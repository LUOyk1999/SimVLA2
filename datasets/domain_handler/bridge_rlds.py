"""Bridge V2 RLDS handler.

Cleans the crowd-sourced instructions, drops episodes whose instruction is
unusable (or keeps them as null conditioning when language dropout is on), and
resamples the 7-D absolute end-effector state on a time grid for every chunk.
"""
from __future__ import annotations

import os
import re
import random
from typing import Iterable, Dict
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp1d
from .base import TFDSHandler


# Module-level counters for logging cleaning/augmentation stats
_clean_stats = {"total": 0, "empty": 0, "garbage": 0, "non_ascii": 0, "kept": 0}
_aug_stats = {"total": 0, "article": 0, "synonym": 0, "lower": 0, "preposition": 0, "skipped": 0}
_LOG_INTERVAL = 10000  # log every N episodes


def _clean_instruction(text: str) -> str:
    """Clean and normalize a raw Bridge V2 instruction.

    Returns empty string for garbage that should be skipped.
    """
    _clean_stats["total"] += 1

    if not text:
        _clean_stats["empty"] += 1
        return ""

    text = text.strip().rstrip(".")

    # Skip URLs, numeric-only, random keyboard mash
    if text.startswith("http") or text.startswith("("):
        _clean_stats["garbage"] += 1
        return ""
    if re.fullmatch(r'[\d\s\-]+', text):
        _clean_stats["garbage"] += 1
        return ""
    # Skip short gibberish (no vowel = likely keyboard mash)
    if len(text) <= 5 and not re.search(r'[aeiouAEIOU]', text):
        _clean_stats["garbage"] += 1
        return ""
    # Skip non-ASCII (French, Spanish, Portuguese annotations, ~71 occurrences)
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        _clean_stats["non_ascii"] += 1
        return ""

    # Skip known garbage
    _garbage = {"nothing", "Nothing", "no change", "doing nothing",
                "move something", "end effector transition from object to object",
                "end effector reaching pot or pan"}
    if text in _garbage:
        _clean_stats["garbage"] += 1
        return ""

    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    _clean_stats["kept"] += 1
    return text


def _augment_instruction(text: str, prob: float = 0.5) -> str:
    """Online instruction augmentation for Bridge V2.

    Applied randomly with probability ``prob`` during training.
    Transformations (one chosen at random per call):
      1. Article toggle: add or remove 'the' before nouns
         "put carrot on plate" <-> "put the carrot on the plate"
      2. Synonym substitution for one object word
         "cube" / "block" -> each other or "object", "pot" / "pan" -> each other
         or "container"
      3. Lowercase (a no-op after ``_clean_instruction``, which already lowercases)
      4. Preposition swap: "into" <-> "in", "onto" <-> "on"
    """
    _aug_stats["total"] += 1
    if random.random() > prob:
        _aug_stats["skipped"] += 1
        return text

    aug_type = random.choice(["article", "synonym", "lower", "preposition"])
    _aug_stats[aug_type] += 1

    if aug_type == "article":
        if " the " in text:
            text = re.sub(r'\bthe\b\s+', lambda m: '' if random.random() < 0.5 else m.group(), text)
            text = re.sub(r'\s{2,}', ' ', text).strip()
        else:
            nouns = [
                "carrot", "plate", "spoon", "towel", "eggplant", "basket",
                "pot", "pan", "bowl", "cup", "cloth", "drawer", "sink",
                "knife", "fork", "corn", "banana", "sushi", "broccoli",
                "potato", "pear", "block", "cube", "object", "lid",
            ]
            for noun in nouns:
                pattern = rf'(?<!\bthe\s)\b({noun})\b'
                if re.search(pattern, text) and random.random() < 0.5:
                    text = re.sub(pattern, r'the \1', text, count=1)

    elif aug_type == "synonym":
        synonyms = {
            "cube": ["block", "object"],
            "block": ["cube", "object"],
            "pot": ["pan", "container"],
            "pan": ["pot", "container"],
        }
        words = text.split()
        for i, w in enumerate(words):
            wl = w.lower().rstrip(".,")
            if wl in synonyms and random.random() < 0.3:
                words[i] = random.choice(synonyms[wl])
                break
        text = " ".join(words)

    elif aug_type == "lower":
        text = text.lower()

    elif aug_type == "preposition":
        swaps = [("into", "in"), ("onto", "on")]
        swap = random.choice(swaps)
        if random.random() < 0.5:
            swap = (swap[1], swap[0])
        text = re.sub(rf'\b{swap[0]}\b', swap[1], text)

    return text.strip()




class BridgeTFRecordHandler(TFDSHandler):
    """Bridge V2 handler for TFRecord format.

    Data layout:
      - state: [7] = [x, y, z, rx, ry, rz, gripper]
      - image_0
      - language_instruction
    """

    dataset_name = "Bridge"

    def _extract_step_data(self, step) -> Dict:
        obs = step['observation']
        state = self._to_numpy(obs.get('state', np.zeros(7)))[:7].astype(np.float32)

        lang = self._to_numpy(step.get('language_instruction', b''))
        if isinstance(lang, bytes):
            lang = lang.decode('utf-8')

        return {
            "state": state,
            "language_instruction": lang.strip() if lang else "",
        }

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None = None,
        lang_dropout_prob: float = 0.0,
        **kwargs
    ) -> Iterable[dict]:
        episode = self._load_episode(traj_idx)
        if episode is None:
            return

        steps = list(episode['steps'])
        T = len(steps)

        if T == 0:
            return

        states = []
        language_instruction = None

        for step in steps:
            step_data = self._extract_step_data(step)
            states.append(step_data["state"])
            if language_instruction is None and step_data["language_instruction"]:
                language_instruction = step_data["language_instruction"]

        if language_instruction:
            language_instruction = _clean_instruction(language_instruction)

        # An unlabeled episode is only usable as null conditioning, so it is kept
        # with "" when language dropout is on and skipped otherwise.
        if not language_instruction:
            if lang_dropout_prob > 0:
                language_instruction = ""
            else:
                return
        elif training and lang_dropout_prob > 0 and random.random() < lang_dropout_prob:
            language_instruction = ""

        if _clean_stats["total"] > 0 and _clean_stats["total"] % _LOG_INTERVAL == 0:
            c = _clean_stats
            a = _aug_stats
            print(
                f"[BridgeV2 InstructionStats] "
                f"clean: total={c['total']} kept={c['kept']} "
                f"empty={c['empty']} garbage={c['garbage']} non_ascii={c['non_ascii']} | "
                f"aug: total={a['total']} applied={a['total']-a['skipped']} "
                f"(article={a['article']} synonym={a['synonym']} "
                f"lower={a['lower']} preposition={a['preposition']})",
                flush=True,
            )

        states = np.stack(states, axis=0).astype(np.float32)

        freq = 5.0
        # Physical duration covered by one action chunk, in seconds. This is the
        # horizon knob: `num_actions` only sets how many interpolation points are
        # sampled inside the window, so each step lasts qdur / num_actions.
        # Changing num_actions alone therefore changes execution speed rather
        # than how far the model plans. To change the plan length without
        # distorting speed, move qdur and num_actions together. Override with
        # BRIDGE_QDUR.
        qdur = float(os.environ.get("BRIDGE_QDUR", 5.0))
        t = np.arange(T, dtype=np.float64) / freq

        interp = interp1d(t, states, axis=0, bounds_error=False, fill_value=(states[0], states[-1]))

        image_mask = self._view_mask(1)

        # Chunks cover every start index range(T-1); windows that step past
        # the episode's last recorded time keep the terminal pose (interp1d
        # fill_value) and mark padded rows via ``action_valid``.
        t_max = float(t.max())
        idxs = list(range(0, max(1, T - 1)))
        if training:
            random.shuffle(idxs)

        for idx in idxs:
            cur = t[idx]

            q = np.linspace(cur, cur + qdur, num_actions + 1, dtype=np.float32)
            valid = int(np.count_nonzero(q[1:] <= t_max + 1e-6))
            q = np.clip(q, float(t[0]), t_max)
            act_seq = torch.tensor(interp(q)).float()

            if (act_seq[1:] - act_seq[:-1]).abs().max() < 1e-6:
                continue

            ins = language_instruction
            if ins:
                ins = self._augment_language(ins, training=training, lang_aug_map=lang_aug_map)
            if ins and training:
                ins = _augment_instruction(ins, prob=0.5)

            obs = steps[idx]['observation']
            np_img = self._to_numpy(obs.get('image_0'))
            if np_img is None:
                continue
            pil_img = Image.fromarray(self._cache_image(np_img))
            image_input = self._stack_views([image_aug(pil_img)])
            proprio_current = torch.tensor(states[idx], dtype=torch.float32)

            sample = {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": act_seq,
                "proprio": proprio_current,
            }
            sample["action_valid"] = torch.tensor(valid, dtype=torch.long)
            yield sample


class BridgeV2Handler(BridgeTFRecordHandler):
    dataset_name = "bridge"

    def __init__(self, meta: dict, num_views: int) -> None:
        meta = meta.copy()
        meta.setdefault("dataset_type", "bridge_v2")
        super().__init__(meta, num_views)
