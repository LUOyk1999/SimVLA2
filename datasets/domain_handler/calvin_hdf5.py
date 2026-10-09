"""CALVIN task_ABC_D HDF5 handler.

Reads the segmented HDF5 dump (one ``seg_XXXXXX.h5`` per contact-free
trajectory) and, for ``action_mode`` ``calvin``, builds the 20-D two-arm
ee6d trajectory SimVLA2 consumes from the recorded robot state (``proprio``
or ``robot_obs``): xyz + rotate6d + gripper per arm, with the second arm all
zeros -- see :meth:`CalvinHandler.build_left_right`.
"""
from __future__ import annotations

import random
from typing import Any, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp1d

from ..utils import decode_image_from_bytes, euler_to_rotate6d
from .base import DomainHandler, _open_h5


class CalvinHandler(DomainHandler):
    """CALVIN HDF5 handler with optional ee6d conversion.

    Each file is opened once per episode; the action window is resampled on a
    ``freq`` time grid by ``interp1d`` (see :meth:`build_left_right`).
    """

    dataset_name = "calvin"

    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys: Sequence[str] = self.meta["observation_key"]
        return [f[k][()] for k in keys]

    def read_instruction(self, f: h5py.File) -> str:
        candidate_keys = []
        meta_key = self.meta.get("language_instruction_key")
        if meta_key:
            candidate_keys.append(meta_key)
        candidate_keys.extend([
            "language_instruction",
            "lang",
            "instruction",
            "task_name",
        ])

        for key in candidate_keys:
            if key not in f:
                continue
            ds = f[key]
            value = ds[()]
            if getattr(ds, "shape", ()) == ():
                if isinstance(value, (bytes, bytearray)):
                    return value.decode()
                return str(value)
            first = value[0]
            if isinstance(first, (bytes, bytearray)):
                return first.decode()
            return str(first)

        # Some CALVIN HDF5 exports omit language entirely; keep training running.
        return ""

    def _load_proprio(self, f: h5py.File) -> np.ndarray:
        if "proprio" in f:
            return f["proprio"][()]
        if "robot_obs" in f:
            return f["robot_obs"][()]
        raise KeyError("No proprio or robot_obs found in Calvin H5 file.")

    def _gripper_to_signed(self, grip_raw: np.ndarray) -> np.ndarray:
        g = np.asarray(grip_raw, dtype=np.float32).reshape(-1)
        if g.size == 0:
            return g.astype(np.float32)

        mode = str(self.meta.get("gripper_mode", "auto")).lower()
        thr = float(self.meta.get("gripper_threshold", 0.5))
        width_thr = float(self.meta.get("gripper_width_threshold", 0.02))
        if mode == "signed":
            return g.astype(np.float32)
        if mode == "prob":
            return np.where(g > thr, 1.0, -1.0).astype(np.float32)
        if mode == "width":
            return np.where(g >= width_thr, 1.0, -1.0).astype(np.float32)

        gmin = np.nanmin(g)
        gmax = np.nanmax(g)
        if gmax <= 0.2:
            return np.where(g >= width_thr, 1.0, -1.0).astype(np.float32)
        if gmin < 0:
            return np.where(g >= 0, 1.0, -1.0).astype(np.float32)
        return np.where(g <= thr, 1.0, -1.0).astype(np.float32)

    def build_left_right(
        self,
        f: h5py.File,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        # Margin must be the window length in frames; store it here because
        # iter_episode calls this method before index_candidates.
        self._window_frames = int(np.ceil(qdur * freq))
        proprio = self._load_proprio(f)
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        proprio = np.asarray(proprio, dtype=np.float32)
        num_steps = proprio.shape[0]

        if proprio.shape[1] >= 6:
            pos = proprio[:, :3]
            euler = proprio[:, 3:6]
        else:
            pos = np.zeros((num_steps, 3), dtype=np.float32)
            euler = np.zeros((num_steps, 3), dtype=np.float32)

        if proprio.shape[1] >= 15:
            grip_raw = proprio[:, -1]
        elif proprio.shape[1] >= 7:
            grip_raw = proprio[:, 6]
        else:
            grip_raw = np.zeros((num_steps,), dtype=np.float32)

        grip_signed = self._gripper_to_signed(grip_raw)
        if grip_signed.ndim == 1:
            grip_signed = grip_signed[:, None]

        action_mode = str(self.meta.get("action_mode", "")).lower()
        if "calvin" in action_mode:
            left = np.concatenate(
                [pos, euler_to_rotate6d(euler, "xyz"), grip_signed],
                axis=-1,
            )
            right = np.zeros_like(left)
        else:
            left = np.concatenate([pos, euler, grip_signed], axis=-1)
            right = np.zeros((num_steps, 0), dtype=np.float32)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        """Start indices that leave room for a full action window.

        The window is never compressed at the end of an episode. The margin is
        ``ceil(qdur * freq)``: what matters for staying in range is the window
        length in frames, not how many points are sampled from it.
        """
        margin = int(getattr(self, "_window_frames", 30))
        return range(0, max(0, T_left - margin))

    @staticmethod
    def _pil_from_arr(arr: Any) -> Image.Image:
        return decode_image_from_bytes(arr) if not isinstance(arr, Image.Image) else arr

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """Open once, yield many samples; file is always closed on exit."""
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]

        with _open_h5(datapath) as f:
            images = self.get_image_datasets(f)
            ins = self.read_instruction(f)
            left, right, lt, rt, freq, qdur = self.build_left_right(f)

        image_mask = self._view_mask(len(images))
        if lt is None: lt = np.arange(left.shape[0], dtype=np.float64) / float(freq)
        if rt is None: rt = np.arange(right.shape[0], dtype=np.float64) / float(freq)

        idxs = list(self.index_candidates(left.shape[0], training))
        if training: random.shuffle(idxs)

        L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        ref = (lt + rt) / 2.0

        V = min(self.num_views, len(images))
        for idx in idxs:
            cur = ref[idx]
            q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
            lseq = torch.tensor(L(q))
            rseq = torch.tensor(R(q))

            if (lseq[1] - lseq[0]).abs().max() < 1e-5 and (rseq[1] - rseq[0]).abs().max() < 1e-5:
                continue

            sample_ins = self._augment_language(ins, training=training, lang_aug_map=lang_aug_map)
            imgs = [image_aug(self._pil_from_arr(images[v][idx])) for v in range(V)]

            yield {
                "language_instruction": sample_ins,
                "image_input": self._stack_views(imgs),
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([lseq, rseq], -1).float()
            }
