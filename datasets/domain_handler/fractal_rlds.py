"""Fractal (RT-1) TFRecord handler.

``abs_trajectory`` holds absolute states; ``idx_for_delta = [0, 1, 2]`` in the
meta turns only the xyz position into a delta, while rotation (Euler xyz) and
gripper stay absolute. Layout:

  state           : [x, y, z, roll, pitch, yaw, gripper]                 (7)
  action target   : [dx, dy, dz, roll, pitch, yaw, gripper]              (7)

The model is fed an all-zero proprio (``fractal_client.py`` sends zeros by
default as well); the state only serves as the origin of the xyz deltas.
"""
from __future__ import annotations

import os
import random
from typing import Iterable
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from .base import TFDSHandler



class FractalHandler(TFDSHandler):
    """Fractal (RT-1) handler.

    Rotations are interpolated with SLERP on quaternions; interpolating the
    Euler angles directly would break wherever an angle wraps around at +-pi.
    """

    dataset_name = "fractal"

    def iter_episode(
        self,
        traj_idx,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None = None,
        **kwargs
    ) -> Iterable[dict]:
        """Yield the samples of one episode.

        ``abs_trajectory`` is ``[num_actions + 1, 7]``: row 0 is the current
        state, rows 1.. the future states. ``action_slice`` (``datasets/utils.py``)
        subtracts row 0 from the xyz columns of the future rows and keeps
        columns 3:7 absolute. The sample carries its own zero ``proprio``, so
        row 0 is not used as the model's proprio.
        """
        episode = self._load_episode(traj_idx)
        if episode is None:
            return

        steps = list(episode['steps'])
        T = len(steps)
        if T == 0:
            return

        # First pass: per-step position, unit quaternion and gripper state.
        positions = []
        quaternions = []
        grippers = []
        language_instruction = None

        for step in steps:
            obs = step['observation']
            
            # [position(3), quaternion(4)]
            pose = self._to_numpy(obs["base_pose_tool_reached"])
            positions.append(pose[:3].astype(np.float32))
            
            # scipy expects [x, y, z, w], and Slerp needs unit quaternions; a
            # degenerate one becomes the identity.
            quat = pose[3:7].astype(np.float32)
            norm = np.linalg.norm(quat)
            if norm > 1e-6: quat /= norm
            else: quat = np.array([0, 0, 0, 1.0], dtype=np.float32)
            quaternions.append(quat)
            
            gripper_raw = obs.get("gripper_closed", obs.get("gripper_closedness_commanded"))
            if gripper_raw is not None:
                gripper_state = self._to_numpy(gripper_raw)
                gripper_state = np.atleast_1d(np.asarray(gripper_state).flatten())
                if len(gripper_state) == 0:
                    gripper_state = np.array([0.0], dtype=np.float32)
                else:
                    gripper_state = gripper_state[:1].astype(np.float32)
            else:
                gripper_state = np.array([0.0], dtype=np.float32)
            grippers.append(gripper_state)
            
            if language_instruction is None:
                lang = self._to_numpy(obs.get("natural_language_instruction", b""))
                if isinstance(lang, bytes): lang = lang.decode("utf-8")
                if lang.strip(): language_instruction = lang.strip()

        if not language_instruction or len(positions) == 0:
            return

        positions = np.stack(positions, axis=0)      # [T, 3]
        quaternions = np.stack(quaternions, axis=0)  # [T, 4]
        grippers = np.stack(grippers, axis=0)        # [T, 1]

        freq = 3.0
        # Physical duration covered by one action chunk, in seconds. `num_actions`
        # only sets the sampling density inside the window, so changing it alone
        # changes execution speed. Default 10 s with H=30 gives 0.333 s per step,
        # matching the native 3 Hz. Override with FRACTAL_QDUR.
        qdur = float(os.environ.get("FRACTAL_QDUR", 10.0))
        t = np.arange(T, dtype=np.float64) / freq

        # Position and gripper interpolate linearly, rotation by Slerp.
        pos_grip = np.concatenate([positions, grippers], axis=-1)
        interp_pos_grip = interp1d(t, pos_grip, axis=0, bounds_error=False, fill_value=(pos_grip[0], pos_grip[-1]))

        rots = R.from_quat(quaternions)
        # Slerp requires strictly increasing, unique timestamps.
        interp_rots = Slerp(t, rots)

        image_mask = self._view_mask(1)

        # Chunk sampler: starts cover every index in the episode
        # (range(T - 1)). Windows that step past the episode's last
        # recorded time keep the terminal pose (interp1d fill_value) and
        # mark the padded rows via ``action_valid``, so the tail of the
        # episode is still sampled without supervising on repeated frames.
        idxs = list(range(0, max(1, T - 1)))
        if training:
            random.shuffle(idxs)

        # Second pass: sample action windows on the interpolated trajectory.
        t_min, t_max = float(t[0]), float(t[-1])
        for idx in idxs:
            cur = t[idx]
            q = np.linspace(cur, cur + qdur, num_actions + 1, dtype=np.float64)
            valid = int(np.count_nonzero(q[1:] <= t_max + 1e-9))
            np.clip(q, t_min, t_max, out=q)

            res_pos_grip = interp_pos_grip(q)  # [num_actions+1, 4]
            res_rots = interp_rots(q)
            res_euler = res_rots.as_euler('xyz').astype(np.float32)  # [num_actions+1, 3]

            abs_traj = np.zeros((num_actions + 1, 7), dtype=np.float32)
            abs_traj[:, :3] = res_pos_grip[:, :3]
            abs_traj[:, 3:6] = res_euler
            abs_traj[:, 6] = res_pos_grip[:, 3]
            
            abs_traj = torch.from_numpy(abs_traj)

            # Skip static segments
            if (abs_traj[1] - abs_traj[0]).abs().max() < 1e-5:
                continue

            ins = self._augment_language(
                language_instruction, training=training, lang_aug_map=lang_aug_map
            )

            obs = steps[idx]['observation']
            np_img = self._to_numpy(obs.get('image'))
            if np_img is None or np_img.size == 0:
                continue
            pil_img = Image.fromarray(self._cache_image(np_img))
            image_input = self._stack_views([image_aug(pil_img)])

            sample = {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": abs_traj,
                "proprio": torch.zeros(7, dtype=torch.float32),
            }
            sample["action_valid"] = torch.tensor(valid, dtype=torch.long)
            yield sample
