"""Shared helpers used by the dataset handlers.

Image decoding, an Euler -> rotate6d converter, and ``action_slice``.
"""
from __future__ import annotations
import numpy as np, cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
from typing import Sequence, Dict
import torch

def decode_image_from_bytes(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x

    if isinstance(x, (bytes, bytearray)):
        x = np.frombuffer(x, dtype=np.uint8)

    arr = np.asarray(x)

    if arr.dtype == object and arr.size == 1:
        return decode_image_from_bytes(arr.item())

    if arr.ndim == 0:
        return decode_image_from_bytes(arr.item())

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if arr.ndim != 1:
        raise ValueError(f"Unsupported image array shape: {arr.shape}")

    rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if rgb is None:
        raw = np.frombuffer(arr.tobytes(), dtype=np.uint8)
        if raw.size == 2764800:
            rgb = raw.reshape(720, 1280, 3)
        elif raw.size == 921600:
            rgb = raw.reshape(480, 640, 3)
        else:
            raise ValueError(f"Failed to decode image buffer with shape {arr.shape}")
    return Image.fromarray(rgb)

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))



def action_slice(abs_traj: torch.Tensor, idx_for_delta: Sequence[int] = ()) -> Dict[str, torch.Tensor]:
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj must be a torch.Tensor")
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    proprio = abs_traj[0]         # [D]
    action = abs_traj[1:].clone() # [H, D]

    if idx_for_delta:
        idx = torch.as_tensor(idx_for_delta, dtype=torch.long, device=abs_traj.device)
        action[:, idx] -= proprio[idx]
    return {"proprio": proprio, "action": action}