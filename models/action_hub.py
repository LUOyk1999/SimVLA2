"""Action-space registry for SimVLA2.

Every action space is a small ``nn.Module`` that owns two things:

  1. shape / channel metadata (``dim_action``, ``dim_proprio``, ``gripper_idx``)
  2. state / action normalization, plus a ``postprocess`` hook applied to
     generated actions

The training objective is not defined here: every benchmark trains on the
flow-matching velocity MSE in ``SimVLA2.forward``.

Normalization is factored into ``BaseActionSpace`` so subclasses only declare
which JSON keys their norm-stats file uses and how missing keys should fall
back. Class attributes:

  * ``ACTION_KEYS``               : keys tried in order for the action stats
  * ``STATE_KEY``                 : key for proprio stats
  * ``FALLBACK_STATE_TO_ACTION``  : reuse action stats if state stats missing
  * ``FALLBACK_ACTION_TO_STATE``  : reuse state stats if action stats missing
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, Type
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------


class NormStats:
    """Container for per-channel normalization statistics."""

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        q01: Optional[np.ndarray] = None,
        q99: Optional[np.ndarray] = None,
    ):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.q01 = torch.as_tensor(q01, dtype=torch.float32) if q01 is not None else None
        self.q99 = torch.as_tensor(q99, dtype=torch.float32) if q99 is not None else None

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if self.q01 is not None:
            self.q01 = self.q01.to(device)
        if self.q99 is not None:
            self.q99 = self.q99.to(device)
        return self


def load_norm_stats(path: str) -> Dict[str, NormStats]:
    """Load a norm-stats JSON, tolerating both the top-level and nested layouts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Norm stats file not found: {p}")
    with open(p) as f:
        data = json.load(f)
    if "norm_stats" in data:
        data = data["norm_stats"]
    out: Dict[str, NormStats] = {}
    for key, stats in data.items():
        if key == "metadata":
            continue
        out[key] = NormStats(
            mean=np.array(stats["mean"], dtype=np.float32),
            std=np.array(stats["std"], dtype=np.float32),
            q01=np.array(stats["q01"], dtype=np.float32) if stats.get("q01") else None,
            q99=np.array(stats["q99"], dtype=np.float32) if stats.get("q99") else None,
        )
    return out


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


ACTION_REGISTRY: Dict[str, Type["BaseActionSpace"]] = {}


def register_action(name: str):
    """Decorator: register an action space under ``name``."""
    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls
    return _wrap


def build_action_space(name: str, **kwargs) -> "BaseActionSpace":
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------


class BaseActionSpace(nn.Module):
    """Abstract base for all action-space definitions.

    Subclasses override:
      * ``dim_action`` / ``dim_proprio`` / ``gripper_idx``: shape metadata

    Optional overrides:
      * ``postprocess(action)``: the default unnormalizes the action
      * ``ACTION_KEYS`` / ``STATE_KEY`` / ``FALLBACK_STATE_TO_ACTION`` /
        ``FALLBACK_ACTION_TO_STATE``: norm-stats layout
    """

    name: str = "base"
    dim_action: int = 0
    dim_proprio: int = 0
    gripper_idx: Tuple[int, ...] = ()

    # Norm-stats layout: subclasses override these class attrs.
    ACTION_KEYS: Tuple[str, ...] = ("actions", "action")
    STATE_KEY: str = "state"
    FALLBACK_STATE_TO_ACTION: bool = False
    FALLBACK_ACTION_TO_STATE: bool = False

    def __init__(
        self,
        norm_stats_path: Optional[str] = None,
        use_quantile_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.use_quantile_norm = use_quantile_norm
        self.state_norm_stats: Optional[NormStats] = None
        self.action_norm_stats: Optional[NormStats] = None
        if norm_stats_path:
            self.load_norm_stats(norm_stats_path)

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load_norm_stats(self, path: str) -> None:
        stats_dict = load_norm_stats(path)
        for k in self.ACTION_KEYS:
            if k in stats_dict:
                self.action_norm_stats = stats_dict[k]
                break
        if self.STATE_KEY in stats_dict:
            self.state_norm_stats = stats_dict[self.STATE_KEY]
        if self.state_norm_stats is None and self.FALLBACK_STATE_TO_ACTION:
            self.state_norm_stats = self.action_norm_stats
        if self.action_norm_stats is None and self.FALLBACK_ACTION_TO_STATE:
            self.action_norm_stats = self.state_norm_stats

        tag = type(self).__name__
        if self.state_norm_stats is None and self.action_norm_stats is None:
            print(f"[{tag}] warning: no norm stats found in {path}")
            return
        if self.state_norm_stats is not None:
            print(f"[{tag}] state  stats dim={len(self.state_norm_stats.mean)}")
        if self.action_norm_stats is not None:
            print(f"[{tag}] action stats dim={len(self.action_norm_stats.mean)}")

    def to(self, device):
        for ns in (self.state_norm_stats, self.action_norm_stats):
            if ns is not None:
                ns.to(device)
        return super().to(device)

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def _norm(self, x: torch.Tensor, stats: Optional[NormStats]) -> torch.Tensor:
        if stats is None:
            return x
        D = x.shape[-1]
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = stats.q01[..., :D].to(device=x.device, dtype=x.dtype)
            q99 = stats.q99[..., :D].to(device=x.device, dtype=x.dtype)
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        mean = stats.mean[..., :D].to(device=x.device, dtype=x.dtype)
        std = stats.std[..., :D].to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + 1e-6)

    def _unnorm(self, x: torch.Tensor, stats: Optional[NormStats]) -> torch.Tensor:
        if stats is None:
            return x
        D = x.shape[-1]
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = stats.q01[..., :D].to(device=x.device, dtype=x.dtype)
            q99 = stats.q99[..., :D].to(device=x.device, dtype=x.dtype)
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        mean = stats.mean[..., :D].to(device=x.device, dtype=x.dtype)
        std = stats.std[..., :D].to(device=x.device, dtype=x.dtype)
        return x * (std + 1e-6) + mean

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x, self.state_norm_stats)

    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x, self.action_norm_stats)

    def unnormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._unnorm(x, self.action_norm_stats)

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return self.unnormalize_action(action)


# -----------------------------------------------------------------------------
# Bridge V2
# -----------------------------------------------------------------------------


@register_action("bridge")
class BridgeActionSpace(BaseActionSpace):
    """Bridge V2 end-effector action, 7-dim [x, y, z, rx, ry, rz, gripper].

    Actions are absolute EEF poses, i.e. they live in the same space as the
    proprioceptive state, so both state and action normalization use the
    ``state`` stats. ``ACTION_KEYS`` is empty: the ``action`` key in the
    shipped norm-stats file is ignored.
    """

    dim_action = 7
    dim_proprio = 7
    gripper_idx = (6,)
    ACTION_KEYS = ()
    FALLBACK_ACTION_TO_STATE = True


# -----------------------------------------------------------------------------
# LIBERO
# -----------------------------------------------------------------------------


@register_action("libero")
class LiberoActionSpace(BaseActionSpace):
    """LIBERO delta end-effector action, 7-dim, with 8-dim proprio.

    Action ``[delta_xyz(3), delta_euler(3), gripper(1)]``; proprio
    ``[ee_pos(3), axis_angle(3), gripper_states(2)]``.
    """

    dim_action = 7
    dim_proprio = 8
    gripper_idx = (6,)


# -----------------------------------------------------------------------------
# CALVIN
# -----------------------------------------------------------------------------


@register_action("calvin")
class CalvinActionSpace(BaseActionSpace):
    """CALVIN ee6d action, 20-dim: two arms x ``[xyz(3), rotate6d(6), gripper(1)]``.

    CALVIN has one arm, so the second 10 dims are always zero.
    """

    dim_action = 20
    dim_proprio = 20
    gripper_idx = (9, 19)
    FALLBACK_STATE_TO_ACTION = True


# -----------------------------------------------------------------------------
# Fractal
# -----------------------------------------------------------------------------


@register_action("fractal")
class FractalActionSpace(BaseActionSpace):
    """Fractal / RT-1 action, 7-dim, with 7-dim proprio.

    Action ``[delta_xyz(3), rpy(3), gripper(1)]``; proprio
    ``[xyz(3), rpy(3), gripper(1)]``. The norm-stats file uses the singular
    ``action`` key (not ``actions``); either key can be missing and the other
    fills in.
    """

    dim_action = 7
    dim_proprio = 7
    gripper_idx = (6,)
    ACTION_KEYS = ("action",)
    FALLBACK_STATE_TO_ACTION = True
    FALLBACK_ACTION_TO_STATE = True


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------


__all__ = [
    "NormStats",
    "load_norm_stats",
    "BaseActionSpace",
    "BridgeActionSpace",
    "LiberoActionSpace",
    "CalvinActionSpace",
    "FractalActionSpace",
    "ACTION_REGISTRY",
    "register_action",
    "build_action_space",
]
