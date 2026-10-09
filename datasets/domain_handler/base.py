"""Base classes for per-dataset handlers.

Each concrete handler inherits from :class:`DomainHandler` and is listed in
``registry.py`` under its ``dataset_name`` key. ``BaseDataReader`` (``dataset_base.py``) looks the
handler up by that key and splits each yielded ``abs_trajectory`` into
``proprio`` and ``action``.
"""
from __future__ import annotations

import io
import random
from abc import ABC, abstractmethod
from typing import Iterable, List

import h5py
import numpy as np
import torch
from mmengine import fileio
from PIL import Image


class DomainHandler(ABC):
    """Minimal domain handler interface.

    Subclasses provide dataset-specific decoding by implementing an iterator
    that yields per-sample dictionaries compatible with the training loop.
    """
    dataset_name: str

    def __init__(self, meta: dict, num_views: int) -> None:
        self.meta = meta
        self.num_views = num_views

    @abstractmethod
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
        """Yield samples for a single episode."""
        ...

    def _view_mask(self, num_valid: int) -> torch.Tensor:
        """``[num_views]`` bool mask with the first ``num_valid`` views set."""
        mask = torch.zeros(self.num_views, dtype=torch.bool)
        mask[:num_valid] = True
        return mask

    def _stack_views(self, imgs: List[torch.Tensor]) -> torch.Tensor:
        """Zero-fill missing views up to ``num_views`` and stack to ``[V, C, H, W]``."""
        while len(imgs) < self.num_views:
            imgs.append(torch.zeros_like(imgs[0]))
        return torch.stack(imgs, dim=0)

    @staticmethod
    def _augment_language(
        instruction: str,
        *,
        training: bool,
        lang_aug_map: dict | None,
        lang_dropout_prob: float = 0.0,
    ) -> str:
        """Training-only paraphrase lookup, then dropout to "" (null conditioning)."""
        if not training:
            return instruction
        if lang_aug_map and instruction in lang_aug_map:
            instruction = random.choice(lang_aug_map[instruction])
        if lang_dropout_prob > 0 and random.random() < lang_dropout_prob:
            instruction = ""
        return instruction


def _open_h5(path: str) -> h5py.File:
    """Open HDF5 from local FS or remote backend via mmengine.fileio."""
    try:
        return h5py.File(path, "r")
    except OSError:
        return h5py.File(io.BytesIO(fileio.get(path)), "r")


class TFDSHandler(DomainHandler):
    """Shared loader for RLDS datasets.

    Reads through ``tensorflow_datasets.builder_from_directory``.
    ``meta["datalist"][traj_idx]`` is ``{"split": ..., "idx": ...}``; TF is
    imported lazily so only RLDS training needs it installed.
    """

    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)
        self._tfds = None
        self._builder = None

        cache_hw = meta.get("cache_image_hw", (224, 224))
        if isinstance(cache_hw, int):
            cache_hw = (cache_hw, cache_hw)
        self._cache_hw = (int(cache_hw[0]), int(cache_hw[1]))
        self._default_split = meta.get("split", "train")

    def _cache_image(self, arr: np.ndarray) -> np.ndarray:
        """Resize to ``cache_image_hw`` (bicubic) before the train transforms."""
        if arr.shape[0] != self._cache_hw[0] or arr.shape[1] != self._cache_hw[1]:
            pil = Image.fromarray(arr)
            pil = pil.resize((self._cache_hw[1], self._cache_hw[0]), Image.BICUBIC)
            arr = np.asarray(pil, dtype=np.uint8)
        else:
            arr = np.asarray(arr, dtype=np.uint8)
        return np.ascontiguousarray(arr)

    def _get_builder(self):
        if self._builder is None:
            import os
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            import tensorflow as tf
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
            try:
                tf.get_logger().setLevel("ERROR")
                tf.get_logger().propagate = False
            except Exception:
                pass
            import tensorflow_datasets as tfds
            self._tfds = tfds

            data_dir = self.meta.get("data_dir")
            if data_dir is None:
                raise ValueError("meta['data_dir'] is required")
            self._builder = tfds.builder_from_directory(builder_dir=data_dir)
        return self._builder

    def _load_episode(self, traj_idx: int):
        builder = self._get_builder()
        traj_info = self.meta["datalist"][traj_idx]
        if isinstance(traj_info, dict):
            split = traj_info.get("split", self._default_split)
            idx = traj_info.get("idx", 0)
        else:
            split, idx = self._default_split, traj_idx

        ds = builder.as_dataset(split=f"{split}[{idx}:{idx + 1}]", shuffle_files=False)
        for episode in self._tfds.as_numpy(ds):
            return episode
        return None

    @staticmethod
    def _to_numpy(value):
        return value.numpy() if hasattr(value, "numpy") else value
