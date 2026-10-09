"""SimVLA2 dataset and DataLoader factory."""
from __future__ import annotations

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset_base import BaseDataReader



def create_dataloader(
    batch_size: int,
    metas_path: str,
    num_actions: int,
    training: bool,
    num_workers: int = 4,
    image_size: int = 256,
    shuffle_buffer_size: int = 500,
    lang_aug: str = None,
    lang_dropout_prob: float = 0.0,
) -> DataLoader:
    """Build a DataLoader for SimVLA2 training (default 256x256)."""

    def worker_init_fn(worker_id: int):
        base_seed = torch.initial_seed() % (2**32)
        seed = base_seed + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass

    dataset = BaseDataReader(
        metas_path=metas_path,
        num_actions=num_actions,
        training=training,
        lang_aug=lang_aug,
        lang_dropout_prob=lang_dropout_prob,
        image_size=image_size,
        shuffle_buffer_size=shuffle_buffer_size if training else 0,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )


__all__ = ["create_dataloader"]
