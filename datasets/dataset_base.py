"""Base dataset for SimVLA2 training.

Iterable dataset with configurable image resolution, ImageNet normalization,
distributed sharding, and a shuffle buffer that mixes episodes into batches.
"""

from __future__ import annotations
from typing import Dict, Iterable, List
import io
import json
import random
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .utils import action_slice
from .domain_handler.registry import get_handler_cls


class ShuffleBuffer:
    """Reservoir-style shuffle buffer.

    Fills up to ``buffer_size`` samples, then on each ``next()`` returns a
    random one and refills from the source. This interleaves samples from
    different episodes so a batch is not correlated by temporal position.
    """
    def __init__(self, data_source: Iterable, buffer_size: int = 2000):
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.buffer: List[dict] = []
        self._source_iter = None
        self._exhausted = False

    def __iter__(self):
        self._source_iter = iter(self.data_source)
        self._exhausted = False
        self.buffer = []
        self._fill_buffer()
        return self

    def _fill_buffer(self):
        while len(self.buffer) < self.buffer_size and not self._exhausted:
            try:
                self.buffer.append(next(self._source_iter))
            except StopIteration:
                self._exhausted = True
                break

    def _try_add_one(self):
        if self._exhausted:
            return
        try:
            self.buffer.append(next(self._source_iter))
        except StopIteration:
            self._exhausted = True

    def __next__(self):
        if len(self.buffer) == 0:
            raise StopIteration
        idx = random.randint(0, len(self.buffer) - 1)
        sample = self.buffer[idx]
        self.buffer[idx] = self.buffer[-1]
        self.buffer.pop()
        self._try_add_one()
        return sample


class BaseDataReader(IterableDataset):
    """Infinite data reader for SimVLA2 training.

    Output sample:
      {
        'language_instruction': str,
        'image_input': FloatTensor[V, C, H, W],
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action],
        'action_valid': LongTensor[],   # only if the handler pads past the episode end
      }
    """

    # ImageNet mean/std; SimVLA2Processor normalizes with the same values at inference.
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        metas_path: str,
        num_actions: int = 10,
        num_views: int = 3,
        training: bool = True,
        lang_aug: str = None,
        lang_dropout_prob: float = 0.0,
        image_size: int = 256,
        shuffle_buffer_size: int = 500,
    ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.lang_dropout_prob = lang_dropout_prob
        self.image_size = image_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.metas: Dict[str, dict] = {}

        print(f"[Dataset] Image size: {self.image_size}x{self.image_size}")
        print(f"[Dataset] shuffle_buffer_size={self.shuffle_buffer_size}")
        
        with io.BytesIO(fileio.get(metas_path)) as f:
            meta = json.load(f)
        print(f"== dataset {meta['dataset_name']} with {len(meta['datalist'])} trajs")
        self.metas[meta["dataset_name"]] = meta

        if lang_aug and training:
            import os as _os
            if _os.path.isfile(lang_aug):
                with open(lang_aug, "r") as f:
                    _aug_map = json.load(f)
                _aug_map = {k: v for k, v in _aug_map.items() if not k.startswith("_")}
                for meta in self.metas.values():
                    meta["lang_aug_map"] = _aug_map
                print(f"[Dataset] Loaded lang_aug_map from {lang_aug} ({len(_aug_map)} entries)")

        self.image_aug = self._build_image_transforms(training)

    def _build_image_transforms(self, training: bool) -> transforms.Compose:
        """Build the image transform for training or eval.

        Bicubic resize to ``self.image_size``, ColorJitter when ``training``,
        then ToTensor and ImageNet normalization.
        """
        transform_list = [
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
        
        if training:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.0
                )
            )
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_STD, inplace=True),
        ])
        
        return transforms.Compose(transform_list)

    def _get_distributed_info(self) -> tuple:
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            global_rank = rank * num_workers + worker_id
            global_world_size = world_size * num_workers
        else:
            global_rank = rank
            global_world_size = world_size

        return global_rank, global_world_size

    def _shard_indices(self, indices: List[int], rank: int, world_size: int) -> List[int]:
        if world_size <= 1:
            return indices
        return indices[rank::world_size]

    def _iter_one_epoch(self, dataset_name: str, rank: int, world_size: int, handler, epoch: int) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        idx_for_delta = meta.get("idx_for_delta", [])

        traj_indices = list(range(len(meta["datalist"])))
        if self.training:
            # Same seed on every rank, so the shards below partition the datalist.
            rng = random.Random(42 + epoch)
            rng.shuffle(traj_indices)

        if world_size > 1:
            traj_indices = self._shard_indices(traj_indices, rank, world_size)
            if rank == 0 and epoch == 0:
                print(f"[Dataset] {dataset_name}: rank {rank}/{world_size} got {len(traj_indices)} episodes")

        if self.training:
            random.shuffle(traj_indices)

        for traj_idx in traj_indices:
            try:
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),
                    lang_dropout_prob=self.lang_dropout_prob,
                ):
                    # A proprio set by the handler wins over row 0 of
                    # abs_trajectory; deltas still use row 0.
                    has_proprio = "proprio" in sample
                    slice_result = action_slice(sample["abs_trajectory"], idx_for_delta)
                    if has_proprio:
                        sample["action"] = slice_result["action"]
                    else:
                        sample.update(slice_result)
                    del sample["abs_trajectory"]
                    yield sample
            except Exception as e:
                import traceback as tb
                print(f"[Dataset] Error in {dataset_name} traj {traj_idx}: {e}")
                tb.print_exc()
                continue

    def _infinite_dataset_iter(self, dataset_name: str, rank: int, world_size: int) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        Handler = get_handler_cls(dataset_name)
        handler = Handler(meta=meta, num_views=self.num_views)
        epoch = 0
        while True:
            yielded_any = False
            for sample in self._iter_one_epoch(dataset_name, rank, world_size, handler, epoch):
                yielded_any = True
                yield sample
            if not yielded_any:
                print(f"[Dataset] WARNING: No samples from {dataset_name}! Breaking.")
                break
            epoch += 1

    def _raw_iter(self):
        rank, world_size = self._get_distributed_info()
        (name,) = self.metas.keys()
        if self.training:
            yield from self._infinite_dataset_iter(name, rank, world_size)
        else:
            Handler = get_handler_cls(name)
            handler = Handler(meta=self.metas[name], num_views=self.num_views)
            yield from self._iter_one_epoch(name, rank, world_size, handler, epoch=0)

    def __iter__(self):
        if self.training and self.shuffle_buffer_size > 0:
            yield from ShuffleBuffer(self._raw_iter(), self.shuffle_buffer_size)
        else:
            yield from self._raw_iter()


__all__ = [
    "BaseDataReader",
    "ShuffleBuffer",
]
