import itertools
import math
from typing import Iterator, Optional, Sized

import numpy as np
import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler


@DATA_SAMPLERS.register_module()
class GroupStreamingSampler(Sampler):

    def __init__(
        self,
        dataset: Sized,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: Optional[int] = 10,
        pad_sequences: bool = False,
        trim_sequences: bool = False,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle

        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        self.batch_size = batch_size
        self.pad_sequences = pad_sequences
        self.trim_sequences = trim_sequences
        self.indices = {}

        self._set_group_indices()
        self._compute_indices(self.epoch)

    def _set_group_indices(self):

        unique_groups = np.unique(self.dataset.flag)
        group_indices = {i: [] for i in unique_groups}
        for i, v in enumerate(self.dataset.flag):
            group_indices[v].append(i)
        self.group_indices = list(group_indices.values())

    def _compute_indices(self, epoch: int):
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            chosen_indices = torch.randperm(len(self.group_indices), generator=g).tolist()
            print("DEBUG: First 10 entries of shuffled indices are: ", chosen_indices[:10])
        else:
            chosen_indices = torch.arange(len(self.group_indices)).tolist()

        self.indices[self.epoch] = []

        for rank in range(self.world_size):
            # subsample
            shuffled_indices = chosen_indices[rank : len(self.group_indices) : self.world_size]
            selected_groups = [self.group_indices[i] for i in shuffled_indices]
            # Divide selected_groups into self.batch_size groups, drop the last if not divisible
            batch_groups = [[] for _ in range(self.batch_size)]
            for i, group in enumerate(selected_groups):
                batch_groups[i % self.batch_size].extend(group)
            indices = []
            while all(len(batch_groups[i]) > 0 for i in range(self.batch_size)):
                for i in range(self.batch_size):
                    indices.append(batch_groups[i].pop(0))
            self.indices[self.epoch].append(indices)

        if self.pad_sequences:
            max_length = max(len(indices) for indices in self.indices[self.epoch])
            for i in range(len(self.indices[self.epoch])):
                self.indices[self.epoch][i] = self.indices[self.epoch][i] + [self.indices[self.epoch][i][-1]] * (
                    max_length - len(self.indices[self.epoch][i])
                )
        elif self.trim_sequences:
            min_length = min(len(indices) for indices in self.indices[self.epoch])
            for i in range(self.world_size):
                self.indices[self.epoch][i] = self.indices[self.epoch][i][:min_length]

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        if self.epoch + 1 not in self.indices:
            self._compute_indices(self.epoch + 1)
        return iter(self.indices[self.epoch][self.rank])

    def __len__(self) -> int:
        return len(self.indices[self.epoch][self.rank])

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
