from typing import Iterator, List, Optional, Sized

import torch
from mmengine.dataset.sampler import DefaultSampler
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DistributedWeightedRandomSampler(DefaultSampler):
    """
    Distributed version of WeightedRandomSampler in https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/sampler.py#L220.

    This sampler will sample a list of indices from the dataset with weights, and if replacement is set to True,
    the same index can be sampled multiple times in the same epoch.
    Note that this sampler will first sample from all indices with torch.multinomial, and then split the sampled indices,
    and thus it might select the same indices in a mini batch since it's a oversampling strategy.

    It has several differences from the PyTorch ``DistributedSampler`` as
    below:

    1. This sampler supports non-distributed environment.

    2. The round up behaviors are a little different.

       - If ``round_up=True``, this sampler will add extra samples to make the
         number of samples is evenly divisible by the world size. And
         this behavior is the same as the ``DistributedSampler`` with
         ``drop_last=False``.
       - If ``round_up=False``, this sampler won't remove or add any samples
         while the ``DistributedSampler`` with ``drop_last=True`` will remove
         tail samples.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(
        self,
        dataset: Sized,
        weights: Optional[List[float]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
        replacement: bool = True,
    ) -> None:
        super(DistributedWeightedRandomSampler, self).__init__(
            dataset=dataset, shuffle=shuffle, seed=seed, round_up=round_up
        )

        assert self.shuffle, "DistributedWeightedRandomSampler only supports shuffle=True"
        if weights is None:
            self.weights = self.dataset.frame_weights

        self.weights = torch.tensor(weights, dtype=torch.double)

        assert len(self.weights) == len(self.dataset), "weights length should be equal to dataset length"
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        # Shuffle must be true to use multinomial sampling
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, len(self.dataset), self.replacement, generator=g).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.world_size]
        return iter(indices)
