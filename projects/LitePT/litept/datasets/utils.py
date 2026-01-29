import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def _offset2batch(offset: torch.Tensor) -> torch.Tensor:
    """
    Convert an offset tensor (cumulative counts) into a per-point batch index tensor.

    Example:
        offset = tensor([3, 7])
        returns tensor([0,0,0, 1,1,1,1])
    """
    if not isinstance(offset, torch.Tensor):
        offset = torch.as_tensor(offset)

    if offset.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=offset.device)

    # Ensure 1D long on same device
    offset = offset.to(dtype=torch.long).view(-1)

    # Counts per batch element
    counts = offset.diff(
        prepend=torch.tensor([0], device=offset.device, dtype=offset.dtype)
    )
    if (counts < 0).any():
        raise ValueError(
            f"offset must be non-decreasing cumulative counts, got: {offset}"
        )

    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=offset.device, dtype=torch.long), counts
    )


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {
            key: (
                collate_fn([d[key] for d in batch])
                if "offset" not in key
                # offset -> bincount -> concat bincount-> concat offset
                else torch.cumsum(
                    collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                    dim=0,
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        if "offset" in batch.keys():
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )

        ### fix bug ###
        # recompute grid coord !!
        grid_coord_new = []
        batch_size = len(batch["offset"])

        batch_mask = _offset2batch(batch["offset"])
        for bs_id in range(batch_size):
            sample_mask = batch_mask == bs_id
            coord_sample = batch["coord"][sample_mask]
            scaled_coord_sample = coord_sample / batch["grid_size"][0]  # hack here!
            grid_coord_sample = torch.floor(scaled_coord_sample).to(torch.int64)
            min_coord_sample = grid_coord_sample.min(0)[0]
            grid_coord_sample -= min_coord_sample

            grid_coord_new.append(grid_coord_sample)

        grid_coord_new = torch.cat(grid_coord_new, dim=0)
        batch["grid_coord"] = grid_coord_new
        ### fix bug ###

    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
