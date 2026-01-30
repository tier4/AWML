from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from mmengine import check_file_exist
from mmengine.fileio import get


class MappingPolicy(Enum):
    CONCAT = "concat"
    FIRST = "first"
    LAST = "last"


def _build_reverse_class_mapping(class_mapping: Dict[str, int], ignore_index: int) -> Dict[int, List[str]]:
    """Build reverse mapping from index to class names."""
    reverse_class_mapping: Dict[int, List[str]] = {}
    for class_name, idx in class_mapping.items():
        if idx == ignore_index:
            continue
        reverse_class_mapping.setdefault(idx, []).append(class_name)

    return reverse_class_mapping


def _join_classes(classes: List[str], policy: MappingPolicy) -> str:
    """Join class names according to the selected policy."""
    if policy == MappingPolicy.CONCAT:
        return "+".join(classes)
    if policy == MappingPolicy.FIRST:
        return classes[0]
    if policy == MappingPolicy.LAST:
        return classes[-1]
    raise ValueError(f"Unknown policy: {policy}")


def _load_raw_semantic_mask(mask_path: str, seg_dtype: np.dtype, backend_args: Optional[dict]) -> np.ndarray:
    """Load raw semantic mask from a local or backend path."""
    try:
        mask_bytes = get(mask_path, backend_args=backend_args)
        # add .copy() to fix read-only bug
        return np.frombuffer(mask_bytes, dtype=seg_dtype).copy()
    except ConnectionError:
        check_file_exist(mask_path)
        return np.fromfile(mask_path, dtype=seg_dtype)


def class_mapping_to_names(
    class_mapping: Dict[str, int], ignore_index: int, policy: MappingPolicy = MappingPolicy.CONCAT
) -> List[str]:
    """Build class names list from a class mapping.

    Args:
        class_mapping: Mapping from class name to target index.
        ignore_index: Index to ignore in the mapping.
        policy: Policy for naming classes for the same index.

    Returns:
        Class names list ordered by sorted class indices.
        The returned list is contiguous; original indices only affect
        ordering (after sorting) and do not map 1:1 to positions.
    """
    reverse_class_mapping = _build_reverse_class_mapping(class_mapping, ignore_index)

    if not reverse_class_mapping:
        return []

    class_names: List[str] = []
    for idx in sorted(reverse_class_mapping.keys()):
        classes = reverse_class_mapping[idx]
        class_names.append(_join_classes(classes, policy))

    return class_names


def class_mapping_to_names_palette_label2cat(
    class_mapping: Dict[str, int],
    ignore_index: int,
    base_palette: List[List[int]],
    base_class_names: List[str],
    policy: MappingPolicy = MappingPolicy.CONCAT,
) -> Tuple[List[str], List[List[int]], Dict[int, str]]:
    """Build class names, palette, and label2cat from a class mapping.

    Args:
        class_mapping: Mapping from class name to target index.
        ignore_index: Index to ignore in the mapping.
        base_palette: List of base palette colors.
        base_class_names: List of base class names matching palette order.
        policy: Policy for naming classes for the same index.

    Returns:
        Tuple of (class_names, palette, label2cat).
        - class_names is ordered by sorted class indices and is contiguous.
        - label2cat maps contiguous indices to class names (not original ids).
        - palette uses the first class name's palette entry.
    """
    reverse_class_mapping = _build_reverse_class_mapping(class_mapping, ignore_index)
    if not reverse_class_mapping:
        return [], [], {}

    if len(base_palette) != len(base_class_names):
        raise ValueError("Length of base_palette and base_class_names must be the same")
    base_palette_by_name: Dict[str, List[int]] = {
        class_name: base_palette[idx] for idx, class_name in enumerate(base_class_names)
    }

    class_names: List[str] = []
    palette: List[List[int]] = []
    label2cat: Dict[int, str] = {}

    for output_idx, idx in enumerate(sorted(reverse_class_mapping.keys())):
        classes = reverse_class_mapping[idx]
        joined_classes = _join_classes(classes, policy)
        class_names.append(joined_classes)
        label2cat[output_idx] = joined_classes
        first_class = classes[0]
        if first_class not in base_palette_by_name:
            raise ValueError(f"Missing palette for class: {first_class}")
        palette.append(base_palette_by_name[first_class])

    return class_names, palette, label2cat


def load_and_map_semantic_mask(
    mask_path: str,
    raw_categories: Dict[str, int],
    class_mapping: Dict[str, int],
    ignore_index: int,
    seg_dtype: np.dtype = np.int64,
    selections: Optional[List[Dict[str, int]]] = None,
    backend_args: Optional[dict] = None,
) -> np.ndarray:
    """Load a raw semantic mask and map it to dataset class indices.

    Args:
        mask_path: Path to the raw semantic mask file.
        raw_categories: Mapping from class name to raw label value.
        class_mapping: Mapping from class name to target class index.
        ignore_index: Index to use for unknown/ignored labels.
        seg_dtype: Numpy dtype for reading the raw mask.
        selections: Optional list of slices with 'idx_begin' and 'length'.
        backend_args: Optional backend arguments for remote file access.

    Returns:
        Mapped semantic mask as np.int64 with contiguous class indices.
    """
    pts_semantic_mask = _load_raw_semantic_mask(mask_path, seg_dtype, backend_args)

    if selections:
        segments = [pts_semantic_mask[s["idx_begin"] : s["idx_begin"] + s["length"]] for s in selections]
        pts_semantic_mask = np.concatenate(segments, axis=0) if segments else pts_semantic_mask[:0]

    if pts_semantic_mask.size == 0:
        return pts_semantic_mask.astype(np.int64)

    raw_to_category = {int(v): k for k, v in raw_categories.items()}
    if not raw_to_category:
        return np.full(pts_semantic_mask.shape, ignore_index, dtype=np.int64)

    raw_labels = list(raw_to_category.keys())
    if all(isinstance(k, (int, np.integer)) for k in raw_labels) and min(raw_labels) >= 0:
        max_raw = max(raw_labels)
        lut = np.full(max_raw + 1, ignore_index, dtype=np.int64)
        for raw_label, category in raw_to_category.items():
            lut[raw_label] = class_mapping.get(category, ignore_index)
        mapped = np.full(pts_semantic_mask.shape, ignore_index, dtype=np.int64)
        in_range = (pts_semantic_mask >= 0) & (pts_semantic_mask <= max_raw)
        mapped[in_range] = lut[pts_semantic_mask[in_range]]
        return mapped

    def _map_segment(raw_label: np.integer) -> int:
        category = raw_to_category.get(int(raw_label), None)
        if category is None:
            return ignore_index
        return class_mapping.get(category, ignore_index)

    return np.vectorize(_map_segment)(pts_semantic_mask).astype(np.int64)
