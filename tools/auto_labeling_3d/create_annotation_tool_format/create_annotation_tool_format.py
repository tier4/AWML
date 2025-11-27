#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Type

from tools.auto_labeling_3d.utils.dataclass.awml_info import AWML3DInfo
from tools.auto_labeling_3d.utils.dataset.annotation_tool_dataset import (
    AnnotationToolDataset,
    DeepenDataset,
    SegmentAIDataset,
)

DATASET_FACTORIES: Dict[str, Type[AnnotationToolDataset]] = {
    "deepen": DeepenDataset,
    "segment.ai": SegmentAIDataset,
}


def convert_pseudo_to_annotation_format(
    info_path: str | Path,
    output_dir: str | Path,
    output_format: str,
    dataset_id: str | None = None,
) -> None:
    """Convert pseudo label to the specified annotation tool format."""

    print(f"Load pseudo label from {info_path}")
    info = AWML3DInfo.load(info_path, dataset_id=dataset_id)

    if output_format not in DATASET_FACTORIES:
        supported = ", ".join(DATASET_FACTORIES.keys())
        raise ValueError(f"Unsupported output format: {output_format}. Supported formats: {supported}")

    dataset_cls = DATASET_FACTORIES[output_format]
    dataset = dataset_cls.load_from_info(info)
    dataset.save(output_dir)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert pseudo label to annotation tool format (e.g., Deepen, Segment.ai)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Pseudo Label pickle file (info.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        required=True,
        choices=["deepen", "segment.ai"],
        help="Output format for annotation tool (currently only 'deepen' is supported)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=False,
        default="Pseudo_DB_J6Gen2_v2_1_Shiojiri_kids_x2_dev_b3902d62-9777-496d-846b-39c3db7b9dcf_2025-04-09_11-34-42_11-35-02",
        help="Dataset ID to use for all annotations (overrides scene_name/sample_idx)",
    )
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    convert_pseudo_to_annotation_format(
        info_path=args.input, output_dir=args.output_dir, output_format=args.output_format, dataset_id=args.dataset_id
    )


if __name__ == "__main__":
    main()
