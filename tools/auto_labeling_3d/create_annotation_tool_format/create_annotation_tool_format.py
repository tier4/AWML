#!/usr/bin/env python3
import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Type

import yaml

from tools.auto_labeling_3d.utils.dataclass.awml_info import AWML3DInfo
from tools.auto_labeling_3d.utils.dataset.annotation_tool_dataset import (
    AnnotationToolDataset,
    DeepenDataset,
    SegmentsAIDataset,
)

ANNOTATION_TOOL_CLASSES: dict[str, Type[AnnotationToolDataset]] = {
    "deepen": DeepenDataset,
    "segments.ai": SegmentsAIDataset,
}


def create_annotation_tool_format(
    info_path: str | Path,
    output_dir: str | Path,
    output_format: str,
    logger: logging.Logger,
    dataname_to_anntool_id: Optional[dict[str, str]] = None,
    annotation_tool_classes: dict[str, Type[AnnotationToolDataset]] = ANNOTATION_TOOL_CLASSES,
) -> list[AnnotationToolDataset]:
    """Load info.pkl, create annotation tool format files for each dataset, and return the results.

    This function is the central executor for the conversion process.

    Args:
        info_path (str | Path): Path to the input pickle file (info.pkl).
        output_dir (str | Path): Directory to save the output annotation tool format files.
        output_format (str): The format for the annotation tool (e.g., 'deepen', 'segment.ai').
        logger (logging.Logger): The logger to use for logging messages.
        dataname_to_anntool_id (Optional[dict[str, str]]):
            An optional dictionary mapping non annotated dataset names to annotation tool format IDs.
            If not provided, new UUIDs will be generated.
        annotation_tool_classes (dict[str, Type[AnnotationToolDataset]]):
            A dictionary mapping format names to dataset classes.

    Returns:
        list[AnnotationToolDataset]: A list of AnnotationToolDataset objects.
    """
    # 1. Load and group data from info.pkl in a single pass
    logger.info(f"Loading and grouping data from: {info_path}")
    awml_datasets: list[AWML3DInfo] = AWML3DInfo.load(info_path=info_path)

    # 2. Prepare the dataset name to tool ID mapping
    if not dataname_to_anntool_id:
        logger.info("ID mapping file not provided. Generating UUIDs for each dataset.")
        dataname_to_anntool_id = {ds.t4_dataset_name: str(uuid.uuid4()) for ds in awml_datasets}

    # 3. Execute conversion and saving for each dataset
    annotation_tool_format_datasets: list[AnnotationToolDataset] = []
    dataset_cls = annotation_tool_classes[output_format]

    for dataset_info in awml_datasets:
        dataset_name = dataset_info.t4_dataset_name
        if dataset_name not in dataname_to_anntool_id:
            logger.warning(f"Dataset '{dataset_name}' has no ID mapping. Skipping.")
            continue

        ann_tool_id = dataname_to_anntool_id[dataset_name]

        annotation_tool_format_dataset = dataset_cls.create_from_info(
            info=dataset_info,
            output_dir=Path(output_dir),
            t4_dataset_name=dataset_name,
            ann_tool_id=ann_tool_id,
        )
        annotation_tool_format_datasets.append(annotation_tool_format_dataset)

    return annotation_tool_format_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert pseudo label to annotation tool format (e.g., Deepen, Segments.ai)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Pseudo Label pickle file (info.pkl) containing one or more datasets.",
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
        choices=["deepen", "segments.ai"],
        help="Output format for annotation tool (currently only 'deepen' is supported)",
    )
    parser.add_argument(
        "--dataname-to-anntool-id",
        type=str,
        default=None,
        help="Path to a YAML file mapping non-annotated dataset names to annotation tool format IDs.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logging.basicConfig(level=args.log_level.upper())
    logger = logging.getLogger(__name__)

    dataname_to_anntool_id = None
    if args.dataname_to_anntool_id:
        logger.info(f"Loading ID mapping from: {args.dataname_to_anntool_id}")
        with open(args.dataname_to_anntool_id, "r") as f:
            dataname_to_anntool_id = yaml.safe_load(f)

    result_datasets = create_annotation_tool_format(
        info_path=args.input,
        output_dir=args.output_dir,
        output_format=args.output_format,
        logger=logger,
        dataname_to_anntool_id=dataname_to_anntool_id,
    )

    # Format the final output as a dictionary of {tool_id: path}
    id_to_path_map = {ds.ann_tool_id: str(ds.ann_tool_file_path) for ds in result_datasets}
    logger.info("--- Conversion Result ---")
    logger.info(json.dumps(id_to_path_map, indent=4))


if __name__ == "__main__":
    main()
