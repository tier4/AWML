"""
Change directory structure for t4dataset.

This script can add a numbered directory (0) to each scene (default behavior)

To reverse the operation (remove numbered directory structure), use the `--annotated-to-non-annotated` flag:

```sh
python tools/auto_labeling_3d/change_directory_structure/change_directory_structure.py --dataset_dir data/t4dataset/pseudo_xx1/ --annotated-to-non-annotated
```
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

from tools.auto_labeling_3d.utils.logger import setup_logger


def move_contents_to_numbered_dir(scene_dir: Path, logger: logging.Logger, version_dir_name: str) -> None:
    """Move all contents of scene directory to a numbered subdirectory (0)."""
    num_dir = scene_dir / version_dir_name

    # Create the numbered directory if it doesn't exist
    num_dir.mkdir(exist_ok=True)

    # Move all contents except the newly created numbered directory
    for item in scene_dir.iterdir():
        if item.name != version_dir_name:
            logger.info(f"  Moving {item.name} to {version_dir_name}/")
            shutil.move(str(item), str(num_dir / item.name))


def move_contents_from_numbered_dir(scene_dir: Path, logger: logging.Logger, version_dir_name: str) -> None:
    """Move all contents from numbered subdirectory (0) back to scene directory."""
    num_dir = scene_dir / version_dir_name

    if not num_dir.exists() or not num_dir.is_dir():
        logger.warning(f"  {num_dir} does not exist, skipping...")
        return

    # Move all contents from numbered directory to parent
    for item in num_dir.iterdir():
        target_path = scene_dir / item.name
        if target_path.exists():
            logger.warning(f"  {target_path} already exists, skipping {item.name}")
            continue
        logger.info(f"  Moving {item.name} from {version_dir_name}/")
        shutil.move(str(item), str(target_path))

    # Remove the now-empty numbered directory
    try:
        num_dir.rmdir()
        logger.info(f"  Removed empty directory {version_dir_name}/")
    except Exception as e:
        raise OSError(f"  Could not remove directory {version_dir_name}/: {e}")


def process_dataset(
    dataset_dir: Path, logger: logging.Logger, annotated_to_non_annotated: bool = False, version_dir_name: str = "0"
) -> None:
    """Process the dataset directory structure."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Directory '{dataset_dir}' does not exist.")

    if not dataset_dir.is_dir():
        raise RuntimeError(f"'{dataset_dir}' is not a directory.")

    operation = "Removing numbered directories" if annotated_to_non_annotated else "Adding numbered directories"
    logger.info(f"{operation} for {dataset_dir}...")

    # Process each directory in the dataset directory
    for scene_dir in dataset_dir.iterdir():
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name
        logger.info(f"Processing {scene_name}...")

        if annotated_to_non_annotated:
            move_contents_from_numbered_dir(scene_dir, logger, version_dir_name)
        else:
            move_contents_to_numbered_dir(scene_dir, logger, version_dir_name)

    logger.info("Directory structure changed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Change directory structure for t4dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          %(prog)s data/t4dataset/pseudo_xx1/
          %(prog)s data/t4dataset/pseudo_xx1/ --annotated-to-non-annotated
                """,
    )

    parser.add_argument(
        "--dataset_dir", type=Path, help="Path to the t4dataset directory (e.g., data/t4dataset/pseudo_xx1/)"
    )
    parser.add_argument(
        "--annotated-to-non-annotated",
        action="store_true",
        help="Remove numbered directory structure (reverse operation)",
    )
    parser.add_argument(
        "--version-dir-name",
        type=str,
        default="0",
        help="Name of the numbered subdirectory to use (default: '0')",
    )
    parser.add_argument(
        "--log-level",
        help="Set log level",
        default="INFO",
        choices=list(logging._nameToLevel.keys()),
    )
    parser.add_argument(
        "--work-dir",
    )
    args = parser.parse_args()

    logger: logging.Logger = setup_logger(args, name="create_pseudo_t4dataset")

    process_dataset(
        args.dataset_dir,
        logger,
        args.annotated_to_non_annotated,
        args.version_dir_name,
    )


if __name__ == "__main__":
    main()
