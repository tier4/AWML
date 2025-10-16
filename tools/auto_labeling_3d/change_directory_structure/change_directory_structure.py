#!/usr/bin/env python3
"""
Change directory structure for t4dataset.

This script can add a numbered directory (0) to each scene (default behavior)

To reverse the operation (remove numbered directory structure), use the `--annotated-to-non-annotated` flag:

```sh
python tools/auto_labeling_3d/change_directory_structure/change_directory_structure.py --dataset_dir data/t4dataset/pseudo_xx1/ --annotated-to-non-annotated
```
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def move_contents_to_numbered_dir(scene_dir: Path) -> None:
    """Move all contents of scene directory to a numbered subdirectory (0)."""
    num_dir = scene_dir / "0"
    
    # Create the numbered directory if it doesn't exist
    num_dir.mkdir(exist_ok=True)
    
    # Move all contents except the newly created numbered directory
    for item in scene_dir.iterdir():
        if item.name != "0":
            print(f"  Moving {item.name} to 0/")
            shutil.move(str(item), str(num_dir / item.name))


def move_contents_from_numbered_dir(scene_dir: Path) -> None:
    """Move all contents from numbered subdirectory (0) back to scene directory."""
    num_dir = scene_dir / "0"
    
    if not num_dir.exists() or not num_dir.is_dir():
        print(f"  Warning: {num_dir} does not exist, skipping...")
        return
    
    # Move all contents from numbered directory to parent
    for item in num_dir.iterdir():
        target_path = scene_dir / item.name
        if target_path.exists():
            print(f"  Warning: {target_path} already exists, skipping {item.name}")
            continue
        print(f"  Moving {item.name} from 0/")
        shutil.move(str(item), str(target_path))
    
    # Remove the now-empty numbered directory
    try:
        num_dir.rmdir()
        print(f"  Removed empty directory 0/")
    except OSError as e:
        print(f"  Warning: Could not remove directory 0/: {e}")


def process_dataset(dataset_dir: Path, annotated_to_non_annotated: bool = False) -> None:
    """Process the dataset directory structure."""
    if not dataset_dir.exists():
        print(f"Error: Directory '{dataset_dir}' does not exist.")
        sys.exit(1)
    
    if not dataset_dir.is_dir():
        print(f"Error: '{dataset_dir}' is not a directory.")
        sys.exit(1)
    
    operation = "Removing numbered directories" if annotated_to_non_annotated else "Adding numbered directories"
    print(f"{operation} for {dataset_dir}...")
    
    # Process each directory in the dataset directory
    for scene_dir in dataset_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        print(f"Processing {scene_name}...")
        
        if annotated_to_non_annotated:
            move_contents_from_numbered_dir(scene_dir)
        else:
            move_contents_to_numbered_dir(scene_dir)
    
    print("Directory structure changed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Change directory structure for t4dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          %(prog)s data/t4dataset/pseudo_xx1/
          %(prog)s data/t4dataset/pseudo_xx1/ --annotated-to-non-annotated
                """
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        help="Path to the t4dataset directory (e.g., data/t4dataset/pseudo_xx1/)"
    )
    
    parser.add_argument(
        "--annotated-to-non-annotated",
        action="store_true",
        help="Remove numbered directory structure (reverse operation)"
    )
    
    args = parser.parse_args()
    
    try:
        process_dataset(args.dataset_dir, args.annotated_to_non_annotated)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
