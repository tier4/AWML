from typing import Dict, Optional, Sequence, Tuple, List
from mmpretrain.registry import DATASETS
from mmpretrain.datasets import BaseDataset
import json
import random
from collections import Counter


def read_json_file(file_path):
    """
    Read a JSON file and return its contents as a Python dictionary.

    Args:
    file_path (str): The path to the JSON file.

    Returns:
    dict: The contents of the JSON file as a Python dictionary.

    Raises:
    FileNotFoundError: If the specified file is not found.
    json.JSONDecodeError: If the file is not valid JSON.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not valid JSON.")

    return {}


@DATASETS.register_module()
class TLRClassificationDataset(BaseDataset):

    def __init__(self,
                 *args,
                 false_instances: int = 0,
                 false_label: int = -1,
                 min_size=0,
                 **kwargs) -> None:
        self.false_instances = false_instances
        self.false_label = false_label
        self.min_size = min_size
        if self.false_instances > 0:
            assert self.false_label >= 0, "false_label must be non-negative when false_instances > 0"
        super().__init__(*args, **kwargs)

    def _check_bbox_overlap(self, bbox1: List[float],
                            bbox2: List[float]) -> bool:
        """Check if two bboxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        x2_min, y2_min, x2_max, y2_max = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min
                    or y2_max < y1_min)

    def _generate_random_bbox(self,
                              img_width: int,
                              img_height: int,
                              existing_bboxes: List[List[float]],
                              max_attempts: int = 100) -> List[float]:
        """Generate a random bbox that doesn't overlap with existing ones."""
        # Assume reasonable bbox size range (e.g., 5-20% of image dimension)
        min_size = min(img_width, img_height) * 0.05
        max_size = min(img_width, img_height) * 0.2

        for _ in range(max_attempts):
            # Generate random width and height
            w = random.uniform(min_size, max_size)
            h = random.uniform(min_size, max_size)

            # Generate random position
            x = random.uniform(0, img_width - w)
            y = random.uniform(0, img_height - h)

            new_bbox = [x, y, x + w, y + h]

            # Check if the new bbox overlaps with any existing bbox
            overlap = False
            for existing_bbox in existing_bboxes:
                if self._check_bbox_overlap(new_bbox, existing_bbox):
                    overlap = True
                    break

            if not overlap:
                return new_bbox

        raise RuntimeError(
            "Could not generate non-overlapping bbox after maximum attempts")

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()
        separated_data_list = []
        img_id = 0

        for data_info in data_list:
            instances = data_info.get('instances', [])
            existing_bboxes = [instance["bbox"] for instance in instances]

            # Add original instances
            for instance in instances:
                single_instance_info = data_info.copy()
                single_instance_info["img_id"] = img_id
                single_instance_info["bbox"] = instance["bbox"]
                single_instance_info["gt_label"] = instance["bbox_label"]
                separated_data_list.append(single_instance_info)

            if self.false_instances > 0:
                # add false examples to training set, not used by default.

                num_false = max(len(instances), 1) * self.false_instances
                img_width = data_info.get('width')
                img_height = data_info.get('height')

                for _ in range(num_false):
                    try:
                        false_bbox = self._generate_random_bbox(
                            img_width, img_height, existing_bboxes)

                        false_instance_info = data_info.copy()
                        false_instance_info["img_id"] = img_id
                        false_instance_info["bbox"] = false_bbox
                        false_instance_info["gt_label"] = self.false_label
                        separated_data_list.append(false_instance_info)

                        # Add this false bbox to existing_bboxes to prevent overlap with future false instances
                        existing_bboxes.append(false_bbox)
                    except RuntimeError as e:
                        print(
                            f"Warning: {e} for image {img_id}. Skipping remaining false instances."
                        )
                        break

            img_id += 1
        class_infos = [x['gt_label'] for x in separated_data_list]
        label_counts = Counter(class_infos)
        print("\nClass Distribution:")
        print("-" * 30)
        for label, count in label_counts.items():
            print(f"Class {self.CLASSES[label]}: {count:,d} samples")
        print("-" * 30)
        print(f"Total: {len(class_infos):,d} samples")
        return separated_data_list
