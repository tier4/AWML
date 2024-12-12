import os
import random
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS, VISUALIZERS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type


@TRANSFORMS.register_module()
class RandomCropWithROI(BaseTransform):
    """Randomly crops an image, bounding boxes, and masks with a focus on a region of interest (ROI).

    This transform crops a portion of the image based on the provided ROI (like a bounding box)
    or a random patch if no ROI is available. During cropping, bounding boxes are adjusted to
    fit within the new cropped region. The crop size is determined either as a relative ratio or
    absolute value and can be tuned based on the aspect ratio or image size.

    Attributes:
        crop_size (Tuple[float, float]): The relative ratio or absolute size (height, width)
            of the crop. Default is (1.0, 20.0).
        process_target_only (bool): If True, only processes and saves the target bounding box and label.
        hw_ratio (Tuple[float, float]): The ratio of height to width for the crop. Default is (0.5, 1.0).
        min_size (int): Minimum size (in pixels) of the bounding boxes to be retained after cropping.
            Default is 5.

    Args:
        crop_size (Tuple[float, float]): Specifies the relative ratio or absolute size of the crop.
        hw_ratio (Tuple[float, float]): Defines the height-to-width ratio for the crop area.
        process_target_only (bool): Whether to process and save only the target bounding box and label.
        min_size (int): Minimum size for valid bounding boxes after cropping.

    Example:
        ```
        original image
             ________________________________
            |                                |
            |                 ROI=`gt_bbox`  |    cropped image
            |                  ________      |      ________
            |                 |        |     | ->  |        |
            |                 |________|     |     |________|
            |                                |
            |________________________________|
        ```

    Methods:
        calculate_roi_crop: Computes the cropped coordinates based on the given ROI and image size.
        calculate_random_crop: Randomly selects a crop when no ROI is available.
        transform: Performs the image crop, adjusts bounding boxes, and updates results.
        save_cropped_image_with_bbox: Saves the cropped image with bounding boxes drawn (for debugging).
        __repr__: Returns a string representation of the transformation.
    """

    def __init__(
        self,
        crop_size: Tuple[float, float] = (1.0, 20.0),
        hw_ratio: Tuple[float, float] = (0.5, 1.0),
        process_target_only: bool = False,
        min_size: int = 5,
    ) -> None:
        assert isinstance(crop_size, tuple) and len(crop_size) == 2
        assert isinstance(hw_ratio, tuple) and len(hw_ratio) == 2
        self.crop_size = crop_size
        self.hw_ratio = hw_ratio
        self.process_target_only = process_target_only
        self.min_size = min_size

    @staticmethod
    def calculate_roi_crop(
        bbox: np.ndarray,
        img_h: int,
        img_w: int,
        crop_size: Tuple[float, float],
        hw_ratio: Tuple[float, float],
    ) -> Tuple[int, int, int, int]:
        """Calculate the ROI crop based on the provided bounding box."""
        bbox_w: int = bbox[2] - bbox[0]
        crop_w = bbox_w * np.random.uniform(*crop_size) + 1
        hw_ratio_val = np.random.uniform(*hw_ratio)
        crop_h = hw_ratio_val * crop_w + 1
        crop_size = np.array([crop_h, crop_w], dtype=np.int64)

        roi_center = ((bbox[2:] + bbox[:2]) / 2.0 + 1.0)[::-1]

        rand_box_center = roi_center + 0.5 * crop_size - np.random.rand(2) * crop_size
        crop_y1, crop_x1 = rand_box_center - crop_size * 0.5
        crop_y2, crop_x2 = rand_box_center + crop_size * 0.5
        crop_y1, crop_x1 = max(0, int(crop_y1)), max(0, int(crop_x1))
        crop_y2, crop_x2 = min(img_h, int(crop_y2)), min(img_w, int(crop_x2))

        return crop_y1, crop_x1, crop_y2, crop_x2

    @staticmethod
    def calculate_random_crop(img_h: int, img_w: int):
        crop_h = int(np.random.uniform(0.1, 0.5) * img_h)
        crop_w = int(np.random.uniform(0.1, 0.5) * img_w)
        crop_y1 = np.random.randint(0, img_h - crop_h)
        crop_x1 = np.random.randint(0, img_w - crop_w)
        crop_y2 = crop_y1 + crop_h
        crop_x2 = crop_x1 + crop_w
        return crop_y1, crop_x1, crop_y2, crop_x2

    @autocast_box_type()
    def transform(self, results: Dict[str, any]) -> Optional[Dict[str, any]]:
        img = results["img"]
        img_h, img_w = img.shape[:2]
        instances = results.get("instances", [])

        if "crop_coords" in results:
            # Used for test and validation set, when the same crop coordinates are used
            crop_y1, crop_x1, crop_y2, crop_x2 = results["crop_coords"]
        elif len(instances) > 0:
            # During training, if there is more than one instance, randomly choose one of the bboxes
            selected_instance = random.choice(instances)
            bbox = np.array(selected_instance["bbox"])
            crop_y1, crop_x1, crop_y2, crop_x2 = self.calculate_roi_crop(
                bbox, img_h, img_w, self.crop_size, self.hw_ratio
            )
        else:
            # During training, if there are no instances, crop out a random patch
            crop_y1, crop_x1, crop_y2, crop_x2 = self.calculate_random_crop(img_h, img_w)

        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results["img"] = img
        cropped_img_h, cropped_img_w = img.shape[:2]

        if instances:
            gt_boxes = []
            gt_labels = []
            ignore_flag = []

            for instance in instances:
                bbox = np.array(instance["bbox"])
                bbox[0::2] -= crop_x1
                bbox[1::2] -= crop_y1
                bbox[0::2] = np.clip(bbox[0::2], 0, cropped_img_w)
                bbox[1::2] = np.clip(bbox[1::2], 0, cropped_img_h)
                instance["bbox"] = bbox.tolist()
                gt_boxes.append(instance["bbox"])
                gt_labels.append(instance["bbox_label"])
                ignore_flag.append(instance["ignore_flag"] == 1)

            valid_indices = []
            for idx, instance in enumerate(instances):
                bbox = np.array(instance["bbox"])
                if (
                    (bbox[2] > bbox[0])
                    and (bbox[2] - bbox[0]) > self.min_size
                    and (bbox[3] > bbox[1])
                    and (bbox[3] - bbox[1]) > self.min_size
                ):
                    valid_indices.append(idx)

            instances = [instances[idx] for idx in valid_indices]

            results["gt_bboxes"] = HorizontalBoxes(
                [gt_boxes[idx] for idx in valid_indices],
                dtype=torch.float32,
            )
            results["gt_bboxes_labels"] = np.array(
                [gt_labels[idx] for idx in valid_indices],
                dtype=np.int64,
            )
            results["gt_ignore_flags"] = np.array(
                [ignore_flag[idx] for idx in valid_indices],
                dtype=bool,
            )
        results["instances"] = instances
        results["img_shape"] = img_shape
        results["ori_shape"] = img_shape
        # Call the function to save the image with bounding boxes
        # self.save_cropped_image_with_bbox(results)

        return results

    def save_cropped_image_with_bbox(self, results: Dict[str, any]) -> None:
        """Save the cropped image with bounding boxes plotted to the ./work_dirs/testimages directory. Used for debugging."""
        img = results["img"]
        # Create the tmp directory if it doesn't exist
        os.makedirs("/workspace/work_dirs/testimages", exist_ok=True)

        # Draw the bounding boxes on the image
        for bbox in results["gt_bboxes"]:
            bbox = [int(x) for x in bbox.numpy()[0]]
            cv2.rectangle(
                img,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=(0, 255, 0),
                thickness=2,
            )

        # Generate a random filename
        filename = f"/workspace/work_dirs/testimages/cropped_img_{random.randint(0, 100)}.png"

        # Save the image
        cv2.imwrite(filename, img)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (
            f"(crop_size={self.crop_size}, hw_ratio={self.hw_ratio}, process_target_only={self.process_target_only})"
        )
        return repr_str

    @staticmethod
    def __get_random_seed(seed_data: List[any]) -> int:
        """A helper method to generate a consistent random seed from a list of seed data."""
        return hash(tuple(seed_data)) % 2**32


@VISUALIZERS.register_module()
class Visualizer:

    def __init__(self) -> None:
        self.root_dir = "/tmp/visualizer"
        shutil.rmtree(self.root_dir, ignore_errors=True)
        os.makedirs(self.root_dir, exist_ok=True)

    def __call__(self, results: Dict[str, any]) -> Optional[Dict[str, any]]:
        filename = os.path.basename(results["filename"])
        save_path = os.path.join(self.root_dir, filename)
        image = results["img"].copy()
        # bboxes = results["gt_bboxes"]
        bboxes = results["img_info"]["ann"]["bboxes"]
        for bbox in bboxes:
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))
            image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image)
        return results
