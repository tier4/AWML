import os
import random
from typing import Dict, Optional, Tuple

import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CropROI(BaseTransform):
    """Crops an image based on a single bounding box (ROI) with optional random offset.

    This transform crops a portion of the image based on a provided bounding box.
    If offset is specified, the crop area can be randomly expanded by (0,offset) pixels
    on each side while staying within image boundaries.

    Attributes:
        min_size (int): Minimum size (in pixels) of the bounding box to be retained after cropping.
            Default is 5.
        offset (int): Maximum random expansion of crop area on each side. Default is None.

    Args:
        min_size (int): Minimum size for a valid bounding box after cropping.
        offset (int, optional): Maximum random expansion on each side. Default: None.

    Example:
        ```python
        original image
             ________________________________
            |                                |
            |             random offset      |    cropped image
            |            ____________        |    ____________
            |           |            |       | -> |          |
            |           |   bbox     |       |    |   bbox   |
            |           |            |       |    |          |
            |           |____________|       |    |__________|
            |                                |
            |________________________________|
        ```
    """

    def __init__(
        self,
        min_size: int = 5,
        offset: Optional[int] = None,
    ) -> None:
        self.min_size = min_size
        self.offset = offset

    @staticmethod
    def calculate_roi_crop(
        bbox: np.ndarray, img_h: int, img_w: int, offset: Optional[int] = None
    ) -> Tuple[int, int, int, int]:
        """Calculate the ROI crop based on the provided bounding box with optional random offset.

        Args:
            bbox (np.ndarray): Original bounding box coordinates [x1, y1, x2, y2].
            img_h (int): Image height.
            img_w (int): Image width.
            offset (int, optional): Maximum random expansion on each side.

        Returns:
            Tuple[int, int, int, int]: Cropping coordinates (y1, x1, y2, x2).
        """
        if offset is not None:
            w = max(int((bbox[2] - bbox[0]) * offset), 1)
            h = max(int((bbox[1] - bbox[0]) * offset), 1)
            # Generate random offsets for each side
            top_offset = np.random.randint(0, h)
            left_offset = np.random.randint(0, w)
            bottom_offset = np.random.randint(0, h)
            right_offset = np.random.randint(0, w)

            # Apply offsets while ensuring we stay within image boundaries
            crop_y1 = max(0, int(bbox[1]) - top_offset)
            crop_x1 = max(0, int(bbox[0]) - left_offset)
            crop_y2 = min(img_h, int(bbox[3]) + bottom_offset)
            crop_x2 = min(img_w, int(bbox[2]) + right_offset)
        else:
            # Original behavior without offset
            crop_y1 = max(0, int(bbox[1]))
            crop_x1 = max(0, int(bbox[0]))
            crop_y2 = min(img_h, int(bbox[3]))
            crop_x2 = min(img_w, int(bbox[2]))

        return crop_y1, crop_x1, crop_y2, crop_x2

    def transform(self, results: Dict[str, any]) -> Optional[Dict[str, any]]:
        img = results["img"]
        img_h, img_w = img.shape[:2]

        if "bbox" in results:
            bbox = np.array(results["bbox"])
            crop_y1, crop_x1, crop_y2, crop_x2 = self.calculate_roi_crop(bbox, img_h, img_w, self.offset)
        else:
            raise ValueError("Bounding box (bbox) is required for CropROI transform.")

        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape

        # Call the function to save the image with bounding box for debugging
        # self.save_cropped_image_with_bbox(results)

        return results

    def save_cropped_image_with_bbox(self, results: Dict[str, any]) -> None:
        """Save the cropped image with bounding boxes plotted to the ./work_dirs/testimages directory. Used for debugging."""
        img = results["img"]
        os.makedirs("/workspace/work_dirs/testimages/CropROI", exist_ok=True)

        filename = f"/workspace/work_dirs/testimages/CropROI/cropped_img_{np.random.randint(1000)}.png"
        cv2.imwrite(filename, img)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(min_size={self.min_size}, offset={self.offset})"
        return repr_str


@TRANSFORMS.register_module()
class CustomResizeRotate(BaseTransform):
    """Randomly resize and rotate an image within specified size limits,
    padding with 0s where necessary after rotation.

    Args:
        min_size (int): Minimum size for the image's width and height after transformation.
        max_size (int): Maximum size for the image's width and height after transformation.
        rotation_range (tuple): Range of rotation angles in degrees, e.g., (-45, 45).
        padding_value (tuple): Padding value for rotated image.

    """

    def __init__(self, min_size=25, max_size=224, rotation_range=(-15, 15), padding_value=None):
        self.min_size = min_size
        self.max_size = max_size
        self.rotation_range = rotation_range
        self.padding_value = padding_value if padding_value else [np.random.randint(0, 255) for _ in range(3)]

    def rotate_and_resize(self, image):
        """Rotate the image randomly within the specified range, then resize to fit bounds."""

        # Step 1: Randomly rotate the image with padding as necessary
        h, w = image.shape[:2]
        angle = random.uniform(*self.rotation_range)
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions to fit the rotated image
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to account for translation to the new bounds
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]

        # Rotate and pad the image with black (0) borders
        rotated_image = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h), borderValue=self.padding_value)

        # Step 2: Resize to ensure dimensions are within min_size and max_size bounds
        final_h, final_w = rotated_image.shape[:2]
        target = np.random.randint(self.min_size, self.max_size)
        scale = target / max(final_w, final_h)
        # Resize to the calculated target dimensions
        target_h, target_w = max(int(final_h * scale), 1), max(int(final_w * scale), 1)
        final_image = cv2.resize(rotated_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return final_image

    def transform(self, results: dict) -> dict:
        """Apply the rotation and resize transformation to the image in results."""
        results["img"] = self.rotate_and_resize(results["img"])
        results["img_shape"] = results["img"].shape[:2]
        # self.save_cropped_image_with_bbox(results)
        return results

    def save_cropped_image_with_bbox(self, results: Dict[str, any]) -> None:
        """Save the cropped image with bounding boxes plotted to the ./work_dirs/testimages directory. Used for debugging."""
        img = results["img"]
        os.makedirs("/workspace/work_dirs/testimages/CustomResizeRotate", exist_ok=True)

        filename = f"/workspace/work_dirs/testimages/CustomResizeRotate/cropped_img_{np.random.randint(1000)}.png"
        cv2.imwrite(filename, img)
