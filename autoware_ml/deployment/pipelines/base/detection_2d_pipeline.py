"""
2D Object Detection Pipeline Base Class.
This module provides the base class for 2D object detection pipelines,
implementing common preprocessing and postprocessing for models like YOLOX, YOLO, etc.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class Detection2DPipeline(BaseDeploymentPipeline):
    """
    Base class for 2D object detection pipelines.

    Provides common functionality for 2D detection tasks including:
    - Image preprocessing (resize, normalize, padding)
    - Postprocessing (NMS, coordinate transformation)
    - Standard detection output format

    Expected output format:
        List[Dict] where each dict contains:
        {
            'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
            'score': float,             # Confidence score
            'class_id': int,            # Class ID
            'class_name': str           # Class name (optional)
        }
    """

    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        num_classes: int = 80,
        class_names: List[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        backend_type: str = "unknown",
    ):
        """
        Initialize 2D detection pipeline.

        Args:
            model: Model object
            device: Device for inference
            num_classes: Number of classes
            class_names: List of class names
            input_size: Model input size (height, width)
            backend_type: Backend type
        """
        super().__init__(model, device, task_type="detection_2d", backend_type=backend_type)

        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.input_size = input_size

    @abstractmethod
    def preprocess(self, input_data: Any, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess input data for 2D detection.

        This method should be implemented by specific detection pipelines.
        For YOLOX, preprocessing is done by MMDetection pipeline before calling this method.

        Args:
            input_data: Preprocessed tensor from MMDetection pipeline or raw input
            **kwargs: Additional preprocessing parameters

        Returns:
            Tuple of (preprocessed_tensor, preprocessing_metadata)
            - preprocessed_tensor: [1, C, H, W]
            - preprocessing_metadata: Dict with preprocessing information
        """
        pass

    @abstractmethod
    def run_model(self, preprocessed_input: torch.Tensor) -> Any:
        """
        Run detection model (backend-specific).

        Args:
            preprocessed_input: Preprocessed tensor [1, C, H, W]

        Returns:
            Model output (backend-specific format)
        """
        pass

    def postprocess(self, model_output: Any, metadata: Dict = None) -> List[Dict]:
        """
        Standard 2D detection postprocessing.

        Steps:
        1. Parse model outputs (boxes, scores, classes)
        2. Apply NMS
        3. Transform coordinates back to original image space
        4. Filter by confidence threshold

        Args:
            model_output: Raw model output
            metadata: Preprocessing metadata

        Returns:
            List of detections in standard format
        """
        # This should be overridden by specific detectors (YOLOX, YOLO, etc.)
        # as output formats differ
        raise NotImplementedError("postprocess() must be implemented by specific detector pipeline")

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
        """
        Non-Maximum Suppression.

        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            iou_threshold: IoU threshold for NMS

        Returns:
            Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)
