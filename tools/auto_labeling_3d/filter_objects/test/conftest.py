import logging
from typing import Any, Dict, List

import pytest


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return logging.getLogger("test_logger")


@pytest.fixture
def sample_classes() -> List[str]:
    """Sample class names for testing."""
    return ["car", "pedestrian", "bicycle"]


@pytest.fixture
def sample_confidence_thresholds() -> Dict[str, float]:
    """Sample confidence thresholds for testing."""
    return {"car": 0.5, "pedestrian": 0.3, "bicycle": 0.4}


@pytest.fixture
def sample_use_label() -> List[str]:
    """Sample use_label for testing."""
    return ["car", "pedestrian", "bicycle"]


@pytest.fixture
def sample_predicted_result_info(sample_classes) -> Dict[str, Any]:
    """Create sample predicted result info for testing.

    Contains 2 frames with various instances for testing different scenarios.
    """
    return {
        "metainfo": {"classes": sample_classes},
        "data_list": [
            # Frame 1
            {
                "pred_instances_3d": [
                    {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},  # car
                    {
                        "bbox_label_3d": 0,  # car
                        "bbox_score_3d": 0.3,  # below car threshold (0.5)
                        "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.2],
                    },
                    {
                        "bbox_label_3d": 1,  # pedestrian
                        "bbox_score_3d": 0.4,  # above pedestrian threshold (0.3)
                        "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0],
                    },
                    {
                        "bbox_label_3d": 2,  # bicycle
                        "bbox_score_3d": 0.2,  # below bicycle threshold (0.4)
                        "bbox_3d": [15.0, 16.0, 0.0, 1.8, 0.8, 1.2, 0.3],
                    },
                ]
            },
            # Frame 2
            {
                "pred_instances_3d": [
                    {
                        "bbox_label_3d": 0,  # car
                        "bbox_score_3d": 0.9,
                        "bbox_3d": [20.0, 21.0, 0.0, 4.0, 2.0, 1.5, 0.4],
                    },
                    {
                        "bbox_label_3d": 1,  # pedestrian
                        "bbox_score_3d": 0.1,  # below pedestrian threshold (0.3)
                        "bbox_3d": [25.0, 26.0, 0.0, 0.6, 0.6, 1.8, 0.0],
                    },
                ]
            },
        ],
    }


@pytest.fixture
def empty_predicted_result_info(sample_classes) -> Dict[str, Any]:
    """Create empty predicted result info for testing."""
    return {"metainfo": {"classes": sample_classes}, "data_list": [{"pred_instances_3d": []}]}


@pytest.fixture
def ensemble_weights() -> List[float]:
    """Sample ensemble weights for testing."""
    return [0.6, 0.4]


@pytest.fixture
def ensemble_settings(ensemble_weights) -> Dict[str, Any]:
    """Sample ensemble settings for testing."""
    return {
        "weights": ensemble_weights,
        "iou_threshold": 0.5,
        "ensemble_label_groups": [["car", "bicycle"], ["pedestrian"]],
    }


@pytest.fixture
def sample_ensemble_results(sample_classes) -> List[Dict[str, Any]]:
    """Create sample results from multiple models for ensemble testing."""
    return [
        # Model 1 results
        {
            "metainfo": {"classes": sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {
                            "bbox_label_3d": 0,  # car
                            "bbox_score_3d": 0.8,
                            "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1],
                        },
                        {
                            "bbox_label_3d": 1,  # pedestrian
                            "bbox_score_3d": 0.7,
                            "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0],
                        },
                    ]
                }
            ],
        },
        # Model 2 results
        {
            "metainfo": {"classes": sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {
                            "bbox_label_3d": 0,  # car - overlapping with model 1
                            "bbox_score_3d": 0.6,
                            "bbox_3d": [1.5, 2.5, 0.0, 4.0, 2.0, 1.5, 0.15],
                        },
                        {
                            "bbox_label_3d": 2,  # bicycle - non-overlapping
                            "bbox_score_3d": 0.5,
                            "bbox_3d": [20.0, 21.0, 0.0, 1.8, 0.8, 1.2, 0.2],
                        },
                    ]
                }
            ],
        },
    ]
