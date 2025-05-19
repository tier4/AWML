import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pytest
from attr import asdict
from attrs import define, field

from tools.auto_labeling_3d.filter_objects.ensemble.ensemble_model import EnsembleModel


@define
class PredInstance3D:
    """Data class for 3D predicted instance."""

    bbox_3d: List[float]
    velocity: List[float]
    instance_id_3d: str
    bbox_label_3d: int
    bbox_score_3d: float

    def asdict(self) -> Dict:
        """Convert object to dictionary."""
        return asdict(self)


# Test fixtures
@pytest.fixture
def ensemble_model():
    """Return an EnsembleModel instance for testing."""
    ensemble_setting = {
        "weights": [0.7, 0.3],
        "iou_threshold": 0.5,
        "skip_box_threshold": 0.2,
        "label": ["car", "truck", "bus", "bicycle", "pedestrian"],
    }
    logger = logging.getLogger("test_ensemble")
    return EnsembleModel(
        ensemble_setting=ensemble_setting,
        logger=logger,
    )


@pytest.fixture
def pred_instances():
    """Return a list of predicted instances for testing."""
    return [
        PredInstance3D(
            bbox_3d=[1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 0.0],
            velocity=[1.0, 0.0],
            instance_id_3d="1",
            bbox_label_3d=0,
            bbox_score_3d=0.9,
        ),
        PredInstance3D(
            bbox_3d=[1.1, 2.1, 3.1, 2.0, 2.0, 2.0, 0.0],
            velocity=[0.0, 1.0],
            instance_id_3d="2",
            bbox_label_3d=0,
            bbox_score_3d=0.8,
        ),
        PredInstance3D(
            bbox_3d=[10.0, 10.0, 3.0, 2.0, 2.0, 2.0, 0.0],
            velocity=[1.0, 0.0],
            instance_id_3d="3",
            bbox_label_3d=1,
            bbox_score_3d=0.85,
        ),
        PredInstance3D(
            bbox_3d=[10.1, 10.1, 3.0, 2.0, 2.0, 2.0, 0.0],
            velocity=[0.0, 1.0],
            instance_id_3d="4",
            bbox_label_3d=1,
            bbox_score_3d=0.75,
        ),
    ]


@pytest.fixture
def create_result_with_instances():
    """Return a function that creates predicted results with specified instances."""

    def _create_result(instances: List[PredInstance3D], metainfo: Optional[Dict] = None) -> Dict:
        """Create test predicted results with specified instances."""
        return {
            "metainfo": metainfo or {"test": "data"},
            "data_list": [
                {
                    "frame_id": 0,
                    "pred_instances_3d": [instance.asdict() for instance in instances],
                }
            ],
        }

    return _create_result


def test_ensemble_single_result(ensemble_model, pred_instances, create_result_with_instances):
    """Test ensembling a single result."""
    # Create a single result
    result = create_result_with_instances([pred_instances[0]])
    # Ensemble should return the input result unchanged
    assert ensemble_model.ensemble([result]) == result


def test_ensemble_multiple_results(ensemble_model, pred_instances, create_result_with_instances):
    """Test ensembling multiple results."""
    # Create test data
    results = [
        create_result_with_instances([pred_instances[0]]),
        create_result_with_instances([pred_instances[1]], metainfo={"test": "data2"}),
    ]

    # Call the ensemble method
    result = ensemble_model.ensemble(results)
    pred_instances_result = result["data_list"][0]["pred_instances_3d"]

    # metainfo should not be changed
    assert result["metainfo"] == results[0]["metainfo"]
    assert len(result["data_list"]) == 1
    assert "pred_instances_3d" in result["data_list"][0]

    # After NMS, only one box should remain
    assert len(pred_instances_result) == 1

    # The box with the highest weighted score should be kept
    # Model 1: 0.9 * 0.7 = 0.63
    # Model 2: 0.8 * 0.3 = 0.24
    # Model 1 has a higher weighted score, so its box should be kept
    assert pytest.approx(pred_instances_result[0]["bbox_score_3d"], abs=1e-5) == 0.63
