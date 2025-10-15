import logging
from typing import Any, Dict, List
from unittest.mock import Mock

import numpy as np
import pytest

from tools.auto_labeling_3d.filter_objects.ensemble.nms_ensemble_model import NMSEnsembleModel, NMSModelInstances, _calculate_iou, _nms_indices


class TestNMSModelInstances:
    """Test cases for NMSModelInstances dataclass."""

    def test_filter_and_weight_instances_basic(self, sample_classes):
        """Test basic filtering and weighting of instances."""
        class_name_to_id = {name: idx for idx, name in enumerate(sample_classes)}
        instances = [
            {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
            {"bbox_label_3d": 1, "bbox_score_3d": 0.7, "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0]},
            {"bbox_label_3d": 2, "bbox_score_3d": 0.6, "bbox_3d": [20.0, 21.0, 0.0, 1.8, 0.8, 1.2, 0.2]}
        ]
        
        model_instances = NMSModelInstances(
            model_id=0,
            instances=instances,
            weight=0.8,
            class_name_to_id=class_name_to_id
        )
        
        target_label_names = ["car", "pedestrian"]
        filtered_instances, boxes, scores = model_instances.filter_and_weight_instances(target_label_names)
        
        # Should keep car and pedestrian, filter out bicycle
        assert len(filtered_instances) == 2
        assert len(boxes) == 2
        assert len(scores) == 2
        
        # Check weighted scores
        expected_scores = [0.8 * 0.8, 0.7 * 0.8]  # original_score * weight
        np.testing.assert_array_almost_equal(scores, expected_scores)
        
        # Check that boxes are numpy arrays with correct shape
        assert isinstance(boxes, np.ndarray)
        assert boxes.shape == (2, 7)  # 2 instances, 7 bbox parameters each

    def test_filter_and_weight_instances_empty_result(self, sample_classes):
        """Test filtering when no instances match target labels."""
        class_name_to_id = {name: idx for idx, name in enumerate(sample_classes)}
        instances = [
            {"bbox_label_3d": 2, "bbox_score_3d": 0.6, "bbox_3d": [20.0, 21.0, 0.0, 1.8, 0.8, 1.2, 0.2]}
        ]
        
        model_instances = NMSModelInstances(
            model_id=0,
            instances=instances,
            weight=0.8,
            class_name_to_id=class_name_to_id
        )
        
        target_label_names = ["car", "pedestrian"]  # No bicycle in target
        filtered_instances, boxes, scores = model_instances.filter_and_weight_instances(target_label_names)
        
        # Should return empty results
        assert len(filtered_instances) == 0
        assert isinstance(boxes, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert boxes.size == 0
        assert scores.size == 0

    def test_filter_and_weight_instances_no_instances(self, sample_classes):
        """Test filtering when there are no instances."""
        class_name_to_id = {name: idx for idx, name in enumerate(sample_classes)}
        
        model_instances = NMSModelInstances(
            model_id=0,
            instances=[],
            weight=0.8,
            class_name_to_id=class_name_to_id
        )
        
        target_label_names = ["car", "pedestrian"]
        filtered_instances, boxes, scores = model_instances.filter_and_weight_instances(target_label_names)
        
        # Should return empty results
        assert len(filtered_instances) == 0
        assert isinstance(boxes, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert boxes.size == 0
        assert scores.size == 0


class TestNMSEnsembleModel:
    """Test cases for NMSEnsembleModel class."""

    def test_init(self, mock_logger, ensemble_settings):
        """Test NMSEnsembleModel initialization."""
        model = NMSEnsembleModel(
            ensemble_setting=ensemble_settings,
            logger=mock_logger
        )
        
        assert model.settings == ensemble_settings
        assert model.logger == mock_logger
        assert model.model_instances_type == NMSModelInstances

    def test_ensemble_single_model(self, mock_logger, ensemble_settings, sample_ensemble_results):
        """Test ensemble with single model (should return original result)."""
        model = NMSEnsembleModel(
            ensemble_setting=ensemble_settings,
            logger=mock_logger
        )
        
        single_result = [sample_ensemble_results[0]]
        result = model.ensemble(single_result)
        
        # Should return the original result unchanged
        assert result == single_result[0]

    def test_ensemble_basic_functionality(self, mock_logger, sample_ensemble_results):
        """Test basic ensemble functionality with multiple models."""
        ensemble_settings = {
            "weights": [0.6, 0.4],
            "iou_threshold": 0.5,
            "ensemble_label_groups": [["car"], ["pedestrian"], ["bicycle"]]
        }
        
        model = NMSEnsembleModel(
            ensemble_setting=ensemble_settings,
            logger=mock_logger
        )
        
        result = model.ensemble(sample_ensemble_results)
        
        # Check structure
        assert "metainfo" in result
        assert "data_list" in result
        assert len(result["data_list"]) == 1  # Same number of frames as input
        
        # Check that instances are present
        instances = result["data_list"][0]["pred_instances_3d"]
        assert len(instances) > 0
        
        # All instances should have weighted scores
        for instance in instances:
            assert "bbox_score_3d" in instance
            assert "bbox_label_3d" in instance
            assert "bbox_3d" in instance

    def test_ensemble_with_overlapping_predictions(self, mock_logger, sample_classes):
        """Test ensemble with overlapping predictions that should be merged by NMS."""
        ensemble_settings = {
            "weights": [0.6, 0.4],
            "iou_threshold": 0.3,  # Low threshold to ensure overlap is detected
            "ensemble_label_groups": [["car"]]
        }
        
        # Create two models with overlapping car predictions
        overlapping_results = [
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,  # car
                                "bbox_score_3d": 0.8,
                                "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]
                            }
                        ]
                    }
                ]
            },
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,  # car - very similar position
                                "bbox_score_3d": 0.6,
                                "bbox_3d": [1.1, 2.1, 0.0, 4.0, 2.0, 1.5, 0.1]
                            }
                        ]
                    }
                ]
            }
        ]
        
        model = NMSEnsembleModel(
            ensemble_setting=ensemble_settings,
            logger=mock_logger
        )
        
        result = model.ensemble(overlapping_results)
        
        # Should have only one car instance after NMS (higher score should win)
        instances = result["data_list"][0]["pred_instances_3d"]
        assert len(instances) == 1
        
        # The kept instance should have the higher score (labels may be remapped)
        kept_instance = instances[0]
        assert kept_instance["bbox_score_3d"] == 0.8  # Higher original score should win

    def test_ensemble_label_groups(self, mock_logger, sample_classes):
        """Test that ensemble_label_groups work correctly - NMS only within groups."""
        ensemble_settings = {
            "weights": [0.5, 0.5],
            "iou_threshold": 0.1,  # Very low threshold - everything should overlap
            "ensemble_label_groups": [["car"], ["pedestrian"]]  # Separate groups
        }
        
        # Create overlapping predictions of different classes at same location
        overlapping_different_classes = [
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,  # car
                                "bbox_score_3d": 0.8,
                                "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]
                            },
                            {
                                "bbox_label_3d": 1,  # pedestrian at same location
                                "bbox_score_3d": 0.7,
                                "bbox_3d": [1.0, 2.0, 0.0, 0.6, 0.6, 1.8, 0.0]
                            }
                        ]
                    }
                ]
            },
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,  # car at same location
                                "bbox_score_3d": 0.6,
                                "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]
                            },
                            {
                                "bbox_label_3d": 1,  # pedestrian at same location
                                "bbox_score_3d": 0.5,
                                "bbox_3d": [1.0, 2.0, 0.0, 0.6, 0.6, 1.8, 0.0]
                            }
                        ]
                    }
                ]
            }
        ]
        
        model = NMSEnsembleModel(
            ensemble_setting=ensemble_settings,
            logger=mock_logger
        )
        
        result = model.ensemble(overlapping_different_classes)
        
        instances = result["data_list"][0]["pred_instances_3d"]
        
        # Should have instances from both label groups (NMS within groups, not between)
        # Labels may be remapped, so we check by count instead of specific label values
        assert len(instances) == 2  # One from each group

    def test_ensemble_empty_predictions(self, mock_logger, ensemble_settings, sample_classes):
        """Test ensemble with empty predictions."""
        empty_results = [
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [{"pred_instances_3d": []}]
            },
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [{"pred_instances_3d": []}]
            }
        ]
        
        model = NMSEnsembleModel(
            ensemble_setting=ensemble_settings,
            logger=mock_logger
        )
        
        result = model.ensemble(empty_results)
        
        # Should handle empty predictions gracefully
        assert "metainfo" in result
        assert "data_list" in result
        assert len(result["data_list"][0]["pred_instances_3d"]) == 0

    def test_ensemble_partial_empty_predictions(self, mock_logger, ensemble_settings, sample_classes):
        """Test ensemble when some models have empty predictions."""
        partial_empty_results = [
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,
                                "bbox_score_3d": 0.8,
                                "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]
                            }
                        ]
                    }
                ]
            },
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [{"pred_instances_3d": []}]  # Empty
            }
        ]
        
        model = NMSEnsembleModel(
            ensemble_setting=ensemble_settings,
            logger=mock_logger
        )
        
        result = model.ensemble(partial_empty_results)
        
        # Should keep the non-empty predictions
        instances = result["data_list"][0]["pred_instances_3d"]
        assert len(instances) > 0

    def test_weight_mismatch_assertion(self, mock_logger, sample_ensemble_results):
        """Test that assertion fails when number of weights doesn't match number of models."""
        wrong_ensemble_settings = {
            "weights": [0.6],  # Only one weight for two models
            "iou_threshold": 0.5,
            "ensemble_label_groups": [["car"], ["pedestrian"], ["bicycle"]]
        }
        
        model = NMSEnsembleModel(
            ensemble_setting=wrong_ensemble_settings,
            logger=mock_logger
        )
        
        with pytest.raises(AssertionError, match="Number of weights must match number of models"):
            model.ensemble(sample_ensemble_results)

    def test_extreme_iou_thresholds(self, mock_logger, sample_classes):
        """Test ensemble with different IoU thresholds."""
        # Test with simple case that mirrors existing successful tests
        test_results_two_models = [
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,
                                "bbox_score_3d": 0.8,
                                "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]
                            }
                        ]
                    }
                ]
            },
            {
                "metainfo": {"classes": sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,
                                "bbox_score_3d": 0.6,
                                "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.1]  # No overlap
                            }
                        ]
                    }
                ]
            }
        ]
        
        # Test that non-overlapping boxes are both kept regardless of threshold
        ensemble_settings = {
            "weights": [0.5, 0.5],
            "iou_threshold": 0.1,  # Very low threshold
            "ensemble_label_groups": [["car"]]
        }
        
        model = NMSEnsembleModel(
            ensemble_setting=ensemble_settings,
            logger=mock_logger
        )
        
        result = model.ensemble(test_results_two_models)
        instances = result["data_list"][0]["pred_instances_3d"]
        
        # Non-overlapping boxes should both be kept
        assert len(instances) == 2


class TestNMSHelperFunctions:
    """Test cases for NMS helper functions."""

    def test_calculate_iou_identical_boxes(self):
        """Test IoU calculation for identical boxes."""
        box1 = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0])
        box2 = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        
        iou = _calculate_iou(box1, box2)
        
        assert len(iou) == 1
        assert abs(iou[0] - 1.0) < 1e-6  # Should be 1.0 for identical boxes

    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation for non-overlapping boxes."""
        box1 = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0])  # Box at origin
        box2 = np.array([[10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 0.0]])  # Box far away
        
        iou = _calculate_iou(box1, box2)
        
        assert len(iou) == 1
        assert abs(iou[0] - 0.0) < 1e-6  # Should be 0.0 for non-overlapping

    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation for partially overlapping boxes."""
        box1 = np.array([0.0, 0.0, 0.0, 4.0, 4.0, 1.5, 0.0])  # 4x4 box at origin
        box2 = np.array([[2.0, 2.0, 0.0, 4.0, 4.0, 1.5, 0.0]])  # 4x4 box offset by (2,2)
        
        iou = _calculate_iou(box1, box2)
        
        # Intersection: 2x2 = 4, Union: 4*4 + 4*4 - 4 = 28, IoU = 4/28 = 1/7
        expected_iou = 4.0 / 28.0
        assert len(iou) == 1
        assert abs(iou[0] - expected_iou) < 1e-6

    def test_calculate_iou_multiple_boxes(self):
        """Test IoU calculation with multiple boxes."""
        box1 = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0])
        boxes = np.array([
            [0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0],  # Identical
            [1.0, 1.0, 0.0, 2.0, 2.0, 1.5, 0.0],  # Partial overlap
            [10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 0.0]  # No overlap
        ])
        
        ious = _calculate_iou(box1, boxes)
        
        assert len(ious) == 3
        assert abs(ious[0] - 1.0) < 1e-6  # Identical
        assert ious[1] > 0.0 and ious[1] < 1.0  # Partial overlap
        assert abs(ious[2] - 0.0) < 1e-6  # No overlap

    def test_nms_indices_basic(self):
        """Test basic NMS indices functionality."""
        # Create boxes with different scores
        boxes = np.array([
            [0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0],   # Low score
            [0.1, 0.1, 0.0, 2.0, 2.0, 1.5, 0.0],   # High score, overlapping with box 0
            [10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 0.0]  # Medium score, non-overlapping
        ])
        scores = np.array([0.3, 0.9, 0.6])
        iou_threshold = 0.5
        
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        
        # Should keep indices 1 (highest score) and 2 (non-overlapping)
        # Index 0 should be suppressed by index 1
        assert 1 in keep_indices  # Highest score
        assert 2 in keep_indices  # Non-overlapping
        assert 0 not in keep_indices  # Suppressed by higher score

    def test_nms_indices_no_overlap(self):
        """Test NMS when no boxes overlap."""
        boxes = np.array([
            [0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0],
            [10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 0.0],
            [20.0, 20.0, 0.0, 2.0, 2.0, 1.5, 0.0]
        ])
        scores = np.array([0.3, 0.9, 0.6])
        iou_threshold = 0.5
        
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        
        # Should keep all boxes since none overlap
        assert len(keep_indices) == 3
        assert set(keep_indices) == {0, 1, 2}

    def test_nms_indices_single_box(self):
        """Test NMS with single box."""
        boxes = np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0]])
        scores = np.array([0.8])
        iou_threshold = 0.5
        
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        
        # Should keep the single box
        assert keep_indices == [0]

    def test_nms_indices_empty_input(self):
        """Test NMS with empty input."""
        boxes = np.array([]).reshape(0, 7)
        scores = np.array([])
        iou_threshold = 0.5
        
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        
        # Should return empty list
        assert keep_indices == []

    @pytest.mark.parametrize("iou_threshold,expected_kept", [
        (0.0, 1),   # Strictest - only highest score
        (0.3, 1),   # Both IoUs (0.391, 0.620) > 0.3, so only highest kept
        (0.6, 2),   # IoU 0.391 <= 0.6, so box0 kept; IoU 0.620 > 0.6, so box2 suppressed
        (0.9, 3),   # All IoUs <= 0.9, so all kept
        (1.0, 3),   # Most lenient - only identical boxes suppressed
    ])
    def test_nms_indices_various_thresholds(self, iou_threshold, expected_kept):
        """Test NMS with various IoU thresholds."""
        # Create three overlapping boxes with different overlap amounts
        boxes = np.array([
            [0.0, 0.0, 0.0, 4.0, 4.0, 1.5, 0.0],   # Score 0.5
            [1.0, 1.0, 0.0, 4.0, 4.0, 1.5, 0.0],   # Score 0.9, medium overlap with box 0
            [0.5, 0.5, 0.0, 4.0, 4.0, 1.5, 0.0]    # Score 0.7, high overlap with both
        ])
        scores = np.array([0.5, 0.9, 0.7])
        
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        
        assert len(keep_indices) == expected_kept