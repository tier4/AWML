import logging
from collections import defaultdict
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from tools.auto_labeling_3d.filter_objects.filter.threshold_filter import ThresholdFilter


class TestThresholdFilter:
    """Test cases for ThresholdFilter class."""

    def test_init(self, mock_logger, sample_confidence_thresholds, sample_use_label):
        """Test ThresholdFilter initialization with valid parameters.
        
        Verifies: Proper initialization of settings and logger assignment
        Expected: All configuration parameters are correctly stored and accessible
        """
        filter_obj = ThresholdFilter(
            confidence_thresholds=sample_confidence_thresholds, use_label=sample_use_label, logger=mock_logger
        )

        assert filter_obj.settings["confidence_thresholds"] == sample_confidence_thresholds
        assert set(filter_obj.settings["use_label"]) == set(sample_use_label)
        assert filter_obj.logger == mock_logger

    def test_init_with_duplicate_use_label(self, mock_logger, sample_confidence_thresholds):
        """Test ThresholdFilter initialization with duplicate labels in use_label list.
        
        Verifies: Proper handling of duplicate labels in use_label parameter
        Expected: Duplicates are automatically removed and unique labels are preserved
        """
        use_label_with_duplicates = ["car", "pedestrian", "car", "bicycle"]
        filter_obj = ThresholdFilter(
            confidence_thresholds=sample_confidence_thresholds, use_label=use_label_with_duplicates, logger=mock_logger
        )

        # Should remove duplicates
        assert len(filter_obj.settings["use_label"]) == 3
        assert set(filter_obj.settings["use_label"]) == {"car", "pedestrian", "bicycle"}

    def test_basic_filtering(
        self, mock_logger, sample_confidence_thresholds, sample_use_label, sample_predicted_result_info
    ):
        """Test basic confidence-based filtering functionality.
        
        Verifies: Objects are filtered based on confidence thresholds for each class
        Expected: Only objects meeting confidence requirements are kept in results
        """
        filter_obj = ThresholdFilter(
            confidence_thresholds=sample_confidence_thresholds, use_label=sample_use_label, logger=mock_logger
        )

        result = filter_obj.filter(sample_predicted_result_info, "test_model")

        # Check structure is preserved
        assert "metainfo" in result
        assert "data_list" in result
        assert result["metainfo"] == sample_predicted_result_info["metainfo"]
        assert len(result["data_list"]) == 2  # Same number of frames

        # Frame 1: Should keep car (0.8 >= 0.5) and pedestrian (0.4 >= 0.3)
        # Should filter out car (0.3 < 0.5) and bicycle (0.2 < 0.4)
        frame1_instances = result["data_list"][0]["pred_instances_3d"]
        assert len(frame1_instances) == 2

        # Check kept instances
        kept_scores = [inst["bbox_score_3d"] for inst in frame1_instances]
        assert 0.8 in kept_scores  # High confidence car
        assert 0.4 in kept_scores  # Above threshold pedestrian
        assert 0.3 not in kept_scores  # Low confidence car filtered
        assert 0.2 not in kept_scores  # Low confidence bicycle filtered

        # Frame 2: Should keep car (0.9 >= 0.5)
        # Should filter out pedestrian (0.1 < 0.3)
        frame2_instances = result["data_list"][1]["pred_instances_3d"]
        assert len(frame2_instances) == 1
        assert frame2_instances[0]["bbox_score_3d"] == 0.9

    def test_use_label_filtering(self, mock_logger, sample_predicted_result_info):
        """Test filtering by use_label parameter to exclude certain classes.
        
        Verifies: Only objects from classes specified in use_label are retained
        Expected: Objects from excluded classes are removed regardless of confidence
        """
        # Only allow car and pedestrian, exclude bicycle
        confidence_thresholds = {"car": 0.1, "pedestrian": 0.1}  # Very low thresholds
        use_label = ["car", "pedestrian"]

        filter_obj = ThresholdFilter(
            confidence_thresholds=confidence_thresholds, use_label=use_label, logger=mock_logger
        )

        result = filter_obj.filter(sample_predicted_result_info, "test_model")

        # Check that bicycle instances are filtered out regardless of confidence
        all_instances = []
        for frame in result["data_list"]:
            all_instances.extend(frame["pred_instances_3d"])

        # Should not contain any bicycle instances (label_id=2)
        bicycle_instances = [inst for inst in all_instances if inst["bbox_label_3d"] == 2]
        assert len(bicycle_instances) == 0

        # Should contain car and pedestrian instances that pass confidence threshold
        car_instances = [inst for inst in all_instances if inst["bbox_label_3d"] == 0]
        pedestrian_instances = [inst for inst in all_instances if inst["bbox_label_3d"] == 1]
        assert len(car_instances) > 0
        assert len(pedestrian_instances) > 0

    def test_boundary_values(self, mock_logger, sample_classes):
        """Test filtering behavior with boundary confidence values (0.0 and 1.0).
        
        Verifies: Correct handling of edge cases with minimum and maximum confidence values
        Expected: Objects with confidence exactly at threshold are included; boundary behaviors work correctly
        """
        # Create data with exact threshold values
        confidence_thresholds = {"car": 0.5}
        use_label = ["car"]

        test_data = {
            "metainfo": {"classes": sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {
                            "bbox_label_3d": 0,  # car
                            "bbox_score_3d": 0.5,  # Exactly at threshold
                            "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1],
                        },
                        {
                            "bbox_label_3d": 0,  # car
                            "bbox_score_3d": 0.4999,  # Just below threshold
                            "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.2],
                        },
                    ]
                }
            ],
        }

        filter_obj = ThresholdFilter(
            confidence_thresholds=confidence_thresholds, use_label=use_label, logger=mock_logger
        )

        result = filter_obj.filter(test_data, "test_model")

        # Should keep exactly at threshold (>=), filter just below
        instances = result["data_list"][0]["pred_instances_3d"]
        assert len(instances) == 1
        assert instances[0]["bbox_score_3d"] == 0.5

    def test_extreme_thresholds(self, mock_logger, sample_classes):
        """Test filtering behavior with extreme threshold values (0.0 and 1.0).
        
        Verifies: Proper handling of extreme thresholds that should keep all or no objects
        Expected: Threshold 0.0 keeps all objects; threshold 1.0 keeps only perfect confidence objects
        """
        test_data = {
            "metainfo": {"classes": sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {
                            "bbox_label_3d": 0,  # car
                            "bbox_score_3d": 0.0,
                            "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1],
                        },
                        {
                            "bbox_label_3d": 0,  # car
                            "bbox_score_3d": 0.5,
                            "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.2],
                        },
                        {
                            "bbox_label_3d": 0,  # car
                            "bbox_score_3d": 1.0,
                            "bbox_3d": [10.0, 11.0, 0.0, 4.0, 2.0, 1.5, 0.3],
                        },
                    ]
                }
            ],
        }

        # Test with threshold 0.0 - should keep all
        filter_obj_zero = ThresholdFilter(confidence_thresholds={"car": 0.0}, use_label=["car"], logger=mock_logger)
        result_zero = filter_obj_zero.filter(test_data, "test_model")
        assert len(result_zero["data_list"][0]["pred_instances_3d"]) == 3

        # Test with threshold 1.0 - should keep only perfect score
        filter_obj_one = ThresholdFilter(confidence_thresholds={"car": 1.0}, use_label=["car"], logger=mock_logger)
        result_one = filter_obj_one.filter(test_data, "test_model")
        instances = result_one["data_list"][0]["pred_instances_3d"]
        assert len(instances) == 1
        assert instances[0]["bbox_score_3d"] == 1.0

    def test_empty_data(
        self, mock_logger, sample_confidence_thresholds, sample_use_label, empty_predicted_result_info
    ):
        """Test proper handling of empty prediction data structures.
        
        Verifies: Filter correctly processes datasets with no predicted instances
        Expected: Empty data structure is preserved without errors; no filtering occurs
        """
        filter_obj = ThresholdFilter(
            confidence_thresholds=sample_confidence_thresholds, use_label=sample_use_label, logger=mock_logger
        )

        result = filter_obj.filter(empty_predicted_result_info, "test_model")

        # Should handle empty data gracefully
        assert "metainfo" in result
        assert "data_list" in result
        assert len(result["data_list"]) == 1
        assert len(result["data_list"][0]["pred_instances_3d"]) == 0

    def test_multiple_frames(self, mock_logger, sample_classes):
        """Test filtering consistency across multiple data frames.
        
        Verifies: Filter applies same logic consistently across all frames in dataset
        Expected: Each frame is processed independently with consistent filtering rules
        """
        confidence_thresholds = {"car": 0.5}
        use_label = ["car"]

        # Create data with 3 frames
        test_data = {
            "metainfo": {"classes": sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.3, "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.2]},
                    ]
                },
                {
                    "pred_instances_3d": [
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.6, "bbox_3d": [10.0, 11.0, 0.0, 4.0, 2.0, 1.5, 0.3]}
                    ]
                },
                {
                    "pred_instances_3d": [
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.4, "bbox_3d": [15.0, 16.0, 0.0, 4.0, 2.0, 1.5, 0.4]}
                    ]
                },
            ],
        }

        filter_obj = ThresholdFilter(
            confidence_thresholds=confidence_thresholds, use_label=use_label, logger=mock_logger
        )

        result = filter_obj.filter(test_data, "test_model")

        # Check each frame individually
        assert len(result["data_list"]) == 3

        # Frame 1: Keep 0.8, filter 0.3
        assert len(result["data_list"][0]["pred_instances_3d"]) == 1
        assert result["data_list"][0]["pred_instances_3d"][0]["bbox_score_3d"] == 0.8

        # Frame 2: Keep 0.6
        assert len(result["data_list"][1]["pred_instances_3d"]) == 1
        assert result["data_list"][1]["pred_instances_3d"][0]["bbox_score_3d"] == 0.6

        # Frame 3: Filter 0.4
        assert len(result["data_list"][2]["pred_instances_3d"]) == 0

    def test_unknown_class_handling(self, mock_logger, sample_classes):
        """Test handling of object classes not defined in confidence_thresholds.
        
        Verifies: Proper handling when objects have classes without threshold definitions
        Expected: Unknown classes are filtered out or handled according to default behavior
        """
        # Define thresholds only for car, but use_label includes pedestrian
        confidence_thresholds = {"car": 0.5}  # Missing pedestrian
        use_label = ["car", "pedestrian"]

        test_data = {
            "metainfo": {"classes": sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
                        {"bbox_label_3d": 1, "bbox_score_3d": 0.9, "bbox_3d": [5.0, 6.0, 0.0, 0.6, 0.6, 1.8, 0.0]},
                    ]
                }
            ],
        }

        filter_obj = ThresholdFilter(
            confidence_thresholds=confidence_thresholds, use_label=use_label, logger=mock_logger
        )

        # Should raise KeyError when pedestrian threshold is not defined
        with pytest.raises(KeyError):
            filter_obj.filter(test_data, "test_model")

    def test_filter_statistics_logging(
        self, mock_logger, sample_confidence_thresholds, sample_use_label, sample_predicted_result_info
    ):
        """Test that filtering statistics are properly logged during operation.
        
        Verifies: Logger captures filtering statistics including counts and model information
        Expected: Info messages are logged with original/filtered object counts per frame
        """
        # Use a mock logger to capture log calls
        mock_logger = Mock(spec=logging.Logger)

        filter_obj = ThresholdFilter(
            confidence_thresholds=sample_confidence_thresholds, use_label=sample_use_label, logger=mock_logger
        )

        filter_obj.filter(sample_predicted_result_info, "test_model")

        # Verify that logging methods were called
        assert mock_logger.info.called

        # Check that statistics information appears in log calls
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        log_text = " ".join(log_calls)

        # Should contain filtering statistics
        assert "Filtering statistics" in log_text
        assert "test_model" in log_text
        assert "Total instances" in log_text
        assert "Filtered instances" in log_text

    @pytest.mark.parametrize(
        "confidence_threshold,expected_kept",
        [
            (0.0, 4),  # Keep all instances in frame 1
            (0.2, 4),  # Keep all (bicycle with 0.2 score is >= 0.2 threshold)
            (0.3, 3),  # Filter car (0.3), keep others
            (0.4, 2),  # Filter car (0.3) and pedestrian (0.4 not >), keep car (0.8)
            (0.8, 1),  # Keep only car (0.8)
            (1.0, 0),  # Filter all
        ],
    )
    def test_various_thresholds_parametrized(self, mock_logger, sample_classes, confidence_threshold, expected_kept):
        """Test filtering behavior across various threshold values using parametrization.
        
        Verifies: Different threshold values produce expected object count results
        Expected: Number of kept objects matches expected count for each threshold level
        """
        # Use same threshold for all classes to simplify testing
        confidence_thresholds = {cls: confidence_threshold for cls in sample_classes}
        use_label = sample_classes

        test_data = {
            "metainfo": {"classes": sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.3, "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.2]},
                        {"bbox_label_3d": 1, "bbox_score_3d": 0.4, "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0]},
                        {"bbox_label_3d": 2, "bbox_score_3d": 0.2, "bbox_3d": [15.0, 16.0, 0.0, 1.8, 0.8, 1.2, 0.3]},
                    ]
                }
            ],
        }

        filter_obj = ThresholdFilter(
            confidence_thresholds=confidence_thresholds, use_label=use_label, logger=mock_logger
        )

        result = filter_obj.filter(test_data, "test_model")

        instances = result["data_list"][0]["pred_instances_3d"]
        assert len(instances) == expected_kept

    def test_should_filter_instance_private_method(self, mock_logger, sample_confidence_thresholds, sample_use_label):
        """Test the _should_filter_instance private method behavior through public interface.
        
        Verifies: Private filtering logic correctly evaluates individual objects
        Expected: Objects are filtered based on class inclusion and confidence thresholds
        """
        filter_obj = ThresholdFilter(
            confidence_thresholds=sample_confidence_thresholds, use_label=sample_use_label, logger=mock_logger
        )

        # Test cases that cover different filtering scenarios
        test_cases = [
            # (instance, category, should_be_filtered)
            ({"bbox_score_3d": 0.8}, "car", False),  # Above threshold, in use_label
            ({"bbox_score_3d": 0.3}, "car", True),  # Below threshold
            ({"bbox_score_3d": 0.9}, "truck", True),  # Not in use_label
            ({"bbox_score_3d": 0.5}, "car", False),  # At threshold (boundary)
        ]

        for instance, category, expected_filtered in test_cases:
            result = filter_obj._should_filter_instance(instance, category)
            assert result == expected_filtered, f"Failed for {instance} with category {category}"
