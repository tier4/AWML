import logging
import unittest
from collections import defaultdict
from typing import Any, Dict, List
from unittest.mock import Mock

from tools.auto_labeling_3d.filter_objects.filter.threshold_filter import ThresholdFilter


class TestThresholdFilter(unittest.TestCase):
    """Test cases for ThresholdFilter class."""

    def setUp(self):
        """
        Initialize shared fixtures before each ThresholdFilter test.

        This setup assumes confidence-threshold filtering across three classes:
        "car", "pedestrian", and "bicycle". It defines:
        - A mock logger used to capture filtering statistics without side effects.
        - Per-class threshold and allow-list configurations for representative vehicles.
        - Sample multi-frame prediction results exercising common filtering paths.
        - An empty prediction template for validating edge-case handling.
        """
        self.mock_logger = Mock(spec=logging.Logger)
        self.sample_confidence_thresholds = {"car": 0.5, "pedestrian": 0.3, "bicycle": 0.4}
        self.sample_use_label = ["car", "pedestrian", "bicycle"]
        self.sample_classes = ["car", "pedestrian", "bicycle"]
        self.sample_predicted_result_info = {
            "metainfo": {"classes": self.sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.3, "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.2]},
                        {"bbox_label_3d": 1, "bbox_score_3d": 0.4, "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0]},
                        {"bbox_label_3d": 2, "bbox_score_3d": 0.2, "bbox_3d": [15.0, 16.0, 0.0, 1.8, 0.8, 1.2, 0.3]},
                    ]
                },
                {
                    "pred_instances_3d": [
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.9, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
                        {"bbox_label_3d": 1, "bbox_score_3d": 0.1, "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0]},
                    ]
                },
            ],
        }
        self.empty_predicted_result_info = {
            "metainfo": {"classes": self.sample_classes},
            "data_list": [{"pred_instances_3d": []}],
        }

    def test_init(self):
        """
        Test case for ThresholdFilter initialization.

        This test verifies that constructing `ThresholdFilter` with explicit
        thresholds, labels, and logger preserves the provided configuration.
        It checks:
        - That the internal settings echo the supplied threshold mapping.
        - That duplicate labels are not introduced during initialization.
        - That the logger reference is stored unchanged.

        Inputs for `ThresholdFilter.__init__`:
            - Confidence thresholds for car, pedestrian, and bicycle classes.
            - Label list containing the three target classes.
            - Mock logger instance shared across tests.
        Expected Outputs from `ThresholdFilter.__init__`:
            - Settings dictionary reflecting the thresholds and labels verbatim.
            - `logger` attribute matching the injected mock.
        """
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds,
            use_label=self.sample_use_label,
            logger=self.mock_logger,
        )
        self.assertEqual(filter_obj.settings["confidence_thresholds"], self.sample_confidence_thresholds)
        self.assertEqual(set(filter_obj.settings["use_label"]), set(self.sample_use_label))
        self.assertEqual(filter_obj.logger, self.mock_logger)

    def test_init_with_duplicate_use_label(self):
        """
        Test case for initialization with duplicate labels in the configuration.

        This test verifies that `ThresholdFilter` collapses duplicate entries in
        the label list during construction.
        It checks:
        - That only unique labels are retained in the stored settings.
        - That the resulting set matches the expected categories.

        Inputs for `ThresholdFilter.__init__`:
            - Confidence thresholds covering three classes.
            - Label list containing duplicate occurrences of "car".
            - Mock logger instance.
        Expected Outputs from `ThresholdFilter.__init__`:
            - Settings with a deduplicated `use_label` containing three entries.
        """
        use_label_with_duplicates = ["car", "pedestrian", "car", "bicycle"]
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds,
            use_label=use_label_with_duplicates,
            logger=self.mock_logger,
        )
        self.assertEqual(len(filter_obj.settings["use_label"]), 3)
        self.assertEqual(set(filter_obj.settings["use_label"]), {"car", "pedestrian", "bicycle"})

    def test_filter_per_class_thresholds(self):
        """
        Test case for filtering detections against per-class score thresholds.

        This test verifies that `filter` removes low-confidence detections while
        preserving those above each class threshold.
        It checks:
        - That metadata keys are retained in the filtered result.
        - That high-confidence car and pedestrian detections remain.
        - That sub-threshold car and bicycle detections are removed.
        - That subsequent frames respect the same thresholds.

        Inputs for `ThresholdFilter.filter`:
            - Prediction results containing two frames with mixed scores.
            - Model identifier string "test_model" (used for logging context).
        Expected Outputs from `ThresholdFilter.filter`:
            - Filtered result mirroring original metadata.
            - First frame containing only detections with scores 0.8 and 0.4.
            - Second frame containing only the 0.9 score detection.
        """
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds,
            use_label=self.sample_use_label,
            logger=self.mock_logger,
        )
        result = filter_obj.filter(self.sample_predicted_result_info, "test_model")
        self.assertIn("metainfo", result)
        self.assertIn("data_list", result)
        self.assertEqual(result["metainfo"], self.sample_predicted_result_info["metainfo"])
        self.assertEqual(len(result["data_list"]), 2)
        frame1_instances = result["data_list"][0]["pred_instances_3d"]
        self.assertEqual(len(frame1_instances), 2)
        kept_scores = [inst["bbox_score_3d"] for inst in frame1_instances]
        self.assertIn(0.8, kept_scores)
        self.assertIn(0.4, kept_scores)
        self.assertNotIn(0.3, kept_scores)
        self.assertNotIn(0.2, kept_scores)
        frame2_instances = result["data_list"][1]["pred_instances_3d"]
        self.assertEqual(len(frame2_instances), 1)
        self.assertEqual(frame2_instances[0]["bbox_score_3d"], 0.9)

    def test_use_label_filtering(self):
        """
        Test case for restricting filtering to a subset of class labels.

        This test verifies that specifying `use_label` limits output detections
        to the approved categories.
        It checks:
        - That bicycle detections are removed when bicycles are absent from
            `use_label`.
        - That car and pedestrian detections remain available for downstream
            processing.

        Inputs for `ThresholdFilter.filter`:
                - Ensemble results containing car, pedestrian, and bicycle detections.
                - Configuration listing only car and pedestrian labels.
        Expected Outputs from `ThresholdFilter.filter`:
                - Filtered detections excluding bicycles while keeping other classes.
        """
        confidence_thresholds = {"car": 0.1, "pedestrian": 0.1}
        use_label = ["car", "pedestrian"]
        filter_obj = ThresholdFilter(
            confidence_thresholds=confidence_thresholds, use_label=use_label, logger=self.mock_logger
        )
        result = filter_obj.filter(self.sample_predicted_result_info, "test_model")
        all_instances = []
        for frame in result["data_list"]:
            all_instances.extend(frame["pred_instances_3d"])
        bicycle_instances = [inst for inst in all_instances if inst["bbox_label_3d"] == 2]
        self.assertEqual(len(bicycle_instances), 0)
        car_instances = [inst for inst in all_instances if inst["bbox_label_3d"] == 0]
        pedestrian_instances = [inst for inst in all_instances if inst["bbox_label_3d"] == 1]
        self.assertGreater(len(car_instances), 0)
        self.assertGreater(len(pedestrian_instances), 0)

    def test_boundary_values(self):
        """
        Test case for handling scores exactly on the threshold boundary.

        This test verifies that `filter` treats scores equal to the threshold as
        valid while discarding scores just below the boundary.
        It checks:
        - That a car detection scoring 0.5 (equal to the threshold) is kept.
        - That a detection with score 0.4999 is filtered out.

        Inputs for `ThresholdFilter.filter`:
            - Single-frame detections with scores straddling the 0.5 boundary.
            - Configuration specifying a car threshold of 0.5.
        Expected Outputs from `ThresholdFilter.filter`:
            - Filtered result retaining only the detection scoring 0.5.
        """
        confidence_thresholds = {"car": 0.5}
        use_label = ["car"]
        test_data = {
            "metainfo": {"classes": self.sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.5, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.4999, "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.2]},
                    ]
                }
            ],
        }
        filter_obj = ThresholdFilter(
            confidence_thresholds=confidence_thresholds, use_label=use_label, logger=self.mock_logger
        )
        result = filter_obj.filter(test_data, "test_model")
        instances = result["data_list"][0]["pred_instances_3d"]
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["bbox_score_3d"], 0.5)

    def test_extreme_thresholds(self):
        """
        Test case for filtering with the lowest and highest possible thresholds.

        This test verifies that `filter` respects edge threshold values of 0.0
        and 1.0 for a single class.
        It checks:
        - That a zero threshold keeps every detection regardless of score.
        - That a threshold of one retains only perfect-score detections.

        Inputs for `ThresholdFilter.filter`:
            - Single-frame detections with scores 0.0, 0.5, and 1.0.
            - Two configurations: one with `car` threshold 0.0, another with 1.0.
        Expected Outputs from `ThresholdFilter.filter`:
            - Three detections kept for the zero threshold scenario.
            - Only the 1.0 detection retained for the unit threshold scenario.
        """
        test_data = {
            "metainfo": {"classes": self.sample_classes},
            "data_list": [
                {
                    "pred_instances_3d": [
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.0, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
                        {"bbox_label_3d": 0, "bbox_score_3d": 0.5, "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.2]},
                        {"bbox_label_3d": 0, "bbox_score_3d": 1.0, "bbox_3d": [10.0, 11.0, 0.0, 4.0, 2.0, 1.5, 0.3]},
                    ]
                }
            ],
        }
        filter_obj_zero = ThresholdFilter(
            confidence_thresholds={"car": 0.0}, use_label=["car"], logger=self.mock_logger
        )
        result_zero = filter_obj_zero.filter(test_data, "test_model")
        self.assertEqual(len(result_zero["data_list"][0]["pred_instances_3d"]), 3)
        filter_obj_one = ThresholdFilter(
            confidence_thresholds={"car": 1.0}, use_label=["car"], logger=self.mock_logger
        )
        result_one = filter_obj_one.filter(test_data, "test_model")
        instances = result_one["data_list"][0]["pred_instances_3d"]
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["bbox_score_3d"], 1.0)

    def test_empty_data(self):
        """
        Test case for filtering when no detections are present.

        This test verifies that `filter` returns an empty prediction list without
        raising errors when given empty inputs.
        It checks:
        - That metadata is preserved even when no detections remain.
        - That the resulting frame contains zero detections.

        Inputs for `ThresholdFilter.filter`:
            - Prediction result with an empty `pred_instances_3d` list.
            - Standard configuration with thresholds and labels for three classes.
        Expected Outputs from `ThresholdFilter.filter`:
            - Response containing metadata and a single frame with no detections.
        """
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds,
            use_label=self.sample_use_label,
            logger=self.mock_logger,
        )
        result = filter_obj.filter(self.empty_predicted_result_info, "test_model")
        self.assertIn("metainfo", result)
        self.assertIn("data_list", result)
        self.assertEqual(len(result["data_list"]), 1)
        self.assertEqual(len(result["data_list"][0]["pred_instances_3d"]), 0)

    def test_multiple_frames(self):
        """
        Test case for filtering across multiple frames with varying scores.

        This test verifies that `filter` applies the same threshold to each frame
        independently.
        It checks:
        - That the first frame keeps only the high-confidence detection.
        - That subsequent frames reflect per-frame filtering results.

        Inputs for `ThresholdFilter.filter`:
            - Three frames of car detections with different scores.
            - Configuration specifying a car threshold of 0.5.
        Expected Outputs from `ThresholdFilter.filter`:
            - First frame containing the 0.8 detection.
            - Second frame retaining the 0.6 detection.
            - Third frame becoming empty because 0.4 falls below the threshold.
        """
        confidence_thresholds = {"car": 0.5}
        use_label = ["car"]
        test_data = {
            "metainfo": {"classes": self.sample_classes},
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
            confidence_thresholds=confidence_thresholds, use_label=use_label, logger=self.mock_logger
        )
        result = filter_obj.filter(test_data, "test_model")
        self.assertEqual(len(result["data_list"]), 3)
        self.assertEqual(len(result["data_list"][0]["pred_instances_3d"]), 1)
        self.assertEqual(result["data_list"][0]["pred_instances_3d"][0]["bbox_score_3d"], 0.8)
        self.assertEqual(len(result["data_list"][1]["pred_instances_3d"]), 1)
        self.assertEqual(result["data_list"][1]["pred_instances_3d"][0]["bbox_score_3d"], 0.6)
        self.assertEqual(len(result["data_list"][2]["pred_instances_3d"]), 0)

    def test_unknown_class_handling(self):
        """
        Test case for encountering detections whose labels lack configured thresholds.

        This test verifies that `filter` raises a `KeyError` when a detection
        references a label missing from the threshold mapping.
        It checks:
        - That calling `filter` with an unknown class triggers a `KeyError`.

        Inputs for `ThresholdFilter.filter`:
            - Single frame containing car and pedestrian detections.
            - Configuration that only defines a threshold for car.
        Expected Outputs from `ThresholdFilter.filter`:
            - Raised `KeyError` while processing the pedestrian detection.
        """
        confidence_thresholds = {"car": 0.5}
        use_label = ["car", "pedestrian"]
        test_data = {
            "metainfo": {"classes": self.sample_classes},
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
            confidence_thresholds=confidence_thresholds, use_label=use_label, logger=self.mock_logger
        )
        with self.assertRaises(KeyError):
            filter_obj.filter(test_data, "test_model")

    def test_filter_statistics_logging(self):
        """
        Test case for verifying logging of filtering statistics.

        This test verifies that `filter` emits informational logs summarizing
        the filtering process.
        It checks:
        - That the mock logger's `info` method is invoked.
        - That the emitted message contains model name and statistics keywords.

        Inputs for `ThresholdFilter.filter`:
            - Sample prediction results spanning multiple frames.
            - Configuration with class thresholds and label usage.
        Expected Outputs from `ThresholdFilter.filter`:
            - Log entry including "Filtering statistics", the model name, and
              total/filtered counts.
        """
        mock_logger = Mock(spec=logging.Logger)
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds,
            use_label=self.sample_use_label,
            logger=mock_logger,
        )
        filter_obj.filter(self.sample_predicted_result_info, "test_model")
        self.assertTrue(mock_logger.info.called)
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        log_text = " ".join(log_calls)
        self.assertIn("Filtering statistics", log_text)
        self.assertIn("test_model", log_text)
        self.assertIn("Total instances", log_text)
        self.assertIn("Filtered instances", log_text)

    def test_various_thresholds(self):
        """
        Test case for sweeping thresholds across multiple expected outcomes.

        This test verifies that `filter` yields the correct number of detections
        as thresholds vary.
        It checks:
        - That the count of retained detections matches the expected value for
            each threshold in the table.

        Inputs for `ThresholdFilter.filter`:
                - Single frame containing detections for three classes with diverse
                    scores.
                - Threshold configurations ranging from 0.0 to 1.0.
        Expected Outputs from `ThresholdFilter.filter`:
                - Per-threshold filtered results with detection counts defined in
                    `test_cases`.
        """
        test_cases = [
            (0.0, 4),
            (0.2, 4),
            (0.3, 3),
            (0.4, 2),
            (0.8, 1),
            (1.0, 0),
        ]
        for confidence_threshold, expected_kept in test_cases:
            confidence_thresholds = {cls: confidence_threshold for cls in self.sample_classes}
            use_label = self.sample_classes
            test_data = {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
                            {"bbox_label_3d": 0, "bbox_score_3d": 0.3, "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.2]},
                            {
                                "bbox_label_3d": 1,
                                "bbox_score_3d": 0.4,
                                "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0],
                            },
                            {
                                "bbox_label_3d": 2,
                                "bbox_score_3d": 0.2,
                                "bbox_3d": [15.0, 16.0, 0.0, 1.8, 0.8, 1.2, 0.3],
                            },
                        ]
                    }
                ],
            }
            filter_obj = ThresholdFilter(
                confidence_thresholds=confidence_thresholds, use_label=use_label, logger=self.mock_logger
            )
            result = filter_obj.filter(test_data, "test_model")
            instances = result["data_list"][0]["pred_instances_3d"]
            self.assertEqual(len(instances), expected_kept)

    def test_should_filter_instance_private_method(self):
        """
        Test case for the `_should_filter_instance` helper logic.

        This test verifies the private method's behavior when assessing whether a
        detection should be removed.
        It checks:
        - That scores meeting the threshold are kept.
        - That scores below the threshold are filtered out.
        - That detections for unknown categories trigger filtering by default.

        Inputs for `ThresholdFilter._should_filter_instance`:
            - Collection of instance-score pairs spanning multiple categories.
            - Expected boolean indicating whether each instance should be filtered.
        Expected Outputs from `ThresholdFilter._should_filter_instance`:
            - Method return values matching the expected boolean in each case.
        """
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds,
            use_label=self.sample_use_label,
            logger=self.mock_logger,
        )
        test_cases = [
            ({"bbox_score_3d": 0.8}, "car", False),
            ({"bbox_score_3d": 0.3}, "car", True),
            ({"bbox_score_3d": 0.9}, "truck", True),
            ({"bbox_score_3d": 0.5}, "car", False),
        ]
        for instance, category, expected_filtered in test_cases:
            result = filter_obj._should_filter_instance(instance, category)
            self.assertEqual(result, expected_filtered, f"Failed for {instance} with category {category}")


if __name__ == "__main__":
    unittest.main()
