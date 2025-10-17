import logging
from collections import defaultdict
from typing import Any, Dict, List
from unittest.mock import Mock
import unittest

from tools.auto_labeling_3d.filter_objects.filter.threshold_filter import ThresholdFilter


class TestThresholdFilter(unittest.TestCase):
    """Test cases for ThresholdFilter class."""

    def setUp(self):
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
            "data_list": [
                {"pred_instances_3d": []}
            ],
        }

    def test_init(self):
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds, use_label=self.sample_use_label, logger=self.mock_logger
        )
        self.assertEqual(filter_obj.settings["confidence_thresholds"], self.sample_confidence_thresholds)
        self.assertEqual(set(filter_obj.settings["use_label"]), set(self.sample_use_label))
        self.assertEqual(filter_obj.logger, self.mock_logger)

    def test_init_with_duplicate_use_label(self):
        use_label_with_duplicates = ["car", "pedestrian", "car", "bicycle"]
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds, use_label=use_label_with_duplicates, logger=self.mock_logger
        )
        self.assertEqual(len(filter_obj.settings["use_label"]), 3)
        self.assertEqual(set(filter_obj.settings["use_label"]), {"car", "pedestrian", "bicycle"})

    def test_basic_filtering(self):
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds, use_label=self.sample_use_label, logger=self.mock_logger
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
        filter_obj_zero = ThresholdFilter(confidence_thresholds={"car": 0.0}, use_label=["car"], logger=self.mock_logger)
        result_zero = filter_obj_zero.filter(test_data, "test_model")
        self.assertEqual(len(result_zero["data_list"][0]["pred_instances_3d"]), 3)
        filter_obj_one = ThresholdFilter(confidence_thresholds={"car": 1.0}, use_label=["car"], logger=self.mock_logger)
        result_one = filter_obj_one.filter(test_data, "test_model")
        instances = result_one["data_list"][0]["pred_instances_3d"]
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["bbox_score_3d"], 1.0)

    def test_empty_data(self):
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds, use_label=self.sample_use_label, logger=self.mock_logger
        )
        result = filter_obj.filter(self.empty_predicted_result_info, "test_model")
        self.assertIn("metainfo", result)
        self.assertIn("data_list", result)
        self.assertEqual(len(result["data_list"]), 1)
        self.assertEqual(len(result["data_list"][0]["pred_instances_3d"]), 0)

    def test_multiple_frames(self):
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
        mock_logger = Mock(spec=logging.Logger)
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds, use_label=self.sample_use_label, logger=mock_logger
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
                            {"bbox_label_3d": 1, "bbox_score_3d": 0.4, "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0]},
                            {"bbox_label_3d": 2, "bbox_score_3d": 0.2, "bbox_3d": [15.0, 16.0, 0.0, 1.8, 0.8, 1.2, 0.3]},
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
        filter_obj = ThresholdFilter(
            confidence_thresholds=self.sample_confidence_thresholds, use_label=self.sample_use_label, logger=self.mock_logger
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
