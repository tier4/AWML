import logging
from typing import Any, Dict, List
from unittest.mock import Mock
import unittest
import numpy as np

from tools.auto_labeling_3d.filter_objects.ensemble.nms_ensemble_model import (
    NMSEnsembleModel,
    NMSModelInstances,
    _calculate_iou,
    _nms_indices,
)


class TestNMSModelInstances(unittest.TestCase):
    """Test cases for NMSModelInstances dataclass."""

    def setUp(self):
        self.sample_classes = ["car", "pedestrian", "bicycle"]

    def test_filter_and_weight_instances_with_target_labels(self):
        class_name_to_id = {name: idx for idx, name in enumerate(self.sample_classes)}
        instances = [
            {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
            {"bbox_label_3d": 1, "bbox_score_3d": 0.7, "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0]},
            {"bbox_label_3d": 2, "bbox_score_3d": 0.6, "bbox_3d": [20.0, 21.0, 0.0, 1.8, 0.8, 1.2, 0.2]},
        ]
        model_instances = NMSModelInstances(
            model_id=0, instances=instances, weight=0.8, class_name_to_id=class_name_to_id
        )
        target_label_names = ["car", "pedestrian"]
        filtered_instances, boxes, scores = model_instances.filter_and_weight_instances(target_label_names)
        self.assertEqual(len(filtered_instances), 2)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(scores), 2)
        expected_scores = [0.8 * 0.8, 0.7 * 0.8]
        np.testing.assert_array_almost_equal(scores, expected_scores)
        self.assertIsInstance(boxes, np.ndarray)
        self.assertEqual(boxes.shape, (2, 7))

    def test_filter_and_weight_instances_no_target_labels(self):
        class_name_to_id = {name: idx for idx, name in enumerate(self.sample_classes)}
        instances = [{"bbox_label_3d": 2, "bbox_score_3d": 0.6, "bbox_3d": [20.0, 21.0, 0.0, 1.8, 0.8, 1.2, 0.2]}]
        model_instances = NMSModelInstances(
            model_id=0, instances=instances, weight=0.8, class_name_to_id=class_name_to_id
        )
        target_label_names = ["car", "pedestrian"]
        filtered_instances, boxes, scores = model_instances.filter_and_weight_instances(target_label_names)
        self.assertEqual(len(filtered_instances), 0)
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(boxes.size, 0)
        self.assertEqual(scores.size, 0)

    def test_filter_and_weight_instances_with_empty_instances(self):
        class_name_to_id = {name: idx for idx, name in enumerate(self.sample_classes)}
        model_instances = NMSModelInstances(model_id=0, instances=[], weight=0.8, class_name_to_id=class_name_to_id)
        target_label_names = ["car", "pedestrian"]
        filtered_instances, boxes, scores = model_instances.filter_and_weight_instances(target_label_names)
        self.assertEqual(len(filtered_instances), 0)
        self.assertIsInstance(boxes, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(boxes.size, 0)
        self.assertEqual(scores.size, 0)


class TestNMSEnsembleModel(unittest.TestCase):
    """Test cases for NMSEnsembleModel class."""

    def setUp(self):
        self.mock_logger = Mock(spec=logging.Logger)
        self.sample_classes = ["car", "pedestrian", "bicycle"]
        self.ensemble_settings = {
            "weights": [0.6, 0.4],
            "iou_threshold": 0.5,
            "ensemble_label_groups": [["car"], ["pedestrian"], ["bicycle"]],
        }
        self.sample_ensemble_results = [
            {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]},
                            {"bbox_label_3d": 1, "bbox_score_3d": 0.7, "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0]},
                        ]
                    }
                ],
            },
            {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {"bbox_label_3d": 2, "bbox_score_3d": 0.6, "bbox_3d": [20.0, 21.0, 0.0, 1.8, 0.8, 1.2, 0.2]},
                        ]
                    }
                ],
            },
        ]

    def test_init(self):
        model = NMSEnsembleModel(ensemble_setting=self.ensemble_settings, logger=self.mock_logger)
        self.assertEqual(model.settings, self.ensemble_settings)
        self.assertEqual(model.logger, self.mock_logger)
        self.assertEqual(model.model_instances_type, NMSModelInstances)

    def test_ensemble_single_model(self):
        model = NMSEnsembleModel(ensemble_setting=self.ensemble_settings, logger=self.mock_logger)
        single_result = [self.sample_ensemble_results[0]]
        result = model.ensemble(single_result)
        self.assertEqual(result, single_result[0])

    def test_ensemble_basic_functionality(self):
        ensemble_settings = {
            "weights": [0.6, 0.4],
            "iou_threshold": 0.5,
            "ensemble_label_groups": [["car"], ["pedestrian"], ["bicycle"]],
        }
        model = NMSEnsembleModel(ensemble_setting=ensemble_settings, logger=self.mock_logger)
        result = model.ensemble(self.sample_ensemble_results)
        self.assertIn("metainfo", result)
        self.assertIn("data_list", result)
        self.assertEqual(len(result["data_list"]), 1)
        instances = result["data_list"][0]["pred_instances_3d"]
        self.assertGreater(len(instances), 0)
        for instance in instances:
            self.assertIn("bbox_score_3d", instance)
            self.assertIn("bbox_label_3d", instance)
            self.assertIn("bbox_3d", instance)

    def test_ensemble_with_overlapping_predictions(self):
        ensemble_settings = {
            "weights": [0.6, 0.4],
            "iou_threshold": 0.3,
            "ensemble_label_groups": [["car"]],
        }
        overlapping_results = [
            {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,
                                "bbox_score_3d": 0.8,
                                "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1],
                            }
                        ]
                    }
                ],
            },
            {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,
                                "bbox_score_3d": 0.6,
                                "bbox_3d": [1.1, 2.1, 0.0, 4.0, 2.0, 1.5, 0.1],
                            }
                        ]
                    }
                ],
            },
        ]
        model = NMSEnsembleModel(ensemble_setting=ensemble_settings, logger=self.mock_logger)
        result = model.ensemble(overlapping_results)
        instances = result["data_list"][0]["pred_instances_3d"]
        self.assertEqual(len(instances), 1)
        kept_instance = instances[0]
        self.assertEqual(kept_instance["bbox_score_3d"], 0.8)

    def test_ensemble_label_groups(self):
        ensemble_settings = {
            "weights": [0.5, 0.5],
            "iou_threshold": 0.1,
            "ensemble_label_groups": [["car"], ["pedestrian"]],
        }
        overlapping_different_classes = [
            {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,
                                "bbox_score_3d": 0.8,
                                "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1],
                            },
                            {
                                "bbox_label_3d": 1,
                                "bbox_score_3d": 0.7,
                                "bbox_3d": [1.0, 2.0, 0.0, 0.6, 0.6, 1.8, 0.0],
                            },
                        ]
                    }
                ],
            },
            {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,
                                "bbox_score_3d": 0.6,
                                "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1],
                            },
                            {
                                "bbox_label_3d": 1,
                                "bbox_score_3d": 0.5,
                                "bbox_3d": [1.0, 2.0, 0.0, 0.6, 0.6, 1.8, 0.0],
                            },
                        ]
                    }
                ],
            },
        ]
        model = NMSEnsembleModel(ensemble_setting=ensemble_settings, logger=self.mock_logger)
        result = model.ensemble(overlapping_different_classes)
        instances = result["data_list"][0]["pred_instances_3d"]
        self.assertEqual(len(instances), 2)

    def test_ensemble_empty_predictions(self):
        empty_results = [
            {"metainfo": {"classes": self.sample_classes}, "data_list": [{"pred_instances_3d": []}]},
            {"metainfo": {"classes": self.sample_classes}, "data_list": [{"pred_instances_3d": []}]},
        ]
        model = NMSEnsembleModel(ensemble_setting=self.ensemble_settings, logger=self.mock_logger)
        result = model.ensemble(empty_results)
        self.assertIn("metainfo", result)
        self.assertIn("data_list", result)
        self.assertEqual(len(result["data_list"][0]["pred_instances_3d"]), 0)

    def test_ensemble_partial_empty_predictions(self):
        partial_empty_results = [
            {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]}
                        ]
                    }
                ],
            },
            {"metainfo": {"classes": self.sample_classes}, "data_list": [{"pred_instances_3d": []}]},
        ]
        model = NMSEnsembleModel(ensemble_setting=self.ensemble_settings, logger=self.mock_logger)
        result = model.ensemble(partial_empty_results)
        instances = result["data_list"][0]["pred_instances_3d"]
        self.assertGreater(len(instances), 0)

    def test_weight_mismatch_assertion(self):
        wrong_ensemble_settings = {
            "weights": [0.6],
            "iou_threshold": 0.5,
            "ensemble_label_groups": [["car"], ["pedestrian"], ["bicycle"]],
        }
        model = NMSEnsembleModel(ensemble_setting=wrong_ensemble_settings, logger=self.mock_logger)
        with self.assertRaises(AssertionError):
            model.ensemble(self.sample_ensemble_results)

    def test_extreme_iou_thresholds(self):
        test_results_two_models = [
            {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {"bbox_label_3d": 0, "bbox_score_3d": 0.8, "bbox_3d": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.1]}
                        ]
                    }
                ],
            },
            {
                "metainfo": {"classes": self.sample_classes},
                "data_list": [
                    {
                        "pred_instances_3d": [
                            {
                                "bbox_label_3d": 0,
                                "bbox_score_3d": 0.6,
                                "bbox_3d": [5.0, 6.0, 0.0, 4.0, 2.0, 1.5, 0.1],
                            }
                        ]
                    }
                ],
            },
        ]
        ensemble_settings = {
            "weights": [0.5, 0.5],
            "iou_threshold": 0.1,
            "ensemble_label_groups": [["car"]],
        }
        model = NMSEnsembleModel(ensemble_setting=ensemble_settings, logger=self.mock_logger)
        result = model.ensemble(test_results_two_models)
        instances = result["data_list"][0]["pred_instances_3d"]
        self.assertEqual(len(instances), 2)


class TestNMSHelperFunctions(unittest.TestCase):
    """Test cases for NMS helper functions."""

    def test_calculate_iou_identical_boxes(self):
        box1 = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0])
        box2 = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        iou = _calculate_iou(box1, box2)
        self.assertEqual(len(iou), 1)
        self.assertAlmostEqual(iou[0], 1.0, places=6)

    def test_calculate_iou_no_overlap(self):
        box1 = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0])
        box2 = np.array([[10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 0.0]])
        iou = _calculate_iou(box1, box2)
        self.assertEqual(len(iou), 1)
        self.assertAlmostEqual(iou[0], 0.0, places=6)

    def test_calculate_iou_partial_overlap(self):
        box1 = np.array([0.0, 0.0, 0.0, 4.0, 4.0, 1.5, 0.0])
        box2 = np.array([[2.0, 2.0, 0.0, 4.0, 4.0, 1.5, 0.0]])
        iou = _calculate_iou(box1, box2)
        expected_iou = 4.0 / 28.0
        self.assertEqual(len(iou), 1)
        self.assertAlmostEqual(iou[0], expected_iou, places=6)

    def test_calculate_iou_multiple_boxes(self):
        box1 = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0])
        boxes = np.array(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0],
                [1.0, 1.0, 0.0, 2.0, 2.0, 1.5, 0.0],
                [10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 0.0],
            ]
        )
        ious = _calculate_iou(box1, boxes)
        self.assertEqual(len(ious), 3)
        self.assertAlmostEqual(ious[0], 1.0, places=6)
        self.assertGreater(ious[1], 0.0)
        self.assertLess(ious[1], 1.0)
        self.assertAlmostEqual(ious[2], 0.0, places=6)

    def test_nms_indices_basic(self):
        boxes = np.array(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0],
                [0.1, 0.1, 0.0, 2.0, 2.0, 1.5, 0.0],
                [10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 0.0],
            ]
        )
        scores = np.array([0.3, 0.9, 0.6])
        iou_threshold = 0.5
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        self.assertIn(1, keep_indices)
        self.assertIn(2, keep_indices)
        self.assertNotIn(0, keep_indices)

    def test_nms_indices_no_overlap(self):
        boxes = np.array(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0],
                [10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 0.0],
                [20.0, 20.0, 0.0, 2.0, 2.0, 1.5, 0.0],
            ]
        )
        scores = np.array([0.3, 0.9, 0.6])
        iou_threshold = 0.5
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        self.assertEqual(len(keep_indices), 3)
        self.assertEqual(set(keep_indices), {0, 1, 2})

    def test_nms_indices_single_box(self):
        boxes = np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0]])
        scores = np.array([0.8])
        iou_threshold = 0.5
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        self.assertEqual(keep_indices, [0])

    def test_nms_indices_empty_input(self):
        boxes = np.array([]).reshape(0, 7)
        scores = np.array([])
        iou_threshold = 0.5
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        self.assertEqual(keep_indices, [])

    def test_nms_indices_various_thresholds(self):
        test_cases = [
            (0.0, 1),
            (0.3, 1),
            (0.6, 2),
            (0.9, 3),
            (1.0, 3),
        ]
        for iou_threshold, expected_kept in test_cases:
            boxes = np.array(
                [
                    [0.0, 0.0, 0.0, 4.0, 4.0, 1.5, 0.0],
                    [1.0, 1.0, 0.0, 4.0, 4.0, 1.5, 0.0],
                    [0.5, 0.5, 0.0, 4.0, 4.0, 1.5, 0.0],
                ]
            )
            scores = np.array([0.5, 0.9, 0.7])
            keep_indices = _nms_indices(boxes, scores, iou_threshold)
            self.assertEqual(len(keep_indices), expected_kept)


if __name__ == "__main__":
    unittest.main()
