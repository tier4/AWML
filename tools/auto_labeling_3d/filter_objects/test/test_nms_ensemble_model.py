import logging
import unittest
from typing import Any, Dict, List
from unittest.mock import Mock

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
        """
        Initialize states before each individual test method.

        This setup assumes an object detection scenario with three classes:
        "car", "pedestrian", and "bicycle".
        """
        self.sample_classes = ["car", "pedestrian", "bicycle"]

    def test_filter_and_weight_instances_with_target_labels(self):
        """
        Test case for filtering and weighting instances with target labels.

        This test verifies that filtering by specified labels and weighting of scores works
        correctly as a preprocessing step for NMS. 
        It checks:
        - That labels other than the target label (e.g., "bicycle") are filtered out.
        - That the weight is correctly applied to the scores.
        Inputs for `filter_and_weight_instances`:
            - Instances: Three instances with labels "car", "pedestrian", and "bicycle".
            - Weight: 0.8
            - Target Labels: ["car", "pedestrian"]
        Expected Outputs from `filter_and_weight_instances`:
            - Filtered Instances: Two instances ("car" and "pedestrian").
            - Scores: Weighted scores for the two instances ([0.64, 0.56]).
        """
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
        """
        Test case for when no instances match the target labels.

        This test verifies that the filtering correctly handles cases where none of the input
        instances match the target labels.
        It checks:
        - That if an instance's label is not in the target labels, it is filtered out.
        - That the function returns empty lists/arrays when no instances are kept.

        Inputs for `filter_and_weight_instances`:
            - Instances: One instance with the label "bicycle".
            - Weight: 0.8
            - Target Labels: ["car", "pedestrian"]
        Expected Outputs from `filter_and_weight_instances`:
            - Filtered Instances: An empty list.
            - Boxes, Scores: An empty numpy array.
        """
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
        """
        Test case for handling an empty list of instances.

        This test verifies that the function correctly handles an empty list of input instances,
        which is an important edge case.
        It checks:
        - That the function executes without errors when given no instances.
        - That the function returns empty lists/arrays as expected.

        Inputs for `filter_and_weight_instances`:
            - Instances: An empty list.
            - Target Labels: ["car", "pedestrian"]
        Expected Outputs from `filter_and_weight_instances`:
            - Filtered Instances: An empty list.
            - Boxes, Scores: An empty numpy array.
        """
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
        """
        Initialize states before each individual test method.

        This setup assumes a scenario for ensembling object detection results from two models
        across three classes: "car", "pedestrian", and "bicycle". 
        It defines:
        - Ensemble settings, including weights for each model and an IoU threshold.
        - Sample prediction results from two models to be used as input for the ensemble tests.
        """
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
                            {
                                "bbox_label_3d": 1,
                                "bbox_score_3d": 0.7,
                                "bbox_3d": [10.0, 11.0, 0.0, 0.6, 0.6, 1.8, 0.0],
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
                                "bbox_label_3d": 2,
                                "bbox_score_3d": 0.6,
                                "bbox_3d": [20.0, 21.0, 0.0, 1.8, 0.8, 1.2, 0.2],
                            },
                        ]
                    }
                ],
            },
        ]

    def test_init(self):
        """
        Test case for NMSEnsembleModel initialization.

        This test verifies that providing ensemble settings and a logger to the
        constructor correctly configures the model.
        It checks:
        - That `model.settings` equals the supplied configuration.
        - That `model.logger` references the injected mock logger.
        - That `model.model_instances_type` resolves to `NMSModelInstances`.

        Inputs for `NMSEnsembleModel.__init__`:
            - `ensemble_setting`: Weights, IoU threshold, and label groups for three classes.
            - `logger`: Mock logger instance shared across the tests.
        Expected outputs from `NMSEnsembleModel.__init__`:
            - The created model exposes the provided configuration and logger.
            - `model.model_instances_type` remains the expected dataclass type.
        """
        model = NMSEnsembleModel(ensemble_setting=self.ensemble_settings, logger=self.mock_logger)
        self.assertEqual(model.settings, self.ensemble_settings)
        self.assertEqual(model.logger, self.mock_logger)
        self.assertEqual(model.model_instances_type, NMSModelInstances)

    def test_ensemble_single_model(self):
        """
        Test case for ensembling results from a single model.

        This test verifies that when `ensemble` receives only one model output, it
        simply returns the original predictions without modification.
        It checks:
        - That the returned dictionary matches the single input entry exactly.

        Inputs for `ensemble`:
            - `ensemble_results`: A list containing one model output with car and pedestrian detections.
        Expected outputs from `ensemble`:
            - The same dictionary that was provided as the lone element of the input list.
        """
        model = NMSEnsembleModel(ensemble_setting=self.ensemble_settings, logger=self.mock_logger)
        single_result = [self.sample_ensemble_results[0]]
        result = model.ensemble(single_result)
        self.assertEqual(result, single_result[0])

    def test_ensemble_basic_functionality(self):
        """
        Test case for combining predictions from two models across three classes.

        This test verifies that `ensemble` merges results while preserving critical
        metadata and required instance fields.
        It checks:
        - That the combined output retains the `metainfo` and `data_list` keys.
        - That the aggregated predictions list is non-empty.
        - That each merged instance includes score, label, and box information.

        Inputs for `ensemble`:
            - `ensemble_results`: Two model outputs covering "car", "pedestrian", and "bicycle" detections.
        Expected outputs from `ensemble`:
            - A single aggregated entry in `data_list` containing merged detections with the required keys present.
        """
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
        """
        Test case for overlapping detections from two models within one label group.

        This test verifies that `ensemble` applies NMS to suppress lower-scoring
        duplicates when two models produce highly overlapping car boxes.
        It checks:
        - That only a single car instance remains after ensembling.
        - That the retained instance corresponds to the higher weighted score (0.8).

        Inputs for `ensemble`:
            - `ensemble_results`: Two model outputs, each predicting a car box with significant overlap.
            - `ensemble_setting`: IoU threshold of 0.3 and weights [0.6, 0.4] for the two models.
        Expected outputs from `ensemble`:
            - One car detection with the higher score preserved and lower-scoring duplicate removed.
        """
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
        """
        Test case for independent label groups with overlapping spatial boxes.

        This test verifies that `ensemble` keeps detections from different label
        groups even when their boxes coincide.
        It checks:
        - That one instance per label group (car and pedestrian) is preserved.
        - That no cross-group suppression occurs despite identical coordinates.

        Inputs for `ensemble`:
            - `ensemble_results`: Two model outputs containing overlapping car and pedestrian detections.
            - `ensemble_setting`: Equal weights with label groups [["car"], ["pedestrian"]].
        Expected outputs from `ensemble`:
            - A result containing two instances, one per label group.
        """
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
        """
        Test case for ensembling when all models output no detections.

        This test verifies that `ensemble` handles empty prediction lists without
        errors and returns an empty set of instances.
        It checks:
        - That the response retains metadata even when no boxes are present.
        - That the resulting `pred_instances_3d` list is empty.

        Inputs for `ensemble`:
            - `ensemble_results`: Two model outputs whose prediction lists are empty.
        Expected outputs from `ensemble`:
            - A dictionary containing `metainfo` and `data_list` with an empty `pred_instances_3d` list.
        """
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
        """
        Test case for ensembling when only one model provides detections.

        This test verifies that `ensemble` keeps predictions from the model with
        detections even when the other model contributes none.
        It checks:
        - That the resulting list of predictions is non-empty.
        - That valid instances from the non-empty model are preserved.

        Inputs for `ensemble`:
            - `ensemble_results`: One output with a single car detection and another empty output.
        Expected outputs from `ensemble`:
            - A result containing the original detection in `pred_instances_3d`.
        """
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
        """
        Test case for mismatched ensemble weights and model count. (2 models, 1 weight)

        This test verifies that `ensemble` raises an assertion error when the
        configuration supplies fewer weights than models.
        It checks:
        - That invoking `ensemble` with inconsistent weights triggers `AssertionError`.

        Inputs for `ensemble`:
            - `ensemble_results`: Two model outputs.
            - `ensemble_setting`: Configuration containing only one weight.
        Expected outputs from `ensemble`:
            - An `AssertionError` indicating the mismatch between weights and models.
        """
        wrong_ensemble_settings = {
            "weights": [0.6],
            "iou_threshold": 0.5,
            "ensemble_label_groups": [["car"], ["pedestrian"], ["bicycle"]],
        }
        model = NMSEnsembleModel(ensemble_setting=wrong_ensemble_settings, logger=self.mock_logger)
        with self.assertRaises(AssertionError):
            model.ensemble(self.sample_ensemble_results)

    def test_extreme_iou_thresholds(self):
        """
        Test case for ensembling with a very low IoU threshold.

        This test verifies that when the IoU threshold is small, `ensemble`
        preserves detections that are spatially separated.
        It checks:
        - That both car detections remain in the final output because their IoU is below the threshold.

        Inputs for `ensemble`:
            - `ensemble_results`: Two model outputs predicting cars in different locations.
            - `ensemble_setting`: Equal weights with IoU threshold 0.1.
        Expected outputs from `ensemble`:
            - Two car detections retained in the aggregated predictions.
        """
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
        """
        Ensure `_calculate_iou` returns 1.0 for perfectly overlapping boxes.

        Inputs:
            - A single axis-aligned box (`box1`).
            - A batch containing one identical box (`box2`).
        Expected Outputs:
            - IoU array with a single value equal to 1.0, confirming full overlap.
        """
        box1 = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0])
        box2 = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        iou = _calculate_iou(box1, box2)
        self.assertEqual(len(iou), 1)
        self.assertAlmostEqual(iou[0], 1.0, places=6)

    def test_calculate_iou_no_overlap(self):
        """
        Ensure `_calculate_iou` returns 0.0 when boxes do not intersect.

        Inputs:
            - A reference box centered at the origin.
            - A far-away box located at (10, 10).
        Expected Outputs:
            - IoU array with a single value equal to 0.0, showing no overlap.
        """
        box1 = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0])
        box2 = np.array([[10.0, 10.0, 0.0, 2.0, 2.0, 1.5, 0.0]])
        iou = _calculate_iou(box1, box2)
        self.assertEqual(len(iou), 1)
        self.assertAlmostEqual(iou[0], 0.0, places=6)

    def test_calculate_iou_partial_overlap(self):
        """
        Ensure `_calculate_iou` captures partial overlap between two boxes.

        Inputs:
            - Two 4x4 axis-aligned boxes offset by (2, 2).
        Expected Outputs:
            - IoU array with one element equal to the analytically computed ratio 4/28.
        """
        box1 = np.array([0.0, 0.0, 0.0, 4.0, 4.0, 1.5, 0.0])
        box2 = np.array([[2.0, 2.0, 0.0, 4.0, 4.0, 1.5, 0.0]])
        iou = _calculate_iou(box1, box2)
        expected_iou = 4.0 / 28.0
        self.assertEqual(len(iou), 1)
        self.assertAlmostEqual(iou[0], expected_iou, places=6)

    def test_calculate_iou_multiple_boxes(self):
        """
        Ensure `_calculate_iou` evaluates IoU against a batch of boxes.

        Inputs:
            - A single box compared against three candidates with varying overlap.
        Expected Outputs:
            - IoU array of length three with values representing identical, partial,
              and zero-overlap cases respectively.
        """
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
        """
        Ensure `_nms_indices` keeps the highest-scoring box among overlaps.

        Inputs:
            - Three boxes, two of which overlap strongly.
            - Scores favoring the second box.
            - IoU threshold of 0.5.
        Expected Outputs:
            - Indices list retaining the high-score overlapping box and the distant box.
        """
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
        """
        Ensure `_nms_indices` keeps all boxes when none overlap.

        Inputs:
            - Three spatially separated boxes with different scores.
            - IoU threshold of 0.5.
        Expected Outputs:
            - All indices returned because suppression is unnecessary.
        """
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
        """
        Ensure `_nms_indices` handles the minimal case with one box.

        Inputs:
            - A single box and associated score.
        Expected Outputs:
            - A list containing only the index 0.
        """
        boxes = np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0]])
        scores = np.array([0.8])
        iou_threshold = 0.5
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        self.assertEqual(keep_indices, [0])

    def test_nms_indices_empty_input(self):
        """
        Ensure `_nms_indices` gracefully handles empty inputs.

        Inputs:
            - Empty arrays for boxes and scores.
        Expected Outputs:
            - An empty list of indices.
        """
        boxes = np.array([]).reshape(0, 7)
        scores = np.array([])
        iou_threshold = 0.5
        keep_indices = _nms_indices(boxes, scores, iou_threshold)
        self.assertEqual(keep_indices, [])

    def test_nms_indices_various_thresholds(self):
        """
        Ensure `_nms_indices` responds to different IoU thresholds.

        Inputs:
            - Three overlapping boxes with descending scores.
            - Multiple IoU thresholds ranging from 0.0 to 1.0.
        Expected Outputs:
            - Number of kept indices matching the expected count for each threshold.
        """
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
