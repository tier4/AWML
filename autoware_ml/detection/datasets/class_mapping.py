"""Class mapping."""

import warnings


def map_classes(
    outputs: list, model_classes: list, dataset_classes: list, custom_mapping: dict = None
):
    """Map classes from model's output to dataset's classes based on the class names.
    This function removes detections that do not belong to dataset's classes by default.
    If you want to customize mapping, specify class names by `custom_mapping`.
    Note that this function modifies the given `outputs` in-place.

    Args:
        outputs (list): Model outputs.
        model_classes (list): A list of class names that the model is supposed to detect.
        dataset_classes (list): A list of class names in the dataset.
        custom_mapping (dict): A dict containing the following key and value:
                                 key: class names in the model's output to map from.
                                 value: class names in the dataset to map to.
                                        (By setting None, you can drop the object class.)
                               For example,
                               ```
                                 custom_mapping={
                                   "car": "vehicle",
                                   "truck": "large_vehicle",
                                   "bus": "large_vehicle",
                                   "pedestrian": None
                                 }
                               ```
                               maps objects classified as `car` by the model to `vehicle`,
                               `truck` and `bus` to `large_vehicle`, and removes those
                               classified as `pedestrian`.
    """
    if len(outputs) == 0:
        return outputs

    _model_classes = [c.lower() for c in model_classes]
    _dataset_classes = [c.lower() for c in dataset_classes]

    if custom_mapping is None:
        class_idx_mapper = {
            _model_classes.index(c): _dataset_classes.index(c)
            for c in set(_model_classes).intersection(set(_dataset_classes))
        }
    else:
        class_idx_mapper = {
            _model_classes.index(c_from.lower()): _dataset_classes.index(c_to.lower())
            for c_from, c_to in custom_mapping.items()
            if c_to is not None
        }

    # warning
    ignored_model_classes = [
        model_classes[idx]
        for idx in set(range(len(model_classes))).difference(set(class_idx_mapper.keys()))
    ]
    if len(ignored_model_classes) > 0:
        warnings.warn(
            f"Object classes {ignored_model_classes} are ignored. "
            "If you do not want to, please configure `model_class_mapping` properly."
        )

    for _output in outputs:
        output = _output["pts_bbox"] if "pts_bbox" in _output.keys() else _output

        # map class indices
        output["labels_3d"].apply_(
            lambda x: class_idx_mapper[x] if x in class_idx_mapper.keys() else -1
        )
        labels_3d = output["labels_3d"]

        # remove detections that do not belong to dataset's classes
        output["boxes_3d"] = output["boxes_3d"][labels_3d != -1]
        output["scores_3d"] = output["scores_3d"][labels_3d != -1]
        output["labels_3d"] = output["labels_3d"][labels_3d != -1]


def map_dataset_classes(
    ground_truths: dict,
    data_class_mapping: dict,
):
    """Map classes from ground_truth to classes based on mapping for evaluation.
    Note that this function modifies the given `ground_truths` in-place.

    Args:
        ground_truths dict(list(dict(str, str))): Ground_truths. gt_dict["results"]
        data_class_mapping (dict): A dict containing the following key and value:
                                 key: class names in dataset to map from.
                                 value: class names in dataset to map to.

                               For example,
                               ```
                                data_class_mapping=dict(
                                    car="vehicle",
                                    pedestrian="others",
                                    bicycle="others",
                                )
                               ```
                               maps objects `car` in dataset to `all`,
                               and `pedestrian` and `bicycle` to `others`.
    """
    if len(ground_truths) == 0:
        return ground_truths
    for scene_token, scene_gts in ground_truths.items():
        for object_gt in scene_gts:
            object_gt["detection_name"]: str = data_class_mapping[object_gt["detection_name"]]
