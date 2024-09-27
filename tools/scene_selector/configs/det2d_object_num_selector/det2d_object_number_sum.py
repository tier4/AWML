scene_selector = dict(
    type = "Det2dObjectNumSelector",
    model_type = "YOLOX",
    confidence_threshold = 0.5,
    target_label = ["car", "bicycle", "pedestrian", "truck", "bus"],
    threshold_object_num = 20,
)
