custom_imports = dict(
    imports=["tools.auto_labeling_3d.filter_objects.filter", "tools.auto_labeling_3d.filter_objects.ensemble"],
    allow_failed_imports=False,
)

centerpoint_pipeline = [
    dict(
        type="ThresholdFilter",
        confidence_thresholds={
            "car": 0.35,
            "truck": 0.35,
            "bus": 0.35,
            "bicycle": 0.35,
            "pedestrian": 0.35,
        },
        use_label=["car", "truck", "bus", "bicycle", "pedestrian"],
    ),
]

bevfusion_pipeline = [
    dict(
        type="ThresholdFilter",
        confidence_thresholds={
            "car": 0.35,
            "truck": 0.35,
            "bus": 0.35,
            "bicycle": 0.35,
            "pedestrian": 0.35,
        },
        use_label=["car", "truck", "bus", "bicycle", "pedestrian"],
    ),
]

filter_pipelines = dict(
    type="Ensemble",
    config=dict(
        type="EnsembleModel",
        ensemble_setting=dict(
            weights=[1.0, 1.0],
            iou_threshold=0.55,
            skip_box_threshold=0.0,
        ),
    ),
    inputs=[
        dict(
            name="centerpoint",
            info_path="./data/t4dataset/info/pseudo_infos_raw_centerpoint.pkl",
            filter_pipeline=centerpoint_pipeline,
        ),
        dict(
            name="bevfusion",
            info_path="./data/t4dataset/info/pseudo_infos_raw_bevfusion.pkl",
            filter_pipeline=bevfusion_pipeline,
        ),
    ],
)
