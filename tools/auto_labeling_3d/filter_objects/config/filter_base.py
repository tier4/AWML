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

filter_pipelines = dict(
    type="Filter",
    input=dict(
        name="centerpoint",
        info_path="./data/t4dataset/info/pseudo_infos_raw_centerpoint.pkl",
        filter_pipeline=centerpoint_pipeline,
    ),
)
