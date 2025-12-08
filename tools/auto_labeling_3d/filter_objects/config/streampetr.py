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

streampetr_pipeline = [
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
        type="NMSEnsembleModel",
        ensemble_setting=dict(
            weights=[1.0],
            iou_threshold=0.55,
            # Ensemble label groups. Each group is processed as one ensemble unit.
            ensemble_label_groups=[
                ["car"],
                ["truck"],
                ["bus"],
                ["pedestrian"],
                ["bicycle"],
            ],
        ),
    ),
    inputs=[
        dict(
            name="streampetr",
            info_path="./data/t4dataset/info/pseudo_infos_raw_streampetr.pkl",
            filter_pipeline=streampetr_pipeline,
        ),
    ],
)
