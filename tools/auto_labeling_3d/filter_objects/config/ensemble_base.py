custom_imports = dict(
    imports=["tools.auto_labeling_3d.filter_objects.filter", "tools.auto_labeling_3d.filter_objects.ensemble"],
    allow_failed_imports=False,
)

centerpoint_pipeline = [
    dict(
        type="ThresholdFilter",
        confidence_thresholds={
            "car": 0.35,
            "truck": 0.4,
            "bus": 0.4,
            "bicycle": 0.4,
            "pedestrian": 0.4,
        },
        use_label=["car", "truck", "bus", "bicycle", "pedestrian"],
    ),
]

bevfusion_pipeline = [
    dict(
        type="ThresholdFilter",
        confidence_thresholds={
            "car": 0.35,
            "truck": 0.4,
            "bus": 0.4,
            "bicycle": 0.4,
            "pedestrian": 0.4,
        },
        use_label=["car", "truck", "bus", "bicycle", "pedestrian"],
    ),
]

streampetr_pipeline = [
    dict(
        type="ThresholdFilter",
        confidence_thresholds={
            "car": 0.36,
            "truck": 0.39,
            "bus": 0.38,
            "bicycle": 0.41,
            "pedestrian": 0.43,
        },
        use_label=["car", "truck", "bus", "bicycle", "pedestrian"],
    ),
]

filter_pipelines = dict(
    type="Ensemble",
    config=dict(
        type="NMSEnsembleModel",
        ensemble_setting=dict(
            weights=[1.0, 1.0, 1.0],
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
            name="centerpoint",
            info_path="/workspace/work_dirs/auto_labeling_3d/pipeline_example/pseudo_infos_raw_centerpoint.pkl",
            filter_pipeline=centerpoint_pipeline,
        ),
        dict(
            name="bevfusion",
            info_path="/workspace/work_dirs/auto_labeling_3d/pipeline_example/pseudo_infos_raw_bevfusion.pkl",
            filter_pipeline=bevfusion_pipeline,
        ),
        dict(
            name="streampetr",
            info_path="/workspace/work_dirs/auto_labeling_3d/pipeline_example/pseudo_infos_raw_streampetr.pkl",
            filter_pipeline=streampetr_pipeline,
        ),
    ],
)
