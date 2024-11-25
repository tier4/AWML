filter_pipelines = EnsembleModel([
    ThresholdFilter(
        info_path="./data/t4dataset/info/pseudo_infos_raw_centerpoint.pkl",
        confidence_threshold=0.40,
        use_label=["car", "track", "bus", "bike", "pedestrian"]),
    ThresholdFilter(
        info_path="./data/t4dataset/info/pseudo_infos_raw_bevfusion.pkl",
        confidence_threshold=0.25,
        use_label=["bike", "pedestrian"]),
])
