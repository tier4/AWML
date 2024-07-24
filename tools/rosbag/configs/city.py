custom_imports = dict(
    imports=["tools.rosbag.rosbag"], allow_failed_imports=False)

open_vocabulary_config_path = "projects/GLIP/configs/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py"
open_vocabulary_model_path = "work_dirs/pretrain/glip/glip_l_mmdet-abfe026b.pth"

selector = dict(
    min_crop_time=5,  #[sec]
    max_crop_time=10,  #[sec]
)

task = [
    dict(
        type="OpenVocabularyDetection",
        text="bicycle",
        confidence=0.5,
        object_num_threshold=1,
        scene_num_threshold=5,
    ),
    dict(
        type="OpenVocabularyDetection",
        text="traffic cone",
        confidence=0.3,
        object_num_threshold=2,
        scene_num_threshold=5,
    ),
    #dict(
    #    type="TrafficConeOnStreet",
    #    confidence=0.3,
    #    object_num_threshold=2,
    #    scene_num_threshold=3,
    #),
]
