t4_dataset_sensor_names = ["CAM_FRONT", "CAM_BACK"]
batch_size = 8

scene_selector = dict(
    type="OpenVocab2dObjectNumSelector",
    model_config_path="/workspace/projects/GLIP/configs/glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py",
    model_checkpoint_path="https://download.openmmlab.com/mmdetection/v3.0/glip/glip_l_mmdet-abfe026b.pth",
    confidence_threshold=0.6,
    target_and_threshold={
        "traffic cone": 1,
        "pedestrians": 10,
        "bicycle": 1,
        "kart": 1,
        "baby stroller": 1,
        "articulated bus": 1,
    },
    batch_size=batch_size,
)
