camera_types = {"CAM_BACK_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT"}

bevfusion_CL_t4xx1_fusion_120m_v1 = dict(
    model="/workspace/work_dirs/BEVFusion-CL_t4xx1_fusion_120m_v1/bevfusion_camera_lidar_voxel_second_secfpn_1xb1_t4xx1.py",
    weights="/workspace/work_dirs/BEVFusion-CL_t4xx1_fusion_120m_v1/epoch_20.pth",
)
transfusion_lidar_20e_t4xx1_90m_768grid = dict(
    model="/workspace/work_dirs/transfusion/transfusion_lidar_pillar_second_secfpn_1xb8-cyclic-20e_t4xx1_90m_768grid.py",
    weights="/workspace/work_dirs/transfusion_v3/epoch_44.pth",
)

scene_selector = dict(
    type="MREMUncertaintyBasedSceneSelector",
    models={
        "BEVFusion_CL_120m": bevfusion_CL_t4xx1_fusion_120m_v1,
        "Transfusion_XX1_90m_v2": transfusion_lidar_20e_t4xx1_90m_768grid,
    },
    iou_threshold=0.5,
    min_score=0.1,
    p_thresh=10,
    d_thresh=50,
    min_rare_objects=3,
    rareness_threshold=0.1131,
    batch_size=1,  # Only batch size 1 is support at the momment, because sometimes images can be missing for some cameras, which makes batched processing infeasible.
)
