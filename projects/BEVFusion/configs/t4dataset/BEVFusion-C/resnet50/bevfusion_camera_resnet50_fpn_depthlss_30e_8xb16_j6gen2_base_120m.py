_base_ = [
    "../default_bevfusion_camera_30e_8xb16_j6gen2_base_120m.py",
    "../../default/models/resnet50/camera_resnet50_fpn_depthlss_120m.py",
]

experiment_group_name = "bevfusion_camera/base/" + _base_.dataset_type
experiment_name = "bevfusion_camera_resnet50_fpn_depthlss_30e_8xb16_j6gen2_base_120m"
work_dir = "work_dirs/" + experiment_group_name + "/" + experiment_name

# model parameter
model = dict(
    type="BEVFusion",
    view_transform=dict(image_size=_base_.image_size),
    bbox_head=dict(
        class_names=_base_.class_names,
        train_cfg=dict(
            point_cloud_range=_base_.point_cloud_range,
            grid_size=_base_.grid_size,
            voxel_size=_base_.voxel_size,
        ),
        test_cfg=dict(
            grid_size=_base_.grid_size,
            voxel_size=_base_.voxel_size[0:2],
            pc_range=_base_.point_cloud_range[0:2],
        ),
        bbox_coder=dict(
            pc_range=_base_.point_cloud_range[0:2],
            voxel_size=_base_.voxel_size[0:2],
        ),
    ),
)
