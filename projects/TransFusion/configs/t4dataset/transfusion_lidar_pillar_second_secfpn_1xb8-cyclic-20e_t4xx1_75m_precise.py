_base_ = [
    "./transfusion_lidar_pillar_second_secfpn_1xb8-cyclic-20e_t4xx1_75m.py"
]

custom_imports = dict(
    imports=["projects.TransFusion.transfusion"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]

# user setting
train_batch_size = 4

# range setting
voxel_size = [0.2, 0.2, 10]
grid_size = [768, 768, 1]

model = dict(
    type="TransFusion",
    data_preprocessor=dict(voxel_layer=dict(voxel_size=voxel_size)),
    pts_voxel_encoder=dict(voxel_size=voxel_size),
    pts_middle_encoder=dict(output_shape=grid_size),
    pts_bbox_head=dict(bbox_coder=dict(voxel_size=voxel_size[:2])),
    train_cfg=dict(pts=dict(
        grid_size=grid_size,
        voxel_size=voxel_size,
    )),
    test_cfg=dict(pts=dict(
        grid_size=grid_size,
        voxel_size=voxel_size[:2],
    )),
)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_batch_size,
    dataset=dict(
        dataset=dict(ann_file=_base_.info_directory_path +
                     _base_.info_train_file_name)))
auto_scale_lr = dict(
    enable=False, base_batch_size=_base_.train_gpu_size * train_batch_size)
