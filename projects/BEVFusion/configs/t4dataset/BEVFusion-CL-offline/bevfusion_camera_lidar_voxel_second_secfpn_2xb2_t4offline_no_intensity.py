_base_ = [
    "../default/bevfusion_lidar_voxel_second_secfpn_1xb1_t4base.py",
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/base.py",
]

custom_imports = dict(imports=["projects.BEVFusion.bevfusion"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/user_name/"
train_gpu_size = 2
train_batch_size = 2
val_interval = 5
max_epochs = 30
backend_args = None

# range setting
point_cloud_range = [-122.4, -122.4, -3.0, 122.4, 122.4, 5.0]
voxel_size = [0.075, 0.075, 0.2]
grid_size = [3264, 3264, 41]
eval_class_range = {
    "car": 121,
    "truck": 121,
    "bus": 121,
    "bicycle": 121,
    "pedestrian": 121,
}

# model parameter
input_modality = dict(use_lidar=True, use_camera=True)
point_load_dim = 5  # x, y, z, intensity, ring_id
point_use_dim = 5
point_intensity_dim = 3
max_num_points = 10
max_voxels = [120000, 160000]
num_proposals = 500
image_size = [256, 704]
lidar_sweep_dims = [0, 1, 2, 4]
num_workers = 1
sweeps_num = 1

model = dict(
    type="BEVFusion",
    data_preprocessor=dict(
        voxelize_cfg=dict(
            max_num_points=max_num_points,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=max_voxels,
        ),
        type="Det3DDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
    ),
    pts_voxel_encoder=dict(type="HardSimpleVFE", num_features=4),
    pts_middle_encoder=dict(in_channels=4, sparse_shape=grid_size),
    img_backbone=dict(
        type="mmdet.SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",  # noqa: E251  # noqa: E501
        ),
    ),
    img_neck=dict(
        type="GeneralizedLSSFPN",
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        act_cfg=dict(type="ReLU", inplace=True),
        upsample_cfg=dict(mode="bilinear", align_corners=False),
    ),
    view_transform=dict(
        type="DepthLSSTransform",
        in_channels=256,
        out_channels=80,
        image_size=image_size,
        feature_size=[32, 88],
        # xbound=[-54.0, 54.0, 0.3],
        # ybound=[-54.0, 54.0, 0.3],
        # xbound=[-122.4, 122.4, 0.68],
        # ybound=[-122.4, 122.4, 0.68],
        xbound=[-122.4, 122.4, 0.3],
        ybound=[-122.4, 122.4, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        # dbound=[1.0, 60.0, 0.5],
        dbound=[1.0, 166.2, 1.4],
        downsample=2,
    ),
    fusion_layer=dict(type="ConvFuser", in_channels=[80, 256], out_channels=256),
    bbox_head=dict(
        num_proposals=num_proposals,
        num_classes=_base_.num_class,
        train_cfg=dict(
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        ),
        test_cfg=dict(
            grid_size=grid_size,
            voxel_size=voxel_size[0:2],
            pc_range=point_cloud_range[0:2],
        ),
        bbox_coder=dict(
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[0:2],
        ),
    ),
)

# TODO: support object sample
# db_sampler = dict(
#    data_root=data_root,
#    info_path=data_root +'nuscenes_dbinfos_train.pkl',
#    rate=1.0,
#    prepare=dict(
#        filter_by_difficulty=[-1],
#        filter_by_min_points=dict(
#            car=5,
#            truck=5,
#            bus=5,
#            trailer=5,
#            construction_vehicle=5,
#            traffic_cone=5,
#            barrier=5,
#            motorcycle=5,
#            bicycle=5,
#            pedestrian=5)),
#    classes=class_names,
#    sample_groups=dict(
#        car=2,
#        truck=3,
#        construction_vehicle=7,
#        bus=4,
#        trailer=6,
#        barrier=2,
#        motorcycle=6,
#        bicycle=6,
#        pedestrian=2,
#        traffic_cone=2),
#    points_loader=dict(
#        type='LoadPointsFromFile',
#        coord_type='LIDAR',
#        load_dim=5,
#        use_dim=[0, 1, 2, 3, 4],
#        backend_args=backend_args))

train_pipeline = [
    dict(
        type="BEVLoadMultiViewImageFromFiles",
        data_root=data_root,
        to_float32=True,
        color_type="color",
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=point_load_dim,
        use_dim=point_use_dim,
        backend_args=backend_args,
    ),
    # TODO: add feature
    # dict(
    #     type="IntensityNorm",
    #     alpha=10.0,
    #     intensity_dim=point_intensity_dim,
    #     div_factor=255.0,
    # ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=5,
        use_dim=lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # TODO: support object sample
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-1.571, 1.571],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[1.0, 1.0, 0.2],
    ),
    dict(type="BEVFusionRandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="ObjectNameFilter",
        classes=[
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ],
    ),
    dict(type="PointShuffle"),
    dict(
        type="Pack3DDetInputs",
        keys=["points", "img", "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_labels"],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "transformation_3d_flow",
            "pcd_rotation",
            "pcd_scale_factor",
            "pcd_trans",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ],
    ),
]

test_pipeline = [
    dict(
        type="BEVLoadMultiViewImageFromFiles",
        data_root=data_root,
        to_float32=True,
        color_type="color",
        backend_args=backend_args,
        test_mode=True,
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=5,
        use_dim=lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
        test_mode=True,
    ),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="Pack3DDetInputs",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "num_pts_feats",
            "num_views",
        ],
    ),
]

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_train_file_name,
        pipeline=train_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=False,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_test_file_name,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    metric="bbox",
    backend_args=backend_args,
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=eval_class_range,
    filter_attributes=_base_.filter_attributes,
)
test_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    metric="bbox",
    backend_args=backend_args,
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=eval_class_range,
    filter_attributes=_base_.filter_attributes,
)

# learning rate
lr = 0.0001
param_scheduler = [
    # learning rate scheduler
    # During the first (max_epochs * 0.4) epochs, learning rate increases from 0 to lr * 10
    # during the next epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=(max_epochs - 8),
        eta_min=lr * 1e-4,
        begin=8,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first (0.4 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=(max_epochs - 8),
        eta_min=1,
        begin=8,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# runtime settings
# Run validation for every val_interval epochs before max_epochs - 10, and run validation every 2 epoch after max_epochs - 10
train_cfg = dict(
    by_epoch=True, max_epochs=max_epochs, val_interval=val_interval, dynamic_intervals=[(max_epochs - 10, 2)]
)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)
auto_scale_lr = dict(enable=False, base_batch_size=train_gpu_size * train_batch_size)

if train_gpu_size > 1:
    sync_bn = "torch"

randomness = dict(seed=0, diff_rank_seed=False, deterministic=True)
