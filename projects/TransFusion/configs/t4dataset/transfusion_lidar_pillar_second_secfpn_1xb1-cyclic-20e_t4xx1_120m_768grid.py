_base_ = [
    "./transfusion_lidar_pillar_second_secfpn_1xb1-cyclic-20e_t4xx1_base.py"
]

# user setting parameter
train_gpu_size = 1
train_batch_size = 4
max_epochs = 50

# range parameter
point_cloud_range = [-122.88, -122.88, -3.0, 122.88, 122.88, 7.0]
voxel_size = [0.32, 0.32, 10]
grid_size = [768, 768, 1]
eval_class_range = {
    "car": 120,
    "truck": 120,
    "bus": 120,
    "bicycle": 120,
    "pedestrian": 120,
}

# model parameter
pillar_feat_channels = [64, 64]
max_voxels = (60000, 60000)
sweeps_num = 1

###############################
##### override parameters #####
###############################

model = dict(
    type="TransFusion",
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=max_voxels,
        ),
    ),
    pts_voxel_encoder=dict(
        type="PillarFeatureNet",
        feat_channels=pillar_feat_channels,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type="PointPillarsScatter", output_shape=(grid_size[0], grid_size[1])),
    pts_bbox_head=dict(
        type="TransFusionHead",
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
        ),
    ),
    train_cfg=dict(
        pts=dict(
            dataset="nuScenes",
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )),
    test_cfg=dict(
        pts=dict(
            dataset="nuScenes",
            grid_size=grid_size,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
        )),
)

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=_base_.backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=_base_.backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-1.571, 1.571],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[1.0, 1.0, 0.2],
    ),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=_base_.class_names),
    dict(type="PointShuffle"),
    dict(
        type="Pack3DDetInputs",
        keys=[
            "points", "img", "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes",
            "gt_labels"
        ],
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
    # TODO: implement
    # dict(type='ObjectMinPointsFilter', min_num_points=5),
    # TODO: implement
    # dict(
    #     type='ObjectSample',
    #     db_sampler=dict(
    #         data_root=data_root,
    #         info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    #         rate=1.0,
    #         prepare=dict(
    #             filter_by_difficulty=[-1],
    #             filter_by_min_points=dict(
    #                 car=5,
    #                 truck=5,
    #                 bus=5,
    #                 trailer=5,
    #                 construction_vehicle=5,
    #                 traffic_cone=5,
    #                 barrier=5,
    #                 motorcycle=5,
    #                 bicycle=5,
    #                 pedestrian=5,
    #             ),
    #         ),
    #         classes=class_names,
    #         sample_groups=dict(
    #             car=2,
    #             truck=3,
    #             construction_vehicle=7,
    #             bus=4,
    #             trailer=6,
    #             barrier=2,
    #             motorcycle=6,
    #             bicycle=6,
    #             pedestrian=2,
    #             traffic_cone=2,
    #         ),
    #         points_loader=dict(
    #             type='LoadPointsFromFile',
    #             coord_type='LIDAR',
    #             load_dim=5,
    #             use_dim=[0, 1, 2, 3, 4],
    #             backend_args=backend_args),
    #     ),
    # ),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=_base_.backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=_base_.backend_args,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type="RandomFlip3D"),
        ],
    ),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_batch_size,
    dataset=dict(dataset=dict(pipeline=train_pipeline)),
)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_evaluator = dict(eval_class_range=eval_class_range)
test_evaluator = dict(eval_class_range=eval_class_range)
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    # learning rate scheduler
    # During the first (max_epochs * 0.4) epochs, learning rate increases from 0 to lr * 10
    # during the next epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=int(max_epochs * 0.4),
        eta_min=_base_.lr * 10,
        begin=0,
        end=int(max_epochs * 0.4),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=int(max_epochs * 0.6),
        eta_min=_base_.lr * 1e-4,
        begin=int(max_epochs * 0.4),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first (0.4 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=int(max_epochs * 0.4),
        eta_min=0.85 / 0.95,
        begin=0,
        end=int(max_epochs * 0.4),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=int(max_epochs * 0.6),
        eta_min=1,
        begin=int(max_epochs * 0.4),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# Default setting for scaling LR automatically
#   - `base_batch_size` = (1 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(base_batch_size=train_gpu_size * train_batch_size)
