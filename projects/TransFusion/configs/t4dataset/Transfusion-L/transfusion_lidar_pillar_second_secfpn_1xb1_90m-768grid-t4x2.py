_base_ = [
    "../default/dataset/transfusion_lidar_pillar_second_secfpn_1xb1_t4x2-base.py",
    "../default/model/transfusion_lidar_pillar_second_secfpn_1xb1_90m-768grid.py",
]

# train parameter
val_interval = 5
sweeps_num = 1

# eval parameter
eval_class_range = {
    "car": 90,
    "truck": 90,
    "bus": 90,
    "bicycle": 90,
    "pedestrian": 90,
}

###############################
##### override parameters #####
###############################

model = dict(
    type="TransFusion",
    pts_bbox_head=dict(num_classes=_base_.num_class),
    train_cfg=dict(val_interval=val_interval),
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
        with_attr_label=False,
    ),
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
    dict(type="PointsRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(type="ObjectNameFilter", classes=_base_.class_names),
    dict(type="PointShuffle"),
    dict(
        type="Pack3DDetInputs",
        keys=[
            "points",
            "img",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_bboxes",
            "gt_labels",
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

train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_evaluator = dict(eval_class_range=eval_class_range)
test_evaluator = dict(eval_class_range=eval_class_range)

train_cfg = dict(val_interval=val_interval)
