_base_ = [
    "./second_secfpn_4xb16_121m_base_amp",
]

point_cloud_range = [-122.40, -122.40, -3.0, 122.40, 122.40, 5.0]
voxel_size = [0.24, 0.24, 8.0]
grid_size = [1020, 1020, 1]  # (122.40 / 0.24 == 510, 510 * 2 == 1020)
sweeps_num = 1
out_size_factor = 2

# user setting
work_dir = "work_dirs/centerpoint_2_5/" + _base_.dataset_type + "/second_secfpn_4xb16_121m_base_amp_rfs/"

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=_base_.point_load_dim,
        use_dim=_base_.point_load_dim,
        backend_args=_base_.backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=_base_.point_load_dim,
        use_dim=_base_.lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=_base_.backend_args,
        test_mode=False,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-1.571, 1.571],
        scale_ratio_range=[0.80, 1.20],
        translation_std=[1.0, 1.0, 0.2],
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes={{_base_.class_names}}),
    dict(type="ObjectMinPointsFilter", min_num_points=5),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

train_frame_object_sampler = dict(
    type="FrameObjectSampler",
    object_samplers=[
        dict(
            type="LowPedestriansObjectSampler",
            height_threshold=1.5,
            bev_distance_thresholds=[
                -50.0,
                -50.0,
                50.0,
                50.0,
            ],
        ),
    ],
)

train_dataloader = dict(
    sampler=dict(type="DistributedWeightedRandomSampler", shuffle=True),
    dataset=dict(
        type="T4FrameSamplerDataset",
        pipeline=train_pipeline,
        modality=_base_.input_modality,
        backend_args=_base_.backend_args,
        data_root=_base_.data_root,
        ann_file=_base_.info_directory_path + _base_.info_train_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        test_mode=False,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        repeat_sampling_factor=0.30,
        frame_object_sampler=train_frame_object_sampler,
    ),
)


model = dict(
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_voxels=(96000, 96000),
            deterministic=True,
        ),
    ),
    pts_voxel_encoder=dict(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
    ),
    pts_middle_encoder=dict(output_shape=(grid_size[0], grid_size[1])),
    pts_neck=dict(
        upsample_strides=[0.5, 1, 2],
    ),
    pts_bbox_head=dict(
        bbox_coder=dict(
            voxel_size=voxel_size,
            pc_range=point_cloud_range,
            out_size_factor=out_size_factor,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=out_size_factor,
        ),
    ),
    test_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
        ),
    ),
)

activation_checkpointing = ["pts_backbone"]
