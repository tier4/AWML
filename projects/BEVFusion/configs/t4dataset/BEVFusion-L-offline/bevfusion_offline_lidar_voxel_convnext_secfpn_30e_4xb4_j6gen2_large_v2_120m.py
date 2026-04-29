_base_ = [
    "../../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/j6gen2_large_full_categories.py",
    "../default/pipelines/default_offline_lidar_intensity_120m.py",
    "../default/schedulers/default_30e_8xb8_adamw_cosine.py",
    "../default/default_misc.py",
]

custom_imports = dict(imports=["projects.BEVFusion.bevfusion"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]
custom_imports["imports"] += ["autoware_ml.detection3d.datasets.transforms"]
custom_imports["imports"] += ["projects.ConvNeXt_PC"]

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/kang/lidardet_large/"

experiment_group_name = "bevfusion_full_20cls"
experiment_name = "convnext_v2_6sweep_large_50e_4xb4"
work_dir = "work_dirs/" + experiment_group_name + "/" + experiment_name

# ---------------------------------------------------------------------------
# v2 changes vs v1 (data-scale only — architecture unchanged for clean warm-start):
#   1. Dataset: lidardet_large (56k frames, +29%)
#   2. sweeps_num: 1 → 5  (6 total frames, denser far-range point cloud)
#   3. max_num_points: 10 → 32  (voxel capacity for denser multi-sweep clouds)
#   4. num_proposals: 500 → 700  (~35/class for 20 classes; better rare-class recall)
#   5. dense_heatmap_pooling_classes: [] (removed — latency not a concern offline)
#   6. max_voxels: [120k,160k] → [150k,200k]  (absorb denser multi-sweep clouds)
#   7. Warm-started from v1 epoch_30 checkpoint (all weights load; no shape change)
#
# large_arch (9×9 DW conv) deliberately excluded — tested separately on Blackwell
# as a fine-tune from this v2 checkpoint for a clean ablation.
# ---------------------------------------------------------------------------

sweeps_num = 5  # 6 frames total: current + 5 previous sweeps

eval_class_range = {cls: 120 for cls in _base_.class_names}

train_gpu_size = 4
train_batch_size = 4
test_batch_size = 2
num_workers = 16

model = dict(
    type="BEVFusion",
    voxelize_cfg=dict(
        max_num_points=32,
        max_voxels=[150000, 200000],  # increased for denser 5-sweep point clouds
        point_cloud_range=_base_.point_cloud_range,
        voxel_size=_base_.voxel_size,
        voxelize_reduce=True,
    ),
    data_preprocessor=dict(type="Det3DDataPreprocessor", pad_size_divisor=32),
    pts_voxel_encoder=dict(type="HardSimpleVFE", num_features=_base_.point_use_dim),
    pts_middle_encoder=dict(
        type="BEVFusionSparseEncoder",
        in_channels=_base_.point_use_dim,
        aug_features_min_values=[],
        aug_features_max_values=[],
        num_aug_features=0,
        sparse_shape=_base_.grid_size,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type="basicblock",
    ),
    pts_backbone=dict(
        type="ConvNeXt_PC",
        in_channels=256,
        out_channels=[256, 384, 384, 384, 384],
        depths=[3, 3, 2, 1, 1],
        out_indices=[1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        with_cp=True,
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[384, 384, 384],
        out_channels=[256, 256, 256],
        upsample_strides=[2, 4, 8],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    bbox_head=dict(
        type="BEVFusionHead",
        num_proposals=700,  # 500→700: ~35/class for 20 classes; better rare-class recall
        auxiliary=True,
        in_channels=768,
        hidden_channel=128,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        class_names=_base_.class_names,
        decoder_layer=dict(
            type="TransformerDecoderLayer",
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            norm_cfg=dict(type="LN"),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
        ),
        train_cfg=dict(
            dataset="t4datasets",
            point_cloud_range=_base_.point_cloud_range,
            grid_size=_base_.grid_size,
            voxel_size=_base_.voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type="HungarianAssigner3D",
                iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                cls_cost=dict(type="mmdet.FocalLossCost", gamma=2.0, alpha=0.25, weight=0.15),
                reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
                iou_cost=dict(type="IoU3DCost", weight=0.25),
            ),
        ),
        test_cfg=dict(
            dataset="t4datasets",
            grid_size=_base_.grid_size,
            out_size_factor=8,
            voxel_size=_base_.voxel_size[0:2],
            pc_range=_base_.point_cloud_range[0:2],
            nms_type=None,
            nms_clusters=[
                dict(
                    class_names=["car", "truck", "bus", "construction_vehicle", "tractor_unit",
                                 "semi_trailer", "train", "emergency_vehicle"],
                    nms_threshold=0.5,
                ),
                dict(class_names=["forklift", "kart", "other_vehicle"], nms_threshold=0.5),
                dict(class_names=["bicycle", "motorcycle"], nms_threshold=0.5),
                dict(
                    class_names=["pedestrian", "personal_mobility", "stroller"],
                    nms_threshold=0.175,
                ),
                dict(
                    class_names=["animal", "traffic_cone", "barrier", "pushable_pullable"],
                    nms_threshold=0.2,
                ),
            ],
        ),
        # Removed dense_heatmap_pooling: all classes use global top-K proposal selection.
        # Latency is not a concern for offline pseudo-labeling; removal improves recall.
        dense_heatmap_pooling_classes=[],
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            post_center_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=_base_.voxel_size[0:2],
            pc_range=_base_.point_cloud_range[0:2],
            code_size=10,
        ),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction="mean",
            loss_weight=1.0,
        ),
        loss_heatmap=dict(type="mmdet.GaussianFocalLoss", reduction="mean", loss_weight=1.0),
        loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
    ),
)

# ---------------------------------------------------------------------------
# Pipeline — sweeps_num=5 (6 frames total)
# ---------------------------------------------------------------------------
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
        use_dim=_base_.lidar_sweep_dims,  # includes time_lag (dim 4) for sweep age
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=_base_.backend_args,
        test_mode=False,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type="BEVFusionGlobalRotScaleTrans",
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=[0.5, 0.5, 0.2],
    ),
    dict(type="BEVFusionRandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(type="ObjectNameFilter", classes=_base_.class_names),
    dict(type="PointShuffle"),
    dict(
        type="Pack3DDetInputs",
        keys=["points", "img", "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_labels"],
        meta_keys=[
            "cam2img", "ori_cam2img", "lidar2cam", "lidar2img", "cam2lidar",
            "ori_lidar2img", "img_aug_matrix", "box_type_3d", "sample_idx",
            "lidar_path", "img_path", "transformation_3d_flow", "pcd_rotation",
            "pcd_scale_factor", "pcd_trans", "lidar_aug_matrix",
            "timestamp", "vehicle_type", "city",
        ],
    ),
]

test_pipeline = [
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
        test_mode=True,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(
        type="Pack3DDetInputs",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "cam2img", "ori_cam2img", "lidar2cam", "lidar2img", "cam2lidar",
            "ori_lidar2img", "img_aug_matrix", "box_type_3d", "sample_idx",
            "lidar_path", "img_path", "num_pts_feats", "num_views",
            "timestamp", "vehicle_type", "city",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=_base_.dataset_type,
        pipeline=train_pipeline,
        modality=_base_.input_modality,
        backend_args=_base_.backend_args,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_train_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        extended_name_mapping=_base_.extended_name_mapping,
        test_mode=False,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        filter_cfg=_base_.filter_cfg,
    ),
)

val_dataloader = dict(
    batch_size=test_batch_size,
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
        extended_name_mapping=_base_.extended_name_mapping,
        modality=_base_.input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=_base_.backend_args,
    ),
)

test_dataloader = dict(
    batch_size=test_batch_size,
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
        extended_name_mapping=_base_.extended_name_mapping,
        modality=_base_.input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=_base_.backend_args,
    ),
)

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    metric="bbox",
    backend_args=_base_.backend_args,
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
    backend_args=_base_.backend_args,
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=eval_class_range,
    filter_attributes=_base_.filter_attributes,
    save_csv=True,
)

# ---------------------------------------------------------------------------
# Scheduler — same LR as v1 (warm start, but data/arch change justifies full schedule)
# ---------------------------------------------------------------------------
lr = 7.0e-5
t_max = 8
max_epochs = 50
val_interval = 1

auto_scale_lr = dict(enable=False, base_batch_size=train_gpu_size * train_batch_size)

if train_gpu_size > 1:
    sync_bn = "torch"

model_wrapper_cfg = dict(type="MMDistributedDataParallel", static_graph=True)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        T_max=t_max,
        eta_min=lr * 10,
        begin=0,
        end=t_max,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=(max_epochs - t_max),
        eta_min=lr * 1e-4,
        begin=t_max,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=t_max,
        eta_min=0.85 / 0.95,
        begin=0,
        end=t_max,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=(max_epochs - t_max),
        eta_min=1,
        begin=t_max,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=val_interval)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=5,
                    save_best="NuScenes metric/T4Metric/mAP"),
)
log_processor = dict(window_size=50)

# Warm-start from v1 best checkpoint.
# Stage-3 depthwise_conv weights (7×7 → 9×9) will mismatch and re-initialize;
# all other weights load correctly. mmengine uses strict=False by default.
load_from = "work_dirs/bevfusion_full_20cls/convnext_30e_4xb4/best_NuScenes metric_T4Metric_mAP_epoch_30.pth"

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
    dict(
        type="MLflowVisBackend",
        save_dir="mlruns",
        exp_name=experiment_group_name,
        run_name=experiment_name,
        tags=dict(
            backbone="ConvNeXt_PC_large_arch",
            dataset="j6gen2_large",
            num_classes=str(len(_base_.class_names)),
            voxel_size="0.075_0.075_0.2",
            sweeps="6",
        ),
    ),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")
