_base_ = [
    "../../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/j6gen2_full_categories.py",
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
info_directory_path = "info/kang/lidardet/"

experiment_group_name = "bevfusion_full_20cls"
experiment_name = "convnext_30e_4xb4"
work_dir = "work_dirs/" + experiment_group_name + "/" + experiment_name

# ---------------------------------------------------------------------------
# Voxel / grid settings (inherited from default_offline_lidar_intensity_120m.py)
#   voxel_size = [0.075, 0.075, 0.2]   (x-y halved vs 0.17m baseline)
#   grid_size  = [3264, 3264, 41]
#
# NOTE on z-voxel: changing z from 0.2→0.5 would change grid_z from 41→16.
# BEVFusionSparseEncoder collapses z via its conv_out stride and produces
# output_channels * D_final features to the backbone. With grid_z=41 the
# chain of stride-2 stages gives D_final=2 → 256 channels (matching
# SECOND in_channels=256). With grid_z=16 D_final changes, breaking that
# assumption. If you want z=0.5, verify the sparse-encoder output channels
# and adjust pts_backbone in_channels accordingly.
# ---------------------------------------------------------------------------

# Override eval_class_range from pipeline base (5-class) to cover all 16 classes
eval_class_range = {cls: 120 for cls in _base_.class_names}

# 4 A100s — reduce batch per GPU to avoid OOM with larger BEV grid + ConvNeXt
train_gpu_size = 4
train_batch_size = 4
test_batch_size = 2
num_workers = 16

# ---------------------------------------------------------------------------
# Model: replace SECOND backbone with ConvNeXt_PC and expand head to 16 classes
# ---------------------------------------------------------------------------
model = dict(
    type="BEVFusion",
    voxelize_cfg=dict(
        max_num_points=10,
        max_voxels=[120000, 160000],
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
    # ConvNeXt_PC replaces SECOND. Input 256ch from sparse encoder → BEV map.
    # with_cp=True enables gradient checkpointing to reduce peak GPU memory.
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
        num_proposals=500,
        auxiliary=True,
        in_channels=768,  # 256 * 3 from SECONDFPN
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
        # Large vehicles produce stable dense heatmaps; small/rare classes use proposal matching
        dense_heatmap_pooling_classes=["car", "truck", "bus", "construction_vehicle",
                                       "tractor_unit", "semi_trailer", "bicycle", "motorcycle"],
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
# Pipeline — override ObjectNameFilter to cover all 16 classes
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
        sweeps_num=_base_.sweeps_num,
        load_dim=_base_.point_load_dim,
        use_dim=_base_.lidar_sweep_dims,
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
        sweeps_num=_base_.sweeps_num,
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
# Scheduler — 4 GPUs × 4 batch (effective 16, down from 8×8=64)
# LR: base 1.414e-4 × sqrt(16/64) = 7e-5. Peak = lr×10 = 7e-4 at epoch 8.
# Dual-phase cosine: warmup (0→t_max), then decay (t_max→max_epochs).
# Must be explicit here because the base file resolves lr at parse time.
# ---------------------------------------------------------------------------
lr = 7.0e-5
t_max = 8
max_epochs = 30
val_interval = 1  # checkpoint + eval every epoch — recover from crashes with ≤1 epoch lost

auto_scale_lr = dict(enable=False, base_batch_size=train_gpu_size * train_batch_size)

if train_gpu_size > 1:
    sync_bn = "torch"

# with_cp=True (gradient checkpointing) causes DDP hooks to fire twice per param.
# static_graph=True tells DDP the computation graph is fixed — the standard fix
# for DDP + torch.utils.checkpoint.
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
        eta_min=lr * 10,  # warmup: lr rises from 7e-5 → 7e-4 over 8 epochs
        begin=0,
        end=t_max,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=(max_epochs - t_max),
        eta_min=lr * 1e-4,  # decay: 7e-4 → 7e-9 over remaining 22 epochs
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

load_from = None

# ---------------------------------------------------------------------------
# Experiment monitoring — TensorBoard + MLflow
# MLflow logs to mlruns/ (file-based). Start the MLflow UI server separately:
#   mlflow server --host 0.0.0.0 --port 5500 --backend-store-uri /workspace/AWML/mlruns
# On resume, mmengine creates a new MLflow run in the same experiment; the
# model/optimizer state is fully restored from checkpoint (--resume auto).
# To log into the same MLflow run across resumes, pass the previous run_id
# via env var: MLFLOW_RUN_ID=<run_id> before launching training.
# ---------------------------------------------------------------------------
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
    dict(
        type="MLflowVisBackend",
        save_dir="mlruns",
        exp_name=experiment_group_name,
        run_name=experiment_name,
        tags=dict(
            backbone="ConvNeXt_PC",
            dataset="j6gen2_full_categories",
            num_classes=str(len(_base_.class_names)),
            voxel_size="0.075_0.075_0.2",
        ),
    ),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")
