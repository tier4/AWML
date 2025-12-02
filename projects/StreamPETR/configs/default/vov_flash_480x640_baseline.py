# 1. Tune hyperparams like seq_len, norm_eval, train_range, missing_image_replacement, large_image_sizes, feature_maps, datasets(xx1,x2,base)
_base_ = [
    "../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../autoware_ml/configs/detection3d/dataset/t4dataset/base.py",
]
custom_imports = dict(
    imports=["projects.StreamPETR.stream_petr"],
    allow_failed_imports=False,
)
custom_imports["imports"] += _base_.custom_imports["imports"]

backbone_norm_cfg = dict(type="LN", requires_grad=True)

info_directory_path = "info/cameraonly/baseline/"
data_root = "data/"


# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)  # fix img_norm

camera_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"]

class_names = _base_.class_names

metainfo = dict(classes=class_names)

batch_size = 4
test_batch_size = 1
val_interval = 5
num_epochs = 35
num_cameras = len(camera_order)
backend_args = None
stride = 16  # downsampling factor of extracted features form image

queue_length = 1
num_frame_losses = 1
collect_keys = [
    "lidar2img",
    "intrinsics",
    "extrinsics",
    "timestamp",
    "img_timestamp",
    "ego_pose",
    "ego_pose_inv",
    # "e2g_matrix",
    # "l2e_matrix",
]

eval_class_range = {
    "car": 51.2,
    "truck": 51.2,
    "bus": 51.2,
    "bicycle": 51.2,
    "pedestrian": 51.2,
}

input_modality = dict(
    use_lidar=True,  # lidar-related information (like ego-pose) is loaded, but pointcloud is not loaded or used
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)
model = dict(
    type="Petr3D",
    stride=stride,
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    img_backbone=dict(
        type="VoVNet",  ###use checkpoint to save memory
        spec_name="V-99-eSE",
        norm_eval=True,  # TODO: make true by default
        frozen_stages=-1,
        input_ch=3,
        out_features=(
            "stage4",
            "stage5",
        ),
    ),
    img_neck=dict(type="CPFPN", in_channels=[768, 1024], out_channels=256, num_outs=2),  ###remove unused parameters
    img_roi_head=dict(
        type="mmdet.FocalHead",
        num_classes=len(class_names),
        in_channels=256,
        bbox_coder=dict(type="mmdet.DistancePointBBoxCoder"),
        loss_cls2d=dict(type="mmdet.QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=2.0),
        loss_centerness=dict(type="mmdet.GaussianFocalLoss", reduction="mean", loss_weight=1.0),
        loss_bbox2d=dict(type="mmdet.L1Loss", loss_weight=5.0),
        loss_iou2d=dict(type="mmdet.GIoULoss", loss_weight=2.0),
        loss_centers2d=dict(type="mmdet.L1Loss", loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type="mmdet.HungarianAssigner2D",
                cls_cost=dict(type="mmdet.FocalLossCostAssigner", weight=2),
                reg_cost=dict(type="mmdet.BBoxL1CostAssigner", weight=5.0, box_format="xywh"),
                iou_cost=dict(type="mmdet.IoUCostAssigner", iou_mode="giou", weight=2.0),
                centers2d_cost=dict(type="mmdet.BBox3DL1CostAssigner", weight=10.0),
            )
        ),
    ),
    pts_bbox_head=dict(
        type="StreamPETRHead",
        num_classes=len(class_names),
        score_thres=0.0,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        with_dn=True,
        match_with_velo=False,
        scalar=10,  ##noise groups
        noise_scale=1.0,
        dn_weight=1.0,  ##dn loss weight
        split=0.75,  ###positive rate
        LID=True,
        with_position=True,
        use_bottom_center=False,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights=[
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],  # you can set the last two to zero will disable optimization for velocity
        transformer=dict(
            type="PETRTemporalTransformer",
            decoder=dict(
                type="PETRTransformerDecoder",
                post_norm_cfg=dict(type="LN"),
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type="PETRTemporalDecoderLayer",
                    attn_cfgs=[
                        dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type="PETRMultiheadFlashAttention", embed_dims=256, num_heads=8, dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=False,  ###use checkpoint to save memory
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                ),
            ),
        ),
        assigner=dict(
            type="mmdet.HungarianAssigner3D",
            cls_cost=dict(type="mmdet.FocalLossCostAssigner", weight=2.0),
            reg_cost=dict(type="mmdet.BBox3DL1CostAssigner", weight=0.25),
            iou_cost=dict(
                type="mmdet.IoUCostAssigner", weight=0.0
            ),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range,
        ),
        train_cfg=dict(
            grid_size=[512, 512, 1],
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            out_size_factor=4,
        ),
        bbox_coder=dict(
            type="mmdet.NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            voxel_size=voxel_size,
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=len(class_names),
        ),
        loss_cls=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type="mmdet.L1Loss", loss_weight=0.25),
        loss_iou=dict(type="mmdet.GIoULoss", loss_weight=0.0),
    ),
)

file_client_args = dict(backend="disk")

ida_aug_conf = {
    "resize_lim": 0.02,
    "final_dim": (480, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "rand_flip": True,
}

ida_aug_conf_test = {
    "resize_lim": 0.02,
    "final_dim": (480, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "rand_flip": False,
}

# augmementations = [
#     {"type": "mmpretrain.ColorJitter", "brightness":0.5, "contrast":0.5, "saturation":0.5, "hue":0.5},
#     {"type": "mmpretrain.GaussianBlur", "magnitude_range":(0,2), "prob":0.5},
# ]
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False,
        with_bbox_depth=False,
    ),
    dict(type="ObjectNameFilter", classes=class_names),
    # dict(type="mmdet.Filter2DByRange", range_2d=61.2),
    dict(type="mmdet.ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=True, with_2d=False),
    dict(
        type="mmdet.GlobalRotScaleTransImage",
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True,
    ),
    dict(type="mmdet.PadMultiViewImage", size_divisor=32),
    # dict(type="mmdet.ImageAugmentation", transforms=augmementations, p=0.75),
    dict(type="mmdet.NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="StreamPETRLoadAnnotations2D"),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys + ["prev_exists"]),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False,
        with_bbox_depth=False,
    ),
    dict(type="mmdet.ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf_test, training=False, with_2d=False),
    dict(type="mmdet.PadMultiViewImage", size_divisor=32),
    dict(type="mmdet.NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="StreamPETRLoadAnnotations2D"),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys + ["prev_exists"]),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type="GroupStreamingSampler", shuffle=True, batch_size=batch_size, trim_sequences=True),
    drop_last=True,
    dataset=dict(
        type="StreamPETRDataset",
        camera_order=camera_order,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_train_file_name,
        pipeline=train_pipeline,
        metainfo=metainfo,
        class_names=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        seq_mode=True,
        random_length=0,
        seq_split_num=1,
        queue_length=queue_length,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
        filter_empty_gt=False,
        reset_origin=True,
    ),
)
val_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="GroupStreamingSampler", shuffle=False, batch_size=test_batch_size),
    dataset=dict(
        type="StreamPETRDataset",
        test_mode=True,
        camera_order=camera_order,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=test_pipeline,
        seq_split_num=1,
        metainfo=metainfo,
        class_names=class_names,
        modality=input_modality,
        seq_mode=True,
        random_length=0,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        queue_length=1,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
        filter_empty_gt=False,
        reset_origin=True,
    ),
)
test_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="GroupStreamingSampler", shuffle=False, batch_size=test_batch_size),
    dataset=dict(
        type="StreamPETRDataset",
        test_mode=True,
        camera_order=camera_order,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_test_file_name,
        pipeline=test_pipeline,
        seq_split_num=1,
        metainfo=metainfo,
        class_names=class_names,
        modality=input_modality,
        seq_mode=True,
        random_length=0,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        queue_length=1,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
        filter_empty_gt=False,
        reset_origin=True,
    ),
)


val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    backend_args=backend_args,
    metric="bbox",
    class_names=class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=eval_class_range,
)
test_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    backend_args=backend_args,
    metric="bbox",
    class_names=class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=eval_class_range,
)


train_cfg = dict(
    by_epoch=True, max_epochs=num_epochs, val_interval=val_interval, dynamic_intervals=[(num_epochs - 5, 1)]
)
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=1000, val_interval=10)
val_cfg = dict()
test_cfg = dict()

lr = 1e-4
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)

# optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1),}))
optim_wrapper = dict(
    type="NoCacheAmpOptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    loss_scale="dynamic",
    clip_grad=dict(max_norm=1, norm_type=2),
)

# lrg policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        eta_min=lr * 1e-4,
    ),
]

vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")


default_hooks = dict(
    logger=dict(type="LoggerHook", interval=100),
    checkpoint=dict(
        interval=2,
        max_keep_ckpts=10,
        by_epoch=True,
        save_best="NuScenes metric/T4Metric/mAP",
        type="CheckpointHook",
    ),  # alternative 'NuScenes metric/T4Metric/NDS'
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl", timeout=3600 * 3),
)  # Since we are doing inference with batch_size=1, it can be slow so timeout needs to be increased

sync_bn = "torch"

load_from = "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/nuscenes/v1.0/nuscenes_vov99_baseline_320x800.pth"

auto_scale_lr = dict(base_batch_size=8, enable=True)

randomness = dict(seed=0, diff_rank_seed=False, deterministic=True)
