_base_ = [
    "../../../../autoware_ml/configs/detection3d/default_runtime.py",
]
custom_imports = dict(
    imports=["projects.StreamPETR.stream_petr"],
    allow_failed_imports=False,
)

backbone_norm_cfg = dict(type="LN", requires_grad=True)
sync_bn = "torch"
data_root = "data/nuscenes/"
data_prefix = dict(
    pts="samples/LIDAR_TOP",
    CAM_FRONT="samples/CAM_FRONT",
    CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
    CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
    CAM_BACK="samples/CAM_BACK",
    CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
    CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
    sweeps="sweeps/LIDAR_TOP",
)


# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)  # fix img_norm
# camera_order = None # This will lead to shuffled camera order
camera_order = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
# camera_order = None

# For nuScenes we usually do 10-class detection
class_names = [
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
]
metainfo = dict(classes=class_names)

num_gpus = 4
batch_size = 4
test_batch_size = 1
val_interval = 5
num_epochs = 90
num_cameras = 6
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
        init_cfg=dict(checkpoint="torchvision://resnet50", type="Pretrained"),
        type="mmpretrain.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        with_cp=False,
        style="pytorch",
    ),
    img_neck=dict(type="CPFPN", in_channels=[1024, 2048], out_channels=256, num_outs=2),  ###remove unused parameters
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
        use_gravity_center=True,
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

data_root = "./data/nuscenes/"
info_directory_path = ""

file_client_args = dict(backend="disk")

ida_aug_conf = {
    "resize_lim": (0.47, 0.625),
    "final_dim": (320, 800),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

ida_aug_conf_test = {
    "resize_lim": (0.47, 0.625),
    "final_dim": (320, 800),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
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
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    # dict(type="mmdet.Filter2DByRange", range_2d=61.2),
    dict(type="mmdet.ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=True, with_2d=True),
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
    # dict(type="StreamPETRLoadAnnotations2D"),
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
    dict(type="mmdet.ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf_test, training=False, with_2d=True),
    dict(type="mmdet.PadMultiViewImage", size_divisor=32),
    dict(type="mmdet.NormalizeMultiviewImage", **img_norm_cfg),
    # dict(type="StreamPETRLoadAnnotations2D"),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys + ["prev_exists"]),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type="GroupStreamingSampler", shuffle=True, batch_size=batch_size, trim_sequences=True),
    drop_last=True,
    dataset=dict(
        type="StreamPETRNuScenesDataset",
        camera_order=camera_order,
        data_root=data_root,
        ann_file=info_directory_path + "nuscenes_infos_train_updated.pkl",
        pipeline=train_pipeline,
        metainfo=metainfo,
        class_names=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        seq_mode=True,
        random_length=0,
        seq_split_num=2,
        queue_length=queue_length,
        data_prefix=data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
        filter_empty_gt=False,
    ),
)
val_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="GroupStreamingSampler", shuffle=False, batch_size=test_batch_size),
    dataset=dict(
        type="StreamPETRNuScenesDataset",
        test_mode=True,
        camera_order=camera_order,
        data_root=data_root,
        ann_file=info_directory_path + "nuscenes_infos_val_updated.pkl",
        pipeline=test_pipeline,
        seq_split_num=1,
        metainfo=metainfo,
        class_names=class_names,
        modality=input_modality,
        seq_mode=True,
        random_length=0,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        queue_length=1,
        data_prefix=data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
        filter_empty_gt=False,
    ),
)
test_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="GroupStreamingSampler", shuffle=False, batch_size=test_batch_size),
    dataset=dict(
        type="StreamPETRNuScenesDataset",
        test_mode=True,
        camera_order=camera_order,
        data_root=data_root,
        ann_file=info_directory_path + "nuscenes_infos_val_updated.pkl",
        pipeline=test_pipeline,
        seq_split_num=1,
        metainfo=metainfo,
        class_names=class_names,
        modality=input_modality,
        seq_mode=True,
        random_length=0,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        queue_length=1,
        data_prefix=data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
        filter_empty_gt=False,
    ),
)


val_evaluator = dict(
    type="NuScenesMetric",
    data_root=data_root,
    ann_file=data_root + "nuscenes_infos_val_updated.pkl",
    metric="bbox",
    backend_args=backend_args,
)
test_evaluator = val_evaluator


train_cfg = dict(
    by_epoch=True, max_epochs=num_epochs, val_interval=val_interval, dynamic_intervals=[(num_epochs - 5, 1)]
)
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=1000, val_interval=10)
val_cfg = dict()
test_cfg = dict()

lr = 2e-4
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)  # bs 8: 2e-4 || bs 16: 4e-4,

optim_wrapper = dict(
    type="NoCacheAmpOptimWrapper",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.25),
        }
    ),
    loss_scale="dynamic",
    clip_grad=dict(max_norm=35, norm_type=2),
)
# lrg policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        eta_min=lr * 1e-3,
    ),
]

vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")


default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=5,
        by_epoch=True,
        type="CheckpointHook",
    ),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl", timeout=3600),
)  # Since we are doing inference with batch_size=1, it can be slow so timeout needs to be increased
