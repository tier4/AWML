_base_ = [
    "../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py",
]
custom_imports = dict(
    imports=["projects.StreamPETR.stream_petr"],
    allow_failed_imports=False,
)
custom_imports["imports"] += _base_.custom_imports["imports"]

backbone_norm_cfg = dict(type="LN", requires_grad=True)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)  # fix img_norm
# For nuScenes we usually do 10-class detection
class_names = _base_.class_names

num_gpus = 1
batch_size = 8
val_interval = 5
num_epochs = 50
num_cameras = 6
backend_args = None
stride = 16  # downsampling factor of extracted features form image
eval_class_range = {
    "car": 75,
    "truck": 75,
    "bus": 75,
    "bicycle": 75,
    "pedestrian": 75,
}

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
    streaming_test_mode=queue_length > 1,
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    img_backbone=dict(
        type="mmpretrain.ResNet",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="./work_dirs/ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth",
            prefix="backbone",
        ),
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        style="pytorch",
    ),
    img_neck=dict(type="CPFPN", in_channels=[1024, 2048], out_channels=256, num_outs=2),  ###remove unused parameters
    # img_roi_head=dict(
    #     type='mmdet.FocalHead',
    #     num_classes=len(class_names),
    #     in_channels=256,
    #     bbox_coder = dict(type='mmdet.DistancePointBBoxCoder'),
    #     loss_cls = dict(
    #         type='QualityFocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=1.0),
    #     loss_cls2d=dict(
    #         type='mmdet.QualityFocalLoss',
    #         use_sigmoid=True,
    #         beta=2.0,
    #         loss_weight=2.0),
    #     loss_centerness=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    #     loss_bbox2d=dict(type='mmdet.L1Loss', loss_weight=5.0),
    #     loss_iou2d=dict(type='mmdet.GIoULoss', loss_weight=2.0),
    #     loss_centers2d=dict(type='mmdet.L1Loss', loss_weight=10.0),
    #     train_cfg=dict(
    #         assigner2d=dict(
    #             type='mmdet.HungarianAssigner2D',
    #             cls_cost=dict(type='mmdet.FocalLossCost', weight=2.),
    #             reg_cost=dict(type='mmdet.BBoxL1Cost', weight=5.0, box_format='xywh'),
    #             iou_cost=dict(type='mmdet.IoUCost', iou_mode='giou', weight=2.0),
    #             centers2d_cost=dict(type='mmdet.BBox3DL1Cost', weight=10.0)))
    #     ),
    pts_bbox_head=dict(
        type="StreamPETRHead",
        num_classes=len(class_names),
        score_thres=0.1,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        with_dn=False,
        match_with_velo=False,
        scalar=10,  ##noise groups
        noise_scale=1.0,
        dn_weight=1.0,  ##dn loss weight
        split=0.75,  ###positive rate
        LID=True,
        with_position=True,
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
            0.0,
            0.0,
        ],  # setting the last two to zero will disable optimization for velocity
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
                        dict(type="PETRMultiheadFlashAttention", embed_dims=256, num_heads=8, dropout=0.1),
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
            cls_cost=dict(type="mmdet.FocalLossCost", weight=2.0),
            reg_cost=dict(type="mmdet.BBox3DL1Cost", weight=0.25),
            iou_cost=dict(
                type="mmdet.IoUCost", weight=0.0
            ),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range,
        ),
        train_cfg=dict(
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

data_root = "./data/"
info_directory_path = "info/cameraonly/streampetr/"

file_client_args = dict(backend="disk")


ida_aug_conf = {
    "resize_lim": (0.45, 0.55),
    "final_dim": (640, 960),  # (528, 720), (800,1200), (1088,1440)
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "rand_flip": True,
}

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
    dict(type="mmdet.ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=True, with_2d=False),
    dict(type="mmdet.PadMultiViewImage", size_divisor=stride),
    dict(type="mmdet.NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="StreamPETRLoadAnnotations2D"),
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
    dict(type="mmdet.ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=False, with_2d=False),
    dict(type="mmdet.PadMultiViewImage", size_divisor=stride),
    dict(type="mmdet.NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="StreamPETRLoadAnnotations2D"),
    dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys + ["prev_exists"]),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    drop_last=True,
    dataset=dict(
        type="StreamPETRDataset",
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_train_file_name,
        pipeline=train_pipeline,
        metainfo=_base_.metainfo,
        class_names=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        random_length=1,
        queue_length=queue_length,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="StreamPETRDataset",
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        class_names=class_names,
        modality=input_modality,
        random_length=0,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        queue_length=1,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="StreamPETRDataset",
        data_root=data_root,
        ann_file="info/cameraonly/streampetr_dummy/" + _base_.info_test_file_name,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        class_names=class_names,
        modality=input_modality,
        random_length=0,
        collect_keys=collect_keys + ["img", "prev_exists", "img_metas"],
        queue_length=1,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        backend_args=backend_args,
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
    ann_file=data_root + "info/cameraonly/streampetr_dummy/" + _base_.info_test_file_name,
    backend_args=backend_args,
    metric="bbox",
    class_names=class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=eval_class_range,
)


train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.01)  # bs 8: 2e-4 || bs 16: 4e-4,

# optim_wrapper = dict(
#     type="DebugOptimWrapper",
#     optimizer=optimizer,
#     clip_grad=dict(max_norm=35, norm_type=2),
# )

optim_wrapper = dict(type="NoCacheAmpOptimWrapper", optimizer=optimizer, clip_grad=dict(max_norm=35, norm_type=2))
# learning policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(
        type="CosineAnnealingLR",
        # TODO Figure out what T_max
        T_max=num_epochs,
        by_epoch=True,
    ),
]

# param_scheduler = [
#     dict(
#         T_max=20, begin=0, by_epoch=True, convert_to_iter_based=True, end=20, eta_min=0.001, type="CosineAnnealingLR"
#     ),
#     dict(
#         T_max=30, begin=20, by_epoch=True, convert_to_iter_based=True, end=80, eta_min=1e-08, type="CosineAnnealingLR"
#     ),
#     dict(
#         T_max=20,
#         begin=0,
#         by_epoch=True,
#         convert_to_iter_based=True,
#         end=20,
#         eta_min=0.8947368421052632,
#         type="CosineAnnealingMomentum",
#     ),
#     dict(
#         T_max=30,
#         begin=20,
#         by_epoch=True,
#         convert_to_iter_based=True,
#         end=80,
#         eta_min=1,
#         type="CosineAnnealingMomentum",
#     ),
# ]

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=10),
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=5,
        save_best="NuScenes metric/T4Metric/mAP",
        type="CheckpointHook",
        # by_epoch=False,
    ),  # alternative 'NuScenes metric/T4Metric/NDS'
)

load_from = None
resume_from = None
