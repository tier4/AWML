_base_ = [
    "../../../../../autoware_ml/configs/detection2d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection2d/schedules/schedule_1x.py",
    "../../../../../autoware_ml/configs/detection2d/dataset/t4dataset/comlops.py",
]

custom_imports = dict(
    imports=[
        "projects.YOLOX_opt_elan.yolox",
        "autoware_ml.detection2d.metrics",
        "autoware_ml.detection2d.datasets",
        "projects.YOLOX_opt_elan.yolox.models",
        "projects.YOLOX_opt_elan.yolox.models.yolox_multitask",
        "projects.YOLOX_opt_elan.yolox.transforms",
    ],
    allow_failed_imports=False,
)

IMG_SCALE = (960, 960)

# parameter settings
img_scale = (960, 960)
max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 1
batch_size = 2
activation = "ReLU6"
num_workers = 4

base_lr = 0.001

# model settings
model = dict(
    type="YOLOXMultiTask",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type="BatchSyncRandomResize",
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10,
            )
        ],
    ),
    backbone=dict(
        type="ELANDarknet",
        deepen_factor=2,
        widen_factor=1,
        out_indices=(2, 3, 4),
        act_cfg=dict(type=activation),
    ),
    neck=dict(
        type="YOLOXPAFPN_ELAN",
        in_channels=[128, 256, 512],
        out_channels=128,
        num_elan_blocks=2,
        act_cfg=dict(type=activation),
    ),
    bbox_head=dict(
        type="YOLOXHead",
        num_classes=40,
        in_channels=128,
        feat_channels=128,
        act_cfg=dict(type=activation),
    ),
    # 新增 mask head
    mask_head=dict(
        type="YOLOXSegHead",  # 你需要在 projects.YOLOX_opt_elan.yolox.models 里实现
        in_channels=[128, 128, 128],  # 来自 neck 的特征通道数
        feat_channels=128,
        num_classes=40,
        act_cfg=dict(type=activation),
        loss=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)

data_root = ""
anno_file_root = "./data/comlops_mini_seg/"
dataset_type = "T4Dataset"

backend_args = None

# pipeline
train_pipeline = [
    # dict(type="Mosaic", img_scale=IMG_SCALE, pad_val=114.0),
    # dict(
    #     type="RandomAffine",
    #     scaling_ratio_range=(0.1, 2),
    #     border=(-IMG_SCALE[0] // 2, -IMG_SCALE[1] // 2),
    # ),
    # dict(type="MixUp", img_scale=IMG_SCALE, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="LoadAnnotations", with_bbox=True, with_seg=True),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Resize", scale=IMG_SCALE, keep_ratio=False),
    dict(
        type="Pad",
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0), seg=255),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    # dict(type="ResizeSegMask", size=IMG_SCALE),
    dict(type="PackDetInputs"),
]

classes = (
    "animal",
    "bicycle",
    "building",
    "bus",
    "car",
    "cone",
    "construction",
    "crosswalk",
    "dashed_lane_marking",
    "deceleration_line",
    "gate",
    "guide_post",
    "laneline_dash_white",
    "laneline_dash_yellow",
    "laneline_solid_green",
    "laneline_solid_red",
    "laneline_solid_white",
    "laneline_solid_yellow",
    "marking_arrow",
    "marking_character",
    "marking_other",
    "motorcycle",
    "other_obstacle",
    "other_pedestrian",
    "other_vehicle",
    "parking_lot",
    "pedestrian",
    "pole",
    "road",
    "road_debris",
    "sidewalk",
    "sky",
    "stopline",
    "striped_road_marking",
    "traffic_light",
    "traffic_sign",
    "train",
    "truck",
    # "unknown",
    "vegetation/terrain",
    "wall/fence",
)

palette = [
    (150, 120, 90),  # 0: animal (褐色)
    (119, 11, 32),  # 1: bicycle (深红)
    (70, 70, 70),  # 2: building (深灰)
    (0, 60, 100),  # 3: bus (深蓝)
    (0, 0, 142),  # 4: car (蓝)
    (250, 170, 30),  # 5: cone (橙色)
    (230, 150, 140),  # 6: construction (浅红)
    (140, 140, 200),  # 7: crosswalk (淡紫)
    (255, 255, 255),  # 8: dashed_lane_marking (白)
    (200, 200, 200),  # 9: deceleration_line (银灰)
    (190, 153, 153),  # 10: gate (浅褐)
    (250, 170, 30),  # 11: guide_post (橙)
    (255, 255, 255),  # 12: laneline_dash_white (白)
    (255, 255, 0),  # 13: laneline_dash_yellow (黄)
    (0, 255, 0),  # 14: laneline_solid_green (绿)
    (255, 0, 0),  # 15: laneline_solid_red (红)
    (255, 255, 255),  # 16: laneline_solid_white (白)
    (255, 215, 0),  # 17: laneline_solid_yellow (金黄)
    (0, 255, 255),  # 18: marking_arrow (青)
    (200, 0, 200),  # 19: marking_character (紫红)
    (150, 0, 150),  # 20: marking_other (紫)
    (0, 0, 230),  # 21: motorcycle (亮蓝)
    (80, 80, 80),  # 22: other_obstacle (灰)
    (250, 170, 160),  # 23: other_pedestrian (肉色)
    (100, 80, 200),  # 24: other_vehicle (紫蓝)
    (180, 165, 180),  # 25: parking_lot (浅灰紫)
    (220, 20, 60),  # 26: pedestrian (鲜红)
    (153, 153, 153),  # 27: pole (中灰)
    (128, 64, 128),  # 28: road (紫色 - 标准道路色)
    (110, 110, 110),  # 29: road_debris (灰)
    (244, 35, 232),  # 30: sidewalk (粉色 - 标准人行道色)
    (70, 130, 180),  # 31: sky (天蓝)
    (220, 220, 220),  # 32: stopline (亮灰)
    (160, 150, 180),  # 33: striped_road_marking (淡蓝灰)
    (250, 170, 30),  # 34: traffic_light (橙)
    (220, 220, 0),  # 35: traffic_sign (黄)
    (0, 80, 100),  # 36: train (蓝绿)
    (0, 0, 70),  # 37: truck (暗蓝)
    (107, 142, 35),  # 38: vegetation/terrain (植物绿)
    (102, 102, 156),  # 39: wall/fence (蓝紫灰)
]
metainfo = dict(classes=classes, palette=palette)

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=anno_file_root + "comlops_mini_seg_infos_train.json",
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=backend_args),
            dict(type="LoadAnnotations", with_bbox=True, with_seg=True),
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=8),
        backend_args=backend_args,
        metainfo=metainfo,
    ),
    pipeline=train_pipeline,
)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_seg=True),
    dict(type="Resize", scale=img_scale, keep_ratio=False),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
    # dict(type="ResizeSegMask", size=(IMG_SCALE)),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=anno_file_root + "comlops_mini_seg_infos_train.json",
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        metainfo=metainfo,
    ),
)

test_dataloader = val_dataloader

val_evaluator = [
    # 检测指标，日志里会显示 det/mAP
    dict(type="VOCMetric", metric="mAP", prefix="det"),
    # 分割指标，日志里会显示 seg/mIoU
    # 注意：这里 type 最好加上 mmseg. 前缀以防万一
    dict(type="mmseg.IoUMetric", ignore_index=255, iou_metrics=["mIoU"], prefix="seg"),
]
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
optimizer = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

# learning rate scheduler
if max_epochs > 5:
    param_scheduler = [
        dict(
            type="mmdet.QuadraticWarmupLR",
            by_epoch=True,
            begin=0,
            end=5,
            convert_to_iter_based=True,
        ),
        dict(
            type="CosineAnnealingLR",
            eta_min=base_lr * 0.05,
            begin=5,
            T_max=max_epochs - num_last_epochs,
            end=max_epochs - num_last_epochs,
            by_epoch=True,
            convert_to_iter_based=True,
        ),
        dict(
            type="ConstantLR",
            by_epoch=True,
            factor=1,
            begin=max_epochs - num_last_epochs,
            end=max_epochs,
        ),
    ]
else:
    param_scheduler = []

# logging
log_config = dict(
    interval=1,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)

default_hooks = dict(
    checkpoint=dict(interval=interval, max_keep_ckpts=3),
    visualization=dict(
        type="DetVisualizationHook",
        draw=True,  # 【关键】必须设为 True，否则只记录日志不画图
        interval=50,  # 验证/测试时，每隔多少个样本画一张（设为 1 则每张都画，设为 50 则抽样画）
        show=False,  # 服务器端设为 False
        wait_time=2,
        test_out_dir="vis_data",  # (可选) 也会把图片保存在本地这个文件夹下
    ),
)

custom_hooks = [
    dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
    dict(type="SyncNormHook", priority=48),
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0001,
        update_buffers=True,
        priority=4,
    ),
]

auto_scale_lr = dict(base_batch_size=batch_size)


vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]

visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")],
    name="visualizer",
    alpha=0.5,  # 设置透明度，方便看清 mask 下的原图
)
