_base_ = [
    "../../../../../autoware_ml/configs/detection2d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection2d/schedules/schedule_1x.py",
]

custom_imports = dict(
    imports=[
        "projects.YOLOX_opt_elan.yolox",
        "autoware_ml.detection2d.metrics",
        "autoware_ml.detection2d.datasets",
        "projects.YOLOX_opt_elan.yolox.models",
        "projects.YOLOX_opt_elan.yolox.models.yolox_multitask",
        "projects.YOLOX_opt_elan.yolox.transforms",
        "mmseg.evaluation.metrics",  # 引入分割评估指标
    ],
    allow_failed_imports=False,
)

# parameter settings
# IMG_SCALE = (960, 960)
IMG_SCALE = (1024, 512)
max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 1
batch_size = 16
activation = "ReLU6"
num_workers = 4
base_lr = 0.001


classes = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
palette = [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

seg_classes = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
seg_palette = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

# metainfo = dict(classes=classes, palette=palette)
metainfo = dict(classes=seg_classes, palette=seg_palette)

model = dict(
    type="YOLOXMultiTask",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,
        # batch_augments=[
        #     dict(
        #         type="BatchSyncRandomResize",
        #         random_size_range=(480, 800),
        #         size_divisor=32,
        #         interval=10,
        #     )
        # ],
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
        num_classes=8,
        in_channels=128,
        feat_channels=128,
        act_cfg=dict(type=activation),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.0),
        loss_bbox=dict(type="IoULoss", loss_weight=0.0),
        loss_obj=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.0),
        loss_l1=dict(type="L1Loss", loss_weight=0.0),
    ),
    mask_head=dict(
        type="YOLOXSegHead",
        in_channels=[128, 128, 128],
        feat_channels=128,
        num_classes=19,
        act_cfg=dict(type=activation),
        loss=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)

dataset_type = "CocoDataset"
data_root = "data/cityscapes/"
backend_args = None

train_pipeline = [
    # dict(type="Mosaic", img_scale=IMG_SCALE, pad_val=114.0),
    # dict(type="MixUp", img_scale=IMG_SCALE, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Resize", scale=IMG_SCALE, keep_ratio=False),
    dict(
        type="Pad",
        pad_to_square=False,
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0), seg=255),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="FixCityscapesPath", data_root=data_root, split="val"),
    dict(type="LoadAnnotations", with_bbox=True, with_seg=True),
    dict(type="Resize", scale=IMG_SCALE, keep_ratio=False),
    dict(type="Pad", pad_to_square=False, size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img="leftImg8bit/train", seg_map_path="gtFine/train"),
        ann_file="annotations/instancesonly_filtered_gtFine_train.json",
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=backend_args),
            dict(type="FixCityscapesPath", data_root=data_root, split="train"),
            dict(type="LoadAnnotations", with_bbox=True, with_seg=True),
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=8),
        backend_args=backend_args,
        metainfo=metainfo,
    ),
    pipeline=train_pipeline,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img="leftImg8bit/val", seg_map_path="gtFine/val"),
        ann_file="annotations/instancesonly_filtered_gtFine_val.json",
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        metainfo=metainfo,
    ),
)
test_dataloader = val_dataloader

val_evaluator = [
    dict(type="mmseg.IoUMetric", ignore_index=255, iou_metrics=["mIoU"], prefix="seg", classes=seg_classes)
]
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

optimizer = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

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

log_config = dict(
    interval=1,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)

default_hooks = dict(
    checkpoint=dict(interval=interval, max_keep_ckpts=3, save_best="seg/mIoU", rule="greater"),
    visualization=dict(
        type="DetVisualizationHook", draw=False, interval=50, show=False, wait_time=2, test_out_dir="vis_data"
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
    alpha=0.5,
)
