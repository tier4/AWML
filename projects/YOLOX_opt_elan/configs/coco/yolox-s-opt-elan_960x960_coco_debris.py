_base_ = [
    "../../../../../autoware_ml/configs/detection2d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection2d/schedules/schedule_1x.py",
    "../../../../../autoware_ml/configs/detection2d/dataset/t4dataset/t4dataset_DynamicRecognition.py",
]

custom_imports = dict(
    imports=[
        "projects.YOLOX_opt_elan.yolox",
        "autoware_ml.detection2d.metrics",
        "projects.YOLOX_opt_elan.yolox.models",
        "projects.YOLOX_opt_elan.yolox.datasets.t4_dataset",
    ],
    allow_failed_imports=False,
)

# conv_yoloxで変換したumedaモデル
load_from = "/workspace/tlr-yolox/autoware-ml/checkpoints/epoch_12_ckpt_mmdet.pth"

img_scale = (960, 960)  # width, height
class_name = ("UNKNOWN", "CAR", "TRUCK", "BUS", "BICYCLE", "MOTORBIKE", "PEDESTRIAN", "ANIMAL", "DEBRIS")

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

mp_start_method = "spawn"

# training settings
max_epochs = 50  # 50
num_last_epochs = 15  # 15
interval = 1
batch_size = 12
base_lr = 0.001
activation = "ReLU6"
num_workers = 4

# model settings
model = dict(
    type="YOLOX",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,
        batch_augments=[
            dict(type="BatchSyncRandomResize", random_size_range=(480, 800), size_divisor=32, interval=10)
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
        num_classes=num_classes,
        in_channels=128,
        feat_channels=128,
        act_cfg=dict(type=activation),
    ),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)

# dataset settings
data_root = "/workspace/dataset/from_umeda/"
dataset_type = "CocoDataset"

backend_args = None

train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/t4_dataset_v1_2_train_cleaned_with_debris_pseudo.json",
        data_prefix=dict(img="train2017/"),
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=backend_args),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=3),
        backend_args=backend_args,
    ),
    pipeline=train_pipeline,
)


test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
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
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/instances_val2017_with_debris_pseudo.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
        test_mode=False,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

val_evaluator = [
    dict(type="VOCMetric", metric="mAP"),
    dict(
        type="CocoMetric",
        ann_file=f"{data_root}/annotations/instances_val2017_with_debris_pseudo.json",
        metric=["bbox"],
        format_only=False,
    ),
]

test_evaluator = val_evaluator

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

# learning rate
if max_epochs > 5:
    param_scheduler = [
        dict(
            # use quadratic formula to warm up 5 epochs
            # and lr is updated by iteration
            # TODO: fix default scope in get function
            type="mmdet.QuadraticWarmupLR",
            by_epoch=True,
            begin=0,
            end=5,
            convert_to_iter_based=True,
        ),
        dict(
            # use cosine lr from 5 to 285 epoch
            type="CosineAnnealingLR",
            eta_min=base_lr * 0.05,
            begin=5,
            T_max=max_epochs - num_last_epochs,
            end=max_epochs - num_last_epochs,
            by_epoch=True,
            convert_to_iter_based=True,
        ),
        dict(
            # use fixed lr during last 15 epochs
            type="ConstantLR",
            by_epoch=True,
            factor=1,
            begin=max_epochs - num_last_epochs,
            end=max_epochs,
        ),
    ]
else:
    param_scheduler = []

log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

default_hooks = dict(checkpoint=dict(interval=interval, max_keep_ckpts=3))  # only keep latest 3 checkpoints

custom_hooks = [
    dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
    dict(type="SyncNormHook", priority=48),
    dict(type="EMAHook", ema_type="ExpMomentumEMA", momentum=0.0001, update_buffers=True, priority=4),
    #    dict(type='MemoryProfilerHook', interval=100)
]

auto_scale_lr = dict(base_batch_size=batch_size)

vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]

visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
