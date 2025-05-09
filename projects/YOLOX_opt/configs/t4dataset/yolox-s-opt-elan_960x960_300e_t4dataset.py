_base_ = [
    "../../../../autoware_ml/configs/detection2d/default_runtime.py",
    "../../../../autoware_ml/configs/detection2d/schedules/schedule_1x.py",
    "../../../../autoware_ml/configs/detection2d/dataset/t4dataset/yolox.py",
]

custom_imports = dict(
    imports=[
        "projects.YOLOX_opt.yolox",
        "autoware_ml.detection2d.metrics",
        "autoware_ml.detection2d.datasets",
        "projects.YOLOX_opt.yolox.models",
    ],
    allow_failed_imports=False,
)

# # dataset type setting
# dataset_type = "T4Dataset"
# info_train_file_name = "t4dataset_base_infos_train.pkl"
# info_val_file_name = "t4dataset_base_infos_val.pkl"
# info_test_file_name = "t4dataset_base_infos_test.pkl"

IMG_SCALE = (960, 960)
# IMG_SCALE = (1280, 960)

# parameter settings
img_scale = (960, 960)
max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 1
batch_size = 4
activation = "ReLU6"
num_workers = 4

base_lr = 0.001
# num_classes = 8

# model settings
model = dict(
    type="YOLOX",
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
        num_classes=len(_base_.classes),
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
# DATA_ROOT = "data/t4dataset/x2/db_v2_0_v1_1_cleaned"
# DATASET_TYPE = "T4Dataset"
# DATASET_CONFIG = "config/dataset_config/2d_linking.yaml"

data_root = ""
anno_file_root = "./data/t4dataset/x2/db_v2_0_v1_1_cleaned/"
dataset_type = "T4Dataset"

backend_args = None

train_pipeline = [
    dict(type="Mosaic", img_scale=IMG_SCALE, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-IMG_SCALE[0] // 2, -IMG_SCALE[1] // 2),
    ),
    dict(type="MixUp", img_scale=IMG_SCALE, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type="Resize", scale=IMG_SCALE, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    # dict(type="DefaultFormatBundle"),
    # dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
    dict(type="PackDetInputs"),
]

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=anno_file_root + "yolox_infos_train.json",
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=backend_args),
            dict(type="LoadAnnotations", with_bbox=True),
            # dict(type="RandomCropWithROI", crop_size=(1.0, 20.0)),
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=8),
        backend_args=backend_args,
    ),
    pipeline=train_pipeline,
)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(
        type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))
    ),
    dict(type="LoadAnnotations", with_bbox=True),
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

# data = dict(
#     samples_per_gpu=12,
#     workers_per_gpu=12,
#     persistent_workers=True,
#     train=train_dataset,
#     val=dict(
#         type=DATASET_TYPE,
#         data_root=DATA_ROOT,
#         version="annotation",
#         pipeline=test_pipeline,
#         dataset_config=DATASET_CONFIG,
#         split_type="val",
#     ),
#     test=dict(
#         type=DATASET_TYPE,
#         data_root=DATA_ROOT,
#         version="annotation",
#         pipeline=test_pipeline,
#         dataset_config=DATASET_CONFIG,
#         split_type="test",
#     ),
# )

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
        ann_file=anno_file_root + "yolox_infos_train.json",
        # Needs to be updated to tlr_infos_test.json once the dataset gets larger, and validation split is also added.
        # The splits were not modified so that we could compare them to previous models.
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
        ann_file=anno_file_root + "yolox_infos_train.json",
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

# val_evaluator = [
#     dict(type="VOCMetric", metric="mAP"),
# ]

# test_evaluator = val_evaluator


val_evaluator = [
    dict(type="VOCMetric", metric="mAP"),
]

test_evaluator = val_evaluator

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
optimizer = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    ),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
# optimizer_config = dict(grad_clip=None)

# learning rate
if max_epochs > 5:
    param_scheduler = [
        dict(
            # use quadratic formula to warm up 5 epochs
            # and lr is updated by iteration
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

log_config = dict(
    interval=1,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)

default_hooks = dict(
    checkpoint=dict(interval=interval, max_keep_ckpts=3)
)  # only keep latest 3 checkpoints

custom_hooks = [
    dict(
        type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48
    ),
    dict(type="SyncNormHook", priority=48),
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0001,
        update_buffers=True,
        priority=4,
    ),
    #    dict(type='MemoryProfilerHook', interval=100)
]

auto_scale_lr = dict(base_batch_size=batch_size)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]

visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)


# max_epochs = 300
# num_last_epochs = 15
# resume_from = None
# interval = 10

# # learning policy
# lr_config = dict(
#     _delete_=True,
#     policy="YOLOX",
#     warmup="exp",
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=5,  # 5 epoch
#     num_last_epochs=num_last_epochs,
#     min_lr_ratio=0.05,
# )

# runner = dict(type="EpochBasedRunner", max_epochs=max_epochs)

# custom_hooks = [
#     dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
#     dict(
#         type="SyncNormHook",
#         num_last_epochs=num_last_epochs,
#         interval=interval,
#         priority=48,
#     ),
#     dict(type="ExpMomentumEMAHook", resume_from=resume_from, momentum=0.0001, priority=49),
# ]
# checkpoint_config = dict(interval=interval)
# evaluation = dict(
#     save_best="auto",
#     # The evaluation interval is 'interval' when running epoch is
#     # less than ‘max_epochs - num_last_epochs’.
#     # The evaluation interval is 1 when running epoch is greater than
#     # or equal to ‘max_epochs - num_last_epochs’.
#     interval=interval,
#     dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
#     metric="bbox",
# )
# log_config = dict(interval=50)