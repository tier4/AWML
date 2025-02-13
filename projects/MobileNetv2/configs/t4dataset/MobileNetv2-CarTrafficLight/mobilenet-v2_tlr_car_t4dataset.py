# Constants and Shared Configurations
BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_EPOCHS = 300
VAL_INTERVAL = 1
IMG_SCALE = 224
CHECKPOINT_INTERVAL = 1
LOG_INTERVAL = 100
ANN_FILE = "./data/info/tlr_classifier_car/"

# Base Configurations
_base_ = [
    "../../../../../autoware_ml/configs/classification2d/default_runtime.py",
    "../../../../../autoware_ml/configs/classification2d/schedules/schedule_1x.py",
    "../../../../../autoware_ml/configs/classification2d/dataset/t4dataset/tlr_classifier_car.py",
]

custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        "projects.MobileNetv2.mobilenet",
        "autoware_ml.classification2d.metrics",
    ],
)

# Shared Pipelines
resize_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="CropROI"),
    dict(type="Resize", scale=IMG_SCALE, backend="pillow", keep_ratio=True),
    dict(type="Pad", pad_to_square=True),
    dict(type="PackInputs"),
]

# Deploy Pipelines
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=IMG_SCALE, backend="pillow", keep_ratio=True),
    dict(type="Pad", pad_to_square=True),
    dict(type="PackInputs"),
]

# train_pipeline = [
#     dict(type="LoadImageFromFile"),
#     dict(type="CropROI", offset=1),
#     # This was removed from the config because horizontal flip can change classes like red_left to red_right.
#     dict(type="RandomFlip", prob=0.5, direction="horizontal"),
#     dict(type="ColorJitter", brightness=0.8, contrast=0.4, saturation=0.1),
#     dict(type="CustomResizeRotate", min_size=10, max_size=IMG_SCALE),
#     dict(type="RandomTranslatePad", aug_translate=True, size=IMG_SCALE),
#     dict(type='PackInputs'),
# ]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="CropROI"),
    dict(type="Resize", scale=IMG_SCALE, backend="pillow", keep_ratio=True),
    dict(type="Pad", pad_to_square=True),
    dict(type="Resize", scale=(IMG_SCALE, IMG_SCALE), backend="pillow"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="ColorJitter", brightness=0.8, contrast=0.4, saturation=0.1),
    dict(type="PackInputs"),
]
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    num_classes=len(_base_.classes),
    to_rgb=True,
)

dataset_type = "TLRClassificationDataset"
default_scope = "mmpretrain"

# Environment Configuration
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

launcher = "none"
load_from = "https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth"
log_level = "INFO"

# Model Definition
model = dict(
    type="ImageClassifier",
    backbone=dict(type="MobileNetV2", widen_factor=1.0),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        in_channels=1280,
        num_classes=len(_base_.classes),
        topk=(1, 2),
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)

# Optimizer and Scheduler
optimizer = dict(type="SGD", lr=5e-4, momentum=0.9, weight_decay=0.00004)
optim_wrapper = dict(optimizer=optimizer)
param_scheduler = dict(type="StepLR", by_epoch=True, gamma=0.98, step_size=1)

# Training Configuration
train_cfg = dict(by_epoch=True, max_epochs=NUM_EPOCHS, val_interval=VAL_INTERVAL)
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(ann_file=ANN_FILE + "tlr_infos_train.json", type=dataset_type, pipeline=train_pipeline),
    collate_fn=dict(type="default_collate"),
)

# Validation and Test Configuration
val_cfg = dict()
val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        ann_file=ANN_FILE
        + "tlr_infos_test.json",  # For the current dataset, val and test set are same. Needs to be fixed in future.
        type=dataset_type,
        pipeline=resize_pipeline,
    ),
    collate_fn=dict(type="default_collate"),
)

test_cfg = dict()
test_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(ann_file=ANN_FILE + "tlr_infos_test.json", type=dataset_type, pipeline=resize_pipeline),
    collate_fn=dict(type="default_collate"),
)

# Evaluators
val_evaluator = dict(type="MultiLabelMetric", topk=1)
test_evaluator = dict(
    type="TLRClassificationMetric",
    class_names=_base_.classes,
    topk=1,
    items=("precision", "recall", "f1-score", "support"),
    average=None,
)
# Hooks and Visualization
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", interval=CHECKPOINT_INTERVAL, save_best="multi-label/f1-score_top1", max_keep_ckpts=3
    ),
    logger=dict(type="LoggerHook", interval=LOG_INTERVAL),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="VisualizationHook", enable=False),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)

# Working Directory
work_dir = "./work_dirs/mobilenet-v2_tlr_car_t4dataset"
