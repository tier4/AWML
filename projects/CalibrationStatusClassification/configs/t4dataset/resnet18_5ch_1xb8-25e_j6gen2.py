_base_ = [
    "mmpretrain::_base_/default_runtime.py",
    "mmpretrain::_base_/models/resnet18.py",
    "../../../../autoware_ml/configs/calibration_classification/dataset/t4dataset/gen2_base.py",
]

data_root = "/workspace/data/t4dataset"
info_directory_path = "/workspace/data/t4dataset/calibration_info/"
train_projection_vis_dir = None
val_projection_vis_dir = None
val_results_vis_dir = None
test_projection_vis_dir = None
test_results_vis_dir = None
binary_save_dir = None
batch_size = 16
max_epochs = 50
warmup_epochs = 5
num_workers = 4
max_depth = 128.0
dilation_size = 1
miscalibration_probability = 0.5

data_preprocessor = dict()

model = dict(
    type="mmpretrain.ImageClassifier",
    backbone=dict(
        type="ResNet",
        depth=18,
        in_channels=5,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        style="pytorch",
        norm_cfg=dict(type="BN", requires_grad=True),
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=2,
        in_channels=512,
        loss=dict(
            type="LabelSmoothLoss",
            loss_weight=1.0,
            label_smooth_val=0.1,
            reduction="mean",
        ),
        topk=(1,),
        init_cfg=dict(type="Normal", layer="Linear", mean=0, std=0.01, bias=0),
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ),
    clip_grad=dict(max_norm=2.0, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs - warmup_epochs,
        by_epoch=True,
        begin=warmup_epochs,
        end=max_epochs,
        eta_min=1e-6,
    ),
]

train_cfg = dict(
    by_epoch=True,
    max_epochs=max_epochs,
    val_interval=1,
    val_begin=5,
)

train_pipeline = [
    dict(
        type="CalibrationClassificationTransform",
        mode="train",
        max_depth=max_depth,
        dilation_size=dilation_size,
        undistort=True,
        miscalibration_probability=miscalibration_probability,
        enable_augmentation=True,
        transform_config=_base_.transform_config,
        data_root=data_root,
        projection_vis_dir=train_projection_vis_dir,
        results_vis_dir=None,
        binary_save_dir=None,
    ),
    dict(
        type="PackInputs",
        input_key="fused_img",
    ),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + _base_.info_train_file_name,
        pipeline=train_pipeline,
        data_root=data_root,
    ),
)

val_cfg = dict()

val_pipeline = [
    dict(
        type="CalibrationClassificationTransform",
        mode="val",
        max_depth=max_depth,
        dilation_size=dilation_size,
        undistort=True,
        miscalibration_probability=miscalibration_probability,
        enable_augmentation=False,
        transform_config=_base_.transform_config,
        data_root=data_root,
        projection_vis_dir=val_projection_vis_dir,
        results_vis_dir=val_results_vis_dir,
        binary_save_dir=None,
    ),
    dict(
        type="PackInputs",
        input_key="fused_img",
        # meta_keys=["img_path", "fused_img", "image", "sample_idx", "frame_id", "frame_idx"],  # Visualization
    ),
]

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=val_pipeline,
        data_root=data_root,
    ),
)

val_evaluator = [
    dict(type="mmpretrain.Accuracy", topk=(1,)),
    dict(type="mmpretrain.SingleLabelMetric"),
]

test_cfg = dict()

test_pipeline = [
    dict(
        type="CalibrationClassificationTransform",
        mode="test",
        max_depth=max_depth,
        dilation_size=dilation_size,
        undistort=True,
        miscalibration_probability=miscalibration_probability,
        enable_augmentation=False,
        transform_config=_base_.transform_config,
        data_root=data_root,
        projection_vis_dir=test_projection_vis_dir,
        results_vis_dir=test_results_vis_dir,
        binary_save_dir=binary_save_dir,
    ),
    dict(
        type="PackInputs",
        input_key="fused_img",
        meta_keys=["img_path", "fused_img", "image", "sample_idx", "frame_id", "frame_idx"],  # Visualization
    ),
]

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    shuffle=False,
    dataset=dict(
        type="T4CalibrationClassificationDataset",
        ann_file=info_directory_path + _base_.info_test_file_name,
        pipeline=test_pipeline,
        data_root=data_root,
        # indices=5,  # Use first 5 samples for testing
    ),
)

test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best=["accuracy/top1"],
        rule="greater",
        max_keep_ckpts=2,
        save_last=True,
        save_optimizer=True,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="VisualizationHook", enable=False),
)

custom_hooks = [
    dict(
        type="EarlyStoppingHook",
        monitor="accuracy/top1",
        patience=15,
        rule="greater",
        min_delta=0.005,
    ),
    dict(
        type="ResultVisualizationHook",
        transform_config=_base_.transform_config,
        data_root=data_root,
        results_vis_dir=val_results_vis_dir,
        phases=["val"],
    ),
    dict(
        type="ResultVisualizationHook",
        transform_config=_base_.transform_config,
        data_root=data_root,
        results_vis_dir=test_results_vis_dir,
        phases=["test"],
    ),
]

randomness = dict(seed=42, deterministic=False, diff_rank_seed=False)

auto_scale_lr = dict(enable=False, base_batch_size=batch_size)

log_processor = dict(
    window_size=20,
    by_epoch=True,
    custom_cfg=[
        dict(data_src="loss", method_name="mean", window_size="global"),
        dict(data_src="accuracy/top1", method_name="max", window_size="global"),
    ],
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends, name="visualizer")
