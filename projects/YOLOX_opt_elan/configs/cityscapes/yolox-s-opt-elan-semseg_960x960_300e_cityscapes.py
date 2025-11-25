# =========================================================
# 基础配置引用
# =========================================================
_base_ = [
    "../../../../../autoware_ml/configs/detection2d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection2d/schedules/schedule_1x.py",
    # 移除 dataset base，因为我们在下面完全重写了 dataset 配置
]

# =========================================================
# 自定义模块导入
# =========================================================
custom_imports = dict(
    imports=[
        "projects.YOLOX_opt_elan.yolox",
        "autoware_ml.detection2d.metrics",
        "autoware_ml.detection2d.datasets",
        "projects.YOLOX_opt_elan.yolox.models",
        "projects.YOLOX_opt_elan.yolox.models.yolox_multitask",
        "projects.YOLOX_opt_elan.yolox.transforms",
        "mmseg.evaluation.metrics",  # 显式导入 mmseg 的指标以防报错
    ],
    allow_failed_imports=False,
)

# =========================================================
# 全局超参数
# =========================================================
IMG_SCALE = (960, 960)  # 输入尺寸
img_scale = (960, 960)
max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 1
batch_size = 2      # 单卡 Batch Size
num_workers = 4     # 单卡 Workers
base_lr = 0.001     # 基础学习率
activation = "ReLU6" # 激活函数

# =========================================================
# 模型设置 (YOLOX + ELAN + SegHead)
# =========================================================
model = dict(
    type="YOLOXMultiTask",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type="BatchSyncRandomResize",
                random_size_range=(480, 800), # 多尺度训练范围
                size_divisor=32,
                interval=10,
            )
        ],
    ),
    backbone=dict(
        type="ELANDarknet",
        deepen_factor=0.33, # 注意：YOLOX-S/M/L 的因子不同，请确认 ELAN 的配置
        widen_factor=0.5,   # 请确认此处是否匹配您想使用的模型大小
        out_indices=(2, 3, 4),
        act_cfg=dict(type=activation),
    ),
    neck=dict(
        type="YOLOXPAFPN_ELAN",
        in_channels=[128, 256, 512], # 请确保与 Backbone 输出通道匹配
        out_channels=128,
        num_elan_blocks=2,
        act_cfg=dict(type=activation),
    ),
    bbox_head=dict(
        type="YOLOXHead",
        num_classes=19, # Cityscapes 19 类? (通常检测是8类，分割是19类，请确认)
        in_channels=128,
        feat_channels=128,
        act_cfg=dict(type=activation),
    ),
    # 自定义语义分割 Head
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

# =========================================================
# 数据集与 Pipeline
# =========================================================
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
# Cityscapes 用于检测通常只关注这8类 (instances)，用于分割关注19类。
# 如果您的目标是全类检测，保持19类；如果只是车辆行人检测，需改为8类。
classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
palette = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

metainfo = dict(classes=classes, palette=palette)
backend_args = None

# 1. 基础 Pipeline
pre_transform = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_seg=True), # 加载分割 Mask
]

# 2. 训练 Pipeline (包含 Mosaic/MixUp)
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
    # MMDetection 3.x 的 Resize 会自动处理 gt_seg_map
    dict(type="Resize", scale=IMG_SCALE, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0), seg=255),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"), # 自动将 gt_seg_map 打包进 data_sample.gt_sem_seg
]

# 3. 最后 15 Epoch 的 Pipeline (关闭 Mosaic/MixUp)
train_pipeline_stage2 = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_seg=True),
    dict(type="Resize", scale=IMG_SCALE, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0), seg=255),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]

# 4. 测试 Pipeline
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=IMG_SCALE, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0), seg=255),
    ),
    dict(
        type="PackDetInputs",
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    ),
]

# =========================================================
# Dataloader 配置
# =========================================================
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="MultiImageMixDataset", # YOLOX 必须的外层封装
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='leftImg8bit/train', 
                seg_map_path='gtFine/train'
            ),
            # 这里指向转换后的 COCO 格式 JSON (检测需要)
            ann_file='annotations/instancesonly_filtered_gtFine_train.json', 
            pipeline=pre_transform, # 只做 Load 操作
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=metainfo,
        ),
        pipeline=train_pipeline, # 这里做 Mosaic 等增强
    ),
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
        data_prefix=dict(
            img_path='leftImg8bit/val', 
            seg_map_path='gtFine/val'
        ),
        ann_file='annotations/instancesonly_filtered_gtFine_val.json',
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=metainfo,
    ),
)
test_dataloader = val_dataloader

# =========================================================
# 评估器 (Evaluator)
# =========================================================
val_evaluator = [
    # 1. 检测指标 (推荐使用 CocoMetric 而非 VOCMetric)
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/instancesonly_filtered_gtFine_val.json',
        metric='bbox',
        prefix='det', # Log显示: det/mAP
        classwise=True
    ),
    # 2. 分割指标
    dict(
        type='mmseg.IoUMetric', 
        ignore_index=255, 
        iou_metrics=['mIoU'], 
        prefix='seg' # Log显示: seg/mIoU
    )
]
test_evaluator = val_evaluator

# =========================================================
# 训练策略与 Hooks
# =========================================================
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=max_epochs, 
    val_interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)] # 最后阶段每 epoch 验证
)

optimizer = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

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

default_hooks = dict(
    checkpoint=dict(interval=interval, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(
        type='DetVisualizationHook',
        draw=True,
        interval=10, 
        show=False,
    ),
)

custom_hooks = [
    # 控制最后 15 epoch 关闭 Mosaic 的关键 Hook
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

# =========================================================
# 可视化后端
# =========================================================
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=0.5,
)