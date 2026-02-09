_base_ = [
    "../model/frnet.py",
    "../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../autoware_ml/configs/segmentation3d/dataset/t4dataset/j6gen2_base.py",
]

custom_imports = dict(
    imports=[
        "projects.FRNet.frnet.datasets",
        "projects.FRNet.frnet.datasets.transforms",
        "projects.FRNet.frnet.models",
    ],
    allow_failed_imports=False,
)
custom_imports["imports"] += _base_.custom_imports["imports"]

# user settings
batch_size = 8
num_workers = 16
iterations = 150000
val_interval = 1500

# LiDAR settings (Hesai OT128)
lidar_h = 128
range_w = 4096
frustum_w = 1024  # 4096 / 4
fov_up = 15.0
fov_down = -25.0
SOURCES = ["LIDAR_FRONT_UPPER", "LIDAR_LEFT_UPPER", "LIDAR_RIGHT_UPPER", "LIDAR_REAR_UPPER"]

dataset_type = _base_.dataset_type
data_root = "data/t4dataset/"
ignore_index = 26

class_mapping = {
    "drivable_surface": 0,
    "other_flat_surface": 1,
    "sidewalk": 2,
    "manmade": 3,
    "vegetation": 4,
    "car": 5,
    "bus": 6,
    "emergency_vehicle": 7,
    "train": 8,
    "truck": 9,
    "tractor_unit": 10,
    "semi_trailer": 11,
    "construction_vehicle": 12,
    "forklift": 13,
    "kart": 14,
    "motorcycle": 15,
    "bicycle": 16,
    "pedestrian": 17,
    "personal_mobility": 18,
    "animal": 19,
    "pushable_pullable": 20,
    "traffic_cone": 21,
    "stroller": 22,
    "debris": 23,
    "other_stuff": 24,
    "noise": 25,
    "ghost_point": 25,
    "out_of_sync": ignore_index,
    "unpainted": ignore_index,
}
num_classes = 27  # 26 classes + 1 for unknown
metainfo = _base_.metainfo
metainfo.update(dict(class_mapping=class_mapping))
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

model = dict(
    data_preprocessor=dict(H=lidar_h, W=frustum_w, fov_up=fov_up, fov_down=fov_down, ignore_index=ignore_index),
    backbone=dict(output_shape=(lidar_h, frustum_w)),
    decode_head=dict(num_classes=num_classes, ignore_index=ignore_index),
    auxiliary_head=[
        dict(
            type="FrustumHead",
            channels=128,
            num_classes=num_classes,
            dropout_ratio=0,
            loss_ce=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type="LovaszLoss", loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type="BoundaryLoss", loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=ignore_index,
        ),
        dict(
            type="FrustumHead",
            channels=128,
            num_classes=num_classes,
            dropout_ratio=0,
            loss_ce=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type="LovaszLoss", loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type="BoundaryLoss", loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=ignore_index,
            indices=2,
        ),
        dict(
            type="FrustumHead",
            channels=128,
            num_classes=num_classes,
            dropout_ratio=0,
            loss_ce=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type="LovaszLoss", loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type="BoundaryLoss", loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=ignore_index,
            indices=3,
        ),
        dict(
            type="FrustumHead",
            channels=128,
            num_classes=num_classes,
            dropout_ratio=0,
            loss_ce=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type="LovaszLoss", loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type="BoundaryLoss", loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=ignore_index,
            indices=4,
        ),
    ],
)

pre_transform = [
    dict(
        type="LoadPointsWithIdentifierFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type="LoadSegAnnotationsWithIdentifier3D",
        seg_3d_dtype="np.uint8",
        backend_args=backend_args,
    ),
    dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
]
train_pipeline = [
    dict(
        type="LoadPointsWithIdentifierFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type="LoadSegAnnotationsWithIdentifier3D",
        seg_3d_dtype="np.uint8",
        backend_args=backend_args,
    ),
    dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(
        type="FrustumMix",
        H=lidar_h,
        W=frustum_w,
        fov_up=fov_up,
        fov_down=fov_down,
        num_areas=[3, 4, 5, 6],
        pre_transform=pre_transform,
        prob=1.0,
    ),
    dict(type="InstanceCopy", instance_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pre_transform=pre_transform, prob=1.0),
    dict(
        type="RangeInterpolation",
        H=lidar_h,
        W=range_w,
        fov_up=fov_up,
        fov_down=fov_down,
        ignore_index=ignore_index,
    ),
    dict(type="Pack3DDetInputs", keys=["points", "pts_semantic_mask"]),
]
test_pipeline = [
    dict(
        type="LoadPointsWithIdentifierFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type="LoadSegAnnotationsWithIdentifier3D",
        seg_3d_dtype="np.uint8",
        backend_args=backend_args,
    ),
    dict(
        type="RangeInterpolation",
        H=lidar_h,
        W=range_w,
        fov_up=fov_up,
        fov_down=fov_down,
        ignore_index=ignore_index,
    ),
    dict(type="Pack3DDetInputs", keys=["points"], meta_keys=["num_points", "lidar_path", "num_pts_feats"]),
]
tta_pipeline = [
    dict(
        type="LoadPointsWithIdentifierFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type="LoadSegAnnotationsWithIdentifier3D",
        seg_3d_dtype="np.uint8",
        backend_args=backend_args,
    ),
    dict(
        type="RangeInterpolation",
        H=lidar_h,
        W=range_w,
        fov_up=fov_up,
        fov_down=fov_down,
        ignore_index=ignore_index,
    ),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=0.0, flip_ratio_bev_vertical=0.0),
                dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=0.0, flip_ratio_bev_vertical=1.0),
                dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=0.0),
                dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=1.0),
            ],
            [
                dict(
                    type="GlobalRotScaleTrans",
                    rot_range=[-3.1415926, 3.1415926],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0.1, 0.1, 0.1],
                )
            ],
            [dict(type="Pack3DDetInputs", keys=["points"], meta_keys=["num_points", "lidar_path", "num_pts_feats"])],
        ],
    ),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        lidar_sources=SOURCES,
        ann_file="info/lidarseg/t4dataset_j6gen2_lidarseg_infos_train.pkl",
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=ignore_index,
        backend_args=backend_args,
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
        lidar_sources=SOURCES,
        ann_file="info/lidarseg/t4dataset_j6gen2_lidarseg_infos_val.pkl",
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=ignore_index,
        test_mode=True,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="SegMetric")
test_evaluator = val_evaluator

vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]

visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")

tta_model = dict(type="Seg3DTTAModel")

lr = 0.01
optim_wrapper = dict(
    type="OptimWrapper", optimizer=dict(type="AdamW", lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6)
)
param_scheduler = [
    dict(
        type="OneCycleLR",
        total_steps=iterations,
        by_epoch=False,
        eta_max=lr,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0,
    )
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=iterations, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=16)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)

default_hooks = dict(checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=-1, save_best="miou"))
