_base_ = ["../model/frnet.py", "../../../../autoware_ml/configs/detection3d/default_runtime.py"]

custom_imports = dict(
    imports=[
        "autoware_ml.segmentation3d.datasets.nuscenes",
        "projects.FRNet.frnet.datasets",
        "projects.FRNet.frnet.datasets.transforms",
        "projects.FRNet.frnet.models",
    ],
    allow_failed_imports=False,
)

# user settings
BATCH_SIZE = 4
NUM_WORKERS = 4
ITERATIONS = 150000
VAL_INTERVAL = 1500

# LiDAR settings (Velodyne HDL-32E)
LIDAR_H = 32
RANGE_W = 1920
FRUSTUM_W = 480  # 1920 / 4
FOV_UP = 10.0
FOV_DOWN = -30.0

dataset_type = "NuScenesSegDataset"
data_root = "data/nuscenes/"
ignore_index = 16
class_names = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]
labels_map = {
    0: 16,
    1: 16,
    2: 6,
    3: 6,
    4: 6,
    5: 16,
    6: 6,
    7: 16,
    8: 16,
    9: 0,
    10: 16,
    11: 16,
    12: 7,
    13: 16,
    14: 1,
    15: 2,
    16: 2,
    17: 3,
    18: 4,
    19: 16,
    20: 16,
    21: 5,
    22: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 12,
    27: 13,
    28: 14,
    29: 16,
    30: 15,
    31: 16,
}
num_classes = 17  # 16 classes + 1 for unknown
metainfo = dict(classes=class_names, seg_label_mapping=labels_map, max_label=31)
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts="samples/LIDAR_TOP", img="", pts_semantic_mask="lidarseg/v1.0-trainval")
backend_args = None

model = dict(
    data_preprocessor=dict(H=LIDAR_H, W=FRUSTUM_W, fov_up=FOV_UP, fov_down=FOV_DOWN, ignore_index=ignore_index),
    backbone=dict(output_shape=(LIDAR_H, FRUSTUM_W)),
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
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=4, backend_args=backend_args),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.uint8",
        backend_args=backend_args,
    ),
    dict(type="PointSegClassMapping"),
    dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
]
train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=4, backend_args=backend_args),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.uint8",
        backend_args=backend_args,
    ),
    dict(type="PointSegClassMapping"),
    dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(
        type="FrustumMix",
        H=LIDAR_H,
        W=FRUSTUM_W,
        fov_up=FOV_UP,
        fov_down=FOV_DOWN,
        num_areas=[3, 4, 5, 6],
        pre_transform=pre_transform,
        prob=1.0,
    ),
    dict(type="InstanceCopy", instance_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pre_transform=pre_transform, prob=1.0),
    dict(type="RangeInterpolation", H=LIDAR_H, W=RANGE_W, fov_up=FOV_UP, fov_down=FOV_DOWN, ignore_index=ignore_index),
    dict(type="Pack3DDetInputs", keys=["points", "pts_semantic_mask"]),
]
test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=4, backend_args=backend_args),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.uint8",
        backend_args=backend_args,
    ),
    dict(type="PointSegClassMapping"),
    dict(type="RangeInterpolation", H=LIDAR_H, W=RANGE_W, fov_up=FOV_UP, fov_down=FOV_DOWN, ignore_index=ignore_index),
    dict(type="Pack3DDetInputs", keys=["points"], meta_keys=["num_points", "lidar_path", "num_pts_feats"]),
]
tta_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=4, backend_args=backend_args),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.uint8",
        backend_args=backend_args,
    ),
    dict(type="PointSegClassMapping"),
    dict(type="RangeInterpolation", H=LIDAR_H, W=RANGE_W, fov_up=FOV_UP, fov_down=FOV_DOWN, ignore_index=ignore_index),
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
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="nuscenes_infos_train.pkl",
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=ignore_index,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="nuscenes_infos_val.pkl",
        data_prefix=data_prefix,
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
        total_steps=ITERATIONS,
        by_epoch=False,
        eta_max=lr,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0,
    )
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=ITERATIONS, val_interval=VAL_INTERVAL)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=16)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)

default_hooks = dict(checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=-1, save_best="miou"))
