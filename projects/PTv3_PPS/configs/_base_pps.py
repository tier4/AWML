"""
Shared base for all PPS experiments. Import via:
    from configs._base_pps import *
or copy the relevant sections into each experiment config.

Varies per experiment: save_path, model.head, (optionally model.freeze_backbone)
"""

# ── Runtime ─────────────────────────────────────────────────────────────────
default_scope = "mmdet3d"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=-1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="Det3DVisualizationHook"),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl", timeout=7200),
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False

num_worker = 32
batch_size = 8
clip_grad = 10.0
sync_bn = False
enable_amp = False
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
mix_prob = 0.8

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=True)

custom_imports = dict(
    imports=[
        "autoware_ml.segmentation3d.datasets.t4dataset",
        "autoware_ml.segmentation3d.datasets.transforms",
    ]
)

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset_type = "T4Dataset"
data_root = "/mnt/t4dataset_ro"
ignore_index = -1
_info_base = "/mnt/t4dataset_ro/info/lidarseg"
info_paths_train = [f"{_info_base}/t4dataset_j6gen2_lidarseg_infos_train.pkl"]
info_paths_val = [f"{_info_base}/t4dataset_j6gen2_lidarseg_infos_val.pkl"]
info_paths_test = [f"{_info_base}/t4dataset_j6gen2_lidarseg_infos_test.pkl"]
class_mapping = dict(
    drivable_surface=0, other_flat_surface=1, sidewalk=2, manmade=3, vegetation=4,
    car=5, bus=6, emergency_vehicle=7, train=8, truck=9, tractor_unit=10,
    semi_trailer=11, construction_vehicle=12, forklift=13, kart=14, motorcycle=15,
    bicycle=16, pedestrian=17, personal_mobility=18, animal=19,
    pushable_pullable=20, traffic_cone=21, stroller=22, debris=23,
    other_stuff=24, noise=25, ghost_point=25, out_of_sync=ignore_index,
    unpainted=ignore_index,
)
num_classes = 26
grid_size = 0.1
point_cloud_range = [-102.4, -102.4, -2.8, 102.4, 102.4, 10.0]
distance_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 102.4)]
metric_options = dict(distance_ranges=distance_ranges)

# Rare class IDs used across experiments:
#   forklift=13, kart=14, personal_mobility=18, traffic_cone=21, stroller=22
RARE_CLASS_IDS = [13, 14, 18, 21, 22]

# ── Backbone (shared across all experiments) ─────────────────────────────────
_BACKBONE = dict(
    type="PT-v3m1",
    in_channels=4,
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4, qkv_bias=True, qk_scale=None,
    attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
    shuffle_orders=True, pre_norm=True, enable_rpe=False, enable_flash=True,
    upcast_attention=False, upcast_softmax=False, cls_mode=False,
    pdnorm_bn=False, pdnorm_ln=False, pdnorm_decouple=True,
    pdnorm_adaptive=False, pdnorm_affine=True,
    pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
)
_WEIGHT_PATH = "/home/linick/Downloads/ptv3_train/reproduce_only_gt/model_best.pth"

# ── Data transforms (shared) ─────────────────────────────────────────────────
_TRAIN_TRANSFORM = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="PointClip", point_cloud_range=point_cloud_range),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
         keys=("coord", "strength", "segment"), return_grid_coord=True),
    dict(type="SphereCrop", point_max=128000, mode="random"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"),
         feat_keys=("coord", "strength")),
]
_VAL_TRANSFORM = [
    dict(type="PointClip", point_cloud_range=point_cloud_range),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
         keys=("coord", "strength", "segment"), return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"),
         feat_keys=("coord", "strength")),
]
_TEST_TRANSFORM = [
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
         keys=("coord", "strength", "segment"), return_inverse=True),
]
_TEST_CFG = dict(
    voxelize=dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
                  mode="test", return_grid_coord=True, keys=("coord", "strength")),
    crop=None,
    post_transform=[
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "index"),
             feat_keys=("coord", "strength")),
    ],
    aug_transform=[
        [dict(type="RandomScale", scale=[0.9, 0.9])],
        [dict(type="RandomScale", scale=[0.95, 0.95])],
        [dict(type="RandomScale", scale=[1, 1])],
        [dict(type="RandomScale", scale=[1.05, 1.05])],
        [dict(type="RandomScale", scale=[1.1, 1.1])],
        [dict(type="RandomScale", scale=[0.9, 0.9]), dict(type="RandomFlip", p=1)],
        [dict(type="RandomScale", scale=[0.95, 0.95]), dict(type="RandomFlip", p=1)],
        [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
        [dict(type="RandomScale", scale=[1.05, 1.05]), dict(type="RandomFlip", p=1)],
        [dict(type="RandomScale", scale=[1.1, 1.1]), dict(type="RandomFlip", p=1)],
    ],
)


def make_data(dataset_type, data_root, info_paths_train, info_paths_val,
              info_paths_test, class_mapping, num_classes, ignore_index):
    return dict(
        num_classes=num_classes,
        ignore_index=ignore_index,
        train=dict(type=dataset_type, split="train", data_root=data_root,
                   info_paths=info_paths_train, transform=_TRAIN_TRANSFORM,
                   test_mode=False, ignore_index=ignore_index,
                   class_mapping=class_mapping, loop=1),
        val=dict(type=dataset_type, split="val", data_root=data_root,
                 info_paths=info_paths_val, transform=_VAL_TRANSFORM,
                 test_mode=False, ignore_index=ignore_index,
                 class_mapping=class_mapping),
        test=dict(type=dataset_type, split="val", data_root=data_root,
                  info_paths=info_paths_test, transform=_TEST_TRANSFORM,
                  test_mode=True, test_cfg=_TEST_CFG,
                  ignore_index=ignore_index, class_mapping=class_mapping),
    )
