# 1. Tune hyperparams like seq_len, norm_eval, train_range, missing_image_replacement, large_image_sizes, feature_maps, datasets(xx1,x2,base)
_base_ = [
    "../default/vov_flash_480x640_baseline.py",
]

# The base config uses HTTPS `load_from` (VoV-99 init). That download can take a long time
# and shows 0% GPU until it finishes. Prefetch once, then point `load_from` at the file:
#   mkdir -p pretrained && wget -c -O pretrained/nuscenes_vov99_baseline_320x800.pth \
#     'https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/nuscenes/v1.0/nuscenes_vov99_baseline_320x800.pth'
# load_from = "work_dirs/t4_base_vov_flash_480x640_bev_2_7_j6gen2/epoch_10.pth"
# load_from = "pretrained/best_NuScenesmetric_T4Metric_mAP_epoch_34.pth"
load_from = "work_dirs/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_base_partialignore/epoch_32.pth"

# info_directory_path = "info/username/"
# data_root = "data/t4dataset/"
info_directory_path = "info/kokseang_2_8/"
data_root = "data/"

batch_size = 8
num_workers = 32

num_epochs = 35
val_interval = 5

info_train_file_name="t4dataset_j6gen2_base_infos_train.pkl"
info_val_file_name="t4dataset_j6gen2_base_infos_val.pkl"
info_test_file_name="t4dataset_j6gen2_base_infos_test.pkl"

# `_base_` pulls multi-split `dataset_test_groups` from autoware_ml t4dataset/base.py.
# Without `_delete_=True`, MMEngine merges dicts and keeps j6gen2/base/... keys → missing pkls.
# `tools/detection3d/test.py` loops each group under `info_directory_path`.
dataset_test_groups = dict(
    _delete_=True,
    # base=(info_test_file_name, True),
    # base=("t4dataset_base_infos_test.pkl", True),
    j6gen2=("t4dataset_j6gen2_infos_test.pkl", True),
    jpntaxi_gen2=("t4dataset_jpntaxi_gen2_infos_test.pkl", False),
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    sampler=dict(type="GroupStreamingSampler", shuffle=True, batch_size=batch_size, trim_sequences=True),
    dataset=dict(
        ann_file=info_directory_path + info_train_file_name,
        data_root=data_root,
        # NAS: StreamPETRDataset.filter_data() defaults to os.path.exists() per camera × every frame at init.
        # That dominates startup (num_workers cannot help). Skip when ann paths are trusted.
        # check_img_paths=False,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(
        ann_file=info_directory_path + info_val_file_name,
        data_root=data_root,
        # check_img_paths=False,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(
        ann_file=info_directory_path + info_test_file_name,
        data_root=data_root,
        # check_img_paths=False,
    ),
)


val_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + info_val_file_name)
test_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + info_test_file_name)


train_cfg = dict(
    by_epoch=True, max_epochs=num_epochs, val_interval=val_interval, dynamic_intervals=[(num_epochs - 5, 1)]
)

lr = 5e-6
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)

# optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1),}))
optim_wrapper = dict(
    type="NoCacheAmpOptimWrapper",
    dtype="bfloat16",
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    loss_scale="dynamic",
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

# lrg policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        eta_min=lr * 1e-4,
    ),
]

auto_scale_lr = dict(base_batch_size=8, enable=False)

train_pipeline = []
for transform in _base_.train_pipeline:
    train_pipeline.append(transform)
    if transform.get("type") == "LoadAnnotations3D":
        train_pipeline.append(dict(type="Filter3DBoxesinBlindSpot", visibility=0.1))

train_dataloader["dataset"]["pipeline"] = train_pipeline
