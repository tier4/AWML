# 1. Tune hyperparams like seq_len, norm_eval, train_range, missing_image_replacement, large_image_sizes, feature_maps, datasets(xx1,x2,base)
_base_ = [
    "../default/vov_flash_480x640_baseline.py",
]

info_directory_path = "info/user/base_model1.0.0/"
data_root = "data/"

batch_size = 1
num_workers = 16

num_epochs = 35
val_interval = 5

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(ann_file=info_directory_path + _base_.info_train_file_name, data_root=data_root),
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(ann_file=info_directory_path + _base_.info_val_file_name, data_root=data_root),
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(ann_file=info_directory_path + _base_.info_test_file_name, data_root=data_root),
)


val_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + _base_.info_val_file_name)
test_evaluator = dict(data_root=data_root, ann_file=data_root + info_directory_path + _base_.info_test_file_name)


train_cfg = dict(
    by_epoch=True, max_epochs=num_epochs, val_interval=val_interval, dynamic_intervals=[(num_epochs - 5, 1)]
)

lr = 1e-4
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)  # bs 8: 2e-4 || bs 16: 4e-4,

# lrg policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(
        type="CosineAnnealingLR",
        by_epoch=True,
        eta_min=lr * 1e-4,
    ),
]

load_from = "/workspace/work_dirs/ckpts/nuscenes_resnet50_320x800_baseline.pth"

auto_scale_lr = dict(base_batch_size=8, enable=True)
