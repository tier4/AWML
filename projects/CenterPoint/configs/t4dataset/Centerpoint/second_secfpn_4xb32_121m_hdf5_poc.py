# CenterPoint config for the HDF5 dataloader throughput PoC (4 GPUs, db_j6gen2_v99).
#
# Identical to second_secfpn_8xb16_121m_base_amp.py except: 4 GPUs, HDF5-capable
# point loaders, no validation/checkpoints/MLflow, and raw timing dumps.
# The point-cloud storage variant is selected with an environment variable so
# one config serves every run:
#   HDF5_POC_VARIANT = bin | none | lz4_l9 | lz4hc_l5 | zstd_l1 | zstd_l5   (default: bin)
#   HDF5_POC_BATCH   = per-GPU batch size                                 (default: 32)
#   HDF5_POC_EPOCHS  = number of epochs                                   (default: 30)
#   HDF5_POC_WORKERS = dataloader workers per GPU                          (default: 32)
#   HDF5_POC_DETERMINISTIC = 1|0  keep the base config's deterministic mode  (default: 1)
# NOTE: `_base_` must come before any import, otherwise mmengine parses this file in lazy-import mode.
_base_ = ["./second_secfpn_8xb16_121m_base_amp.py"]

import copy  # noqa: E402
import os  # noqa: E402

variant = os.environ.get("HDF5_POC_VARIANT", "bin")
train_gpu_size = 4
train_batch_size = int(os.environ.get("HDF5_POC_BATCH", 32))
max_epochs = int(os.environ.get("HDF5_POC_EPOCHS", 30))
num_workers = int(os.environ.get("HDF5_POC_WORKERS", 32))
randomness = dict(seed=0, diff_rank_seed=False, deterministic=os.environ.get("HDF5_POC_DETERMINISTIC", "1") == "1")

data_root = "data/t4dataset/"
info_directory_path = "info/hdf5_poc/"
_info_stem = "t4dataset_hdf5_poc" + ("" if variant == "bin" else f"_{variant}")
info_train_file_name = _info_stem + "_infos_train.pkl"
info_val_file_name = _info_stem + "_infos_val.pkl"
info_test_file_name = _info_stem + "_infos_test.pkl"

# Runs with a non-default worker count get their own work_dir so they don't overwrite the main matrix.
_run_name = variant if num_workers == 32 else f"{variant}_w{num_workers}"
experiment_name = f"second_secfpn_4xb{train_batch_size}_121m_hdf5_poc_{_run_name}"
work_dir = "work_dirs/hdf5_poc/" + _run_name


def _hdf5_pipeline(pipeline):
    """Swap the stock point loaders for the HDF5-capable subclasses (same args)."""
    pipeline = copy.deepcopy(pipeline)
    for transform in pipeline:
        if transform["type"] == "LoadPointsFromFile":
            transform["type"] = "LoadPointsFromFileHDF5"
        elif transform["type"] == "LoadPointsFromMultiSweeps":
            transform["type"] = "LoadPointsFromMultiSweepsHDF5"
    return pipeline


train_pipeline = _hdf5_pipeline(_base_.train_pipeline)
test_pipeline = _hdf5_pipeline(_base_.test_pipeline)
eval_pipeline = _hdf5_pipeline(_base_.eval_pipeline)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    dataset=dict(pipeline=train_pipeline, ann_file=info_directory_path + info_train_file_name),
)
test_dataloader = dict(
    num_workers=num_workers,
    dataset=dict(pipeline=test_pipeline, ann_file=info_directory_path + info_test_file_name),
)
test_evaluator = dict(ann_file=data_root + info_directory_path + info_test_file_name)

# Throughput only: no validation loop, so no val dataloader/evaluator either.
val_dataloader = None
val_cfg = None
val_evaluator = None
train_cfg = dict(_delete_=True, by_epoch=True, max_epochs=max_epochs, val_interval=max_epochs + 1)

# Same schedule shape as the base config, re-derived from the shorter max_epochs
# (the second cosine phase is dropped when no epochs remain for it, e.g. 1-epoch smoke runs).
lr = _base_.lr
t_max = min(max(1, round(max_epochs * 0.3)), max_epochs)
param_scheduler = [
    dict(type="CosineAnnealingLR", T_max=t_max, eta_min=lr * 10, begin=0, end=t_max, by_epoch=True, convert_to_iter_based=True),
    dict(
        type="CosineAnnealingMomentum",
        T_max=t_max,
        eta_min=0.85 / 0.95,
        begin=0,
        end=t_max,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
if max_epochs > t_max:
    param_scheduler += [
        dict(
            type="CosineAnnealingLR",
            T_max=max_epochs - t_max,
            eta_min=lr * 1e-4,
            begin=t_max,
            end=max_epochs,
            by_epoch=True,
            convert_to_iter_based=True,
        ),
        dict(
            type="CosineAnnealingMomentum",
            T_max=max_epochs - t_max,
            eta_min=1,
            begin=t_max,
            end=max_epochs,
            by_epoch=True,
            convert_to_iter_based=True,
        ),
    ]

auto_scale_lr = dict(enable=False, base_batch_size=train_gpu_size * train_batch_size)
sync_bn = "torch"

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=5),
    checkpoint=dict(_delete_=True, type="CheckpointHook", interval=-1, save_last=False),
)
custom_hooks = _base_.custom_hooks + [dict(type="IterTimingDumpHook")]
