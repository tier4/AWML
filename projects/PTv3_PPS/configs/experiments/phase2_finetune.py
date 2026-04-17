"""
Phase 2 — Full fine-tuning (40 epochs), shared across all experiments.

Usage:
    python train.py --config-file configs/experiments/phase2_finetune.py \
        --options resume=True \
                  load_from=exp/A_full_sup/phase1/model/model_last.pth \
                  save_path=exp/A_full_sup/phase2 \
                  model.head.supervised_class_ids=None  # override if needed

The model.head config is reconstructed from the checkpoint — the only
fields that matter here are freeze_backbone=False and the optimizer/scheduler.
Override model.head.* via --options if the experiment needs specific settings
carried into phase 2 (e.g. supervised_class_ids for Exp B/F).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs._base_pps import *  # noqa

save_path = "exp/phase2"   # override via --options save_path=exp/X/phase2
epoch = 40
eval_epoch = 5
batch_size_val = None

# lr scaled ×2 for 2-GPU training (effective batch_size = 16).
# If running on 1 GPU, halve these: lr=0.0005, max_lr=[0.0005, 0.00005].
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

model = dict(
    type="PPSSegmentor",
    num_classes=num_classes,
    backbone_out_channels=64,
    freeze_backbone=False,      # phase 2: full fine-tune
    weight_path=None,           # loaded from checkpoint via resume/load_from
    backbone=_BACKBONE,
    head=dict(
        ignore_index=ignore_index,
        temperature=0.07,
        ema_momentum=0.99,
        conf_threshold=0.6,
        proto_loss_weight=0.2,
        af3_loss_weight=1.0,
        ortho_loss_weight=0.1,
        # Override via --options if experiment needs it:
        # supervised_class_ids=[13, 14, 18, 21, 22]
        # adaptive_ema=True
        # rare_class_ids=[13, 14, 18, 21, 22]
        # rare_temperature=0.04
    ),
)

data = make_data(dataset_type, data_root, info_paths_train, info_paths_val,
                 info_paths_test, class_mapping, num_classes, ignore_index)
