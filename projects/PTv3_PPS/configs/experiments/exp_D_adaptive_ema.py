"""
Exp D — Novel design: adaptive EMA momentum (Phase 1, 10 epochs).
adaptive_ema=True: per-class momentum scales with running class frequency.
  momentum_c = base_ema + (1 - base_ema) * (freq_c / max_freq)
  → rare classes (low freq): momentum ≈ 0.99 → larger per-step prototype update
  → common classes (high freq): momentum ≈ 1.0 → very slow update (stable)

Hypothesis: rare classes see far fewer labeled points per epoch; adaptive EMA
            compensates by making each rare-class update more impactful,
            while keeping common-class prototypes stable.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs._base_pps import *  # noqa

save_path = "exp/D_adaptive_ema/phase1"
epoch = 10
eval_epoch = 10
batch_size_val = None

optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.005)
scheduler = dict(type="OneCycleLR", max_lr=[0.005], pct_start=0.1,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
param_dicts = []

model = dict(
    type="PPSSegmentor",
    num_classes=num_classes,
    backbone_out_channels=64,
    freeze_backbone=True,
    weight_path=_WEIGHT_PATH,
    backbone=_BACKBONE,
    head=dict(
        ignore_index=ignore_index,
        temperature=0.07,
        ema_momentum=0.99,          # base momentum (rare classes use this)
        conf_threshold=0.6,
        proto_loss_weight=0.2,
        af3_loss_weight=1.0,
        ortho_loss_weight=0.1,
        adaptive_ema=True,          # ← novel design
    ),
)

data = make_data(dataset_type, data_root, info_paths_train, info_paths_val,
                 info_paths_test, class_mapping, num_classes, ignore_index)
