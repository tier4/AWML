"""
Exp F — Combined: partial supervision + adaptive EMA (Phase 1, 10 epochs).
Combines the two strongest individual hypotheses:
  - Only the 5 rare classes are supervised (removes common-class noise from loss)
  - Adaptive EMA ensures rare prototypes update aggressively despite few examples

This is the "maximum rare-class focus" configuration. If both B and D help
individually, their combination should amplify the effect. The risk is that
with no common-class supervision, val mIoU on common classes degrades.
The val metric will reveal if this tradeoff is worth it.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs._base_pps import *  # noqa

save_path = "exp/F_partial_adaptive/phase1"
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
        ema_momentum=0.99,
        conf_threshold=0.6,
        proto_loss_weight=0.2,
        af3_loss_weight=1.0,
        ortho_loss_weight=0.1,
        supervised_class_ids=RARE_CLASS_IDS,   # partial supervision
        adaptive_ema=True,                      # adaptive EMA
    ),
)

data = make_data(dataset_type, data_root, info_paths_train, info_paths_val,
                 info_paths_test, class_mapping, num_classes, ignore_index)
