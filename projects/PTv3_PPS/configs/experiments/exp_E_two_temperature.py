"""
Exp E — Novel design: two-temperature system (Phase 1, 10 epochs).
Rare class columns in the cosine logit matrix use rare_temperature=0.04
(sharper boundary), while common classes use temperature=0.10 (softer).

Intuition: rare-class decision boundaries need to be sharper to prevent
common-class predictions from bleeding into rare-class regions. Common
classes already have enough training signal and benefit from softer
boundaries (less overfit to training distribution).

temperature mapping:
  common classes (default): 0.10  → logit scale = 10×
  rare classes [13,14,18,21,22]:  0.04  → logit scale = 25×
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs._base_pps import *  # noqa

save_path = "exp/E_two_temperature/phase1"
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
        temperature=0.10,               # common class temperature (softer)
        rare_temperature=0.04,          # rare class temperature (sharper)
        rare_class_ids=RARE_CLASS_IDS,  # [13, 14, 18, 21, 22]
        ema_momentum=0.99,
        conf_threshold=0.6,
        proto_loss_weight=0.2,
        af3_loss_weight=1.0,
        ortho_loss_weight=0.1,
    ),
)

data = make_data(dataset_type, data_root, info_paths_train, info_paths_val,
                 info_paths_test, class_mapping, num_classes, ignore_index)
