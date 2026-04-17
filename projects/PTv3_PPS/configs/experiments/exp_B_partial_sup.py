"""
Exp B — Partial supervision: only rare 5 classes supervised (Phase 1, 10 epochs).
supervised_class_ids = [13 forklift, 14 kart, 18 personal_mobility,
                        21 traffic_cone, 22 stroller]
All other GT labels are remapped to ignore_index during training.
Hypothesis: forcing the model to focus exclusively on rare classes
            prevents them from being drowned out by majority classes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from configs._base_pps import *  # noqa

save_path = "exp/B_partial_sup/phase1"
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
        supervised_class_ids=RARE_CLASS_IDS,   # [13, 14, 18, 21, 22]
    ),
)

data = make_data(dataset_type, data_root, info_paths_train, info_paths_val,
                 info_paths_test, class_mapping, num_classes, ignore_index)
