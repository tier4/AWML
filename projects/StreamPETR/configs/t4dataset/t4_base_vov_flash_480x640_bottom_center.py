# 1. Tune hyperparams like seq_len, norm_eval, train_range, missing_image_replacement, large_image_sizes, feature_maps, datasets(xx1,x2,base)
_base_ = [
    "./t4_base_vov_flash_480x640_baseline.py",
]

model = dict(pts_bbox_head=dict(use_bottom_center=True))
