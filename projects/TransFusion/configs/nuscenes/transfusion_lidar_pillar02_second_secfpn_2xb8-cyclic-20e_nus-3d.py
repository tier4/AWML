_base_ = [
    "./transfusion_lidar_pillar02_second_secfpn_1xb8-cyclic-20e_nus-3d.py"
]
custom_imports = dict(
    imports=["projects.TransFusion.transfusion"], allow_failed_imports=False)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (1 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
