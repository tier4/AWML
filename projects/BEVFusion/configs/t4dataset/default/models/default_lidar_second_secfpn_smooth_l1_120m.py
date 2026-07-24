_base_ = [
    "./default_lidar_second_secfpn_120m.py"
]

model = dict(
    loss_bbox=dict(type="mmdet.SmoothL1Loss", reduction="mean", loss_weight=0.25)
)
