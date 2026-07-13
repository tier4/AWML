_base_ = [
    "./default_lidar_second_secfpn_120m.py",
]

model = dict(
    bbox_head=dict(
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2], iou=[1, 2]),
        loss_iou=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=1.0),
    ),
)
