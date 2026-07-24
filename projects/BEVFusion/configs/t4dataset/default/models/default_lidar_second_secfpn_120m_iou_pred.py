_base_ = ["./default_lidar_second_secfpn_120m.py"]

model = dict(
    bbox_head=dict(
        loss_iou=dict(_delete_=True, type="CustomBCEWithLogitsLoss", reduction="mean", weight=1.0),
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2], iou=[1, 2]),
    )
)
