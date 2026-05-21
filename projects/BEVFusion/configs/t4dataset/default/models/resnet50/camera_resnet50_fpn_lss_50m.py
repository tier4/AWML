_base_ = [
    "./camera_resnet50_fpn_depthlss_120m.py",
]
num_proposals = 200 

# Image network
model = dict(
    view_transform=dict(
        type="LSSTransformV2",
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60, 0.5],
        downsample=2,
    ),
    bbox_head=dict(
        in_channels=80,
        num_proposals=num_proposals,
        bbox_coder=dict(
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        ),
    ),
)
