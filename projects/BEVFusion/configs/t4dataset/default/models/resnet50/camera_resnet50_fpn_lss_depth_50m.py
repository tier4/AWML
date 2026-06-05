_base_ = [
    "./camera_resnet50_fpn_depthlss_120m.py",
]
num_proposals = 200
out_size_factor = 4

# Image network
model = dict(
    depth_gt_downsample=8,
    loss_depth_weight=1.0,
    view_transform=dict(
        # type="LSSTransformV2",
        type="LSSTransformV2DepthAware",
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60, 0.5],
        downsample=1,
        camera_depth_aware_configs=dict(mlp_drop_out=0.0, downsample=8, num_camera_depth_parameters=27),
    ),
    bbox_head=dict(
        num_proposals=num_proposals,
        bbox_coder=dict(
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        ),
        train_cfg=dict(out_size_factor=4),
        test_cfg=dict(out_size_factor=4),
        bbox_coder=dict(out_size_factor=4),
    ),
)
