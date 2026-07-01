_base_ = [
    "./default_camera_swin_fpn_depthlss_120m.py",
]

# Image network
model = dict(
	  view_transform=dict(
        type="LSSTransformV2DepthAware",
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60, 0.5],
        downsample=2,
        camera_depth_aware_configs=dict(mlp_drop_out=0.0, downsample=8, num_camera_depth_parameters=27),
    ),
    bbox_head=dict(
        bbox_coder=dict(
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        ),
    ),
)
