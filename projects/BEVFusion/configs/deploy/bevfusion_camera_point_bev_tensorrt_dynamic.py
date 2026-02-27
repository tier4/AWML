codebase_config = dict(type="mmdet3d", task="VoxelDetection", model_type="end2end")

custom_imports = dict(
    imports=[
        "projects.BEVFusion.deploy",
        "projects.BEVFusion.bevfusion",
        "projects.SparseConvolution",
    ],
    allow_failed_imports=False,
)

depth_bins = 129
# TODO(KokSeang): Read this parameter from a base config
feature_dims = (48, 96)
# image_dims = (640, 576)

backend_config = dict(
    type="tensorrt",
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                geom_feats=dict(
                    min_shape=[0 * depth_bins * feature_dims[0] * feature_dims[1], 4],
                    opt_shape=[6 * depth_bins * feature_dims[0] * feature_dims[1] // 2, 4],
                    max_shape=[6 * depth_bins * feature_dims[0] * feature_dims[1], 4],
                ),
                kept=dict(
                    min_shape=[0 * depth_bins * feature_dims[0] * feature_dims[1]],
                    opt_shape=[6 * depth_bins * feature_dims[0] * feature_dims[1]],
                    max_shape=[6 * depth_bins * feature_dims[0] * feature_dims[1]],
                ),
                ranks=dict(
                    min_shape=[0 * depth_bins * feature_dims[0] * feature_dims[1]],
                    opt_shape=[6 * depth_bins * feature_dims[0] * feature_dims[1] // 2],
                    max_shape=[6 * depth_bins * feature_dims[0] * feature_dims[1]],
                ),
                indices=dict(
                    min_shape=[0 * depth_bins * feature_dims[0] * feature_dims[1]],
                    opt_shape=[6 * depth_bins * feature_dims[0] * feature_dims[1] // 2],
                    max_shape=[6 * depth_bins * feature_dims[0] * feature_dims[1]],
                ),
                image_feats=dict(
                    min_shape=[1, 256, feature_dims[0], feature_dims[1]],
                    opt_shape=[6, 256, feature_dims[0], feature_dims[1]],
                    max_shape=[6, 256, feature_dims[0], feature_dims[1]],
                ),
                # TODO(TIERIV): Optimize. Now, using points will increase latency significantly,
                # we always include intensity as well even we dont need them
                points=dict(min_shape=[5000, 5], opt_shape=[50000, 5], max_shape=[200000, 5]),
                lidar2image=dict(min_shape=[1, 4, 4], opt_shape=[6, 4, 4], max_shape=[6, 4, 4]),
                img_aug_matrix=dict(min_shape=[1, 4, 4], opt_shape=[6, 4, 4], max_shape=[6, 4, 4]),
            )
        )
    ],
)

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=17,
    save_file="camera_point_bev.onnx",
    input_names=["geom_feats", "kept", "ranks", "indices", "image_feats", "points", "lidar2image", "img_aug_matrix"],
    output_names=["bbox_pred", "score", "label_pred"],
    dynamic_axes={
        "geom_feats": {
            0: "num_kept",
        },
        "kept": {
            0: "num_geom_feats",
        },
        "ranks": {
            0: "num_kept",
        },
        "indices": {
            0: "num_kept",
        },
        "image_feats": {
            0: "num_imgs",
        },
        "points": {
            0: "num_points",
        },
        "lidar2image": {
            0: "num_imgs",
        },
        "img_aug_matrix": {
            0: "num_imgs",
        },
    },
    input_shape=None,
    verbose=True,
)
