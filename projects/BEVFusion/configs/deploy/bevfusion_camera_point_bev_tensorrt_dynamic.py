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
feature_dims = (60, 80)
# image_dims = (640, 576)

backend_config = dict(
    type="tensorrt",
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                # TODO(TIERIV): Optimize. Now, using points will increase latency significantly
                points=dict(min_shape=[5000, 4], opt_shape=[50000, 4], max_shape=[200000, 4]),
                lidar2image=dict(min_shape=[1, 4, 4], opt_shape=[6, 4, 4], max_shape=[6, 4, 4]),
                img_aug_matrix=dict(min_shape=[1, 4, 4], opt_shape=[6, 4, 4], max_shape=[6, 4, 4]),
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
                    min_shape=[0, 256, feature_dims[0], feature_dims[1]],
                    opt_shape=[6, 256, feature_dims[0], feature_dims[1]],
                    max_shape=[6, 256, feature_dims[0], feature_dims[1]],
                ),
            )
        )
    ],
)


input_names = ["points", "lidar2image", "img_aug_matrix", "geom_feats", "kept", "ranks", "indices", "image_feats"]

dynamic_axes = (
    {
        "points": {
            0: "num_points",
        },
        "lidar2image": {
            0: "num_imgs",
        },
        "img_aug_matrix": {
            0: "num_imgs",
        },
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
    },
)


onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=17,
    save_file="camera_point_bev_network.onnx",
    input_names=input_names,
    output_names=["bbox_pred", "score", "label_pred"],
    dynamic_axes=dynamic_axes,
    input_shape=None,
    verbose=True,
)
