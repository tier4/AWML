codebase_config = dict(type="mmdet3d", task="VoxelDetection", model_type="end2end")

custom_imports = dict(
    imports=[
        "projects.BEVFusion.deploy",
        "projects.BEVFusion.bevfusion",
        "projects.SparseConvolution",
    ],
    allow_failed_imports=False,
)

backend_config = dict(
    type="tensorrt",
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                # points=dict(
                #    min_shape=[5000, 5],
                #    opt_shape=[50000, 5],
                #    max_shape=[200000, 5]),
                voxels=dict(min_shape=[1, 5], opt_shape=[64000, 5], max_shape=[256000, 5]),
                coors=dict(min_shape=[1, 4], opt_shape=[64000, 4], max_shape=[256000, 4]),
                num_points_per_voxel=dict(min_shape=[1], opt_shape=[64000], max_shape=[256000]),
                imgs=dict(min_shape=[1, 3, 256, 704], opt_shape=[6, 3, 256, 704], max_shape=[6, 3, 256, 704]),
                lidar2image=dict(min_shape=[1, 4, 4], opt_shape=[6, 4, 4], max_shape=[6, 4, 4]),
                geom_feats=dict(
                    min_shape=[0 * 118 * 32 * 88, 4],
                    opt_shape=[6 * 118 * 32 * 88 // 2, 4],
                    max_shape=[6 * 118 * 32 * 88, 4],
                ),
                kept=dict(
                    min_shape=[0 * 118 * 32 * 88],
                    opt_shape=[6 * 118 * 32 * 88],
                    max_shape=[6 * 118 * 32 * 88],
                ),
                ranks=dict(
                    min_shape=[0 * 118 * 32 * 88],
                    opt_shape=[6 * 118 * 32 * 88 // 2],
                    max_shape=[6 * 118 * 32 * 88],
                ),
                indices=dict(
                    min_shape=[0 * 118 * 32 * 88],
                    opt_shape=[6 * 118 * 32 * 88 // 2],
                    max_shape=[6 * 118 * 32 * 88],
                ),
            )
        )
    ],
)

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=17,
    save_file="end2end.onnx",
    input_names=[  # 'points',
        "voxels",
        "coors",
        "num_points_per_voxel",
        "imgs",
        "lidar2image",
        "cam2image",
        "camera2lidar",
        "geom_feats",
        "kept",
        "ranks",
        "indices",
    ],
    output_names=["bbox_pred", "score", "label_pred"],
    dynamic_axes={
        # 'points': {
        #    0: 'num_points',
        # },
        "voxels": {
            0: "num_voxels",
        },
        "coors": {
            0: "num_voxels",
        },
        "num_points_per_voxel": {
            0: "num_voxels",
        },
        "imgs": {
            0: "num_imgs",
        },
        "lidar2image": {
            0: "num_imgs",
        },
        "cam2image": {
            0: "num_imgs",
        },
        "camera2lidar": {
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
    },
    input_shape=None,
    verbose=True,
)
