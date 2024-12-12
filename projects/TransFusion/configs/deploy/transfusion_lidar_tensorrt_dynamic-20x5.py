codebase_config = dict(type="mmdet3d", task="VoxelDetection", model_type="end2end")

backend_config = dict(
    type="tensorrt",
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                voxels=dict(
                    min_shape=[5000, 20, 5],
                    opt_shape=[30000, 20, 5],
                    max_shape=[60000, 20, 5],
                ),
                num_points=dict(
                    min_shape=[5000],
                    opt_shape=[30000],
                    max_shape=[60000],
                ),
                coors=dict(
                    min_shape=[5000, 4],
                    opt_shape=[30000, 4],
                    max_shape=[60000, 4],
                ),
            )
        )
    ],
)

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="end2end.onnx",
    input_names=["voxels", "num_points", "coors"],
    output_names=["cls_score0", "bbox_pred0", "dir_cls_pred0"],
    dynamic_axes={
        "voxels": {
            0: "voxels_num",
        },
        "num_points": {
            0: "voxels_num",
        },
        "coors": {
            0: "voxels_num",
        },
    },
    input_shape=None,
)
