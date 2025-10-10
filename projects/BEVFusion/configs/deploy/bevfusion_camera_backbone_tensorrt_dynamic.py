codebase_config = dict(type="mmdet3d", task="VoxelDetection", model_type="end2end")

custom_imports = dict(
    imports=[
        "projects.BEVFusion.deploy",
        "projects.BEVFusion.bevfusion",
        "projects.SparseConvolution",
    ],
    allow_failed_imports=False,
)

image_dims = (384, 576)

backend_config = dict(
    type="tensorrt",
    common_config=dict(max_workspace_size=1 << 32),
    model_inputs=[
        dict(
            input_shapes=dict(
                imgs=dict(
                    min_shape=[1, 3, image_dims[0], image_dims[1]],
                    opt_shape=[6, 3, image_dims[0], image_dims[1]],
                    max_shape=[6, 3, image_dims[0], image_dims[1]],
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
    save_file="image_backbone.onnx",
    input_names=[
        "imgs",
    ],
    output_names=["image_feats"],
    dynamic_axes={
        "imgs": {
            0: "num_imgs",
        },
    },
    input_shape=None,
    verbose=True,
)
