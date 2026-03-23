_base_ = ["../../t4dataset/frnet_1xb8_t4dataset-ot128-seg.py"]

custom_imports = dict(
    imports=[
        "projects.FRNet.frnet.models",
    ],
    allow_failed_imports=False,
)

num_classes = _base_.num_classes
tensorrt_config = dict(
    points=dict(min_shape=[5000, 4], opt_shape=[60000, 4], max_shape=[160000, 4]),
    coors=dict(min_shape=[5000, 3], opt_shape=[60000, 3], max_shape=[160000, 3]),
    voxel_coors=dict(min_shape=[3000, 3], opt_shape=[30000, 3], max_shape=[60000, 3]),
    inverse_map=dict(min_shape=[5000], opt_shape=[60000], max_shape=[160000]),
)

onnx_config = dict(
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=16,
    do_constant_folding=True,
    input_names=["points", "coors", "voxel_coors", "inverse_map"],
    output_names=["pred_probs"],
    dynamic_axes={
        "points": {0: "num_points"},
        "coors": {0: "num_points"},
        "voxel_coors": {0: "num_unique_coors"},
        "inverse_map": {0: "num_points"},
        "pred_probs": {0: "num_points"},
    },
)
