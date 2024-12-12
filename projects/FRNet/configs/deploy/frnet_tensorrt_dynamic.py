custom_imports = dict(
    imports=[
        "projects.FRNet.frnet.models",
    ],
    allow_failed_imports=False,
)

tensorrt_config = dict(
    points=dict(min_shape=[5000, 4], opt_shape=[30000, 4], max_shape=[60000, 4]),
    coors=dict(min_shape=[5000, 3], opt_shape=[30000, 3], max_shape=[60000, 3]),
    voxel_coors=dict(min_shape=[5000, 3], opt_shape=[30000, 3], max_shape=[60000, 3]),
    inverse_map=dict(min_shape=[5000], opt_shape=[30000], max_shape=[60000]),
    seg_logit=dict(min_shape=[5000, 17], opt_shape=[30000, 17], max_shape=[60000, 17]),
)

onnx_config = dict(
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=16,
    do_constant_folding=True,
    input_names=["points", "coors", "voxel_coors", "inverse_map"],
    output_names=["seg_logit"],
    dynamic_axes={
        "points": {0: "num_points"},
        "coors": {0: "num_points"},
        "voxel_coors": {0: "num_unique_coors"},
        "inverse_map": {0: "num_points"},
        "seg_logit": {0: "num_points"},
    },
)
