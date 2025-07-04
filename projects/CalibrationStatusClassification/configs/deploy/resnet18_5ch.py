codebase_config = dict(type="mmpretrain", task="Classification", model_type="end2end")

backend_config = dict(
    type="tensorrt",
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 5, 1860, 2880],
                    opt_shape=[1, 5, 1860, 2880],
                    max_shape=[1, 5, 1860, 2880],
                ),
            )
        )
    ],
)

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=16,
    do_constant_folding=True,
    save_file="end2end.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    input_shape=None,
)
