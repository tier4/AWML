codebase_config = dict(type="mmpretrain", task="Classification", model_type="end2end")

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="traffic_light_classifier_mobilenetv2_batch_1.onnx",
    input_shape=[224, 224],
    input_names=["input"],
    output_names=["output"],
)
backend_config = dict(
    type="tensorrt",
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 224, 224],
                    opt_shape=[1, 3, 224, 224],
                    max_shape=[1, 3, 224, 224],
                )
            )
        )
    ],
)
