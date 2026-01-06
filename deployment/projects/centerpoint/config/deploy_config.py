"""
CenterPoint Deployment Configuration
"""

# ============================================================================
# Task type for pipeline building
# Options: 'detection2d', 'detection3d', 'classification', 'segmentation'
# ============================================================================
task_type = "detection3d"

# ============================================================================
# Checkpoint Path - Single source of truth for PyTorch model
# ============================================================================
checkpoint_path = "work_dirs/centerpoint/best_checkpoint.pth"

# ============================================================================
# Device settings (shared by export, evaluation, verification)
# ============================================================================
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

# ============================================================================
# Export Configuration
# ============================================================================
export = dict(
    mode="both",
    work_dir="work_dirs/centerpoint_deployment",
    onnx_path=None,
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    info_file="data/t4dataset/info/t4dataset_j6gen2_infos_val.pkl",
    sample_idx=1,
)

# ============================================================================
# Model Input/Output Configuration
# ============================================================================
model_io = dict(
    input_name="voxels",
    input_shape=(32, 4),
    input_dtype="float32",
    additional_inputs=[
        dict(name="num_points", shape=(-1,), dtype="int32"),
        dict(name="coors", shape=(-1, 4), dtype="int32"),
    ],
    head_output_names=("heatmap", "reg", "height", "dim", "rot", "vel"),
    batch_size=None,
    dynamic_axes={
        "voxels": {0: "num_voxels"},
        "num_points": {0: "num_voxels"},
        "coors": {0: "num_voxels"},
    },
)

# ============================================================================
# ONNX Export Configuration
# ============================================================================
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
    multi_file=True,
    components=dict(
        voxel_encoder=dict(
            name="pts_voxel_encoder",
            onnx_file="pts_voxel_encoder.onnx",
            engine_file="pts_voxel_encoder.engine",
        ),
        backbone_head=dict(
            name="pts_backbone_neck_head",
            onnx_file="pts_backbone_neck_head.onnx",
            engine_file="pts_backbone_neck_head.engine",
        ),
    ),
)

# ============================================================================
# Backend Configuration (mainly for TensorRT)
# ============================================================================
backend_config = dict(
    common_config=dict(
        precision_policy="auto",
        max_workspace_size=2 << 30,
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input_features=dict(
                    min_shape=[1000, 32, 11],
                    opt_shape=[20000, 32, 11],
                    max_shape=[64000, 32, 11],
                ),
                spatial_features=dict(
                    min_shape=[1, 32, 760, 760],
                    opt_shape=[1, 32, 760, 760],
                    max_shape=[1, 32, 760, 760],
                ),
            )
        )
    ],
)

# ============================================================================
# Evaluation Configuration
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=1,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=devices["cuda"],
        ),
        onnx=dict(
            enabled=True,
            device=devices["cuda"],
            model_dir="work_dirs/centerpoint_deployment/onnx/",
        ),
        tensorrt=dict(
            enabled=True,
            device=devices["cuda"],
            engine_dir="work_dirs/centerpoint_deployment/tensorrt/",
        ),
    ),
)

# ============================================================================
# Verification Configuration
# ============================================================================
verification = dict(
    enabled=False,
    tolerance=1e-1,
    num_verify_samples=1,
    devices=devices,
    scenarios=dict(
        both=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        onnx=[
            dict(ref_backend="pytorch", ref_device="cpu", test_backend="onnx", test_device="cpu"),
        ],
        trt=[
            dict(ref_backend="onnx", ref_device="cuda", test_backend="tensorrt", test_device="cuda"),
        ],
        none=[],
    ),
)
