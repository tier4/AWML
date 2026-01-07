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

# Derived artifact directories
_WORK_DIR = str(export["work_dir"]).rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Unified Component Configuration (Single Source of Truth)
#
# Each component defines:
#   - name: Component identifier used in export
#   - onnx_file: Output ONNX filename
#   - engine_file: Output TensorRT engine filename
#   - io: Input/output specification for ONNX export
#   - tensorrt_profile: TensorRT optimization profile (min/opt/max shapes)
# ============================================================================
components = dict(
    voxel_encoder=dict(
        name="pts_voxel_encoder",
        onnx_file="pts_voxel_encoder.onnx",
        engine_file="pts_voxel_encoder.engine",
        io=dict(
            inputs=[
                dict(name="input_features", dtype="float32"),
            ],
            outputs=[
                dict(name="pillar_features", dtype="float32"),
            ],
            dynamic_axes={
                "input_features": {0: "num_voxels", 1: "num_max_points"},
                "pillar_features": {0: "num_voxels"},
            },
        ),
        tensorrt_profile=dict(
            input_features=dict(
                min_shape=[1000, 32, 11],
                opt_shape=[20000, 32, 11],
                max_shape=[64000, 32, 11],
            ),
        ),
    ),
    backbone_head=dict(
        name="pts_backbone_neck_head",
        onnx_file="pts_backbone_neck_head.onnx",
        engine_file="pts_backbone_neck_head.engine",
        io=dict(
            inputs=[
                dict(name="spatial_features", dtype="float32"),
            ],
            outputs=[
                dict(name="heatmap", dtype="float32"),
                dict(name="reg", dtype="float32"),
                dict(name="height", dtype="float32"),
                dict(name="dim", dtype="float32"),
                dict(name="rot", dtype="float32"),
                dict(name="vel", dtype="float32"),
            ],
            dynamic_axes={
                "spatial_features": {0: "batch_size", 2: "height", 3: "width"},
                "heatmap": {0: "batch_size", 2: "height", 3: "width"},
                "reg": {0: "batch_size", 2: "height", 3: "width"},
                "height": {0: "batch_size", 2: "height", 3: "width"},
                "dim": {0: "batch_size", 2: "height", 3: "width"},
                "rot": {0: "batch_size", 2: "height", 3: "width"},
                "vel": {0: "batch_size", 2: "height", 3: "width"},
            },
        ),
        tensorrt_profile=dict(
            spatial_features=dict(
                min_shape=[1, 32, 760, 760],
                opt_shape=[1, 32, 760, 760],
                max_shape=[1, 32, 760, 760],
            ),
        ),
    ),
)

# ============================================================================
# Runtime I/O settings
# ============================================================================
runtime_io = dict(
    info_file="data/t4dataset/info/t4dataset_j6gen2_infos_val.pkl",
    sample_idx=1,
)

# ============================================================================
# ONNX Export Settings (shared across all components)
# ============================================================================
onnx_config = dict(
    opset_version=16,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

# ============================================================================
# TensorRT Build Settings (shared across all components)
# ============================================================================
tensorrt_config = dict(
    precision_policy="auto",
    max_workspace_size=2 << 30,
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
            model_dir=_ONNX_DIR,
        ),
        tensorrt=dict(
            enabled=True,
            device=devices["cuda"],
            engine_dir=_TENSORRT_DIR,
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
