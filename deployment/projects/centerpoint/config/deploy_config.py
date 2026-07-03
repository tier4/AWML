"""
CenterPoint Deployment Configuration

Layout (single file, grouped by concern):
  1. SHARED VALUES  - single source of truth reused across sections (paths, devices, shapes).
  2. EXPORT         - export mode, ONNX/TensorRT build settings, component definitions.
  3. EVALUATION     - per-backend evaluation settings.
  4. VERIFICATION   - cross-backend numerical verification scenarios.

Only the top-level names `checkpoint_path`, `deploy_log_path`, `devices`, `export`,
`components`, `onnx_config`, `tensorrt_config`, `evaluation`, `verification` are read by
`BaseDeploymentConfig`. Names prefixed with `_` are local helpers (single-source literals)
and are intentionally not consumed directly.
"""

# ============================================================================
# 1. SHARED VALUES (single source of truth)
#    Change a path/device/shape here once; every section below references it.
# ============================================================================

# Checkpoint - single source of truth for the PyTorch model (used by export + PyTorch eval).
checkpoint_path = "work_dirs/centerpoint/best_checkpoint.pth"

# Log file path (relative paths are resolved under export.work_dir). None disables file logging.
deploy_log_path = "deployment.log"

# Device settings (shared by export, evaluation, verification).
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)
# Alias reused by the per-backend evaluation settings below so the CUDA device is written once.
_CUDA = devices["cuda"]

# Deployment output layout. _ONNX_DIR / _TENSORRT_DIR are the single source for both the
# export outputs and the evaluation backends' model_dir / engine_dir (kept in sync here).
_DEPLOY_WORK_DIR = "work_dirs/centerpoint_deployment"
_WORK_DIR = _DEPLOY_WORK_DIR.rstrip("/")
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# TensorRT profile shapes (hoisted so repeated/grid-derived literals live in one place).
# Voxel encoder input: [num_voxels, num_points_per_voxel, voxel_feature_dim].
_NUM_POINTS_PER_VOXEL = 32
_VOXEL_FEATURE_DIM = 11
# Backbone/neck/head input: [batch, channels, grid_h, grid_w] (check grid size in model config).
# min == opt == max here because the BEV grid is fixed for this model.
_SPATIAL_FEATURE_SHAPE = [1, 32, 1020, 1020]

# ============================================================================
# 2. EXPORT
# ============================================================================

# Export Configuration
# mode: "onnx", "trt", "both", "none"
# work_dir: path to the deployment output root
# onnx_path: path to the ONNX output directory (if mode="trt" and ONNX already exists)
# sample_idx: dataset index of the sample used to trace/shape the exported model
export = dict(
    mode="both",
    work_dir=_DEPLOY_WORK_DIR,
    onnx_path=_ONNX_DIR,
    sample_idx=0,
)

# ONNX Export Settings (shared across all components).
onnx_config = dict(
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

# TensorRT Build Settings (shared across all components).
# Supports `auto`, `fp16`, `fp32_tf32`, and `strongly_typed`.
tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=2 << 30,
)

# Unified Component Configuration (Single Source of Truth)
#
# Component key is the unique identifier (used for config lookup, filenames, logs).
# Each component defines:
#   - onnx_file: Output ONNX filename
#   - engine_file: Output TensorRT engine filename
#   - io: Input/output specification for ONNX export
#   - tensorrt_profile: TensorRT optimization profile (min/opt/max shapes)
components = dict(
    pts_voxel_encoder=dict(
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
                # Make sure to match the shape of the input to the model.
                # [num_voxels, num_points_per_voxel, voxel_feature_dim]
                min_shape=[1000, _NUM_POINTS_PER_VOXEL, _VOXEL_FEATURE_DIM],
                opt_shape=[20000, _NUM_POINTS_PER_VOXEL, _VOXEL_FEATURE_DIM],
                max_shape=[96000, _NUM_POINTS_PER_VOXEL, _VOXEL_FEATURE_DIM],
            ),
        ),
    ),
    pts_backbone_neck_head=dict(
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
                # Make sure to match the shape of the input to the model.
                # check grid size in the model config
                min_shape=_SPATIAL_FEATURE_SHAPE,
                opt_shape=_SPATIAL_FEATURE_SHAPE,
                max_shape=_SPATIAL_FEATURE_SHAPE,
            ),
        ),
    ),
)

# ============================================================================
# 3. EVALUATION
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=5,
    num_warmup=3,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=True,
            device=_CUDA,
        ),
        onnx=dict(
            enabled=True,
            device=_CUDA,
            model_dir=_ONNX_DIR,
        ),
        tensorrt=dict(
            enabled=True,
            device=_CUDA,
            engine_dir=_TENSORRT_DIR,
        ),
    ),
)

# ============================================================================
# 4. VERIFICATION
#
# Tolerance is backend- and machine-dependent:
# - The same scenario can show very different max/mean diffs on different machines: GPU
#   architecture, driver, ORT/CUDA/TRT versions, and ORT's CUDA graph partitioning (CPU
#   fallback nodes for small ops) all change numerics. ONNX on CPU, ONNX on CUDA, and
#   TensorRT on CUDA are not directly comparable to each other as "one true" references.
# - Additionally, the verification configuration should use a precision-aware tolerance,
#   especially when FP16 is enabled.
# ============================================================================
verification = dict(
    enabled=False,
    # TODO(vividf): double check the tolerance value
    tolerance=1,
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
