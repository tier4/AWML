"""
BEVFusion-L Deployment Configuration — split sparse+dense, merged & optimized (FP16).

Layout (single file, grouped by concern; mirrors centerpoint/config/deploy_config.py):
  1. SHARED VALUES  - single source of truth reused across sections (paths, devices, shapes).
  2. EXPORT         - spconv/export flags, ONNX/TensorRT build settings, component definitions.
  3. EVALUATION     - per-backend evaluation settings.
  4. VERIFICATION   - cross-backend numerical verification scenarios.

Only the top-level names `checkpoint_path`, `devices`, `export`, `components`, `runtime_io`,
`onnx_config`, `tensorrt_config`, `evaluation`, `verification`, plus the BEVFusion-only flags
`spconv_do_sort`, `spconv_fuse_implicit_gemm_relu`, `fuse_spconv_bn`, `bevfusion_merge`, are read
(by `BaseDeploymentConfig` / `BEVFusionDeploymentConfig`). Names prefixed with `_` are local
single-source helpers and are intentionally not consumed directly.

Requirements:
  - LiDAR-only model (`fusion_layer is None`, `img_backbone is None`).
  - `components.bevfusion_dense.tensorrt_profile.lidar_bev` H,W must equal
    `grid_size[0:2] // out_size_factor` (e.g. 1440/8 → 180×180). Do NOT widen H/W: `bbox_head`
    uses fixed `bev_pos` and heatmap length H*W, so a wide profile breaks Reshape/Gather and
    yields garbage mAP. Adjust channel C (default 256) to the sparse encoder output.

CLI::

    python -m deployment.cli.main bevfusion_l \\
        deployment/projects/bevfusion_l/config/deploy_config.py \\
        <your_model_cfg.py>
"""

# ============================================================================
# 1. SHARED VALUES (single source of truth)
# ============================================================================

# Checkpoint - single source of truth for the PyTorch model (used by export + PyTorch eval).
checkpoint_path = "work_dirs/bevfusion/bevfusion_2_8/best_epoch_25.pth"
# checkpoint_path = "vivid/bench_comparison/bevfusion_2_7/best_epoch_28.pth"

# Device settings (shared by export, evaluation, verification).
devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)
# Alias reused by the per-backend evaluation settings below so the CUDA device is written once.
_CUDA = devices["cuda"]

# Deployment output layout. _ONNX_DIR / _TENSORRT_DIR are the single source for both the export
# outputs and the evaluation backends' engine_dir (kept in sync here).
_WORK_DIR = "work_dirs/bevfusion_deployment_2_8"
_ONNX_DIR = f"{_WORK_DIR}/onnx"
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# Dense head BEV feature map: [batch, channels, grid_h, grid_w] = grid_size // out_size_factor.
# Fully static (min == opt == max) so constant folding can drop the head's shape-glue (~177
# nodes) and the split graph's node count matches the monolithic export.
_LIDAR_BEV_SHAPE = [1, 256, 180, 180]

# Sparse (spconv) voxel input profile: [num_voxels, max_points_per_voxel, voxel_feature_dim].
_MAX_POINTS_PER_VOXEL = 32
_VOXEL_FEATURE_DIM = 5
_VOXELS_OPT = 64000
_VOXELS_MAX = 256000

# ============================================================================
# 2. EXPORT
# ============================================================================

# Bake the pair-mask argsort into GetIndicePairsImplicitGemm.do_sort_i at ONNX symbolic export.
spconv_do_sort = False

# Sparse ONNX postprocess: fuse each `ImplicitGemm -> Relu` into an activated ImplicitGemm
# (applied as a post-export transform by export/component_builder.py after exporting the sparse ONNX).
# - True  : bake activation into ImplicitGemm (act_type)
# - False : keep explicit Relu nodes
spconv_fuse_implicit_gemm_relu = True

# Fuse SparseConv + BN in `pts_middle_encoder` before ONNX export (eval-mode Conv-BN fold),
# producing a BN-free sparse subgraph in the exported ONNX.
fuse_spconv_bn = True

# Export mode: "onnx", "trt", "both", "none". sample_idx: dataset index used to trace/shape.
export = dict(
    mode="none",
    work_dir=_WORK_DIR,
    onnx_path=_ONNX_DIR,
    sample_idx=0,
)

# Keep the split component definitions but also emit a single merged full-graph ONNX/engine.
# - enabled=False: split sparse+dense ONNX/engine only.
# - enabled=True : additionally emit one merged ONNX + engine + backend pipeline.
bevfusion_merge = dict(
    enabled=True,
    onnx_file="bevfusion_lidar_fp16_opt.onnx",
    engine_file="bevfusion_lidar_fp16_opt.engine",
)

# ONNX export settings (shared across all components).
# BEVFusion 2.8.x exports at opset 18 (matches projects/BEVFusion/configs/deploy/*_tensorrt_dynamic.py).
onnx_config = dict(
    opset_version=18,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

# TensorRT build settings (shared across all components). plugin_libraries loads the spconv
# ImplicitGemm plugin before engine build/deserialize.
tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=1 << 32,
    plugin_libraries=["/opt/plugins/libautoware_tensorrt_plugins.so"],
)

# Split components (must keep keys `bevfusion_sparse` + `bevfusion_dense`; the merged full graph
# is derived from this pair by BEVFusionDeploymentConfig when bevfusion_merge is enabled).
components = dict(
    bevfusion_sparse=dict(
        onnx_file="bevfusion_sparse.onnx",
        engine_file="bevfusion_sparse.engine",
        io=dict(
            inputs=[
                dict(name="voxels", dtype="float32"),
                dict(name="coors", dtype="int32"),
                dict(name="num_points_per_voxel", dtype="int32"),
            ],
            outputs=[
                dict(name="lidar_bev", dtype="float32"),
            ],
            dynamic_axes={
                "voxels": {0: "voxels_num"},
                "coors": {0: "voxels_num"},
                "num_points_per_voxel": {0: "voxels_num"},
            },
        ),
        tensorrt_profile=dict(
            voxels=dict(
                min_shape=[1, _MAX_POINTS_PER_VOXEL, _VOXEL_FEATURE_DIM],
                opt_shape=[_VOXELS_OPT, _MAX_POINTS_PER_VOXEL, _VOXEL_FEATURE_DIM],
                max_shape=[_VOXELS_MAX, _MAX_POINTS_PER_VOXEL, _VOXEL_FEATURE_DIM],
            ),
            coors=dict(
                min_shape=[1, 3],
                opt_shape=[_VOXELS_OPT, 3],
                max_shape=[_VOXELS_MAX, 3],
            ),
            num_points_per_voxel=dict(
                min_shape=[1],
                opt_shape=[_VOXELS_OPT],
                max_shape=[_VOXELS_MAX],
            ),
        ),
    ),
    bevfusion_dense=dict(
        onnx_file="bevfusion_dense.onnx",
        engine_file="bevfusion_dense.engine",
        io=dict(
            inputs=[
                dict(name="lidar_bev", dtype="float32"),
            ],
            outputs=[
                dict(name="bbox_pred", dtype="float32"),
                dict(name="score", dtype="float32"),
                dict(name="label_pred", dtype="int64"),
            ],
            # Static lidar_bev input (see _LIDAR_BEV_SHAPE), so no dynamic axes.
            dynamic_axes={},
        ),
        # H,W fixed to head grid (grid_size // out_size_factor). Widen only batch dim if needed.
        tensorrt_profile=dict(
            lidar_bev=dict(
                min_shape=_LIDAR_BEV_SHAPE,
                opt_shape=_LIDAR_BEV_SHAPE,
                max_shape=_LIDAR_BEV_SHAPE,
            ),
        ),
    ),
)

runtime_io = dict(
    info_file="info/t4dataset_j6gen2_base_infos_test.pkl",
)

# ============================================================================
# 3. EVALUATION
#    ONNX *inference* is unsupported for BEVFusion (sparse graph needs the TRT plugin).
# ============================================================================
evaluation = dict(
    enabled=True,
    num_samples=5,
    num_warmup=2,
    verbose=True,
    backends=dict(
        pytorch=dict(
            enabled=False,
            device=_CUDA,
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
# ============================================================================
verification = dict(
    enabled=False,
    tolerance=1,
    num_verify_samples=1,
    devices=devices,
    scenarios=dict(
        both=[],
        onnx=[],
        trt=[],
        none=[],
    ),
)
