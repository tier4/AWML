"""
BEVFusion deploy config — **split ONNX / TensorRT (route 1)**

Use this instead of ``deploy_config.py`` when you want:
  - ``bevfusion_sparse.onnx`` / ``.engine`` — voxelization stays outside; graph is
    ``pts_middle_encoder`` only (spconv / plugins / libspconv).
  - ``bevfusion_dense.onnx`` / ``.engine`` — ``pts_backbone`` + ``pts_neck`` + ``bbox_head``
    (+ the head postprocess). Suitable for plain TensorRT without spconv ops.

**Requirements**
  - LiDAR-only model: ``fusion_layer is None`` and ``img_backbone is None``.

  - Set ``bevfusion_dense.tensorrt_profile.lidar_bev`` **H,W** to ``grid_size[0:2] // out_size_factor``
    (e.g. 1440/8 → **180×180**). Do **not** use a wide H/W range: ``bbox_head`` uses fixed ``bev_pos``
    and heatmap length ``H*W``; TRT profiles like 32×32 or 2048×2048 break ``Reshape``/``Gather``
    consistency and yield garbage mAP.
  - Adjust channel **C** (default ``256``) to match ``pts_backbone.in_channels`` / sparse encoder output.

CLI::

    python -m deployment.cli.main bevfusion_l \\
        deployment/projects/bevfusion_l/config/deploy_config_without_opt.py \\
        <your_model_cfg.py>
"""

spconv_do_sort = False

# ============================================================================
# Sparse ONNX postprocess (FP): fuse ImplicitGemm with trailing Relu/Add(const)+Relu.
# ----------------------------------------------------------------------------
# Applied automatically as a post-export transform by
# deployment/projects/bevfusion_l/export/component_builder.py after exporting ``bevfusion_sparse.onnx``.
# - True  : bake activation into ImplicitGemm (act_type / optional 6th bias input)
# - False : keep explicit Relu/Add nodes
# ============================================================================
spconv_fuse_implicit_gemm_relu = False

# Fuse SparseConv + BN in ``pts_middle_encoder`` before ONNX export (eval-mode Conv-BN fold).
# Produces a BN-free sparse subgraph in the exported ONNX.
fuse_spconv_bn = False

# ============================================================================
# Checkpoint Path
# ============================================================================
checkpoint_path = "work_dirs/bevfusion/bevfusion_2_8/best_epoch_25.pth"
# checkpoint_path = "vivid/bench_comparison/bevfusion_2_7/best_epoch_28.pth"


devices = dict(
    cpu="cpu",
    cuda="cuda:0",
)

export = dict(
    mode="trt",
    work_dir="work_dirs/bevfusion_deployment_2_8_original",
    onnx_path="work_dirs/bevfusion_deployment_2_8_original/onnx",
    # Dataset index of the sample used to trace/shape the exported model (read by ExportConfig,
    # same as CenterPoint's export.sample_idx).
    sample_idx=0,
)


# Optional: keep split component definitions but also emit one merged full-graph ONNX/engine.
# - False: split sparse+dense ONNX/engine (default)
# - True : one ONNX + one engine + one backend pipeline
bevfusion_merge = dict(
    enabled=True,
    onnx_file="bevfusion_lidar.onnx",
    engine_file="bevfusion_lidar.engine",
)


_WORK_DIR = str(export["work_dir"]).rstrip("/")
_TENSORRT_DIR = f"{_WORK_DIR}/tensorrt"

# ============================================================================
# Split components (must keep keys ``bevfusion_sparse`` + ``bevfusion_dense``)
# ============================================================================
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
                # lidar_bev output is a fixed dense grid [1, C*D, 180, 180] regardless of voxel
                # count; marking its batch/H/W dynamic re-dynamizes the boundary and defeats
                # constant folding (adds ~50 shape-glue nodes). Keep it static.
            },
        ),
        tensorrt_profile=dict(
            voxels=dict(
                min_shape=[1, 32, 5],
                opt_shape=[64000, 32, 5],
                max_shape=[256000, 32, 5],
            ),
            coors=dict(
                min_shape=[1, 3],
                opt_shape=[64000, 3],
                max_shape=[256000, 3],
            ),
            num_points_per_voxel=dict(
                min_shape=[1],
                opt_shape=[64000],
                max_shape=[256000],
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
            # Spatial dims must stay at head BEV resolution; see module docstring.
            # lidar_bev input is fully static ([1,256,180,180]; TRT profile is min=opt=max),
            # so no dynamic axes — this lets constant folding remove the head's shape-glue
            # (~177 nodes) and aligns the split graph node count with the monolithic export.
            dynamic_axes={},
        ),
        # H,W fixed to head grid (grid_size // out_size_factor). Widen only batch dim if needed.
        tensorrt_profile=dict(
            lidar_bev=dict(
                min_shape=[1, 256, 180, 180],
                opt_shape=[1, 256, 180, 180],
                max_shape=[1, 256, 180, 180],
            ),
        ),
    ),
)

runtime_io = dict(
    info_file="info/t4dataset_j6gen2_base_infos_test.pkl",
)

onnx_config = dict(
    # BEVFusion 2.8.x exports at opset 18 (matches projects/BEVFusion/configs/deploy/*_tensorrt_dynamic.py).
    opset_version=18,
    do_constant_folding=True,
    export_params=True,
    keep_initializers_as_inputs=False,
    simplify=False,
)

tensorrt_config = dict(
    precision_policy="fp16",
    max_workspace_size=1 << 32,
    plugin_libraries=["/opt/plugins/libautoware_tensorrt_plugins.so"],
)

evaluation = dict(
    enabled=True,
    num_samples=5,
    num_warmup=2,
    verbose=True,
    # ONNX *inference* is unsupported for BEVFusion.
    backends=dict(
        pytorch=dict(
            enabled=False,
            device=devices["cuda"],
        ),
        tensorrt=dict(
            enabled=True,
            device=devices["cuda"],
            engine_dir=_TENSORRT_DIR,
        ),
    ),
)

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
