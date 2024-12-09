default_hooks = dict(
    checkpoint=dict(interval=1, type="CheckpointHook"),
    logger=dict(interval=100, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(enable=False, type="VisualizationHook"),
)
default_scope = "mmpretrain"
env_cfg = dict(
    cudnn_benchmark=False, dist_cfg=dict(backend="nccl"), mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0)
)
launcher = "none"
