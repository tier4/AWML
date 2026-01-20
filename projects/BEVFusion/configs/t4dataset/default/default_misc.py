vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="NuScenes metric/T4Metric/mAP"),
)
custom_hooks = [dict(type="DisableObjectSampleHook", disable_after_epoch=15)]
log_processor = dict(window_size=50)

randomness = dict(seed=0, diff_rank_seed=True, deterministic=True)

# Load the best checkpoint from the previous training if finetuning
load_from = None
