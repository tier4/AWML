vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]

visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")
custom_hooks = [dict(type="DisableObjectSampleHook", disable_after_epoch=15)]
randomness = dict(seed=0, diff_rank_seed=True, deterministic=True)
