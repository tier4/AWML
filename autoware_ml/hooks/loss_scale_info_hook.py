from mmengine.hooks import Hook
from mmengine.hooks.runtime_info_hook import DATA_BATCH
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LossScaleInfoHook(Hook):
    """A hook that updates momentum information into message hub.

    E.g. ``epoch``, ``iter``, ``max_epochs``, and ``max_iters`` for the
    training state. Components that cannot access the runner can get runtime
    information through the message hub.
    """

    priority = "VERY_HIGH"

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Update current iter and learning rate information before every
        iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        if hasattr(runner.optim_wrapper, "loss_scaler"):
            runner.message_hub.update_scalar(f"train/loss_scaler", runner.optim_wrapper.loss_scaler.get_scale())
