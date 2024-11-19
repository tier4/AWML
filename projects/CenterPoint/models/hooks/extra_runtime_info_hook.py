from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.hooks.runtime_info_hook import DATA_BATCH


@HOOKS.register_module()
class ExtraRuntimeInfoHook(Hook):
    """A hook that updates extra runtime information into message hub.

    E.g. ``epoch``, ``iter``, ``max_epochs``, and ``max_iters`` for the
    training state. Components that cannot access the runner can get runtime
    information through the message hub.
    """

    priority = 'VERY_HIGH'

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        """Update current iter and learning rate information before every
        iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        runner.message_hub.update_info('iter', runner.iter)
        momentum_dict = runner.optim_wrapper.get_momentum()
        assert isinstance(momentum_dict, dict), (
            '`runner.optim_wrapper.get_lr()` should return a dict '
            'of learning rate when training with OptimWrapper(single '
            'optimizer) or OptimWrapperDict(multiple optimizer), '
            f'but got {type(momentum_dict)} please check your optimizer '
            'constructor return an `OptimWrapper` or `OptimWrapperDict` '
            'instance')
        for name, momentum in momentum_dict.items():
            runner.message_hub.update_scalar(f'train/{name}', momentum[0])