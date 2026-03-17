from pathlib import Path
from typing import Optional, Union

from mmengine.hooks.logger_hook import SUFFIX_TYPE
from mmengine.hooks.logger_hook import LoggerHook as _LoggerHook
from mmengine.registry import HOOKS


@HOOKS.register_module(force=True)
class LoggerHook(_LoggerHook):
    """A custom logger hook for logging information, for example, logging to tensorboard during inference."""

    def __init__(
        self,
        interval: int = 10,
        ignore_last: bool = True,
        interval_exp_name: int = 1000,
        out_dir: Optional[Union[str, Path]] = None,
        out_suffix: SUFFIX_TYPE = (".json", ".log", ".py", "yaml"),
        keep_local: bool = True,
        file_client_args: Optional[dict] = None,
        log_metric_by_epoch: bool = True,
        backend_args: Optional[dict] = None,
        logging_inference_to_tensorboard: bool = False,
        log_metrics_to_tensorboard: bool = True,
    ) -> None:
        """
        Inherited from LoggerHook, please check the base class.
        :param logging_inference_to_tensorboard: Set True to logging information to tensorboard during inference.
        """
        super(LoggerHook, self).__init__(
            interval=interval,
            ignore_last=ignore_last,
            interval_exp_name=interval_exp_name,
            out_dir=out_dir,
            out_suffix=out_suffix,
            keep_local=keep_local,
            file_client_args=file_client_args,
            log_metric_by_epoch=log_metric_by_epoch,
            backend_args=backend_args,
        )
        self._logging_inference_to_tensorboard = logging_inference_to_tensorboard
        self._log_metrics_to_tensorboard = log_metrics_to_tensorboard
        # There's no test iter in https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L477 where
        # runner.iter doesn't increase by 1 after an iteration during inference
        # Note that we assume that it's running in a single-gpu environment
        self._test_iter = 0

    def after_val_epoch(self, runner, metrics=None) -> None:
        """Optionally skip default TensorBoard metric logging for validation."""
        tag, log_str = runner.log_processor.get_log_after_epoch(runner, len(runner.val_dataloader), "val")
        runner.logger.info(log_str)
        if not self._log_metrics_to_tensorboard:
            return
        if self.log_metric_by_epoch:
            if isinstance(runner._train_loop, dict) or runner._train_loop is None:
                epoch = 0
            else:
                epoch = runner.epoch
            runner.visualizer.add_scalars(tag, step=epoch, file_path=self.json_log_path)
        else:
            if isinstance(runner._train_loop, dict) or runner._train_loop is None:
                iter = 0
            else:
                iter = runner.iter
            runner.visualizer.add_scalars(tag, step=iter, file_path=self.json_log_path)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Everything is the same to LoggerHook except it saves info to tensorboard as well."""

        if self._logging_inference_to_tensorboard:
            if self.every_n_inner_iters(batch_idx, self.interval):
                tag, log_str = runner.log_processor.get_log_after_iter(runner, batch_idx, "test")
                runner.logger.info(log_str)
                # This line saves info to tensorboard
                runner.visualizer.add_scalars(tag, step=self._test_iter + 1, file_path=self.json_log_path)
        else:
            super().after_test_iter(runner=runner, batch_idx=batch_idx, data_batch=data_batch, outputs=outputs)

        self._test_iter += 1
