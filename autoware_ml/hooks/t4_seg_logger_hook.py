from mmengine.registry import HOOKS

from .logger_hook import LoggerHook


@HOOKS.register_module()
class T4SegLoggerHook(LoggerHook):
    """Logger hook for T4 segmentation configs using custom TensorBoard metric tags."""

    def after_val_epoch(self, runner, metrics=None) -> None:
        """Log validation results without the default TensorBoard scalar dump."""
        _, log_str = runner.log_processor.get_log_after_epoch(runner, len(runner.val_dataloader), "val")
        runner.logger.info(log_str)
