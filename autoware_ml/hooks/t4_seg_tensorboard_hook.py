import matplotlib.pyplot as plt
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.visualization import Visualizer

from autoware_ml.segmentation3d.evaluation import (
    T4SegMetric,
    build_t4_seg_tb_scalars,
    figure_to_numpy,
    iter_t4_seg_confusion_matrix_figures,
)


@HOOKS.register_module()
class T4SegTensorboardHook(Hook):
    """Log shared T4 segmentation TensorBoard tags for MMEngine runners."""

    priority = "LOW"

    def after_val_epoch(self, runner, metrics=None):
        self._log_stage(runner, stage="val", step=runner.iter)

    def after_test_epoch(self, runner, metrics=None):
        self._log_stage(runner, stage="test", step=0)

    def _log_stage(self, runner, stage: str, step: int) -> None:
        metric = self._get_metric(runner, stage)
        if metric is None or metric.last_eval_result is None:
            return

        try:
            vis = Visualizer.get_current_instance()
        except Exception:
            return

        class_names = [metric.last_label2cat[i] for i in sorted(metric.last_label2cat)]
        scalars = build_t4_seg_tb_scalars(
            metrics=metric.last_eval_result.metrics,
            class_names=class_names,
            stage=stage,
            distance_ranges=metric.distance_ranges,
        )
        if scalars:
            vis.add_scalars(scalars, step=step)

        for tag, fig in iter_t4_seg_confusion_matrix_figures(metric.last_eval_result, class_names, stage):
            try:
                vis.add_image(tag, figure_to_numpy(fig), step=step)
            except Exception:
                pass
            finally:
                plt.close(fig)

    @staticmethod
    def _get_metric(runner, stage: str):
        loop = runner.val_loop if stage == "val" else runner.test_loop
        evaluator = getattr(loop, "evaluator", None)
        if evaluator is None:
            return None
        for metric in getattr(evaluator, "metrics", []):
            if isinstance(metric, T4SegMetric):
                return metric
        return None
