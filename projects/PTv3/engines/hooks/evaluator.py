"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import utils.comm as comm

from autoware_ml.segmentation3d.datasets.utils import class_mapping_to_names
from autoware_ml.segmentation3d.evaluation import (
    SegEvalResult,
    build_t4_seg_tb_scalars,
    iter_t4_seg_confusion_matrix_figures,
    t4_seg_eval_from_hists,
    update_seg_eval_histograms,
)

from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        cfg = self.trainer.cfg
        num_classes = cfg.data.num_classes
        ignore_index = cfg.data.ignore_index
        metric_options = getattr(cfg, "metric_options", None) or {}
        distance_ranges = metric_options.get("distance_ranges") or []
        reduce_device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )

        total_hist = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=reduce_device)
        range_hist_tensors = {
            f"{lo:g}-{hi:g}m": torch.zeros((num_classes, num_classes), dtype=torch.float64, device=reduce_device)
            for lo, hi in distance_ranges
        }
        loss_sum = 0.0
        loss_count = 0

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1].detach().cpu().numpy()
            segment = input_dict["segment"].detach().cpu().numpy()

            # Extract BEV coordinate for range-based metrics.
            coord_np = None
            if "coord" in input_dict:
                coord = input_dict["coord"]
                if isinstance(coord, torch.Tensor):
                    coord_np = coord.detach().cpu().numpy()
                    if coord_np.ndim != 2 or coord_np.shape[1] < 2:
                        coord_np = None

            sample_total_hist = np.zeros((num_classes, num_classes), dtype=np.float64)
            sample_range_hists = {
                label: np.zeros((num_classes, num_classes), dtype=np.float64) for label in range_hist_tensors
            }
            update_seg_eval_histograms(
                total_hist=sample_total_hist,
                pred=pred,
                gt=segment,
                num_classes=num_classes,
                ignore_index=ignore_index,
                range_hists=sample_range_hists,
                coord=coord_np,
                distance_ranges=distance_ranges if distance_ranges else None,
            )
            total_hist += torch.from_numpy(sample_total_hist).to(device=total_hist.device)
            for label, hist in sample_range_hists.items():
                range_hist_tensors[label] += torch.from_numpy(hist).to(device=total_hist.device)
            loss_sum += float(loss.item())
            loss_count += 1

            info = f"Test: [{i + 1}/{len(self.trainer.val_loader)}] "
            if "origin_coord" in input_dict:
                info = "Interp. " + info
            self.trainer.logger.info(info + f"Loss {loss.item():.4f}")

        comm.synchronize()
        if comm.get_world_size() > 1:
            dist.reduce(total_hist, dst=0)
            for hist in range_hist_tensors.values():
                dist.reduce(hist, dst=0)
        loss_reduced = comm.reduce_dict(
            {
                "loss_sum": torch.tensor(loss_sum, dtype=torch.float64, device=reduce_device),
                "loss_count": torch.tensor(loss_count, dtype=torch.float64, device=reduce_device),
            },
            average=False,
        )
        if not comm.is_main_process():
            return

        loss_avg = float(loss_reduced["loss_sum"] / loss_reduced["loss_count"].clamp_min(1.0))

        mapped_class_names = class_mapping_to_names(cfg.class_mapping, ignore_index)
        assert len(mapped_class_names) == num_classes, (
            "class_mapping_to_names length must match num_classes: " f"{len(mapped_class_names)} vs {num_classes}"
        )
        label2cat = {i: mapped_class_names[i] for i in range(num_classes)}

        eval_result: SegEvalResult = t4_seg_eval_from_hists(
            total_hist=total_hist.cpu().numpy(),
            label2cat=label2cat,
            ignore_index=ignore_index,
            range_hists={label: hist.cpu().numpy() for label, hist in range_hist_tensors.items()},
            logger=self.trainer.logger,
        )

        epoch = self.trainer.epoch + 1
        writer = self.trainer.writer
        if writer is not None:
            writer.add_scalar("val/loss", loss_avg, epoch)
            for tag, value in build_t4_seg_tb_scalars(
                metrics=eval_result.metrics,
                class_names=mapped_class_names,
                stage="val",
                distance_ranges=distance_ranges,
            ).items():
                writer.add_scalar(tag, value, epoch)

            for tag, fig in iter_t4_seg_confusion_matrix_figures(eval_result, mapped_class_names, "val"):
                writer.add_figure(tag, fig, epoch)
                plt.close(fig)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = eval_result.metrics.get("miou", 0.0)
        self.trainer.comm_info["current_metric_name"] = "miou"

    def after_train(self):
        self.trainer.logger.info("Best {}: {:.4f}".format("miou", self.trainer.best_metric_value))
