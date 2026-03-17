"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np
import torch
import utils.comm as comm

from autoware_ml.segmentation3d.datasets.utils import class_mapping_to_names
from autoware_ml.segmentation3d.evaluation import (
    SegEvalResult,
    figure_to_numpy,
    plot_confusion_matrix,
    t4_seg_eval,
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

        local_results = []
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

            local_results.append(dict(pred=pred, gt=segment, coord=coord_np))
            loss_sum += float(loss.item())
            loss_count += 1

            info = f"Test: [{i + 1}/{len(self.trainer.val_loader)}] "
            if "origin_coord" in input_dict:
                info = "Interp. " + info
            self.trainer.logger.info(info + f"Loss {loss.item():.4f}")

        comm.synchronize()
        gathered = comm.gather(
            dict(results=local_results, loss_sum=loss_sum, loss_count=loss_count),
            dst=0,
        )
        if not comm.is_main_process():
            return

        merged_results = []
        total_loss_sum = 0.0
        total_loss_count = 0
        for item in gathered:
            merged_results.extend(item["results"])
            total_loss_sum += float(item["loss_sum"])
            total_loss_count += int(item["loss_count"])
        loss_avg = total_loss_sum / max(total_loss_count, 1)

        mapped_class_names = class_mapping_to_names(cfg.class_mapping, ignore_index)
        assert len(mapped_class_names) == num_classes, (
            "class_mapping_to_names length must match num_classes: " f"{len(mapped_class_names)} vs {num_classes}"
        )
        label2cat = {i: mapped_class_names[i] for i in range(num_classes)}

        eval_result: SegEvalResult = t4_seg_eval(
            gt_labels=[r["gt"] for r in merged_results],
            seg_preds=[r["pred"] for r in merged_results],
            label2cat=label2cat,
            ignore_index=ignore_index,
            coords_list=[r.get("coord") for r in merged_results] if distance_ranges else None,
            distance_ranges=distance_ranges if distance_ranges else None,
            logger=self.trainer.logger,
        )

        epoch = self.trainer.epoch + 1
        writer = self.trainer.writer
        if writer is not None:
            writer.add_scalar("val/loss", loss_avg, epoch)
            m = eval_result.metrics
            for key in ("miou", "acc", "acc_cls", "mprecision", "mrecall", "mf1"):
                writer.add_scalar(f"val/{key}", m.get(key, 0.0), epoch)
            for name in mapped_class_names:
                writer.add_scalar(f"val/class_iou/{name}", m.get(name, 0.0), epoch)
                for sub in ("precision", "recall", "f1"):
                    writer.add_scalar(f"val/class_{sub}/{name}", m.get(f"{sub}/{name}", 0.0), epoch)
            for lo, hi in distance_ranges:
                lbl = f"{lo:g}-{hi:g}m"
                for key in ("miou", "acc", "acc_cls", "mprecision", "mrecall", "mf1"):
                    writer.add_scalar(f"val/range/{lbl}/{key}", m.get(f"{lbl}/{key}", 0.0), epoch)
                for name in mapped_class_names:
                    writer.add_scalar(f"val/range/{lbl}/class_iou/{name}", m.get(f"{lbl}/{name}", 0.0), epoch)
                    for sub in ("precision", "recall", "f1"):
                        writer.add_scalar(
                            f"val/range/{lbl}/class_{sub}/{name}",
                            m.get(f"{lbl}/{sub}/{name}", 0.0),
                            epoch,
                        )

            if eval_result.cm is not None and eval_result.cm.sum() > 0:
                fig = plot_confusion_matrix(eval_result.cm, mapped_class_names)
                writer.add_figure("val/confusion_matrix", fig, epoch)
                plt.close(fig)
            for lbl, rcm in eval_result.range_cms.items():
                if rcm is not None and rcm.sum() > 0:
                    fig = plot_confusion_matrix(rcm, mapped_class_names, label=lbl)
                    tag = f"val/confusion_matrix_{lbl.replace('-', '_').replace(' ', '_')}"
                    writer.add_figure(tag, fig, epoch)
                    plt.close(fig)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = eval_result.metrics.get("miou", 0.0)
        self.trainer.comm_info["current_metric_name"] = "miou"

    def after_train(self):
        self.trainer.logger.info("Best {}: {:.4f}".format("miou", self.trainer.best_metric_value))
