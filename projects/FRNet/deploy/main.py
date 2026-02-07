"""FRNet deployment – main entry point.

Runs the full pipeline: load configs, prepare dataset, export model
(torch -> ONNX -> TensorRT), run inference and optionally visualize.

Example:
    python main.py checkpoint.pth \\
        --model-cfg  configs/t4dataset/frnet_1xb8_t4dataset-ot128-seg.py \\
        --deploy-cfg configs/deploy/t4dataset/frnet_tensorrt_dynamic.py \\
        --execution tensorrt --num-samples 4 --verbose --show
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from dataset import create_dataset_handler
from mmdeploy.utils import load_config
from mmdet3d.evaluation.functional import seg_eval
from mmdet3d.utils import register_all_modules
from mmengine.logging import MMLogger
from onnx_model import OnnxModel
from postprocessing import Postprocessing
from preprocessing import Preprocessing
from torch_model import TorchModel
from trt_model import TrtModel
from visualizer import Visualizer

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Exiting...")
    exit(1)

register_all_modules()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="FRNet deployment and inference.")
    parser.add_argument("checkpoint", type=str, help="Path to PyTorch checkpoint file.")
    parser.add_argument(
        "--execution",
        choices=["torch", "onnx", "tensorrt"],
        default=None,
        help="Backend used for inference.  When omitted only model export is performed.",
    )
    parser.add_argument("--model-cfg", type=str, required=True, help="Model config file path.")
    parser.add_argument("--deploy-cfg", type=str, required=True, help="Deploy config file path.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset root directory.  Falls back to data_root from the model config.",
    )
    parser.add_argument("--threshold", type=float, default=-999.9, help="Score threshold for predictions.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to run inference on.")
    parser.add_argument("--no-deploy", dest="deploy", action="store_false", help="Skip model export (use existing).")
    parser.add_argument("--show", action="store_true", help="Show interactive 3-D visualization of predictions.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    return parser.parse_args()


def main() -> None:
    """Run the deployment / inference pipeline."""
    args = parse_args()

    logger = MMLogger.get_current_instance()

    model_cfg, deploy_cfg = load_config(args.model_cfg, args.deploy_cfg)
    dataset_dir = args.dataset_dir or model_cfg.data_root
    onnx_path = os.path.join(os.path.dirname(args.checkpoint), "frnet.onnx")

    logger.info(f"Dataset type: {model_cfg.dataset_type}")
    logger.info(f"Dataset directory: {dataset_dir}")

    dataset = create_dataset_handler(model_cfg, dataset_dir)

    samples = dataset.get_samples()
    num_samples = min(args.num_samples, len(samples))
    samples = samples[:num_samples]
    logger.info(f"Using {num_samples} sample(s) for inference.")

    preprocessing = Preprocessing(config=model_cfg)
    postprocessing = Postprocessing(
        score_threshold=args.threshold,
        ignore_index=model_cfg.ignore_index,
    )

    # A representative input is needed to trace the ONNX graph.
    representative_input = preprocessing.preprocess(dataset.load_points(samples[0]))

    torch_model = TorchModel(model_cfg=model_cfg, checkpoint_path=args.checkpoint)
    onnx_model = OnnxModel(
        deploy_cfg=deploy_cfg,
        model=torch_model.model,
        batch_inputs_dict=representative_input,
        onnx_path=onnx_path,
        deploy=args.deploy,
        verbose=args.verbose,
    )

    trt_model = None
    if args.execution == "tensorrt" or args.deploy:
        trt_model = TrtModel(
            deploy_cfg=deploy_cfg,
            onnx_path=onnx_path,
            deploy=args.deploy,
            verbose=args.verbose,
        )

    visualizer = Visualizer(class_names=dataset.class_names, palette=dataset.palette) if args.show else None

    if args.execution is None:
        return

    models = {
        "torch": torch_model,
        "onnx": onnx_model,
        "tensorrt": trt_model,
    }
    model = models[args.execution]

    gt_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []

    logger.info(f"Score threshold: {args.threshold}")
    for sample in samples:
        logger.info("-" * 80)
        source_name = sample.get("source_name")
        if source_name is not None:
            logger.info(f"Source: {source_name}")

        points = dataset.load_points(sample)
        batch_inputs_dict = preprocessing.preprocess(points)
        num_points = batch_inputs_dict["num_points"]

        logger.info(f"Running {args.execution} inference...")
        predictions = model.inference(batch_inputs_dict)
        result = postprocessing.postprocess(predictions)

        # Slice predictions to original points (drop interpolated points)
        result_original = result[:num_points]

        # Collect GT / pred pairs for evaluation
        gt = dataset.load_gt(sample)
        if gt is not None:
            gt_list.append(gt)
            pred_list.append(result_original)

        if visualizer is not None:
            visualizer.visualize(batch_inputs_dict, result)

    if gt_list:
        logger.info("=" * 80)
        logger.info(f"Evaluating {len(gt_list)} sample(s) with ground-truth labels...")
        seg_eval(
            gt_labels=gt_list,
            seg_preds=pred_list,
            label2cat=dataset.label2cat,
            ignore_index=model_cfg.ignore_index,
            logger=logger,
        )
    else:
        logger.warning("No ground-truth annotations available – skipping evaluation.")


if __name__ == "__main__":
    main()
