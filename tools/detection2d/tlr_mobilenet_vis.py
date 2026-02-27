#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmpretrain.apis import init_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MobileNet TLR predictions from infos JSON.")
    parser.add_argument("--config", required=True, help="Model config path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--infos", required=True, help="Path to tlr_infos_{split}.json")
    parser.add_argument("--out-dir", default="work_dirs/tlr_vis", help="Output directory for visualizations.")
    parser.add_argument("--device", default="cuda:0", help="Device for inference.")
    parser.add_argument("--max-samples", type=int, default=50, help="Max number of instances to visualize.")
    parser.add_argument("--class-name", default=None, help="Optional GT class filter, e.g. 'red,straight'.")
    return parser.parse_args()


def _get_infer_size(cfg: Config) -> int:
    pipeline = None
    if hasattr(cfg, "resize_pipeline"):
        pipeline = cfg.resize_pipeline
    elif hasattr(cfg, "test_pipeline"):
        pipeline = cfg.test_pipeline
    else:
        pipeline = cfg.val_dataloader.dataset.pipeline

    for t in pipeline:
        if t.get("type") != "Resize":
            continue
        scale = t.get("scale")
        if isinstance(scale, int):
            return int(scale)
        if isinstance(scale, (tuple, list)) and len(scale) > 0:
            return int(max(scale))

    return 224


def _safe_crop(img: np.ndarray, bbox):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    return img[y1:y2, x1:x2, :], (x1, y1, x2, y2)


def _resize_and_pad(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img

    scale = float(size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas


def _to_model_input(img: np.ndarray, cfg: Config, device: str) -> torch.Tensor:
    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 1, 3)
    std = np.array(dp.get("std", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)
    to_rgb = bool(dp.get("to_rgb", False))

    x = img.astype(np.float32)
    if to_rgb:
        x = x[..., ::-1]
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    x = np.ascontiguousarray(x)
    return torch.from_numpy(x).unsqueeze(0).to(device)


def _extract_pred(pred):
    pred_label = None
    pred_score = None

    if hasattr(pred, "pred_label"):
        try:
            label_tensor = torch.as_tensor(pred.pred_label).detach().flatten()
            if label_tensor.numel() == 1:
                pred_label = int(label_tensor.item())
            elif label_tensor.numel() > 1:
                pred_label = int(torch.argmax(label_tensor).item())
        except Exception:
            pred_label = None

    scores = None
    if hasattr(pred, "pred_score"):
        scores = pred.pred_score
    elif hasattr(pred, "pred_scores"):
        scores = pred.pred_scores

    if scores is not None:
        try:
            score_tensor = torch.as_tensor(scores).detach().flatten().float()
            if score_tensor.numel() > 0:
                top_idx = int(torch.argmax(score_tensor).item())
                pred_score = float(score_tensor[top_idx].item())
                if pred_label is None:
                    pred_label = top_idx
        except Exception:
            pred_score = None

    return pred_label, pred_score


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)

    infer_size = _get_infer_size(cfg)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    with open(args.infos, "r") as f:
        infos = json.load(f)

    classes = infos.get("metainfo", {}).get("classes", [])
    class_filter = args.class_name.strip().lower() if args.class_name else None

    os.makedirs(args.out_dir, exist_ok=True)

    saved = 0
    for item in infos.get("data_list", []):
        img_path = item.get("img_path")
        if not img_path:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        for inst_idx, inst in enumerate(item.get("instances", [])):
            label = inst.get("bbox_label")
            if label is None or label >= len(classes):
                continue

            gt_name = classes[label]
            if class_filter and gt_name.lower() != class_filter:
                continue

            bbox = inst.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            crop, draw_bbox = _safe_crop(img, bbox)
            if crop is None or crop.size == 0:
                continue

            crop = _resize_and_pad(crop, infer_size)
            model_input = _to_model_input(crop, cfg, args.device)

            with torch.no_grad():
                pred = model(inputs=model_input, mode="predict")[0]

            pred_label, pred_score = _extract_pred(pred)
            pred_name = classes[pred_label] if pred_label is not None and pred_label < len(classes) else "unknown"

            vis = img.copy()
            x1, y1, x2, y2 = draw_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if pred_score is not None:
                text = f"pred: {pred_name} ({pred_score:.3f}) | gt: {gt_name}"
            else:
                text = f"pred: {pred_name} | gt: {gt_name}"
            cv2.putText(vis, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)

            base = Path(img_path).stem
            out_path = os.path.join(args.out_dir, f"{saved:06d}_{base}_i{inst_idx}.jpg")
            cv2.imwrite(out_path, vis)

            saved += 1
            if saved >= args.max_samples:
                print(f"Saved {saved} visualizations to {args.out_dir}")
                return

    print(f"Saved {saved} visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
