import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ENTROPY_KINDS = ("sum", "argmax", "normalized")
ENTROPY_AGGS = ("mean", "max")
SCATTER_SIZE = 15
SCATTER_ALPHA = 0.75
GRID_ALPHA = 0.2


def load_prediction_results(paths: List[Path]) -> List[dict]:
    records: List[dict] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def load_frame_summary(paths: List[Path]) -> List[dict]:
    records: List[dict] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def rankdata(arr: np.ndarray) -> np.ndarray:
    """Return average ranks (1-based) with tie handling."""
    order = np.argsort(arr)
    sorted_arr = arr[order]
    ranks = np.zeros(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        rank_val = 0.5 * (i + j) + 1.0  # average rank, 1-based
        ranks[i : j + 1] = rank_val
        i = j + 1
    out = np.empty(len(arr), dtype=float)
    out[order] = ranks
    return out


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without external deps."""
    if len(x) != len(y) or len(x) == 0:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def apply_precision_jitter(values: np.ndarray, jitter: float) -> np.ndarray:
    if jitter <= 0:
        return values
    rng = np.random.default_rng(0)
    jittered = values + rng.uniform(-jitter, jitter, size=values.shape)
    return np.clip(jittered, 0.0, 1.0)


def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Compute AUC using ranks (pos=1, neg=0)."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    ranks = rankdata(scores)
    pos_ranks = ranks[labels == 1]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled = ((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b) / (
        len(group_a) + len(group_b) - 2
    )
    if pooled <= 0:
        return float("nan")
    return float((np.mean(group_b) - np.mean(group_a)) / np.sqrt(pooled))


def summarize_tp_fp(tp_vals: List[float], fp_vals: List[float]) -> dict:
    tp = np.array(tp_vals, dtype=float)
    fp = np.array(fp_vals, dtype=float)
    return {
        "tp_count": int(len(tp)),
        "fp_count": int(len(fp)),
        "tp_mean": float(np.mean(tp)) if len(tp) else float("nan"),
        "fp_mean": float(np.mean(fp)) if len(fp) else float("nan"),
        "tp_median": float(np.median(tp)) if len(tp) else float("nan"),
        "fp_median": float(np.median(fp)) if len(fp) else float("nan"),
        "tp_std": float(np.std(tp)) if len(tp) else float("nan"),
        "fp_std": float(np.std(fp)) if len(fp) else float("nan"),
        "fp_minus_tp_mean": float(np.mean(fp) - np.mean(tp)) if len(tp) and len(fp) else float("nan"),
        "auc_fp_vs_tp": auc_from_scores(fp, tp),
        "cohen_d_fp_vs_tp": cohen_d(tp, fp),
    }


def label_matches(record: dict, label_filter: Optional[str]) -> bool:
    if label_filter is None:
        return True
    label = record.get("label")
    if isinstance(label, str) and label.lower() == label_filter.lower():
        return True
    if label_filter.isdigit():
        label_index = record.get("label_index")
        if label_index is not None and int(label_index) == int(label_filter):
            return True
    return False


def extract_tp_fp_entropy_per_box(
    records: List[dict],
    entropy_kind: str,
    label_filter: Optional[str],
    score_min: Optional[float],
    score_max: Optional[float],
    require_entropy_included: bool,
) -> Tuple[List[float], List[float], Dict[str, int]]:
    counts = {
        "total": 0,
        "missing_tp_fp": 0,
        "missing_entropy": 0,
        "filtered_entropy_included": 0,
        "filtered_score": 0,
        "filtered_label": 0,
    }
    entropy_key = f"entropy_{entropy_kind}"
    tp_vals: List[float] = []
    fp_vals: List[float] = []

    for rec in records:
        counts["total"] += 1
        tp_fp = rec.get("tp_fp")
        if tp_fp not in ("tp", "fp"):
            counts["missing_tp_fp"] += 1
            continue

        entropy_val = rec.get(entropy_key)
        if entropy_val is None or (isinstance(entropy_val, float) and math.isnan(entropy_val)):
            counts["missing_entropy"] += 1
            continue

        if require_entropy_included and not rec.get("entropy_included", False):
            counts["filtered_entropy_included"] += 1
            continue

        if not label_matches(rec, label_filter):
            counts["filtered_label"] += 1
            continue

        score_val = rec.get("score")
        if score_min is not None:
            if score_val is None or float(score_val) < score_min:
                counts["filtered_score"] += 1
                continue
        if score_max is not None:
            if score_val is None or float(score_val) > score_max:
                counts["filtered_score"] += 1
                continue

        if tp_fp == "tp":
            tp_vals.append(float(entropy_val))
        else:
            fp_vals.append(float(entropy_val))

    return tp_vals, fp_vals, counts


def find_threshold_key(ap_dict: dict, threshold: float) -> Optional[str]:
    for key in ap_dict.keys():
        try:
            if abs(float(key) - threshold) < 1e-6:
                return key
        except (TypeError, ValueError):
            continue
    return None


def available_thresholds(frame_records: List[dict], match_mode: str) -> List[float]:
    for record in frame_records:
        metrics = record.get("metrics", {})
        matching = metrics.get(match_mode, {})
        for cls_data in matching.values():
            ap_dict = cls_data.get("ap", {})
            if ap_dict:
                return sorted({float(k) for k in ap_dict.keys()})
    return []


def compute_frame_map(
    metrics: dict,
    match_mode: str,
    threshold: float,
    label_filter: Optional[str],
) -> Optional[float]:
    matching = metrics.get(match_mode, {})
    aps: List[float] = []
    for cls_name, cls_data in matching.items():
        if label_filter and cls_name.lower() != label_filter.lower():
            continue
        ap_dict = cls_data.get("ap", {})
        ap_key = find_threshold_key(ap_dict, threshold)
        if not ap_key:
            continue
        ap_val = ap_dict.get(ap_key)
        if ap_val is None:
            continue
        if isinstance(ap_val, float) and math.isnan(ap_val):
            continue
        aps.append(float(ap_val))
    if not aps:
        return None
    return float(np.mean(aps))


def aggregate_map_by_group(
    frame_records: List[dict],
    scope: str,
    match_mode: str,
    threshold: float,
    label_filter: Optional[str],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    counts = {
        "total": 0,
        "missing_group": 0,
        "missing_metrics": 0,
        "missing_map": 0,
    }
    groups: Dict[str, List[float]] = {}

    for record in frame_records:
        counts["total"] += 1
        scene_id = record.get("scene_id")
        frame_id = record.get("frame_id")
        if not scene_id or (scope == "frame" and frame_id is None):
            counts["missing_group"] += 1
            continue
        metrics = record.get("metrics")
        if not metrics:
            counts["missing_metrics"] += 1
            continue
        map_val = compute_frame_map(metrics, match_mode, threshold, label_filter)
        if map_val is None:
            counts["missing_map"] += 1
            continue
        group_key = scene_id if scope == "scene" else f"{scene_id}/{frame_id}"
        groups.setdefault(group_key, []).append(map_val)

    group_map = {key: float(np.mean(vals)) for key, vals in groups.items() if vals}
    return group_map, counts


def aggregate_entropy_by_group(
    records: List[dict],
    scope: str,
    entropy_kind: str,
    entropy_agg: str,
    label_filter: Optional[str],
    score_min: Optional[float],
    score_max: Optional[float],
    require_entropy_included: bool,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    counts = {
        "total": 0,
        "missing_group": 0,
        "missing_entropy": 0,
        "filtered_entropy_included": 0,
        "filtered_score": 0,
        "filtered_label": 0,
    }
    entropy_key = f"entropy_{entropy_kind}"
    groups: Dict[str, List[float]] = {}

    for rec in records:
        counts["total"] += 1
        scene_id = rec.get("scene_id")
        frame_id = rec.get("frame_id")
        if not scene_id or (scope == "frame" and frame_id is None):
            counts["missing_group"] += 1
            continue

        entropy_val = rec.get(entropy_key)
        if entropy_val is None or (isinstance(entropy_val, float) and math.isnan(entropy_val)):
            counts["missing_entropy"] += 1
            continue

        if require_entropy_included and not rec.get("entropy_included", False):
            counts["filtered_entropy_included"] += 1
            continue

        if not label_matches(rec, label_filter):
            counts["filtered_label"] += 1
            continue

        score_val = rec.get("score")
        if score_min is not None:
            if score_val is None or float(score_val) < score_min:
                counts["filtered_score"] += 1
                continue
        if score_max is not None:
            if score_val is None or float(score_val) > score_max:
                counts["filtered_score"] += 1
                continue

        group_key = scene_id if scope == "scene" else f"{scene_id}/{frame_id}"
        groups.setdefault(group_key, []).append(float(entropy_val))

    agg_fn = np.mean if entropy_agg == "mean" else np.max
    group_entropy = {key: float(agg_fn(vals)) for key, vals in groups.items() if vals}
    return group_entropy, counts


def combine_entropy_map_stats(
    entropy_by_group: Dict[str, float],
    map_by_group: Dict[str, float],
) -> List[Tuple[float, float, str]]:
    stats: List[Tuple[float, float, str]] = []
    for group_key, entropy_val in entropy_by_group.items():
        map_val = map_by_group.get(group_key)
        if map_val is None or (isinstance(map_val, float) and math.isnan(map_val)):
            continue
        stats.append((float(entropy_val), float(map_val), group_key))
    return stats


def aggregate_group_stats(
    records: List[dict],
    scope: str,
    entropy_kind: str,
    entropy_agg: str,
    label_filter: Optional[str],
    score_min: Optional[float],
    score_max: Optional[float],
    require_entropy_included: bool,
) -> Tuple[List[Tuple[float, float, str]], List[float], List[float], Dict[str, int]]:
    counts = {
        "total": 0,
        "missing_group": 0,
        "missing_tp_fp": 0,
        "missing_entropy": 0,
        "filtered_entropy_included": 0,
        "filtered_score": 0,
        "filtered_label": 0,
    }
    entropy_key = f"entropy_{entropy_kind}"
    groups: Dict[str, Dict[str, List[float]]] = {}

    for rec in records:
        counts["total"] += 1
        scene_id = rec.get("scene_id")
        frame_id = rec.get("frame_id")
        if not scene_id or (scope == "frame" and frame_id is None):
            counts["missing_group"] += 1
            continue

        tp_fp = rec.get("tp_fp")
        if tp_fp not in ("tp", "fp"):
            counts["missing_tp_fp"] += 1
            continue

        entropy_val = rec.get(entropy_key)
        if entropy_val is None or (isinstance(entropy_val, float) and math.isnan(entropy_val)):
            counts["missing_entropy"] += 1
            continue

        if require_entropy_included and not rec.get("entropy_included", False):
            counts["filtered_entropy_included"] += 1
            continue

        if not label_matches(rec, label_filter):
            counts["filtered_label"] += 1
            continue

        score_val = rec.get("score")
        if score_min is not None:
            if score_val is None or float(score_val) < score_min:
                counts["filtered_score"] += 1
                continue
        if score_max is not None:
            if score_val is None or float(score_val) > score_max:
                counts["filtered_score"] += 1
                continue

        group_key = scene_id if scope == "scene" else f"{scene_id}/{frame_id}"
        group = groups.setdefault(group_key, {"all": [], "tp": [], "fp": []})
        group["all"].append(float(entropy_val))
        if tp_fp == "tp":
            group["tp"].append(float(entropy_val))
        else:
            group["fp"].append(float(entropy_val))

    agg_fn = np.mean if entropy_agg == "mean" else np.max
    stats: List[Tuple[float, float, str]] = []
    tp_agg_vals: List[float] = []
    fp_agg_vals: List[float] = []

    for group_key, group in groups.items():
        total = len(group["tp"]) + len(group["fp"])
        if total == 0 or not group["all"]:
            continue
        entropy_val = float(agg_fn(group["all"]))
        precision = len(group["tp"]) / total
        stats.append((entropy_val, float(precision), group_key))
        if group["tp"]:
            tp_agg_vals.append(float(agg_fn(group["tp"])))
        if group["fp"]:
            fp_agg_vals.append(float(agg_fn(group["fp"])))

    return stats, tp_agg_vals, fp_agg_vals, counts


def plot_entropy_vs_precision(
    stats: List[Tuple[float, float, str]],
    output: Path,
    entropy_kind: str,
    entropy_agg: str,
    scope: str,
    precision_jitter: float,
) -> None:
    entropies = np.array([s[0] for s in stats], dtype=float)
    precision = np.array([s[1] for s in stats], dtype=float)
    corr = spearman_corr(entropies, precision) if len(stats) > 1 else float("nan")
    precision_plot = apply_precision_jitter(precision, precision_jitter)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. Install matplotlib to generate a figure.")
        print(f"Spearman rank correlation: {corr:.4f}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(entropies, precision_plot, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, edgecolors="none")
    ax.set_xlabel(f"{entropy_kind} Entropy ({entropy_agg})")
    ax.set_ylabel(f"{scope.title()} Precision")
    ax.set_title(f"Entropy vs. Precision (Spearman={corr:.3f})")
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")
    print(f"Spearman rank correlation: {corr:.4f}")


def plot_entropy_vs_map(
    stats: List[Tuple[float, float, str]],
    output: Path,
    entropy_kind: str,
    entropy_agg: str,
    scope: str,
) -> None:
    entropies = np.array([s[0] for s in stats], dtype=float)
    maps = np.array([s[1] for s in stats], dtype=float)
    corr = spearman_corr(entropies, maps) if len(stats) > 1 else float("nan")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. Install matplotlib to generate a figure.")
        print(f"Spearman rank correlation: {corr:.4f}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(entropies, maps, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, edgecolors="none")
    ax.set_xlabel(f"{entropy_kind} Entropy ({entropy_agg})")
    ax.set_ylabel(f"{scope.title()} mAP")
    ax.set_title(f"Entropy vs. mAP (Spearman={corr:.3f})")
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")
    print(f"Spearman rank correlation: {corr:.4f}")


def plot_tp_fp_hist(
    tp_vals: List[float],
    fp_vals: List[float],
    output: Path,
    entropy_kind: str,
    bins: int,
    density: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. Install matplotlib to generate a figure.")
        return

    tp = np.array(tp_vals, dtype=float)
    fp = np.array(fp_vals, dtype=float)
    if len(tp) == 0 or len(fp) == 0:
        raise SystemExit("No valid TP/FP entropies found after aggregation.")

    all_vals = np.concatenate([tp, fp])
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(tp, bins=bin_edges, alpha=0.6, label=f"TP (n={len(tp)})", density=density)
    ax.hist(fp, bins=bin_edges, alpha=0.6, label=f"FP (n={len(fp)})", density=density)
    ax.set_xlabel(f"{entropy_kind} Entropy")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title("Entropy Distribution: TP vs FP (per-box)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def build_combinations(entropy_kind: Optional[str], entropy_agg: Optional[str]) -> List[Tuple[str, str]]:
    kinds = [entropy_kind] if entropy_kind else list(ENTROPY_KINDS)
    aggs = [entropy_agg] if entropy_agg else list(ENTROPY_AGGS)
    return [(kind, agg) for kind in kinds for agg in aggs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze prediction_results.jsonl and generate entropy plots."
    )
    parser.add_argument(
        "--prediction-results",
        type=Path,
        nargs="+",
        required=True,
        help="One or more prediction_results.jsonl files (all records are accumulated)",
    )
    parser.add_argument(
        "--frame-summary",
        type=Path,
        nargs="*",
        default=None,
        help="Optional frame_summary.jsonl files to enable mAP plots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("entropy_plots"),
        help="Directory to write plots",
    )
    parser.add_argument(
        "--scope",
        choices=["scene", "frame"],
        default=None,
        help="Aggregate by a single scope (deprecated; use --scopes)",
    )
    parser.add_argument(
        "--scopes",
        choices=["scene", "frame"],
        nargs="+",
        default=None,
        help="Scopes to run (default: scene and frame)",
    )
    parser.add_argument(
        "--match-mode",
        default="center_distance_bev",
        help="Match mode key inside frame summary (default: center_distance_bev)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Distance threshold to use for AP when computing mAP",
    )
    parser.add_argument(
        "--entropy-kind",
        choices=list(ENTROPY_KINDS),
        default=None,
        help="Entropy kind to plot (default: all kinds)",
    )
    parser.add_argument(
        "--entropy-agg",
        choices=list(ENTROPY_AGGS),
        default=None,
        help="Entropy aggregation to plot (default: mean and max)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins for TP/FP plot",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot density instead of counts in TP/FP histogram",
    )
    parser.add_argument(
        "--precision-jitter",
        type=float,
        default=None,
        help="Add vertical jitter to precision scatter (default: 0.015 for frame, 0 for scene)",
    )
    parser.add_argument(
        "--score-min",
        type=float,
        default=None,
        help="Minimum score to include predictions",
    )
    parser.add_argument(
        "--score-max",
        type=float,
        default=None,
        help="Maximum score to include predictions",
    )
    parser.add_argument(
        "--require-entropy-included",
        action="store_true",
        help="Only include predictions where entropy_included is true",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Filter to a single label name (e.g., car) or numeric label_index",
    )
    args = parser.parse_args()

    records = load_prediction_results(args.prediction_results)
    if not records:
        raise SystemExit("No records loaded from prediction_results.jsonl")

    print("TP/FP histograms use per-box entropies; scope and entropy-agg do not apply.")
    hist_kinds = [args.entropy_kind] if args.entropy_kind else list(ENTROPY_KINDS)
    for entropy_kind in hist_kinds:
        tp_vals, fp_vals, counts = extract_tp_fp_entropy_per_box(
            records,
            entropy_kind=entropy_kind,
            label_filter=args.label,
            score_min=args.score_min,
            score_max=args.score_max,
            require_entropy_included=args.require_entropy_included,
        )
        if not tp_vals or not fp_vals:
            print(
                f"Skipping TP/FP hist for {entropy_kind}: insufficient TP/FP data. "
                f"missing_tp_fp={counts['missing_tp_fp']} missing_entropy={counts['missing_entropy']}"
            )
            continue
        summary = summarize_tp_fp(tp_vals, fp_vals)
        print(
            f"TP/FP summary {entropy_kind}: "
            f"tp_mean={summary['tp_mean']:.4f} fp_mean={summary['fp_mean']:.4f} "
            f"fp_minus_tp_mean={summary['fp_minus_tp_mean']:.4f} "
            f"auc_fp_vs_tp={summary['auc_fp_vs_tp']:.4f} "
            f"cohen_d_fp_vs_tp={summary['cohen_d_fp_vs_tp']:.4f}"
        )
        tp_fp_path = args.output_dir / f"tp_fp_entropy_box_{entropy_kind}.png"
        plot_tp_fp_hist(
            tp_vals,
            fp_vals,
            tp_fp_path,
            entropy_kind,
            bins=max(args.bins, 1),
            density=args.density,
        )

    if args.scopes:
        scopes = args.scopes
    elif args.scope:
        scopes = [args.scope]
    else:
        scopes = ["scene", "frame"]

    frame_records: List[dict] = []
    if args.frame_summary:
        frame_records = load_frame_summary(args.frame_summary)
        if not frame_records:
            print("No frame summary records loaded; skipping mAP plots.")
        else:
            available = available_thresholds(frame_records, args.match_mode)
            if available and args.threshold not in available:
                raise SystemExit(f"Threshold {args.threshold} not found. Available thresholds: {available}")
            label_for_map = None
            if args.label and not args.label.isdigit():
                label_for_map = args.label
            elif args.label and args.label.isdigit():
                print("Label filter is numeric; mAP plots use class names so label filter is ignored.")
            print(
                f"Loaded {len(frame_records)} frame summaries for mAP "
                f"(match_mode={args.match_mode}, threshold={args.threshold})"
            )

    combos = build_combinations(args.entropy_kind, args.entropy_agg)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scope in scopes:
        precision_jitter = args.precision_jitter
        if precision_jitter is None:
            precision_jitter = 0.015 if scope == "frame" else 0.0
        if precision_jitter:
            print(f"Applying precision jitter={precision_jitter:.4f} for {scope} plots (display only).")

        map_by_group: Dict[str, float] = {}
        if frame_records:
            label_for_map = None
            if args.label and not args.label.isdigit():
                label_for_map = args.label
            elif args.label and args.label.isdigit():
                print("Label filter is numeric; mAP plots use class names so label filter is ignored.")
            map_by_group, map_counts = aggregate_map_by_group(
                frame_records,
                scope=scope,
                match_mode=args.match_mode,
                threshold=args.threshold,
                label_filter=label_for_map,
            )
            print(
                f"[{scope}] mAP groups={len(map_by_group)} missing_metrics={map_counts['missing_metrics']} "
                f"missing_map={map_counts['missing_map']}"
            )

        for entropy_kind, entropy_agg in combos:
            entropy_by_group, entropy_counts = aggregate_entropy_by_group(
                records,
                scope=scope,
                entropy_kind=entropy_kind,
                entropy_agg=entropy_agg,
                label_filter=args.label,
                score_min=args.score_min,
                score_max=args.score_max,
                require_entropy_included=args.require_entropy_included,
            )

            if not entropy_by_group:
                print(
                    f"[{scope}] Skipping {entropy_kind}/{entropy_agg}: no entropy after filtering. "
                    f"missing_entropy={entropy_counts['missing_entropy']}"
                )
                continue

            stats, _, _, counts = aggregate_group_stats(
                records,
                scope=scope,
                entropy_kind=entropy_kind,
                entropy_agg=entropy_agg,
                label_filter=args.label,
                score_min=args.score_min,
                score_max=args.score_max,
                require_entropy_included=args.require_entropy_included,
            )

            print(
                f"[{scope}] {entropy_kind}/{entropy_agg}: groups={len(stats)} total_records={counts['total']} "
                f"missing_tp_fp={counts['missing_tp_fp']} missing_entropy={counts['missing_entropy']} "
                f"filtered_score={counts['filtered_score']} filtered_label={counts['filtered_label']} "
                f"filtered_entropy_included={counts['filtered_entropy_included']}"
            )

            scatter_path = args.output_dir / f"entropy_vs_precision_{scope}_{entropy_kind}_{entropy_agg}.png"
            if stats:
                plot_entropy_vs_precision(
                    stats,
                    scatter_path,
                    entropy_kind,
                    entropy_agg,
                    scope,
                    precision_jitter,
                )
            else:
                print(f"[{scope}] Skipping precision plot for {entropy_kind}/{entropy_agg}: no valid stats.")

            if map_by_group:
                stats_map = combine_entropy_map_stats(entropy_by_group, map_by_group)
                map_path = args.output_dir / f"entropy_vs_map_{scope}_{entropy_kind}_{entropy_agg}.png"
                if stats_map:
                    plot_entropy_vs_map(stats_map, map_path, entropy_kind, entropy_agg, scope)
                else:
                    print(f"[{scope}] Skipping mAP plot for {entropy_kind}/{entropy_agg}: no aligned groups.")


if __name__ == "__main__":
    main()
