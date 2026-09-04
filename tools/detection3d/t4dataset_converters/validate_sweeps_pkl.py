"""Validation utilities for t4dataset info pkls with (optionally bidirectional) lidar sweeps.

Standalone: requires only numpy (pkls are plain pickle files). Three modes:

  --compare OLD.pkl NEW.pkl   Deep-diff two pkls for bit-identity (regression check).
  --check PKL                 Verify sweep invariants (ordering, frame_offset signs,
                              time lags, transform consistency, file existence).
  --dump-cloud PKL            Accumulate key frame + all sweeps of one sample into a
                              single cloud (npz + ply, colored by time lag) using
                              lidar_points.lidar2ego as the sweep->key-lidar transform.

Examples:
  python tools/detection3d/t4dataset_converters/validate_sweeps_pkl.py \
      --compare old_infos_train.pkl new_infos_train.pkl
  python tools/detection3d/t4dataset_converters/validate_sweeps_pkl.py \
      --check infos_train.pkl --data-root ./data/t4dataset
  python tools/detection3d/t4dataset_converters/validate_sweeps_pkl.py \
      --dump-cloud infos_train.pkl --index 100 --data-root ./data/t4dataset --out /tmp/cloud
"""

import argparse
import os
import os.path as osp
import pickle
import sys
from collections import Counter

import numpy as np

LIDAR_HZ = 10.0


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# --compare
# ---------------------------------------------------------------------------


def deep_diff(a, b, path, diffs, max_diffs):
    if len(diffs) >= max_diffs:
        return
    if type(a) is not type(b):
        diffs.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return
    if isinstance(a, dict):
        for key in a.keys() | b.keys():
            if key not in a:
                diffs.append(f"{path}.{key}: missing in OLD")
            elif key not in b:
                diffs.append(f"{path}.{key}: missing in NEW")
            else:
                deep_diff(a[key], b[key], f"{path}.{key}", diffs, max_diffs)
            if len(diffs) >= max_diffs:
                return
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
            return
        # gt_attrs is semantically a set: merged objects build it via list(set(...)),
        # whose order varies with PYTHONHASHSEED across processes (t4converter.py:545).
        if path.endswith(".gt_attrs") and all(isinstance(x, str) for x in a):
            a, b = sorted(a), sorted(b)
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", diffs, max_diffs)
            if len(diffs) >= max_diffs:
                return
    elif isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            diffs.append(f"{path}: ndarray mismatch (dtype/shape/values)")
    elif isinstance(a, float):
        if a != b and not (np.isnan(a) and np.isnan(b)):
            diffs.append(f"{path}: {a!r} != {b!r}")
    else:
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")


def run_compare(old_path, new_path, max_diffs):
    old, new = load_pkl(old_path), load_pkl(new_path)
    # The --version CLI arg is embedded in metainfo; different version strings are expected.
    for pkl in (old, new):
        if isinstance(pkl, dict):
            pkl.get("metainfo", {}).pop("version", None)
    diffs = []
    deep_diff(old, new, "", diffs, max_diffs)
    if diffs:
        print(f"NOT IDENTICAL ({len(diffs)} difference(s) shown, cap {max_diffs}):")
        for d in diffs:
            print(f"  {d}")
        return 1
    print("IDENTICAL (ignoring metainfo.version)")
    return 0


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------


def check_transforms(sweep, errors, prefix):
    lidar2ego = np.asarray(sweep["lidar_points"]["lidar2ego"], dtype=np.float64)
    lidar2sensor = np.asarray(sweep["lidar_points"]["lidar2sensor"], dtype=np.float64)
    residual = lidar2ego @ lidar2sensor - np.eye(4)
    if np.abs(residual).max() > 1e-3:
        errors.append(f"{prefix}: lidar2ego @ lidar2sensor deviates from I by {np.abs(residual).max():.2e}")
    translation_norm = float(np.linalg.norm(lidar2ego[:3, 3]))
    if translation_norm > 60.0:
        errors.append(f"{prefix}: implausible sweep->key translation {translation_norm:.1f} m")


def run_check(pkl_path, data_root):
    pkl = load_pkl(pkl_path)
    infos = pkl["data_list"]
    errors = []
    warnings = []
    past_counts, future_counts = Counter(), Counter()
    has_offset = 0

    for idx, info in enumerate(infos):
        key_ts = float(info["timestamp"])
        sweeps = info.get("lidar_sweeps", [])
        lags = [key_ts - float(s["timestamp"]) for s in sweeps]
        prefix = f"info[{idx}]"

        abs_lags = [abs(lag) for lag in lags]
        # Tolerance: the converter sorts by the lidar sample_data timestamp, while
        # info["timestamp"] is the sample timestamp (differs by ~ms) — near-ties between
        # +k and -k sweeps may legitimately appear swapped when measured from here.
        if any(b < a - 0.02 for a, b in zip(abs_lags, abs_lags[1:])):
            errors.append(f"{prefix}: lidar_sweeps not sorted by |time_lag|: {[round(l, 3) for l in lags]}")

        n_past = sum(1 for lag in lags if lag > 0)
        n_future = sum(1 for lag in lags if lag < 0)
        past_counts[n_past] += 1
        future_counts[n_future] += 1

        for j, (sweep, lag) in enumerate(zip(sweeps, lags)):
            sweep_prefix = f"{prefix}.sweep[{j}]"
            if abs(lag) > 5.0:
                errors.append(f"{sweep_prefix}: |time_lag| {lag:.3f} s implausible (unit bug?)")
            offset = sweep.get("frame_offset")
            if offset is not None:
                has_offset += 1
                if offset == 0 or (offset > 0) != (lag > 0):
                    errors.append(f"{sweep_prefix}: frame_offset {offset} inconsistent with time_lag {lag:.3f}")
                expected = abs(offset) / LIDAR_HZ
                if abs(abs(lag) - expected) > 0.05 * max(1, abs(offset)):
                    warnings.append(
                        f"{sweep_prefix}: |time_lag| {abs(lag):.3f} vs expected ~{expected:.1f} s "
                        f"for frame_offset {offset}"
                    )
            check_transforms(sweep, errors, sweep_prefix)
            if data_root:
                sweep_path = osp.join(data_root, sweep["lidar_points"]["lidar_path"])
                if not osp.exists(sweep_path):
                    errors.append(f"{sweep_prefix}: missing file {sweep_path}")

    print(f"{len(infos)} infos in {pkl_path}")
    print(f"past sweep count histogram   {{count: infos}}: {dict(sorted(past_counts.items()))}")
    print(f"future sweep count histogram {{count: infos}}: {dict(sorted(future_counts.items()))}")
    print(f"sweep entries with frame_offset: {has_offset}")
    for w in warnings[:20]:
        print(f"WARN  {w}")
    if len(warnings) > 20:
        print(f"WARN  ... {len(warnings) - 20} more warnings")
    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for e in errors[:50]:
            print(f"  {e}")
        return 1
    print("CHECK PASSED")
    return 0


# ---------------------------------------------------------------------------
# --dump-cloud
# ---------------------------------------------------------------------------


def load_points(path, num_features):
    return np.fromfile(path, dtype=np.float32).reshape(-1, num_features)


def run_dump_cloud(pkl_path, index, data_root, out_dir):
    pkl = load_pkl(pkl_path)
    info = pkl["data_list"][index]
    key_ts = float(info["timestamp"])
    num_features = int(info["lidar_points"].get("num_pts_feats") or 5)

    key_points = load_points(osp.join(data_root, info["lidar_points"]["lidar_path"]), num_features)
    xyz = [key_points[:, :3]]
    lag = [np.zeros(len(key_points), dtype=np.float32)]

    for sweep in info.get("lidar_sweeps", []):
        points = load_points(osp.join(data_root, sweep["lidar_points"]["lidar_path"]), num_features)
        # lidar2ego is (despite the name) the full sweep-lidar -> key-lidar transform.
        matrix = np.asarray(sweep["lidar_points"]["lidar2ego"], dtype=np.float64)
        transformed = points[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
        xyz.append(transformed.astype(np.float32))
        lag.append(np.full(len(points), key_ts - float(sweep["timestamp"]), dtype=np.float32))

    xyz = np.concatenate(xyz)
    lag = np.concatenate(lag)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = osp.join(out_dir, f"accumulated_{index}.npz")
    np.savez_compressed(npz_path, xyz=xyz, time_lag=lag)

    # PLY colored by lag: past = blue..cyan, key = white, future = red..yellow.
    max_abs = max(float(np.abs(lag).max()), 1e-6)
    ratio = np.clip(np.abs(lag) / max_abs, 0.0, 1.0)
    rgb = np.zeros((len(lag), 3), dtype=np.uint8)
    past, future, key = lag > 1e-6, lag < -1e-6, np.abs(lag) <= 1e-6
    rgb[key] = 255
    rgb[past, 2] = 255
    rgb[past, 1] = (255 * (1 - ratio[past])).astype(np.uint8)
    rgb[future, 0] = 255
    rgb[future, 1] = (255 * ratio[future]).astype(np.uint8)
    ply_path = osp.join(out_dir, f"accumulated_{index}.ply")
    with open(ply_path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(xyz)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")

    lags = sorted({round(float(v), 3) for v in np.unique(lag)})
    print(f"{len(xyz)} points ({len(info.get('lidar_sweeps', []))} sweeps + key), time lags: {lags}")
    print(f"wrote {npz_path}")
    print(f"wrote {ply_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"), help="deep-diff two pkls")
    mode.add_argument("--check", metavar="PKL", help="verify sweep invariants")
    mode.add_argument("--dump-cloud", metavar="PKL", help="accumulate one sample's sweeps to npz/ply")
    parser.add_argument("--data-root", default=None, help="dataset root for resolving lidar paths")
    parser.add_argument("--index", type=int, default=0, help="sample index for --dump-cloud")
    parser.add_argument("--out", default=".", help="output directory for --dump-cloud")
    parser.add_argument("--max-diffs", type=int, default=20, help="max differences reported by --compare")
    args = parser.parse_args()

    if args.compare:
        return run_compare(args.compare[0], args.compare[1], args.max_diffs)
    if args.check:
        return run_check(args.check, args.data_root)
    if not args.data_root:
        parser.error("--dump-cloud requires --data-root")
    return run_dump_cloud(args.dump_cloud, args.index, args.data_root, args.out)


if __name__ == "__main__":
    sys.exit(main())
