"""
Local PTv3 inference helper for T4Dataset layouts.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PTV3_ROOT = Path(__file__).resolve().parents[1]
if str(PTV3_ROOT) not in sys.path:
    sys.path.insert(0, str(PTV3_ROOT))


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _has_t4_dataset_layout(dataset_root: Path) -> bool:
    return (
        (dataset_root / "annotation" / "sample_data.json").is_file()
        and (dataset_root / "annotation" / "category.json").is_file()
        and (dataset_root / "data" / "LIDAR_CONCAT").is_dir()
    )


def _resolve_data_root_and_lidar_dir(input_root: Path) -> tuple[Path, Path]:
    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")

    if _has_t4_dataset_layout(input_root):
        return input_root, input_root / "data" / "LIDAR_CONCAT"

    child_dirs = sorted(
        [path for path in input_root.iterdir() if path.is_dir() and not path.name.startswith(".")],
        key=lambda p: _natural_key(p.name),
    )
    candidates = [path for path in child_dirs if _has_t4_dataset_layout(path)]
    if candidates:
        version_dir = candidates[-1]
        return version_dir, version_dir / "data" / "LIDAR_CONCAT"

    raise FileNotFoundError(
        "Could not find a valid T4Dataset root with annotation/ + data/LIDAR_CONCAT. "
        f"Tried: {input_root} and its non-hidden child directories."
    )


def _build_inference_infos(
    data_root: Path, lidar_dir: Path, fallback_masks_dir: Path, keyframes_only: bool = False
) -> dict:
    annotation_dir = data_root / "annotation"
    with (annotation_dir / "sample_data.json").open("r", encoding="utf-8") as file:
        sample_data = json.load(file)

    lidarseg_path = annotation_dir / "lidarseg.json"
    if lidarseg_path.is_file():
        with lidarseg_path.open("r", encoding="utf-8") as file:
            lidarseg = json.load(file)
    else:
        lidarseg = []

    with (annotation_dir / "category.json").open("r", encoding="utf-8") as file:
        categories = json.load(file)

    raw_categories = {category["name"]: idx for idx, category in enumerate(categories)}
    lidarseg_by_token = {record["sample_data_token"]: record for record in lidarseg}

    lidar_records = sorted(
        [
            record
            for record in sample_data
            if str(record.get("filename", "")).startswith("data/LIDAR_CONCAT/")
            and bool(record.get("token"))
            # keyframes_only: build_pseudo only consumes keyframe tokens, and there are
            # ~10 LiDAR sweeps per keyframe — restricting here is a ~10x compute saving.
            and (not keyframes_only or bool(record.get("is_key_frame")))
        ],
        key=lambda record: _natural_key(Path(record["filename"]).name),
    )

    fallback_masks_dir.mkdir(parents=True, exist_ok=True)
    data_list = []
    missing_tokens = []
    for lidar_record in lidar_records:
        token = str(lidar_record["token"])
        mask_record = lidarseg_by_token.get(token)
        if mask_record is None:
            missing_tokens.append(token)
            lidar_path = data_root / str(lidar_record["filename"])
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
            dummy_mask = fallback_masks_dir / f"{token}.bin"
            np.zeros((points.shape[0],), dtype=np.uint8).tofile(dummy_mask)
            mask_path = str(dummy_mask)
        else:
            mask_path = str(mask_record["filename"])

        data_list.append(
            {
                "lidar_points": {"lidar_path": str(lidar_record["filename"])},
                "pts_semantic_mask_path": mask_path,
                "pts_semantic_mask_categories": raw_categories,
                "token": token,
            }
        )

    if missing_tokens:
        print(
            "WARN: Missing lidarseg entries for some LIDAR sample_data tokens; "
            "falling back to ignore-index labels for inference-only runs. "
            f"First few: {missing_tokens[:5]}"
        )

    return {"data_list": data_list}


def _default_output_dir(input_root: Path, data_root: Path, cfg) -> Path:
    model_name = "ptv3"
    if "inference_model_name" in cfg:
        model_name = str(cfg.inference_model_name)
    if data_root == input_root:
        return data_root / "model_outputs" / model_name
    return input_root / data_root.name / "model_outputs" / model_name


def _cleanup_output_dir(output_dir: Path) -> None:
    for artifact_name in ("inference_infos.pkl", "model_best.normalized.pth", "test.log"):
        (output_dir / artifact_name).unlink(missing_ok=True)
    result_dir = output_dir / "result"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    fallback_masks_dir = output_dir / ".ptv3_local_infer_masks"
    if fallback_masks_dir.exists():
        shutil.rmtree(fallback_masks_dir)


def _promote_result_files(run_dir: Path, output_dir: Path) -> None:
    result_dir = run_dir / "result"
    if not result_dir.is_dir():
        raise FileNotFoundError(f"PTv3 result directory not found: {result_dir}")
    for result_file in sorted(result_dir.glob("*.npz")):
        destination = output_dir / result_file.name
        destination.unlink(missing_ok=True)
        shutil.move(str(result_file), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PTv3 inference on a T4Dataset root")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--keyframes-only",
        action="store_true",
        help="Only run inference on keyframe LIDAR_CONCAT frames (is_key_frame=true).",
    )
    args = parser.parse_args()

    import utils.comm as comm
    from engines.test import TESTERS
    from utils.config import Config

    input_root = Path(args.input).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    config_for_load = config_path
    staged_config_path = None
    configs_root = PTV3_ROOT / "configs"
    try:
        config_path.relative_to(configs_root)
    except ValueError:
        configs_root.mkdir(parents=True, exist_ok=True)
        with config_path.open("r", encoding="utf-8") as src:
            content = src.read()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".py", prefix="local_infer_", dir=str(configs_root), delete=False
        ) as dst:
            dst.write(content)
            config_for_load = Path(dst.name)
            staged_config_path = config_for_load

    try:
        data_root, lidar_dir = _resolve_data_root_and_lidar_dir(input_root)
        cfg = Config.fromfile(str(config_for_load))
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else _default_output_dir(input_root, data_root, cfg)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_output_dir(output_dir)

        fallback_masks_dir = output_dir / ".ptv3_local_infer_masks"
        info_dict = _build_inference_infos(
            data_root, lidar_dir, fallback_masks_dir, keyframes_only=args.keyframes_only
        )
        info_path = output_dir / "inference_infos.pkl"
        with info_path.open("wb") as file:
            pickle.dump(info_dict, file)

        # Run dir holds the tensorboard log + transient result npz; keep it on LOCAL disk
        # (not the NAS data_root) to avoid slow NFS writes and rmtree cleanup races.
        run_base = os.environ.get("PP3D_RUN_TMP", "") or None  # None -> system temp (/tmp)
        with tempfile.TemporaryDirectory(dir=run_base, prefix="ptv3_local_infer_") as run_dir_str:
            run_dir = Path(run_dir_str)
            cfg.weight = str(checkpoint_path)
            cfg.save_path = str(run_dir)
            cfg.data_root = str(data_root)
            cfg.data.test.data_root = str(data_root)
            cfg.data.test.info_paths = [str(info_path)]
            cfg.info_paths_test = [str(info_path)]
            cfg.inference_only = True
            # Throughput overrides (see engines/test.py): batch fragments per forward and use
            # AMP. Env-tunable for sweeps; defaults chosen for A100-80GB.
            cfg.fragment_batch_size = int(os.environ.get("PP3D_FRAG_BS", "16"))
            cfg.enable_amp = os.environ.get("PP3D_AMP", "1") == "1"
            cfg.amp_dtype = os.environ.get("PP3D_AMP_DTYPE", "fp16")
            # 1-partition-per-aug TTA: 1 forward per aug + inverse scatter (≈4-5x fewer
            # fragments than full multi-partition). On by default; PP3D_SINGLE_PART=0 disables.
            if os.environ.get("PP3D_SINGLE_PART", "1") == "1":
                try:
                    cfg.data.test.test_cfg.voxelize.single_partition = True
                except Exception:
                    pass

            if "resume" not in cfg:
                cfg.resume = False
            if "find_unused_parameters" not in cfg:
                cfg.find_unused_parameters = False
            if "show" not in cfg:
                cfg.show = False
            if "empty_cache" not in cfg:
                cfg.empty_cache = False
            if "keywords" not in cfg:
                cfg.keywords = ""
            if "replacement" not in cfg:
                cfg.replacement = ""
            if "test" not in cfg:
                cfg.test = dict(type="SemSegTester", verbose=True)

            world_size = comm.get_world_size()
            batch_size_test = cfg.batch_size_test if "batch_size_test" in cfg and cfg.batch_size_test is not None else 1
            if batch_size_test < world_size:
                batch_size_test = world_size
            cfg.batch_size_test_per_gpu = max(1, batch_size_test // world_size)

            test_cfg = cfg.test
            tester_type = test_cfg["type"] if isinstance(test_cfg, dict) else test_cfg.type
            tester = TESTERS.build(dict(type=tester_type, cfg=cfg))
            tester.test()
            _promote_result_files(run_dir, output_dir)
    finally:
        if staged_config_path is not None and staged_config_path.exists():
            staged_config_path.unlink()


if __name__ == "__main__":
    main()
