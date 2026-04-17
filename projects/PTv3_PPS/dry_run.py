"""
Dry-run: verify data loading + model forward pass without full training.

Checks:
  1. T4Dataset loads train/val splits from NAS correctly
  2. DataLoader produces batches with expected keys/shapes
  3. PPSSegmentor forward pass runs (train mode: returns loss; eval: returns seg_logits)
  4. Prototype warm-start loads correctly from model_best.pth

Usage (inside autoware-ml-integration container):
    cd /workspace/projects/PTv3_PPS
    python dry_run.py

If all checks pass, real training should work.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PTv3 = os.path.abspath(os.path.join(_HERE, "..", "PTv3"))
sys.path.insert(0, _HERE)
sys.path.insert(0, _PTv3)

import torch
import models  # noqa: registers PTv3 models
import pps_models  # noqa: registers PPSSegmentor

_INFO_BASE = "/mnt/t4dataset_ro/info/lidarseg"
_DATA_ROOT = "/mnt/t4dataset_ro"
_WEIGHT_PATH = "/home/linick/Downloads/ptv3_train/reproduce_only_gt/model_best.pth"

CLASS_MAPPING = dict(
    drivable_surface=0, other_flat_surface=1, sidewalk=2, manmade=3, vegetation=4,
    car=5, bus=6, emergency_vehicle=7, train=8, truck=9, tractor_unit=10,
    semi_trailer=11, construction_vehicle=12, forklift=13, kart=14, motorcycle=15,
    bicycle=16, pedestrian=17, personal_mobility=18, animal=19,
    pushable_pullable=20, traffic_cone=21, stroller=22, debris=23,
    other_stuff=24, noise=25, ghost_point=25, out_of_sync=-1, unpainted=-1,
)
GRID_SIZE = 0.1
POINT_CLOUD_RANGE = [-102.4, -102.4, -2.8, 102.4, 102.4, 10.0]
NUM_CLASSES = 26
IGNORE_INDEX = -1


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def test_dataset():
    print("\n=== 1. Dataset loading ===")
    from datasets.t4dataset import T4Dataset  # PTv3's T4Dataset wrapper

    transform = [
        dict(type="PointClip", point_cloud_range=POINT_CLOUD_RANGE),
        dict(type="GridSample", grid_size=GRID_SIZE, hash_type="fnv", mode="train",
             keys=("coord", "strength", "segment"), return_grid_coord=True),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "segment"),
             feat_keys=("coord", "strength")),
    ]

    ok = True
    for split, info_file in [
        ("train", "t4dataset_j6gen2_lidarseg_infos_train.pkl"),
        ("val",   "t4dataset_j6gen2_lidarseg_infos_val.pkl"),
    ]:
        try:
            ds = T4Dataset(
                split=split,
                data_root=_DATA_ROOT,
                info_paths=[os.path.join(_INFO_BASE, info_file)],
                transform=transform,
                test_mode=False,
                ignore_index=IGNORE_INDEX,
                class_mapping=CLASS_MAPPING,
            )
            n = len(ds)
            sample = ds[0]
            keys = list(sample.keys())
            coord_shape = sample["coord"].shape
            seg_shape = sample["segment"].shape
            ok &= check(
                f"{split}: {n} frames, coord={coord_shape}, segment={seg_shape}",
                n > 0 and coord_shape[1] == 3 and seg_shape[0] == coord_shape[0],
            )
        except Exception as e:
            check(f"{split}: load failed", False, str(e))
            ok = False
    return ok


def test_dataloader():
    print("\n=== 2. DataLoader (2 batches) ===")
    from datasets.t4dataset import T4Dataset
    from torch.utils.data import DataLoader
    from datasets.utils import point_collate_fn  # PTv3 collate

    transform = [
        dict(type="PointClip", point_cloud_range=POINT_CLOUD_RANGE),
        dict(type="GridSample", grid_size=GRID_SIZE, hash_type="fnv", mode="train",
             keys=("coord", "strength", "segment"), return_grid_coord=True),
        dict(type="SphereCrop", point_max=32000, mode="random"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "segment"),
             feat_keys=("coord", "strength")),
    ]

    try:
        ds = T4Dataset(
            split="train",
            data_root=_DATA_ROOT,
            info_paths=[os.path.join(_INFO_BASE, "t4dataset_j6gen2_lidarseg_infos_train.pkl")],
            transform=transform,
            test_mode=False,
            ignore_index=IGNORE_INDEX,
            class_mapping=CLASS_MAPPING,
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2,
                            collate_fn=point_collate_fn)
        for i, batch in enumerate(loader):
            if i >= 2:
                break
            check(
                f"batch {i}: feat={batch['feat'].shape}, segment={batch['segment'].shape}",
                batch["feat"].shape[1] == 4 and batch["segment"].shape[0] > 0,
            )
        return True
    except Exception as e:
        check("dataloader", False, str(e))
        return False


def test_model_forward():
    print("\n=== 3. Model forward pass ===")
    from models.builder import build_model

    model_cfg = dict(
        type="PPSSegmentor",
        num_classes=NUM_CLASSES,
        backbone_out_channels=64,
        freeze_backbone=True,
        weight_path=_WEIGHT_PATH if os.path.exists(_WEIGHT_PATH) else None,
        backbone=dict(
            type="PT-v3m1",
            in_channels=4,
            order=["z", "z-trans", "hilbert", "hilbert-trans"],
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True, enable_rpe=False,
            enable_flash=True, upcast_attention=False, upcast_softmax=False,
            cls_mode=False, pdnorm_bn=False, pdnorm_ln=False,
            pdnorm_decouple=True, pdnorm_adaptive=False, pdnorm_affine=True,
            pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
        ),
        head=dict(
            ignore_index=IGNORE_INDEX,
            temperature=0.07, ema_momentum=0.99, conf_threshold=0.6,
            proto_loss_weight=0.2, af3_loss_weight=1.0, ortho_loss_weight=0.1,
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    try:
        model = build_model(model_cfg).to(device)
        N = 4096
        fake_input = {
            "coord": torch.randn(N, 3).to(device),
            "grid_coord": torch.randint(0, 100, (N, 3)).to(device),
            "feat": torch.randn(N, 4).to(device),
            "offset": torch.tensor([N], dtype=torch.long).to(device),
            "segment": torch.randint(0, NUM_CLASSES, (N,)).to(device),
        }

        # Train mode
        model.train()
        out = model(fake_input)
        check("train forward → loss scalar", "loss" in out and out["loss"].ndim == 0,
              f"loss={out['loss'].item():.4f}")

        # Eval mode (no segment)
        model.eval()
        fake_test = {k: v for k, v in fake_input.items() if k != "segment"}
        with torch.no_grad():
            out = model(fake_test)
        check("eval forward → seg_logits", "seg_logits" in out,
              f"shape={out['seg_logits'].shape}")

        # Prototype warm-start check
        if os.path.exists(_WEIGHT_PATH):
            proto = model.head.prototypes.data
            check("prototypes non-zero", proto.abs().max().item() > 0.01,
                  f"norm={proto.norm(dim=1).mean().item():.3f}")

        return True
    except Exception as e:
        import traceback
        check("model forward", False, str(e))
        traceback.print_exc()
        return False


def main():
    print("PTv3_PPS dry-run")
    print(f"  data_root  : {_DATA_ROOT}  exists={os.path.isdir(_DATA_ROOT)}")
    print(f"  info_base  : {_INFO_BASE}  exists={os.path.isdir(_INFO_BASE)}")
    print(f"  weight_path: {_WEIGHT_PATH}  exists={os.path.exists(_WEIGHT_PATH)}")
    print(f"  CUDA       : {torch.cuda.is_available()}")

    results = []
    results.append(test_dataset())
    results.append(test_dataloader())
    results.append(test_model_forward())

    print("\n=== Summary ===")
    labels = ["Dataset", "DataLoader", "Model forward"]
    all_pass = True
    for label, ok in zip(labels, results):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        all_pass &= ok
    print()
    print("All checks passed — ready to train." if all_pass else "Some checks FAILED — see above.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
