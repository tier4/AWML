"""
PTv3_PPS training entry point.

Usage (single GPU):
    python train.py --config-file configs/semseg-ptv3-pps-t4dataset-phase1.py

Usage (multi-GPU, e.g. 4):
    python train.py --config-file configs/semseg-ptv3-pps-t4dataset-phase2.py --num-gpus 4

Sys.path is configured here so that:
  - PTv3/  is importable (models/, engines/, utils/, datasets/)
  - PTv3_PPS/ is importable (pps_models/, configs/)
PTv3 must come BEFORE PTv3_PPS so 'import models' resolves to PTv3/models/.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PTv3 = os.path.abspath(os.path.join(_HERE, "..", "PTv3"))

# PTv3 first so 'models', 'engines', 'utils', 'datasets' resolve to PTv3
sys.path.insert(0, _HERE)   # pps_models/, configs/ etc.
sys.path.insert(0, _PTv3)   # models/, engines/, utils/, datasets/

# Register all PTv3 models (DefaultSegmentorV2, PT-v3m1, losses, …)
import models  # noqa: E402, F401

# Register PPSSegmentor into the same MODELS registry
import pps_models  # noqa: E402, F401

from engines.defaults import (  # noqa: E402
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.launch import launch  # noqa: E402
from engines.train import TRAINERS  # noqa: E402


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
