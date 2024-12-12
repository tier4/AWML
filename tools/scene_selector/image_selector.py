import glob
from argparse import ArgumentParser

import cv2
from mmengine.config import Config

from autoware_ml.registry import DATA_SELECTOR


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file")
    parser.add_argument(
        "inputs",
        type=str,
        help="Input image file path or directory path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./work_dirs",
        help="Output directory of images or predictiggon results.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    files = glob.glob(args.inputs)

    scene_selector = DATA_SELECTOR.build(cfg.scene_selector)

    result = scene_selector.is_target_scene(files, results_path=args.out_dir)
    print("Result: ", result)


if __name__ == "__main__":
    main()
