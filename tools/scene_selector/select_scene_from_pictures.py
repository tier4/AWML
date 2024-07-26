import glob
from argparse import ArgumentParser

import cv2
from mmengine.config import Config

from tools.scene_selector.select_scene import SelectScene


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'scenario_config', type=str, help='Config for scenario file')
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='work_dirs',
        help='Output directory of images or prediction results.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.scenario_config)
    files = glob.glob(args.inputs)

    scene_selector = SelectScene(cfg.tasks)
    scene_selector.init()

    for file in files:
        image = cv2.imread(file)
        scene_selector.inference(image)


if __name__ == '__main__':
    main()
