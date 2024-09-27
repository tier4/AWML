import glob
from argparse import ArgumentParser

import cv2
from mmengine.config import Config
from mmengine.registry import Registry

SELECTOR = Registry('selector')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='Config file')
    parser.add_argument(
        'inputs',
        type=str,
        help='Input image file or folder path.',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='work_dirs',
        help='Output directory of images or predictiggon results.',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    files = glob.glob(args.inputs)

    scene_selector = SELECTOR.build(cfg)

    for file in files:
        image = cv2.imread(file)
        result = scene_selector.is_target_scene(image)
        print("Result: ", result)


if __name__ == '__main__':
    main()
