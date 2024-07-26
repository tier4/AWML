from argparse import ArgumentParser

from mmengine.config import Config

from tools.scene_selector.rosbag.rosbag import Rosbag
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

    scene_selector = SelectScene()
    scene_selector.init()
    rosbag = Rosbag()

    for scene in rosbag.get_scenes():
        for image in scene.get_images():
            scene_selector.inference(image)
    scene_selector.get_results()


if __name__ == '__main__':
    main()
