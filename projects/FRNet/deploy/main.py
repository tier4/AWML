import argparse
import os

import numpy as np
import numpy.typing as npt
import torch
from mmdeploy.utils import load_config
from mmdet3d.utils import register_all_modules
from nuscenes.nuscenes import NuScenes
from onnx_model import OnnxModel
from postprocessing import Postprocessing
from preprocessing import Preprocessing
from torch_model import TorchModel
from trt_model import TrtModel
from visualizer import Visualizer

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} is available.')
else:
    print('No GPU available. Exiting...')
    exit(1)

register_all_modules()


def parse_args():
    parser = argparse.ArgumentParser(description='Perform inference.')
    parser.add_argument(
        'checkpoint', type=str, help='Path to PyTorch checkpoint file.')
    parser.add_argument(
        '--execution',
        choices=['torch', 'onnx', 'tensorrt'],
        default=None,
        help='Specify the execution method for provided model.')
    parser.add_argument(
        '--model-cfg',
        type=str,
        default=
        '/workspace/projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py',
        help='Model config file path.')
    parser.add_argument(
        '--deploy-cfg',
        type=str,
        default=
        '/workspace/projects/FRNet/configs/deploy/frnet_tensorrt_dynamic.py',
        help='Deploy config file path.')
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='/workspace/data/nuscenes',
        help='The directory to dataset with LiDAR sweeps and label files.')
    parser.add_argument(
        '--threshold', type=float, default=-999.9, help='Threshold for score.')
    parser.add_argument(
        '--no-deploy',
        dest='deploy',
        action='store_false',
        help='Skip model deploy.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode.')

    return parser.parse_args()


def get_pcd_paths(dataset_dir: str) -> list:
    pcd_paths = []
    nusc = NuScenes(version='v1.0-test', dataroot=dataset_dir, verbose=True)

    print(f'Total test samples: {len(nusc.scene)}')
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
        pcd_paths.append(lidar_path)

    return pcd_paths


def load_input(pcd_path: str) -> npt.ArrayLike:
    points = np.fromfile(pcd_path, dtype=np.float32)
    points = points.reshape(-1, 5)[:, :4]
    return points


def main():
    args = parse_args()

    model_cfg_path = args.model_cfg
    deploy_cfg_path = args.deploy_cfg
    model_cfg, deploy_cfg = load_config(model_cfg_path, deploy_cfg_path)

    preprocessing = Preprocessing(config=model_cfg)
    postprocessing = Postprocessing(score_threshold=args.threshold)

    pcd_paths = get_pcd_paths(dataset_dir=args.dataset_dir)
    onnx_path = os.path.join(os.path.dirname(args.checkpoint), 'frnet.onnx')

    batch_inputs_dict = preprocessing.preprocess(load_input(pcd_paths[0]))

    torch_model = TorchModel(
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        checkpoint_path=args.checkpoint)

    onnx_model = OnnxModel(
        deploy_cfg=deploy_cfg,
        model=torch_model.model,
        batch_inputs_dict=batch_inputs_dict,
        onnx_path=onnx_path,
        deploy=args.deploy,
        verbose=args.verbose)

    trt_model = TrtModel(
        deploy_cfg=deploy_cfg,
        onnx_path=onnx_path,
        deploy=args.deploy,
        verbose=args.verbose)

    visualizer = Visualizer(class_names=torch_model.class_names)

    if args.execution == None:
        exit()

    print(f'Score threshold: {args.threshold}')
    for pcd_path in pcd_paths:
        print('-' * 80)
        points = load_input(pcd_path)
        batch_inputs_dict = preprocessing.preprocess(points)

        print(f'Running {args.execution} inference...')
        if args.execution == 'torch':
            predictions = torch_model.inference(batch_inputs_dict)
        elif args.execution == 'onnx':
            predictions = onnx_model.inference(batch_inputs_dict)
        elif args.execution == 'tensorrt':
            predictions = trt_model.inference(batch_inputs_dict)

        result = postprocessing.postprocess(predictions)

        visualizer.visualize(batch_inputs_dict, result)


if __name__ == '__main__':
    main()
