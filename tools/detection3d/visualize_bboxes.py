"""
Script to generate predictions with a 3D perception model to visualize bboxes.
"""

import logging
import argparse
import os
from pathlib import Path

from autoware_ml.detection3d.runners.prediction_runner import PredictionRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export CenterPoint model to backends.')
    parser.add_argument('model_cfg_path', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument(
        '--work-dir', default='', help='the dir to save logs and models')
    parser.add_argument(
        '--data-root',
        default='',
        help='the dir to save datasets. Set it to overwrite the default config'
    )
    parser.add_argument(
        '--bboxes-score-threshold',
        default=0.10,
        type=float,
        help='Score threshold to filter out bboxes')
    parser.add_argument(
        '--frame-range',
        default=None,
        nargs="*",
        type=int,
        help='Index to render data')
    parser.add_argument(
        '--ann-file-path',
        default='',
        help='the dir to save pkl infos. Set it to overwrite the default config'
    )
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default='gpu',
        help="Set running device!")
    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help="Maximum cpu workers running in multiprocessing")
    args = parser.parse_args()
    return args


def build_prediction_runner(args) -> PredictionRunner:
    """ Build a PredictionRunner. """
    model_cfg_path = args.model_cfg_path
    checkpoint_path = args.checkpoint
    experiment_name = Path(model_cfg_path).stem
    work_dir = Path(
        os.getcwd()
    ) / 'work_dirs' / 'predictions' / experiment_name if not args.work_dir else Path(
        args.work_dir)

    prediction_runner = PredictionRunner(
        experiment_name=experiment_name,
        model_cfg_path=model_cfg_path,
        checkpoint_path=checkpoint_path,
        work_dir=work_dir,
        max_workers=args.max_workers,
        data_root=args.data_root,
        ann_file_path=args.ann_file_path,
        device=args.device,
        frame_range=args.frame_range,
        bboxes_score_threshold=args.bboxes_score_threshold)
    return prediction_runner


if __name__ == '__main__':
    """ Run an InferenceRunner. """
    args = parse_args()

    # Build PredictionRunner
    prediction_runner = build_prediction_runner(args=args)

    # Start running PredictionRunner
    prediction_runner.run()
