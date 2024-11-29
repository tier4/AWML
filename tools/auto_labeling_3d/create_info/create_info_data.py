import argparse
import datetime
import logging
from pathlib import Path
import pickle
from typing import List, Dict, Any

from mmdeploy.utils import get_root_logger
from mmengine.config import Config
from mmengine.registry import init_default_scope

from tools.auto_labeling_3d.create_info.create_data_non_annotated_t4dataset import _create_non_annotated_info
from tools.auto_labeling_3d.create_info.inference import inference

def create_info_data(
   non_annotated_dataset_path: Path,
   model_config: Config,
   model_checkpoint_path: str,
   model_name: str,
   out_dir: str,
   logger: logging.Logger,
) -> None:
    """Create info file(.pkl) with pseudo label from non-annotated dataset.

    Args:
        non_annotated_dataset_path (Path): Path to non-annotated dataset directory. e.g, "./data/t4dataset/pseudo_xx1/"
        model_config (Config): Config for the model used for auto labeling.
        model_checkpoint_path (str): Path to the model checkpoint(.pth) used for auto labeling.
        model_name (str): Name of the model used for auto labeling.
        out_dir (str): Path to output directory for pseudo labeled info file.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
        None: Results are saved to pickle file.
    """

    # create non_annotated info
    non_annotated_info_file_path: Path = _create_non_annotated_info(
        cfg=model_config,
        dataset_version=Path(non_annotated_dataset_path).stem,
        logger=logger
    )

    # predict pseudo label
    pseudo_labeled_dataset_info: Dict[str, Any] = inference(model_config, model_checkpoint_path, non_annotated_info_file_path.name)

    # delete non_annotated info
    non_annotated_info_file_path.unlink()

    # dump pseudo label
    output_pseudo_label_pkl_path = Path(out_dir) / f"pseudo_infos_raw_{model_name}.pkl"
    with open(output_pseudo_label_pkl_path, 'wb') as f:
        pickle.dump(pseudo_labeled_dataset_info, f)
    logger.info(f"Saved pseudo labeled info file in {output_pseudo_label_pkl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create pseudo labeled info data from non-annotated dataset')
    
    parser.add_argument(
        '--root-path',
        type=str,
        required=True,
        help='Path to non-annotated T4dataset directory'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='Path to output directory for pseudo labeled info file'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file of the model used for auto labeling'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        required=True,
        help='Path to checkpoint file of the model used for auto labeling'
    )
    parser.add_argument(
        "--log-level",
        help="Set log level",
        default="INFO",
        choices=list(logging._nameToLevel.keys()),
    )
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    return parser.parse_args()


def main():
    init_default_scope("mmdet3d")

    args = parse_args()

    # Check if non-annotated T4dataset directory exists
    root_path = Path(args.root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"Input directory not found: {args.root_path}")
    
    # Load config
    cfg = Config.fromfile(args.config)
    model_name: str = Path(args.config).stem

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir_path = Path('work_dirs') / "auto_labeling_3d" / "create_info_data"
        work_dir_path.mkdir(parents=True, exist_ok=True)
        cfg.work_dir = str(work_dir_path)
    
    # set logger
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(log_level)
    
    file_handler = logging.FileHandler(
        Path(cfg.work_dir) / f"create_info_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
      )
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # log for debug
    logger.debug(f"args.root_path: {args.root_path}")
    logger.debug(f"args.out_dir: {args.out_dir}")
    logger.debug(f"args.config: {args.config}")
    logger.debug(f"args.ckpt: {args.ckpt}")
    logger.debug(f"=========config=========\n{cfg}\n")

    create_info_data(
        non_annotated_dataset_path=root_path,
        model_config=cfg,
        model_checkpoint_path=args.ckpt,
        model_name=model_name,
        out_dir=args.out_dir,
        logger=logger,
    )

if __name__ == '__main__':
    main()
