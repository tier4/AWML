import argparse
import datetime
import logging
from pathlib import Path

from mmdeploy.utils import get_root_logger

def setup_logger(args: argparse.Namespace, name: str) -> logging.Logger:
    """Set up a logger with file and stream handlers.

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - work_dir (str, optional): Directory for log files
            - log_level (str): Logging level for stream handler
        name (str): Base name for the log file and directory

    Returns:
        logging.Logger: Configured logger instance with both stream and file handlers

    Note:
        - Work directory is determined in this priority: CLI > 'work_dirs/auto_labeling_3d/{name}'
        - Stream handler level is set according to args.log_level
        - File handler is always set to DEBUG level
        - Log file name format: "{name}_{YYYYMMDD_HHMMSS}.log"
        - All command line arguments are logged at DEBUG level
    """
    # work_dir is determined in this priority: CLI > 'work_dirs/auto_labeling_3d/{name}'
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        log_dir = Path(args.work_dir)
    else:
        # use config filename as default work_dir if args.work_dir is None
        log_dir = Path('work_dirs') / "auto_labeling_3d" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # set logger
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(log_level)
    
    file_handler = logging.FileHandler(
        log_dir / f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
      )
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # log for debug
    for arg_name, arg_value in vars(args).items():
        logger.debug(f"args.{arg_name}: {arg_value}")

    return logger
