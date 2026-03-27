"""
CLI helpers: argument parsing and logging setup.

Config schema does not depend on argparse/logging; this module is the single place for CLI concerns.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

_LOG_FORMAT = "%(levelname)s:%(name)s:%(message)s"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(level=getattr(logging, level), format=_LOG_FORMAT)
    return logging.getLogger("deployment")


def add_deployment_file_logging(log_file_path: str) -> None:
    """
    Append a UTF-8 file handler to the root logger so all log records are also written to disk.

    Idempotent for the same absolute path. Creates parent directories when needed.

    Args:
        log_file_path: Absolute or resolved path to the log file.
    """
    path = os.path.abspath(os.path.expanduser(log_file_path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            if os.path.abspath(getattr(h, "baseFilename", "")) == path:
                return

    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    fh.setLevel(root.level)
    root.addHandler(fh)


def parse_base_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """
    Create argument parser with common deployment arguments.

    Args:
        parser: Optional existing ArgumentParser to add arguments to

    Returns:
        ArgumentParser with deployment arguments
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Deploy model to ONNX/TensorRT",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add_argument("deploy_cfg", help="Deploy config path")
    parser.add_argument("model_cfg", help="Model config path")
    # Optional overrides
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser
