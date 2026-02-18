"""CenterPoint CLI extensions."""

from __future__ import annotations

import argparse


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register CenterPoint-specific CLI flags onto a project subparser."""
    parser.add_argument(
        "--rot-y-axis-reference",
        action="store_true",
        help="Convert rotation to y-axis clockwise reference (CenterPoint ONNX-compatible format)",
    )
