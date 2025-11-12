"""Script to compute analysis of T4 datasets."""

import argparse

from mmengine.logging import print_log

from tools.analysis_2d.analysis_runner import AnalysisRunner2D


def parse_args():
    """Add args and parse them through CLI."""
    parser = argparse.ArgumentParser(description="analysis of T4dataset in 2D")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="config for T4dataset",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="output directory of info file",
    )
    args = parser.parse_args()
    return args


def main():
    """Main enrtypoint to run the Runner."""
    args = parse_args()
    # Build AnalysesRunner
    print_log("Building AnalysisRunner2D...", logger="current")
    analysis_runner = AnalysisRunner2D(
        data_root_path=args.data_root_path,
        config_path=args.config_path,
        out_path=args.out_dir,
    )
    print_log("Built AnalysisRunner!")

    # Run AnalysesRunner
    analysis_runner.run()


if __name__ == "__main__":
    main()
