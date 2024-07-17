"""Download T4Dataset from DATA SEARCH with webauto command.

Setup a tool and configuration following <https://github.com/tier4/WebAutoCLI>.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Union
from tempfile import TemporaryDirectory

import yaml


def divide_file_path(full_path: str) -> Union[str, str, str, str, str]:
    """
    Args
        full_path (str): './dir/subdir/filename.ext.ext2'
    Return
        ["./dir/subdir", "filename.ext", "subdir", "filename" "ext.ext2" ]
    """
    dir_name, base_name = os.path.split(full_path)
    subdir_name = os.path.basename(os.path.dirname(full_path))
    basename_without_ext, extension = base_name.split(".", 1)

    return dir_name, base_name, subdir_name, basename_without_ext, extension


def download_t4dataset(
    project_id: str,
    t4dataset_id: str,
    output_dir: str,
    config_path: str,
) -> None:
    """
    Return: None

    Args:
        project_id (str): The project id of webauto command.
        t4dataset_id (str): The t4dataset id of webauto command.
        output_dir (str): The output directory path.
        config_path (str): The path to config.
    """
    download_command = "webauto data annotation-dataset pull --project-id {} --annotation-dataset-id {} --asset-dir {}"

    with TemporaryDirectory() as temp_dir:
        download_command_ = download_command.format(
            project_id,
            t4dataset_id,
            temp_dir,
        )
        print(download_command_)
        subprocess.call(download_command_, shell=True)

        # rename directory
        _, _, _, database_name, _ = divide_file_path(config_path)
        from_directory = os.path.join(temp_dir, "annotation_dataset")
        to_directory = os.path.join(output_dir, database_name)

        mv_command = "mv {} {}".format(from_directory, to_directory)
        print(mv_command)
        subprocess.call(mv_command, shell=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="list of t4dataset for train/val/test")
    parser.add_argument(
        "--project-id",
        type=str,
        choices=["prd_jt", "x2_dev"],
        required=True,
        help="project id of dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/t4dataset",
        help="directory path for data to be downloaded",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_path = Path(args.config)
    assert config_path.exists() and config_path.is_file()

    output_dir = Path(args.output)
    assert output_dir.exists() and output_dir.is_dir()

    with open(config_path) as f:
        data_splits = yaml.safe_load(f)
    required_keys = ["train", "val", "test"]
    assert isinstance(data_splits, dict) and all([
        isinstance(data_splits[key], list) for key in required_keys
    ]), "config file must be a type of `dict[str, list]`"
    t4dataset_ids = sum([data_splits[key] for key in required_keys], [])

    for t4dataset_id in t4dataset_ids:
        download_t4dataset(
            args.project_id,
            t4dataset_id,
            output_dir,
            config_path,
        )


if __name__ == "__main__":
    main()
