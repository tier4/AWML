"""Download T4Dataset from DATA SEARCH with webauto command.

Setup a tool and configuration following <https://github.com/tier4/WebAutoCLI>.
"""

import argparse
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import json
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

def check_t4dataset_latest_version(
    project_id: str,
    t4dataset_id: str,
) -> int:
    """Return the latest version of t4dataset using `webauto data annotation-dataset describe` command.
    Args:
        project_id (str): The project id of webauto command.
        t4dataset_id (str): The t4dataset id of webauto command.
    """
    describe_command = "webauto data annotation-dataset describe --project-id {} --annotation-dataset-id {} --output json".format(project_id, t4dataset_id)
    result = json.loads(subprocess.run(describe_command, shell=True, check=True, capture_output=True, text=True).stdout)
    return result['version_id']

def download_t4dataset(
    project_id: str,
    t4dataset_id: str,
    output_dir: str,
    config_path: str,
    delete_rosbag: bool,
) -> None:
    """Download t4dataset using webauto CLI. When there are multiple versions in t4dataset, it would automatically 
    Return: None

    Args:
        project_id (str): The project id of webauto command.
        t4dataset_id (str): The t4dataset id of webauto command.
        output_dir (str): The output directory path.
        config_path (str): The path to config.
    """
    download_command = "webauto data annotation-dataset pull --project-id {} --annotation-dataset-id {} --asset-dir {}"

    with TemporaryDirectory() as temp_dir:
        t4dataset_version_id: int = check_t4dataset_latest_version(project_id, t4dataset_id)
        print(f"\n***************** start downloading t4dataset id {t4dataset_id} with version id {t4dataset_version_id}")

        _, _, _, database_name, _ = divide_file_path(config_path)
        from_directory = os.path.join(
            temp_dir,
            "annotation_dataset",
            t4dataset_id,
            str(t4dataset_version_id),
        )
        to_directory = os.path.join(output_dir, database_name, t4dataset_id)

        # skip if the target t4dataset already exists
        to_directory_with_version_id = os.path.join(to_directory, str(t4dataset_version_id))
        if os.path.exists(to_directory_with_version_id):
            print(f"t4dataset already exists at {to_directory_with_version_id}")
            return

        download_command_ = download_command.format(
            project_id,
            t4dataset_id,
            temp_dir,
        )
        print(download_command_)
        subprocess.call(download_command_, shell=True)

        if delete_rosbag is True:
            rosbag_path: str = os.path.join(from_directory, "input_bag")
            rm_command = "rm -r {}".format(rosbag_path)
            print(rm_command)
            subprocess.call(rm_command, shell=True)

        # rename directory
        os.makedirs(to_directory, exist_ok=True)

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
    parser.add_argument(
        '--delete-rosbag',
        action='store_true',
        help="Delete rosbag file from T4dataset",
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
            args.delete_rosbag,
        )


if __name__ == "__main__":
    main()
