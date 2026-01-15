"""Download T4Dataset from DATA SEARCH with webauto command.

Setup a tool and configuration following <https://github.com/tier4/WebAutoCLI>.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import yaml
from packaging import version

# Required webauto version
WEBAUTO_VERSION = "v0.50.0"


def check_webauto_version(webauto_path: str) -> None:
    """
    Check if the webauto version is >= the required version.

    Args:
        webauto_path (str): The path to WebAutoCLI.

    Raises:
        Exception: If webauto version check fails or is below required version.
    """
    command = f"{webauto_path} --version"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception(f"Failed to get webauto version. {result.stderr.decode('utf-8')}")

    version_output = result.stdout.decode("utf-8").strip()

    # Extract version from output (assuming format like "webauto v0.50.0" or just "v0.50.0")
    version_match = re.search(r"v?(\d+\.\d+\.\d+)", version_output)
    if not version_match:
        raise Exception(f"Could not parse version from webauto output: {version_output}")

    current_version = version_match.group(1)
    required_version = WEBAUTO_VERSION.lstrip("v")  # Remove 'v' prefix if present

    if version.parse(current_version) < version.parse(required_version):
        raise Exception(f"Webauto version {current_version} is below required version {required_version}")


def get_t4dataset_ids(config_path: str) -> list[str]:
    """
    Get T4Dataset IDs like "0df0328e-39ea-42f1-844a-b455c91dc6cc".

    Args:
        config_path (str): The path to config.

    Returns: list[str]
    """
    with open(config_path) as f:
        data_splits = yaml.safe_load(f)
    required_keys = ["train", "val", "test"]
    assert isinstance(data_splits, dict) and all(
        [isinstance(data_splits[key], list) for key in required_keys]
    ), "config file must be a type of `dict[str, list]`"

    all_t4dataset_ids = set()
    for key in required_keys:
        for t4dataset_ids in data_splits[key]:
            t4dataset_ids = t4dataset_ids.split("   ")
            if len(t4dataset_ids) == 2:
                all_t4dataset_ids.add((t4dataset_ids[0], t4dataset_ids[1]))  # (id, version)
            elif len(t4dataset_ids) == 1:
                all_t4dataset_ids.add((t4dataset_ids[0], -1))  # -1 indicates no version specified
            else:
                raise ValueError(f"Invalid T4Dataset format in {t4dataset_ids}. Use format 'id   version' or 'id'.")
    return list(all_t4dataset_ids)


def divide_file_path(full_path: str) -> Union[str, str, str, str, str]:
    """
    Get each element of path for file path.

    Args:
        full_path (str): "./dir/subdir/filename.ext.ext2"

    Return:
        ["./dir/subdir", "filename.ext", "subdir", "filename" "ext.ext2" ]
    """
    dir_name, base_name = os.path.split(full_path)
    subdir_name = os.path.basename(os.path.dirname(full_path))
    basename_without_ext, extension = base_name.split(".", 1)

    return dir_name, base_name, subdir_name, basename_without_ext, extension


def check_t4dataset_latest_version(
    webauto_path: str,
    project_id: str,
    t4dataset_id: str,
) -> int:
    """
    Return the latest version of t4dataset using `webauto data annotation-dataset describe` command.

    Args:
        webauto_path (str): The path to WebAutoCLI.
        project_id (str): The project id of webauto command.
        t4dataset_id (str): The t4dataset id of webauto command.
    Return: The latest version of T4dataset (int)
    """

    describe_command = (
        "{} data annotation-dataset describe --project-id {} --annotation-dataset-id {} --output json".format(
            webauto_path, project_id, t4dataset_id
        )
    )
    result = subprocess.run(
        describe_command,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if result == "" or result == "\n":
        print(f"fail to check the version in {describe_command}")
    result_json = json.loads(result)
    return result_json["version_id"]


def pull_t4dataset(
    webauto_path: str,
    project_id: str,
    t4dataset_id: str,
    t4dataset_version_id: int,
    output_dir: str,
) -> None:
    """
    Pull T4dataset using WebAutoCLI.

    Args:
        webauto_path (str): The path to WebAutoCLI.
        project_id (str): The project id of webauto command.
        t4dataset_id (str): The t4dataset id of webauto command.
        t4dataset_version_id (int): The version id of T4dataset.
        output_dir (str): The output directory path.
    Return: None
    """

    download_command = "{} data annotation-dataset pull --project-id {} --annotation-dataset-id {} --annotation-dataset-version-id {} --asset-dir {}"
    download_command_ = download_command.format(
        webauto_path,
        project_id,
        t4dataset_id,
        t4dataset_version_id,
        output_dir,
    )
    print(download_command_)
    subprocess.call(download_command_, shell=True)


def move_t4dataset(
    from_directory: str,
    to_directory: str,
) -> None:
    """
    Move temp directory to local directory for T4dataset.

    Args:
        from_directory (str): Path to move from
        to_directory (str): Path to move to
    Returns: None
    """

    os.makedirs(to_directory, exist_ok=True)
    mv_command = "mv {} {}".format(from_directory, to_directory)
    print(mv_command)
    subprocess.call(mv_command, shell=True)


def download_t4dataset(
    config_path: Path,
    webauto_path: str,
    project_id: str,
    t4dataset_id: str,
    t4dataset_version_id: int,
    output_dir: str,
    temp_dir: str,
    delete_rosbag: bool,
    download_latest: bool = False,
) -> None:
    """
    Download T4dataset.

    Args:
        config_path (Path): The path to the config file.
        webauto_path (str): The path to WebAutoCLI.
        project_id (str): The project id of webauto command.
        t4dataset_id (str): The t4dataset id of webauto command.
        t4dataset_version_id (int): The version id of T4dataset.
        output_dir (str): The output directory path.
        temp_dir (str): The temp directory to download T4dataset.
        delete_rosbag: (bool): If the this value is true, the rosbag file is deleted.
        download_latest (bool): If true, download the latest version of T4dataset.
    Return: None
    """

    # check the latest version of T4dataset
    if download_latest or t4dataset_version_id == -1:
        t4dataset_version_id: int = check_t4dataset_latest_version(
            webauto_path,
            project_id,
            t4dataset_id,
        )

    print(f"\n***** Start downloading T4Dataset ID:{t4dataset_id} with version:{t4dataset_version_id}")

    # set path to directory
    _, _, _, database_name, _ = divide_file_path(config_path)
    from_directory = os.path.join(
        temp_dir,
        "annotation_dataset",
        t4dataset_id,
        str(t4dataset_version_id),
    )
    to_directory = os.path.join(output_dir, database_name, t4dataset_id)
    to_directory_with_version_id = os.path.join(to_directory, str(t4dataset_version_id))

    # skip if the target t4dataset already exists
    if os.path.exists(to_directory_with_version_id):
        print(f"t4dataset already exists at {to_directory_with_version_id}")
        return

    pull_t4dataset(
        webauto_path,
        project_id,
        t4dataset_id,
        t4dataset_version_id,
        temp_dir,
    )

    # delete rosbag
    if delete_rosbag is True:
        rosbag_path: str = os.path.join(from_directory, "input_bag")
        rm_command = "rm -r {}".format(rosbag_path)
        print(rm_command)
        subprocess.call(rm_command, shell=True)

    # check if the dataset is successfully downloaded
    path_to_annotation = os.path.join(from_directory, "annotation")
    if not os.path.isdir(path_to_annotation):
        print(f"fail to download annotation files at {path_to_annotation}")
        return

    move_t4dataset(from_directory, to_directory)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="list of t4dataset for train/val/test",
    )
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
        "--delete-rosbag",
        action="store_true",
        help="Delete rosbag file from T4dataset",
    )
    parser.add_argument(
        "--webauto-path",
        type=str,
        help="The path to WebAutoCLI binary file for executing the CLI command",
        default="webauto",
    )
    parser.add_argument(
        "--download-latest",
        action="store_true",
        help="Download the latest version of T4dataset",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Check webauto version before proceeding
    check_webauto_version(args.webauto_path)

    config_path = Path(args.config)
    assert config_path.exists() and config_path.is_file()
    output_dir = Path(args.output)
    assert output_dir.exists() and output_dir.is_dir()

    t4dataset_ids = get_t4dataset_ids(config_path)

    for t4dataset_id, t4dataset_version_id in t4dataset_ids:
        with TemporaryDirectory() as temp_dir:
            download_t4dataset(
                config_path,
                args.webauto_path,
                args.project_id,
                t4dataset_id,
                t4dataset_version_id,
                output_dir,
                temp_dir,
                args.delete_rosbag,
                args.download_latest,
            )


if __name__ == "__main__":
    main()
