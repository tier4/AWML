import argparse
import logging
from pathlib import Path

import requests
import yaml


def download_checkpoint(url: str, save_path: Path, logger: logging.Logger) -> None:
    """
    Downloads a file from the specified URL and saves it to the given path.
    Creates parent directories if they do not exist.

    Args:
        url (str): The URL of the file to download.
        save_path (Path): The local path where the file will be saved.
        logger (logging.Logger): Logger for logging messages.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} to {save_path} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Download completed.")
