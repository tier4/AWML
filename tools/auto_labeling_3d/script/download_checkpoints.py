import argparse
from pathlib import Path
import requests
import yaml
import logging
from tools.auto_labeling_3d.utils.logger import setup_logger

def download_file(url: str, save_path: Path, logger: logging.Logger) -> None:
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

def main() -> None:
    """
    Main entry point for downloading checkpoints specified in a config YAML file.
    Parses command-line arguments, loads the config, and downloads each checkpoint.
    """
    parser = argparse.ArgumentParser(description="Download checkpoints from config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument(
        "--log-level",
        help="Set log level",
        default="INFO",
        choices=list(logging._nameToLevel.keys()),
    )
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing logs or outputs",
    )
    args = parser.parse_args()

    logger: logging.Logger = setup_logger(args, name="download_checkpoints")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    create_info = config.get("create_info", {})
    model_list = create_info.get("model_list", [])
    for model in model_list:
        checkpoint = model.get("checkpoint", {})
        url = checkpoint.get("model_zoo_url")
        save_path = checkpoint.get("checkpoint_path")
        if url and save_path:
            download_file(url, Path(save_path), logger)
        else:
            logger.warning(f"Skipping model: missing url or save_path: {model}")

if __name__ == "__main__":
    main()
