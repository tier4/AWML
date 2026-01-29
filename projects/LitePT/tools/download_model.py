from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download

MODEL_NAMES = (
    # semantic segmentation
    "nuscenes-semseg-litept-small-v1m1",
    "scannet-semseg-litept-small-v1m1",
    "structured3d-semseg-litept-base-v1m1",
    "structured3d-semseg-litept-large-v1m1",
    "structured3d-semseg-litept-small-v1m1",
    "waymo-semseg-litept-small-v1m1",
    # instance segmentation
    "scannet-insseg-litept-small-v1m2",
    "scannet200-insseg-litept-small-v1m2",
    # object detection
    "waymo-objdet-litept-small-v1m3",
)


def download_model(model_name: str, output_dir: str) -> None:
    model_path = hf_hub_download(
        repo_id="prs-eth/LitePT",
        filename=f"{model_name}/model/model_best.pth",
        repo_type="model",
        local_dir=output_dir,
    )
    print(f"✅ Model downloaded successfully: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download LitePT models via HuggingFace Hub"
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=[
            # semantic segmentation
            "nuscenes-semseg-litept-small-v1m1",
            "scannet-semseg-litept-small-v1m1",
            "structured3d-semseg-litept-base-v1m1",
            "structured3d-semseg-litept-large-v1m1",
            "structured3d-semseg-litept-small-v1m1",
            "waymo-semseg-litept-small-v1m1",
            # instance segmentation
            "scannet-insseg-litept-small-v1m2",
            "scannet200-insseg-litept-small-v1m2",
            # object detection
            "waymo-objdet-litept-small-v1m3",
            # download all models
            "all",
        ],
    )
    parser.add_argument("-o", "--output", type=str, default="workdir")
    args = parser.parse_args()

    if args.model_name == "all":
        for model_name in MODEL_NAMES:
            download_model(model_name, args.output)
    else:
        download_model(args.model_name, args.output)


if __name__ == "__main__":
    main()
