import argparse
import os
import os.path as osp
import re
import warnings
from pathlib import Path
from typing import Dict, List

import cv2
import hydra
import numpy as np
import supervision as sv
import torch
import yaml
from collections import defaultdict
from groundingdino.util.inference import load_image, load_model, predict
from hydra import initialize
from mmengine.config import Config
from mmengine.logging import print_log
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from t4_devkit import Tier4
from torchvision.ops import box_convert
from tqdm import tqdm


class SAM2Wrapper:

    def __init__(self, cfg):

        self.cfg = cfg
        self.sam2_classes = self.cfg["sam2_classes"]
        self.text_prompt = ". ".join(self.sam2_classes) + "."

        self.sam2_checkpoint = self.cfg["sam2_checkpoint"]
        self.sam2_cfg = self.cfg["sam2_cfg"]
        self.grounding_dino_checkpoint = self.cfg["grounding_dino_checkpoint"]
        self.grounding_dino_cfg = self.cfg["grounding_dino_cfg"]
        self.background_value = self.cfg["background_value"]

        self.box_threshold = self.cfg["box_threshold"]
        self.text_threshold = self.cfg["text_threshold"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # environment settings
        # use bfloat16

        # build SAM2 image predictor

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        config_dir = ""
        with initialize(config_path=config_dir):
            self.sam2_model = build_sam2(self.sam2_cfg, self.sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=self.grounding_dino_cfg,
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=self.device,
        )

    def get_best_label(self, sam2_label: str, sam2_classes: List[str]):

        sam2_label_list = sam2_label.split(" ")

        for i in range(len(sam2_label_list)):
            candidate = " ".join(sam2_label_list[0 : i + 1])
            if candidate in sam2_classes:
                return candidate

        return ""

    def segment(self, img_path, override):

        img_path = Path(img_path)
        seg_img_path = img_path.with_name(img_path.stem + "_seg.png")
        anno_img_path = img_path.with_name(img_path.stem + "_anno.jpg")

        if seg_img_path.exists() and not override:
            return None

        image_source, image = load_image(str(img_path))

        self.sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            if len(input_boxes) > 0:

                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

            else:

                masks = np.array([]).reshape(0, h, w)
                scores = []
                logits = []

        # Class image creation

        class_image = np.full((h, w, 1), self.background_value, dtype=np.uint8)
        label_to_class_idx = {}

        for idx, label in enumerate(self.sam2_classes):
            label_to_class_idx[label] = idx

        for instance_idx in reversed(range(len(confidences))):
            instance_label = self.get_best_label(labels[instance_idx], self.sam2_classes)

            if instance_label in label_to_class_idx:
                class_idx = label_to_class_idx[instance_label]
            else:
                print(f"Unrecognized label: {labels[instance_idx]}")
                continue

            mask = masks[instance_idx].squeeze().astype(np.bool_)

            # if mask[240, 800]:
            #    x = 0

            class_image[mask] = class_idx

        cv2.imwrite(str(seg_img_path), class_image)

        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences)]

        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids  # (n, 4)  # (n, h, w)
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(str(anno_img_path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])

        return annotated_frame


def get_scene_root_dir_path(
    root_path: str,
    dataset_version: str,
    scene_id: str,
) -> str:
    """
    This function checks if the provided `scene_root_dir_path` follows the new directory structure
    of the T4 Dataset, which should look like `$T4DATASET_VERSION/$T4DATASET_ID/$VERSION_ID/`.
    If the `scene_root_dir_path` does contain a version directory, it searches for the latest version directory
    under the `scene_root_dir_path` and returns the updated path.
    If no version directory is found, it prints a deprecation warning and returns the original `scene_root_dir_path`.

    Args:
        root_path (str): The root path of the T4 Dataset.
        dataset_version (str): The dataset version like 'db_jpntaxi_v2'
        scene_id: The scene id token.
    Returns:
        str: The updated path containing the version directory if it exists,
            otherwise the original `scene_root_dir_path`.
    """
    # an integer larger than or equal to 0
    version_pattern = re.compile(r"^\d+$")

    scene_root_dir_path = osp.join(root_path, dataset_version, scene_id)

    version_dirs = [d for d in os.listdir(scene_root_dir_path) if version_pattern.match(d)]

    if version_dirs:
        version_id = sorted(version_dirs, key=int)[-1]
        return os.path.join(scene_root_dir_path, version_id)
    else:
        warnings.simplefilter("always")
        warnings.warn(
            f"The directory structure of T4 Dataset is deprecated. In the newer version, the directory structure should look something like `$T4DATASET_ID/$VERSION_ID/`. Please update your Web.Auto CLI to the latest version.",
            DeprecationWarning,
        )
        return scene_root_dir_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create data info for T4dataset")

    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="config for T4dataset",
    )

    parser.add_argument(
        "--segmentation_config",
        type=str,
        required=True,
        help="config for sam2 + grounding dino",
    )

    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )

    parser.add_argument(
        "--out_videos",
        type=str,
        required=True,
        help="directory to save segmented videos",
    )

    args = parser.parse_args()
    return args


def make_video(video_folder, scene_id, cam_name, images):

    if len(images) == 0:
        print("Empty list. Already processed (?)")
        return

    height, width, layers = images[0].shape

    # Define output video settings
    output_file = Path(video_folder) / f"{scene_id}_{cam_name}.mp4"
    fps = 2  # frames per second
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create the video writer
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width // 2, height // 2))

    # Write each image to the video
    for image in images:
        image = cv2.resize(image, (width // 2, height // 2))
        video_writer.write(image)

    video_writer.release()
    print(f"Video created successfully: {output_file}")


def main():
    args = parse_args()

    # load config
    dataset_cfg = Config.fromfile(args.dataset_config)
    os.makedirs(args.out_videos, exist_ok=True)

    # load config
    with open(args.segmentation_config, "r") as f:
        segmentation_cfg = yaml.safe_load(f)["sam2"]

    model = SAM2Wrapper(segmentation_cfg)

    for dataset_version in tqdm(dataset_cfg.dataset_version_list):
        dataset_list = osp.join(dataset_cfg.dataset_version_config_root, dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

        for split in tqdm(["train", "val", "test"]):
            print_log(f"Segmenting images from split: {split}", logger="current")
            for scene_id in tqdm(dataset_list_dict.get(split, [])):
                print_log(f"Segmented images from scene: {scene_id}")
                scene_root_dir_path = get_scene_root_dir_path(
                    args.root_path,
                    dataset_version,
                    scene_id,
                )

                if not osp.isdir(scene_root_dir_path):
                    raise ValueError(f"{scene_root_dir_path} does not exist.")

                t4 = Tier4(version="annotation", data_root=scene_root_dir_path, verbose=False)
                #scene_seg_images_dict = {camera_name: [] for camera_name in dataset_cfg.camera_types}
                scene_seg_images_dict = defaultdict(list)

                for i, sample_data in enumerate(tqdm(t4.sample_data)):

                    if sample_data.fileformat not in ("jpg", "png") or (
                        segmentation_cfg["only_key_frames"] and not sample_data.is_key_frame
                    ):
                        continue

                    cam_name = sample_data.channel

                    seg_img = model.segment(os.path.join(scene_root_dir_path, sample_data.filename), segmentation_cfg["override"])

                    if seg_img is None:
                        continue

                    scene_seg_images_dict[cam_name].append(seg_img)

                for cam_name, images in scene_seg_images_dict.items():
                    make_video(args.out_videos, scene_id, cam_name, images)


if __name__ == "__main__":
    main()
