import argparse
import os
import os.path as osp
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import mmengine
import numpy as np
import cv2
import yaml
from mmengine.config import Config
from mmengine.logging import print_log
from t4_devkit import Tier4
from t4_devkit.schema import ObjectAnn


@dataclass
class Instance:
    bbox: List[float]
    bbox_label: int
    mask: List[List[int]] = field(default_factory=list)
    extra_anns: List[str] = field(default_factory=list)

@dataclass
class DataEntry:
    img_path: str
    width: int
    height: int
    instances: List[Instance] = field(default_factory=list)
    surfaces: List[Instance] = field(default_factory=list)

@dataclass
class DetectionData:
    metainfo: Dict[str, str]
    data_list: List[DataEntry] = field(default_factory=list)

def save_semantic_mask_png(semantic_mask, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, semantic_mask.astype(np.uint8))

def convert_entry_instances_to_semantic(entry, allowed_classes):
    height, width = entry.height, entry.width
    semantic_mask = np.full((height, width), 255, dtype=np.uint8)

    for surf in entry.surfaces:
        cls_id = surf.bbox_label
        for poly_flat in surf.mask:
            if len(poly_flat) < 6: 
                continue
            poly = np.array(poly_flat, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(semantic_mask, [poly], color=cls_id)

    for inst in entry.instances:
        cls_id = inst.bbox_label
        for poly_flat in inst.mask:
            if len(poly_flat) < 6: 
                continue
            poly = np.array(poly_flat, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(semantic_mask, [poly], color=cls_id)
    
    return semantic_mask

def generate_colormap(num_classes):
    np.random.seed(42)
    colors = np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    return colors

def save_colored_mask(semantic_mask, output_path, colormap):
    height, width = semantic_mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for cls_id, color in enumerate(colormap):
        color_mask[semantic_mask == cls_id] = color

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, color_mask[:, :, ::-1])  # OpenCV BGR

def update_detection_data_annotations(
    data_list: Dict[str, DataEntry],
    object_ann: List[ObjectAnn],
    surface_ann: List[ObjectAnn],
    attributes: Dict[str, str],
    categories: Dict[str, str],
    class_mappings: Dict[str, str],
    allowed_classes: List[str],
    root_path: str,
    save_colored_masks: bool = False,
) -> None:

    # Instance (Objects) 
    for ann in object_ann:
        class_name = class_mappings.get(categories[ann.category_token], None)
        if class_name not in allowed_classes:
            continue
        bbox_label = allowed_classes.index(class_name)

        import cv2
        import numpy as np
        
        binary = ann.mask.decode().astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        flat_polygons = []
        for cnt in contours:
            if len(cnt) >= 3:  
                poly = cnt.reshape(-1, 2).tolist()  # [[x1,y1], [x2,y2], ...]
                poly_flat = [coord for point in poly for coord in point]  
                flat_polygons.append(poly_flat)

        instance = Instance(
            bbox=ann.bbox,
            bbox_label=bbox_label,
            mask=flat_polygons,
            extra_anns=[attributes[x] for x in ann.attribute_tokens],
        )
        data_list[ann.sample_data_token].instances.append(instance)

    # Surface (Background)
    for ann in surface_ann:
        class_name = class_mappings.get(categories[ann.category_token], None)
        
        if class_name not in allowed_classes:
            continue
            
        bbox_label = allowed_classes.index(class_name)

        binary = ann.mask.decode().astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        flat_polygons = []
        for cnt in contours:
            if len(cnt) >= 3:
                poly = cnt.reshape(-1, 2).tolist()
                poly_flat = [coord for point in poly for coord in point]
                flat_polygons.append(poly_flat)

        surface_instance = Instance(
            bbox=ann.bbox if hasattr(ann, 'bbox') else [0,0,0,0], 
            bbox_label=bbox_label,
            mask=flat_polygons,
            extra_anns=[] 
        )

        if ann.sample_data_token in data_list:
            data_list[ann.sample_data_token].surfaces.append(surface_instance)

    # generate and save semantic masks images
    colormap = generate_colormap(len(allowed_classes) + 1)
    for key, entry in data_list.items():
        gray_mask = convert_entry_instances_to_semantic(entry, allowed_classes)

        mask_file = f"{root_path}/masks/{(entry.img_path).split('/')[-5] +'_' + (entry.img_path).split('/')[-2] +'_' + (entry.img_path).split('/')[-1]}.png"
        os.makedirs(os.path.dirname(mask_file), exist_ok=True)
        cv2.imwrite(mask_file, gray_mask.astype(np.uint8))
        entry.gt_semantic_seg = mask_file

        if save_colored_masks:
            color_file = f"{root_path}/masks_color/{(entry.img_path).split('/')[-5] +'_' +(entry.img_path).split('/')[-2] +'_' + (entry.img_path).split('/')[-1]}.png"
            save_colored_mask(gray_mask, color_file, colormap)

def get_scene_root_dir_path(
    root_path: str,
    dataset_version: str,
    scene_id: str,
) -> str:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data info for T4dataset")
    parser.add_argument("--config", type=str, required=True, help="config for T4dataset")
    parser.add_argument("--root_path", type=str, required=True, help="specify the root path of dataset")
    parser.add_argument("--data_name", type=str, required=True, help="dataset name. example: tlr")
    parser.add_argument(
        "--use_available_dataset_version",
        action="store_true",
        help="Will resort to using the available dataset version if the one specified in the config file does not exist.",
    )
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="output directory of info file")
    parser.add_argument(
        "--save_colored_masks",
        action="store_true",
        help="Whether to save colored semantic masks.",
    )
    return parser.parse_args()


def get_detection_data_empty_dict(data_name: str, classes: List[str]) -> DetectionData:
    return DetectionData(
        metainfo={"dataset_type": data_name, "task_name": "detection_task", "classes": classes}, data_list=[]
    )


def assign_ids_and_save_detection_data(
    split_name: str, data_entries: List[DataEntry], out_dir: str, data_name: str, classes: List[str]
) -> None:
    detection_data = get_detection_data_empty_dict(data_name, classes)
    detection_data.data_list.extend(data_entries)

    # Convert to dict
    detection_data_dict = {
        "metainfo": detection_data.metainfo,
        "data_list": [
            {
                "img_id": i,
                "img_path": entry.img_path,
                "width": entry.width,
                "height": entry.height,
                "instances": [
                    {
                        "bbox": instance.bbox,
                        "bbox_label": instance.bbox_label,
                        "mask": instance.mask,
                        "extra_anns": instance.extra_anns,
                        "ignore_flag": 0,
                    }
                    for instance in entry.instances
                ],
                "seg_map_path": entry.gt_semantic_seg,
            }
            for i, entry in enumerate(detection_data.data_list)
        ],
    }

    save_path = osp.join(out_dir, f"{data_name}_infos_{split_name}.json")
    mmengine.dump(detection_data_dict, save_path)
    print(f"DetectionData annotations saved to {save_path}")


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    data_infos = {
        "train": [],
        "val": [],
        "test": [],
    }

    for dataset_version in cfg.dataset_version_list:
        dataset_list = osp.join(cfg.dataset_version_config_root, dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

        for split in ["train", "val", "test"]:
            print_log(f"Creating data info for split: {split}", logger="current")
            for scene_id in dataset_list_dict.get(split, []):
                print_log(f"Creating data info for scene: {scene_id}")

                if len(scene_id.split("   ")) == 2:
                    t4_dataset_id, t4_dataset_version_id = parts
                else:
                    t4_dataset_id = scene_id.strip()
                    t4_dataset_version_id = None

                if t4_dataset_version_id and os.path.exists(osp.join(args.root_path, t4_dataset_id, t4_dataset_version_id)):
                    scene_root_dir_path = osp.join(args.root_path, t4_dataset_id, t4_dataset_version_id)
                elif os.path.exists(osp.join(args.root_path, dataset_version, t4_dataset_id)):
                    print(
                        "Warning: f{t4_dataset_id} has no t4_dataset_version_id or the specified version is missing. "
                        "Using the available version on disk."
                    )
                    
                    scene_root_dir_path = get_scene_root_dir_path(args.root_path, dataset_version, t4_dataset_id)
                else:
                    raise ValueError(
                        f"{t4_dataset_id} does not exist."
                    )

                t4 = Tier4(
                    data_root=scene_root_dir_path,
                    verbose=False,
                )

                data_list: Dict[str, DataEntry] = {}
                for tmp in t4.sample_data:
                    if not tmp.is_key_frame:
                        continue
                    if not os.path.basename(tmp.filename)[-3:] == "bin":
                        data_entry = DataEntry(
                            img_path=os.path.abspath(os.path.join(t4.data_root, tmp.filename)),
                            width=tmp.width,
                            height=tmp.height,
                        )
                        data_list[tmp.token] = data_entry

                attributes = {tmp.token: tmp.name for tmp in t4.attribute}
                categories = {tmp.token: tmp.name for tmp in t4.category}

                update_detection_data_annotations(
                    data_list,
                    t4.object_ann,
                    t4.surface_ann,
                    attributes,
                    categories,
                    cfg.class_mappings,
                    cfg.classes,
                    args.root_path,
                    save_colored_masks=args.save_colored_masks,
                )
                data_infos[split].extend(data_list.values())

    # Save each split separately
    for split in ["train", "val", "test"]:
        assign_ids_and_save_detection_data(
            split,
            data_infos[split],
            args.out_dir,
            args.data_name,
            cfg.classes,
        )

    # Save combined splits
    assign_ids_and_save_detection_data(
        "trainval",
        data_infos["train"] + data_infos["val"],
        args.out_dir,
        args.data_name,
        cfg.classes,
    )
    assign_ids_and_save_detection_data(
        "all",
        data_infos["train"] + data_infos["val"] + data_infos["test"],
        args.out_dir,
        args.data_name,
        cfg.classes,
    )


if __name__ == "__main__":
    main()
