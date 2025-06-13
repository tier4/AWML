import argparse
import os
import os.path as osp
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import mmengine
import numpy as np
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


@dataclass
class DetectionData:
    metainfo: Dict[str, str]
    data_list: List[DataEntry] = field(default_factory=list)


def update_detection_data_annotations(
    data_list: Dict[str, DataEntry],
    object_ann: List[ObjectAnn],
    attributes: Dict[str, str],
    categories: Dict[str, str],
    class_mappings: Dict[str, str],
    allowed_classes: List[str],
) -> None:
    for ann in object_ann:
        class_name = class_mappings[categories[ann.category_token]]
        if class_name not in allowed_classes:
            continue
        bbox_label = allowed_classes.index(class_name)
        instance = Instance(
            bbox=ann.bbox,
            bbox_label=bbox_label,
            # TODO(someone): Please check this operation is correct!!!
            mask=[[int(x), int(y)] for y, x in zip(*np.where(ann.mask.decode() == 1))],
            extra_anns=[attributes[x] for x in ann.attribute_tokens],
        )
        data_list[ann.sample_data_token].instances.append(instance)


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
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="output directory of info file")
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
                scene_root_dir_path = get_scene_root_dir_path(
                    args.root_path,
                    dataset_version,
                    scene_id,
                )

                if not osp.isdir(scene_root_dir_path):
                    raise ValueError(f"{scene_root_dir_path} does not exist.")
                t4 = Tier4(
                    version="annotation",
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
                    attributes,
                    categories,
                    cfg.class_mappings,
                    cfg.classes,
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
