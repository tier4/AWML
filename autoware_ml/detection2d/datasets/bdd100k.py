import copy
import os.path as osp
from typing import Dict, List, Optional

import numpy as np
from mmengine.fileio import load
from mmengine.utils import track_iter_progress
from mmdet.datasets import CocoDataset, BaseDetDataset
from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO


@DATASETS.register_module()
class BDD100kDataset(CocoDataset):
    """Dataset class for BDD100k based on CocoDataset"""

    CLASSES = (
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    )

    METAINFO = dict(classes=CLASSES)
    NAME_MAPPING = None

    def load_data_list(self) -> List[dict]:
        self.coco = COCO()
        categories: List[dict] = [
            dict(id=i, name=cat_name) for i, cat_name in enumerate(self.CLASSES)
        ]
        self.coco.dataset["categories"] = categories
        self.coco.dataset["images"] = []
        self.coco.dataset["annotations"] = []

        cat2id: Dict[str, int] = {k_v["name"]: k_v["id"] for k_v in categories}

        data_infos: List[dict] = []
        width: int = 1280
        height: int = 720

        ann_infos: List[dict] = load(self.ann_file)
        for i, ann in enumerate(track_iter_progress(ann_infos)):
            if ann.get("labels") is None:
                continue

            image = dict(
                file_name=osp.join(self.data_prefix["img"], ann["name"]),
                height=height,
                width=width,
                id=i,
            )
            bboxes = []
            labels = []
            for label_info in ann["labels"]:
                category_name: str = (
                    self.NAME_MAPPING[label_info["category"]]
                    if self.NAME_MAPPING is not None
                    else label_info["category"]
                )

                if category_name not in self.CLASSES:
                    continue

                x1: float = label_info["box2d"]["x1"]
                y1: float = label_info["box2d"]["y1"]
                x2: float = label_info["box2d"]["x2"]
                y2: float = label_info["box2d"]["y2"]

                coco_ann = dict(
                    iscrowd=0,
                    image_id=image["id"],
                    file_name=osp.join(self.data_prefix["img"], ann["name"]),
                    category_id=cat2id[category_name],
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    area=float((x2 - x1) * (y2 - y1)),
                    segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
                    id=len(self.coco.dataset["annotations"]),
                )
                self.coco.dataset["annotations"].append(coco_ann)
                bboxes.append([x1, x2, y1, y2])
                labels.append(cat2id[category_name])

            self.coco.dataset["images"].append(image)

            data_infos.append(
                dict(
                    id=i,
                    filename=osp.join(self.data_prefix["img"], ann["name"]),
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes, dtype=np.float32),
                        lables=np.array(labels, dtype=np.int64),
                    ),
                )
            )

        self.coco.createIndex()
        self.cat_ids = self.coco.get_cat_ids()
        self.img_ids = self.coco.get_img_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        data_list = []
        total_ann_ids = []
        for img_id in self.img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info["img_id"] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                {"raw_ann_info": raw_ann_info, "raw_img_info": raw_img_info}
            )
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
