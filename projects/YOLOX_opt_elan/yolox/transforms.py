import os.path as osp

import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ResizeSegMask:
    def __init__(self, size):
        self.size = size  # (H_out, W_out)

    def __call__(self, results):
        if "gt_seg_map" in results and results["gt_seg_map"] is not None:
            seg = results["gt_seg_map"]  # numpy array (H, W)
            seg = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
            seg = F.interpolate(seg, size=self.size, mode="nearest")
            results["gt_seg_map"] = seg.squeeze(0).squeeze(0).long().numpy()  # back to (H_out,W_out)
        return results


@TRANSFORMS.register_module()
class FixCityscapesPath(BaseTransform):
    def __init__(self, data_root, split="train"):
        self.data_root = data_root
        self.split = split

    def transform(self, results):
        img_path = results["img_path"]
        filename = osp.basename(img_path)

        seg_filename = filename.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
        city = filename.split("_")[0]
        seg_path = osp.join(self.data_root, "gtFine", self.split, city, seg_filename)

        results["seg_map_path"] = seg_path

        return results
