# convert yolox_s_opt weight to mmdet weight

import argparse
import os
import sys
import re
from collections import OrderedDict
from subprocess import call
from urllib import request

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import yolox_s_opt_to_mmdet_key


def create_yolox_checkpoint(yolox_ml_ckpt: str, modified_official_ckpt_path):
    """
    Based on specified model, download the official yolox checkpoint and update the weights with autoware_ml_ckpt
    and save to work_dir
    Args:
        autoware_ml_ckpt (str): path to the autoware_ml yolox checkpoint
        model (str): yolox model name
        work_dir (str): path to save the modified yolox checkpoint
    """

    def get_class_num(mmdet_ckpt):
        cls_tensor = mmdet_ckpt["bbox_head.multi_level_conv_cls.0.weight"]
        return cls_tensor.shape[0]

    official_ckpt_save_path = os.path.join(yolox_ml_ckpt)
    official_ckpt = torch.load(official_ckpt_save_path)

    new_state_dict = OrderedDict()
    new_state_dict["state_dict"] = {}

    for yolox_key in official_ckpt["model"].keys():
        mmdet_key = yolox_s_opt_to_mmdet_key(yolox_key)
        if mmdet_key in new_state_dict["state_dict"]:
            print(f"duplicate keys:{mmdet_key}")
            assert False
        new_state_dict["state_dict"][mmdet_key] = official_ckpt["model"][yolox_key]

    torch.save(new_state_dict, modified_official_ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLOX_s_opt checkpoint to ONNX")

    parser.add_argument(
        "yolox_ckpt",
        help="Model checkpoint",
    )

    parser.add_argument(
        "mmdet_ckpt",
        help="Model checkpoint",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    create_yolox_checkpoint(args.yolox_ckpt, args.mmdet_ckpt)


if __name__ == "__main__":
    main()
