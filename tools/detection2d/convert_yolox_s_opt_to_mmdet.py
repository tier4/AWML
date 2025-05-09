# convert yolox_s_opt weight to mmdet weight

import argparse
import os
import re
import sys
from collections import OrderedDict
from subprocess import call
from urllib import request

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import yolox_s_opt_to_mmdet_key


def yolox_to_mmdet_key(key):
    x = key
    x = re.sub(r"backbone.backbone", r"backbone", x)
    x = re.sub(r"stem.down1", r"stem.conv.0", x)
    x = re.sub(r"stem.conv2", r"stem.conv.1", x)
    x = re.sub(r"(?<=darknet)[1-9]", lambda exp: str(int(exp.group(0)) - 1), x)
    x = re.sub(r"dark(?=[0-9].[0-9].)", r"stage", x)
    x = re.sub(r"(?<=stage)[1-9]", lambda exp: str(int(exp.group(0)) - 1), x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.1|4.2).)conv1", r"main_conv", x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.1|4.2).)conv2", r"short_conv", x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.1|4.2).)conv3", r"final_conv", x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.1|4.2).)m(?=\.)", r"blocks", x)
    x = re.sub(r"backbone.lateral_conv(?=[0-9])", r"neck.reduce_layers.", x)
    x = re.sub(r"(backbone.C3_(?=p[3-4]))", r"neck.top_down_blocks.", x)
    x = re.sub(r"p4.conv1", r"0.main_conv", x)
    x = re.sub(r"p4.conv2", r"0.short_conv", x)
    x = re.sub(r"p4.conv3", r"0.final_conv", x)
    x = re.sub(r"p4.m", r"0.blocks", x)
    x = re.sub(r"backbone.reduce_conv(?=[0-9]\.)", r"neck.reduce_layers.", x)
    x = re.sub(r"p3.conv1", r"1.main_conv", x)
    x = re.sub(r"p3.conv2", r"1.short_conv", x)
    x = re.sub(r"p3.conv3", r"1.final_conv", x)
    x = re.sub(r"p3.m", r"1.blocks", x)
    x = re.sub(r"backbone.bu_conv2", r"neck.downsamples.0", x)
    x = re.sub(r"(backbone.C3_(?=n[3-4]))", r"neck.bottom_up_blocks.", x)
    x = re.sub(r"n3.conv1", r"0.main_conv", x)
    x = re.sub(r"n3.conv2", r"0.short_conv", x)
    x = re.sub(r"n3.conv3", r"0.final_conv", x)
    x = re.sub(r"n3.m", r"0.blocks", x)
    x = re.sub(r"backbone.bu_conv1", r"neck.downsamples.1", x)
    x = re.sub(r"n4.conv1", r"1.main_conv", x)
    x = re.sub(r"n4.conv2", r"1.short_conv", x)
    x = re.sub(r"n4.conv3", r"1.final_conv", x)
    x = re.sub(r"n4.m", r"1.blocks", x)
    x = re.sub(r"head.cls_convs", r"bbox_head.multi_level_cls_convs", x)
    x = re.sub(r"head.reg_convs", r"bbox_head.multi_level_reg_convs", x)
    x = re.sub(r"head.cls_preds", r"bbox_head.multi_level_conv_cls", x)
    x = re.sub(r"head.reg_preds", r"bbox_head.multi_level_conv_reg", x)
    x = re.sub(r"head.obj_preds", r"bbox_head.multi_level_conv_obj", x)

    x = re.sub(r"head.stems", r"neck.out_convs", x)

    return x


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
        # mmdet_key = yolox_s_opt_to_mmdet_key(yolox_key)
        # if mmdet_key in new_state_dict["state_dict"]:
        #     print(f"duplicate keys:{mmdet_key}")
        #     assert False
        mmdet_key = yolox_to_mmdet_key(yolox_key)
        if mmdet_key in new_state_dict["state_dict"]:
            print(f"duplicate keys:{mmdet_key}")
            assert False
        new_state_dict["state_dict"][mmdet_key] = official_ckpt["model"][yolox_key]

    torch.save(new_state_dict, modified_official_ckpt_path)

    return modified_official_ckpt_path


def convert_yolox_checkpoint(args):

    modified_official_ckpt_path = create_yolox_checkpoint(args.yolox_ckpt, args.mmdet_ckpt)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLOX checkpoint to ONNX")

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
    # create_yolox_checkpoint(args.yolox_ckpt, args.mmdet_ckpt)
    convert_yolox_checkpoint(args)


if __name__ == "__main__":
    main()
