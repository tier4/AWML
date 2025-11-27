import argparse
import os
import re
import shutil
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

sys.path.append(".")
from tools.detection2d.deploy_yolox import add_efficientnms_trt


def current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def create_yolox_checkpoint(autoware_ml_ckpt: str, work_dir: str):
    """
    Based on specified model, download the tier4 yolox checkpoint and update the weights with autoware_ml_ckpt
    and save to work_dir
    Args:
        autoware_ml_ckpt (str): path to the autoware_ml yolox checkpoint
        work_dir (str): path to save the modified yolox checkpoint
    """

    def get_class_num(mmdet_ckpt):
        cls_tensor = mmdet_ckpt["bbox_head.multi_level_conv_cls.0.weight"]
        return cls_tensor.shape[0]

    tmp_dir = os.path.join(work_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    tier4_ckpt_save_path = os.path.join(work_dir, "temp_ckpt.pth")
    modified_tier4_ckpt_path = os.path.join(tmp_dir, "yolox_s_opt_modified.pth")
    modified_tier4_ckpt_path = os.path.abspath(modified_tier4_ckpt_path)
    if not os.path.isfile(tier4_ckpt_save_path):
        # request.urlretrieve(url, tier4_ckpt_save_path)
        print("tier4 ckpt {} does not exist.".format(tier4_ckpt_save_path))

    tier4_ckpt = torch.load(tier4_ckpt_save_path, weights_only=False)

    mmdet_ckpt = torch.load(autoware_ml_ckpt, map_location="cuda:0", weights_only=False)

    if "state_dict" in mmdet_ckpt.keys():
        mmdet_ckpt = mmdet_ckpt["state_dict"]
    class_num = get_class_num(mmdet_ckpt)

    new_state_dict = OrderedDict()
    new_state_dict["model"] = {}

    for yolox_key in tier4_ckpt["model"].keys():
        mmdet_key = yolox_s_opt_to_mmdet_key(yolox_key)
        new_state_dict["model"][yolox_key] = mmdet_ckpt[mmdet_key]
    torch.save(new_state_dict, modified_tier4_ckpt_path)
    return modified_tier4_ckpt_path, class_num


def create_temp_yolox_checkpoint(work_dir: str, exp_file: str):
    """
    Based on specified model, download the tier4 yolox checkpoint and update the weights with autoware_ml_ckpt
    and save to work_dir
    Args:
        autoware_ml_ckpt (str): path to the autoware_ml yolox checkpoint
        work_dir (str): path to save the modified yolox checkpoint
    """
    # generate yolox exp config file
    tmp_path = os.path.join(work_dir, "tmp_exp.py")
    shutil.copy(exp_file, tmp_path)

    # generate script to save dummy yolox checkpoint
    src_script = "create_yolox_dummy_checkpoint.py"
    dest_path = os.path.join(work_dir, src_script)
    shutil.copy(os.path.join(f"{current_dir()}/sample", src_script), dest_path)

    call(
        f"python {src_script} -f tmp_exp.py",
        cwd=work_dir,
        shell=True,
    )


def install_t4_yolox(work_dir: str) -> str:
    """
    download the tier4 yolox codes and install the environment

    Args:
        work_dir (str): path to the workdir

    Returns:
        str: directory the tier4 yolox is installed
    """
    yolox_dir = os.path.join(work_dir, "YOLOX-T4")
    if not os.path.isdir(yolox_dir):
        url = "https://github.com/tier4/YOLOX.git"
        commit_id = "d1c7ab7173803e72e497d2d3646adb24497b052d"  #  edge_ai_optimization, Mar 8, 2024
        call(f"git clone {url} {yolox_dir}", cwd="./", shell=True)
        call(f"git checkout {commit_id}", cwd=yolox_dir, shell=True)
        call("python setup.py develop", cwd=yolox_dir, shell=True)

        commit_id = "9f385b74a9f42151d5f44021ebbc0f2c733091cf"
        call(
            f"pip3 install git+https://github.com/wep21/yolox_onnx_modifier.git@{commit_id}", cwd=yolox_dir, shell=True
        )
    else:
        print("yolox tier4 package exists. skip clone and install")
    return yolox_dir


def export_onnx(
    yolox_dir: str,
    modified_tier4_ckpt_path: str,
    class_num: int,
    output_onnx_file: str,
    nms: bool,
    dynamic: bool,
    batch_size: int,
):
    """
    export the pytorch yolox model to onnx format
    Args:
        yolox_dir (str): _description_
        modified_tier4_ckpt_path (str): _description_
        class_num (int): _description_
        output_onnx_file (str): _description_
        dynamic (bool): _description_
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f".:~/{yolox_dir}:" + env.get("PYTHONPATH", "")

    call(
        f"python tools/export_onnx.py\
        --output-name {output_onnx_file}\
        -f tmp_exp.py \
        -c {modified_tier4_ckpt_path}\
        {'--decode_in_inference' if nms else ''}\
        {'--dynamic' if dynamic else ''}\
        --batch-size {batch_size}",
        cwd=yolox_dir,
        shell=True,
        env=env,
    )


def convert_yolox_checkpoint(args):
    print("*" * 20 + f"install tier4 yolox package to {args.work_dir}" + "*" * 20)
    yolox_dir = install_t4_yolox(args.work_dir)

    print("*" * 20 + "create temp tier4 yolox checkpoint" + "*" * 20)
    create_temp_yolox_checkpoint(yolox_dir, args.yolox_exp_file)

    print("*" * 20 + "convert mmdet to tier4 yolox checkpoint" + "*" * 20)
    modified_tier4_ckpt_path, class_num = create_yolox_checkpoint(args.autoware_ml_ckpt, yolox_dir)

    print("*" * 20 + "converting to onnx" + "*" * 20)
    output_onnx_file = args.output_onnx_file
    if output_onnx_file is None:
        output_onnx_file = f"yolox_s_opt_mmdet.onnx"
    output_onnx_file = os.path.abspath(os.path.join(args.work_dir, output_onnx_file))

    export_onnx(
        yolox_dir,
        modified_tier4_ckpt_path,
        class_num,
        output_onnx_file,
        args.nms,
        args.dynamic,
        args.batch_size,
    )

    if args.nms:
        print("*" * 20 + "add EfficientNMS_TRT" + "*" * 20)
        add_efficientnms_trt(
            output_onnx_file, output_onnx_file, args.max_output_boxes, args.iou_threshold, args.score_threshold
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLOX checkpoint to ONNX")

    parser.add_argument(
        "autoware_ml_ckpt",
        help="Model checkpoint",
    )

    parser.add_argument(
        "yolox_exp_file",
        help="YOLOX experiment configuration file",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="static inference batch size",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="whether the input shape should be dynamic or not",
    )
    parser.add_argument(
        "--max_output_boxes",
        type=int,
        default=100,
        help="max number of output boxes",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.65,
        help="NMS iou threshold",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="score threshold",
    )
    parser.add_argument(
        "--output_onnx_file",
        type=str,
        default=None,
        help="output onnx file name",
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs",
        help="the directory to save the converted checkpoint",
    )
    parser.add_argument(
        "--nms", action="store_true", help="whether add a efficientNMS plugin to the head of the model"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    convert_yolox_checkpoint(args)


if __name__ == "__main__":
    main()
