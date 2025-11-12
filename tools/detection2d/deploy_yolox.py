import argparse
import os
import re
from collections import OrderedDict
from subprocess import call
from urllib import request

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch


def yolox_to_mmdet_key(key):
    x = key
    x = re.sub(r"backbone.backbone", r"backbone", x)
    x = re.sub(r"(?<=darknet)[1-9]", lambda exp: str(int(exp.group(0)) - 1), x)
    x = re.sub(r"dark(?=[0-9].[0-9].)", r"stage", x)
    x = re.sub(r"(?<=stage)[1-9]", lambda exp: str(int(exp.group(0)) - 1), x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.2).)conv1", r"main_conv", x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.2).)conv2", r"short_conv", x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.2).)conv3", r"final_conv", x)
    x = re.sub(r"(?<=stage(1.1|2.1|3.1|4.2).)m(?=\.)", r"blocks", x)
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


def create_yolox_checkpoint(autoware_ml_ckpt: str, model: str, work_dir: str):
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

    # download official yolox checkpoint
    model2url = {
        "yolox-s": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
        "yolox-m": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",
        "yolox-l": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
        "yolox-x": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
    }
    url = model2url[model]
    tmp_dir = os.path.join(work_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    official_ckpt_save_path = os.path.join(tmp_dir, model + ".pth")
    modified_official_ckpt_path = os.path.join(tmp_dir, model + "_modified.pth")
    modified_official_ckpt_path = os.path.abspath(modified_official_ckpt_path)
    if not os.path.isfile(official_ckpt_save_path):
        request.urlretrieve(url, official_ckpt_save_path)

    official_ckpt = torch.load(official_ckpt_save_path, weights_only=False)

    mmdet_ckpt = torch.load(autoware_ml_ckpt, map_location="cuda:0", weights_only=False)

    if "state_dict" in mmdet_ckpt.keys():
        mmdet_ckpt = mmdet_ckpt["state_dict"]
    class_num = get_class_num(mmdet_ckpt)

    new_state_dict = OrderedDict()
    new_state_dict["model"] = {}

    for yolox_key in official_ckpt["model"].keys():
        mmdet_key = yolox_to_mmdet_key(yolox_key)
        new_state_dict["model"][yolox_key] = mmdet_ckpt[mmdet_key]
    torch.save(new_state_dict, modified_official_ckpt_path)
    return modified_official_ckpt_path, class_num


def install_official_yolox(work_dir: str) -> str:
    """
    download the official yolox codes and install the environment

    Args:
        work_dir (str): path to the workdir

    Returns:
        str: directory the official yolox is installed
    """
    yolox_dir = os.path.join(work_dir, "YOLOX")
    if not os.path.isdir(yolox_dir):
        url = "https://github.com/Megvii-BaseDetection/YOLOX.git"
        commit_id = "f00a798c8bf59f43ab557a2f3d566afa831c8887"  # Jul 30 2024 commit
        call(f"git clone {url} {yolox_dir}", cwd="./", shell=True)
        call(f"git checkout {commit_id}", cwd=yolox_dir, shell=True)
        call("python setup.py develop", cwd=yolox_dir, shell=True)

        commit_id = "9f385b74a9f42151d5f44021ebbc0f2c733091cf"
        call(
            f"pip3 install git+https://github.com/wep21/yolox_onnx_modifier.git@{commit_id}", cwd=yolox_dir, shell=True
        )
    else:
        print("yolox official package exists. skip clone and install")
    return yolox_dir


def export_onnx(
    yolox_dir: str,
    modified_official_ckpt_path: str,
    model: str,
    class_num: int,
    input_size: list,
    output_onnx_file: str,
    nms: bool,
    dynamic: bool,
    batch_size: int,
):
    """
    export the pytorch yolox model to onnx format
    Args:
        yolox_dir (str): _description_
        modified_official_ckpt_path (str): _description_
        model (str): _description_
        class_num (int): _description_
        input_size (list): _description_
        output_onnx_file (str): _description_
        dynamic (bool): _description_
    """
    model2depthWidth = {
        "yolox-s": (0.33, 0.50),
        "yolox-m": (0.67, 0.75),
        "yolox-l": (1.0, 1.0),
        "yolox-x": (1.33, 1.25),
    }
    # generate yolox exp config file
    tmp_exp_path = os.path.join(yolox_dir, "tmp_exp.py")
    with open(tmp_exp_path, "w") as f:
        f.write("import os\n")
        f.write("from yolox.exp import Exp as MyExp\n")
        f.write("class Exp(MyExp):\n")
        f.write("    def __init__(self):\n")
        f.write("        super(Exp, self).__init__()\n")
        f.write(f"        self.depth = {model2depthWidth[model][0]}\n")
        f.write(f"        self.width = {model2depthWidth[model][1]}\n")
        f.write(f"        self.test_size=({input_size[0]}, {input_size[1]})\n")
        f.write(f"        self.num_classes = {class_num}\n")
    call(
        f"python tools/export_onnx.py\
        --output-name {output_onnx_file}\
        -f {tmp_exp_path}\
        -c {modified_official_ckpt_path}\
        {'--decode_in_inference' if nms else ''}\
        {'--dynamic' if dynamic else ''}\
        --batch-size {batch_size}",
        cwd=yolox_dir,
        shell=True,
    )


def add_efficientnms_trt(
    input_onnx_file: str, output_onnx_file: str, max_output_boxes: int, iou_threshold: float, score_threshold: float
):
    """
    Add EfficientNMS_TRT plugin to the head of the network.
    Source modified from:
    `https://github.com/wep21/yolox_onnx_modifier`
    Args:
        input_onnx_file (str): source yolox onnx model generated with --decode_in_inference during export2onnx
        output_onnx_file (str): path to save modified onnx model
        max_output_boxes (int): maximum number of detections
        iou_threshold (float): NMS iou threshold
        score_threshold (float): minimum detection score
    """
    graph = gs.import_onnx(onnx.load(input_onnx_file))
    batch_size = graph.inputs[0].shape[0]
    # handle the tensor size problem caused by dynamic batch size
    if not isinstance(batch_size, int):
        for node in graph.nodes:
            if "/head/Reshape" in node.name:
                node.outputs[0].shape = [
                    batch_size,
                    node.inputs[0].shape[1],
                    node.inputs[0].shape[2] * node.inputs[0].shape[3],
                ]
            elif node.name in ["/head/Concat_6", "/head/Concat_24"]:
                s1 = node.inputs[0].shape
                s2 = node.inputs[1].shape
                s3 = node.inputs[2].shape
                node.outputs[0].shape = [batch_size, s1[1], s1[2] + s2[2] + s3[2]]
            elif node.name == "/head/Transpose":
                s1 = node.inputs[0].shape
                node.outputs[0].shape = [s1[0], s1[2], s1[1]]
            elif node.name in ["/head/Slice_3", "/head/Slice_4", "/head/Slice_5"]:
                start, stop, axis, interval = [int(inp.values[0]) for inp in node.inputs[1:5]]
                stop = min(stop, node.inputs[0].shape[axis])
                node.outputs[0].shape = [node.inputs[0].shape[0], node.inputs[0].shape[1], (stop - start) // interval]
            elif node.name in ["/head/Add", "/head/Exp", "/head/Mul", "/head/Mul_1"]:
                node.outputs[0].shape = node.inputs[0].shape
            else:
                for inp in node.inputs:
                    if isinstance(inp.shape[0], str) and "unk" in inp.shape[0]:
                        inp.shape[0] = batch_size
                for outp in node.outputs:
                    if isinstance(outp.shape[0], str) and "unk" in outp.shape[0]:
                        outp.shape[0] = batch_size

    scatter_node = graph.outputs[0].inputs[0]
    box_slice_starts = gs.Constant(name="box_slice_starts", values=np.array([0], dtype=np.int64))
    box_slice_ends = gs.Constant(name="box_slice_ends", values=np.array([4], dtype=np.int64))
    box_slice_axes = gs.Constant(name="box_slice_axes", values=np.array([2], dtype=np.int64))
    box_slice_steps = gs.Constant(name="box_slice_steps", values=np.array([1], dtype=np.int64))
    boxes = gs.Variable(
        name="boxes",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], scatter_node.outputs[0].shape[1], 4),
    )
    box_slice_node = gs.Node(
        "Slice",
        inputs=[
            scatter_node.outputs[0],
            box_slice_starts,
            box_slice_ends,
            box_slice_axes,
            box_slice_steps,
        ],
        outputs=[boxes],
    )
    graph.nodes.append(box_slice_node)
    class_slice_starts = gs.Constant(name="class_slice_starts", values=np.array([5], dtype=np.int64))
    class_slice_ends = gs.Constant(
        name="class_slice_ends", values=np.array([scatter_node.outputs[0].shape[2]], dtype=np.int64)
    )
    class_slice_axes = gs.Constant(name="class_slice_axes", values=np.array([2], dtype=np.int64))
    class_slice_steps = gs.Constant(name="class_slice_steps", values=np.array([1], dtype=np.int64))
    classes = gs.Variable(
        name="classes",
        dtype=np.float32,
        shape=(
            scatter_node.outputs[0].shape[0],
            scatter_node.outputs[0].shape[1],
            scatter_node.outputs[0].shape[2] - 5,
        ),
    )
    class_slice_node = gs.Node(
        "Slice",
        inputs=[
            scatter_node.outputs[0],
            class_slice_starts,
            class_slice_ends,
            class_slice_axes,
            class_slice_steps,
        ],
        outputs=[classes],
    )
    graph.nodes.append(class_slice_node)
    gather_indices = gs.Constant(name="gather_indices", values=np.array(4, dtype=np.int64))
    gather_output = gs.Variable(
        name="gather_output",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], scatter_node.outputs[0].shape[1]),
    )
    gather_node = gs.Node(
        "Gather",
        attrs={
            "axis": 2,
        },
        inputs=[scatter_node.outputs[0], gather_indices],
        outputs=[gather_output],
    )
    graph.nodes.append(gather_node)
    unsqueeze_output = gs.Variable(
        name="unsqueeze_output",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], scatter_node.outputs[0].shape[1], 1),
    )
    unsqueeze_node = gs.Node(
        "Unsqueeze",
        attrs={
            "axes": [2],
        },
        inputs=[gather_output],
        outputs=[unsqueeze_output],
    )
    graph.nodes.append(unsqueeze_node)
    scores = gs.Variable(
        name="scores",
        dtype=np.float32,
        shape=(
            scatter_node.outputs[0].shape[0],
            scatter_node.outputs[0].shape[1],
            scatter_node.outputs[0].shape[2] - 5,
        ),
    )
    mul_node = gs.Node(
        "Mul",
        inputs=[
            unsqueeze_output,
            classes,
        ],
        outputs=[scores],
    )
    graph.nodes.append(mul_node)
    num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=(scatter_node.outputs[0].shape[0], 1))
    detection_boxes = gs.Variable(
        name="detection_boxes",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], max_output_boxes, 4),
    )
    detection_scores = gs.Variable(
        name="detection_scores",
        dtype=np.float32,
        shape=(scatter_node.outputs[0].shape[0], max_output_boxes),
    )
    detection_classes = gs.Variable(
        name="detection_classes",
        dtype=np.int32,
        shape=(scatter_node.outputs[0].shape[0], max_output_boxes),
    )
    nms_node = gs.Node(
        "EfficientNMS_TRT",
        attrs={
            "background_class": -1,
            "box_coding": 1,
            "iou_threshold": iou_threshold,
            "max_output_boxes": max_output_boxes,
            "plugin_namespace": "",
            "plugin_version": "1",
            "score_activation": 0,
            "score_threshold": score_threshold,
        },
        inputs=[
            boxes,
            scores,
        ],
        outputs=[
            num_detections,
            detection_boxes,
            detection_scores,
            detection_classes,
        ],
    )
    graph.nodes.append(nms_node)
    graph.outputs = [
        num_detections,
        detection_boxes,
        detection_scores,
        detection_classes,
    ]
    graph.cleanup().toposort()
    onnx.save_model(gs.export_onnx(graph), output_onnx_file)


def convert_yolox_checkpoint(args):
    print("*" * 20 + "downloading official yolox checkpoint" + "*" * 20)
    modified_official_ckpt_path, class_num = create_yolox_checkpoint(args.autoware_ml_ckpt, args.model, args.work_dir)
    print("*" * 20 + f"install official yolox package to {args.work_dir}" + "*" * 20)
    yolox_dir = install_official_yolox(args.work_dir)
    print("*" * 20 + "converting to onnx" + "*" * 20)
    output_onnx_file = args.output_onnx_file
    if output_onnx_file is None:
        output_onnx_file = f"{args.model}.onnx"
    output_onnx_file = os.path.abspath(os.path.join(args.work_dir, output_onnx_file))
    export_onnx(
        yolox_dir,
        modified_official_ckpt_path,
        args.model,
        class_num,
        args.input_size,
        output_onnx_file,
        args.nms,
        args.dynamic,
        args.batch_size,
    )
    print("*" * 20 + "add EfficientNMS_TRT" + "*" * 20)
    if args.nms:
        add_efficientnms_trt(
            output_onnx_file, output_onnx_file, args.max_output_boxes, args.iou_threshold, args.score_threshold
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLOX checkpoint to ONNX")

    parser.add_argument(
        "autoware_ml_ckpt",
        help="Model checkpoint",
    )
    parser.add_argument("--input_size", type=int, nargs="+", help="input image size of the model")
    parser.add_argument(
        "--model",
        type=str,
        default="yolox-s",
        choices=["yolox-s", "yolox-m", "yolox-l", "yolox-x"],
        help="the type of yolox model",
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
    input_size = args.input_size
    if isinstance(input_size, int):
        input_size = [input_size, input_size]
    elif isinstance(input_size, list):
        assert input_size.__len__() == 2
    convert_yolox_checkpoint(args)


if __name__ == "__main__":
    main()
