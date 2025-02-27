import argparse
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn
from mmcv.parallel import MMDataParallel

# Additions
from mmcv.runner import load_checkpoint, save_checkpoint
from mmcv.runner.fp16_utils import auto_fp16
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from onnxsim import simplify
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from torchpack.utils.config import configs

from .lean.exptool import export_onnx
from .lean.funcs import fuse_relu_only, layer_fusion_bn, layer_fusion_bn_relu
from .lean.quantize import (
    calibrate_model,
    disable_quantization,
    initialize,
    print_quantizer_status,
    quantize_decoder,
    quantize_encoders_camera_branch,
    quantize_encoders_lidar_branch,
    replace_to_quantization_module,
    set_quantizer_fast,
)

# from tools.onnx.lean.train import qat_train


def parse_args():
    parser = argparse.ArgumentParser(description="Export bevfusion model")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--input_data", type=str, default="runs/example-data/example-data.pth")
    parser.add_argument("--quantization", type=str, default="fp16", help="fp16, int8")
    # SCN num of input channels
    parser.add_argument("--in-channel", type=int, default=5)
    # help="Transfer the coordinate order of the index from xyz to zyx",
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--calibrate_batch", type=int, default=300)
    args = parser.parse_args()
    return args


def divide_file_path(full_path):
    """
    Args
        full_path (str): './dir/subdir/filename.ext.ext2'
    Return
        ["./dir/subdir", "filename.ext", "subdir", "filename" "ext.ext2" ]
    """

    # ./dir/subdir, filename.ext
    dir_name, base_name = os.path.split(full_path)
    # subdir
    subdir_name = os.path.basename(os.path.dirname(full_path))
    # filename # .ext
    basename_without_ext, extension = base_name.split(".", 1)

    return dir_name, base_name, subdir_name, basename_without_ext, extension


def fuse_conv_bn(module):
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, QuantConv2d) or isinstance(
            child, nn.Conv2d
        ):  # or isinstance(child, QuantConvTranspose2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


class SubclassCameraModule(nn.Module):

    def __init__(self, model):
        super(SubclassCameraModule, self).__init__()
        self.model = model

    def forward(self, img, depth):
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        feat = self.model.encoders.camera.backbone(img)
        feat = self.model.encoders.camera.neck(feat)
        if not isinstance(feat, torch.Tensor):
            feat = feat[0]

        BN, C, H, W = map(int, feat.size())
        feat = feat.view(B, int(BN / B), C, H, W)

        def get_cam_feats(self, x, d):
            B, N, C, fH, fW = map(int, x.shape)
            d = d.view(B * N, *d.shape[2:])
            x = x.view(B * N, C, fH, fW)

            d = self.dtransform(d)
            x = torch.cat([d, x], dim=1)
            x = self.depthnet(x)

            depth = x[:, : self.D].softmax(dim=1)
            feat = x[:, self.D : (self.D + self.C)].permute(0, 2, 3, 1)
            return feat, depth

        return get_cam_feats(self.model.encoders.camera.vtransform, feat, depth)


class SubclassHeadBBox(nn.Module):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        # ["batch_score", "batch_rot", "batch_dim", "batch_center", "batch_height", "batch_vel"]
        self.classes_eye = nn.Parameter(torch.eye(parent.heads.object.num_classes).float())

    @staticmethod
    @auto_fp16(apply_to=("inputs", "classes_eye"))
    def head_forward(self, inputs, classes_eye):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = int(inputs.shape[0])
        lidar_feat = self.shared_conv(inputs)

        ################################################
        # image to BEV
        ################################################
        lidar_feat_flatten = lidar_feat.view(batch_size, int(lidar_feat.shape[1]), -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.to(lidar_feat.dtype).repeat(batch_size, 1, 1).to(lidar_feat.device)

        ################################################
        # image guided query initialization
        ################################################
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if len(classes_eye[0]) == 10:
            print("Nuscenes model")
            # local_max[:,8,] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            # local_max[:,9,] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
            local_max[:, 8] = heatmap[:, 8]
            local_max[:, 9] = heatmap[:, 9]

        elif len(classes_eye[0]) == 3:  # for Pedestrian & Cyclist in Waymo
            print("Waymo model")
            # local_max[
            #     :,
            #     1,
            # ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            # local_max[
            #     :,
            #     2,
            # ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            local_max[:, 1] = heatmap[:, 1]
            local_max[:, 2] = heatmap[:, 2]
        elif len(classes_eye[0]) == 5:
            print("T4dataset model")
            # local_max[:, 3] = heatmap[:, 3]
            # local_max[:, 4] = heatmap[:, 4]

        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, int(heatmap.shape[1]), -1)
        # top #num_proposals among all classes
        # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
        #     ..., : self.num_proposals
        # ]
        top_proposals = heatmap.view(batch_size, -1).topk(k=self.num_proposals, dim=-1, largest=True)[1]
        top_proposals_class = top_proposals // int(heatmap.shape[-1])
        top_proposals_index = top_proposals % int(heatmap.shape[-1])
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        # self.one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
        #     0, 2, 1
        # ).half()
        self.one_hot = classes_eye.index_select(0, top_proposals_class.view(-1))[None].permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(self.one_hot)
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        ################################################
        # transformer decoder layer (LiDAR feature as K,V)
        ################################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)

            # Prediction
            print(self.prediction_heads[i])
            res_layer = self.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        ################################################
        # transformer decoder layer (img feature as K,V)
        ################################################
        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]["dense_heatmap"] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return ret_dicts[-1]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return new_res

    def get_bboxes(self, preds_dict, one_hot):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        # batch_score = preds_dict["heatmap"][..., -self.num_proposals :].sigmoid()
        batch_score = preds_dict["heatmap"].sigmoid()
        # if self.loss_iou.loss_weight != 0:
        #    batch_score = torch.sqrt(batch_score * preds_dict['iou'][..., -self.num_proposals:].sigmoid())
        # one_hot = F.one_hot(
        #     query_labels, num_classes=num_classes
        # ).permute(0, 2, 1)
        batch_score = batch_score * preds_dict["query_heatmap_score"] * one_hot
        # batch_center = preds_dict["center"][..., -self.num_proposals :]
        # batch_height = preds_dict["height"][..., -self.num_proposals :]
        # batch_dim = preds_dict["dim"][..., -self.num_proposals :]
        # batch_rot = preds_dict["rot"][..., -self.num_proposals :]
        # batch_vel = None
        # if "vel" in preds_dict:
        #     batch_vel = preds_dict["vel"][..., -self.num_proposals :]
        batch_center = preds_dict["center"]
        batch_height = preds_dict["height"]
        batch_dim = preds_dict["dim"]
        batch_rot = preds_dict["rot"]
        batch_vel = None
        if "vel" in preds_dict:
            batch_vel = preds_dict["vel"]

        return [
            batch_score,
            batch_rot,
            batch_dim,
            batch_center,
            batch_height,
            batch_vel,
        ]

    def forward(self, x):
        for type, head in self.parent.heads.items():
            if type == "object":
                pred_dict = self.head_forward(head, x, self.classes_eye)
                return self.get_bboxes(pred_dict, head.one_hot)
            else:
                raise ValueError(f"unsupported head: {type}")


class SubclassFuser(nn.Module):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @auto_fp16(apply_to=("features",))
    def forward(self, features):
        if self.parent.fuser is not None:
            x = self.parent.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.parent.decoder["backbone"](x)
        x = self.parent.decoder["neck"](x)
        return x[0]


def make_ptq_model(args, model, cfg, save_root):
    # Create model
    quantize_encoders_lidar_branch(model.encoders.lidar.backbone)
    quantize_encoders_camera_branch(model.encoders.camera)
    replace_to_quantization_module(model.fuser)
    quantize_decoder(model.decoder)
    model.encoders.lidar.backbone = layer_fusion_bn(model.encoders.lidar.backbone)
    model = fuse_conv_bn(model)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Load dataset
    dataset_train = build_dataset(cfg.data.train)
    dataset_test = build_dataset(cfg.data.test)
    distributed = False
    data_loader_train = build_dataloader(
        dataset_train,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=distributed,
        seed=cfg.seed,
    )
    print("DataLoad Info:", data_loader_train.batch_size, data_loader_train.num_workers)

    initialize()
    save_path = os.path.job(save_root, "bevfusion_ptq.pth")
    os.makedirs(save_root, exist_ok=True)

    # set random seeds
    if cfg.seed is not None:
        print(f"Set random seed to {cfg.seed}, " f"deterministic mode: {cfg.deterministic}")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    print("train nums:{} val nums:{}".format(len(dataset_train), len(dataset_test)))

    ## Calibrate
    print("🔥 start calibrate 🔥 ")
    set_quantizer_fast(model)
    calibrate_model(model, data_loader_train, 0, None, args.calibrate_batch)
    disable_quantization(model.module.encoders.lidar.backbone.conv_input).apply()
    disable_quantization(model.module.decoder.neck.deblocks[0][0]).apply()
    print_quantizer_status(model)

    print(f"Done due to ptq only! Save checkpoint to {save_path}")
    model.module.encoders.lidar.backbone = fuse_relu_only(model.module.encoders.lidar.backbone)
    torch.save(model, save_path)


def make_camera_onnx(args, model, cfg, save_root, points, img):
    if args.quantization == "fp16":
        disable_quantization(model).apply()

    camera_model = SubclassCameraModule(model)
    camera_model.cuda().eval()
    depth = torch.zeros(len(points), img.shape[1], 1, img.shape[-2], img.shape[-1]).cuda()

    downsample_model = model.encoders.camera.vtransform.downsample
    downsample_model.cuda().eval()
    downsample_in = torch.zeros(1, 80, 360, 360).cuda()

    with torch.no_grad():
        camera_backbone_onnx = f"{save_root}/camera.backbone.onnx"
        camera_vtransform_onnx = f"{save_root}/camera.vtransform.onnx"
        TensorQuantizer.use_fb_fake_quant = True
        torch.onnx.export(
            camera_model,
            (img, depth),
            camera_backbone_onnx,
            input_names=["img", "depth"],
            output_names=["camera_feature", "camera_depth_weights"],
            opset_version=13,
            do_constant_folding=True,
        )

        onnx_orig = onnx.load(camera_backbone_onnx)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_simp, camera_backbone_onnx)
        print(f"The export is completed. ONNX save as {camera_backbone_onnx}")

        torch.onnx.export(
            downsample_model,
            downsample_in,
            camera_vtransform_onnx,
            input_names=["feat_in"],
            output_names=["feat_out"],
            opset_version=13,
            do_constant_folding=True,
        )
        print(f"The export is completed. ONNX save as {camera_vtransform_onnx}")


def make_fuser_onnx(args, model, cfg, save_root, points, img):
    if args.quantization == "fp16":
        disable_quantization(model).apply()

    model.eval()
    fuser = SubclassFuser(model).cuda()
    TensorQuantizer.use_fb_fake_quant = True
    with torch.no_grad():
        camera_features = torch.randn(1, 80, 180, 180).cuda()
        lidar_features = torch.randn(1, 256, 180, 180).cuda()

        fuser_onnx_path = f"{save_root}/fuser.onnx"
        torch.onnx.export(
            fuser,
            [camera_features, lidar_features],
            fuser_onnx_path,
            opset_version=13,
            input_names=["camera", "lidar"],
            output_names=["middle"],
        )
        print(f"The export is completed. ONNX save as {fuser_onnx_path}")


def make_head_onnx(args, model, cfg, save_root, points, img):
    if args.quantization == "fp16":
        disable_quantization(model).apply()

    model.eval()
    headbbox = SubclassHeadBBox(model).cuda().half()
    TensorQuantizer.use_fb_fake_quant = True

    with torch.no_grad():
        boxhead_onnx_path = f"{save_root}/head.bbox.onnx"
        head_input = torch.randn(1, 512, 180, 180).cuda().half()
        torch.onnx.export(
            headbbox,
            head_input,
            f"{save_root}/head.bbox.onnx",
            opset_version=13,
            input_names=["middle"],
            output_names=["score", "rot", "dim", "reg", "height", "vel"],
        )
        print(f"The export is completed. ONNX save as {boxhead_onnx_path}")


def make_lidar_onnx(args, model, cfg, save_root, points, img):
    if args.inverse:
        lidar_backbone_onnx_name = "lidar.backbone.zyx.onnx"
    else:
        lidar_backbone_onnx_name = "lidar.backbone.xyz.onnx"
    lidar_backbone_onnx_path = os.path.join(save_root, lidar_backbone_onnx_name)
    lidar_backbone_model = layer_fusion_bn_relu(model.encoders.lidar.backbone)
    lidar_backbone_model.eval().cuda().half()
    disable_quantization(lidar_backbone_model).apply()

    for name, module in lidar_backbone_model.named_modules():
        module.precision = "int8"
        module.output_precision = "int8"
    lidar_backbone_model.conv_input.precision = "fp16"
    lidar_backbone_model.conv_out.output_precision = "fp16"
    voxels = torch.zeros(1, args.in_channel).cuda().half()
    coors = torch.zeros(1, 4).int().cuda()
    batch_size = 1
    export_onnx(
        lidar_backbone_model,
        voxels,
        coors,
        batch_size,
        args.inverse,
        lidar_backbone_onnx_path,
    )


if __name__ == "__main__":
    args = parse_args()

    # path
    dir_name, base_name, _, _, _ = divide_file_path(args.checkpoint)
    config_path = os.path.join(dir_name, "configs.yaml")
    save_root = os.path.join(dir_name, f"onnx_{args.quantization}")
    os.makedirs(save_root, exist_ok=True)

    # Load config
    configs.load(config_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_path)

    # Load model
    model_ = build_model(cfg.model)
    checkpoint = load_checkpoint(model_, args.checkpoint, map_location="cpu")

    # PTQ
    # make_ptq_model(args, model, cfg, dir_name)

    # Make onnx
    data = torch.load(args.input_data)
    img = data["img"].data[0].cuda()
    points = [i.cuda() for i in data["points"].data[0]]
    make_camera_onnx(args, model_, cfg, save_root, points, img)
    make_fuser_onnx(args, model_, cfg, save_root, points, img)
    make_head_onnx(args, model_, cfg, save_root, points, img)
    make_lidar_onnx(args, model_, cfg, save_root, points, img)
