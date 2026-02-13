# Copyright (c) OpenMMLab. All rights reserved.

import logging
import os.path as osp
from typing import Optional

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from builder import ExportBuilder
from containers import TrtBevFusionCameraOnlyContainer, TrtBevFusionImageBackboneContainer, TrtBevFusionMainContainer
from data_classes import ModelData, SetupConfigs
from mmdeploy.core import RewriterContext
from mmdeploy.utils import (
    get_root_logger,
)


class Torch2OnnxExporter:

    def __init__(self, setup_configs: SetupConfigs, log_level: str):
        """Initialization of Torch2OnnxExporter."""
        self.setup_configs = setup_configs
        log_level = logging.getLevelName(log_level)
        self.logger = get_root_logger()
        self.logger.setLevel(log_level)
        self.output_prefix = osp.join(
            self.setup_configs.work_dir,
            osp.splitext(osp.basename(self.setup_configs.deploy_cfg.onnx_config.save_file))[0],
        )
        self.output_path = self.output_prefix + ".onnx"
        self.builder = ExportBuilder(self.setup_configs)

    def export(self) -> None:
        """
        Export Pytorch Model to ONNX.
        """
        self.logger.info(f"Export PyTorch model to ONNX: {self.output_path}.")

        # Build the model data and configs
        builder_data = self.builder.build()

        # Export the model
        self._export_model(
            model_data=builder_data.model_data,
            context_info=builder_data.context_info,
            patched_model=builder_data.patched_model,
            ir_configs=builder_data.ir_configs,
        )

        # Fix the ONNX graph
        self._fix_onnx_graph()

        self.logger.info(f"ONNX exported to {self.output_path}")

    def _export_model(
        self, model_data: ModelData, context_info: dict, patched_model: torch.nn.Module, ir_configs: dict
    ) -> None:
        """
        Export torch model to ONNX.
        Args:
          model_data (ModelData): Dataclass with data inputs.
          context_info (dict): Context when deploying to rewrite some configs.
          patched_model (torch.nn.Module): Patched Pytorch model.
          ir_configs (dict): Configs for intermediate representations in ONNX.
        """
        with RewriterContext(**context_info), torch.no_grad():
            image_feats = None
            if "img_backbone" in self.setup_configs.model_cfg.model:
                image_feats = self._export_image_backbone(model_data, ir_configs, patched_model)
                # If the image backbone feat is None, it's exported to ONNX and exit
                if image_feats is None:
                    return

            # Export the camera bev only network
            if self.setup_configs.module == "camera_bev_only_network":
                self._export_camera_bev_only_network(
                    model_data=model_data, ir_configs=ir_configs, patched_model=patched_model, image_feats=image_feats
                )

            # Export the main network with camera or lidar-only
            elif self.setup_configs.module == "main_body":
                self._export_main_body(
                    model_data=model_data, ir_configs=ir_configs, patched_model=patched_model, image_feats=image_feats
                )

    def _export_image_backbone(
        self, model_data: ModelData, ir_configs: dict, patched_model: torch.nn.Module
    ) -> Optional[torch.Tensor]:
        """Export the image backbone.

        Args:
          model_data (ModelData): Dataclass with data inputs.
          context_info (dict): Context when deploying to rewrite some configs.
          patched_model (torch.nn.Module): Patched Pytorch model.
          ir_configs (dict): Configs for intermediate representations in ONNX.

        Returns:
            Image feats.
        """
        data_preprocessor = model_data.input_metas["data_preprocessor"]
        model_inputs_data = model_data.model_inputs
        device = self.setup_configs.device

        imgs = model_inputs_data.imgs
        images_mean = data_preprocessor.mean.to(device)
        images_std = data_preprocessor.std.to(device)
        image_backbone_container = TrtBevFusionImageBackboneContainer(patched_model, images_mean, images_std)
        model_inputs = (imgs.to(device=device, dtype=torch.uint8),)

        if self.setup_configs.module == "image_backbone":
            torch.onnx.export(
                image_backbone_container,
                model_inputs,
                self.output_path,
                export_params=True,
                input_names=ir_configs["input_names"],
                output_names=ir_configs["output_names"],
                opset_version=ir_configs["opset_version"],
                dynamic_axes=ir_configs["dynamic_axes"],
                keep_initializers_as_inputs=ir_configs["keep_initializers_as_inputs"],
                verbose=ir_configs["verbose"],
            )
            self.logger.info(f"Image backbone exported to {self.output_path}")
            return

        image_feats = image_backbone_container(*model_inputs)
        self.logger.info(f"Converted Image backbone")
        return image_feats

    def _export_camera_bev_only_network(
        self,
        model_data: ModelData,
        ir_configs: dict,
        patched_model: torch.nn.Module,
        image_feats: Optional[torch.Tensor],
    ) -> None:
        """Export the camera bev only network to an ONNX file.

        Args:
          model_data (ModelData): Dataclass with data inputs.
          context_info (dict): Context when deploying to rewrite some configs.
          patched_model (torch.nn.Module): Patched Pytorch model.
          ir_configs (dict): Configs for intermediate representations in ONNX.
        """
        main_container = TrtBevFusionCameraOnlyContainer(patched_model)
        data_samples = model_data.input_metas["data_samples"]
        imgs = model_data.model_inputs.imgs
        lidar2img = model_data.model_inputs.lidar2img
        geom_feats = model_data.model_inputs.geom_feats
        kept = model_data.model_inputs.kept
        ranks = model_data.model_inputs.ranks
        indices = model_data.model_inputs.indices
        points = model_data.model_inputs.points
        img_aug_matrix = imgs.new_tensor(np.stack(data_samples[0].img_aug_matrix))
        device = self.setup_configs.device

        model_inputs = (
            lidar2img.to(device).float(),
            img_aug_matrix.to(device).float(),
            geom_feats.to(device).float(),
            kept.to(device),
            ranks.to(device).long(),
            indices.to(device).long(),
            image_feats,
        )

        if "points" in ir_configs["input_names"]:
            model_inputs += (points.to(device).float(),)

        torch.onnx.export(
            main_container,
            model_inputs,
            self.output_path.replace(".onnx", "_temp_to_be_fixed.onnx"),
            export_params=True,
            input_names=ir_configs["input_names"],
            output_names=ir_configs["output_names"],
            opset_version=ir_configs["opset_version"],
            dynamic_axes=ir_configs["dynamic_axes"],
            keep_initializers_as_inputs=ir_configs["keep_initializers_as_inputs"],
            verbose=ir_configs["verbose"],
        )
        self.logger.info(f"Camera bev only network exported to {self.output_path}")

    def _export_main_body(
        self,
        model_data: ModelData,
        ir_configs: dict,
        patched_model: torch.nn.Module,
        image_feats: Optional[torch.Tensor],
    ) -> None:
        """Export the main body (lidar-only or camera-lidar) to an ONNX file.

        Args:
          model_data (ModelData): Dataclass with data inputs.
          context_info (dict): Context when deploying to rewrite some configs.
          patched_model (torch.nn.Module): Patched Pytorch model.
          ir_configs (dict): Configs for intermediate representations in ONNX.
        """
        main_container = TrtBevFusionMainContainer(patched_model)
        data_samples = model_data.input_metas["data_samples"]
        voxels = model_data.model_inputs.voxels
        coors = model_data.model_inputs.coors
        num_points_per_voxel = model_data.model_inputs.num_points_per_voxel
        device = self.setup_configs.device
        model_inputs = (
            voxels.to(device),
            coors.to(device),
            num_points_per_voxel.to(device),
        )

        if image_feats is not None:
            imgs = model_data.model_inputs.imgs
            points = model_data.model_inputs.points
            lidar2img = model_data.model_inputs.lidar2img
            img_aug_matrix = imgs.new_tensor(np.stack(data_samples[0].img_aug_matrix))
            geom_feats = model_data.model_inputs.geom_feats
            kept = model_data.model_inputs.kept
            ranks = model_data.model_inputs.ranks
            indices = model_data.model_inputs.indices
            model_inputs += (
                points.to(device).float(),
                lidar2img.to(device).float(),
                img_aug_matrix.to(device).float(),
                geom_feats.to(device).float(),
                kept.to(device),
                ranks.to(device).long(),
                indices.to(device).long(),
                image_feats,
            )

        torch.onnx.export(
            main_container,
            model_inputs,
            self.output_path.replace(".onnx", "_temp_to_be_fixed.onnx"),
            export_params=True,
            input_names=ir_configs["input_names"],
            output_names=ir_configs["output_names"],
            opset_version=ir_configs["opset_version"],
            dynamic_axes=ir_configs["dynamic_axes"],
            keep_initializers_as_inputs=ir_configs["keep_initializers_as_inputs"],
            verbose=ir_configs["verbose"],
        )
        if image_feats is None:
            model_name = "lidar-only"
        else:
            model_name = "camera-lidar"
        self.logger.info(f"Main body network with {model_name} exported to {self.output_path}")

    def _fix_onnx_graph(self) -> None:
        """Fix the ONNX graph with an ONNX file."""
        self.logger.info("Attempting to fix the graph (TopK's K becoming a tensor)")
        model = onnx.load(self.output_path.replace(".onnx", "_temp_to_be_fixed.onnx"))
        graph = gs.import_onnx(model)

        # Fix TopK
        topk_nodes = [node for node in graph.nodes if node.op == "TopK"]
        assert len(topk_nodes) == 1
        topk = topk_nodes[0]
        k = self.setup_configs.model_cfg.get("num_proposals", None)
        if k is None:
            raise ValueError(f"num_proposals is not found in the model configs!")
        topk.inputs[1] = gs.Constant("K", values=np.array([k], dtype=np.int64))
        topk.outputs[0].shape = [1, k]
        topk.outputs[0].dtype = topk.inputs[0].dtype if topk.inputs[0].dtype else np.float32
        topk.outputs[1].shape = [1, k]
        topk.outputs[1].dtype = np.int64

        graph.cleanup().toposort()
        onnx.save_model(gs.export_onnx(graph), self.output_path)

        self.logger.info(f"(Fixed) ONNX exported to {self.output_path}")
