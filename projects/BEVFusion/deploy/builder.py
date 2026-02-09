import logging
import os.path as osp
from typing import Any

import numpy as np
import torch
from containers import TrtBevFusionCameraOnlyContainer, TrtBevFusionImageBackboneContainer, TrtBevFusionMainContainer
from data_classes import BackendConfigs, BuilderData, ModelData, ModelInputs, SetupConfigs
from mmdeploy.apis import build_task_processor
from mmdeploy.apis.onnx.passes import optimize_onnx
from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import (
    IR,
    Backend,
    get_backend,
    get_dynamic_axes,
    get_ir_config,
    get_onnx_config,
    get_root_logger,
    load_config,
)
from mmdet3d.registry import MODELS
from mmengine.registry import RUNNERS

from projects.BEVFusion.deploy.torch2onnx import backend


class ExportBuilder:

    def __init__(self, setup_configs: SetupConfigs):
        self.setup_configs = setup_configs

    def build(self):
        """Build the model.

        Returns:
            Model data.
        """
        # Build the model data
        model_data = self._build_model_data()

        # Build the backend configs
        backend = self._build_backend()

        # Build the optimize configs
        optimize = self._build_optimize_configs(backend)

        # Build the IR configs
        ir_configs = self._build_ir_configs()

        # Update the deploy config
        self._update_dpeloy_cfg(ir_configs, backend)

        # Build the intermediate representations
        ir = self._build_intermediate_representations(ir_configs)

        # Build the context info
        context_info = self._build_context_info(ir, ir_configs, backend, optimize)

        # Patch the model
        patched_model = self._build_patched_model(model_data, backend, ir)

        return BuilderData(
            model_data=model_data,
            ir_configs=ir_configs,
            context_info=context_info,
            patched_model=patched_model,
        )

    def _build_model_data(self):
        """Build the model.

        Args:
            setup_config: Setup configuration for the model.

        Returns:
            Model data.
        """
        data_preprocessor = MODELS.build(self.setup_configs.data_preprocessor_cfg)

        # load a sample
        runner = RUNNERS.build(self.setup_configs.model_cfg)
        runner.load_or_resume()
        data = runner.test_dataloader.dataset[self.setup_configs.sample_idx]

        # create model an inputs
        task_processor = build_task_processor(
            self.setup_configs.model_cfg, self.setup_configs.deploy_cfg, self.setup_configs.device
        )

        torch_model = task_processor.build_pytorch_model(self.setup_configs.checkpoint_path)
        data, model_inputs = task_processor.create_input(data, data_preprocessor=data_preprocessor, model=torch_model)

        if isinstance(model_inputs, list) and len(model_inputs) == 1:
            model_inputs = model_inputs[0]

        data_samples = data["data_samples"]
        input_metas = {"data_samples": data_samples, "mode": "predict", "data_preprocessor": data_preprocessor}

        (
            voxels,
            coors,
            num_points_per_voxel,
            points,
            camera_mask,
            imgs,
            lidar2img,
            cam2image,
            camera2lidar,
            geom_feats,
            kept,
            ranks,
            indices,
        ) = model_inputs

        return ModelData(
            model_inputs=ModelInputs(
                voxels=voxels,
                coors=coors,
                num_points_per_voxel=num_points_per_voxel,
                points=points,
                camera_mask=camera_mask,
                imgs=imgs,
                lidar2img=lidar2img,
                cam2image=cam2image,
                camera2lidar=camera2lidar,
                geom_feats=geom_feats,
                kept=kept,
                ranks=ranks,
                indices=indices,
            ),
            torch_model=torch_model,
            input_metas=input_metas,
        )

    @staticmethod
    def _add_or_update(cfg: dict, key: str, val: Any) -> None:
        if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val

    def update_deploy_cfg(self, ir_configs: dict, backend: Backend) -> None:
        """Update the deploy config.

        Args:
            ir_configs: IR configs.
            backend_configs: Backend configs.
        """
        self._add_or_update(self.setup_configs.deploy_cfg, "ir_config", ir_configs)
        self._add_or_update(self.setup_configs.deploy_cfg, "backend_config", dict(type=backend))

    def _build_patched_model(self, model_data: ModelData, backend: str, ir: IR) -> torch.nn.Module:
        """Build the patched model.

        Returns:
            Patched model.
        """
        patched_model = patch_model(model_data.torch_model, cfg=self.setup_configs.deploy_cfg, backend=backend, ir=ir)
        # Set Patched model to eval() for inference status
        patched_model.eval()
        patched_model.to(self.setup_configs.device)
        return patched_model

    def _build_backend(self) -> str:
        """Build the backend configs.

        Returns:
            Backend configs.
        """
        return get_backend(self.setup_configs.deploy_cfg).value

    def _build_optimize_configs(self, backend: str) -> dict:
        """Build the optimize configs.

        Returns:
            Optimize configs.
        """
        optimize = self.setup_configs.onnx_cfg.get("optimize", False)
        if backend == Backend.NCNN.value:
            """NCNN backend needs a precise blob counts, while using onnx optimizer
            will merge duplicate initilizers without reference count."""
            optimize = False
        return optimize

    def _build_ir_configs(self) -> dict:
        """Build the IR configs.

        Returns:
            IR configs.
        """
        onnx_cfg = self.setup_configs.onnx_cfg
        input_names = onnx_cfg["input_names"]
        output_names = onnx_cfg["output_names"]
        axis_names = input_names + output_names
        dynamic_axes = get_dynamic_axes(self.setup_configs.deploy_cfg, axis_names)
        verbose = not onnx_cfg.get("strip_doc_string", True) or onnx_cfg.get("verbose", False)
        keep_initializers_as_inputs = onnx_cfg.get("keep_initializers_as_inputs", True)
        opset_version = onnx_cfg.get("opset_version", 11)

        ir_configs = dict(
            type="onnx",
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
        )
        return ir_configs

    def _build_intermediate_representations(self) -> IR:
        """Build the intermediate representations (IR).

        Returns:
            Intermediate representation (IR).
        """
        return IR.get(get_ir_config(self.setup_configs.deploy_cfg)["type"])

    def _build_context_info(self, ir: IR, ir_configs: dict, backend: str, optimize: bool) -> dict:
        """Build the context info.

        Returns:
            Context info.
        """
        if optimize:
            onnx_custom_passes = optimize_onnx
        else:
            onnx_custom_passes = None

        return dict(
            deploy_cfg=self.setup_configs.deploy_cfg,
            ir=ir,
            backend=backend,
            opset=ir_configs["opset_version"],
            cfg=self.setup_configs.deploy_cfg,
            onnx_custom_passes=onnx_custom_passes,
        )
