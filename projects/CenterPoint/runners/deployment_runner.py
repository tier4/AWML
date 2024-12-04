from typing import Optional, Union
from pathlib import Path

from autoware_ml.detection3d.runners.base_runner import BaseRunner
from mmengine.registry import MODELS
from mmengine.registry import init_default_scope

from torch import nn


class DeploymentRunner(BaseRunner):
    """ Runner to run deploment of mmdet3D model to generate ONNX with random inputs. """

    def __init__(self,
                 model_cfg_path: str,
                 checkpoint_path: str,
                 work_dir: Path,
                 rot_y_axis_reference: bool = False,
                 device: str = 'gpu',
                 replace_onnx_models: bool = False,
                 default_scope: str = 'mmengine',
                 experiment_name: str = "",
                 log_level: Union[int, str] = 'INFO',
                 log_file: Optional[str] = None) -> None:
        """
        :param model_cfg_path: MMDet3D model config path.
        :param checkpoint_path: Checkpoint path to load weights.
        :param work_dir: Working directory to save outputs.
        :param rot_y_axis_reference: Set True to convert rotation 
            from x-axis counterclockwiese to y-axis clockwise.
        :param device: Working devices, only 'gpu' or 'cpu' supported.
        :param replace_onnx_models: Set True to replace model with ONNX, 
            for example, CenterHead -> CenterHeadONNX.
        :param default_scope: Default scope in mmdet3D.
        :param experiment_name: Experiment name.
        :param log_level: Logging and display log messages above this level.
        :param log_file: Logger file.
        """
        super(DeploymentRunner, self).__init__(
            model_cfg_path=model_cfg_path,
            checkpoint_path=checkpoint_path,
            work_dir=work_dir,
            device=device,
            default_scope=default_scope,
            experiment_name=experiment_name,
            log_level=log_level,
            log_file=log_file)

        # We need init deafault scope to mmdet3d to search registries in the mmdet3d scope
        init_default_scope("mmdet3d")

        self._rot_y_axis_reference = rot_y_axis_reference
        self._replace_onnx_models = replace_onnx_models

    def build_model(self) -> nn.Module:
        """
        Build a model. Replace the model by ONNX model if replace_onnx_model is set.
        :return torch.nn.Module. A torch module.
        """
        self._logger.info("===== Building CenterPoint model ====")
        model_cfg = self._cfg.get('model')
        # Update Model type to ONNX
        if self._replace_onnx_models:
            self._logger.info("Replacing ONNX models!")
            model_cfg.type = "CenterPointONNX"
            model_cfg.point_channels = model_cfg.pts_voxel_encoder.in_channels
            model_cfg.device = self._device
            model_cfg.pts_voxel_encoder.type = "PillarFeatureNetONNX" if model_cfg.pts_voxel_encoder.type == "PillarFeatureNet" else "BackwardPillarFeatureNetONNX"
            model_cfg.pts_bbox_head.type = "CenterHeadONNX"
            model_cfg.pts_bbox_head.separate_head.type = "SeparateHeadONNX"
            model_cfg.pts_bbox_head.rot_y_axis_reference = self._rot_y_axis_reference

        model = MODELS.build(model_cfg)
        model.to(self._torch_device)

        self._logger.info(model)
        self._logger.info("===== Built CenterPoint model ====")
        return model

    def run(self) -> None:
        """ Start running the Runner. """
        # Building a model
        model = self.build_model()

        # Loading checkpoint to the model
        self.load_verify_checkpoint(model=model)

        assert hasattr(
            model,
            "save_onnx"), "The model must have the function: save_onnx()!"

        # Run and save onnx model!
        model.save_onnx(save_dir=self._work_dir)
