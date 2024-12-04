import os
import time
from pathlib import Path
from typing import Optional, Union, Tuple

import mmengine
from mmengine import MMLogger
from mmengine.config import Config
from mmengine.registry import DefaultScope, MODELS
from mmengine.runner import Runner

import torch
from torch import nn
from torch.utils.data import DataLoader


class BaseRunner:
    """ Base runner to run a mmdect3D model. """

    def __init__(self,
                 model_cfg_path: str,
                 checkpoint_path: str,
                 work_dir: Path,
                 device: str = 'gpu',
                 default_scope: str = 'mmengine',
                 experiment_name: str = "",
                 log_level: Union[int, str] = 'INFO',
                 log_file: Optional[str] = None) -> None:
        """
        :param model_cfg_path: MMDet3D model config path.
        :param checkpoint_path: Checkpoint path to load weights.
        :param work_dir: Working directory to save outputs.
        :param device: Working devices, only 'gpu' or 'cpu' supported.
        :param default_scope: Default scope in mmdet3D.
        :param experiment_name: Experiment name.
        :param log_level: Logging and display log messages above this level.
        :param log_file: Logger file.
        """
        self._experiment_name = experiment_name
        self._log_level = log_level
        self._log_file = log_file
        self._model_cfg_path = Path(model_cfg_path)
        self._checkpoint_path = Path(checkpoint_path)
        self._timestamp = ""
        self._device = device

        assert self._device in ["gpu",
                                "cpu"], "Only gpu or cpu supported in device!"
        self._torch_device = torch.device(
            'cuda:0') if self._device == 'gpu' else torch.device('cpu')

        # Setup env
        self._setup_env()

        if default_scope is not None:
            default_scope = DefaultScope.get_instance(  # type: ignore
                self._experiment_name,
                scope_name=default_scope)

        self.default_scope = default_scope

        # Load model configs
        self._cfg = Config.fromfile(model_cfg_path)

        self._work_dir = work_dir / self._timestamp

        # create work_dir if not
        mmengine.mkdir_or_exist(os.path.abspath(self._work_dir))

        # Building and configuration
        self._logger = self._build_logger()

    def run(self) -> None:
        """
        Start running the Runner.
        """
        raise NotImplementedError

    def _get_weight_statistics(self, model: nn.Module) -> Tuple[float, float]:
        """
        Compute statistics of the weights from the model. 
        :return: (mean, variance) of the weights.
        """
        sum_weights = torch.Tensor(
            [torch.sum(param) for param in model.state_dict().values()])
        return (torch.mean(sum_weights), torch.var(sum_weights))

    def build_model(self) -> nn.Module:
        """ Build a model. """
        self._logger.info("===== Building the model ====")
        model_cfg = self._cfg.get('model')

        model = MODELS.build(model_cfg)
        model.to(self._torch_device)

        self._logger.info(model)
        self._logger.info("===== Built the model ====")
        return model

    def build_test_dataloader(self,
                              data_root: str = '',
                              ann_file_path: str = '') -> DataLoader:
        """
        Build a test dataloader.
        :param data_root: Overwritting data root in the dataloader config.
        :param ann_file_path: Overwritting ann_file_path in the dataloader config.
        :return: Pytorch Dataloader for a test set.
        """
        self._logger.info("===== Building test dataloader ====")
        self._cfg.val_dataloader.batch_size = 1
        self._cfg.test_dataloader.batch_size = 1

        if data_root:
            self._logger.info(f"Replace data_root to {data_root}")
            self._cfg.test_dataloader.dataset.data_root = data_root

        if ann_file_path:
            self._logger.info(f"Replace ann_file to {ann_file_path}")
            self._cfg.test_dataloader.dataset.ann_file = ann_file_path

        # build dataset
        test_dataloader = Runner.build_dataloader(self._cfg.test_dataloader)

        self._logger.info("===== Built test dataloader ====")
        return test_dataloader

    def load_verify_checkpoint(self, model: nn.Module):
        """
        Load checkpoint and make verification.
        :param model: Pytorch NN module.
        """
        self._logger.info("Loading Checkpoint")
        mean_before, variance_before = self._get_weight_statistics(model=model)

        # Load checkpoint
        checkpoint_state_dict = torch.load(
            self._checkpoint_path, map_location=self._torch_device)

        # Load model weights
        model.load_state_dict(checkpoint_state_dict['state_dict'])

        mean_after, variance_after = self._get_weight_statistics(model=model)

        # Verify if loading works
        self._logger.info(
            f"Mean of weights before loading: {mean_before} and after loading: {mean_after}"
        )
        self._logger.info(
            f"variance of weights before loading: {variance_before} and after loading: {variance_after}"
        )

    def _setup_env(self) -> None:
        """ Setup environment. """
        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        self._timestamp = time.strftime('%Y%m%d_%H%M%S',
                                        time.localtime(timestamp.item()))

    def _build_logger(self, **kwargs) -> MMLogger:
        """
        Build a global asscessable MMLogger when we are not running a runner.
        :param log_level (int or str): The log level of MMLogger handlers.
            Defaults to 'INFO'.
        :param log_file (str, optional): Path of filename to save log.
            Defaults to None.
        **kwargs: Remaining parameters passed to ``MMLogger``.
        :return MMLogger: A MMLogger object build from ``logger``.
        """
        if self._log_file is None:
            self._log_file = os.path.join(self._work_dir,
                                          f'{self._timestamp}.log')

        log_cfg = dict(
            log_level=self._log_level, log_file=self._log_file, **kwargs)
        log_cfg.setdefault('name', self._experiment_name)
        # `torch.compile` in PyTorch 2.0 could close all user defined handlers
        # unexpectedly. Using file mode 'a' can help prevent abnormal
        # termination of the FileHandler and ensure that the log file could
        # be continuously updated during the lifespan of the runner.
        log_cfg.setdefault('file_mode', 'a')

        return MMLogger.get_instance(**log_cfg)  # type: ignore
