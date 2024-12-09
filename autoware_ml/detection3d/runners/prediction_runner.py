import concurrent.futures
import os
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.det3d_data_sample import Det3DDataSample
from mmengine.registry import init_default_scope
from mmengine.runner import autocast
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoware_ml.detection3d.runners.base_runner import BaseRunner
from autoware_ml.detection3d.visualization.decoded_bboxes import BatchDecodedBboxes, DecodedBboxes


def visualization_wrapper(
    batch_decoded_bboxes: BatchDecodedBboxes,
    vis_dir: Path,
    xlim: Tuple[int, int],
    ylim: Tuple[int, int],
) -> None:
    """
    A wrapper function to call a single BatchDecodedBboxes for executing visualization
    in concurrent.futures.
    """
    batch_decoded_bboxes.visualize(vis_dir=vis_dir, xlim=xlim, ylim=ylim)


class PredictionRunner(BaseRunner):
    """Runner to generate predictions over a test dataloader, and visualize predictions."""

    def __init__(
        self,
        model_cfg_path: str,
        checkpoint_path: str,
        work_dir: Path,
        data_root: str,
        ann_file_path: str,
        batch_size: int = 0,
        max_workers: int = 8,
        frame_range: Optional[Tuple[int, int]] = None,
        bboxes_score_threshold: float = 0.10,
        device: str = "gpu",
        default_scope: str = "mmengine",
        experiment_name: str = "",
        log_level: Union[int, str] = "INFO",
        log_file: Optional[str] = None,
    ) -> None:
        """
        :param model_cfg_path: MMDet3D model config path.
        :param checkpoint_path: Checkpoint path to load weights.
        :param work_dir: Working directory to save outputs.
        :param data_root: Path where to save date, it will overwrite configs in model_cfg_path
            if it's set.
        :param ann_file_path: Path where to save annotation file, it will overwrite configs in
            model_cfg_path if it's set.
        :param max_workers: Get maximum number of cpu workers when running in multiprocessing.
        :param device: Working devices, only 'gpu' or 'cpu' supported.
        :param default_scope: Default scope in mmdet3D.
        :param experiment_name: Experiment name.
        :param log_level: Logging and display log messages above this level.
        :param log_file: Logger file.
        """
        super(PredictionRunner, self).__init__(
            model_cfg_path=model_cfg_path,
            checkpoint_path=checkpoint_path,
            work_dir=work_dir,
            device=device,
            default_scope=default_scope,
            experiment_name=experiment_name,
            log_level=log_level,
            log_file=log_file,
        )

        # We need init deafault scope to mmdet3d to search registries in the mmdet3d scope
        init_default_scope("mmdet3d")

        self._data_root = data_root
        self._ann_file_path = ann_file_path
        self._bboxes_score_threshold = bboxes_score_threshold
        self._xlim = [-120, 120]
        self._ylim = [-120, 120]
        self._frame_range = frame_range
        self._max_workers = max_workers
        self._batch_size = batch_size

    def run(self) -> None:
        """Start running the Runner."""
        # Update config
        if hasattr(self._cfg.model, "pts_bbox_head"):
            self._cfg.model.pts_bbox_head.bbox_coder.score_threshold = self._bboxes_score_threshold
        elif hasattr(self._cfg.model, "bbox_head"):
            self._cfg.model.bbox_head.bbox_coder.score_threshold = self._bboxes_score_threshold
        else:
            raise ValueError("The model is not supported!")

        # Build a model
        model = self.build_model()

        # Load the checkpoint
        self.load_verify_checkpoint(model=model)

        # Build a test dataloader
        test_dataloader = self.build_test_dataloader(data_root=self._data_root, ann_file_path=self._ann_file_path)

        batch_decoded_bboxes = self.predict(model=model, dataloader=test_dataloader)

        if len(batch_decoded_bboxes):
            self._visualize(batch_decoded_bboxes=batch_decoded_bboxes)

    def _visualize(
        self,
        batch_decoded_bboxes: List[BatchDecodedBboxes],
    ) -> None:
        """Visualize the decoded predictions."""
        self._logger.info(f"Visualizing with cpu workers: {self._max_workers}")

        vis_dir = Path(self._work_dir) / "vis"
        partial_visualization_wrapper = partial(
            visualization_wrapper,
            vis_dir=vis_dir,
            xlim=self._xlim,
            ylim=self._ylim,
        )
        # Use ProcessPoolExecutor to execute tasks concurrently
        with concurrent.futures.ProcessPoolExecutor(max_workers=self._max_workers) as executor:

            # Use tqdm to show progress in the terminal
            with tqdm(total=len(batch_decoded_bboxes), desc="Visualizing") as pbar:
                # Submit tasks to the executor and store futures
                futures = [
                    executor.submit(partial_visualization_wrapper, decoded_bboxes)
                    for decoded_bboxes in batch_decoded_bboxes
                ]

                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    _ = future.result()
                    pbar.update(1)  # Update progress bar for each completed task

        self._logger.info(f"saved visualization to {vis_dir}!")

    def _decode_outputs(
        self,
        output: Det3DDataSample,
        lidar_pointclouds: npt.NDArray[np.float64],
        data_sample: Det3DDataSample,
    ) -> BatchDecodedBboxes:
        """
        Decode outputs from a model and save their metadata into DecodedBboxes.
        :param outputs: Output from a model in Det3DDataSample.
        :param lidar_pointcloud: Lidar pointclouds.
        :param data_sample: Groundtruth and metadata from input.
        :return Decoded predictions.
        """
        bboxes = output.pred_instances_3d["bboxes_3d"].tensor.detach().cpu()
        scores = output.pred_instances_3d["scores_3d"].detach().cpu()
        labels = output.pred_instances_3d["labels_3d"].detach().cpu()
        lidar_bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        img_paths = data_sample.img_path if all(data_sample.img_path) else []
        full_img_paths = [
            os.path.join(self._cfg.test_dataloader.dataset.data_root, img_path) for img_path in img_paths
        ]
        lidar2cam = data_sample.lidar2cam if hasattr(data_sample, "lidar2cam") else []
        cam2img = data_sample.cam2img if hasattr(data_sample, "cam2img") else []
        return DecodedBboxes(
            lidar_bboxes=lidar_bboxes,
            lidar_pointclouds=lidar_pointclouds,
            labels=labels,
            scores=scores,
            class_names=self._cfg.class_names,
            img_paths=full_img_paths,
            lidar2cams=lidar2cam,
            cam2imgs=cam2img,
        )

    def predict(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> List[BatchDecodedBboxes]:
        """
        Generate predictions from every data sample with the model and dataloader.
        :param model: Torch NN module.
        :param dataloader: Torch Iterable Dataloader.
        """
        data_root = self._cfg.test_dataloader.dataset.data_root
        batch_decoded_bboxes = []
        with torch.no_grad():
            model.eval()
            for index, data in enumerate(tqdm(dataloader, desc="Generating predictions...")):

                if self._frame_range is not None:
                    if index < self._frame_range[0]:
                        continue
                    if index > self._frame_range[1]:
                        self._logger.info(f"Done visualizing the frames: {self._frame_range}")
                        break

                lidar_path = Path(data["data_samples"][0].lidar_path)
                lidar_path = lidar_path.relative_to(data_root)
                lidar_path = str(lidar_path).split("/")
                # Always assume the first two are database_name/dataset_id
                scene_name = "/".join(lidar_path[:2])
                # Always assume the last two are LIDAR_CONCAT/lidar_frame_index
                lidar_filename = "_".join(lidar_path[-2:]) + ".png"

                with autocast(enabled=True):
                    outputs = model.test_step(data)

                decoded_bboxes = self._decode_outputs(
                    output=outputs[0],
                    lidar_pointclouds=data["inputs"]["points"][0],
                    data_sample=data["data_samples"][0],
                )

                batch_decoded_bboxes.append(
                    BatchDecodedBboxes(
                        scene_name=scene_name,
                        lidar_filename=lidar_filename,
                        decoded_bboxes=decoded_bboxes,
                    )
                )

                if self._batch_size > 0 and len(batch_decoded_bboxes) % self._batch_size == 0:
                    self._visualize(batch_decoded_bboxes=batch_decoded_bboxes)
                    batch_decoded_bboxes = []

        return batch_decoded_bboxes
