import os
from math import ceil
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple
from pathlib import Path
from matplotlib import pyplot as plt

import mmengine
from mmengine.registry import init_default_scope
from mmengine.runner import autocast
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.visualization.vis_utils import proj_lidar_bbox3d_to_img
from mmdet3d.structures.det3d_data_sample import Det3DDataSample
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path as Plt_Path
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from projects.CenterPoint.runners.base_runner import BaseRunner
from tools.detection3d.visualize_bev import OBJECT_PALETTE


@dataclass(frozen=True)
class DecodedOutputs:
    """ 
    Dataclass to save decoded outputs from CenterPoint and their metadata.
    :param lidar_bboxes: Decoded bboxes in lidar.
    :param lidar_pointclouds: Raw lidar pointclouds.
    :param scores: Scores for each bbox.
    :param labels: Labels for each bbox.
    :param class_name: Available class names for the outputs.
    :param img_paths: Available image paths for the outputs.
    :param lidar2cams [<4, 4>]: Intrinsic and extrinsic from lidar to cameras.
    :param cam2imgs [<3, 3>]: Intrinsic and extrinsic from cameras to images. 
    """
    lidar_bboxes: LiDARInstance3DBoxes
    lidar_pointclouds: npt.NDArray[np.float64]
    scores: torch.tensor
    labels: torch.tensor
    class_names: List[str]
    img_paths: List[str]
    lidar2cams: List[npt.NDArray[np.float64]]
    cam2imgs: List[npt.NDArray[np.float64]]

    def project_lidar_bboxex_to_img(
            self, lidar2img: np.ndarray) -> npt.NDArray[np.float64]:
        """
        Project Bboxes in lidar view to an image view.
        :param lidar2img <4, 4>: intrinsic and extrinsic from lidar to an image.
        :return <N, 8, 2> (Number of bboxes, 8 corners, x and y coordinates). 
        Projected bboxes in an image. 
        """
        return proj_lidar_bbox3d_to_img(
            self.lidar_bboxes, input_meta={'lidar2img': lidar2img})

    def compute_lidar_to_imgs(self) -> List[npt.NDArray[np.float64]]:
        """
        Compute Extrinsic and intrinsic from lidar to images.
        :return List of <4, 4> for extrinsic and intrinsic from a lidar to every images.
        """
        lidar2imgs = []
        for lidar2cam, cam2img in zip(self.lidar2cams, self.cam2imgs):
            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam2img).astype(np.float32)
            lidar2cam_array = np.asarray(lidar2cam, dtype=np.float32)
            lidar2imgs.append(cam2img_array @ lidar2cam_array)
        return lidar2imgs

    def visualize_bboxes_to_lidar(self,
                                  fig: Figure,
                                  grid_spec: GridSpec,
                                  xlim: Tuple[int, int],
                                  ylim: Tuple[int, int],
                                  radius: float = 0.1,
                                  thickness: int = 1,
                                  line_styles: str = '-',
                                  draw_front=True) -> None:
        """
        Visualize bboxes in LiDAR.
        :param ax: Axes to visualize bboxes in lidar.
        :param fpath: Path to save the visualization.
        :param xlim: Range in x-axis (-min, max).
        :param ylim: Range in y-axis (-min, max).
        :param draw_front: Set True to draw a line to indicate direction of bboxes.
        """
        ax = fig.add_subplot(grid_spec[-1, :], facecolor="black")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        # ax.set_aspect("auto", adjustable="datalim")
        ax.axis('auto')

        circle_patches = [
            Circle((pcd[0], pcd[1]), radius=radius)
            for pcd in self.lidar_pointclouds
        ]

        c = PatchCollection(circle_patches, facecolors="white")
        ax.add_collection(c)

        if self.lidar_bboxes is not None and len(self.lidar_bboxes) > 0:
            lines_verts_idx = [0, 3, 7, 4, 0]
            coords = self.lidar_bboxes.corners[:, lines_verts_idx, :2]
            codes = [Plt_Path.LINETO] * coords.shape[1]
            codes[0] = Plt_Path.MOVETO
            pathpatches = []
            edge_color_norms = []
            center_bottom_patches = []
            for index in range(coords.shape[0]):
                verts = coords[index]
                pth = Plt_Path(verts, codes)
                pathpatches.append(PathPatch(pth))

                label = self.labels[index]
                name = self.class_names[label]
                edge_color_norms.append(np.array(OBJECT_PALETTE[name]) / 255)
                if draw_front:
                    # Draw line indicating the front
                    center_bottom_forward = torch.mean(
                        coords[index, 2:4, :2], axis=0, keepdim=True)
                    center_bottom = torch.mean(
                        coords[index, [0, 1, 2, 3], :2], axis=0, keepdim=True)
                    center_bottom_pth = Plt_Path(
                        torch.concat([center_bottom, center_bottom_forward],
                                     axis=0),
                        codes=codes[0:2])
                    center_bottom_patches.append(PathPatch(center_bottom_pth))

            p = PatchCollection(
                pathpatches,
                facecolors='none',
                edgecolors=edge_color_norms,
                linewidths=thickness,
                linestyles=line_styles)
            ax.add_collection(p)

            if len(center_bottom_patches):
                line_collections = PatchCollection(
                    center_bottom_patches,
                    facecolors='none',
                    edgecolors=edge_color_norms,
                    linewidths=thickness,
                    linestyles=line_styles)
                ax.add_collection(line_collections)

        ax.set_title("LiDAR")

    def visualize_bboxes_to_image(self,
                                  ax: plt.axes,
                                  img_path: str,
                                  lidar2img: npt.NDArray[np.float64],
                                  img_title_index: int = -1,
                                  alpha: float = 0.8,
                                  line_widths: int = 2,
                                  line_styles: str = '-') -> None:
        """
        Visualize bboxes from LiDAR to an image. 
        This function is modified from mmdet3d.visualization.local_visualizer.draw_proj_bboxes_3d.
        :param ax: Matplotlib axis.
        :param img_path: Full image path.
        :param lidar2img <4, 4>: intrinsic and extrinsic from lidar to an image.
        :param img_title_index: Index to indicate an image name from img_path.
        :param alpha: Transparency of polygons.
        :param line_width: Thickness of bboxes' edges.
        :param line_styles: Style of bboxes's edges.
        :return np.float64 <N, 8, 2> (Number of bboxes, 8 corners, x and y coordinates). 
        """
        # Draw the image to axis
        img = plt.imread(img_path)
        ax.imshow(img)

        # Metadata about image
        h, w, _ = img.shape
        img_size = (w, h)
        ax_title = img_path.split("/")[img_title_index]

        corners_2d = self.project_lidar_bboxex_to_img(lidar2img=lidar2img)
        edge_color_norms = []
        if img_size is not None:
            # Filter out the bbox where half of stuff is outside the image.
            # This is for the visualization of multi-view image.
            valid_point_idx = (corners_2d[..., 0] >= 0) & \
                        (corners_2d[..., 0] <= img_size[0]) & \
                        (corners_2d[..., 1] >= 0) & (corners_2d[..., 1] <= img_size[1])  # noqa: E501
            valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 4
            valid_bbox_labels = self.labels[valid_bbox_idx]
            corners_2d = corners_2d[valid_bbox_idx]
            for label in valid_bbox_labels:
                name = self.class_names[label]
                edge_color_norms.append(np.array(OBJECT_PALETTE[name]) / 255)

        lines_verts_idx = [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 5, 1, 2, 6]
        lines_verts = corners_2d[:, lines_verts_idx, :]
        front_polys = corners_2d[:, 4:, :]
        codes = [Plt_Path.LINETO] * lines_verts.shape[1]
        codes[0] = Plt_Path.MOVETO
        pathpatches = []
        for i in range(len(corners_2d)):
            verts = lines_verts[i]
            pth = Plt_Path(verts, codes)
            pathpatches.append(PathPatch(pth))

        p = PatchCollection(
            pathpatches,
            facecolors='none',
            edgecolors=edge_color_norms,
            linewidths=line_widths,
            linestyles=line_styles)
        ax.add_collection(p)

        # Draw a mask on the front of project bboxes
        front_polys = [front_poly for front_poly in front_polys]
        face_colors = edge_color_norms
        polygon_collection = PolyCollection(
            front_polys,
            alpha=alpha,
            facecolor=face_colors,
            linestyles=line_styles,
            edgecolors=edge_color_norms,
            linewidths=line_widths)
        ax.add_collection(polygon_collection)

        # Setting the axis
        ax.set_title(ax_title)
        ax.set_axis_off()

    def visualize_bboxes_to_images(self,
                                   fig: Figure,
                                   grid_spec: GridSpec,
                                   spec_cols: int = 3,
                                   alpha: float = 0.8) -> None:
        """
        Visualize bboxes from lidar to ever image.
        :param fpath: Path to save the visualization.
        :param ax_cols: Number of cols in the visualization.
        :param fig_size: Figure size.
        :param alpha: Transparency of polygons.
        """
        lidar2imgs = self.compute_lidar_to_imgs()
        assert len(self.img_paths) == len(lidar2imgs)
        selected_row = 0
        for index, (img_path,
                    lidar2img) in enumerate(zip(self.img_paths, lidar2imgs)):
            selected_col = index % spec_cols
            selected_row = index // spec_cols
            ax = fig.add_subplot(grid_spec[selected_row, selected_col])
            if img_path is not None:
                self.visualize_bboxes_to_image(
                    ax=ax,
                    img_path=img_path,
                    lidar2img=np.asarray(lidar2img),
                    alpha=alpha,
                    img_title_index=-2)

    def visualize_bboxes(
        self,
        fpath: str,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        spec_cols: int = 3,
        alpha: float = 0.8,
        fig_size: Tuple[int, int] = (15, 15)) -> None:
        """
        Visualize bboxes to both imgs and lidar in BEV.
        :param fpath: Path to save the visualization.
        :param xlim: x-axis range for lidar.
        :param ylim: y-axis range for lidar.
        :param spec_cols: Number of columns for gridspec in the visualization.
        :param alpha: Transparency of polygons in images.
        :param fig_size: Figure size.
        """
        # Init axes
        # Get the number of rows
        image_rows = ceil(len(self.img_paths) / spec_cols)
        fig = plt.figure(figsize=fig_size)

        # Images + Lidar
        grid_spec = fig.add_gridspec(image_rows + 1, spec_cols)

        # Add subplots for images
        if len(self.img_paths):
            self.visualize_bboxes_to_images(
                fig=fig, grid_spec=grid_spec, spec_cols=spec_cols, alpha=alpha)

        # Add subplot for lidar
        self.visualize_bboxes_to_lidar(
            fig=fig,
            grid_spec=grid_spec,
            xlim=xlim,
            ylim=ylim,
            draw_front=True)

        plt.tight_layout()
        plt.savefig(
            fname=fpath,
            format="png",
            # dpi=15,
            bbox_inches="tight",
        )
        plt.close()


class InferenceRunner(BaseRunner):
    """ Runner to run inference over a test dataloader, and visualize inferences. """

    def __init__(self,
                 model_cfg_path: str,
                 checkpoint_path: str,
                 work_dir: Path,
                 data_root: str,
                 ann_file_path: str,
                 frame_range: Optional[Tuple[int, int]] = None,
                 bboxes_score_threshold: float = 0.10,
                 device: str = 'gpu',
                 default_scope: str = 'mmengine',
                 experiment_name: str = "",
                 log_level: Union[int, str] = 'INFO',
                 log_file: Optional[str] = None) -> None:
        """
        :param model_cfg_path: MMDet3D model config path.
        :param checkpoint_path: Checkpoint path to load weights.
        :param work_dir: Working directory to save outputs.
        :param data_root: Path where to save date, it will overwrite configs in model_cfg_path 
            if it's set.
        :param ann_file_path: Path where to save annotation file, it will overwrite configs in
            model_cfg_path if it's set.
        :param device: Working devices, only 'gpu' or 'cpu' supported.
        :param default_scope: Default scope in mmdet3D.
        :param experiment_name: Experiment name.
        :param log_level: Logging and display log messages above this level.
        :param log_file: Logger file.
        """
        super(InferenceRunner, self).__init__(
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

        self._data_root = data_root
        self._ann_file_path = ann_file_path
        self._bboxes_score_threshold = bboxes_score_threshold
        self._xlim = [-120, 120]
        self._ylim = [-120, 120]
        self._frame_range = frame_range

    def run(self) -> None:
        """
        Start running the Runner.
        """
        # Update config
        self._cfg.model.pts_bbox_head.bbox_coder.score_threshold = self._bboxes_score_threshold

        # Build a model
        model = self.build_model()

        # Load the checkpoint
        self.load_verify_checkpoint(model=model)

        # Build a test dataloader
        test_dataloader = self.build_test_dataloader(
            data_root=self._data_root, ann_file_path=self._ann_file_path)

        self.inference(model=model, dataloader=test_dataloader)

    def _decode_outputs(self, output: Det3DDataSample,
                        lidar_pointclouds: npt.NDArray[np.float64],
                        data_sample: Det3DDataSample) -> DecodedOutputs:
        """
        Decode outputs from a model and save their metadata into DecodedOutputs.
        :param outputs: Output from a model in Det3DDataSample. 
        :param lidar_pointcloud: Lidar pointclouds.
        :param data_sample: Groundtruth and metadata from input.
        :return DecodedOutputs. 
        """
        bboxes = output.pred_instances_3d["bboxes_3d"].tensor.detach().cpu()
        scores = output.pred_instances_3d["scores_3d"].detach().cpu()
        labels = output.pred_instances_3d["labels_3d"].detach().cpu()
        lidar_bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        img_paths = data_sample.img_path if all(data_sample.img_path) else []
        full_img_paths = [
            os.path.join(self._cfg.test_dataloader.dataset.data_root, img_path)
            for img_path in img_paths
        ]
        lidar2cam = data_sample.lidar2cam if hasattr(data_sample,
                                                     "lidar2cam") else []
        cam2img = data_sample.cam2img if hasattr(data_sample,
                                                 "cam2img") else []
        return DecodedOutputs(
            lidar_bboxes=lidar_bboxes,
            lidar_pointclouds=lidar_pointclouds,
            labels=labels,
            scores=scores,
            class_names=self._cfg.class_names,
            img_paths=full_img_paths,
            lidar2cams=lidar2cam,
            cam2imgs=cam2img)

    def inference(self, model: nn.Module, dataloader: DataLoader) -> None:
        """
        Inference outputs from every data samples with the model and dataloader.
        :param model: Torch NN module.
        :param dataloader: Torch Iterable Dataloader.
        """
        vis_dir = Path(self._work_dir) / "vis"
        with torch.no_grad():
            model.eval()
            for index, data in enumerate(
                    tqdm(
                        dataloader,
                        desc="Running inference and visualizing!")):

                if self._frame_range is not None:
                    if index < self._frame_range[0]:
                        continue
                    if index > self._frame_range[1]:
                        self._logger.info(
                            f"Done visualizing the frames: {self._frame_range}"
                        )
                        break

                lidar_path = data["data_samples"][0].lidar_path.split("/")
                scene_token = lidar_path[3]
                file_name = "_".join(lidar_path[3:8]) + ".png"

                with autocast(enabled=True):
                    outputs = model.test_step(data)

                decoded_outputs = self._decode_outputs(
                    output=outputs[0],
                    lidar_pointclouds=data["inputs"]["points"][0],
                    data_sample=data["data_samples"][0])

                # Scene frame
                scene_path = vis_dir / scene_token
                mmengine.mkdir_or_exist(scene_path)
                lidar_fpath = scene_path / file_name

                # Visualize bboxes in both cameras and bev
                decoded_outputs.visualize_bboxes(
                    fpath=lidar_fpath,
                    xlim=self._xlim,
                    ylim=self._ylim,
                    alpha=0.5)
            self._logger.info(f"Saved visualization to {vis_dir}!")
