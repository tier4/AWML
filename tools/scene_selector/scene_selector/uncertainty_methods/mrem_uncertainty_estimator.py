# Standard libraries
import copy
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib

# Third-party libraries
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Custom base transform
from mmcv.transforms.base import BaseTransform
from mmdet3d.apis.inferencers import LidarDet3DInferencer
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Det3DDataSample, bbox_overlaps_3d
from mmdet3d.structures import ops as box_np_ops
from mmdet3d.structures.bbox_3d import get_box_type
from mmdet3d.utils import ConfigType

# MMDetection3D libraries
from mmengine import dump
from mmengine.dataset import Compose
from rich.progress import track

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]


@TRANSFORMS.register_module()
class LidarWithCamDet3DInferencerLoader(BaseTransform):
    """Load point cloud in the Inferencer's pipeline.

    Added keys:
      - points
      - timestamp
      - axis_align_matrix
      - box_type_3d
      - box_mode_3d
    """

    def __init__(self, coord_type="LIDAR", **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(dict(type="LoadPointsFromFile", coord_type=coord_type, **kwargs))
        self.from_ndarray = TRANSFORMS.build(dict(type="LoadPointsFromDict", coord_type=coord_type, **kwargs))
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)

    def transform(self, single_input: dict) -> dict:
        """Transform function to add image meta information.
        Args:
            single_input (dict): Single input.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert "points" in single_input, "key 'points' must be in input dict"
        if isinstance(single_input["points"], str):
            inputs = dict(
                lidar_points=dict(lidar_path=single_input["points"]),
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d,
            )
        elif isinstance(single_input["points"], np.ndarray):
            inputs = dict(
                points=single_input["points"],
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d,
            )
        else:
            raise ValueError("Unsupported input points type: " f"{type(single_input['points'])}")
        if "points" in inputs:
            single_input.update(self.from_ndarray(inputs))
        single_input.update(self.from_file(inputs))
        return single_input


class REMCustomInferencer(LidarDet3DInferencer):

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg, "LoadPointsFromFile")
        if load_point_idx == -1:
            raise ValueError("LoadPointsFromFile is not found in the test pipeline")

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg["coord_type"], load_cfg["load_dim"]
        self.use_dim = (
            list(range(load_cfg["use_dim"])) if isinstance(load_cfg["use_dim"], int) else load_cfg["use_dim"]
        )

        pipeline_cfg[load_point_idx]["type"] = "LidarWithCamDet3DInferencerLoader"
        return Compose(pipeline_cfg)

    def pred2dict(
        self,
        data_sample: Det3DDataSample,
        pred_out_dir: str = "",
    ) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        result = {}
        if "pred_instances_3d" in data_sample:
            pred_instances_3d = data_sample.pred_instances_3d.numpy()
            result = {
                "labels_3d": pred_instances_3d.labels_3d.tolist(),
                "scores_3d": pred_instances_3d.scores_3d.tolist(),
                "bboxes_3d": pred_instances_3d.bboxes_3d.tensor.cpu().tolist(),
            }

        if "pred_pts_seg" in data_sample:
            pred_pts_seg = data_sample.pred_pts_seg.numpy()
            result["pts_semantic_mask"] = pred_pts_seg.pts_semantic_mask.tolist()

        if pred_out_dir != "":
            if "lidar_path" in data_sample:
                lidar_path = osp.basename(data_sample.lidar_path)
                lidar_path = osp.splitext(lidar_path)[0]
                out_json_path = osp.join(pred_out_dir, "preds", lidar_path + ".json")
            elif "img_path" in data_sample:
                img_path = osp.basename(data_sample.img_path)
                img_path = osp.splitext(img_path)[0]
                out_json_path = osp.join(pred_out_dir, "preds", img_path + ".json")
            else:
                out_json_path = osp.join(pred_out_dir, "preds", f"{str(self.num_visualized_imgs).zfill(8)}.json")
            dump(result, out_json_path)

        return result

    def __call__(
        self,
        inputs: InputsType,
        batch_size: int = 1,
        return_datasamples: bool = False,
        **kwargs,
    ) -> Optional[dict]:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Batch size. Defaults to 1.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.


        Returns:
            dict: Inference and visualization results.
        """

        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        cam_type = preprocess_kwargs.pop("cam_type", "CAM2")
        ori_inputs = self._inputs_to_list(inputs, cam_type=cam_type)
        inputs = self.preprocess(ori_inputs, batch_size=batch_size, **preprocess_kwargs)

        results_dict = {"predictions": [], "visualization": []}
        for data in track(inputs, description="Inference") if self.show_progress else inputs:
            preds = self.forward(data, **forward_kwargs)
            visualization = self.visualize(ori_inputs, preds, **visualize_kwargs)
            results = self.postprocess(preds, visualization, return_datasamples, **postprocess_kwargs)
            results_dict["predictions"].extend(results["predictions"])
            if results["visualization"] is not None:
                results_dict["visualization"].extend(results["visualization"])
        return results_dict


class ModelRareExampleMining:
    """
    Class for performing rare example mining using multiple models' 3D bounding box predictions on LiDAR data.
    Based on https://waymo.com/research/improving-the-intra-class-long-tail-in-3d-detection-via-rare-example-mining/

    Parameters:
    models (dict): Dictionary where keys are model names and values are their respective configurations.
    iou_threshold (float): Intersection-over-Union threshold for merging predictions from different models (default=0.4).
    min_score (float): Minimum confidence score for considering a bounding box in the final predictions (default=0.05).
    p_thresh (int): Minimum number of LiDAR points in a bounding box to be considered as a valid detection (default=20).
    d_thresh (int): Maximum distance from the sensor to consider a bounding box (default=75).

    Methods:
    __call__(sensor_info, results_path=""): Runs the rare example mining pipeline and optionally saves BEV visualizations.
    compute_3d_iou_batch(bboxes1, bboxes2): Computes the IoU between one set of bounding boxes and another.
    get_point_count_3d(boxes, point_cloud): Counts the number of LiDAR points within each 3D bounding box.
    get_distance_2d(boxes): Computes the distances from the sensor to the centers of bounding boxes.
    calculate_scores(model_predictions, point_cloud): Calculates rare example scores for each predicted object.
    plot_bev(pcd, bboxes, scores, model_name, output_path): Plots and saves a BEV visualization of the point cloud and bounding boxes.
    """

    def __init__(
        self,
        models,
        iou_threshold=0.4,
        min_score=0.05,
        p_thresh=20,
        d_thresh=75,
        batch_size=8,
    ):
        self.inferences = {k: REMCustomInferencer(**v) for k, v in models.items()}
        self.iou_threshold = iou_threshold
        self.p_thresh = p_thresh
        self.d_thresh = d_thresh
        self.min_score = min_score
        self.batch_size = batch_size

    def __call__(self, sensor_info, results_path: str = ""):
        """
        Runs the rare example mining pipeline, processing sensor information with multiple models,
        and optionally saving BEV visualizations for each sensor input.

        Parameters:
        sensor_info (list): List of sensor information dictionaries, each containing LiDAR point cloud file paths.
        results_path (str): Directory where BEV visualizations should be saved. If empty, no visualizations are saved.

        Returns:
        unique_bboxes (list): List of unique bounding boxes across models.
        data (list): Confidence score matrix for the bounding boxes.
        variances (list): Variance of confidence scores for each bounding box.
        h_i (list): Hard example filter flags (1 if box passes the point count threshold, 0 otherwise).
        r_i (list): Rare example scores (product of hard example flags and variances).
        """
        preds = []
        model_names = []

        for mname, model in self.inferences.items():
            preds.append(model(copy.deepcopy(sensor_info), self.batch_size)["predictions"])
            model_names.append(mname)

        unique_bboxes, data, variances, h_i, r_i = [], [], [], [], []

        for i, all_model_preds in enumerate(zip(*preds)):
            pcd = np.fromfile(sensor_info[i]["points"], dtype=np.float32).reshape((-1, 5))
            unique_bboxes_inst, data_inst, variances_inst, h_i_inst, r_i_inst = self.calculate_scores(
                all_model_preds, pcd
            )
            unique_bboxes.append(unique_bboxes_inst)
            data.append(data_inst)
            variances.append(variances_inst)
            h_i.append(h_i_inst)
            r_i.append(r_i_inst)

            # Prepare the bounding boxes and scores for each model
            model_bboxes = [model_pred["bboxes_3d"] for model_pred in all_model_preds]
            model_scores = [model_pred["scores_3d"] for model_pred in all_model_preds]

            # If a results_path is provided, save the BEV visualization
            if results_path:
                pcd_file_name = os.path.splitext(os.path.basename(sensor_info[i]["points"]))[0]
                output_path = os.path.join(results_path, f"{pcd_file_name}.png")
                self.plot_bev(pcd, model_bboxes, model_scores, model_names, unique_bboxes_inst, output_path)

        return unique_bboxes, data, variances, h_i, r_i

    def compute_3d_iou_batch(self, bboxes1, bboxes2):
        """
        Computes the Intersection-over-Union (IoU) between two sets of 3D bounding boxes.

        Parameters:
        bboxes1 (ndarray): Array of bounding boxes (N, 7) [x, y, z, dx, dy, dz, heading].
        bboxes2 (ndarray): Array of bounding boxes (M, 7) [x, y, z, dx, dy, dz, heading].

        Returns:
        ndarray: IoU values between each pair of bounding boxes in bboxes1 and bboxes2.
        """
        box_tensor = torch.tensor(bboxes1[:, :7])
        boxes_tensor = torch.tensor(bboxes2[:, :7])
        ious = bbox_overlaps_3d(box_tensor, boxes_tensor).numpy()
        return ious

    def get_point_count_3d(self, boxes, point_cloud):
        """
        Counts the number of LiDAR points within each 3D bounding box.

        Parameters:
        boxes (ndarray): Array of bounding boxes (N, 7) [x, y, z, dx, dy, dz, heading].
        point_cloud (ndarray): LiDAR point cloud data (N_points, 4).

        Returns:
        ndarray: Number of points inside each bounding box.
        """
        masks = box_np_ops.points_in_rbbox(point_cloud[:, :3], boxes)
        return masks.sum(axis=0)

    def get_distance_2d(self, boxes):
        """
        Computes the distances from the sensor to the centers of each bounding box.

        Parameters:
        boxes (ndarray): Array of bounding boxes (N, 7) [x, y, z, dx, dy, dz, heading].

        Returns:
        ndarray: Distances to the centers of the bounding boxes.
        """
        centers = boxes[:, :2]
        distances = np.linalg.norm(centers, axis=1)
        return distances

    def calculate_scores(self, model_predictions, point_cloud):
        """
        Calculates rare example scores by combining predictions from multiple models and applying clustering.

        Parameters:
        model_predictions (list): List of model prediction dictionaries with 'labels_3d', 'scores_3d', 'bboxes_3d'.
        point_cloud (ndarray): LiDAR point cloud data (N_points, 4).

        Returns:
        tuple: Unique bounding boxes, confidence score matrix, variances, hard example flags, and rare example scores.
        """
        all_bboxes_list, all_scores_list, model_indices_list = [], [], []

        for model_idx, model_pred in enumerate(model_predictions):
            bboxes = np.array(model_pred["bboxes_3d"])
            scores = np.array(model_pred["scores_3d"])
            distances = self.get_distance_2d(bboxes)

            mask = (scores > self.min_score) & (distances < self.d_thresh)
            bboxes = bboxes[mask]
            scores = scores[mask]

            all_bboxes_list.append(bboxes)
            all_scores_list.append(scores)
            model_indices_list.append(np.full(len(bboxes), model_idx))

        all_bboxes = np.vstack(all_bboxes_list)
        if not all_bboxes.shape[0]:
            print(f"WARNING!! No bounding boxes were found at score threshold: {self.min_score}")
            return tuple(np.array([]) for _ in range(5))
        all_scores = np.hstack(all_scores_list)
        model_indices = np.hstack(model_indices_list)

        iou_matrix = self.compute_3d_iou_batch(all_bboxes, all_bboxes)
        np.fill_diagonal(iou_matrix, 0)
        adjacency_matrix = (iou_matrix > self.iou_threshold).astype(int)

        graph = csr_matrix(adjacency_matrix)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        unique_bboxes = []
        counts = 0
        for c in range(n_components):
            indices = np.where(labels == c)[0]
            counts = counts + len(indices)
            max_score_idx = indices[np.argmax(all_scores[indices])]
            unique_bboxes.append(all_bboxes[max_score_idx])
        assert counts == len(all_bboxes), f"Different {counts}, {len(all_bboxes)}"
        unique_bboxes = np.array(unique_bboxes)

        data = []
        for c in range(n_components):
            indices = np.where(labels == c)[0]
            cluster_model_indices = model_indices[indices]
            cluster_scores = all_scores[indices]
            scores_per_model = [
                (
                    np.max(all_scores[indices[cluster_model_indices == model_idx]])
                    if model_idx in cluster_model_indices
                    else 0
                )
                for model_idx in range(len(model_predictions))
            ]
            data.append(scores_per_model)

        data = np.array(data)
        variances = np.var(data, axis=1, ddof=0)
        point_counts = self.get_point_count_3d(unique_bboxes, point_cloud)

        h_i = np.where(point_counts > self.p_thresh, 1, 0)
        r_i = h_i * variances

        return unique_bboxes, data, variances, h_i, r_i

    def plot_bev(self, pcd, model_bboxes, model_scores, model_names, unique_bboxes_inst, output_path):
        """
        Plots and saves the Bird's Eye View (BEV) visualization of the point cloud and bounding boxes for each model.

        Parameters:
        pcd (ndarray): LiDAR point cloud data (N_points, 5).
        model_bboxes (list of ndarrays): List of bounding boxes for each model (N, 7 for each model).
        model_scores (list of lists): Confidence scores for each model's bounding boxes.
        model_names (list of str): Names of the models corresponding to the bounding boxes.
        output_path (str): File path to save the BEV image.
        """
        plt.figure(figsize=(10, 10))

        # Filter point cloud to show only the points within [-self.d_thresh, self.d_thresh] range
        mask = (
            (pcd[:, 0] >= -self.d_thresh)
            & (pcd[:, 0] <= self.d_thresh)
            & (pcd[:, 1] >= -self.d_thresh)
            & (pcd[:, 1] <= self.d_thresh)
        )
        filtered_pcd = pcd[mask]

        # Plot the filtered point cloud with 50% transparency
        plt.scatter(filtered_pcd[:, 0], filtered_pcd[:, 1], s=0.5, color="blue", alpha=0.25, label="Point Cloud")

        # Define a color palette for different models
        colors = ["red", "green", "orange", "purple", "cyan"]  # Add more colors if you have more models
        unified_color = "magenta"
        # Plot bounding boxes and confidence scores for each model with different colors
        for model_idx, (bboxes, scores, model_name) in enumerate(zip(model_bboxes, model_scores, model_names)):
            color = colors[model_idx % len(colors)]  # Cycle through colors if more models than colors

            for bbox, score in zip(bboxes, scores):
                if np.isscalar(score):  # Check if score is a scalar
                    if score > self.min_score:
                        self._plot_bbox_and_score(bbox, score, color)
                elif np.any(score > self.min_score):  # Handle arrays by checking if any score is above the threshold
                    max_score = np.max(score)  # Choose the highest score in the array
                    self._plot_bbox_and_score(bbox, max_score, color)

            # Add each model to the legend
            plt.scatter([], [], color=color, label=f"{model_name}")
        for bbox in unique_bboxes_inst:
            self._plot_bbox_and_score(bbox, 1, unified_color, linestyle=":")
        plt.scatter([], [], color=unified_color, label="combined_bbox")

        # Set the axis limits to match the range of d_thresh
        plt.xlim(-self.d_thresh, self.d_thresh)
        plt.ylim(-self.d_thresh, self.d_thresh)

        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.title("BEV Visualization")
        plt.legend()  # Add the legend to the plot
        plt.savefig(output_path)
        plt.close()

    def _plot_bbox_and_score(self, bbox, score, color, linestyle=None):
        """
        Helper method to plot bounding box and its confidence score for a given model.

        Parameters:
        bbox (ndarray): Bounding box to plot (7 elements: [x, y, z, dx, dy, dz, heading]).
        score (float): Confidence score of the bounding box.
        color (str): Color for the bounding box and the corresponding confidence score text.
        """
        # Extract bounding box parameters

        x, y, dx, dy, heading = bbox[0], bbox[1], bbox[3], bbox[4], bbox[6]

        if np.sqrt(x * x + y * y) > self.d_thresh:
            return
        # Compute the four corners of the bounding box
        corners = np.array(
            [
                [-dx / 2, -dy / 2],  # Bottom-left corner
                [dx / 2, -dy / 2],  # Bottom-right corner
                [dx / 2, dy / 2],  # Top-right corner
                [-dx / 2, dy / 2],  # Top-left corner
            ]
        )

        # Rotation matrix for the heading angle
        rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])

        # Rotate the corners around the center of the bounding box
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # Translate corners to the actual position (centered at (x, y))
        rotated_corners[:, 0] += x
        rotated_corners[:, 1] += y

        # Create a polygon using the rotated corners
        polygon = patches.Polygon(
            rotated_corners,
            fill=False,
            edgecolor=color,
            linewidth=2,  # Bolden the bounding box lines
            alpha=0.75,  # Slight transparency
            linestyle=linestyle,
        )
        plt.gca().add_patch(polygon)

        # Plot the confidence score text at the center of the bounding box
        plt.text(
            x, y, f"{score:.2f}", color=color, alpha=0.75, fontsize=10, ha="center", va="center", fontweight="bold"
        )
