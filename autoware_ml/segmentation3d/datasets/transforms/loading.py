import numpy as np

from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations
from mmdet3d.structures.points import get_points_type
from mmengine import check_file_exist
from mmengine.fileio import get
from mmengine.registry import TRANSFORMS
from pyquaternion import Quaternion
from typing import List, Optional, Union


@TRANSFORMS.register_module()
class LoadPointsWithIdentifierFromFile(BaseTransform):
    """Load Points With Identifier From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normalize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normalize the elongation. This is
            usually used in Waymo dataset. Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        coord_type: str,
        lidar_sources: List[str],
        load_dim: int = 6,
        use_dim: Union[int, List[int]] = [0, 1, 2],
        shift_height: bool = False,
        use_color: bool = False,
        norm_intensity: bool = False,
        norm_elongation: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if not isinstance(lidar_sources, list):
            raise TypeError("lidar_sources must be a list of channel names.")
        if len(lidar_sources) == 0:
            raise ValueError("lidar_sources must not be empty.")
        self.lidar_sources = lidar_sources
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results["lidar_points"]["lidar_path"]
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if "lidar_sources_info" not in results or "lidar_sources" not in results:
            raise KeyError("Expected 'lidar_sources_info' and 'lidar_sources' in results.")
        source_map = results["lidar_sources"]
        sources_info = results["lidar_sources_info"]
        sources = sources_info.get("sources", [])
        token_to_range = {s["sensor_token"]: (s["idx_begin"], s["length"]) for s in sources}
        selected_points = []
        selected_sources = []
        for channel in self.lidar_sources:
            if channel not in source_map:
                raise KeyError(f"lidar_sources does not contain channel '{channel}'.")
            sensor_token = source_map[channel]["sensor_token"]
            if sensor_token not in token_to_range:
                raise KeyError(f"lidar_sources_info missing sensor_token '{sensor_token}'.")
            idx_begin, length = token_to_range[sensor_token]
            points_transformed = points[idx_begin : idx_begin + length]
            translation = source_map[channel]["translation"]
            rotation = Quaternion(source_map[channel]["rotation"]).rotation_matrix
            # Stored transform is sensor_to_base (T_sensor_to_base).
            # Points are stored in base frame; transform back to sensor frame (inverse).
            # p_sensor = (p_base - t_sensor_to_base) @ R_sensor_to_base
            points_transformed[:, :3] = (points_transformed[:, :3] - translation) @ rotation
            selected_points.append(points_transformed)
            selected_sources.append(dict(sensor_token=sensor_token, idx_begin=idx_begin, length=length))
        points = np.concatenate(selected_points, axis=0) if selected_points else points[:0]
        results["lidar_sources_info"] = dict(
            stamp=sources_info.get("stamp"),
            sources=selected_sources,
        )
        results["lidar_sources_selection"] = selected_sources
        if self.norm_intensity:
            assert (
                len(self.use_dim) >= 4
            ), f"When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}"  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert (
                len(self.use_dim) >= 5
            ), f"When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}"  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results["points"] = points

        return results


@TRANSFORMS.register_module()
class LoadSegAnnotationsWithIdentifier3D(LoadAnnotations):
    """Load AnnotationsWithIdentifier3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Required Keys:

    - pts_semantic_mask_path (str): Path of semantic mask file.
    - pts_semantic_mask_categories (dict): Categories of the current 3D semantic mask.
    - dataset (object): Dataset object.

        - metainfo (dict): Metainfo of the dataset.

            - class_mapping (dict): Class mapping of the dataset.

    Added Keys:

    - pts_semantic_mask (np.int64): Semantic mask of each point.


    Args:
        seg_3d_dtype (str): String of dtype of 3D semantic masks.
            Defaults to 'np.int64'.
        seg_offset (int): The offset to split semantic and instance labels from
            panoptic labels. Defaults to None.
        dataset_type (str): Type of dataset used for splitting semantic and
            instance labels. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        seg_3d_dtype: str = "np.int64",
        seg_offset: int = None,
        dataset_type: str = None,
        backend_args: Optional[dict] = None,
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_mask=False,
            with_seg=True,
            poly2mask=False,
            backend_args=backend_args,
        )
        self.seg_3d_dtype = eval(seg_3d_dtype)
        self.seg_offset = seg_offset
        self.dataset_type = dataset_type

    def _load_semantic_seg_3d(self, results: dict) -> dict:
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        assert "pts_semantic_mask_path" in results, "pts_semantic_mask_path key is missing in input dictionary"
        assert "lidar_sources_selection" in results, "lidar_sources_selection key is missing in input dictionary"
        assert (
            "pts_semantic_mask_categories" in results
        ), "pts_semantic_mask_categories key is missing in input dictionary"
        assert "dataset" in results, "dataset key is missing in input dictionary"
        assert hasattr(results["dataset"], "metainfo"), "metainfo attribute is missing in dataset"
        assert "class_mapping" in results["dataset"].metainfo, "class_mapping key is missing in metainfo"

        pts_semantic_mask_path = results["pts_semantic_mask_path"]

        try:
            mask_bytes = get(pts_semantic_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(pts_semantic_mask_path, dtype=np.int64)

        selections = results["lidar_sources_selection"]
        segments = [pts_semantic_mask[s["idx_begin"] : s["idx_begin"] + s["length"]] for s in selections]
        pts_semantic_mask = np.concatenate(segments, axis=0) if segments else pts_semantic_mask[:0]

        pts_semantic_mask_categories = results["pts_semantic_mask_categories"]
        class_mapping = results["dataset"].metainfo["class_mapping"]
        raw_to_category = {v: k for k, v in pts_semantic_mask_categories.items()}

        def _map_segment(raw_label):
            category = raw_to_category.get(raw_label, None)
            if category is None:
                return self.ignore_index
            return class_mapping.get(category, self.ignore_index)

        pts_semantic_mask = np.vectorize(_map_segment)(pts_semantic_mask).astype(np.int64)

        if self.dataset_type == "semantickitti":
            pts_semantic_mask = pts_semantic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        # nuScenes loads semantic and panoptic labels from different files.

        results["pts_semantic_mask"] = pts_semantic_mask

        # 'eval_ann_info' will be passed to evaluator
        if "eval_ann_info" in results:
            results["eval_ann_info"]["pts_semantic_mask"] = pts_semantic_mask
        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        results = super().transform(results)
        results = self._load_semantic_seg_3d(results)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = "    "
        repr_str = self.__class__.__name__ + "(\n"
        repr_str += f"{indent_str}seg_offset={self.seg_offset})"

        return repr_str
