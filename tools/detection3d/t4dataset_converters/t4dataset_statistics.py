import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
from t4_devkit.schema import Sample, SampleData


@dataclass(frozen=True)
class T4DatasetSceneMetadata:
    """Class to store metadata for a T4Dataset."""

    scene_id: str
    location: str
    vehicle_type: str

    @property
    def frame_prefix(self) -> str:
        return self.location + "/" + self.vehicle_type


class T4DatasetStatistics:
    """
    Class to generate statistics for a split in T4Dataset.
    TODO(KokSeang): This class will be unified with T4MetricV2DataFrame.
    """

    def __init__(self, output_dir: Path, split_name: str, version: str, class_names: List[str]):
        self.output_dir = output_dir
        self.split_name = split_name
        self.version = version
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.statistics = {}
        self.class_names = class_names

        self.schema = {
            "location": pl.String,
            "vehicle_type": pl.String,
            "suffix_name": pl.String,
            f"{self.split_name}_total_num_frames": pl.Int64,
            f"{self.split_name}_num_frame_keys": pl.List(pl.String),
            f"{self.split_name}_num_frame_values": pl.List(pl.Int64),
            f"{self.split_name}_ego_pose_translation_x_keys": pl.List(pl.String),
            f"{self.split_name}_ego_pose_translation_x_values": pl.List(pl.List(pl.Float64)),
            f"{self.split_name}_ego_pose_translation_y_keys": pl.List(pl.String),
            f"{self.split_name}_ego_pose_translation_y_values": pl.List(pl.List(pl.Float64)),
            f"{self.split_name}_ego_pose_translation_z_keys": pl.List(pl.String),
            f"{self.split_name}_ego_pose_translation_z_values": pl.List(pl.List(pl.Float64)),
        }
        for class_name in self.class_names:
            label_schema = {
                f"{self.split_name}_{class_name}_frequency_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_frequency_values": pl.List(pl.Int64),
                f"{self.split_name}_{class_name}_area_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_area_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_volume_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_volume_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_orientation_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_orientation_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_translation_x_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_translation_x_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_translation_y_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_translation_y_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_translation_z_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_translation_z_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_bev_radial_distance_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_bev_radial_distance_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_velocity_x_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_velocity_x_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_velocity_y_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_velocity_y_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_speed_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_speed_values": pl.List(pl.List(pl.Float64)),
                f"{self.split_name}_{class_name}_num_lidar_pts_keys": pl.List(pl.String),
                f"{self.split_name}_{class_name}_num_lidar_pts_values": pl.List(pl.List(pl.Float64)),
            }
            self.schema.update(label_schema)

    def add_samples(
        self,
        samples: List[Sample],
        infos: List[Dict[str, Any]],
        bucket_name: str,
        scene_metadata: T4DatasetSceneMetadata,
        bev_distance_range: Tuple[float, float],
    ):
        """Add samples to the statistics."""
        # Initialize the bucket if it doesn't exist
        if bucket_name not in self.statistics:
            self.statistics[bucket_name] = {
                "metadata": {
                    f"metadata/{self.split_name}_num_frame": defaultdict(int),
                    f"metadata/{self.split_name}_total_num_frames": 0,
                    f"metadata/{self.split_name}_ego_pose_translation_x": defaultdict(list),
                    f"metadata/{self.split_name}_ego_pose_translation_y": defaultdict(list),
                    f"metadata/{self.split_name}_ego_pose_translation_z": defaultdict(list),
                },
                "metadata_label": {
                    class_name: {
                        f"metadata_label/{self.split_name}_{class_name}_frequency": defaultdict(int),
                        f"metadata_label/{self.split_name}_{class_name}_area": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_volume": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_orientation": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_translation_x": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_translation_y": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_translation_z": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_bev_radial_distance": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_velocity_y": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_velocity_x": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_speed": defaultdict(list),
                        f"metadata_label/{self.split_name}_{class_name}_num_lidar_pts": defaultdict(list),
                    }
                    for class_name in self.class_names
                },
            }

        self.statistics[bucket_name]["metadata"][f"metadata/{self.split_name}_num_frame"][
            scene_metadata.frame_prefix
        ] += len(samples)
        self.statistics[bucket_name]["metadata"][f"metadata/{self.split_name}_total_num_frames"] += len(samples)

        for info in infos:
            if not len(info):
                continue

            # Save ego pose translation
            self.statistics[bucket_name]["metadata"][f"metadata/{self.split_name}_ego_pose_translation_x"][
                scene_metadata.frame_prefix
            ].append(info["ego2global"][0][3])
            self.statistics[bucket_name]["metadata"][f"metadata/{self.split_name}_ego_pose_translation_y"][
                scene_metadata.frame_prefix
            ].append(info["ego2global"][1][3])
            self.statistics[bucket_name]["metadata"][f"metadata/{self.split_name}_ego_pose_translation_z"][
                scene_metadata.frame_prefix
            ].append(info["ego2global"][2][3])

            # Save object distribution
            object_distributions = self._get_object_distributions(info, scene_metadata, bev_distance_range)
            for class_name, distributions in object_distributions.items():
                for distribution_name, distribution_data in distributions.items():
                    self.statistics[bucket_name]["metadata_label"][class_name][distribution_name][
                        scene_metadata.frame_prefix
                    ] += distribution_data[scene_metadata.frame_prefix]

    def save_to_json(self):
        output_path = self.output_dir / f"t4dataset_{self.version}_statistics_{self.split_name}.json"
        with open(output_path, "w") as f:
            json.dump(self.statistics, f, indent=4)

    def save_to_parquet(self, filename: str = None) -> None:
        """Save data to a parquet file.

        Args:
            filename: Output filename. If None, generates a default name based on
                     split_name and version. Defaults to None.
        """
        if filename is None:
            filename = f"t4dataset_{self.version}_statistics_{self.split_name}.parquet"

        output_path = self.output_dir / filename
        df = self._dict_to_dataframe()
        # Save to parquet
        df.write_parquet(output_path)

    def _get_object_distributions(
        self, info: Dict[str, Any], scene_metadata: T4DatasetSceneMetadata, bev_distance_range: Tuple[float, float]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get object distributions."""
        object_distributions = {
            class_name: {
                f"metadata_label/{self.split_name}_{class_name}_frequency": defaultdict(int),
                f"metadata_label/{self.split_name}_{class_name}_area": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_volume": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_orientation": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_translation_x": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_translation_y": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_translation_z": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_bev_radial_distance": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_velocity_y": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_velocity_x": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_speed": defaultdict(list),
                f"metadata_label/{self.split_name}_{class_name}_num_lidar_pts": defaultdict(list),
            }
            for class_name in self.class_names
        }

        for instance in info["instances"]:
            if not (instance["num_lidar_pts"] > 0 and instance["bbox_label_3d"] > -1):
                continue

            radial_distance = np.linalg.norm(instance["bbox_3d"][:2])
            if radial_distance < bev_distance_range[0] or radial_distance >= bev_distance_range[1]:
                continue

            class_name = self.class_names[instance["bbox_label_3d"]]

            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_frequency"][
                scene_metadata.frame_prefix
            ] += 1
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_area"][
                scene_metadata.frame_prefix
            ].append(instance["bbox_3d"][3] * instance["bbox_3d"][4])
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_volume"][
                scene_metadata.frame_prefix
            ].append(instance["bbox_3d"][3] * instance["bbox_3d"][4] * instance["bbox_3d"][5])
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_orientation"][
                scene_metadata.frame_prefix
            ].append(instance["bbox_3d"][6])
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_translation_x"][
                scene_metadata.frame_prefix
            ].append(instance["bbox_3d"][0])
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_translation_y"][
                scene_metadata.frame_prefix
            ].append(instance["bbox_3d"][1])
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_translation_z"][
                scene_metadata.frame_prefix
            ].append(instance["bbox_3d"][2])
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_bev_radial_distance"][
                scene_metadata.frame_prefix
            ].append(radial_distance)
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_velocity_x"][
                scene_metadata.frame_prefix
            ].append(instance["velocity"][0])
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_velocity_y"][
                scene_metadata.frame_prefix
            ].append(instance["velocity"][1])
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_speed"][
                scene_metadata.frame_prefix
            ].append(np.linalg.norm(instance["velocity"][:2]))
            object_distributions[class_name][f"metadata_label/{self.split_name}_{class_name}_num_lidar_pts"][
                scene_metadata.frame_prefix
            ].append(instance["num_lidar_pts"])

        return object_distributions

    @staticmethod
    def _parse_column_name(metric_name: str) -> str:
        """
        Parse the metric column name.

        Args:
            metric_name (str): The metric name.
        """
        # Remove prefix, such as "metrics/" or "metadata/"
        metric_name = metric_name.split("/")[-1]
        return metric_name.replace("/", "_").replace(".", "_")

    def _parse_column_data(self, column_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the column data.

        Args:
            column_data: The column data.
        """
        df = defaultdict(list)
        for key, column_value in column_data.items():
            column_name = self._parse_column_name(key)
            if isinstance(column_value, dict):
                df[f"{column_name}_keys"].append(list(column_value.keys()))
                df[f"{column_name}_values"].append(list(column_value.values()))
            else:
                df[column_name].append(column_value)
        return df

    def _parse_label_column_data(self, column_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the label column data.

        Args:
            column_data: The column data.
        """
        df = defaultdict(list)
        for _, distributions in column_data.items():
            for distribution_name, distribution_data in distributions.items():
                column_name = self._parse_column_name(distribution_name)
                if isinstance(distribution_data, dict):
                    df[f"{column_name}_keys"].append(list(distribution_data.keys()))
                    df[f"{column_name}_values"].append(list(distribution_data.values()))
                else:
                    df[column_name].append(distribution_data)
        return df

    def _dict_to_dataframe(self) -> pl.DataFrame:
        """Convert nested dictionary to flat DataFrame.

        Args:
            data: Nested dictionary to flatten.

        Returns:
            DataFrame with flattened structure where dict values are flattened to keys and values.
        """
        df = defaultdict(list)
        for bucket_name, columns in self.statistics.items():
            location, vehicle_type, suffix_name = bucket_name.split("/")
            df["location"].append(location)
            df["vehicle_type"].append(vehicle_type)
            df["suffix_name"].append(suffix_name)

            current_df = defaultdict(list)
            for column_header_name, column_data in columns.items():
                if column_header_name == "metadata_label":
                    current_df.update(self._parse_label_column_data(column_data))
                else:
                    current_df.update(self._parse_column_data(column_data))

            for column_name, data in current_df.items():
                df[column_name].extend(data)

        return pl.from_dict(df, schema=self.schema)
