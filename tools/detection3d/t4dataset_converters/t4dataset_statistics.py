import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
from t4_devkit.schema import Sample, SampleData


@dataclass(frozen=True)
class T4DatasetSceneMetadata:
    """Class to store metadata for a T4Dataset."""

    scene_id: str
    location: str
    vehicle_type: str

    @property
    def get_frame_prefix(self) -> str:
        return self.location + "/" + self.vehicle_type


class T4DatasetStatistics:
    """
    Class to generate statistics for a split in T4Dataset.
    TODO(KokSeang): This class will be unified with T4MetricV2DataFrame.
    """

    def __init__(self, output_dir: Path, split_name: str, version: str):
        self.output_dir = output_dir
        self.split_name = split_name
        self.version = version
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.statistics = {}

    def add_samples(self, samples: List[Sample], bucket_name: str, scene_metadata: T4DatasetSceneMetadata):
        """Add samples to the statistics."""
        # Initialize the bucket if it doesn't exist
        if bucket_name not in self.statistics:
            self.statistics[bucket_name] = {
                "metadata": {
                    f"metadata/{self.split_name}_num_frame_distribution": defaultdict(int),
                    f"metadata/{self.split_name}_total_num_frames": 0,
                },
                "metadata_label": {},
            }

        self.statistics[bucket_name]["metadata"][f"metadata/{self.split_name}_num_frame_distribution"][
            scene_metadata.get_frame_prefix
        ] += len(samples)
        self.statistics[bucket_name]["metadata"][f"metadata/{self.split_name}_total_num_frames"] += len(samples)

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
            for _, column_data in columns.items():
                current_df.update(self._parse_column_data(column_data))

            for column_name, data in current_df.items():
                df[column_name].extend(data)

        return pl.from_dict(df)
