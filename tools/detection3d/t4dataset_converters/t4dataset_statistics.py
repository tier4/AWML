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
    """Class to generate statistics for a split in T4Dataset."""

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
        df = self._dict_to_dataframe(data)
        # Save to parquet
        df.write_parquet(output_path)

    def _dict_to_dataframe(self, data: Dict[str, Any]) -> pl.DataFrame:
        """Convert nested dictionary to flat DataFrame.

        Args:
            data: Nested dictionary to flatten.

        Returns:
            DataFrame with flattened structure where dict values are flattened to keys and values.
        """
        for bucket_name, columns in self.statistics.items():
            location, vehicle_type, range_filter_name, bev_distance_range = bucket_name.split("/")
        # rows = []

        # def flatten_dict(d: Dict[str, Any],
        #                  parent_key: str = "",
        #                  sep: str = "_"):
        #     """Recursively flatten nested dictionary. Dict values are flattened to keys and values."""
        #     items = {}
        #     for k, v in d.items():
        #         new_key = f"{parent_key}{sep}{k}" if parent_key else k
        #         if isinstance(v, defaultdict):
        #             # Convert defaultdict to regular dict, then flatten it
        #             v = dict(v)
        #             if isinstance(v, dict) and v:
        #                 # Flatten dict values: each key becomes a column, each value becomes the cell value
        #                 for dict_key, dict_value in v.items():
        #                     # Use dict_key as part of the column name
        #                     column_key = f"{new_key}{sep}{dict_key}"
        #                     items[column_key] = dict_value
        #             else:
        #                 items[new_key] = v
        #         elif isinstance(v, dict):
        #             # Flatten dict values: each key becomes a column, each value becomes the cell value
        #             if v:
        #                 for dict_key, dict_value in v.items():
        #                     # Use dict_key as part of the column name
        #                     column_key = f"{new_key}{sep}{dict_key}"
        #                     items[column_key] = dict_value
        #             else:
        #                 items[new_key] = None
        #         elif isinstance(v, (list, tuple)):
        #             # Convert lists/tuples to JSON string for parquet compatibility
        #             items[new_key] = json.dumps(v) if v else None
        #         else:
        #             items[new_key] = v
        #     return items

        # # Handle top-level structure
        # if isinstance(data, dict):
        #     for bucket_name, bucket_data in data.items():
        #         flattened = flatten_dict({bucket_name: bucket_data})
        #         flattened["bucket_name"] = bucket_name
        #         rows.append(flattened)

        # if rows:
        #     return pl.DataFrame(rows)
        # else:
        #     # If no nested structure, create a single-row DataFrame
        #     return pl.DataFrame([data])
