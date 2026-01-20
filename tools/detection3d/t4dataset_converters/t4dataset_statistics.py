import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
        output_path = self.output_dir / f"t4dataset_{self.version}_{self.split_name}_statistics.json"
        with open(output_path, "w") as f:
            json.dump(self.statistics, f, indent=4)
