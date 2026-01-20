from collections import defaultdict
import json
from pathlib import Path 
from typing import Dict, Any

import polars as pl 

class T4MetricV2DataFrame:

    def __init__(self, output_dataframe_path: Path) -> None:
        """
        Initialize the T4MetricV2DataFrame.

        Args:
            output_path (Path): The path to save the dataframe.
        """
        self.output_dataframe_path = output_dataframe_path
        self.output_dataframe_path.parent.mkdir(parents=True, exist_ok=True)

    def read_json_to_dict(self, json_path: Path) -> Dict[str, Any]:
        """
        Read a JSON file and return its contents as a Python dictionary.

        Args:
            json_path (Path): The path to the JSON file.
        """
        with open(json_path, "r") as file:
            return json.load(file)

    def __call__(
        self, 
        aggregated_metric_scalars: Dict[str, Any], 
        aggregated_metric_data: Dict[str, Any]) -> pl.DataFrame:
        """
        Convert JSON compatible format to a parquet dataframe, where special characters "/" and "." 
        are converted to "_".
        Both aggregated metric scalars and aggregated metric data share the same structure in the format of:
        ```
        {
            "evaluator_name": {
                "metrics": {
                    "metric_name": metric_value,
                },
                "aggregated_metric_label": {
                    "label_name": {
                        "metric_name": [],
                    },
                },
                "metadata": {
                    "metadata_name": metadata_value,
                },
                "metadata_label": {
                    "label_name": {
                        "metadata_name": metadata_value,
                    },
                },
            },
        }
        ```
        Args:
            aggregated_metric_scalars (Dict[str, Any]): The aggregated metric scalars and metadata.
            aggregated_metric_data (Dict[str, Any]): The aggregated metric data for a task, 
                for example, detection/precisions.
        """
        df = defaultdict(list)
        for evaluator_name, metric_dict in aggregated_metric_scalars.items():
            selected_evaluator_metric_data = aggregated_metric_data.get(evaluator_name, None)
            if selected_evaluator_metric_data is None:
                raise ValueError(f"Evaluator {evaluator_name} not found in aggregated metric data.")
            
            # Parse evaluator name to location/vehicle_type/evaluator_bucket_name
            location, vehicle_type, evaluator_bucket_name = evaluator_name.split("/")
            df["location"].append(location)
            df["vehicle_type"].append(vehicle_type)
            df["evaluator_bucket_name"].append(evaluator_bucket_name)
            
            # Parse aggregated_metric_scalars
            current_df = defaultdict(list)
            for metric_header_name, metric_header_data in metric_dict.items():
                metric_header_data = self._parse_metric_header_data(
                    metric_header_name=metric_header_name, 
                    metric_header_data=metric_header_data
                )
                # update the dataframe with the metric header data
                current_df.update(metric_header_data)
            
            # Parse aggregated_metric_data
            for metric_header_name, metric_header_data in selected_evaluator_metric_data.items():
                metric_header_data = self._parse_metric_header_data(
                    metric_header_name=metric_header_name, 
                    metric_header_data=metric_header_data
                )
                # update the dataframe with the metric header data
                current_df.update(metric_header_data)

            for metric_column_name, metric_column_data in current_df.items():
                df[metric_column_name].extend(metric_column_data)

        df = pl.from_dict(df)
        return df
    
    def _parse_metric_header_data(self, metric_header_name: str, metric_header_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the metric header data.

        Args:
            metric_header_data (Dict[str, Any]): The metric header data, for example, 
            given the metric_header_name "metrics", the metric_header_data is:
            ```
            {
                "metric_name": metric_value,
            },
            ```
        """
        if metric_header_name in ["metadata_label", "aggregated_metric_label"]:
            return self._parse_metric_label_column_data(metric_header_data)
        else:
            return self._parse_metric_column_data(metric_header_data)
    
    def _parse_metric_label_column_data(self, metric_header_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the metric label column data.

        Args:
            metric_header_data (Dict[str, Any]): The metric header data.
        """
        df = defaultdict(list)
        for _, metric_dict in metric_header_data.items():
            for metric_name, metric_value in metric_dict.items():
                metric_column_name = self._parse_metric_column_name(metric_name)
                df[metric_column_name].append(metric_value)
        return df
    
    def _parse_metric_column_data(self, metric_header_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the metric label column data.

        Args:
            metric_header_data (Dict[str, Any]): The metric header data.
        """
        df = defaultdict(list)
        for metric_name, metric_value in metric_header_data.items():
            metric_column_name = self._parse_metric_column_name(metric_name)
            # Nested dict type
            if isinstance(metric_value, dict):
                # Make it list of values
                values = list(metric_value.values())
                # Make it list of keys 
                keys = list(metric_value.keys())
                df[f"{metric_column_name}_keys"].append(keys)
                df[f"{metric_column_name}_values"].append(values)
            else:
                df[metric_column_name].append(metric_value)
        return df
        
    @staticmethod
    def _parse_metric_column_name(metric_name: str) -> str:
        """
        Parse the metric column name.

        Args:
            metric_name (str): The metric name.
        """
        # Remove prefix, such as "metrics/" or "metadata/"
        metric_name = metric_name.split("/")[-1]
        return metric_name.replace("/", "_").replace(".", "_")

    def save_dataframe(self, df: pl.DataFrame) -> None:
        """
        Save the dataframe to the output path.

        Args:
            df (pl.DataFrame): The dataframe to save.
        """
        df.write_parquet(self.output_dataframe_path)


# if __name__ == "__main__":
#     aggregated_metric_scalars = "work_dirs/centerpoint/j6gen2_base/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_amp_t4metric_v2/20260119_042508/testing/db_largebus/aggregated_metrics.json"
#     aggregated_metric_data = "work_dirs/centerpoint/j6gen2_base/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_amp_t4metric_v2/20260119_042508/testing/db_largebus/aggregated_metrics_data.json"
#     t4metric_v2_dataframe = T4MetricV2DataFrame(output_dataframe_path=Path("work_dirs/centerpoint/j6gen2_base/T4Dataset/second_secfpn_4xb16_121m_j6gen2_base_amp_t4metric_v2/20260119_042508/testing/db_largebus/output_dataframe.parquet"))
    
#     aggregated_metric_scalars = t4metric_v2_dataframe.read_json_to_dict(aggregated_metric_scalars)
#     aggregated_metric_data = t4metric_v2_dataframe.read_json_to_dict(aggregated_metric_data)
#     df = t4metric_v2_dataframe(aggregated_metric_scalars, aggregated_metric_data)
#     t4metric_v2_dataframe.save_dataframe(df)
#     print(df.columns)