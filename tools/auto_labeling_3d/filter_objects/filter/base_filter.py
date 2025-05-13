import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, NewType


class BaseFilter(ABC):
    def __init__(self, logger: logging.Logger):
        self.settings = {}
        self.logger = logger

    @abstractmethod
    def _should_filter_instance(self, pred_instance_3d: Dict[str, Any], **kwargs) -> bool:
        """
        Check if an instance should be filtered based on specific criteria.
        This is an abstract method that must be implemented in derived classes.

        Args:
            pred_instance_3d (Dict[str, Any]): Predicted result of each instance in pseudo label.
            **kwargs: Additional arguments that may be needed by derived classes

        Returns:
            bool: True if instance should be filtered out, False otherwise
        """
        pass

    @abstractmethod
    def filter(self, predicted_result_info: Dict[str, Any], info_name: str) -> Dict:
        """
        Apply filtering to the pseudo labels.
        This is an abstract method that must be implemented in derived classes.

        Args:
            predicted_result_info (Dict[str, Any]): Info dict that contains predicted result.
            info_name (str): Name of each model used for generating info file.

        Returns:
            Dict[str, Any]: Filtered dataset info
        """
        pass

    def _report_filter_statistics(
        self, total_instances: Dict[str, int], filtered_instances: Dict[str, int], info_name: str
    ) -> Dict:
        """Report filtering statistics and generate summary.

        Args:
            total_instances (Dict[str, int]): Total instances count per category
            filtered_instances (Dict[str, int]): Filtered instances count per category
            info_name (str): Name of each model used for generating info file.

        Returns:
            Dict: Dictionary containing all filtering statistics
        """
        # Calculate filtering percentages for each category
        filtering_percentages = {
            category: (
                (filtered_instances[category] / total_instances[category] * 100)
                if total_instances[category] > 0
                else 0
            )
            for category in total_instances.keys()
        }

        # Calculate total statistics
        total_all = sum(total_instances.values())
        filtered_all = sum(filtered_instances.values())
        total_percentage = (filtered_all / total_all * 100) if total_all > 0 else 0

        # Log statistics
        self.logger.info(f"Filtering statistics of {self.__class__.__name__} for {info_name}")
        self.logger.info("Per category:")
        for category in total_instances.keys():
            self.logger.info(f"  {category}:")
            self.logger.info(f"    Total instances: {total_instances[category]}")
            self.logger.info(f"    Filtered instances: {filtered_instances[category]}")
            self.logger.info(f"    Remaining instances: {total_instances[category] - filtered_instances[category]}")
            self.logger.info(f"    Filtered percentage: {filtering_percentages[category]:.1f}%")

        self.logger.info("Total across all categories:")
        self.logger.info(f"  Total instances: {total_all}")
        self.logger.info(f"  Filtered instances: {filtered_all}")
        self.logger.info(f"  Remaining instances: {total_all - filtered_all}")
        self.logger.info(f"  Filtered percentage: {total_percentage:.1f}%")
        self.logger.info(f"\n")

        # Return statistics dictionary
        return {
            "total_instances": dict(total_instances),
            "filtered_instances": dict(filtered_instances),
            "remaining_instances": {
                category: total_instances[category] - filtered_instances[category]
                for category in total_instances.keys()
            },
            "filtering_percentages": filtering_percentages,
        }
