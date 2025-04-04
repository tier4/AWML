from typing import List

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from sklearn.cluster import DBSCAN, HDBSCAN

from tools.data_splitting.utils import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT, compute_kl_divergence


# Define a custom distance function
def custom_distance(x, y):
    return np.abs(x - y)


class DataSplittingProblem(ElementwiseProblem):
    def __init__(
        self,
        scenario_dataframe: pd.DataFrame,
    ):
        self.scenario_dataframe = scenario_dataframe
        self.total_len = len(self.scenario_dataframe)
        # All available categories
        self.categories = []
        self.timestamp_clusters = self._compute_timestamp_clustering()
        self.spatial_clusters = self._compute_spatial_clustering()
        self.data_index = []
        super().__init__(n_var=self.total_len, n_obj=3, n_constr=1, xl=0, xu=1)  # Split ratio for Group 1 and Group 2

    def _compute_spatial_clustering(self):
        coordinates = self.scenario_dataframe[["ego_pose_x", "ego_pose_y"]].values
        # Apply HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=2, min_samples=2)
        clusters = clusterer.fit_predict(coordinates)

        max_label = max(clusters)
        acc_cluster = np.arange(len(clusters[clusters == -1])) + 1
        noisy_clusters = acc_cluster + max_label
        clusters[clusters == -1] = noisy_clusters
        return clusters

    def _compute_timestamp_clustering(self):
        timestamps = self.scenario_dataframe["starting_timestamp"].values
        # Apply DBSCAN
        # 3 days × 24 × 60 × 60 =259,200 seconds
        clusterer = DBSCAN(min_samples=2, eps=259200, metric=custom_distance)
        clusters = clusterer.fit_predict(timestamps)

        max_label = max(clusters)
        acc_cluster = np.arange(len(clusters[clusters == -1])) + 1
        noisy_clusters = acc_cluster + max_label
        clusters[clusters == -1] = noisy_clusters
        return clusters

    def _compute_temporal_distribution_difference(
        self, train_split: pd.DataFrame, test_split: pd.DataFrame, val_split: pd.DataFrame, weights: List[float]
    ):
        """
        Calculate the absolute difference in class distribution (e.g., cars and pedestrians)
        between three groups.

        Arguments:
            - group1, group2, group3: Class distributions in each group (e.g., [cars_percentage, pedestrians_percentage])

        Returns:
            - Total difference in class distribution
        """
        train_split_clusters = {i: 0 for i in range(max(self.timestamp_clusters) + 1)}
        test_split_clusters = {i: 0 for i in range(max(self.timestamp_clusters) + 1)}
        val_split_clusters = {i: 0 for i in range(max(self.timestamp_clusters) + 1)}

        for i in self.timestamp_clusters[train_split.index]:
            train_split_clusters[i] += 1

        for i in self.timestamp_clusters[test_split.index]:
            test_split_clusters[i] += 1

        for i in self.timestamp_clusters[val_split.index]:
            val_split_clusters[i] += 1

        train_split_distribution = np.asarray([i / len(train_split) for i in train_split_clusters.values()])
        test_split_distribution = np.asarray([i / len(test_split) for i in test_split_clusters.values()])
        val_split_distribution = np.asarray([i / len(val_split) for i in val_split_clusters.values()])

        # P(train_split_distribution || test_split_distribution)
        kl_train_test_split = compute_kl_divergence(train_split_distribution, test_split_distribution)

        # P(train_split_distribution || val_split_distribution)
        kl_train_val_split = compute_kl_divergence(train_split_distribution, val_split_distribution)

        # P(test_split_distribution || val_split_distribution)
        kl_test_val_split = compute_kl_divergence(test_split_distribution, val_split_distribution)

        return weights[0] * kl_train_test_split + weights[1] * kl_train_val_split + weights[2] * kl_test_val_split

    def _compute_class_distribution_difference(
        self, train_split: pd.DataFrame, test_split: pd.DataFrame, val_split: pd.DataFrame
    ):
        """
        Calculate the absolute difference in class distribution (e.g., cars and pedestrians)
        between three groups.

        Arguments:
        - group1, group2, group3: Class distributions in each group (e.g., [cars_percentage, pedestrians_percentage])

        Returns:
        - Total difference in class distribution
        """
        # Calculate Kl-divergence between group1 and group2
        train_split_total_counts = train_split[self.categories].sum()
        train_split_distribution = np.asarray(
            [train_split[category].sum() / train_split_total_counts for category in self.categories]
        )

        test_split_total_counts = test_split[self.categories].sum()
        test_split_distribution = np.asarray(
            [test_split[category].sum() / test_split_total_counts for category in self.categories]
        )

        val_split_total_counts = val_split[self.categories].sum()
        val_split_distribution = np.asarray(
            [val_split[category].sum() / val_split_total_counts for category in self.categories]
        )

        # P(train_split_distribution || test_split_distribution)
        kl_train_test_split = compute_kl_divergence(train_split_distribution, test_split_distribution)

        # P(train_split_distribution || val_split_distribution)
        kl_train_val_split = compute_kl_divergence(train_split_distribution, val_split_distribution)

        # P(test_split_distribution || val_split_distribution)
        kl_test_val_split = compute_kl_divergence(test_split_distribution, val_split_distribution)

        return kl_train_test_split + kl_train_val_split + kl_test_val_split

    def _compute_spatial_distribution_difference(self, train_split, test_split, val_split) -> None:
        """ """
        train_split_clusters = {i: 0 for i in range(max(self.spatial_clusters) + 1)}
        test_split_clusters = {i: 0 for i in range(max(self.spatial_clusters) + 1)}
        val_split_clusters = {i: 0 for i in range(max(self.spatial_clusters) + 1)}

        for i in self.spatial_clusters[train_split.index]:
            train_split_clusters[i] += 1

        for i in self.spatial_clusters[test_split.index]:
            test_split_clusters[i] += 1

        for i in self.spatial_clusters[val_split.index]:
            val_split_clusters[i] += 1

        # Should get uniform distribution as much as possible
        true_distribution = np.asarray([1 / len(self.spatial_clusters) for _ in range(max(self.spatial_clusters) + 1)])
        # train_split_distribution = np.asarray([i / len(train_split) for i in train_split_clusters.values()])
        test_split_distribution = np.asarray([i / len(test_split) for i in test_split_clusters.values()])
        val_split_distribution = np.asarray([i / len(val_split) for i in val_split_clusters.values()])

        # P(true_distribution || test_split_distribution)
        kl_true_test_split = compute_kl_divergence(true_distribution, test_split_distribution)

        # P(true_distribution || val_split_distribution)
        kl_true_val_split = compute_kl_divergence(true_distribution, val_split_distribution)

        return kl_true_test_split + kl_true_val_split

    def _evaluate(self, x, out, *args, **kwargs):
        train_split = self.scenario_dataframe[(x < TRAIN_SPLIT)]  # Train split
        test_split = self.scenario_dataframe[(x >= TEST_SPLIT[0]) & (x < TEST_SPLIT[1])]  # Test split
        val_split = self.scenario_dataframe[(x >= VAL_SPLIT)]  # Val split

        class_kl_d = self._compute_class_distribution_difference(train_split, test_split, val_split)
        temporal_kl_d = self._compute_temporal_distribution_difference(
            train_split, test_split, val_split, [3.0, 1.0, 1.0]
        )
        spatial_kl_d = self._compute_spatial_distribution_difference(train_split, test_split, val_split)

        # Return the objectives (maximize the gaps)
        out["F"] = [class_kl_d, -temporal_kl_d, spatial_kl_d]  # Negative because pymoo minimizes by default

        constraint_violation = (
            abs(len(train_split) / self.total_len - 0.75)
            + abs(len(test_split) / self.total_len - 0.20)
            + abs(len(val_split) / self.total_len - 0.05)
        )

        # Constraint is added to the `out` dictionary
        out["G"] = [constraint_violation]  # Ensure the constraint is satisfied (penalty)
