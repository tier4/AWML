from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from mmengine.logging import print_log
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from tools.data_splitting.data_split_problem import DataSplittingProblem
from tools.data_splitting.utils import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT


class NSGARunner:
    def __init__(
        self,
        out_path: Path,
        random_seed: int = 0,
        population_size: int = 100,
    ):
        self.out_path = out_path
        self.population_size = population_size
        self.random_seed = random_seed

    def _write_split_to_file(
        self, train_split: pd.DataFrame, test_split: pd.DataFrame, val_split: pd.DataFrame
    ) -> None:
        splits = {
            "train": train_split["scenario_token"].tolist(),
            "val": val_split["scenario_token"].to_list(),
            "test": test_split["scenario_token"].to_list(),
        }

        with open(self.out_path / "data_splits.yaml", "w") as file:
            yaml.dump(splits, file)

    def run(self, scenario_df: pd.DataFrame) -> None:
        print_log("RUnning NSGA-II")
        data_splitting_problem = DataSplittingProblem(scenario_dataframe=scenario_df)
        algorithm = NSGA2(
            pop_size=self.population_size,  # Population size
            seed=self.random_seed,
            crossover=SBX(prob=0.9, eta=15),  # Simulated Binary Crossover (SBX)
            mutation=PolynomialMutation(prob=0.1, eta=20),  # Polynomial mutation
        )
        termination_criteria = DefaultMultiObjectiveTermination(
            xtol=1e-10,
            cvtol=1e-8,
            ftol=0.00025,
            n_max_gen=1000,
        )

        # Run the optimization
        res = minimize(data_splitting_problem, algorithm, termination=termination_criteria, save_history=True)

        # We use the best timestamp splitting to split the data
        best_solution_idx = np.argmin(res.F[:, 1])
        best_solution = res.X[best_solution_idx]  # Best solution with maximum gap between splits

        train_split = scenario_df[best_solution < TRAIN_SPLIT]  # Training split
        test_split = scenario_df[(best_solution >= TEST_SPLIT[0]) & (best_solution < TEST_SPLIT[1])]  # Test split
        val_split = scenario_df[(best_solution >= VAL_SPLIT)]  # Validation split
        print_log(f"Train split: {len(train_split)}, Test split: {len(test_split)}, Val split: {len(val_split)}")

        self._write_split_to_file(train_split, test_split, val_split)
        print_log("Data splitting completed.")
