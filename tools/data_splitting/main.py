from pathlib import Path

import fire

from tools.data_splitting.nsga_runner import NSGARunner
from tools.data_splitting.split_runner import SplitRunner


def run(config_path: str, data_root: str, out_path: str) -> None:
    """ """
    split_runner = SplitRunner(
        config_path=Path(config_path),
        data_root_path=Path(data_root),
        out_path=Path(out_path),
    )
    scenario_df = split_runner.run()
    nsga = NSGARunner(out_path=Path(out_path), random_seed=0, population_size=200)
    nsga.run(scenario_df=scenario_df)


if __name__ == "__main__":
    fire.Fire(run)
