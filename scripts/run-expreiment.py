import os
import typing as T

import click

from src.explib.config_parser import read_config

# In principle one can cast it to os.path.Path
Pathable = T.Union[str, os.PathLike]


@click.command()
@click.option("--report_dir", default="./reports", help="Report file")
@click.option(
    "--config", default="./config.yaml", help="Prefix for config items"
)
@click.option("--storage_path", default=None, help="Prefix for config items")
def run(report_dir: Pathable, config: Pathable, storage_path: Pathable):
    """Loads an experiment from config file conducts the experiment it.

    Args:
        report_dir (str): Directory to save report to.
        config (str): Path to config file. The report is expected to be specified in .yaml format with
        support to some special key functionalities (see :func:`~laplace_flows.experiments.utils.read_config)
        Defaults to "./config.yaml".
        storage_path (str): Path to Ray storage directory. Defaults to None.
    """
    sepline = "\n" + ("-" * 80) + "\n" + ("-" * 80) + "\n"
    print(f"{sepline}Parsing config file:{sepline}")
    config = os.path.abspath(config)
    experiment = read_config(config)
    print(f"{sepline}Done.{sepline}")
    print(f"{sepline}Conducting experiment{sepline}")
    # Conduct experiment
    experiment.conduct(report_dir, storage_path=storage_path)
    print(f"{sepline}Done.{sepline}")


if __name__ == "__main__":
    run()
