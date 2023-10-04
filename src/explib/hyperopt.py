import json
import logging
import os
import shutil
import typing as T
from datetime import datetime
from typing import Any, Dict, Iterable, Literal

import pandas as pd
import torch
from matplotlib import pyplot as plt
from pyro import distributions as dist
from pyro.distributions.transforms import AffineCoupling, Permute
from ray import tune
from ray.air import RunConfig, session
from torch.utils.data import DataLoader

from src.explib.base import Experiment
from src.explib.config_parser import from_checkpoint
from src.veriflow.flows import NiceFlow
from src.veriflow.networks import AdditiveAffineNN
from src.veriflow.transforms import ScaleTransform

HyperParams = Literal[
    "train",
    "test",
    "coupling_layers",
    "coupling_nn_layers",
    "split_dim",
    "epochs",
    "iters",
    "batch_size",
    "optim",
    "optim_params",
    "base_dist",
]


class HyperoptExperiment(Experiment):
    """Hyperparameter optimization experiment."""

    def __init__(
        self,
        trial_config: Dict[str, Any],
        num_hyperopt_samples: int,
        gpus_per_trial: int,
        cpus_per_trial: int,
        scheduler: tune.schedulers.FIFOScheduler,
        tuner_params: T.Dict[str, T.Any],
        *args,
        **kwargs,
    ) -> None:
        """Initialize hyperparameter optimization experiment.

        Args:
            trial_config (Dict[str, Any]): trial configuration
            num_hyperopt_samples (int): number of hyperparameter optimization samples
            gpus_per_trial (int): number of GPUs per trial
            cpus_per_trial (int): number of CPUs per trial
            scheduler (tune.schedulers.FIFOScheduler): scheduler class
            scheduler_params (Dict[str, T.Any]): scheduler parameters
            tuner_params (T.Dict[str, T.Any]): tuner parameters
        """
        super().__init__(*args, **kwargs)
        self.trial_config = trial_config
        self.scheduler = scheduler
        self.num_hyperopt_samples = num_hyperopt_samples
        self.gpus_per_trial = gpus_per_trial
        self.cpus_per_trial = cpus_per_trial
        self.tuner_params = tuner_params

    @classmethod
    def _trial(
        cls, config: T.Dict[str, T.Any], device: torch.device = "cpu"
    ) -> Dict[str, float]:
        """Worker function for hyperparameter optimization.

        Args:
            config (T.Dict[str, T.Any]): configuration
            device (torch.device, optional): device. Defaults to "cpu".
        Returns:
            Dict[str, float]: trial performance metrics
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                # torch.mps.empty_cache()
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        dataset = config["dataset"]
        data_train = dataset.get_train()
        data_test = dataset.get_test()
        data_val = dataset.get_val()

        flow = config["model_cfg"]["type"](**(config["model_cfg"]["params"]))
        flow.to(device)

        best_loss = float("inf")
        strikes = 0
        for _ in range(config["epochs"]):
            train_loss = flow.fit(
                data_train,
                config["optim_cfg"]["optimizer"],
                config["optim_cfg"]["params"],
                batch_size=config["batch_size"],
                device=device,
            )[-1]

            val_loss = 0
            for i in range(0, len(data_val), config["batch_size"]):
                j = min([len(data_test), i + config["batch_size"]])
                val_loss += float(
                    -flow.log_prob(data_val[i:j][0].to(device)).sum()
                )
            val_loss /= len(data_val)

            session.report(
                {"train_loss": train_loss, "val_loss": val_loss},
                checkpoint=None,
            )
            if val_loss < best_loss:
                strikes = 0
                best_loss = val_loss
                torch.save(flow.state_dict(), "./checkpoint.pt")
            else:
                strikes += 1
                if strikes >= config["patience"]:
                    break

        return {
            "val_loss_best": best_loss,
            "val_loss": val_loss,
        }

    def conduct(
        self, report_dir: os.PathLike, storage_path: os.PathLike = None
    ):
        """Run hyperparameter optimization experiment.

        Args:
            report_dir (os.PathLike): report directory
            storage_path (os.PathLike, optional): Ray logging path. Defaults to None.
        """
        home = os.path.expanduser("~")

        if storage_path is not None:
            tuner_config = {"run_config": RunConfig(storage_path=storage_path)}
        else:
            storage_path = os.path.expanduser("~/ray_results")
            tuner_config = {}

        exptime = str(datetime.now())

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(HyperoptExperiment._trial),
                resources={
                    "cpu": self.cpus_per_trial,
                    "gpu": self.gpus_per_trial,
                },
            ),
            tune_config=tune.TuneConfig(
                scheduler=self.scheduler,
                num_samples=self.num_hyperopt_samples,
                **(self.tuner_params),
            ),
            param_space=self.trial_config,
            **(tuner_config),
        )
        results = tuner.fit()

        # TODO: hacky way to determine the last experiment
        exppath = (
            storage_path
            + [
                "/" + f
                for f in sorted(os.listdir(storage_path))
                if f.startswith("_trial")
            ][-1]
        )
        report_file = os.path.join(
            report_dir, f"report_{self.name}_" + exptime + ".csv"
        )
        results = self._build_report(
            exppath, report_file=report_file, config_prefix="param_"
        )
        best_result = results.iloc[results["val_loss_best"].argmax()].copy()

        self._test_best_model(best_result, exppath, report_dir)

    def _test_best_model(
        self,
        best_result: pd.Series,
        expdir: str,
        report_dir: str,
        device: torch.device = "cpu",
    ) -> pd.Series:
        trial_id = best_result.trial_id
        for d in os.listdir(expdir):
            if trial_id in d:
                shutil.copyfile(
                    os.path.join(expdir, d, "checkpoint.pt"),
                    os.path.join(report_dir, f"{self.name}_best_model.pt"),
                )
                shutil.copyfile(
                    os.path.join(expdir, d, "params.pkl"),
                    os.path.join(report_dir, f"{self.name}_best_config.pkl"),
                )
                break

        best_model = from_checkpoint(
            os.path.join(report_dir, f"{self.name}_best_config.pkl"),
            os.path.join(report_dir, f"{self.name}_best_model.pt"),
        )

        data_test = self.trial_config["dataset"].get_test()
        test_loss = 0
        for i in range(0, len(data_test), 100):
            j = min([len(data_test), i + 100])
            test_loss += float(
                -best_model.log_prob(data_test[i:j][0].to(device)).sum()
            )
        test_loss /= len(data_test)

        best_result["test_loss"] = test_loss
        best_result.to_csv(
            os.path.join(report_dir, f"{self.name}_best_result.csv")
        )

        return best_result

    def _build_report(
        self, expdir: str, report_file: str, config_prefix: str = ""
    ) -> pd.DataFrame:
        """Builds a report of the hyperopt experiment.

        Args:
            expdir (str): The expdir parameter is the path to the experiment directory (ray results folder).
            report_file (str): The report_file parameter is the path to the report file.
            config_prefix: The config_prefix parameter is the prefix for the config items.
        """
        report = None
        print(os.listdir(expdir))
        for d in os.listdir(expdir):
            if os.path.isdir(expdir + "/" + d):
                try:
                    with open(expdir + "/" + d + "/result.json", "r") as f:
                        result = json.loads(
                            '{"val_loss_best'
                            + f.read().split('{"val_loss_best')[-1]
                        )
                except:
                    print(f"error at {expdir + '/' + d}")
                    continue

                config = result["config"]
                for k in config.keys():
                    result[config_prefix + k] = (
                        config[k]
                        if not isinstance(config[k], Iterable)
                        else str(config[k])
                    )
                result.pop("config")

                if report is None:
                    report = pd.DataFrame(result, index=[0])
                else:
                    report = pd.concat(
                        [report, pd.DataFrame(result, index=[0])],
                        ignore_index=True,
                    )

        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        report.to_csv(report_file, index=False)
        return report
