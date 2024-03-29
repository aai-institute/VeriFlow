import json
import logging
import os
import shutil
import typing as T
from datetime import datetime
from typing import Any, Dict, Iterable, Literal, Optional
import warnings

import numpy as np
import pandas as pd
from pickle import dump
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import plotly.express as px
from pyro import distributions as dist
from pyro.distributions.transforms import AffineCoupling, Permute
from ray import tune
from ray.air import RunConfig, session

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
    def _trial(cls, config: T.Dict[str, T.Any], device: torch.device = "cpu") -> Dict[str, float]:
        """Worker function for hyperparameter optimization.
        
        Args:
            config (T.Dict[str, T.Any]): configuration
            device (torch.device, optional): device. Defaults to "cpu".
        Returns:
            Dict[str, float]: trial performance metrics
        """
        writer = SummaryWriter()
        # warnings.simplefilter("error")
        torch.autograd.set_detect_anomaly(True)
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

        model_hparams = config["model_cfg"]["params"]
        flow = config["model_cfg"]["type"](**model_hparams)
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
                val_loss += float(-flow.log_prob(data_val[i:j][0].to(device)).sum())
            val_loss /= len(data_val)

            session.report(
                {"train_loss": train_loss, "val_loss": val_loss},
                checkpoint=None,
            )
            if val_loss < best_loss:
                strikes = 0
                best_loss = val_loss
                
                # Create checkpoint
                torch.save(flow.state_dict(), "./checkpoint.pt")
                
                # Advanced logging
                cfg_log = config["logging"]
                if "images" in cfg_log and cfg_log["images"]:
                    img_sample(
                        flow, 
                        "./sample.png", 
                        reshape=cfg_log["image_shape"], 
                        device=device
                    )
                if "scatter" in cfg_log and cfg_log["scatter"]:
                    scatter_sample(
                        flow, 
                        "./scatter.png", 
                        device=device
                    )
                    
                    density_contour_sample(flow, "./density_contour.png", writer=writer, device=device)
                
            else:
                strikes += 1
                if strikes >= config["patience"]:
                    break

        return {
            "val_loss_best": best_loss,
            "val_loss": val_loss,
        }

    def conduct(self, report_dir: os.PathLike, storage_path: os.PathLike = None):
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

        #search_alg = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
        #search_alg = ConcurrencyLimiter(search_alg, max_concurrent=100)
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(HyperoptExperiment._trial),
                resources={"cpu": self.cpus_per_trial, "gpu": self.gpus_per_trial},
            ),
            tune_config=tune.TuneConfig(
                scheduler=self.scheduler,
                #search_alg=search_alg,
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
        results = self._build_report(exppath, report_file=report_file, config_prefix="param_")
        best_result = results.iloc[results["val_loss_best"].argmin()].copy()

        self._test_best_model(best_result, exppath, report_dir, exp_id=exptime)
    
    def _test_best_model(self, best_result: pd.Series, expdir: str, report_dir: str, device: torch.device = "cpu", exp_id: str = "foo" ) -> pd.Series:
        trial_id = best_result.trial_id
        id = f"exp_{exp_id}_{trial_id}"
        for d in os.listdir(expdir):
            if trial_id in d:
                shutil.copyfile(
                    os.path.join(expdir, d, f"checkpoint.pt"), 
                    os.path.join(report_dir, f"{self.name}_{id}_best_model.pt")
                )
                shutil.copyfile(
                    os.path.join(expdir, d, "params.pkl"), 
                    os.path.join(report_dir, f"{self.name}_{id}_best_config.pkl")
                )
                
                if self.trial_config["logging"]["images"]:
                    shutil.copyfile(
                        os.path.join(expdir, d, "sample.png"), 
                        os.path.join(report_dir, f"{self.name}_{id}_sample.png")
                    )
                break
        
        best_model = from_checkpoint(
            os.path.join(report_dir, f"{self.name}_{id}_best_config.pkl"),
            os.path.join(report_dir, f"{self.name}_{id}_best_model.pt")
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
        
    def _build_report(self, expdir: str, report_file: str, config_prefix: str = "") -> pd.DataFrame:
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
                        result = json.loads('{"val_loss_best' + f.read().split('{"val_loss_best')[-1])
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
                        [report, pd.DataFrame(result, index=[0])], ignore_index=True
                    )

        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        report.to_csv(report_file, index=False)
        return report


def img_sample(model, path = "./sample.png", reshape: Optional[Iterable[int]] = None, n=2, device="cpu"):
    """Sample images from a model.
    
    Args:
        model: model to sample from
        path: path to save the image
        reshape: reshape the image
        n: number of samples
        device: device to sample from

    """
    with torch.no_grad():
        sample = np.uint8(np.clip(model.sample(torch.tensor([n, n])).numpy(), 0, 1) * 255)
        
    if reshape is not None:
        reshape = [n, n] + reshape
        sample = sample.reshape(reshape)
    
    sample = np.concatenate(sample, axis=-1)
    sample = np.concatenate(sample, axis=0)
        
    Image.fromarray(sample, mode="L").save(path, dpi=(300, 300))

def scatter_sample(model, path = "./sample.png", n=1000, device="cpu"):
    """Sample images from a model.
    
    Args:
        model: model to sample from
        path: path to save the image
        n: number of samples
        device: device to sample from

    """
    with torch.no_grad():
        sample = model.sample(torch.tensor([n])).numpy()
        
    fig = px.scatter(x=sample[:, 0], y=sample[:, 1])
    fig.write_image(path, width=800, height=800)

def density_contour_sample(model, path="./density_contour.png", n=1000, device="cpu"):
    """Generate a density contour plot from samples of a model and save it as an image.

    Args:
        model: model to sample from.
        path: path to save the image.
        n: number of samples.
        device: device to generate samples from.

    """
    with torch.no_grad():
        # Sample from the model
        sample = model.sample(torch.tensor([n])).to(device).numpy()

    # Create a density contour plot
    fig = px.density_contour(x=sample[:, 0], y=sample[:, 1])

    # Save the plot as an image
    fig.write_image(path, width=800, height=800)