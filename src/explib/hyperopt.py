import io
import json
import logging
import os
import shutil
import typing as T
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from pickle import dump
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
import plotly.express as px
import ray
from ray import tune
from ray.air import RunConfig, session

from src.explib.base import Experiment
from src.explib.config_parser import from_checkpoint
from src.veriflow.flows import NiceFlow
from src.veriflow.networks import AdditiveAffineNN
from src.veriflow.transforms import ScaleTransform



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
            device (torch.device, optional): device. Defaults to None.
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
        for epoch in range(config["epochs"]):
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
                
                torch.save(flow.state_dict(), f"./checkpoint.pt")
                
                # Advanced logging
                try:
                    cfg_log = config["logging"]
                    if "images" in cfg_log and cfg_log["images"]:
                        img_sample(
                            flow, 
                            f"sample",
                            step=epoch, 
                            reshape=cfg_log["image_shape"], 
                            device=device,
                            writer=writer
                        )
                    if "scatter" in cfg_log and cfg_log["scatter"]:
                        scatter_sample(
                            flow, 
                            f"scatter",
                            step=epoch, 
                            device=device,
                            writer=writer
                        )
                        
                        density_contour_sample(
                            flow,
                            f"density_contour",
                            step=epoch,
                            writer=writer,
                            device=device
                        )
                except KeyError:
                    pass
                
            else:
                strikes += 1
                if strikes >= config["patience"]:
                    break
        writer.close()
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
        
        ray.init(_temp_dir=f"{storage_path}/temp/")
        #ray.init()
        
        if storage_path is not None:
            runcfg = RunConfig(storage_path=storage_path)
            runcfg.local_dir = f"{storage_path}/local/"
            tuner_config = {"run_config": runcfg}
        else:
            storage_path = os.path.expanduser("~/ray_results")
            tuner_config = {}

        exptime = str(datetime.now())
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
        ray.shutdown()
    
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
    
    @classmethod  
    def _build_report(self, expdir: str, report_file: str, config_prefix: str = "") -> pd.DataFrame:
        """Builds a report of the hyperopt experiment.

        Args:
            expdir (str): The expdir parameter is the path to the experiment directory (ray results folder).
            report_file (str): The report_file parameter is the path to the report file.
            config_prefix: The config_prefix parameter is the prefix for the config items.
        """
        report = None
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


def img_sample(
    model, 
    name = "sample", 
    step=0, 
    reshape: Optional[Iterable[int]] = None, 
    n=2, 
    writer: SummaryWriter = None, 
    device="cpu"
    ):
    """Sample images from a model.
    
    Args:
        model: model to sample from
        path: path to save the image
        reshape: reshape the image
        n: number of samples
        device: device to sample from

    """
    if writer is None:
        writer = SummaryWriter("./")
    with torch.no_grad():
        sample = torch.clip(model.sample(torch.tensor([n, n])), 0, 1) * 255
        
    if reshape is not None:
        reshape = [n, n] + reshape
        sample = sample.reshape(reshape)
    
    sample = torch.cat([x for x in sample], dim=-1)
    sample = torch.cat([x for x in sample], dim=0)
    
    writer.add_image(name, sample, global_step=step, dataformats="HW")
    writer.flush()
    writer.close()

def scatter_sample(
    model, 
    name = "sample", 
    step=0, 
    n=1000, 
    writer: SummaryWriter = None,
    device="cpu"
    ):
    """Sample images from a model.
    
    Args:
        model: model to sample from
        path: path to save the image
        n: number of samples
        device: device to sample from

    """
    if writer is None:
        writer = SummaryWriter("./")
    with torch.no_grad():
        sample = model.sample(torch.tensor([n]))
        
    fig = px.scatter(x=sample[:, 0], y=sample[:, 1])
    fig_bytes = fig.to_image(format="png")

    # Load image into PIL
    image = Image.open(io.BytesIO(fig_bytes))

    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Convert NumPy array to PyTorch Tensor
    tensor = torch.from_numpy(image_np)
    
    writer.add_image(name, tensor, global_step=step, dataformats="HWC")
    writer.flush()
    writer.close()

def density_contour_sample(
    model, 
    name = "sample", 
    step=0, 
    n=1000, 
    device="cpu",
    writer: SummaryWriter = None
    
    ):
    """Generate a density contour plot from samples of a model and save it as an image.

    Args:
        model: model to sample from.
        path: path to save the image.
        n: number of samples.
        device: device to generate samples from.

    """
    if writer is None:
        writer = SummaryWriter("./")
    with torch.no_grad():
        # Sample from the model
        sample = model.sample(torch.tensor([n])).to(device).numpy()

    # Create a density contour plot
    fig = px.density_contour(x=sample[:, 0], y=sample[:, 1])
    fig_bytes = fig.to_image(format="png")
    # Load image into PIL
    image = Image.open(io.BytesIO(fig_bytes))
    # Convert PIL Image to NumPy array
    image_np = np.array(image)
    # Convert NumPy array to PyTorch Tensor
    tensor = torch.from_numpy(image_np)
    
    writer.add_image(name, tensor, global_step=step, dataformats="HWC")
    writer.flush()
    writer.close()