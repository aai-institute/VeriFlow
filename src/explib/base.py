import os
import typing as T


class Experiment(object):
    """Base class for experiments."""

    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self.name = name

    @classmethod
    def _init_rec(cls, cfg):
        if isinstance(cfg, dict):
            if "experiment" in cfg:
                experiment_type = cfg["experiment"]["experiment_type"]
                params = cls._init_rec(cfg["experiment"]["experiment_params"])

                return experiment_type(**params)
            else:
                return {k: cls._init_rec(v) for k, v in cfg.items()}
        elif isinstance(cfg, list):
            return [cls._init_rec(v) for v in cfg]
        else:
            return cfg

    @classmethod
    def from_dict(cls, config: T.Dict[str, T.Any]) -> "Experiment":
        if "experiment" not in config:
            raise ValueError(
                "Invalid config file. The config file needs to contain an experiment field."
            )
        return cls._init_rec(config)

    def conduct(
        self, report_dir: os.PathLike, storage_path: os.PathLike = None
    ) -> None:
        """Conducts the experiment and saves the results to the report directory. The method is expected to store all results in report_dir."""
        raise NotImplementedError


class ExperimentCollection(Experiment):
    """Implements an experiment that consists of several jointly conducted but independent experiments."""

    def __init__(
        self, experiments: T.Iterable[Experiment], *args, **kwargs
    ) -> None:
        """
        The function initializes an object with a list of experiments based on a given configuration.

        :param experiments: The "experiments" parameter is an iterable object that contains a list of
        experiments. Each experiment is represented by a configuration object
        :type experiments: Iterable *args
        """
        super().__init__(*args, **kwargs)
        self.experiments = experiments

    @classmethod
    def from_dict(cls, config: T.Dict[str, T.Any]) -> "ExperimentCollection":
        config = deepcopy(config)
        for i, exp_cfg in enumerate(
            config["experiment_params"]["experiments"]
        ):
            config["experiment_params"]["experiments"][
                i
            ] = Experiment.from_dict(exp_cfg)

        return Experiment.from_dict(config)

    def conduct(
        self, report_dir: os.PathLike, storage_path: os.PathLike = None
    ):
        for i, exp in enumerate(self.experiments):
            exp.conduct(
                os.path.join(report_dir, f"{i}_{exp.name}"),
                storage_path=storage_path,
            )
