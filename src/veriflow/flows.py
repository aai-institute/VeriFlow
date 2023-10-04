from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pyro
import torch
from pyro import distributions as dist
from pyro.distributions.transforms import (AffineCoupling, LowerCholeskyAffine,
                                           SoftplusTransform)
from pyro.infer import SVI
from pyro.nn import DenseNN
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.veriflow.networks import AdditiveAffineNN, ConvNet2D
from src.veriflow.transforms import (BaseTransform, LeakyReLUTransform,
                                     LUTransform, MaskedCoupling, Permute,
                                     ScaleTransform)


class Flow(torch.nn.Module):
    """Base implementation of a flow model"""

    # Export mode determines whether the log_prob or the sample function is exported to onnx
    export_modes = Literal["log_prob", "sample"]
    export = "log_prob"

    def forward(self, x: torch.Tensor):
        """Dummy implementation of forward method for onnx export. The self.export attribute
        determines whether the log_prob or the sample function is exported to onnx
        """
        if self.export == "log_prob":
            return self.log_prob(x)
        elif self.export == "sample":
            return self.sample()
        else:
            raise ValueError(f"Unknown export mode {self.export}")

    def __init__(self, base_distribution, layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = layers
        self.trainable_layers = torch.nn.ModuleList(
            [l for l in layers if isinstance(l, torch.nn.Module)]
        )
        self.base_distribution = base_distribution

        self.transform = dist.TransformedDistribution(
            base_distribution, layers
        )

    def fit(
        self,
        data_train: Dataset,
        optim: torch.optim.Optimizer = torch.optim.Adam,
        optim_params: Dict[str, Any] = None,
        batch_size: int = 32,
        shuffe: bool = True,
        gradient_clip: float = None,
        device: torch.device = None,
        jitter: float = 1e-6,
        epochs: int = 1,
    ) -> float:
        """
        Wrapper function for the fitting procedure. Allows basic configuration of the optimizer and other
        fitting parameters.

        Args:
            data_train: training data.
            batch_size: number of samples per optimization step.
            optim: optimizer class.
            optimizer_params: optimizer parameter dictionary.
            jitter: Determines the amount of jitter that is added if the optimization leaves the feasible region.
            epochs: number of epochs.

        Returns:
            Loss curve (negative log-likelihood).
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        model = self.to(device)

        if optim_params is not None:
            optim = optim(model.trainable_layers.parameters(), **optim_params)
        else:
            optim = optim(model.trainable_layers.parameters())

        N = len(data_train)

        epoch_losses = []
        for _ in range(epochs):
            losses = []
            if shuffe:
                perm = np.random.choice(N, N, replace=False)
                data_train = data_train[perm]

            for idx in range(0, N, batch_size):
                idx_end = min(idx + batch_size, N)
                try:
                    sample = torch.Tensor(data_train[idx:idx_end][0]).to(
                        device
                    )
                except:
                    continue
                optim.zero_grad()
                while not self.is_feasible():
                    self.add_jitter(jitter)
                loss = -model.transform.log_prob(sample).mean()
                losses.append(float(loss.detach()))
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )
                optim.step()
                while not self.is_feasible():
                    self.add_jitter(jitter)

                model.transform.clear_cache()
            epoch_losses.append(np.mean(losses))

        return epoch_losses

    def to_onnx(
        self, path: str, export_mode: export_modes = "log_prob"
    ) -> None:
        """Saves the model as onnx file

        Args:
            path: path to save the model.
            export_mode: export mode. Can be "log_prob" or "sample".
        """
        mode_cache = self.export
        self.export = export_mode
        dummy_input = self.base_distribution.sample()
        torch.onnx.export(self, dummy_input, path, verbose=True)
        self.export = mode_cache

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the models log-densities for the given samples

        Args:
            x: sample tensor.
        """
        return self.transform.log_prob(x)

    def sample(self, sample_shape: Iterable[int] = None) -> torch.Tensor:
        """Returns n_sample samples from the distribution

        Args:
            n_sample: sample shape.
        """
        if sample_shape is None:
            sample_shape = [1]

        return self.transform.sample(sample_shape)

    def to(self, device) -> None:
        """Moves the model to the given device"""
        self.device = device
        # self.layers = torch.nn.ModuleList([l.to(device) for l in self.layers])
        self.trainable_layers = torch.nn.ModuleList(
            [l.to(device) for l in self.trainable_layers]
        )
        return super().to(device)

    def is_feasible(self) -> bool:
        """Checks is the model parameters meet all constraints"""
        return all(
            [
                l.is_feasible()
                for l in self.layers
                if isinstance(l, BaseTransform)
            ]
        )

    def add_jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to meet non-zero constraints"""
        for l in self.layers:
            if isinstance(l, BaseTransform) and not l.is_feasible():
                l.add_jitter(jitter)


class NiceFlow(Flow):
    Permutation = Literal["random", "half", "LU"]

    """Implementation of the NICE flow architecture by using fully connected coupling layers"""

    def __init__(
        self,
        base_distribution: dist.Distribution,
        coupling_layers: int,
        coupling_nn_layers: List[int],
        split_dim: int,
        scale_every_coupling=False,
        nonlinearity: Optional[torch.nn.Module] = None,
        permutation: Permutation = "LU",
        *args,
        **kwargs,
    ) -> None:
        """Initialization

        Args:
            base_distribution: base distribution,
            coupling_layers: number of coupling layers. All coupling layers share the same architecture but not the same weights.
            coupling_nn_layers: number of neurons in the hidden layers of the dense neural network that computes the coupling loc parameter.
            split_dim: split dimension for the coupling, i.e. input dimension of the conditioner.
            scale_every_coupling: if True, a scale transform is applied after every coupling layer. Otherwise, a single scale transform is applied after all coupling layers.
            nonlinearity: nonlinearity of the coupling network.
            permutation: permutation type. Can be "random" or "half".
        """
        input_dim = base_distribution.sample().shape[0]
        self.input_dim = input_dim
        self.coupling_layers = coupling_layers
        self.coupling_nn_layers = coupling_nn_layers
        self.split_dim = split_dim

        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()

        rdim = input_dim - split_dim
        layers = []
        for i in range(coupling_layers):
            layers.append(self._get_permutation(permutation, i))
            layers.append(
                AffineCoupling(
                    split_dim,
                    AdditiveAffineNN(
                        split_dim,
                        coupling_nn_layers,
                        rdim,
                        nonlinearity=nonlinearity,
                    ),
                )
            )

            if scale_every_coupling:
                layers.append(ScaleTransform(input_dim))

        if not scale_every_coupling:
            layers.append(ScaleTransform(input_dim))

        super().__init__(base_distribution, layers, *args, **kwargs)

    def _get_permutation(self, permtype: Permutation, i=0):
        """Returns a permutation layer"""
        if permtype == "random":
            return Permute(torch.randperm(self.input_dim, dtype=torch.long))
        elif permtype == "half":
            if i % 2 == 0:  # every 2nd pixel
                perm = torch.arange(self.input_dim, dtype=torch.long)
                perm = perm.reshape(-1, 2).moveaxis(0, 1).reshape(-1)
            elif (
                i % 2 == 1
            ):  # interchange conditioning variables and output variables
                perm = torch.arange(self.input_dim, dtype=torch.long)
                perm = perm.reshape(2, -1).flip(0).reshape(-1)
            else:  # random permutation
                perm = torch.randperm(self.input_dim, dtype=torch.long)
            return Permute(perm)
        elif permtype == "LU":
            return LUTransform(self.input_dim)
        else:
            raise ValueError(f"Unknown permutation type {permtype}")


class NiceMaskedConvFlow(Flow):
    """Implementation of the NICE flow architecture using fully connected coupling layers
    and a checkerboard permutation"""

    def __init__(
        self,
        base_distribution: dist.Distribution,
        coupling_layers: int,
        conv_layers: int,
        kernel_size: int,
        nonlinearity: Optional[torch.nn.Module] = None,
        c_hidden: int = 32,
        rescale_hidden: Union[int, Tuple[int]] = 4,
        *args,
        **kwargs,
    ) -> None:
        """Initialization

        Args:
            base_distribution: base distribution,
            coupling_layers: number of coupling layers. All coupling layers share the same architecture but not the same weights.
            coupling_nn_layers: number of hidden convolutional layers of the network that computes the coupling loc parameter.
            kernel_size: kernel size of the convolutional layers.
            nonlinearity: nonlinearity of the convolutional layers.
            c_hidden: number of hidden channels of the convolutional layers.
            rescale_hidden: rescaling of hight and width for the hidden layers.
        """

        self.coupling_layers = coupling_layers

        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()

        c, h, w = base_distribution.sample().shape
        mask = NiceMaskedConvFlow.create_checkerboard_mask(h, w)

        layers = []
        self.masks = []
        for i in range(coupling_layers):
            layers.append(
                MaskedCoupling(
                    mask,
                    ConvNet2D(
                        mask.shape[0],
                        num_layers=conv_layers,
                        nonlinearity=nonlinearity,
                        kernel_size=kernel_size,
                        c_hidden=c_hidden,
                        rescale_hidden=rescale_hidden,
                    ),
                )
            )
            self.masks.append(mask)
            mask = 1 - mask

        layers.append(ScaleTransform(mask.shape))

        super().__init__(base_distribution, layers, *args, **kwargs)

    @classmethod
    def create_checkerboard_mask(
        cls, h: int, w: int, invert: bool = False
    ) -> torch.Tensor:
        """Creates a checkerboard mask of size $(h,w)$.

        Args:
            h (_type_): height
            w (_type_): width
            invert (bool, optional): If True, inverts the mask. Defaults to False.
        Returns:
            Checkerboard mask of height $h$ and width $w$.
        """
        x, y = torch.arange(h, dtype=torch.int32), torch.arange(
            w, dtype=torch.int32
        )
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        mask = torch.fmod(xx + yy, 2)
        mask = mask.to(torch.float32).view(1, 1, h, w)
        if invert:
            mask = 1 - mask
        return mask


class LUFlow(Flow):
    """Implementation of the NICE flow architecture using fully connected coupling layers
    and a checkerboard permutation
    """

    def __init__(
        self,
        base_distribution: dist.Distribution,
        n_layers: int,
        nonlinearity: Optional[torch.distributions.Transform] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialization

        Args:
            base_distribution: base distribution,
            n_layers: number of LU-layers.
            nonlinearity: nonlinearity of the convolutional layers.
        """

        self.n_layers = n_layers

        if nonlinearity is None:
            nonlinearity = LeakyReLUTransform

        layers = []

        for i in range(n_layers):
            layers.append(
                LUTransform(
                    base_distribution.sample().shape[0],
                )
            )
            layers.append(nonlinearity())

        super().__init__(base_distribution, layers, *args, **kwargs)

    def is_feasible(self):
        """Checks if all LU layers are feasible"""
        return all(
            [
                l.is_feasible()
                for l in self.layers
                if isinstance(l, LUTransform)
            ]
        )

    def add_jitter(self, jitter: float = 1e-6) -> None:
        for layer in self.layers:
            if isinstance(layer, LUTransform) and not layer.is_feasible():
                layer.add_jitter(jitter)
