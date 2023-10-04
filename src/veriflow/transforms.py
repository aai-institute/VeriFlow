import math
from abc import abstractmethod
from typing import List

import numpy as np
import pyro
import torch
from pyro import distributions as dist
from pyro.distributions import constraints
from pyro.distributions.transforms import (
    AffineCoupling,
    LowerCholeskyAffine,
    Permute,
)
from pyro.infer import SVI
from pyro.nn import DenseNN
from sklearn.datasets import load_digits
from torch import linalg as LA
from torch.distributions.transforms import Transform
from torch.distributions.utils import lazy_property
from torch.functional import F
from torch.nn import init
from tqdm import tqdm

from src.veriflow.linalg import solve_triangular


class BaseTransform(dist.TransformModule):
    """Base class for transforms. Implemented as a thin layer on top of pyro's TransformModule. The baseTransform
    provides additional methods for checking and constraints on the parameters of the transform.
    """

    def __init__(self, *args, **kwargs):
        super(dist.TransformModule, self).__init__(*args, **kwargs)

    @abstractmethod
    def is_feasible(self) -> bool:
        """Checks if the layer is feasible."""
        return True

    @abstractmethod
    def jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to the layer. This is useful to ensure that the transformation is invertible."""
        pass


class ScaleTransform(BaseTransform):
    """Implementation of a bijective scale transform. Applies a transform $y = \mathrm{diag}(\mathbf{scale})x$, where scale is a learnable parameter of dimension $\mathbf{dim}$

    *Note:* The implementation does not enforce the non-zero constraint of the diagonal elements of $\mathbf{U}$ during training.
    See :func:`add_jitter` and :func:`is_feasible` for a way to ensure that the transformation is invertible.
    """

    def __init__(self, dim: torch.Tensor, *args, **kwargs):
        """Initializes the scale transform."""
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.scale = torch.nn.Parameter(torch.empty(dim))
        self.init_params()

        self.bijective = True
        self.domain = dist.constraints.real_vector
        self.codomain = dist.constraints.real_vector

    def init_params(self):
        """initialization of the parameters"""
        dim = self.dim
        bound = 1 / math.sqrt(dim) if dim > 0 else 0
        init.uniform_(self.scale, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the affine transform $\mathbf{scale}x$

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: transformed tensor $\mathbf{scale}x$
        """
        return x * self.scale

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the inverse transform $\mathbf{scale}^{-1}x$


        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: transformed tensor $\mathbf{scale}^{-1}x$
        """
        return x / self.scale

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :func:`backward`"""
        return self.backward(x)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Computes the log absolute determinant of the Jacobian of the transform $\mathbf{scale}x$.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            float: log absolute determinant of the Jacobian of the transform $\mathbf{scale}x$
        """
        return self.scale.abs().log().sum()

    def sign(self) -> int:
        """Computes the sign of the determinant of the Jacobian of the transform $\mathbf{scale}x$."""
        return 1 if (self.scale < 0).int().sum() % 2 == 0 else -1

    def is_feasible(self) -> bool:
        """Checks if the layer is feasible, i.e. if the diagonal elements of $\mathbf{U}$ are all positive"""
        return (self.scale != 0).all()

    def add_jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to the diagonal elements of $\mathbf{U}$."""
        perturbation = torch.randn(self.dim, device=self.U_raw.device) * jitter
        self.U_raw = self.scale + perturbation


class Permute(BaseTransform):
    """Permutation transform."""

    bijective = True
    volume_preserving = True

    def __init__(
        self, permutation: torch.tensor, *, dim: int = -1, cache_size: int = 1
    ) -> None:
        """Initializes the permutation transform.

        Args:
            permutation (torch.Tensor): permutation vector
        """
        super().__init__(cache_size=cache_size)

        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")

        self.permutation = permutation
        self.dim = dim

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        """Returns the domain of the transform."""
        return constraints.independent(constraints.real, -self.dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        """Returns the codomain of the transform."""
        return constraints.independent(constraints.real, -self.dim)

    @lazy_property
    def inv_permutation(self):
        """Returns the inverse permutation."""
        result = torch.empty_like(self.permutation, dtype=torch.long)
        result[self.permutation] = torch.arange(
            self.permutation.size(0),
            dtype=torch.long,
            device=self.permutation.device,
        )
        return result.to(self.permutation.device)

    def _call(self, x: torch.Tensor):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        return x.index_select(self.dim, self.permutation)

    def _inverse(self, y: torch.Tensor):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        return y.index_select(self.dim, self.inv_permutation)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the element-wise determinant of the log Jacobian, i.e.
        log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])). Note that this type of
        transform is not auto-regressive, so the log Jacobian is not the sum of the
        previous expression. However, it turns out it's always 0 (since the
        determinant is -1 or +1), and so returning a vector of zeros works.
        """

        return torch.zeros(
            x.size()[: -self.event_dim],
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )

    def with_cache(self, cache_size: int = 1):
        """Returns a new :class:`Permute` instance with a given cache size."""
        if self._cache_size == cache_size:
            return self
        return Permute(self.permutation, cache_size=cache_size)


class LUTransform(BaseTransform):
    """Implementation of a linear bijection transform. Applies a transform $y = (\mathbf{L}\mathbf{U})^{-1}x$, where $\mathbf{L}$ is a
    lower triangular matrix with unit diagonal and $\mathbf{U}$ is an upper triangular matrix. Bijectivity is guaranteed by
    requiring that the diagonal elements of $\mathbf{U}$ are positive and the diagonal elements of  $\mathbf{L}$ are all $1$.

    *Note:* The implementation does not enforce the non-zero constraint of the diagonal elements of $\mathbf{U}$ during training.
    See :func:`add_jitter` and :func:`is_feasible` for a way to ensure that the transformation is invertible.
    """

    bijective = True
    volume_preserving = False
    domain = dist.constraints.real_vector
    codomain = dist.constraints.real_vector

    def __init__(self, dim: int, *args, **kwargs):
        """Initializes the LU transform.

        Args:
            dim (int): dimension of the input and output
        """
        super().__init__(*args, **kwargs)
        self.L_raw = torch.nn.Parameter(torch.empty(dim, dim))
        self.U_raw = torch.nn.Parameter(torch.empty(dim, dim))
        self.bias = torch.nn.Parameter(torch.empty(dim))
        self.dim = dim

        self.init_params()

        self.input_shape = dim

        self.L_mask = torch.tril(torch.ones(dim, dim), diagonal=-1)
        self.U_mask = torch.triu(torch.ones(dim, dim), diagonal=0)

        self.L_raw.register_hook(lambda grad: grad * self.L_mask)
        self.U_raw.register_hook(lambda grad: grad * self.U_mask)

    def init_params(self):
        """Parameter initialization
        Adopted from pytorch's Linear layer parameter initialization.
        """

        init.kaiming_uniform_(self.L_raw, nonlinearity="relu")
        with torch.no_grad():
            self.L_raw.copy_(self.L_raw.tril(diagonal=-1).fill_diagonal_(1))

        init.kaiming_uniform_(self.U_raw, nonlinearity="relu")
        with torch.no_grad():
            self.U_raw.copy_(self.U_raw.triu())

        if self.bias is not None:
            fan_in = self.dim
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the affine transform $y = (LU)^{-1}x + \mathrm{bias}$.
        The value $y$ is computed by solving the linear equation system
        \begin{align*}
            Ly_0 &= x + LU\textrm{bias} \\
            Uy &= y_0
        \end{align*}

        :param x: input tensor
        :type x: torch.Tensor
        :return: transformed tensor $(LU)x + \mathrm{bias}$
        """
        x0 = x + torch.functional.F.linear(self.bias, self.inv_weight)
        y0 = solve_triangular(self.L, x0)
        return solve_triangular(self.U, y0)

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """Computes the inverse transform $(LU)(y - \mathrm{bias})$

        :param y: input tensor
        :type y: torch.Tensor
        :return: transformed tensor $(LU)^{-1}(y - \mathrm{bias})$"""
        return torch.functional.F.linear(y - self.bias, self.inv_weight)

    @property
    def L(self) -> torch.Tensor:
        """The lower triangular matrix $\mathbf{L}$ of the layers LU decomposition"""
        return self.L_raw.tril(-1) + torch.eye(self.dim)

    @property
    def U(self) -> torch.Tensor:
        """The upper triangular matrix $\mathbf{U}$ of the layers LU decomposition"""
        return self.U_raw.triu()

    @property
    def inv_weight(self) -> torch.Tensor:
        """Inverse weight matrix of the affine transform"""
        return LA.matmul(self.L, self.U)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Alias for :func:`backward`"""
        return self.backward(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Computes the log absolute determinant of the Jacobian of the transform $(LU)x + \mathrm{bias}$.

        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): transformed tensor

        Returns:
            float: log absolute determinant of the Jacobian of the transform $(LU)x + \mathrm{bias}$
        """
        # log |Det(LU)| =  sum(log(|diag(U)|))
        # (as L is lower triangular with all 1s on the diag, i.e. log|Det(L)| = 0, and U is upper triangular)
        # However, since onnx export of diag() is currently not supported, we have use
        # a reformulation. Note dU keeps the quadratic structure but replace all values
        # outside the diagonal with 1. Then sum(log(|diag(U)|)) = sum(log(|dU|))
        U = self.U
        dU = U - U.triu(1) + (torch.ones_like(U) - torch.eye(self.dim))
        return dU.abs().log().sum()

    def sign(self) -> int:
        """Computes the sign of the determinant of the Jacobian of the transform $(LU)x + \mathrm{bias}$.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            float: sign of the determinant of the Jacobian of the transform $(LU)x + \mathrm{bias}$
        """
        return self.L.diag().prod().sign() * self.U.diag().prod().sign()

    def to(self, device) -> None:
        """Moves the layer to a given device

        Args:
            device (torch.device): target device
        """
        self.L_raw = self.L_raw.to(device)
        self.U_raw = self.U_raw.to(device)
        self.bias = self.bias.to(device)
        return super().to(device)

    def is_feasible(self) -> bool:
        """Checks if the layer is feasible, i.e. if the diagonal elements of $\mathbf{U}$ are all positive"""
        return (self.U_raw.diag() != 0).all()

    def add_jitter(self, jitter: float = 1e-6) -> None:
        """Adds jitter to the diagonal elements of $\mathbf{U}$. This is useful to ensure that the transformation
        is invertible.

        Args:
            jitter (float, optional): jitter strength. Defaults to 1e-6.
        """
        perturbation = torch.randn(self.dim, device=self.U_raw.device) * jitter
        with torch.no_grad():
            self.U_raw.copy_(
                self.U_raw
                + perturbation * torch.eye(self.dim, device=self.U_raw.device)
            )


class MaskedCoupling(BaseTransform):
    """Implementation of a masked coupling layer. The layer is defined by a mask that specifies which dimensions are passed through unchanged and which are transformed.
    The layer is defined by a bijective function $y = \mathrm{mask} \odot x + (1 - \mathrm{mask}) \odot (x + \mathrm{transform}(x))$, where $\mathrm{mask}$ is a binary mask,
    $\mathrm{transform}$ is a bijective function, and $\odot$ denotes element-wise multiplication.
    """

    def __init__(
        self, mask: torch.Tensor, conditioner: torch.nn.Module, *args, **kwargs
    ) -> None:
        """Initializes the masked coupling layer.

        Args:
            mask (torch.Tensor): binary mask
            conditioner (torch.nn.Module): bijective function $\mathrm{transform}$
        """
        super().__init__(*args, **kwargs)
        self.mask = mask
        self.conditioner = conditioner
        self.input_shape = mask.shape
        self.bijective = True
        self.domain = dist.constraints.real_vector
        self.codomain = dist.constraints.real_vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the affine transform
        $\mathrm{mask} \odot x + (1 - \mathrm{mask}) \odot (x + \mathrm{transform}(x))$

        Args:
            x (torch.Tensor): input tensor
        """
        x_masked = x * self.mask
        x_transformed = x + (1 - self.mask) * self.conditioner(x_masked)
        return x_transformed

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """Computes the inverse transform

        Args:
            y (torch.Tensor): input tensor

        Returns:
            torch.Tensor: transformed tensor
        """
        y_masked = y * self.mask
        y_transformed = y - (1 - self.mask) * self.conditioner(y_masked)
        return y_transformed

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Alias for :func:`backward`"""
        return self.backward(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Computes the log absolute determinant of the Jacobian of the transform

        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor

        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        x_masked = x * self.mask
        return 0.0

    def sign(self) -> int:
        """Computes the sign of the determinant of the Jacobian of the transform

        Args:
            x (torch.Tensor): input tensor

        Returns:
            int: sign of the determinant of the Jacobian of the transform
        """
        return 1.0

    def to(self, device):
        """Moves the layer to a given device

        Args:
            device (torch.device): target device
        """
        self.mask = self.mask.to(device)
        return super().to(device)


class LeakyReLUTransform(BaseTransform):
    bijective = True
    domain = dist.constraints.real
    codomain = dist.constraints.real
    sign = 1

    def __init__(self, alpha: float = 0.01, *args, **kwargs) -> None:
        """Initializes the LeakyReLU transform.

        Args:
            alpha (float, optional): slope of the negative part of the function. Defaults to 0.01.
        """
        if alpha == 0:
            raise ValueError("alpha must be positive")
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the LeakyReLU transform

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: transformed tensor
        """
        return F.leaky_relu(x, negative_slope=self.alpha)

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """Computes the inverse transform

        Args:
            y (torch.Tensor): input tensor

        Returns:
            torch.Tensor: transformed tensor
        """
        return F.leaky_relu(y, negative_slope=1 / self.alpha)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :func:`forward`"""
        return self.forward(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Alias for :func:`backward`"""
        return self.backward(y)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Computes the log absolute determinant of the Jacobian of the transform

        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): output tensor

        Returns:
            float: log absolute determinant of the Jacobian of the transform
        """
        return torch.log(y / x).sum()
