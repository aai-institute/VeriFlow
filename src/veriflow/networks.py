from math import ceil
from typing import List, Optional, Tuple, Union

import torch
from pyro.nn import DenseNN
from torch import nn


class AdditiveAffineNN(torch.nn.Module):
    """Provides a dense NN that computes loc and log_scale parameter for an affine transform that is purely additive, i.e. the log_scale component
    always returns the 0 vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        nonlinearity: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        if nonlinearity is None:
            nonlinearity = torch.nn.ReLU()

        self.loc_fnc = DenseNN(
            input_dim, hidden_dims, [output_dim], nonlinearity=nonlinearity
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loc = self.loc_fnc(x)
        log_scale = torch.zeros_like(loc)
        return [loc, log_scale]


class LayerNormChannels(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        """
        This module applies layer norm across channels in an image.
        Args:
            c_in: Number of channels of the input
            eps: Small constant to stabilize std
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        y = y * self.gamma + self.beta
        return y


class GatedConv(nn.Module):
    def __init__(self, c_in, c_hidden):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Args:
            c_in: Number of channels of the input
            c_hidden: Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * c_in, c_hidden, kernel_size=3, padding=1),
            nn.Conv2d(2 * c_hidden, 2 * c_in, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards method

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: network output.
        """
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class ConvNet2D(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int = 3,
        rescale_hidden: int = 2,
        c_out: int = -1,
        num_layers: int = 3,
        nonlinearity: any = nn.ReLU(),
        kernel_size: int = 3,
        padding: int = None,
    ):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Args:
            c_in: Number of input channels
            c_hidden: Number of hidden dimensions to use within the network
            rescale_hidden: Factor by which to rescale hight and width the hidden before and after the hidden layers.
            c_out: Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers: Number of gated ResNet blocks to apply
        """
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.nonlinearity = nonlinearity
        c_out = c_out if c_out > 0 else c_in
        layers = []
        layers += [
            nn.Conv2d(
                c_in, c_hidden, kernel_size=kernel_size, padding=padding
            ),
        ]
        if rescale_hidden != 1:
            layers += [nn.MaxPool2d(rescale_hidden)]

        for layer_index in range(num_layers):
            layers += [
                nn.Conv2d(
                    c_hidden,
                    c_hidden,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nonlinearity,
                LayerNormChannels(c_hidden),
            ]

        # compute padding and output padding for rescaling via transposed convolutions
        if rescale_hidden != 1:
            diff = rescale_hidden - kernel_size
            if diff < 0:
                outpad = diff % 2
                pad = ceil(abs(diff) / 2.0)
            else:
                outpad = diff
                pad = 0

            layers += [
                nn.ConvTranspose2d(
                    c_hidden,
                    c_hidden,
                    kernel_size=kernel_size,
                    stride=rescale_hidden,
                    output_padding=outpad,
                    padding=pad,
                ),
                nonlinearity,
            ]

        layers += [
            nn.Conv2d(
                c_hidden, c_out, kernel_size=kernel_size, padding=padding
            )
        ]
        self.nn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards method

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: network output.
        """
        return self.nn(x)
