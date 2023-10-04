import os
import typing as T
from abc import abstractmethod

import idx2numpy
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.datasets import (
    make_blobs,
    make_checkerboard,
    make_circles,
    make_moons,
)
from torch import Tensor
from torchvision.datasets import MNIST, FashionMNIST


class DequantizedDataset(torch.utils.data.Dataset):
    """
    A dataset that dequantizes the data by adding uniform noise to each pixel.
    """

    def __init__(
        self,
        dataset: T.Union[os.PathLike, torch.utils.data.Dataset, np.ndarray],
        num_bits: int = 8,
    ):
        if isinstance(dataset, torch.utils.data.Dataset) or isinstance(
            dataset, np.ndarray
        ):
            self.dataset = dataset
        else:
            self.dataset = pd.read_csv(dataset).values

        self.num_bits = num_bits
        self.num_levels = 2**num_bits
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / self.num_levels),
                transforms.Lambda(
                    lambda x: x + torch.rand_like(x) / self.num_levels
                ),
            ]
        )

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        x = Tensor(self.transform(x))
        return x, y

    def __len__(self):
        return len(self.dataset)


class DataSplit:
    def __init__(*agrs, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_train(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_test(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_val(self) -> torch.utils.data.Dataset:
        raise NotImplementedError


class SimpleSplit(DataSplit):
    """
    Split of dataset
    """

    def __init__(
        self,
        train: torch.utils.data.Dataset,
        test: torch.utils.data.Dataset,
        val: torch.utils.data.Dataset,
    ):
        """Create split of dataset

        Args:
            train (torch.utils.data.Dataset): training set
            test (torch.utils.data.Dataset): test set
            val (torch.utils.data.Dataset): validation set
        """
        self.train = train
        self.test = test
        self.val = val

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val


GENERATORS = {
    "make_moons": make_moons,
    "make_blobs": make_blobs,
    "make_checkerboard": make_checkerboard,
    "make_circles": make_circles,
}


class SyntheticDataset(torch.utils.data.Dataset):
    """
    Dataset from generator function
    """

    def __init__(
        self,
        generator: T.Union[T.Callable[..., np.ndarray], str],
        params: T.Dict[str, T.Any],
        *args,
        **kwargs
    ):
        """Create dataset from generator function

        Args:
            generator (function): generator function
            params: ]dict]: parameters for generator function
        """
        super().__init__(*args, **kwargs)
        if isinstance(generator, str):
            generator = GENERATORS[generator]

        self.dataset = generator(**params)[0]

    def __getitem__(self, index: int):
        x = self.dataset[index]
        x = Tensor(x)
        return x, torch.zeros_like(x)

    def __len__(self):
        return len(self.dataset)


class SyntheticSplit(SimpleSplit):
    """
    Split of synthetic dataset
    """

    def __init__(
        self,
        generator: T.Union[T.Callable[..., np.ndarray], str],
        params_train: T.Dict[str, T.Any],
        params_test: T.Dict[str, T.Any],
        params_val: T.Dict[str, T.Any],
        *args,
        **kwargs
    ):
        """Create dataset from generator function

        Args:
            generator (function): generator function
            params: ]dict]: parameters for generator function
        """
        if isinstance(generator, str):
            generator = GENERATORS[generator]

        train = SyntheticDataset(generator, params_train)
        test = SyntheticDataset(generator, params_test)
        val = SyntheticDataset(generator, params_val)
        super().__init__(train=train, test=test, val=val, *args, **kwargs)


class FlattenedDataset(torch.utils.data.Dataset):
    """
    A dataset that flattens the data.
    """

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        x = x.flatten()
        return x, y

    def __len__(self):
        return len(self.dataset)


class FashionMnistDequantized(DequantizedDataset):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        train: bool = True,
        label: T.Optional[int] = None,
    ):
        rel_path = (
            "FashionMNIST/raw/train-images-idx3-ubyte"
            if train
            else "FashionMNIST/raw/t10k-images-idx3-ubyte"
        )
        path = os.path.join(dataloc, rel_path)
        if not os.path.exists(path):
            FashionMNIST(path, train=train, download=True)
        # TODO: remove hardcoding of 3x3 downsampling, vectorizing
        dataset = idx2numpy.convert_from_file(path)[:, ::3, ::3]
        dataset = dataset.reshape(dataset.shape[0], -1)
        if label is not None:
            rel_path = (
                "FashionMNIST/raw/train-labels-idx1-ubyte"
                if train
                else "FashionMNIST/raw/t10k-labels-idx1-ubyte"
            )
            path = os.path.join(dataloc, rel_path)
            labels = idx2numpy.convert_from_file(path)
            dataset = dataset[labels == label]
        super().__init__(dataset, num_bits=8)

    def __getitem__(self, index: int):
        x = Tensor(self.dataset[index].copy())
        x = self.transform(x)
        return x, 0


class MnistDequantized(DequantizedDataset):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        train: bool = True,
        digit: T.Optional[int] = None,
        flatten=True,
    ):
        if train:
            rel_path = "MNIST/raw/train-images-idx3-ubyte"
        else:
            rel_path = "MNIST/raw/t10k-images-idx3-ubyte"
        path = os.path.join(dataloc, rel_path)
        if not os.path.exists(path):
            MNIST(dataloc, train=train, download=True)

        # TODO: remove hardcoding of 3x3 downsampling
        dataset = idx2numpy.convert_from_file(path)[:, ::3, ::3]
        if flatten:
            dataset = dataset.reshape(dataset.shape[0], -1)
        if digit is not None:
            if train:
                rel_path = "MNIST/raw/train-labels-idx1-ubyte"
            else:
                rel_path = "MNIST/raw/t10k-labels-idx1-ubyte"
            path = os.path.join(dataloc, rel_path)
            labels = idx2numpy.convert_from_file(path)
            dataset = dataset[labels == digit]
        super().__init__(dataset, num_bits=8)

    def __getitem__(self, index: int):
        x = Tensor(self.dataset[index].copy())
        x = self.transform(x)
        return x, 0


class DataSplitFromCSV(DataSplit):
    def __init__(
        self, train: os.PathLike, test: os.PathLike, val: os.PathLike
    ):
        self.train = train
        self.test = test
        self.val = val

    def get_train(self) -> torch.utils.data.Dataset:
        return pd.read_csv(self.train).values

    def get_test(self) -> torch.utils.data.Dataset:
        return pd.read_csv(self.test).values

    def get_val(self) -> torch.utils.data.Dataset:
        return pd.read_csv(self.val).values


class FashionMnistSplit(DataSplit):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        val_split: float = 0.1,
        label: T.Optional[int] = None,
    ):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        self.dataloc = dataloc
        self.train = FashionMnistDequantized(
            self.dataloc, train=True, label=label
        )
        shuffle = torch.randperm(len(self.train))
        self.val = torch.utils.data.Subset(
            self.train, shuffle[: int(len(self.train) * val_split)]
        )
        self.train = torch.utils.data.Subset(
            self.train, shuffle[int(len(self.train) * val_split) :]
        )
        self.test = FashionMnistDequantized(
            self.dataloc, train=False, label=label
        )

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val


class MnistSplit(DataSplit):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        val_split: float = 0.1,
        digit: T.Optional[int] = None,
    ):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        self.dataloc = dataloc
        self.train = MnistDequantized(self.dataloc, train=True, digit=digit)
        shuffle = torch.randperm(len(self.train))
        self.val = torch.utils.data.Subset(
            self.train, shuffle[: int(len(self.train) * val_split)]
        )
        self.train = torch.utils.data.Subset(
            self.train, shuffle[int(len(self.train) * val_split) :]
        )
        self.test = MnistDequantized(self.dataloc, train=False, digit=digit)

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val
