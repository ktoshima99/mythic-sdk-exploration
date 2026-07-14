from abc import ABC
from abc import abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from vnnort.data.container import InputData
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase


class DatasetBase(ABC):
    """Base class for all dataset accessors, which defines a common interface all classes need to implement.

    All datasets implementing this interface can be used like this:
    Example:
        dataset = Dataset()
        length = len(dataset)  # Number of entries available
        entry = dataset[0]
    """

    @abstractmethod
    def __len__(self) -> int:
        """Length of the dataset.

        Returns:
            int: The number of samples in the dataset.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> InputData:
        """Access a data sample from the dataset.

        Args:
            index (int): needs to be >= 0 and < len(self)

        Returns:
            InputData: The data sample at the given index.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_benchmark(self) -> type[BenchmarkBase]:
        """Return a subclass of BenchmarkBase which can be used to benchmark a model on this dataset.

        Returns:
            type[BenchmarkBase]: A subclass of BenchmarkBase

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    def subset(self, indices: Sequence[int] | NDArray[np.intp]) -> "SubDataset":
        """Return a SubDataset with the given indices."""
        return SubDataset(self, indices)


class SubDataset(DatasetBase):
    """A view into a DatasetBase restricted to a subset of indices."""

    def __init__(self, dataset: DatasetBase, indices: Sequence[int] | NDArray[np.intp]):
        """Initialize with a parent dataset and the indices to expose."""
        self.parent_dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return len(self.indices)

    def __getitem__(self, index: int) -> InputData:
        """Return the sample at the given subset integer index."""
        return self.parent_dataset[int(self.indices[index])]

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Return the benchmark class from the parent dataset."""
        return self.parent_dataset.get_benchmark()
