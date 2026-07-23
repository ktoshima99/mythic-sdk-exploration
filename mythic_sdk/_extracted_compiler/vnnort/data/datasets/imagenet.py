import numpy as np
from torchvision.datasets import ImageNet

from vnnort import get_env_variable
from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import ImageClassificationInput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase

IMAGENET_DATASET_PATH = get_env_variable("VNNORT_IMAGENET_DATASET_PATH")


class ImagenetDataset(DatasetBase):
    """This class provides access to the validation set of the Imagenet1k dataset."""

    def __init__(self, dataset_path: str = IMAGENET_DATASET_PATH) -> None:
        """Initialize the dataset.

        Internally it utilizes the torchvision dataset implementation and thus requires the same file system structure.
        This means that the dataset must be downloaded manually beforehand.
        See https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html for more details.


        Args:
            dataset_path (str, optional): The path to the dataset. Defaults to vnnort.DEFAULT_IMAGENET_DATASET_PATH.
        """
        # We use the torchvision dataset accessor to parse and load the data
        self.dataset = ImageNet(root=dataset_path, split="val", transform=None)

        # We try to evenly sample from all classes
        # all classes have 50 samples
        # index [0, 49] is class 0
        # index [50, 99] is class 1
        indices = []
        for image_index in range(50):
            for class_index in range(1000):
                indices.append(class_index * 50 + image_index)
        self.indices = np.array(indices)

    def __getitem__(self, index: int) -> ImageClassificationInput:
        """Access a data sample from the dataset.

        Args:
            index (int): needs to be >= 0 and < len(self)

        Returns:
            ImageClassificationInput: The data sample at the given index.
        """
        image, label = self.dataset[self.indices[index]]

        data = ImageClassificationInput(image=image, label=label)
        return data

    def __len__(self) -> int:
        """Length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.indices)

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Return a class which can be used to benchmark a model on this dataset.

        Returns:
            type[BenchmarkBase]: A class which can be used to benchmark a model on this dataset
        """
        from vnnort.inference.evaluation.image_classification import ImageClassificationBenchmark

        return ImageClassificationBenchmark
