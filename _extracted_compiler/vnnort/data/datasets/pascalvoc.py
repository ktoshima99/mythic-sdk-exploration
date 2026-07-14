import os

import numpy as np
from torchvision.datasets import VOCSegmentation

from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import ImageSegmentationInput
from vnnort.data.container import InputData
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase

DATASET_SPLIT = "val"
VNNORT_PASCALVOC_PATH = os.getenv("VNNORT_PASCALVOC_PATH")


class PascalVOCDataset(DatasetBase):
    """Dataset for the Pascal VOC segmentation task."""

    def __init__(self) -> None:
        """Initialize the dataset."""
        super().__init__()
        # Use the Pascal VOC segmentation dataset implementation from torchvision
        self.ds = VOCSegmentation(root=VNNORT_PASCALVOC_PATH, image_set=DATASET_SPLIT, download=False)

    def __getitem__(self, idx: int) -> InputData:
        """Access a data sample from the dataset.

        Args:
            idx (int): needs to be >= 0 and < len(self)

        Returns:
            InputData: Input data to the model
        """
        image, segmentation_map = self.ds[idx]
        result = ImageSegmentationInput(image, np.array(segmentation_map, dtype=np.uint32))
        return result

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.ds)

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Return a subclass of BenchmarkBase which can be used to benchmark a model on this dataset.

        Returns:
            type[BenchmarkBase]: A subclass of BenchmarkBase
        """
        from vnnort.inference.evaluation.image_segmentation import ImageSegmentationBenchmark

        return ImageSegmentationBenchmark
