from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sys
from typing import Generator

import numpy as np
from torchvision.datasets import CocoDetection

from vnnort import get_env_variable
from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import ImageDetectionInput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase

COCO_DATASET_PATH = get_env_variable("VNNORT_COCO_DATASET_PATH")


class CocoClassRemapper:
    """Helper class to map COCO class labels between the original (91 classes) and cleaned up dataset (80 classes).

    The original dataset published in 2014 had plans for 91 classes. However the actual published annotations
    only had 80 of these left. Unfortunately, there are many model variations are either trained on 91 or 80 classes.
    This class helps to match between either of these versions.
    """

    # fmt: off
    _original_coco_classes = [
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    # fmt: on

    def __init__(self) -> None:
        """Initialize the remapper."""
        original_name_to_label = {key: index for index, key in enumerate(self._original_coco_classes)}
        clean_coco_classes = [name for name in self._original_coco_classes if name != "N/A"]

        clean_name_to_label = {key: index for index, key in enumerate(clean_coco_classes)}
        clean_name_to_label["N/A"] = -1  # This is not included in the original

        self._clean_to_original_label = {
            index: original_name_to_label[name] for index, name in enumerate(clean_coco_classes)
        }

        self._original_to_clean_label = {
            index: clean_name_to_label[name] for index, name in enumerate(self._original_coco_classes)
        }
        self.invalid_labels = {index for index, name in enumerate(self._original_coco_classes) if name == "N/A"}

    def clean_to_original_label(self, label: int) -> int:
        """Map from a clean label [0, 79] to the original label [0, 90].

        Args:
            label (int): The clean label in range [0, 79].

        Returns:
            int: Original label in range [0, 90].
        """
        return self._clean_to_original_label[label]

    def original_to_clean_label(self, original_label: int) -> int:
        """Map from a clean label [0, 90] to the original label [0, 79].

        Args:
            original_label (int): The original label in range [0, 90].

        Returns:
            int: Clean label in range [0, 79].

        Raises:
            ValueError: If the original label is not valid for the clean COCO dataset.
        """
        if original_label in self.invalid_labels:
            raise ValueError(f"Label {original_label} is not a valid label for the clean COCO dataset.")
        return self._original_to_clean_label[original_label]

    def clean_label_to_name(self, clean_label: int) -> str:
        """Return the corresponding COCO class name given the clean label in range [0, 79]."""
        original_label = self.clean_to_original_label(clean_label)
        return self._original_coco_classes[original_label]


@contextlib.contextmanager
def suppress_stdout() -> Generator[None, None, None]:
    """Supress the output of any print statement within this context."""
    with open(os.devnull, "w") as fnull:
        original_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = original_stdout


class CocoDataset(DatasetBase):
    """Wrapper class to load samples from the COCO dataset validation dataset.

    This class wraps around the torchvision datasets class for accessing the COCO dataset. By default behaviour
    this class also automatically downloads the dataset to the provided file location.
    """

    def __init__(self, dataset_path: str = COCO_DATASET_PATH):
        """Construct the dataset wrapper for loading the dataset and/or download the dataset to specified location.

        Args:
            dataset_path (str, optional): Path to dataset directory. Defaults to COCO_DATASET_PATH.
        """
        # Wrapper from huggingface

        dataset_path = Path(dataset_path)
        image_path = dataset_path / "val2014"
        annotation_path = dataset_path / "annotations/instances_val2014.json"

        # Inititalize the coco dataset and suppress all print statements therein
        with suppress_stdout():
            self.dataset = CocoDetection(
                root=image_path,
                annFile=annotation_path,
            )

        # The pytorch wrapper class returns labels from the original annotations
        # We need the cleaned version
        self.class_mapper = CocoClassRemapper()

    def __getitem__(self, index: int) -> ImageDetectionInput:
        """Access a data sample from the dataset.

        Args:
            index (int): needs to be >= 0 and < len(self)

        Returns:
            ImageDetectionInput: The data sample at the given index.
        """
        image, annotations = self.dataset[index]

        # Make sure images are RGB (even grayscale images)
        if image.mode != "RGB":
            # Convert the image to RGB mode
            image = image.convert("RGB")

        # Convert boxes from xywh to xyxy
        boxes = np.array([a["bbox"] for a in annotations])
        if len(boxes) > 0:
            boxes = np.stack(
                [boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]],
                axis=1,
                dtype=np.float32,
            )
        else:
            boxes = np.zeros([0, 4], dtype=np.float32)
        labels = np.array([self.class_mapper.original_to_clean_label(a["category_id"]) for a in annotations])

        result = ImageDetectionInput(
            image=image,
            boxes=boxes,
            labels=labels,
        )
        return result

    def __len__(self) -> int:
        """Length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataset)

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Get the benchmark class for this dataset.

        Returns:
            type[BenchmarkBase]: The benchmark class for this dataset
        """
        from vnnort.inference.evaluation.image_detection import ImageDetectionBenchmark

        return ImageDetectionBenchmark
