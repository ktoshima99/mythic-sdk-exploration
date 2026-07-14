import json
import os
from typing import Any

import PIL.Image
import numpy as np

from vnnort import get_env_variable
from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import ImageDetectionInput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase

VNNORT_NUIMAGES_PATH = get_env_variable("VNNORT_NUIMAGES_PATH")

categories_to_label = {
    class_name: index
    for index, class_name in enumerate(
        [
            "car",
            "truck",
            "trailer",
            "bus",
            "construction_vehicle",
            "bicycle",
            "motorcycle",
            "pedestrian",
            "trafficcone",
            "barrier",
        ]
    )
}


class NuimagesDataset(DatasetBase):
    """Nuimages dataset for ImageDetection."""

    def __init__(self, path_to_dataset: str = VNNORT_NUIMAGES_PATH, split: str = "val") -> None:
        """Initialize the nuimages dataset."""
        if split == "mini":
            annotation_file = "nuimages_v1.0-mini.json"
        elif split == "val":
            annotation_file = "nuimages_v1.0-val.json"
        else:
            raise ValueError("Split must be val or mini")

        annotations_path = os.path.join(path_to_dataset, "nuimages_coco/annotations", annotation_file)
        with open(annotations_path, mode="r") as f:
            annotations_file = json.load(f)
        image_names = [entry["file_name"] for entry in annotations_file["images"]]
        annotations: dict[int, Any] = {index: {"labels": [], "bboxes": []} for index in range(len(image_names))}
        for entry in annotations_file["annotations"]:
            annotations[entry["image_id"]]["labels"].append(entry["category_id"])
            annotations[entry["image_id"]]["bboxes"].append(entry["bbox"])

        self.image_names = image_names
        self.annotations = annotations
        self.path_to_dataset = path_to_dataset

    def __getitem__(self, index: int) -> ImageDetectionInput:
        """Access a data sample from the dataset."""
        image_name = self.image_names[index]
        image_path = os.path.join(self.path_to_dataset, "nuimages", image_name)
        image = PIL.Image.open(image_path)
        annotation = self.annotations[index]

        # Convert boxes from xywh to xyxy
        boxes = np.array(annotation["bboxes"])
        if len(boxes) > 0:
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            labels = np.array(annotation["labels"])
        else:
            boxes = np.empty([0, 4], dtype=np.int32)
            labels = np.empty([0], dtype=np.int32)
        result = ImageDetectionInput(
            image=image,
            boxes=boxes,
            labels=labels,
        )
        return result

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_names)

    def get_benchmark(self) -> type[BenchmarkBase]:
        """Get the benchmark class for this dataset.

        Returns:
            type[BenchmarkBase]: The benchmark class for this dataset
        """
        from vnnort.inference.evaluation.image_detection import ImageDetectionBenchmark

        return ImageDetectionBenchmark
