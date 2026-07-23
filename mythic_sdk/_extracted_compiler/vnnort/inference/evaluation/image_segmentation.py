from typing import Any

import numpy as np
from numpy.typing import NDArray

from vnnort.data.container import InputData
from vnnort.data.container import OutputData
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase


def compute_mIoU(pred_map: NDArray[np.uint32], gt_map: NDArray[np.uint32]) -> float:
    """
    Compute mean Intersection over Union (mIoU) between segmentation maps.

    Args:
        pred_map (NDArray[np.uint32]): Predicted segmentation map with same shape as gt_map
        gt_map (NDArray[np.uint32]): Ground truth segmentation map same shape as pred_map.

    Returns:
        float: mean IoU score.

    Raises:
        ValueError: If dtypes or shapes are incorrect.
    """
    # Check input correctness
    if pred_map.dtype != np.uint32 or gt_map.dtype != np.uint32:
        msg = "Segmentation maps need to be of type uint32."
        raise ValueError(msg)
    if pred_map.shape != gt_map.shape:
        msg = "Segmentation maps need to have the same shape"
        raise ValueError(msg)

    # Flatten the inputs
    pred_map = pred_map.flatten()
    gt_map = gt_map.flatten()

    # Confusion matrix: row = gt, col = pred
    num_classes = max(int(np.max(gt_map)) + 1, int(np.max(pred_map)) + 1)
    mask = (gt_map >= 0) & (gt_map < num_classes)  # valid mask
    confusion = np.bincount(num_classes * gt_map[mask] + pred_map[mask], minlength=num_classes**2).reshape(
        num_classes, num_classes
    )

    intersection = np.diag(confusion)
    union = (
        confusion.sum(axis=0)  # predicted
        + confusion.sum(axis=1)  # ground truth
        - intersection  # subtract intersection
    )

    iou = intersection / np.maximum(union, 1)  # avoid division by zero
    mIoU = np.mean(iou[union > 0])  # ignore classes with no presence

    return float(mIoU)


class ImageSegmentationBenchmark(BenchmarkBase):
    """Class for benchmarking image segmentation models on the mIOU metric."""

    def setup_and_reset(self) -> None:
        """Set up and reset the benchmark.

        This function is called before each benchmark run to perform any necessary setup or reset operations.
        Overwrite this, if you want to initialize your benchmark with something specific and stateful.

        Returns:
            None: None
        """
        self.mIoUs: list[float] = []

    def update(self, input_data: list[InputData], output_data: list[OutputData]) -> None:
        """Update the benchmark with new input and output data.

        This function is called for each batch of input and output data during the benchmark run.

        Args:
            input_data (list[InputData]): Input data to the model
            output_data (list[OutputData]): Output data from the model

        Returns:
            None: None

        Raises:
            ValueError: If the length of input and output data list is not the same
        """
        if not len(input_data) == len(output_data):
            raise ValueError("Length of input and output data list need to be the same")

        for current_input, current_output in zip(input_data, output_data):
            gt_map = current_input.segmentation_map  # type: ignore
            pred_map = current_output.segmentation_map  # type: ignore
            mIoU = compute_mIoU(pred_map, gt_map)
            self.mIoUs.append(mIoU)

    def compute(self) -> Any:
        """Compute the benchmark results.

        This function is called after the benchmark run to compute the final results.

        Returns:
            Any: The computed benchmark results
        """
        result = np.mean(self.mIoUs)
        return {"mIoU": result}
