import logging
from typing import Any

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from vnnort.data.container import ImageDetectionInput
from vnnort.data.container import ImageDetectionOutput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase

logger = logging.getLogger(__name__)


class ImageDetectionBenchmark(BenchmarkBase):
    """Class for benchmarking image detection models."""

    def setup_and_reset(self) -> None:
        """Set up and reset the benchmark.

        This function is called before each benchmark run to perform any necessary setup or reset operations.
        Overwrite this, if you want to initialize your benchmark with something specific and stateful.
        """
        self.metric = MeanAveragePrecision()

    def update(self, input_data: list[ImageDetectionInput], output_data: list[ImageDetectionOutput]) -> None:  # type: ignore[override]
        """Update the benchmark with new input and output data.

        This function is called for each batch of input and output data during the benchmark run.

        Args:
            input_data (list[ImageDetectionInput]): Input data to the model
            output_data (list[ImageDetectionOutput]): Output data from the model

        Returns:
            None: None

        Raises:
            TypeError: If input_data or output_data is not a list of ImageDetectionInput or ImageDetectionOutput
        """
        # Check that data types are correct
        if not isinstance(input_data, list) and not all(isinstance(entry, ImageDetectionInput) for entry in input_data):
            raise TypeError("Expected input_data to be a list of ImageDetectionInput")

        if not isinstance(output_data, list) and not all(
            isinstance(entry, ImageDetectionOutput) for entry in output_data
        ):
            raise TypeError("Expected output_data to be a list of ImageDetectionOutput")

        # Convert to torch tensors to use the torch metric library
        groundtruths = [
            {
                "boxes": torch.tensor(entry.boxes),
                "labels": torch.tensor(entry.labels),
            }
            for entry in input_data
        ]

        predictions = [
            {
                "boxes": torch.tensor(prediction.boxes),
                "labels": torch.tensor(prediction.labels),
                "scores": torch.tensor(prediction.scores),
            }
            for prediction in output_data
        ]
        self.metric.update(predictions, groundtruths)

    def compute(self) -> Any:
        """Compute the benchmark results.

        This function is called after the benchmark run to compute the final results.

        Returns:
            Any: The computed benchmark results
        """
        all_metrics = self.metric.compute()

        # Only keep single value metrics
        result = {name: float(value) for name, value in all_metrics.items() if value.numel() == 1}
        return result
