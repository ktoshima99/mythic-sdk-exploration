from pathlib import Path
import pickle
from typing import Any

from numpy.typing import NDArray

from vnnort.utils.onnx_utils.graph_helper import ONNXGraphHelper

# Default percentile statistics, which are collected per tensor
TENSOR_STATISTIC_PERCENTILES = [50.0, 99.0, 100.0]


class QuantizationReport:
    """Wrapper class to contain all information about the result of the quantization process."""

    def __init__(
        self,
        graph: ONNXGraphHelper,
        tensor_statistics: dict[str, dict[str, NDArray[Any]]],
        layer_metrics: dict[str, Any] | None = None,
    ):
        """Initialize the quantization report.

        Args:
            graph (ONNXGraphHelper): Onnx graph of the original model.
            tensor_statistics (dict[str, dict[str, NDArray[Any]]]): A dictionary containing the percentile statistics for each tensor
            layer_metrics (dict[str, Any] | None): A dictionary containing the quantization accuracy for each layer
        """
        self.network_graph = graph
        self.tensor_statistics = tensor_statistics
        self.layer_metrics = layer_metrics

    def save(self, path: str | Path) -> None:
        """Save this report to path."""
        # Save data with pickle
        with open(path, "wb") as f:

            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "QuantizationReport":
        """Load the report saved at path.

        Args:
            path (str | Path): Path to load the report from.

        Raises:
            ValueError: If the path does not point to a quantization report.

        Returns:
            QuantizationReport: The loaded report
        """
        # Load data with pickle
        with open(path, "rb") as f:
            report = pickle.load(f)
            if not isinstance(report, QuantizationReport):
                raise ValueError(f"Expected {path} to point to a quantization report.")
            return report
