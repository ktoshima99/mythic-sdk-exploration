from typing import Any, List

from sklearn.metrics import accuracy_score

from vnnort.data.container import ImageClassificationInput
from vnnort.data.container import ImageClassificationOutput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase


class ImageClassificationBenchmark(BenchmarkBase):
    """Class for benchmarking image classification models."""

    def setup_and_reset(self) -> None:
        """Set up and reset the benchmark.

        This function is called before each benchmark run to perform any necessary setup or reset operations.
        Overwrite this, if you want to initialize your benchmark with something specific and stateful.

        Returns:
            None: None
        """
        self.gt_labels: List[int] = []
        self.pred_labels: List[int] = []

    def update(self, input_data: list[ImageClassificationInput], output_data: list[ImageClassificationOutput]) -> None:  # type: ignore
        """Update the benchmark with new input and output data.

        This function is called for each batch of input and output data during the benchmark run.

        Args:
            input_data (list[ImageClassificationInput]): Input data to the model
            output_data (list[ImageClassificationOutput]): Output data from the model

        Returns:
            None: None
        """
        input_labels = [input_sample.label for input_sample in input_data]
        output_labels = [output.label for output in output_data]
        self.gt_labels.extend(output_labels)
        self.pred_labels.extend(input_labels)

    def compute(self) -> Any:
        """Compute the benchmark results.

        This function is called after the benchmark run to compute the final results.

        Returns:
            Any: The computed benchmark results
        """
        result = accuracy_score(self.gt_labels, self.pred_labels)
        return {"Accuracy": result}
