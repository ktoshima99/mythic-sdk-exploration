from typing import Any, List

from vnnort.data.container import TextGenerationInput, TextGenerationOutput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase


class TextGenerationBenchmark(BenchmarkBase):
    """Class for benchmarking text generation models."""

    def setup_and_reset(self) -> None:
        """Set up and reset the benchmark.

        This function is called before each benchmark run to perform any necessary setup or reset operations.
        Overwrite this, if you want to initialize your benchmark with something specific and stateful.

        Returns:
            None: None
        """
        self.gt_answers: List[str] = []
        self.pred_answers: List[str] = []

    def update(self, input_data: list[TextGenerationInput], output_data: list[TextGenerationOutput]) -> None:  # type: ignore
        """Update the benchmark with new input and output data.

        This function is called for each batch of input and output data during the benchmark run.

        Args:
            input_data (list[TextGenerationInput]): Input data to the model
            output_data (list[TextGenerationOutput]): Output data from the model

        Returns:
            None: None
        """
        gt_answers = [input_sample.expected_text for input_sample in input_data]
        pred_answers = [output.output_text for output in output_data]
        self.gt_answers.extend(gt_answers)
        self.pred_answers.extend(pred_answers)

    def compute(self) -> Any:
        """Compute the benchmark results.

        This function is called after the benchmark run to compute the final results.

        Returns:
            Any: The computed benchmark results
        """
        correct = 0
        for gt, pred in zip(self.gt_answers, self.pred_answers):
            if gt.strip().lower() == pred.strip().lower():
                correct += 1
        result = correct / len(self.gt_answers) if self.gt_answers else 0.0
        return {"Accuracy": result}
