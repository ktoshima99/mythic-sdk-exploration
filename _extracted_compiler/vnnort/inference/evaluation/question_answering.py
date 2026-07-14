from typing import Any

from evaluate import load

from vnnort.data.container import QuestionAnsweringInput
from vnnort.data.container import QuestionAnsweringOutput
from vnnort.inference.evaluation.benchmark_base import BenchmarkBase


class QuestionAnsweringBenchmark(BenchmarkBase):
    """Class to benchmark a question answering model."""

    def setup_and_reset(self) -> None:
        """Set up and reset the benchmark.

        This function is called before each benchmark run to perform any necessary setup or reset operations.
        Overwrite this, if you want to initialize your benchmark with something specific and stateful.

        Returns:
            None: None
        """
        self.predictions: list[dict[str, Any]] = []
        self.references: list[dict[str, Any]] = []

    def update(self, input_data: list[QuestionAnsweringInput], output_data: list[QuestionAnsweringOutput]) -> None:  # type: ignore
        """Update the benchmark with new input and output data.

        This function is called for each batch of input and output data during the benchmark run.

        Args:
            input_data (list[QuestionAnsweringInput]): Input data to the model
            output_data (list[QuestionAnsweringOutput]): Output data from the model

        Returns:
            None: None
        """
        # For some reason the library expects an id string. While this is intended to come from the original dataset
        # this is completely unnecessary and can be any unique string.
        index_str = str(len(self.predictions))
        for input_entry, output_entry in zip(input_data, output_data):
            self.predictions.append({"prediction_text": output_entry.answer, "id": index_str})
            self.references.append({"answers": input_entry.answers, "id": index_str})

    def compute(self) -> Any:
        """Compute the benchmark results.

        This function is called after the benchmark run to compute the final results.

        Returns:
            Any: The computed benchmark results
        """
        squad_metric = load("squad")
        result = squad_metric.compute(predictions=self.predictions, references=self.references)
        return {"F1": result["f1"]}
