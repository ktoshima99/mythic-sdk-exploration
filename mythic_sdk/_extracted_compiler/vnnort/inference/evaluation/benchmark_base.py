from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from vnnort.data.container import InputData
from vnnort.data.container import OutputData
from vnnort.data.utils import batch_model_input_data
from vnnort.data.utils import unbatch_model_output_data

if TYPE_CHECKING:
    from vnnort.data.dataloader import Dataloader
    from vnnort.inference.engine import InferenceEngine  # Only imported during type checking for type annotations


class BenchmarkBase(ABC):
    """Abstract base class for all benchmark class implementations.

    This class provides a common interface for all benchmark classes. It defines the `run` method which is used to
    run the benchmark on a given dataset and expects its subclasses to implement the `update` and `compute` methods.

    Optionally, you may also implement the `setup_and_reset` method which is called before each benchmark run to
    perform any necessary setup or reset operations.

    Example:
        class MyBenchmark(BenchmarkBase):
            def setup_and_reset(self):
                self.metric = MeanAveragePrecision()

            def update(self, input_data, output_data):
                self.metric.update(input_data, output_data)

            def compute(self):
                return self.metric.compute()

        # Create a benchmark and run an experiment
        engine = InferenceEngine(model)
        dataloader = Dataloader(...)
        benchmark = MyBenchmark(engine, dataloader)
        results = benchmark.run()
    """

    def __init__(self, engine: "InferenceEngine", dataloader: "Dataloader", verbose: bool = True):
        """Initialize a new BenchmarkBase instance.

        Args:
            engine (InferenceEngine): The inference engine to use for inference
            dataloader (Dataloader): The dataloader to use for benchmarking
            verbose (bool, optional): Whether to print progress bars. Defaults to True.
        """
        self.engine = engine
        self.dataloader = dataloader
        self.verbose = verbose
        self.vid_model = self.engine._vid_model

        self.setup_and_reset()

    def run(self) -> Any:
        """Run the benchmark.

        This method runs the benchmark on the provided dataloader and returns the results.

        Returns:
            Any: The results of the benchmark
        """
        for input_data, model_input in tqdm(self.dataloader, desc="Running benchmark", disable=not self.verbose):
            input_data, model_input = self._prepare_model_input(input_data, model_input)
            model_outputs = self.engine.run(model_input)
            output_data = self._prepare_output_data(model_outputs, input_data)  # type: ignore
            self.update(input_data, output_data)

        results = self.compute()

        # Reset state so that the benchmark can be run again
        self.setup_and_reset()

        return results

    def _prepare_model_input(
        self,
        input_data: InputData | list[InputData],
        model_input: dict[str, NDArray[np.float32]] | None | list[dict[str, NDArray[np.float32]]],
    ) -> tuple[list[InputData], list[dict[str, NDArray[np.float32]]]]:
        """Prepare model input.

        This function takes in the output and makes sure that it is in the correct format. The dataloader output can be
        varying, depending on the provided arguments (batch_size, preprocessing_func). This function makes sure
        that the output is in the correct format for the model.
        """
        # Always work in batched mode
        if not isinstance(input_data, list):
            input_data = [input_data]
            if model_input is None:
                model_input = [None]  # type: ignore

        # In case no preprocessing_func was provided to the dataloader, apply it now
        if all(value is None for value in model_input):  # type: ignore
            model_input = [self.vid_model.preprocess(input_sample) for input_sample in input_data]
            model_input = batch_model_input_data(model_input)

        return input_data, model_input  # type: ignore

    def _prepare_output_data(
        self, model_outputs: dict[str, NDArray[np.float32]], input_data: list[InputData]
    ) -> list[OutputData]:
        """Handle the model output and transform to correct format.

        This function takes in the model output, unbatches it and calls the VidModel postprocessing function to
        transform the output to the correct format.

        Args:
            model_outputs (dict[str, NDArray[np.float32]]): The model output as returned by the InferenceEngine
            input_data (list[InputData]): The InputData objects fed to the model. May be used by the postprocessing
                function.

        Returns:
            list[OutputData]: The processed output of the model
        """
        model_outputs = unbatch_model_output_data(model_outputs)
        output_data = [
            self.vid_model.postprocess(output, input_data) for input_data, output in zip(input_data, model_outputs)
        ]
        return output_data

    def setup_and_reset(self) -> None:
        """Set up and reset the benchmark.

        This function is called before each benchmark run to perform any necessary setup or reset operations.
        Overwrite this, if you want to initialize your benchmark with something specific and stateful.
        """
        pass

    @abstractmethod
    def update(self, input_data: list[InputData], output_data: list[OutputData]) -> None:
        """Update the benchmark with new input and output data.

        This function is called for each batch of input and output data during the benchmark run.

        Args:
            input_data (list[InputData]): Input data to the model
            output_data (list[OutputData]): Output data from the model

        Returns:
            None: None
        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Any:
        """Compute the benchmark results.

        This function is called after the benchmark run to compute the final results.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.

        Returns:
            Any: The computed benchmark results
        """
        raise NotImplementedError
