from abc import ABC
from abc import abstractmethod
from typing import Any

from vnnort.data.container import InputData


class BaseRuntime(ABC):
    """Base class for all runtime implementations (e.g. ONNX Runtime and emulator or actual hardware).

    Runtimes are used to run inference on models.
    """

    @abstractmethod
    def __call__(self, model_input: InputData) -> Any:
        """Run a single inference step with a model.

        Args:
            model_input (InputData): The input data to the model

        Returns:
            Any: The output of the model

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources used by the runtime."""
        raise NotImplementedError
