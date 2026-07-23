from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class OnnxHookBase(ABC):
    """Base class for all hooks of the HookedOnnxInferenceSession.

    This class can be used as a base class for hooks, which can be used in conjunction with HookedOnnxInferenceSession.
    The callback `on_call` will be attached to a specific tensor and is called during each step of
    HookedOnnxInferenceSession.run() with the data the tensor holds at that time. It is up to the concrete
    implementation of the hook to handle the data and keep track of intermediate results. In the end, compute() can
    be called to aggregate all results.
    """

    @abstractmethod
    def on_call(self, data: Any) -> None:
        """Handle next data output of the tensor this hook is attached to."""
        raise NotImplementedError


class OnnxOutputHook(OnnxHookBase):
    """Hook to keep track of all intermediate results of a tensor."""

    def __init__(self) -> None:
        """Initialize the hook with a list to store all intermediate results."""
        self._outputs: list[Any] = []

    def on_call(self, data: Any) -> None:
        """Add data to internal data list."""
        self._outputs.append(data.copy())

    def compute(self) -> list[Any]:
        """Return all intermediate results."""
        return self._outputs


class ONNXTensorCompareHook(OnnxHookBase):
    """Hook to check if the contents of a tensors stay the same with different model inputs.

    This hook can be used in conjunction with the HookedONNXInferenceSession class to track the contents of
    of a specific tensor of time. This hook has the advantage that only one whole tensor has to be kept in
    memory at all times.

    As of now this Hook is meant to be called exactly twice. After the second call, output_is_static() returns,
    whether the tensor has changed or not. The last tensor data is contained in the `output` member variable.
    """

    def __init__(self) -> None:
        """Initialize the hook with a list to store all intermediate results."""
        self.output: Any = None
        self._same_output: bool = None
        self._is_compressed = False

        self.calls = 0

    def on_call(self, data: NDArray[Any]) -> None:
        """Parse data by HookedONNXInferenceSession for corresponding tensor."""
        self.calls += 1
        """Add data to internal data list."""
        if self._same_output is not None:  # The hook should not be called more than twice
            msg = "on_call was called more than twice"
            raise RuntimeError(msg)

        # First call:
        if self.output is None:
            if data.size > 32:
                data = self._calculate_data_features(data)  # type: ignore
                self._is_compressed = True
            self.output = data

        # Second call
        else:
            if data.size > 32:
                second_compressed_data = self._calculate_data_features(data)
                first_compressed_data = self.output
                self._same_output = self._compare_features(first_compressed_data, second_compressed_data)

                # Assign original data as output since it is still needed
                self.output = data

            else:
                self._same_output = self._compare_features(data, self.output)

    def output_is_static(self) -> bool:
        """Return all intermediate results."""
        return self._same_output

    def _calculate_data_features(self, data: NDArray[Any]) -> list[NDArray[Any]]:
        result = []
        for dim in range(data.ndim):
            current_axes = tuple([i for i in range(data.ndim) if i != dim])
            result.append(data.mean(axis=current_axes))
        return result

    def _compare_features(self, data1: Any, data2: Any) -> bool:
        if isinstance(data1, list):
            return all(np.allclose(d1, d2, atol=1e-08, rtol=0) for d1, d2 in zip(data1, data2))
        elif isinstance(data1, np.ndarray):
            return np.allclose(data1, data2, atol=1e-08, rtol=0)
        else:
            msg = "Invalid data types"
            raise RuntimeError(msg)
