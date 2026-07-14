from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from vnnort.utils.onnx_utils.onnx_hooks import OnnxHookBase


class MinMaxHook(OnnxHookBase):
    """OnnxHookBase used in MinMaxCalibrator to collect minmax statistics of tensors."""

    def __init__(self, tensor_name: str, axis: int | Sequence[int]):
        """Initialize a new hook for some tensor, which collects statistics for all channels on axis `axis`.

        Args:
            tensor_name (str): Name of the tensor this hook is registred for.
            axis (int | Sequence[int]): Tensor axes on which collect statistics.
                All other axis will be flattened and the maximum absolute values for each channel in axes are collected.

        Raises:
            ValueError: If axis is not int or sequence of ints
        """
        super().__init__()
        self._tensor_name = tensor_name
        if isinstance(axis, (list, tuple)):
            self._axis = list(axis)
        elif isinstance(axis, int):
            self._axis = [axis]
        else:
            raise ValueError("axis needs to be int or sequence of ints.")

        # Keep track of intermediate results
        self._max_values: NDArray[np.float32] | None = None

    def on_call(self, data: NDArray[Any]) -> None:
        """Update the statistics with new data.

        Args:
            data (NDArray[Any]): next data of the onnx tensor this hook is registred for.

        Returns:
            None: updates statistics internally
        """
        flattened_data = self._reshape_data_to_target_axes(data)
        max_values = self._calculate_max_abs_values(flattened_data)

        self._update_max_value_buffer(max_values)

    def compute_tensor_ranges(self) -> NDArray[np.float32]:
        """Return the collected max_values.

        Returns:
            NDArray[np.float32]: max_values with shape of provided axis argument

        Raises:
            RuntimeError: If _max_values is not set at this point.
        """
        if self._max_values is None:
            raise RuntimeError("'_max_values' should be set at this point.")
        return self._max_values

    def _reshape_data_to_target_axes(self, data: NDArray[Any]) -> NDArray[Any]:
        """Reshape data so that the axes requested by axis are in the first axes and the rest are flattened.

        Assuming data has shape [d1, d2, d3, d4, d5] and requested axis is [1, 3] then the reshaped data will have
        the shape [d2, d4, d1*d3*d5].
        Args:
            data (NDArray[Any]): Data to reshape.

        Returns:
            NDArray[Any]: Reshaped data.
        """
        data_rank = data.ndim
        new_axes = list(self._axis)
        for axis in range(data_rank):
            if axis not in new_axes:
                new_axes.append(axis)
        permuted_data = np.transpose(data, new_axes)

        # Flatten axes that are not required
        flattened_shape = list(permuted_data.shape[: len(self._axis)]) + [-1]
        flattened_data = np.reshape(permuted_data, flattened_shape)

        return flattened_data

    def _calculate_max_abs_values(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate the maximum absolute value over the last axis of data.

        Args:
            data (NDArray[np.float32]): Data to calculate the maximum absolute value over the last axis.

        Returns:
            NDArray[np.float32]: Maximum absolute values with shape of provided axis argument
        """
        absolute_values = np.abs(data)
        max_values = np.max(absolute_values, axis=-1)
        return np.array(max_values)

    def _update_max_value_buffer(self, max_values: NDArray[np.float32]) -> None:
        """Update the internal buffer with new max values or initialize it if not yet initialized.

        Args:
            max_values (NDArray[np.float32]): Maximum absolute values with shape of provided axis arguments

        Returns:
            None: updates buffers internally

        Raises:
            RuntimeWarning: If shapes do not match
        """
        # Update internal buffer and keep track of highest values
        if self._max_values is None:
            self._max_values = max_values
        else:
            # The shape must not change in between calibration steps!
            if max_values.shape != self._max_values.shape:
                raise RuntimeWarning("Shapes must stay the same for hooked tensors.")

            larger_values_mask = self._max_values < max_values
            self._max_values[larger_values_mask] = max_values[larger_values_mask]
