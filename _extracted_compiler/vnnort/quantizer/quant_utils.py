from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from vnnort.quantizer.calibrator.histogram_hook import NDHistogram
from vnnort.utils.onnx_utils.graph_helper import Tensor


def round_up_to_power_of_two(tensor: NDArray[np.float32]) -> NDArray[np.float32]:
    """Round all entries of tensor to the nearest higher power of two.

    Args:
        tensor (NDArray[np.float32]): The tensor to round.

    Raises:
        ValueError: If tensor contains negative values.
    Returns:
        NDArray[np.float32]: A new tensor with rounded values.
    """
    # Check for negative/0 values
    if np.any(tensor <= 0):
        raise ValueError("Negative values not supported")
    return np.array(2 ** np.ceil(np.log2(tensor)))


def fill_zero_entries(
    tensor: NDArray[np.float32], epsilon: float = 1e-9, fill_value: float = 1.0
) -> NDArray[np.float32]:
    """Remove (near) zero entries from tensors by filling them with fill_value.

    Args:
        tensor (NDArray[np.float32]): The tensor to modify.
        epsilon (float, optional): Minimum value to consider as zero. Defaults to 1e-9.
        fill_value (float, optional): Value to fill zero entries with. Defaults to 1.0.

    Returns:
        NDArray[np.float32]: The modified tensor.
    """
    # Everything smaller than this is considered zero
    abs_max_values = np.abs(tensor)
    max_values_are_zero = abs_max_values < epsilon

    result = np.copy(tensor)
    result[max_values_are_zero] = fill_value

    return result


def calculate_quantization_range_from_histogram(histogram: NDHistogram, percentile: float) -> NDArray[Any]:
    """Given an NDHistogram, calculate the quantization range values based on the provided percentile.

    Args:
        histogram (NDHistogram): NDHistogram to use for calculating the quantization ranges
        percentile (float): Which percentile to use

    Raises:
        ValueError: If percentile is not between 0.0 and 100.0

    Returns:
        NDArray[Any]: 1D Array of shape histogram.n_channels, containing the quantization ranges per channel
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("Percentile needs to be 0.0 <= percentile <= 100.0")

    # We are interested in the absolute max value, so calculate both the lower and upper percentiles
    upper_percentiles = np.abs(histogram.ndim_percentile(percentile))
    lower_percentiles = np.abs(histogram.ndim_percentile(100.0 - percentile))
    result: NDArray[Any] = np.where(upper_percentiles > lower_percentiles, upper_percentiles, lower_percentiles)
    return result


def calculate_quantization_range_from_data(
    data: NDArray[Any], axes: int | tuple[int, ...], percentile: float
) -> NDArray[Any]:
    """Given a data tensor, calculate the quantization range values based on the provided percentile on axes.

    Args:
        data (NDArray[Any]): array of any shape of at least rank axis
        axes (int | tuple[int, ...]): Over which axes to compute the quantization ranges
        percentile (float): Which percentile to use

    Returns:
        NDArray[Any]: 1D Array of shape data.shape[axis], containing the quantization ranges per channel

    Raises:
        ValueError: For invalid inputs
    """
    if type(axes) is int:
        axes = (axes,)
    axes = cast(tuple[int, ...], axes)

    if any(ax >= data.ndim or ax < -data.ndim for ax in axes):
        raise ValueError("Axis out of range")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("Percentile needs to be 0.0 <= percentile <= 100.0")
    if data.ndim == 0:
        raise ValueError("Data needs to be at least 1D tensor")

    # Reshape so that target axes are in front
    new_order = list(range(data.ndim))
    for i, ax in enumerate(axes):
        new_order.remove(ax)
        new_order.insert(i, ax)
    reordered_data = np.transpose(data, new_order)

    # Flatten
    target_shape = [data.shape[ax] for ax in axes] + [-1]
    reordered_data = reordered_data.reshape(target_shape)

    # Calculate percentiles
    absolute_data = np.abs(reordered_data)
    result = np.percentile(absolute_data, percentile, axis=-1)
    return np.array(result)


def make_quantization_ranges_tensor_wide(quantization_ranges: NDArray[np.float32]) -> NDArray[np.float32]:
    """Given a 1D tensor of range values, calculate maximum and set all values to it.

    Args:
        quantization_ranges (NDArray[np.float32]): The quantization ranges to modify.
    Returns:
        NDArray[np.float32]: The modified quantization ranges.
    Raises:
        ValueError: If quantization_ranges is not 1D tensor

    """
    # tensor ranges needs to be 1D tensor
    if not quantization_ranges.ndim == 1:
        raise ValueError("tensor_ranges needs to be 1D tensor")

    result = np.empty_like(quantization_ranges)
    result[:] = np.max(quantization_ranges)
    return result


def power_of_two_values_to_exponents(tensor_ranges: NDArray[np.float32], epsilon: float = 1e-9) -> NDArray[np.int8]:
    """Convert power of two tensor values to max exponents representation.

    Given a value which is a power of two (e.g. 1, 2, 4, 8, ...), it can also be represented by its 2-logarithm.
    2**x
    Args:
        tensor_ranges (NDArray[np.float32]): The tensor values to convert.
        epsilon (float, optional): Minimum value to consider as zero. Defaults to 1e-9.

    Raises:
        ValueError: If tensor ranges are not power of two
    Returns:
        NDArray[np.int8]: The 2log exponents representation.
    """
    absolute_values = np.abs(tensor_ranges)
    absolute_values[absolute_values < epsilon] = epsilon
    max_exponents_fp32 = np.log2(absolute_values)
    max_exponents = np.round(max_exponents_fp32).astype(np.int8)

    # All tensor ranges need to be power of two, so there should be no difference between log and rounding
    if not np.all(np.isclose(max_exponents, max_exponents_fp32)):
        raise ValueError("Tensor ranges need to be power of two")
    return np.array(max_exponents)


def quantize_values(
    fp_data: NDArray[np.float32], max_exponents: NDArray[np.int8], n_bits: int, axis: int = 0
) -> NDArray[np.int8] | NDArray[np.int16]:
    """Quantize values in fp_data based on max exponents and n_bits to use.

    Args:
        fp_data (NDArray[np.float32]): The data to quantize. Can be of any shape [D1, ..., Dn]
        max_exponents (NDArray[np.int8]): The max exponents to use. Must be 1D vector of shape [Dx], where Dx refers
            to the axis denoted by the `axis` parameter.
        n_bits (int): The number of bits to use.
        axis (int): axis where to apply the max exponents.
    Raises:
        ValueError: If n_bits is not 8 or 16 or max_exponents is not 1D tensor or fp_data and max_exponents do not
            have same length
    Returns:
        NDArray[np.int8] | NDArray[np.int16]: The quantized data.
    """
    if n_bits != 8 and n_bits != 16:
        raise ValueError("n_bits cannot be larger than 16")

    if max_exponents.ndim != 1:
        raise ValueError("max_exponents needs to be 1D tensor")

    if fp_data.shape[axis] != max_exponents.shape[0] and max_exponents.shape[0] != 1:
        raise ValueError("fp_data and max_exponents need to have same length in the axis dimension")

    fp_data_shape = fp_data.shape

    # Reshape max_exponents to [1, 1, ..., fp_data.shape[axis], 1, 1, ..., 1] to utilize broadcasting
    new_shape = np.ones(len(fp_data_shape), dtype=np.int64)
    new_shape[axis] = max_exponents.shape[0]
    max_exponents = max_exponents.reshape(new_shape)

    TargetDataType = np.int16 if n_bits == 16 else np.int8
    max_value = 2 ** (n_bits - 1)  # This corresponds the data  type being used (-1 one for sign)
    scale_factors = max_value / (2.0**max_exponents)
    result = np.round(fp_data * scale_factors)
    result = np.clip(result, -max_value, max_value - 1).astype(TargetDataType)

    return result  # type: ignore


def calculate_nd_kurtosis(histogram: NDHistogram) -> NDArray[Any]:
    """Calculate the kurtosis values for all histograms in histogram.

    Args:
        histogram(NDHistogram): The histogram, which will be used as basis.

    Returns:
        NDArray[Any]: An N-dim vector, with each entry being the kurtosis value of the corresponding channel in the histogram
    """
    bin_centers = histogram.bin_values
    bin_counts = histogram.bin_counts

    # Normalize counts to probabilities
    total_count = np.sum(bin_counts, axis=1, keepdims=True)
    probabilities = bin_counts / total_count

    # Calculate mean
    mean = np.sum(probabilities * bin_centers, axis=1, keepdims=True)

    # Calculate variance
    variance = np.sum(probabilities * (bin_centers - mean) ** 2, axis=1, keepdims=True)

    # Calculate kurtosis
    kurtosis = np.sum(probabilities * (bin_centers - mean) ** 4, axis=1, keepdims=True) / (variance**2)

    return np.array(kurtosis)


@dataclass
class TensorQuantInfo:  # noqa: DOC601,DOC603
    """Class to store quantization information for a tensor.

    Parameters:
        tensor (Tensor): The tensor to quantize.
        n_bits (int): The number of bits to use.
        n_fraction_bits (int): Number of fraction bits to use (out of n_bits)
        axis (int | tuple[int, ...]): The axis to quantize.
        power_of_two_scaling_only (bool, optional): Whether to use power of two scaling only. Defaults to False.
        tensor_histograms (NDHistogram | None): A ND histogram representing the distribution of the tensor
            along the specified axis This will be populated by a calibrator tasked with collecting histograms.
            Defaults to None.
        quantization_ranges (NDArray[Any] | None): The Quantization Ranges of the Tensor.
        max_exponents (NDArray[np.int8] | None): The max exponents of this tensor.
        adjusted_max_exponents (NDArray[np.int8] | None): The adjusted max exponents of this tenso (debugging field)
    """

    tensor: Tensor
    n_bits: int
    n_fraction_bits: int
    axis: int | tuple[int, ...]
    power_of_two_scaling_only: bool = False
    tensor_histograms: NDHistogram | None = None
    quantization_ranges: NDArray[Any] | None = None
    max_exponents: NDArray[np.int8] | None = None
    adjusted_max_exponents: NDArray[np.int8] | None = None  # Debugging field. May go away later

    def __repr__(self) -> str:
        """Return string representation of TensorQuantInfo."""
        return "TensorQuantInfo"
