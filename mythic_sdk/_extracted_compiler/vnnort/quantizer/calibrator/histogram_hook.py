import logging
import traceback
from typing import Any, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# from vnnort.utils.onnx_utils.hooked_inference import HookedOnnxInferenceSession
from vnnort.utils.onnx_utils.onnx_hooks import OnnxHookBase

# Maximum absolute value allowed as histogram input. Everything else is clipped
MAX_ABS_VALUE = 25000


class NDHistogram:
    """A class, which can be used to represent multiple histograms for NDimensional data."""

    def __init__(self, bin_counts: NDArray[Any], bin_widths: NDArray[Any], min_values: NDArray[Any]):
        """Construct an dimensional histogram.

        Args:
            bin_counts (NDArray[Any]): Array of shape [n_channels, n_bins] containing the counts of each histogram bin
            bin_widths (NDArray[Any]): Array of shape [n_channels] containing the differences between adjacent
                histogram bins per channel.
            min_values (NDArray[Any]): Array of shape [n_channels, n_bins] containing the values of the smallest bins
                per channel.
        """
        self.bin_counts = bin_counts  # [n_channels, n_bins]
        self.bin_widths = bin_widths  # [n_channels]
        self.min_values = min_values  # [n_channels]

    @property
    def n_channels(self) -> int:
        """Return the n_channels this ND Histogram is based on.

        Returns:
            int: Number of channels.
        """
        return int(self.bin_counts.shape[0])

    @property
    def n_bins(self) -> int:
        """Return the number of bins used for all histograms.

        Returns:
            int: Number of bins.
        """
        return int(self.bin_counts.shape[1])

    @property
    def max_values(self) -> NDArray[Any]:
        """For all histograms, return the max values they contain.

        Returns:
            NDArray[Any]: 1D array of shape [n_channels].
        """
        return np.array(self.min_values + self.bin_widths * self.n_bins)

    @property
    def bin_values(self) -> NDArray[Any]:
        """For all histograms and all their bins, return the values they represent.

        Returns:
            NDArray[Any]: 2D array of shape [n_channels, n_bins].
        """
        return np.array(
            self.min_values.reshape(self.n_channels, 1)
            + self.bin_widths.reshape(self.n_channels, 1) * np.arange(self.n_bins).reshape(1, self.n_bins)
        )

    @staticmethod
    def calculate(data: NDArray[Any], n_bins: int) -> "NDHistogram":
        """Calculate N histograms of size n_bins for data.

        Args:
            data (NDArray[Any]): A 2D array of shape [n_channels, n_datapoints] containing the data for all histograms
            n_bins (int): Number of bins to use for all histograms.

        Raises:
            ValueError: if data is not 2D

        Returns:
            NDHistogram: The resulting histogram.
        """
        if not data.ndim == 2:
            raise ValueError("data needs to be 2D")
        n_channels = data.shape[0]

        # Filter out illegal data points
        data[data < -MAX_ABS_VALUE] = -MAX_ABS_VALUE
        data[data > MAX_ABS_VALUE] = MAX_ABS_VALUE
        data[~np.isfinite(data)] = MAX_ABS_VALUE

        min_values = np.min(data, axis=1)
        max_values = np.max(data, axis=1)
        # Take care of the case, where we only have one value
        same_values = min_values == max_values
        max_values[same_values] = min_values[same_values] + 1

        bin_indices = np.round(
            (data - min_values[:, None]) / (max_values[:, None] - min_values[:, None]) * (n_bins - 1)
        ).astype(np.int32)

        # Initialize output counts array of shape [M, N]
        counts = np.zeros((n_channels, n_bins), dtype=int)

        # Use np.add.at for in-place addition to accumulate bin counts
        np.add.at(counts, (np.arange(n_channels)[:, None], bin_indices), 1)
        bin_widths = (max_values - min_values) / n_bins

        return NDHistogram(counts, bin_widths, min_values)

    def ndim_percentile(self, percentile: float) -> NDArray[Any]:
        """Calculate the nth percentile for all histograms.

        Args:
            percentile (float): the nth percentile [0.0, 100.0] to calculate.

        Raises:
            ValueError: If percentile is not between 0.0 and 100.0

        Returns:
            NDArray[Any]: A 1D array of shape [n_channels] containing the nth percentiles per channel.
        """
        # Ensure percentile is a valid value
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100")

        results = []
        for counts, values in zip(self.bin_counts, self.bin_values):
            result = self._compute_percentile(counts, values, percentile)
            results.append(result)
        return np.stack(results)

    @staticmethod
    def _compute_percentile(bin_counts: NDArray[Any], bin_values: NDArray[Any], percentile: float) -> float:
        """
        Compute the nth percentile of a 1D histogram.

        Args:
            bin_counts (NDArray[Any]): Counts in each bin.
            bin_values (NDArray[Any]): Bin centers.
            percentile (float): Percentile to compute (0-100).

        Returns:
            float: The estimated nth percentile value.
        """
        # Step 1: Compute the cumulative distribution function (CDF)
        cumulative_counts = np.cumsum(bin_counts)
        total_count = cumulative_counts[-1]
        cdf = cumulative_counts / total_count  # Normalize CDF to [0, 1]

        # Step 2: Find the bin where the nth percentile falls
        target_fraction = percentile / 100.0  # Convert percentile to a fraction
        bin_index = np.searchsorted(cdf, target_fraction)  # Returns the index which would maintain order (<=)

        # Step 3: Interpolate to find the precise percentile value
        if bin_index == 0:
            # If it's in the first bin, return the first bin value, because it is <= the min value
            return float(bin_values[0])
        else:
            # Linear interpolation between bin_values[bin_index-1] and bin_values[bin_index]
            cdf_low = cdf[bin_index - 1]
            cdf_high = cdf[bin_index]
            value_low = bin_values[bin_index - 1]
            value_high = bin_values[bin_index]

            # Fractional position within the bin
            fraction_within_bin = (target_fraction - cdf_low) / (cdf_high - cdf_low)
            percentile_value = value_low + fraction_within_bin * (value_high - value_low)

            return float(percentile_value)


class DynamicNDHistogram(OnnxHookBase):
    """Wrapper around the NDHistogram class capable of dynamically updating histograms, to be used as OnnxHookBase."""

    def __init__(self, tensor_name: str, axis: int | Sequence[int], n_bins: int = 256, abs_values: bool = True):
        """Initialize a new hook for some tensor, which collects statistics for all channels on axis `axis`.

        Args:
            tensor_name (str): Name of the tensor this hook is registred for.
            axis (int | Sequence[int]): Tensor axes on which to collect statistics.
                All other axis will be flattened and the histograms for each channel in axes are collected.
            n_bins (int): Number of bins to use.
            abs_values (bool): Whether to only consider absolute values during histogram calculation

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
        self.histograms: NDHistogram | None = None
        self._data_shape: Tuple[int, ...] | None = None

        self._n_bins = n_bins
        self._abs_values = abs_values

    def on_call(self, data: NDArray[Any]) -> None:
        """Update the statistics with new data.

        Args:
            data (NDArray[Any]): next data of the onnx tensor this hook is registred for.

        Returns:
            None: this function just updates statistics internally

        Raises:
            RuntimeError: If data shape changes
            Exception: on undefined error
        """
        if self._data_shape is None:
            self._data_shape = data.shape
        else:
            if self._data_shape != data.shape:
                raise RuntimeError(f"Data shape must not change over time for tensor {self._tensor_name}")
            self._data_shape = data.shape
        try:
            flattened_data = self._reshape_data_to_target_axes(data)
            new_histogram = NDHistogram.calculate(flattened_data, self._n_bins)

            self._update_histogram(new_histogram)
        except Exception:
            # Print traceback
            logging.error(f"Error for tensor: {self._tensor_name}")
            traceback.print_exc()
            raise Exception

    def compute_percentile(self, percentile: float) -> NDArray[np.float32]:
        """Return the collected max_values.

        Args:
            percentile(float): The percentile to calculate [0, 100]
        Raises:
            RuntimeError: If run on_call() at least once before computing percentiles
        Returns:
            NDArray[np.float32]: max_values with shape of provided axis argument
        """
        if self.histograms is None or self._data_shape is None:
            raise RuntimeError("Run on_call() at least once before computing percentiles")
        max_values = self.histograms.ndim_percentile(percentile=percentile)

        # In case we want to return absolute values we also calculate the opposite percentile and take the max abs
        if self._abs_values:
            min_values = self.histograms.ndim_percentile(percentile=percentile)
            max_values = np.maximum(np.abs(max_values), np.abs(min_values))
        target_shape = [self._data_shape[ax] for ax in self._axis]
        return max_values.reshape(target_shape)

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
        permuted_data = np.transpose(data, new_axes)  # [ax1, ax2, ..., axn, d1, d2, .., dn]

        # Flatten histogram and data axes
        n_channels = np.prod([data.shape[axis] for axis in self._axis])
        flattened_shape = [n_channels, -1]
        flattened_data = np.reshape(permuted_data, flattened_shape)  # [ax1 * ax2 * ... * axn, d1 * d2 * ... * dn]

        return flattened_data

    def _update_histogram(self, histogram: NDHistogram) -> None:
        if self.histograms is None:
            self.histograms = histogram
        else:
            self.histograms = self._merge(self.histograms, histogram)

    @staticmethod
    def _merge(hist1: "NDHistogram", hist2: "NDHistogram") -> "NDHistogram":
        """Merge two histograms.

        Args:
            hist1 (NDHistogram): First histogram to merge.
            hist2 (NDHistogram): Second histogram to merge.

        Raises:
            ValueError: If histograms have different number of bins
        Returns:
            NDHistogram: Merged histogram

        """
        if not hist1.n_bins == hist2.n_bins:
            raise ValueError("Histograms need to have the same number of bins.")

        # Calculate new range
        new_min_values = np.minimum(hist1.min_values, hist2.min_values)
        new_max_values = np.maximum(hist1.max_values, hist2.max_values)
        bin_widths = (new_max_values - new_min_values) / hist1.n_bins

        # Update new bins with values from both histograms
        new_bin_count = np.zeros((hist1.n_channels, hist1.n_bins), dtype=int)
        for histogram in [hist1, hist2]:
            bin_values = histogram.bin_values

            # Match existing bins to to new bins
            new_bin_indices = np.round(
                (bin_values - new_min_values[:, None])
                / (new_max_values[:, None] - new_min_values[:, None])
                * (hist1.n_bins - 1)
            ).astype(np.int32)

            # Add values to those bins
            np.add.at(new_bin_count, (np.arange(hist1.n_channels)[:, None], new_bin_indices), histogram.bin_counts)

        return NDHistogram(new_bin_count, bin_widths, new_min_values)
