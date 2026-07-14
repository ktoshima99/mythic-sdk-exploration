from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from vnnort.quantizer.calibrator.histogram_hook import NDHistogram
from vnnort.quantizer.quant_utils import TensorQuantInfo


def generate_distribution_figure(
    tensor_name: str, tensor_statistics: dict[str, NDArray[Any]], sort_by_channels: bool = True
) -> go.Figure:
    """Generate a plot showing the distribution of the tensor corresponding to quant_info.

    Args:
        tensor_name (str): Name of the tensor
        tensor_statistics (dict[str, NDArray[Any]]): Statistics of the tensor
        sort_by_channels (bool, optional): Whether to sort the channel by their values magnitude. Defaults to True.

    Returns:
        go.Figure: Plotly figure containing the tensors values distribution.
    """
    # Create the figure
    fig = go.Figure()
    percentile_50 = tensor_statistics["percentile_50"]
    percentile_99 = tensor_statistics["percentile_99"]
    max_values = tensor_statistics["max_values"]
    quantization_ranges = tensor_statistics["quantization_ranges"]

    n_channels = len(max_values)
    x = np.arange(n_channels)
    channels = x
    ch_texts = [f"Ch: {ch}" for ch in channels]
    if sort_by_channels:
        sorted_indices = np.argsort(-max_values)
        quantization_ranges = quantization_ranges[sorted_indices]
        percentile_50 = percentile_50[sorted_indices]
        percentile_99 = percentile_99[sorted_indices]
        max_values = max_values[sorted_indices]
        channels = channels[sorted_indices]  # type: ignore
        ch_texts = [f"Ch: {ch}" for ch in channels]

    # Add 50% Percentile
    fig.add_trace(
        go.Scatter(
            x=x,
            y=percentile_50,
            fill="tozeroy",
            mode="lines",
            line=dict(color="green", width=2),
            name="50% Percentile",
            text=ch_texts,
        )
    )

    # Add 99% Percentile
    fig.add_trace(
        go.Scatter(
            x=x,
            y=percentile_99,
            fill="tonexty",
            mode="lines",
            line=dict(color="orange", width=2),
            name="99% Percentile",
            text=ch_texts,
        )
    )

    # Add Maximum
    fig.add_trace(
        go.Scatter(
            x=x,
            y=max_values,
            fill="tonexty",
            mode="lines",
            line=dict(color="darkred", width=2),
            name="Maximum",
            text=ch_texts,
        )
    )

    # Add Tensor Range
    fig.add_trace(
        go.Scatter(
            x=x,
            y=quantization_ranges,
            # fill='tonexty',
            mode="lines",
            line=dict(color="blue", width=2, shape="hv"),
            name="Quantization Range",
            text=ch_texts,
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Distribution over all channels for {tensor_name}",
        xaxis_title="Sorted Channels",
        yaxis_title="Values",
        legend=dict(title="Legend"),
        template="simple_white",
    )
    return fig


def generate_channel_histogram(quant_info: TensorQuantInfo, channel: int = 0) -> go.Figure:
    """Generate a plotly histogram for channel of tensor data in quant_info.

    Args:
        quant_info (TensorQuantInfo): TensorQuantInfo object for which histogram is built.
        channel (int, optional): Channel for which histogram is plotted. Defaults to 0.

    Returns:
        go.Figure: Resulting Plotly figure.

    Raises:
        RuntimeError: If TensorQuantInfo has no histograms and no tensor_data or if tensor data is not set.
    """
    tensor_name = quant_info.tensor.name
    histograms = quant_info.tensor_histograms
    tensor_data = quant_info.tensor.data
    if histograms is None and tensor_data is not None:
        tensor_data = quant_info.tensor.data
        if tensor_data is None:
            raise RuntimeError("Tensor data should be set at this point.")
        if tensor_data.ndim > 1:
            tensor_data = tensor_data.reshape([tensor_data.shape[0], -1])
        else:
            tensor_data = tensor_data.reshape([1, -1])
        histograms = NDHistogram.calculate(tensor_data, n_bins=128)
    elif histograms is None:
        raise RuntimeError("Either histograms or data need to be present in TensorQuantInfo")
    bin_values = histograms.bin_values[channel]
    bin_count = histograms.bin_counts[channel]
    x = bin_values
    y = bin_count

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(
        title=f"Histogram of channel {channel} for {tensor_name}",
        xaxis_title="Values",
        yaxis_title="Count",
        template="simple_white",
    )
    return fig


def layerwise_metrics_figure(fp32_result: float, metric_values: dict[str, float]) -> go.Figure:
    """Generate a plotly figure for layerwise metrics.

    Args:
        fp32_result (float): The full fp32 accuracy.
        metric_values (dict[str, float]): A dictionary containing the quantization accuracy for each layer.

    Returns:
        go.Figure: Plotly figure

    """
    layer_indices = np.arange(len(metric_values.keys()))
    layer_names = list(metric_values.keys())
    metric_values = list(metric_values.values())

    fig = go.Figure(
        data=[
            go.Scatter(
                x=layer_indices, y=metric_values, text=layer_names, mode="lines+markers", name="Layerwise Metric"
            )
        ],
        layout=go.Layout(
            title="Layerwise Quantization Metrics",
            xaxis_title="Layer Index",
            yaxis_title="Metric",
            template="simple_white",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=layer_indices,
            y=[fp32_result] * len(layer_indices),
            mode="lines",
            name="Fp32 Result",
            line=dict(dash="dash", color="red"),
        )
    )
    return fig
