from __future__ import annotations

from dataclasses import dataclass, field

from vnnort.utils.config.base_config import BaseConfig


@dataclass
class QuantizationConfig(BaseConfig):  # noqa: DOC601,DOC603
    """This class contains the configuration for the quantization process.

    This dataclass exposes only the most relevant settings and sets sensible default values for all
    other settings.

    Parameters:
        calibration_dataset_size (int): The number of samples to use for calibration. Defaults to 20.
        percentile (float): Which percentile [0, 100.0] of all values to assume as the maximum range a tensor can take.
        percentile_histogram_bins (int): How many histogram bins to use to use for statistic collection.
        disable_last_layer_channelwise (bool): Whether to disable channelwise quantization for the last layer.
            This is very important for classification models. In case not all classes are present in the calibration
            dataset, channelwise quantization will give very bad results. Defaults to True.
        tensor_n_bits (list[tuple[str, int]]): A list of tuples with tensor names and number of bits to quantize them
    """

    calibration_dataset_size: int = 20
    percentile: float = 100.0
    percentile_histogram_bins: int = 64
    disable_last_layer_channelwise: bool = True
    tensor_n_bits: list[tuple[str, int]] = field(default_factory=list)


QuantizationConfig()
