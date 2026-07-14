"""Define a common hardware configuration class and default values for Boreas A."""
from dataclasses import dataclass
import dataclasses
from functools import partial
from typing import Callable, List, Union, Optional
# TODO: Remove this import once we stop using 3.8 for RMIRs.
import sys

if sys.version_info < (3, 9):
    import importlib_resources as resources
else:
    from importlib import resources

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from munc import _pattern_detector
from munc._hw_config_registry import hw_config_registry
import munc.hydra_configs.noise_config
import munc.hydra_configs.training_model
from munc._constants import HardwareType, ONNXType


@dataclass
class ConfigBase:
    """Base class for hardware configuration objects."""

    name: str

    def __repr__(self):
        """Return printable representation of hardware configuration file."""
        return f'{self.__class__.__name__}(' f'{self.name!r})'

    def derived_copy(self, **kwargs):
        """Return a deep copy of the hardware configuration object."""
        return dataclasses.replace(self, **kwargs)


@dataclass
class HWConfig(ConfigBase):
    """Base hardware configuration class."""

    weight_min: int
    weight_max: int
    weight_fractional_bits: int
    bias_rows: int

    input_bits: int
    max_inputs: int
    signed: bool

    num_of_adcs: int

    pFSR_values: List[Union[int, float]]
    iFSR_values: List[Union[int, float]]

    ds_max_mult: int
    ds_max_shift: int

    accum_clip: List[int]
    max_abs_dot_product_value: int
    RELU_clip: List[int]
    HTAN_clip: List[int]
    HARDSIGMOID_clip: List[int]
    SOFTMAX_clip: List[int]

    target_range_fcn: Callable

    @property
    def bias_min(self):
        """Bias minimum class property."""
        return self.weight_min * self.bias_rows

    @property
    def bias_max(self):
        """Bias maximum class property."""
        return self.weight_max * self.bias_rows

    @property
    def max_unsigned_input(self):
        """Maximum unsigned input value."""
        return 2 ** self.input_bits - 1


@dataclass
class NoiseConfigBase(ConfigBase):
    """Base noise and distortion parameters."""

    temp_delta: float
    local_temp_delta: float

    # Other parameters
    # Used in mythic_linear (o2t), mythic_quantized_mul(o2t)
    ds_trainable_range: List[Union[int, float]]
    # break_FSR_into_pFSR_and_iFSR (op) passes it calculate_half_pfsr_ifsr_and_digital_scale
    half_pFSR_arr: List[Union[int, float]]
    # break_FSR_into_pFSR_and_iFSR (op) passes it calculate_half_pfsr_ifsr_and_digital_scale
    # mythic_linear (o2t) uses it for limits on a trainable iFSR.
    half_iFSR_arr: List[Union[int, float]]


@dataclass
class BoreasNoiseConfig(NoiseConfigBase):
    """Boreas noise and distortion parameters."""

    # Noise parameters
    ADC_noise_lsb_at_10ifsr: float
    weight_noise_percentage: float
    weight_noise_additive: float
    weight_linear_slope: float
    weight_linear_offset: float
    adc_linear_slope: float
    adc_linear_offset: float


@dataclass
class DenaliNoiseConfig(NoiseConfigBase):
    """Denali noise and distortion parameters."""

    nonidealities: dict
    model_common_mode: bool
    flash_model_name: Optional[str]


def get_hw_config(node):
    """Return hardware configuration of an ONNX node."""
    # Default to Boreas if no hardware configuration is specified for compatibility with older models.
    return node.model.hwconfig or boreas_hw_config


_BOREAS_TARGET_RANGES = [
    # (EDGE_PRED, RANGE) pairs

    # MM to MM without an explicit action. "Without an explicit action" assumes hardtanh which is effectively
    # a clip operation.
    (_pattern_detector.is_MM_output_MM_block_input, [-127.5, 127.5]),
    # If edge is feeding the matrix multiply, set target scale to 255.5 if unsigned and 127.5 if signed
    (_pattern_detector.is_nonMM_output_MM_block_unsigned_input, [0, 255.5]),
    (_pattern_detector.is_nonMM_output_MM_block_signed_input, [-127.5, 127.5]),
    # MM block without an explicit activation assumes hardtanh which is effectively
    # a clip operation.
    (_pattern_detector.implies_htanh_at_this_edge, [-127.5, 127.5]),
    # Scale is 255.5 if edge input to RELU following MM
    (partial(_pattern_detector.is_input_to_activation, op_type=ONNXType.RELU), [0, 255.5]),
    (partial(_pattern_detector.is_input_to_activation, op_type=ONNXType.LEAKY_RELU), [-127.5, 127.5]),
    (partial(_pattern_detector.is_input_to_activation, op_type=ONNXType.SWISH), [-127.5, 127.5]),
    # Scale is 255.5 if edge is output of HARDSIGMOID or ReLU6
    (partial(_pattern_detector.is_output_of_activation, op_type=ONNXType.CLIP), [0, 255.5]),
    # Scale is 255.5 if edge input to RELU following MM
    (_pattern_detector.is_composite_scale_output, [-255.5, 255.5]),
    # Output of a standalone MUL node that is used to scale input of an ADD.
    # TODO: Support signed output. Currently input scaling nodes are only added after a ReLU, but we do not want
    # to depend on it.
    (_pattern_detector.is_output_of_add_input_scaler, [0, 255.5]),

]


_DENALI_TARGET_RANGES = [
    # (EDGE_PRED, RANGE) pairs

    # MM to MM without an explicit action. "Without an explicit action" assumes hardtanh which is effectively
    # a clip operation.
    (_pattern_detector.is_MM_output_MM_block_input, [-127.5, 127.5]),
    # If edge is feeding the matrix multiply, set target scale to 255.5 if unsigned and 127.5 if signed
    (_pattern_detector.is_nonMM_output_MM_block_unsigned_input, [0, 255.5]),
    (_pattern_detector.is_nonMM_output_MM_block_signed_input, [-127.5, 127.5]),
    # MM block without an explicit activation assumes hardtanh which is effectively
    # a clip operation.
    (_pattern_detector.implies_htanh_at_this_edge, [-127.5, 127.5]),
    # Scale is 255.5 if edge input to RELU following MM
    (partial(_pattern_detector.is_input_to_activation, op_type=ONNXType.RELU), [0, 255.5]),
    (partial(_pattern_detector.is_input_to_activation, op_type=ONNXType.LEAKY_RELU), [-127.5, 127.5]),
    (partial(_pattern_detector.is_input_to_activation, op_type=ONNXType.SWISH), [-127.5, 127.5]),
    # Scale is 255.5 if edge is output of HARDSIGMOID or ReLU6
    (partial(_pattern_detector.is_output_of_activation, op_type=ONNXType.CLIP), [0, 255.5]),
    # Scale is 127.5 if edge is a composite scale node output.
    (_pattern_detector.is_composite_scale_output, [-127.5, 127.5]),
    # Output of a standalone MUL node that is used to scale input of an ADD.
    # TODO: Support signed output. Currently input scaling nodes are only added after a ReLU, but we do not want
    # to depend on it.
    (_pattern_detector.is_output_of_add_input_scaler, [0, 255.5]),
    (_pattern_detector.is_output_of_softmax, [0, 255.5]),
    (_pattern_detector.is_output_of_softmax_output_multiplier, [0, 255.5]),

    # Temporary signed-only inputs and outputs for 2-input Mul and MatMul inputs
    (_pattern_detector.is_input_of_multi_input_mul_matmul, [-127.5, 127.5]),
    (_pattern_detector.is_output_of_multi_input_mul_matmul, [-127.5, 127.5]),
    (_pattern_detector.is_output_of_multi_input_mul_or_matmul_output_multiplier, [-127.5, 127.5]),
]


def get_target_range(target_ranges, model, edge):
    """Return target range for a specific edge based on `target_ranges` specifications.

    Parameters
    ----------
    target_ranges : List[Tuple[Callable[[ONNXModel, str], Bool], List]]
        A list of edge matching predicates and their corresponding valid range, i.e. of (EDGE_PRED, RANGE) pairs.
        An edge predicate takes an ONNXModel and an edge, and returns true if the corresponding range should be used.
    model : munc._onnx_model.ONNXModel
        MUNC's ONNX model wrapper.
    edge : str
        Edge name to retrieve range for.

    Returns
    -------
    list
        List represnetation of the valid range (i.e. [min, max]).
    """
    # Include margin (if needed)
    target_range_multiplier = 1.0

    def shrink(taget_range):
        return [x * target_range_multiplier for x in taget_range]

    def find_first_match(pred, seq):
        return next(filter(pred, seq), None)

    def spec_pred(spec):
        return spec[0]

    def spec_range(spec):
        return spec[1]

    def is_match(spec):
        return spec_pred(spec)(model, edge)

    # Take the first range definition that matches `edge`.
    match = find_first_match(is_match, target_ranges)
    return shrink(spec_range(match)) if match else None


boreas_hw_config = HWConfig(
    name=HardwareType.BOREAS,

    weight_min=-128,
    weight_max=127,
    weight_fractional_bits=0,
    bias_rows=6,

    input_bits=8,
    max_inputs=1008,
    signed=False,

    num_of_adcs=256,

    pFSR_values=[2, 4, 6, 8],
    iFSR_values=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],

    ds_max_mult=255,
    ds_max_shift=7,

    accum_clip=[-256, 255],
    max_abs_dot_product_value=255,
    RELU_clip=[0, 255],
    HTAN_clip=[-128, 127],
    HARDSIGMOID_clip=[0, 255],
    SOFTMAX_clip=[0, 255],

    target_range_fcn=partial(get_target_range, _BOREAS_TARGET_RANGES),
)
hw_config_registry.register(boreas_hw_config)


denali_hw_config = HWConfig(
    name=HardwareType.DENALI,

    weight_min=-128,
    weight_max=127,
    weight_fractional_bits=8,
    bias_rows=15,

    input_bits=8,
    max_inputs=1200,
    signed=True,

    num_of_adcs=256,

    pFSR_values=[1.0, 3.0, 5.0, 10.0],
    iFSR_values=[1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0, 11.25, 12.5, 13.75, 15.0, 16.25, 17.5, 18.75, 20.0],

    ds_max_mult=255,
    ds_max_shift=7,

    accum_clip=[-128, 127],
    max_abs_dot_product_value=128,
    RELU_clip=[0, 255],
    HTAN_clip=[-128, 127],
    HARDSIGMOID_clip=[0, 255],
    SOFTMAX_clip=[0, 255],

    target_range_fcn=partial(get_target_range, _DENALI_TARGET_RANGES),
)
hw_config_registry.register(denali_hw_config)


def register_noise_models(package, define_global=False):
    """Register noise models provided in `.yaml` files in a package."""
    with initialize_config_module(package, version_base=None):
        for f in resources.files(package).iterdir():
            if f.name.endswith('.yaml'):
                name = f.name[:-5]
                cfg = compose(config_name=name)
                cfg.name = name
                noise_config = instantiate(cfg)
                if define_global:
                    globals()[name] = noise_config


register_noise_models(munc.hydra_configs.noise_config.__name__, True)


def get_make_analog_model(training_model_name, noise_config, overrides=[], instantiate_model=True):
    """Return a function that creates an analog model with the specified training model and noise configuration."""
    with initialize_config_module(munc.hydra_configs.training_model.__name__, version_base=None):
        cfg = compose(config_name=training_model_name, overrides=overrides)
        cfg.noise_config = noise_config
        return instantiate(cfg) if instantiate_model else cfg
