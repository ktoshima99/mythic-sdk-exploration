# noqa-flake8-docstrings
import logging

import torch.nn as nn
import torch

from munc._pytorch.layers import quantize, clip

logger = logging.getLogger(__name__)


def _identity_weight_model(weight, bias, global_temp):
    return (weight, bias)


def _identity_input_model(X, global_temp):
    return X


class BaseAnalogModel(nn.Module):
    """The base class of Mythic MMA layer models.

    This model represents a hardware implementation of dot product as
    `output_model(dot(input_model(X), weight_model(W)))`.
    A weight model, an input model, and an ADC model are constructed from configurations passed to the constructor.
    By default they are defined as identity functions.
    To define a separable model providing weight, input, and ADC models is sufficient. To change how dot product is
    computed, the `_compute_adc_input` method should be overridden.
    The default implementation of `forward` computes ideal (FP) output.
    """

    def __init__(self, signed_input, mma_func, pFSR, hw_config, noise_config,
                 weight_model=None, input_model=None, adc_model=None):
        super().__init__()
        self.hw_config = hw_config
        self.noise_config = noise_config
        # The noise_scale multiplies all noises and can be used to change noise by a schedule.
        self._noise_scale = 1.0
        self.mma_func = mma_func
        self.signed_input = signed_input
        self.weight_model = weight_model or _identity_weight_model
        self.input_model = input_model or _identity_input_model
        self.adc_model = adc_model or BaseADCModel(hw_config=hw_config, mma_func=None,
                                                   pFSR=pFSR, num_fractional_bits=0,
                                                   clip_dot=False, quantize_dot=False)

    @property
    def noise_scale(self):
        return self._noise_scale

    @noise_scale.setter
    def noise_scale(self, value):
        self._noise_scale = value
        if isinstance(self.weight_model, nn.Module):
            self.weight_model.noise_scale = value
        if isinstance(self.input_model, nn.Module):
            self.input_model.noise_scale = value
        if isinstance(self.adc_model, nn.Module):
            self.adc_model.noise_scale = value

    def _apply_mma_func(self, X, weight, bias):
        hw_config = self.hw_config
        X_norm = 2 ** hw_config.input_bits - 1
        W_norm = max(abs(hw_config.weight_min), abs(hw_config.weight_max))
        return self.mma_func(X / X_norm, weight / W_norm, bias / W_norm)

    def _compute_adc_input(self, X, weight, bias):
        return self._apply_mma_func(X, weight, bias)

    def forward(self, X, weight, bias, global_temp, iFSR):
        weight_model_output, bias_model_output = self.weight_model(weight, bias, global_temp)
        input_model_output = self.input_model(X, global_temp)
        adc_input = self._compute_adc_input(input_model_output, weight_model_output, bias_model_output)
        adc_output = self.adc_model(adc_input, X, weight, bias, iFSR=iFSR)
        return adc_output


def make_analog_model(signed_input, mma_func, pFSR, hw_config, noise_config, hardware_config_name=None,
                      make_weight_model=None, make_input_model=None, make_adc_model=None, model_class=BaseAnalogModel,
                      name='', **kwargs):
    """Create weight, input, and ADC models using the provided functions and create an AnalogModel from them."""
    assert hardware_config_name is None or hw_config.name == hardware_config_name, \
        f"Expected hardware configuration {hardware_config_name}, got {hw_config.name}"
    weight_model = make_weight_model and make_weight_model(hw_config=hw_config, pFSR=pFSR, signed_input=signed_input)
    input_model = make_input_model and make_input_model(hw_config=hw_config, pFSR=pFSR, signed_input=signed_input)
    adc_model = make_adc_model and make_adc_model(hw_config=hw_config, pFSR=pFSR, mma_func=mma_func)
    model = model_class(signed_input=signed_input, mma_func=mma_func, pFSR=pFSR,
                        hw_config=hw_config, noise_config=noise_config, weight_model=weight_model,
                        input_model=input_model, adc_model=adc_model, **kwargs)
    return model


class BaseInputModel(nn.Module):
    """A input model that only handles weight duplication required to support signed input."""

    def __init__(self, hw_config, pFSR, signed_input, duplicate_weight=False):
        super().__init__()
        self.duplicate_weight = duplicate_weight
        self.signed_input = signed_input

    def forward(self, X, global_temp):
        if self.duplicate_weight and self.signed_input:
            pos_inputs = torch.clamp(X, min=0)
            neg_inputs = torch.clamp(-X, min=0)
            X = torch.cat([pos_inputs, neg_inputs], dim=1)
        return X


class BaseADCModel(nn.Module):
    """An ADC model that only applies clipping and quantization."""

    def __init__(self, hw_config, mma_func, pFSR, num_fractional_bits=0, clip_dot=True, quantize_dot=True):
        super().__init__()
        self.clip_dot = clip_dot
        self.hw_config = hw_config
        self.quantize_dot = quantize_dot
        self.num_fractional_bits = num_fractional_bits
        self.pFSR = pFSR

    def _scale_to_adc_output_range(self, z, iFSR):
        return (self.pFSR / iFSR) * self.hw_config.max_abs_dot_product_value * z

    def _quantize_and_clip(self, z):
        fractional_bits_mul = (2 ** self.num_fractional_bits)
        z = z * fractional_bits_mul
        z = quantize(z) if self.quantize_dot else z
        z = z / fractional_bits_mul
        if self.clip_dot and self.hw_config.accum_clip is not None:
            z = torch.clamp(z, *self.hw_config.accum_clip)
        return z

    def forward(self, z, X, weight, bias, iFSR):
        return self._quantize_and_clip(self._scale_to_adc_output_range(z, iFSR))


class BaseWeightModel(nn.Module):
    """A weight model that only applies clipping and quantization."""

    def __init__(self, hw_config, pFSR, clip_weight=True, quantize_weight=True, duplicate_weight=False,
                 signed_input=False):
        super().__init__()
        self.clip_weight = clip_weight
        self.hw_config = hw_config
        self.quantize_weight = quantize_weight
        self.duplicate_weight = duplicate_weight
        self.signed_input = signed_input

    def forward(self, weight, bias, global_temp):
        def clip_weight(weight, min_val, max_val):
            return clip(weight, min_val, max_val) if self.clip_weight else weight

        hw_config = self.hw_config

        weight = quantize(weight, self.hw_config.weight_fractional_bits) if self.quantize_weight else weight
        bias = quantize(bias, self.hw_config.weight_fractional_bits) if self.quantize_weight else bias
        weight = clip_weight(weight, hw_config.weight_min, hw_config.weight_max)
        bias = clip_weight(bias, hw_config.bias_min, hw_config.bias_max)

        if self.duplicate_weight and self.signed_input:
            weight = torch.cat([weight, -weight], dim=1)

        return weight, bias
