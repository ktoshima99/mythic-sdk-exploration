# noqa-flake8-docstrings
import logging
from math import sqrt

import torch
import torch.nn as nn

from munc._pytorch.layers import quantize, clip
from munc._pytorch import noise
from munc.bcm.bcm_models.digitalmodel import PytorchDigitalMMA
from munc._ace_model import BaseInputModel, BaseADCModel


logger = logging.getLogger(__name__)


class BoreasWeightModel(nn.Module):
    """A model of (non-ideal) Boreas weights."""

    def __init__(self, clip_weight, hw_config, noise_config, quantize_weight, pFSR, back_prop_weight_noise,
                 duplicate_weight=False, signed_input=False):
        super().__init__()
        self.clip_weight = clip_weight
        self.hw_config = hw_config
        self.noise_config = noise_config
        self.noise_scale = 1.0
        self.quantize_weight = quantize_weight
        self.pFSR = pFSR
        self.back_prop_weight_noise = back_prop_weight_noise
        self.duplicate_weight = duplicate_weight
        self.signed_input = signed_input

    def forward(self, weight, bias, global_temp):
        def clip_weight(weight, min_val, max_val):
            return clip(weight, min_val, max_val) if self.clip_weight else weight

        noise_config = self.noise_config
        local_temp_delta = noise_config.local_temp_delta * self.noise_scale

        def process_parameters_stored_in_NVM(weight, weight_noise_additive, weight_noise_percentage):
            weight = quantize(weight) if self.quantize_weight else weight
            noisy_weight = weight
            if self.noise_scale != 0:
                noisy_weight = _linear_transform(noisy_weight, noise_config.weight_linear_offset,
                                                 noise_config.weight_linear_slope)
                noisy_weight = noise.temp_shift(noisy_weight, self.pFSR, global_temp, local_temp_delta)
                noisy_weight = noise.weight_noise(noisy_weight, weight_noise_additive,
                                                  weight_noise_percentage,
                                                  not self.back_prop_weight_noise)
            noisy_weight = torch.where(weight != 0, noisy_weight, weight)
            # At this point, noisy_weight may exceed the [-128, 127] range. This is how noise behaves in hardware.
            return noisy_weight

        weight_noise_percentage = noise_config.weight_noise_percentage * self.noise_scale
        weight_noise_additive = noise_config.weight_noise_additive * self.noise_scale

        # Clip at the beginning to prevent gradients outside of physically possible range
        weight = clip_weight(weight, self.hw_config.weight_min, self.hw_config.weight_max)
        bias = clip_weight(bias, self.hw_config.bias_min, self.hw_config.bias_max)

        weight = process_parameters_stored_in_NVM(weight, weight_noise_additive, weight_noise_percentage)
        # FIXME: Bias is not split, so both proportional and additive noise computations for it must take it into
        # account. A simple but not exact solution used here assumes equal splitting.
        bias = process_parameters_stored_in_NVM(bias, weight_noise_additive * sqrt(self.hw_config.bias_rows),
                                                weight_noise_percentage / sqrt(self.hw_config.bias_rows))

        if self.duplicate_weight and self.signed_input:
            weight = torch.cat([weight, -weight], dim=1)

        return weight, bias


class BoreasInputModel(BaseInputModel):
    """A model of Boreas inputs."""


class BoreasADCModel(BaseADCModel):
    """A Boreas ADC model."""

    def __init__(self, clip_dot, hw_config, noise_config, quantize_dot, pFSR, model_multicycle_processing,
                 mma_func, num_fractional_bits=0):
        super().__init__(hw_config=hw_config, mma_func=mma_func, pFSR=pFSR,
                         num_fractional_bits=num_fractional_bits, clip_dot=clip_dot, quantize_dot=quantize_dot)
        self.noise_scale = 1.0
        self.model_multicycle_processing = model_multicycle_processing
        self.noise_config = noise_config
        self.mma_func = mma_func

    def forward(self, z, X, weight, bias, iFSR):
        z = self._scale_to_adc_output_range(z, iFSR)
        if self.model_multicycle_processing:
            adc_clipping_error = _compute_adc_clipping_error(X, z, weight, bias, pFSR=self.pFSR,
                                                             iFSR=iFSR, mma_func=self.mma_func)
            z = z + adc_clipping_error

        z = _linear_transform(z, self.noise_config.adc_linear_offset, self.noise_config.adc_linear_slope)
        ADC_noise_lsb_at_10ifsr = self.noise_config.ADC_noise_lsb_at_10ifsr * self.noise_scale
        z = noise.adc_noise(z, ADC_noise_lsb_at_10ifsr / (iFSR / 2))
        # Note: We do not add BoreasB support for the digital datapath here.
        # The minor BoreasB modeling improvements would not support gradient propagation.
        # Clipping here does not directly represent ADC clipping (or any other real hardware clipping point).
        # It ensures modeled dot product values are not larger than the maximum integer value we can get in
        # a MMA accumulator after summing up 8 ADC samples (one per bit).
        z = self._quantize_and_clip(z)
        return z


def _linear_transform(value, offset, slope):
    return value * slope + offset


@torch.no_grad()
def _compute_adc_clipping_error(inputs, ideal_output, weight, bias, pFSR, iFSR, mma_func):
    bias = bias.unsqueeze(1)
    mma = PytorchDigitalMMA(weight, bias, pFSR=pFSR, iFSR=iFSR, weight_scale=1)
    dot = mma.dot(inputs, dot_op=mma_func)
    return dot - ideal_output
