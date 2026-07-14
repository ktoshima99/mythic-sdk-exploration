"""Denali reference model wrappers for MythicMMA.

This module depends on the reference model (internal package).
"""
from dataclasses import replace
from functools import partial
import math

import torch.nn.functional as F
import torch

from mythic.acm.denali.ref.reference_flash_cell_model import (ReferenceFlashCellModel, find_temp_code,
                                                              make_conv2d, make_linear, merge_flash_arrays,
                                                              select_flash_array_rows)
from mythic.acm.denali.ref.reference_aidac_model import ReferenceAIDACModel, concat_aidacs, select_aidacs
from mythic.acm.denali.ref.reference_adc_model import ReferenceADCModel
from mythic.acm.denali.ref.lut_sar_adc_model import LUTSARADCModel
from munc._session_tools import split_bias
from munc._constants import BiasSplittingMethod
from munc._ace_model import BaseAnalogModel, make_analog_model
from munc._denali_ace_separable_model import (BaseDenaliADCModel, _hw_parameters_as_dict, _randomize_hw_model)
from mythic.acm.denali.training.polynomial_separable_model import DEFAULT_AIDAC_GAIN


class DenaliReferenceADCModel(BaseDenaliADCModel):
    """A BaseADCModel wrapper around the Denali reference ADC model.

    It implements the interface MythicMMA expects from hardware models.
    """

    adc_model_class = ReferenceADCModel

    def __init__(self, hw_config, mma_func, pFSR, num_fractional_bits=0, clip_dot=True, quantize_dot=True,
                 model_common_mode=True, nonidealities={}):
        super().__init__(hw_config=hw_config, mma_func=mma_func, pFSR=pFSR,
                         num_fractional_bits=num_fractional_bits, clip_dot=clip_dot, quantize_dot=quantize_dot,
                         model_common_mode=model_common_mode, nonidealities=nonidealities)
        assert self.clip_dot, "Denali Reference ADC model always clips the dot product."
        assert self.quantize_dot, "Denali Reference ADC model always quantizes the dot product."

    def _make_hw_model(self, fsr, num_adcs, device):
        hw_model = self.adc_model_class(fsr=fsr, num_adcs=num_adcs, device=device)
        hw_model = _randomize_hw_model(hw_model, self.nonidealities)
        hw_model = hw_model.calibrate_comparator(100)
        return hw_model

    def configure_nonidealities(self, nonidealities):
        """Configure nonidealities for the ADC model."""
        super().configure_nonidealities(nonidealities)
        # FIXME: If we are not changing the ADC offset, we should not recalibrate?
        self.hw_model = self.hw_model.calibrate_comparator(100)


# TODO: It may be better to have one DenaliADCModel class and automatically switch between
# ReferenceADCModel/LUTSARADCModel/ApproximateADCModel within it based on a parameter,
# e.g. on _update_models_every_time.
class DenaliLUTSARADCModel(DenaliReferenceADCModel):
    """A BaseADCModel wrapper around the Denali LUT SAR ADC model.

    It implements the interface MythicMMA expects from hardware models.
    """

    adc_model_class = LUTSARADCModel


class DenaliReferenceModel(BaseAnalogModel):
    """An AnalogModel wrapper around the Denali reference model.

    It implements the interface MythicMMA expects from hardware models.
    """

    def __init__(self, signed_input, mma_func, pFSR, hw_config, weight_model, input_model, adc_model, noise_config,
                 model_common_mode, aidac_model_nonidealities={}, flash_model_nonidealities={}, flash_model=None,
                 flash_model_chunk_size=0):
        mma_func = self._get_denali_reference_model_mma_func(mma_func, compute_common_mode_current=model_common_mode)
        super().__init__(signed_input=signed_input, mma_func=mma_func, pFSR=pFSR, hw_config=hw_config,
                         weight_model=weight_model, input_model=input_model, adc_model=adc_model,
                         noise_config=noise_config)
        self.aidac_model_nonidealities = aidac_model_nonidealities
        self.flash_model_nonidealities = flash_model_nonidealities
        self.flash_model_name = flash_model
        self.pFSR = pFSR
        self._update_models_every_time = True
        self.aidac_model = None
        self.flash_model = None
        self.temperature = noise_config.nonidealities.get('temperature', 20.0)
        self.inference_temperature = noise_config.nonidealities.get('inference_temperature', self.temperature)
        self._init_temp_codes()
        self.inference_veg = noise_config.nonidealities.get('inference_veg', self.veg_code)
        self.aidac_gain = DEFAULT_AIDAC_GAIN
        self.inference_aidac_gain = noise_config.nonidealities.get('inference_aidac_gain', self.aidac_gain)
        # Setting this to non-zero enables chunking flash cell model inputs into `self._get_chunk_size` batches to
        # reduce memory usage.
        self.flash_model_chunk_size = flash_model_chunk_size

    @property
    def update_models_every_time(self):
        """Whether to create new hardware models every time the MMA function is called."""
        return self._update_models_every_time

    @update_models_every_time.setter
    def update_models_every_time(self, value):
        """Set whether to create new hardware models every time the MMA function is called."""
        self._update_models_every_time = value
        self.adc_model.update_models_every_time = value
        self.weight_model.update_models_every_time = value
        if value:
            self.reset_hardware_parameters()

    def reset_hardware_parameters(self):
        """Reset the underlying ADC hardware model to force creating a new instance next time the model is used."""
        self.aidac_model = None
        self.flash_model = None

    def _compute_adc_input(self, X, weight, bias):
        hw_config = self.hw_config
        W_norm = max(abs(hw_config.weight_min), abs(hw_config.weight_max))
        # Split bias into rows.
        bias_rows = split_bias(bias, num_bias_splits=hw_config.bias_rows, max_abs_weight=W_norm,
                               method=BiasSplittingMethod.OVERFLOW, use_fp=True)
        return self._apply_mma_func(X, weight, bias_rows)

    def _init_temp_codes(self):
        # This function and _make_hw_models must use the same parameters (temperature, pFSR, AIDAC gain).
        aidac_model = ReferenceAIDACModel(temperature=torch.as_tensor(self.temperature, dtype=torch.float64),
                                          pFSR=self.pFSR)
        flash_model = ReferenceFlashCellModel(temperature=torch.as_tensor(self.temperature, dtype=torch.float64),
                                              pFSR=self.pFSR)
        self.vref_temp_code, self.veg_code = find_temp_code(self.temperature, pFSR=self.pFSR,
                                                            aidac_model=aidac_model, flash_model=flash_model)

    def _make_hw_models(self, device, array_size):
        hw_config = self.hw_config
        temperature = torch.as_tensor(self.temperature, dtype=torch.float64)
        # This function and _init_temp_codes must use the same parameters (temperature, pFSR, AIDAC gain).
        w_aidac_model = ReferenceAIDACModel(device=device, num_dacs=array_size[0], temperature=temperature,
                                            vref_temp_code=self.vref_temp_code, pFSR=self.pFSR)
        w_aidac_model = _randomize_hw_model(w_aidac_model, self.aidac_model_nonidealities)
        b_aidac_model = ReferenceAIDACModel(device=device, num_dacs=hw_config.bias_rows, temperature=temperature,
                                            vref_temp_code=self.vref_temp_code, pFSR=self.pFSR)
        b_aidac_model = _randomize_hw_model(b_aidac_model, self.aidac_model_nonidealities)
        w_flash_model = ReferenceFlashCellModel(device=device, array_size=array_size, pFSR=self.pFSR,
                                                temperature=temperature,
                                                flash_params=self.flash_model_name, veg_code=self.veg_code)
        w_flash_model = _randomize_hw_model(w_flash_model, self.flash_model_nonidealities)
        b_flash_model = ReferenceFlashCellModel(device=device, array_size=(hw_config.bias_rows, array_size[1]),
                                                pFSR=self.pFSR, temperature=temperature,
                                                flash_params=self.flash_model_name,
                                                veg_code=self.veg_code)
        b_flash_model = _randomize_hw_model(b_flash_model, self.flash_model_nonidealities)

        return (w_aidac_model, b_aidac_model), (w_flash_model, b_flash_model)

    def configure_nonidealities(self, aidac_model_nonidealities, flash_model_nonidealities, adc_model_nonidealities):
        """Configure nonidealities for AIDAC and Flash models."""
        self.aidac_model_nonidealities = aidac_model_nonidealities
        self.flash_model_nonidealities = flash_model_nonidealities
        self.aidac_model = tuple(_randomize_hw_model(m, self.aidac_model_nonidealities) for m in self.aidac_model)
        self.flash_model = tuple(_randomize_hw_model(m, self.flash_model_nonidealities) for m in self.flash_model)
        self.adc_model.configure_nonidealities(adc_model_nonidealities)

    def _get_hw_models(self, device, array_size):
        if self.update_models_every_time:
            return self._make_hw_models(device, array_size)

        if self.aidac_model is None:
            self.aidac_model, self.flash_model = self._make_hw_models(device, array_size)
        elif self.aidac_model[0].device != device or self.flash_model[0].device != device:
            self.aidac_model = tuple(replace(m, device=device) for m in self.aidac_model)
            self.flash_model = tuple(replace(m, device=device) for m in self.flash_model)
        return self.aidac_model, self.flash_model

    def _get_chunk_size(self, weight):
        # This converts .self_chunk_size, which is in memory size-ish units, to a batch size for flash cell model.
        return math.ceil(self.flash_model_chunk_size / (weight.numel() * 4))

    def _get_adjust_models_for_inference(self):
        adjust_temperature = self.temperature != self.inference_temperature
        adjust_veg = self.veg_code != self.inference_veg
        adjust_aidac_gain = self.aidac_gain != self.inference_aidac_gain
        if not (adjust_temperature or adjust_veg or adjust_aidac_gain):
            return None

        def adjust_models_for_inference(aidac_models, flash_models):
            aidac_kwargs = {}
            flash_kwargs = {}
            if adjust_temperature:
                aidac_kwargs['temperature'] = torch.tensor(self.inference_temperature, dtype=torch.float64)
                flash_kwargs['temperature'] = aidac_kwargs['temperature']
            if adjust_veg:
                flash_kwargs['veg_code'] = self.inference_veg
            if adjust_aidac_gain:
                aidac_kwargs['aidac_gain'] = self.inference_aidac_gain
            aidac_models = tuple(replace(m, **aidac_kwargs) for m in aidac_models) if aidac_kwargs else aidac_models
            flash_models = tuple(replace(m, **flash_kwargs) for m in flash_models) if flash_kwargs else flash_models
            return aidac_models, flash_models

        return adjust_models_for_inference

    def _linear_mma_func(self, x, weight, bias, compute_common_mode_current):
        aidac_model, flash_model = self._get_hw_models(x.device, tuple(reversed(weight.shape)))
        linear = make_linear(weight, bias, compute_common_mode_current=compute_common_mode_current,
                             aidac_model=aidac_model, flash_model=flash_model, chunk_size=self._get_chunk_size(weight),
                             adjust_models_for_inference=self._get_adjust_models_for_inference())
        return linear(x)

    def _conv_mma_func(self, x, weight, bias, mma_func, compute_common_mode_current):
        aidac_model, flash_model = self._get_hw_models(x.device, (math.prod(weight.shape[1:]), weight.shape[0]))

        conv2d = make_conv2d(weight, bias,
                             pad=mma_func.keywords['padding'],
                             stride=mma_func.keywords['stride'],
                             group=mma_func.keywords['groups'],
                             dilation=mma_func.keywords['dilation'],
                             compute_common_mode_current=compute_common_mode_current,
                             aidac_model=aidac_model, flash_model=flash_model,
                             chunk_size=self._get_chunk_size(weight),
                             adjust_models_for_inference=self._get_adjust_models_for_inference())
        return conv2d(x)

    def _get_denali_reference_model_mma_func(self, mma_func, compute_common_mode_current):
        """Return a Denali reference model-based replacement for the provided mma_func (F.Conv2d or F.linear).

        As the model is not separable, we have to use a hardware model provided implementation of `dot` instead of
        Pytorch's Conv/Linear.
        """
        if isinstance(mma_func, partial) and mma_func.func == F.conv2d:
            return partial(self._conv_mma_func, mma_func=mma_func,
                           compute_common_mode_current=compute_common_mode_current)
        elif mma_func == F.linear:
            return partial(self._linear_mma_func, compute_common_mode_current=compute_common_mode_current)
        else:
            raise ValueError(f"Can't adjust the model to the provided mma_func {mma_func}")

    def get_hardware_parameters(self):
        """Return hardware parameters for AIDAC, Flash, and ADC models."""
        return {
            # Combine weight and bias models into one.
            'aidac_model': _hw_parameters_as_dict(concat_aidacs(*self.aidac_model)),
            # Combine weight and bias models into one. We always use one bank.
            'flash_models': [_hw_parameters_as_dict(merge_flash_arrays(*self.flash_model))],
            'adc_model': self.adc_model.get_hardware_parameters(),
        }

    def set_hardware_parameters(self, params):
        """Update hardware parameters of AIDAC, Flash, and ADC models using dataclasses.replace."""
        assert self.aidac_model and self.flash_model
        # Split the model into a weight and a bias models.
        expected_num_dacs = self.aidac_model[0].num_dacs + self.aidac_model[1].num_dacs
        aidac_model = replace(self.aidac_model[0], **params['aidac_model'])
        assert aidac_model.num_dacs == expected_num_dacs, \
            f"Expected {expected_num_dacs} DACs, got {aidac_model.num_dacs}"
        self.aidac_model = (select_aidacs(aidac_model, slice(self.aidac_model[0].num_dacs)),
                            select_aidacs(aidac_model, slice(self.aidac_model[0].num_dacs, None)))
        # Split the model into a weight and a bias models. We always use bank 0.
        expected_array_size = ((self.flash_model[0].array_size[0] + self.flash_model[1].array_size[0],)
                               + self.flash_model[0].array_size[1:])
        flash_model = replace(self.flash_model[0], **params['flash_models'][0])
        assert flash_model.array_size == expected_array_size, \
            f"Expected array size {expected_array_size}, got {flash_model.array_size}"
        self.flash_model = (select_flash_array_rows(flash_model, slice(self.flash_model[0].array_size[0])),
                            select_flash_array_rows(flash_model, slice(self.flash_model[0].array_size[0], None)))
        self.adc_model.set_hardware_parameters(params['adc_model'])


make_denali_reference_model = partial(make_analog_model, model_class=DenaliReferenceModel)
