"""Denali separable (training) model wrappers for MythicMMA.

This module does not depend on the reference model (internal package).
"""
from dataclasses import replace, asdict
from functools import partial

import torch
from funcy import omit, project

from mythic.acm.denali.training.polynomial_separable_model import FSR_UNIT
from mythic.acm.denali.training.approximate_adc_model import ApproximateADCModel
from munc._session_tools import split_bias
from munc._constants import BiasSplittingMethod
from munc._ace_model import (BaseAnalogModel, make_analog_model,
                             BaseInputModel, BaseWeightModel, BaseADCModel)
from mythic.acm.denali.training import polynomial_separable_model
from munc._pytorch import noise


# TODO: Move the programming error parameters to acm-m2000.
# Additive weight programming error in the same units as weights at pFSR=1 (hw_config.weight_min/max), i.e.
# this number should be divided by pFSR before it's applied to weights.
# ([0.5nA, 1.5nA, 3nA] / FSR_UNIT) * hw_config.weight_max -> [0.63, 1.92, 3.84].
WEIGHT_ADDITIVE_NOISE_SIGMA = 1.92  # 1.5nA
# Proportional weight programming error
WEIGHT_PROPORTIONAL_NOISE_SIGMA = 0.0


def _hw_parameters_as_dict(hw_model):
    return omit(asdict(hw_model), 'device')


def _randomize_hw_model(hw_model, nonidealities):
    return hw_model.randomize(**omit(nonidealities, 'enable')) if nonidealities.get('enable') else hw_model


class BaseDenaliADCModel(BaseADCModel):
    """A base class of Denali ADC models.

    It's an adapter between the Denali ADC models and the MythicMMA interface.
    """

    def __init__(self, hw_config, mma_func, pFSR, num_fractional_bits=0, clip_dot=True, quantize_dot=True,
                 model_common_mode=True, nonidealities={}):
        super().__init__(hw_config=hw_config, mma_func=mma_func, pFSR=pFSR,
                         num_fractional_bits=num_fractional_bits, clip_dot=clip_dot, quantize_dot=quantize_dot)
        self.model_common_mode = model_common_mode
        self.nonidealities = nonidealities
        assert self.num_fractional_bits == 0, "Denali ADC doesn't support fractional bits."
        self._update_models_every_time = True
        self.hw_model = None

    def _make_hw_model(self, fsr, num_adcs, device):
        raise NotImplementedError("_make_hw_model must be implemented in a subclass.")

    def forward(self, z, X, weight, bias, iFSR):  # noqa: D102
        if self.model_common_mode:
            input_current, common_mode_current = z
        else:
            input_current = z
            common_mode_current = torch.zeros(1, dtype=input_current.dtype)
        hw_model = self._get_hw_model(fsr=iFSR, num_adcs=input_current.shape[1], device=z.device)
        # ReferenceADCModel expects input in nA.
        # The Denal ADC models do not support fp16 (yet?).
        z = hw_model.convert(input_current.to(dtype=torch.float32) * FSR_UNIT * self.pFSR,
                             common_mode_current.to(dtype=torch.float32) * FSR_UNIT * self.pFSR)
        return z.to(dtype=input_current.dtype)

    def _get_hw_model(self, fsr, num_adcs, device):
        if self.update_models_every_time:
            return self._make_hw_model(fsr, num_adcs, device)

        if self.hw_model is None:
            self.hw_model = self._make_hw_model(fsr, num_adcs, device)
        elif self.hw_model.fsr != fsr:
            self.hw_model = replace(self.hw_model, fsr=fsr)
        if self.hw_model.device != device:
            self.hw_model = replace(self.hw_model, device=device)
        return self.hw_model

    @property
    def update_models_every_time(self):
        """Whether to create new hardware models every time the MMA function is called."""
        return self._update_models_every_time

    @update_models_every_time.setter
    def update_models_every_time(self, value):
        """Set whether to create new hardware models every time the MMA function is called."""
        self._update_models_every_time = value
        if value:
            self.reset_hardware_parameters()

    def get_hardware_parameters(self):
        """Return ADC hardware dataclass parameters as a dict, omitting device."""
        return _hw_parameters_as_dict(self.hw_model)

    def set_hardware_parameters(self, params):
        """Replace the underlying ADC hardware model using the provided parameters."""
        self.hw_model = replace(self.hw_model, **params)

    def reset_hardware_parameters(self):
        """Reset the underlying ADC hardware model to force creating a new instance next time the model is used."""
        self.hw_model = None

    def configure_nonidealities(self, nonidealities):
        """Configure nonidealities for the ADC model."""
        self.nonidealities = nonidealities
        self.hw_model = _randomize_hw_model(self.hw_model, nonidealities)


class DenaliInputModel(BaseInputModel):
    """A Denali input model.

    This model is a part of the Denali polynomial separable model.
    """

    def __init__(self, hw_config, pFSR, signed_input, duplicate_weight=False, flash_model=None, nonidealities={}):
        super().__init__(hw_config=hw_config, signed_input=signed_input,
                         duplicate_weight=duplicate_weight, pFSR=pFSR)
        self.nonidealities = nonidealities
        self.flash_model = flash_model
        self.pFSR = pFSR
        self.hw_model = None
        self._update_models_every_time = True
        self.hw_config = hw_config
        self.temperature = torch.tensor(20.0, dtype=torch.float64)
        self.vref_temp_code = polynomial_separable_model.find_temp_code(self.temperature.item(), pFSR=self.pFSR,
                                                                        model_data=flash_model)

    def _make_hw_model(self, device):
        hw_model = polynomial_separable_model.InputModel(pFSR=self.pFSR, model_data=self.flash_model, device=device,
                                                         vref_temp_code=self.vref_temp_code)
        assert hw_model.temperature == self.temperature
        return _randomize_hw_model(hw_model, self.nonidealities)

    def _get_hw_model(self, device):
        if self.update_models_every_time:
            return self._make_hw_model(device)

        if self.hw_model is None:
            self.hw_model = self._make_hw_model(device)
        elif self.hw_model.device != device:
            self.hw_model = replace(self.hw_model, device=device)
        return self.hw_model

    @property
    def update_models_every_time(self):
        """Whether to create new hardware models every time the MMA function is called."""
        return self._update_models_every_time

    @update_models_every_time.setter
    def update_models_every_time(self, value):
        """Set whether to create new hardware models every time the MMA function is called."""
        self._update_models_every_time = value
        if value:
            self.reset_hardware_parameters()

    def reset_hardware_parameters(self):
        """Reset the underlying ADC hardware model to force creating a new instance next time the model is used."""
        self.hw_model = None

    def forward(self, X, global_temp):  # noqa: D102
        X = super().forward(X, global_temp)
        hw_model = self._get_hw_model(X.device)
        hw_config = self.hw_config
        X_norm = 2 ** hw_config.input_bits - 1
        # * X_norm is needed here, because _apply_mma_func normalizes by X_norm.
        return hw_model(X) * X_norm

    def get_hardware_parameters(self):
        """Return Input hardware dataclass parameters as a dict, omitting device."""
        return _hw_parameters_as_dict(self.hw_model)

    def set_hardware_parameters(self, params):
        """Replace the underlying Input hardware model using the provided parameters."""
        self.hw_model = replace(self.hw_model, **params)

    def configure_nonidealities(self, nonidealities):
        """Configure InputModel nonidealities."""
        self.nonidealities = nonidealities
        self.hw_model = _randomize_hw_model(self.hw_model, nonidealities)


class DenaliBaseWeightModel(BaseWeightModel):
    """A base class for Denali weight models.

    This model implements weight programming errors.
    """

    def __init__(self, hw_config, pFSR, clip_weight=True, quantize_weight=True,
                 duplicate_weight=False, signed_input=False, nonidealities={}):
        super().__init__(hw_config=hw_config, pFSR=pFSR,
                         clip_weight=clip_weight, quantize_weight=quantize_weight, duplicate_weight=duplicate_weight,
                         signed_input=signed_input)
        programming_keys = ['weight_additive_noise', 'weight_proportional_noise', 'weight_noise_back_prop']
        self.nonidealities = omit(nonidealities, programming_keys)
        self.programming_nonidealities = project(nonidealities, programming_keys + ['enable'])
        self.pFSR = pFSR
        self._update_models_every_time = True

    @property
    def update_models_every_time(self):
        """Whether to create new hardware models every time the MMA function is called."""
        return self._update_models_every_time

    @update_models_every_time.setter
    def update_models_every_time(self, value):
        """Set whether to create new hardware models every time the MMA function is called."""
        self._update_models_every_time = value

    def forward(self, weight, bias, global_temp):  # noqa: D102
        weight, bias = super().forward(weight, bias, global_temp)
        if self.update_models_every_time:
            def add_errors(weight, adjust_for_bias_split=False):
                return apply_programming_errors(weight, programming_nonidealities=self.programming_nonidealities,
                                                pFSR=self.pFSR, hw_config=self.hw_config,
                                                adjust_for_bias_split=adjust_for_bias_split)
            return add_errors(weight), add_errors(bias, True)
        else:
            # In this case we assume that programming errors are applied by a caller at the desired frequency.
            return weight, bias


def apply_programming_errors(weight, programming_nonidealities, pFSR, hw_config, adjust_for_bias_split=False):
    """Apply programming errors to the provided weight tensor."""
    def param(param_name, default):
        return programming_nonidealities.get(param_name, default)

    if param('enable', False):
        additive_noise = param('weight_additive_noise', WEIGHT_ADDITIVE_NOISE_SIGMA) / pFSR
        proportional_noise = param('weight_proportional_noise', WEIGHT_PROPORTIONAL_NOISE_SIGMA)
        # FIXME: Bias is not split, so both proportional and additive noise computations for it must take it
        # into account. A simple but not exact solution used here assumes greedy splitting.
        if adjust_for_bias_split:
            m = torch.sqrt(1 + weight.abs() / hw_config.weight_max)
            additive_noise = additive_noise * m
            proportional_noise = proportional_noise / m
        noisy_weight = weight
        noisy_weight = noise.weight_noise(noisy_weight, additive_noise=additive_noise,
                                          mult_sigma=proportional_noise,
                                          ste=not param('weight_noise_back_prop', False))
        noisy_weight = torch.where(weight != 0, noisy_weight, weight)
        # At this point, noisy_weight may exceed the nominal range. This is how programming errors work.
        return noisy_weight
    else:
        return weight


def default_weight_randomizer(node, programming_nonidealities):
    """Return weight/bias initializers with programming errors applied.

    This function can be used as a weight_randomizer in the randomization schedule to apply programming errors.
    See `munc._monte_carlo.chip_instance_generator.random_model_instances`.

    Parameters
    ----------
    node : onnx.NodeProto
        The ONNX node representing a layer with weights and bias.

    Returns
    -------
    dict
        A mapping from initializer names to numpy arrays with programming errors applied.
    """
    pFSR = node.attrs['__pFSR']
    hw_config = node.model.hwconfig

    def add_errors(weight, adjust_for_bias_split=False):
        return apply_programming_errors(torch.as_tensor(weight.copy()),
                                        programming_nonidealities=programming_nonidealities,
                                        pFSR=pFSR, hw_config=hw_config,
                                        adjust_for_bias_split=adjust_for_bias_split).numpy()

    initializers = {node.input[1]: add_errors(node.initializer[1])}
    bias = node.initializer[2] if len(node.input) > 2 else None
    if bias is not None:
        initializers[node.input[2]] = add_errors(bias, True)
    return initializers


class DenaliWeightModel(DenaliBaseWeightModel):
    """A Denali weight model.

    This model is a part of the Denali polynomial separable model.
    """

    def __init__(self, hw_config, pFSR, clip_weight=True, quantize_weight=True,
                 duplicate_weight=False, signed_input=False, flash_model=None, nonidealities={}):
        super().__init__(hw_config=hw_config, pFSR=pFSR,
                         clip_weight=clip_weight, quantize_weight=quantize_weight, duplicate_weight=duplicate_weight,
                         signed_input=signed_input, nonidealities=nonidealities)
        self.flash_model = flash_model
        self.weight_shape = None
        self.hw_weight_model = None
        self.hw_bias_model = None

    @DenaliBaseWeightModel.update_models_every_time.setter
    def update_models_every_time(self, value):
        """Set whether to create new hardware models every time the MMA function is called."""
        DenaliBaseWeightModel.update_models_every_time.fset(self, value)
        if value:
            self.reset_hardware_parameters()

    def reset_hardware_parameters(self):
        """Reset the underlying ADC hardware model to force creating a new instance next time the model is used."""
        self.hw_weight_model = None
        self.hw_bias_model = None

    def _make_hw_models(self, weight_shape, device):
        hw_weight_model = polynomial_separable_model.WeightModel(model_data=self.flash_model, device=device,
                                                                 shape=weight_shape, pFSR=self.pFSR)
        hw_bias_model = polynomial_separable_model.BiasModel(model_data=self.flash_model, device=device,
                                                             pFSR=self.pFSR,
                                                             shape=(weight_shape[0], self.hw_config.bias_rows))
        hw_weight_model = _randomize_hw_model(hw_weight_model, self.nonidealities)
        hw_bias_model = _randomize_hw_model(hw_bias_model, self.nonidealities)
        return hw_weight_model, hw_bias_model

    def _get_hw_models(self, weight_shape, device):
        if self.update_models_every_time:
            return self._make_hw_models(weight_shape, device)

        if self.hw_weight_model is None:
            #  TODO: Do we need `or self.hw_weight_model.shape != weight_shape` in the condition above?
            self.hw_weight_model, self.hw_bias_model = self._make_hw_models(weight_shape, device)
        elif self.hw_weight_model.device != device:
            self.hw_weight_model = replace(self.hw_weight_model, device=device)
            self.hw_bias_model = replace(self.hw_bias_model, device=device)
        return self.hw_weight_model, self.hw_bias_model

    def forward(self, weight, bias, global_temp):  # noqa: D102
        weight, bias = super().forward(weight, bias, global_temp)
        hw_config = self.hw_config
        W_norm = max(abs(hw_config.weight_min), abs(hw_config.weight_max))
        # Split bias into rows.
        bias_rows = split_bias(bias, num_bias_splits=hw_config.bias_rows, max_abs_weight=W_norm,
                               method=BiasSplittingMethod.OVERFLOW, use_fp=True)

        hw_weight_model, hw_bias_model = self._get_hw_models(weight.shape, weight.device)
        # TODO: Does it make sense to change the Denali model to use [-W_norm, W_norm] range instead of [-1, 1]?
        weight_model_output = hw_weight_model(weight / W_norm) * W_norm
        bias_model_output = hw_bias_model(bias_rows / W_norm).sum(-1) * W_norm
        return weight_model_output, bias_model_output

    def get_hardware_parameters(self):
        """Return Weight/Bias hardware dataclass parameters as a dict, omitting device."""
        return {
            'weight': _hw_parameters_as_dict(self.hw_weight_model),
            'bias': _hw_parameters_as_dict(self.hw_bias_model),
        }

    def set_hardware_parameters(self, params):
        """Replace the underlying Weight and Bias hardware models using the provided parameters."""
        self.hw_weight_model = replace(self.hw_weight_model, **params['weight'])
        self.hw_bias_model = replace(self.hw_bias_model, **params['bias'])

    def configure_nonidealities(self, nonidealities):
        """Configure Weight/BiasModel nonidealities."""
        self.nonidealities = nonidealities
        self.hw_weight_model = _randomize_hw_model(self.hw_weight_model, self.nonidealities)
        self.hw_bias_model = _randomize_hw_model(self.hw_bias_model, self.nonidealities)


class DenaliADCModel(BaseDenaliADCModel):
    """A BaseADCModel wrapper around the approximate Denali ADC model.

    It implements the interface MythicMMA expects from hardware models.
    """

    def _make_hw_model(self, fsr, num_adcs, device):
        hw_model = ApproximateADCModel(fsr=fsr, num_adcs=num_adcs, device=device, clip_output=self.clip_dot,
                                       quantize_output=self.quantize_dot)
        hw_model = _randomize_hw_model(hw_model, self.nonidealities)
        return hw_model


class DenaliSeparableModel(BaseAnalogModel):
    """An AnalogModel wrapper around a Denali polynomial separable model.

    It implements the interface MythicMMA expects from hardware models.
    """

    def __init__(self, signed_input, mma_func, pFSR, hw_config, weight_model, input_model, adc_model, noise_config):
        mma_func = partial(self._mma_func_wrapper, mma_func=mma_func)
        super().__init__(signed_input=signed_input, mma_func=mma_func, pFSR=pFSR,
                         hw_config=hw_config, weight_model=weight_model, input_model=input_model,
                         adc_model=adc_model, noise_config=noise_config)
        # TODO: Make the offset model configurable similarly to WeightModel and InputModel.
        offset_model = polynomial_separable_model.OffsetModel(model_data=self.weight_model.flash_model, pFSR=pFSR)
        self.offset = offset_model.offset
        self._update_models_every_time = True
        self.pFSR = pFSR

    @property
    def update_models_every_time(self):
        """Whether to create new hardware models every time the MMA function is called."""
        return self._update_models_every_time

    @update_models_every_time.setter
    def update_models_every_time(self, value):
        """Set whether to create new hardware models every time the MMA function is called."""
        self._update_models_every_time = value
        self.input_model.update_models_every_time = value
        self.weight_model.update_models_every_time = value
        self.adc_model.update_models_every_time = value

    def reset_hardware_parameters(self):
        """Reset the underlying ADC hardware model to force creating a new instance next time the model is used."""
        self.input_model.reset_hardware_parameters()
        self.weight_model.reset_hardware_parameters()
        self.adc_model.reset_hardware_parameters()

    def _mma_func_wrapper(self, input_model_output, weight_model_output, bias_model_output, mma_func):
        model_common_mode = self.noise_config.model_common_mode
        return polynomial_separable_model.compute_model_output(input_model_output, weight_model_output,
                                                               bias_model_output, self.offset, mma_func, pFSR=self.pFSR,
                                                               compute_common_mode_current=model_common_mode)

    def get_hardware_parameters(self):
        """Return grouped hardware parameters for input, weight, and ADC models."""
        return {
            'input_model': self.input_model.get_hardware_parameters(),
            'weight_model': self.weight_model.get_hardware_parameters(),
            'adc_model': self.adc_model.get_hardware_parameters(),
        }

    def set_hardware_parameters(self, params):
        """Replace underlying hardware models for input, weight, and ADC using provided parameters."""
        self.input_model.set_hardware_parameters(params['input_model'])
        self.weight_model.set_hardware_parameters(params['weight_model'])
        self.adc_model.set_hardware_parameters(params['adc_model'])

    def configure_nonidealities(self, input_model_nonidealities, weight_model_nonidealities, adc_model_nonidealities):
        """Configure nonidealities for AIDAC, Flash, and ADC models."""
        self.weight_model.configure_nonidealities(weight_model_nonidealities)
        self.input_model.configure_nonidealities(input_model_nonidealities)
        self.adc_model.configure_nonidealities(adc_model_nonidealities)

    def forward(self, X, weight, bias, global_temp, iFSR):  # noqa: D102
        # Make sure computations are done in fp32 to avoid accuracy issues in the separable model. It likely can be
        # tweaked to work in fp16, but it will require more work.
        orig_dtype = weight.dtype
        weight = weight.to(dtype=torch.float32)
        bias = bias.to(dtype=torch.float32)
        X = X.to(dtype=torch.float32)
        res = super().forward(X, weight, bias, global_temp, iFSR)
        return res.to(dtype=orig_dtype)


make_denali_separable_model = partial(make_analog_model, model_class=DenaliSeparableModel)
