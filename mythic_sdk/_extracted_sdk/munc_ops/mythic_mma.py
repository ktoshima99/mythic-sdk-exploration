"""A superclass of MythicLinear and MythicConv2d."""
import logging

import numpy as np
import torch
import torch.nn as nn

from munc._pytorch.layers import clip, floor_ste, retrieve_activation_function, quantize_dsf, quantize_fsr
from munc._o2t_module import O2TModule
from munc._node_utils import is_node_signed
from munc._session_tools import apply_momentum, calculate_digital_scale_factors
from munc._ace_model import BaseAnalogModel
from munc.hw_specs import get_hw_config


logger = logging.getLogger(__name__)


class MythicMMA(O2TModule):
    """A superclass of MythicLinear and MythicConv2d.

    This module implements the TorchNet interface and manages the training parameters of a Mythic MMA layer.
    Actual computations and noise modeling are delegated to an instance of an AnalogModel subclass that is created
    using `make_analog_model`.
    MythicLinear and MythicConv2d are almost identical, they only differ by a pytorch function(F.linear vs F.conv2d).
    The subclasses parametrize this class by an appropriate function (attribute `mma_func`).
    """

    arg_check = (2, 4)

    model_multicycle_processing = False

    def __init__(self, node, o2t,
                 make_analog_model,
                 quantize_activation=True,
                 clip_activation=True,
                 quantize_dsf=False,
                 # The noise_scale multiplies all noises and can be used to change noise by a schedule.
                 noise_scale=1,
                 weight_scale_trainable=False,
                 ifsr_trainable=False,
                 min_ifsr=None,
                 max_ifsr=None,
                 global_temp_module=None):
        super().__init__(node, o2t)
        self.global_temp_module = global_temp_module
        self.activation_func = retrieve_activation_function(node)
        self._init_mma_func()
        self.hw_config = get_hw_config(node)

        self.analog_model = make_analog_model(hw_config=self.hw_config, pFSR=self.pFSR,
                                              signed_input=is_node_signed(node),
                                              mma_func=self.mma_func)
        # The noise_scale multiplies all noises and can be used to change noise by a schedule.
        self.analog_model.noise_scale = noise_scale
        self.noise_config = self.analog_model.noise_config
        self.quantize_activation = quantize_activation
        self.clip_activation = clip_activation
        self.quantize_dsf = quantize_dsf
        # A function to compute an optimal weight scale. It takes a layer (an instance of this class),
        # inputs, weights, and biases.
        self.compute_weight_scale = None
        self.weight_scale_momentum = 0.99
        self.weight_scale_trainable = weight_scale_trainable
        self.ifsr_trainable = ifsr_trainable
        self.min_ifsr = min_ifsr or (min(self.noise_config.half_iFSR_arr) * 2)
        self.max_ifsr = max_ifsr or (max(self.noise_config.half_iFSR_arr) * 2)
        self.register_buffer('available_fsrs', torch.tensor(self.noise_config.half_iFSR_arr) * 2)

        self.node_name = node.name
        self.per_channel_dsf = isinstance(self.multiplier, np.ndarray) and self.multiplier.ndim > 0
        dsf = self._compute_initial_dsf()
        if self.trainable_dsf:
            self.dsf = nn.Parameter(dsf)
        else:
            self.register_buffer('dsf', dsf)

        # Register `weight_scale` as a buffer to let pytorch know it needs to save it and synchronize it between
        # processes.
        weight_scale = self.weight_scale
        if not isinstance(weight_scale, torch.Tensor):
            weight_scale = torch.from_numpy(np.array(weight_scale))
        del self.weight_scale
        if self.weight_scale_trainable:
            self.weight_scale = nn.Parameter(weight_scale)
        else:
            self.register_buffer('weight_scale', weight_scale)

        self.iFSR = torch.tensor(self.iFSR, dtype=torch.float32)
        if self.ifsr_trainable:
            self.iFSR = nn.Parameter(self.iFSR)
        else:
            # Check that the initial iFSR is valid. It can be a scalar or a vector.
            ifsr_list = [self.iFSR] if self.iFSR.numel() == 1 else self.iFSR
            if any(all(abs(ifsr / 2 - x) > 0.01 for x in self.noise_config.half_iFSR_arr) for ifsr in ifsr_list):
                raise ValueError(f'iFSR/2={self.iFSR / 2} is not in the list of allowed half iFSR values: '
                                 f'{self.noise_config.half_iFSR_arr}')

        self.ideal_model = BaseAnalogModel(signed_input=False, mma_func=self.mma_func, pFSR=self.pFSR,
                                           hw_config=self.hw_config, noise_config=None)
        self.ideal_model.noise_scale = 0.0

    def clipped_ifsr(self):
        return (quantize_fsr(clip(self.iFSR, self.min_ifsr, self.max_ifsr), self.available_fsrs)
                if self.ifsr_trainable else self.iFSR)

    def _compute_ideal_output(self, X, weight, bias):
        acc = self.ideal_model(X, weight * self.weight_scale, bias * self.weight_scale, global_temp=0,
                               iFSR=self.clipped_ifsr())
        return self.apply_activation(acc * self.effective_digital_scale(acc.shape), self.clip_activation)

    def _layer_op(self, X, weight, bias=None, global_temp=0):
        if bias is None:
            bias = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)

        global_temp = self.global_temp_module() if self.global_temp_module is not None else 0
        global_temp = global_temp * self.analog_model.noise_scale
        if isinstance(global_temp, torch.Tensor):
            global_temp = global_temp.to(dtype=weight.dtype)

        if self.compute_weight_scale and self.training:
            with torch.no_grad():
                weight_scale = self.compute_weight_scale(self, X, weight, bias)
                weight_scale = apply_momentum(weight_scale, self.weight_scale, self.weight_scale_momentum)
                self.weight_scale = weight_scale

        # `self.weight_scale` is 1.0 unless `self.compute_weight_scale` or `self.weight_scale_trainable` is enabled.
        weight = weight * self.weight_scale
        bias = bias * self.weight_scale

        z = self.analog_model(X, weight, bias, global_temp=global_temp, iFSR=self.clipped_ifsr())
        z = self.effective_digital_scale(z.shape) * z
        z = floor_ste(z) if self.quantize_activation else z
        return self.apply_activation(z, self.clip_activation)

    def _compute_initial_dsf(self):
        """Return the initial value of the training parameter representing digital scaling.

        A digital scaling factor stored in ONNX nodes (as multiplier and a shift) does not have to be directly used as
        a training parameter. Instead this function computes the initial training digital scale value from the
        multiplier, the shift, and other node parameters. `self.effective_digital_scale` does the opposite - it computes
        the effective digital scale from training parameters. This allows us to experiment with training parameters
        without changing node attributes. The current implementation of training parameters is based on the following
        formula:
        ADC((W * weight_scale) * X / ifsr) * dsf / (weight_scale / ifsr)
        to ensure `weight_scale` and `ifsr` only control noise, and don't affect the output scale of the layer.
        """
        # In the computation of the ADC output we multiply weights by `weight_scale` and divide by `iFSR`.
        # To prevent this scaling from changing the output scale of the layer the effective digital scale includes
        # `1 / (weight_scale / iFSR)`. To make the initial effective digital scale the same as the DSF specified
        # in the node attributes we need to multiply it by `(weight_scale / iFSR)`.
        # Scaling by iFSR is disabled for now. If you want to enable it, don't forget to update
        # `effective_digital_scale` too.
        # dsf = torch.as_tensor(...) * (self.weight_scale / self.iFSR)
        dsf = torch.as_tensor(self.multiplier / 2 ** self.shift, dtype=torch.float32) * self.weight_scale
        return dsf

    def effective_digital_scale(self, expected_shape=None):
        """Return the effective digital scale that is going to be used on hardware.

        See `_compute_initial_dsf()` for the opposite operation.
        """
        bounds = self.noise_config.ds_trainable_range
        # Scaling by iFSR is disabled for now. If you want to enable it, don't forget to update
        # `_compute_initial_dsf` too.
        # dsf = torch.clamp(self.dsf / (self.weight_scale / self.clipped_ifsr()), bounds[0], bounds[1])
        dsf = torch.clamp(self.dsf / self.weight_scale, bounds[0], bounds[1])
        dsf = quantize_dsf(dsf, self.hw_config) if self.quantize_dsf else dsf
        # Make the shape compatible (broadcastable) with the expected output shape if needed.
        if expected_shape is not None and dsf.ndim > 0:
            shape = [1] * len(expected_shape)
            shape[1] = -1
            dsf = dsf.view(shape)
        return dsf

    def apply_activation(self, val, clip_activation):
        """Apply activation function to the value and clip it if activation clipping is enabled."""
        if self.activation_func is not None:
            val = self.activation_func(val)
        if not self.activation_clip:
            return val
        elif clip_activation:
            return torch.clamp(val, *self.activation_clip)
        else:
            return torch.clamp(val, 0) if self.activation_clip[0] == 0 else val

    def update_back_to_onnx(self):
        if self.trainable_dsf:
            digital_scale = self.effective_digital_scale().detach().cpu().numpy()
            # logger.info(f'{self.node_name}: in update_back_to_onnx dsf = {digital_scale}, param = {self.dsf.item()}')
            multiplier, divider = calculate_digital_scale_factors(digital_scale, self.hw_config.ds_max_mult,
                                                                  self.hw_config.ds_max_shift)
            self.multiplier = multiplier
            self.shift = np.round(np.log2(divider)).astype(int)

        return {
            '__iFSR': self.clipped_ifsr().item(),
            '__pFSR': self.pFSR,
            '__multiplier': self.multiplier if self.per_channel_dsf else int(self.multiplier),
            '__shift': self.shift if self.per_channel_dsf else int(self.shift),
            '__weight_scale': self.weight_scale.detach().cpu().numpy()
        }

    def update_parameters_back_to_onnx(self, model, node):
        # Return the number of trainable parameters.
        return len(list(self.parameters()))

    def get_group_parameters(self, group):
        return (list(self.parameters()) if group == self.trainable_dsf else [])
