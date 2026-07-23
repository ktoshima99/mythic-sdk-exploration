# noqa-flake8-docstrings
import logging
import torch
import torch.nn.functional as F
import copy

from munc.bcm.registry import SimpleAttributes
from munc.bcm.debug_containers import SimpleMMADebugContainer

logger = logging.getLogger(__name__)

FACTORY_NAME = 'munc_simple'


class PytorchSimpleMMA:
    """Pytorch port of the VectorModel:
        - multi-cycle
        - SAR ADC noise
        - Weights noise
    """
    _ALLOWED_FIXABLE_NOISE = (None,)

    def __init__(self, weights, biases, mma_attr=None, pFSR=2.0, iFSR=2.0, name=None, seed=None,
                 weight_scale=128):
        """
        To make Pytorch model computation deterministic and reproducible, see
        https://pytorch.org/docs/stable/notes/randomness.html
        https://pytorch.org/docs/stable/torch.html?highlight=rand#torch.rand
        https://pytorch.org/docs/master/torch.html#generators
        https://github.com/pytorch/pytorch/issues/17079

        *** This will lead to severe speed penalty ***

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        """
        self.name = name
        self.mma_attr = mma_attr
        if mma_attr is None:
            self.mma_attr = SimpleAttributes()
        self.mma_attr = copy.deepcopy(self.mma_attr)
        # We need to know the device of the weights to cast the inputs to the correct device when running multi-gpu
        self.device = weights.device
        self.dtype = weights.dtype

        if not weights.is_cuda and self.dtype is not torch.float32:
            self.dtype = torch.float32
            weights = weights.type(self.dtype)
            biases = biases.type(self.dtype)

        self.iFSR = iFSR
        self.pFSR = pFSR
        self.weight_scale = weight_scale
        self.num_inputs = 1  # need to know this to generate unique noise per input

        self.weights = weights
        self.biases = biases

        self.out_channels = None
        self.iflash_weights = None
        self.iflash_biases = None

        # SAR ADC cycle scaling parameters
        pows2 = torch.logspace(0, 7, 8, 2, device=self.device, dtype=torch.uint8)
        self.bit_scales = pows2.reshape(8, *((1,) * weights.dim()))

        # internal state to manage and monitor the randomization of the noises
        self.mma_base_attr = copy.deepcopy(self.mma_attr)
        self.adc_noise_gen = None
        self.adc_offset_and_inl_gen = None
        self.popcorn_noise_gen = None
        self.mc_w_rand_p = None
        self.mc_w_rand_n = None
        self.mc_b_rand_p = None
        self.mc_b_rand_n = None
        self.adc_noise = None
        self.pop_stats_p = 0
        self.pop_stats_n = 0

        self.randomize()
        self.debug_container = None

    def __str__(self):
        if self.name is None:
            str_name = f"{FACTORY_NAME} mma"
        else:
            str_name = f"{FACTORY_NAME} mma / {self.name}"
        return str_name

    def dump_attrs(self):
        return self.mma_attr.__dict__

    def randomize(self, fix_vals=None, random_state=None):
        if fix_vals is None:
            fix_vals = {}

        if self.out_channels is None:
            self.lazy_randomization_requested = True
            self.random_state = random_state
        else:
            self.randomize_adc_parameters(random_state)

        for key, val in fix_vals.items():
            if key in self._ALLOWED_FIXABLE_NOISE:
                setattr(self, key, val)
            else:
                raise KeyError("You are requesting a noise source to be fixed that cannot be fixed. "
                               f"Allowed values are {self._ALLOWED_FIXABLE_NOISE}")

        self.iflash_weights, self.iflash_biases = self.mod_weights_torch(temp_delta=fix_vals.get('temp_delta', None))

    def randomize_adc_parameters(self, random_state=None):
        self.simple_offset, self.simple_inl = self.get_adc_offset_and_inl()
        # pows2 shape: (8, 1, num_inputs, out_channels)
        pows2 = torch.cat([
            (2 + self.simple_inl) ** x
            for x in torch.arange(2, 10, 1, dtype=torch.float32, device=self.device)]
        )
        # This is an equivalent of 'sar_scale_factor' in bcm repo
        self.sar_bases = (2. * self.iFSR / pows2).type(self.dtype)

    def get_adc_offset_and_inl(self):
        random_normal = torch.normal(
            0, 1, (1, self.num_inputs, 2*self.out_channels), dtype=self.dtype, device=self.device,
            generator=self.adc_offset_and_inl_gen
        )
        # offset shape: (1, 1, num_inputs, out_channels)
        scaled_simple_offset = 10e6 * self.mma_attr.simple_offset
        offset = scaled_simple_offset * random_normal[:, :, 0::2]
        inl = self.mma_attr.simple_inl * random_normal[:, :, 1::2]
        return offset, inl

    def get_adc_noise(self):
        scaled_simple_noise = 10e6 * self.mma_attr.simple_noise
        noise = scaled_simple_noise * torch.normal(0, 1, (8, 8, self.num_inputs, self.out_channels),
                                                   generator=self.adc_noise_gen,
                                                   dtype=self.dtype, device=self.device)
        return noise

    def mod_weights_torch(self, decay_rate=None, decay_hours=None, temp_delta=None,
                          mc_mult_sigma_lsb=None, pop_fraction=None, mc_mult=None,
                          linear_beta0=None, linear_beta1=None):
        """Apply noise models to flash weights

        Sequence of weight noise models applied:
            1) Linear Charge Decay based on decay_rate and decay_hours
            2) Exponential Temperature Change based on temp_delta
            3) Apply Weights Variation (proportional weights error model) based on mc_mult and mc_mult_sigma_lsb
            4) Pop-Corn noise based on pop_fraction, pop_lognorm_mean and pop_lognorm_sigma
            5) Linear noise model based on linear_beta0 and linear_beta1

        Parameters
        -------
        decay_rate : float
            Decay rate for linear charge decay.
        decay_hours : float
            Decay hours for linear charge decay.
        temp_delta : float
            Temperature delta for Exponential Temperature Change.
        mc_mult : float
            Sigma for proportional weights error model.
        mc_mult_sigma_lsb : float
            Sigma lsb for proportional weights error model.
        pop_fraction : float
            Proportion of weights subject to PopCorn noise.
        linear_beta0 : float
            Intercept parameter for linear noise model.
        linear_beta1 : float
            Slope parameter for linear noise model.

        Returns
        -------
        torch.tensor
            Weights after noise models are applied.
        """
        # First Class internal vars: self.decay_rate and self.decay_hours are updated
        # If no specific parameters are passed, Previously stored/passed parameters for mod_weights
        # function are used
        self.decay_rate = decay_rate
        if decay_rate is None:
            self.decay_rate = self.mma_attr.decay_rate
        self.decay_hours = decay_hours
        if decay_hours is None:
            self.decay_hours = self.mma_attr.decay_hours
        self.temp_delta = temp_delta
        if temp_delta is None:
            self.temp_delta = self.mma_attr.temp_delta
        self.mc_mult = mc_mult
        if mc_mult is None:
            self.mc_mult = self.mma_attr.mc_mult
        self.mc_mult_sigma_lsb = mc_mult_sigma_lsb
        if mc_mult_sigma_lsb is None:
            self.mc_mult_sigma_lsb = self.mma_attr.mc_mult_sigma_lsb
        self.pop_fraction = pop_fraction
        if pop_fraction is None:
            self.pop_fraction = self.mma_attr.pop_fraction
        self.linear_beta0 = linear_beta0
        if linear_beta0 is None:
            self.linear_beta0 = self.mma_attr.linear_beta0
        self.linear_beta1 = linear_beta1
        if linear_beta1 is None:
            self.linear_beta1 = self.mma_attr.linear_beta1

        def apply_noise(weights, mc_w_rand_p, mc_w_rand_n):
            # program flash (w=1 corresponds to 100nA flash current)
            iflash_weights = weights * (self.weight_scale / 128.0) * self.pFSR

            # Apply Charge Decay (Linear for now. Expect to be updated later)
            iflash_weights = linear_charge_decay(iflash_weights, self.decay_rate, self.decay_hours)

            # Modify flash values due to temperature effects
            iflash_weights = exponential_temp_change(iflash_weights, self.temp_delta)

            wp = torch.clamp(iflash_weights, min=0)
            wn = torch.clamp(-iflash_weights, min=0)

            # Apply Flash MonteCarlo variation (Proportional weights error model)
            wp = apply_weight_var_flatmc(wp, self.mc_mult, self.mc_mult_sigma_lsb, mc_w_rand_p, pFSR=self.pFSR)
            wn = apply_weight_var_flatmc(wn, self.mc_mult, self.mc_mult_sigma_lsb, mc_w_rand_n, pFSR=self.pFSR)

            # Apply pop-corn noise
            if self.mma_attr.pop_lognorm_sigma == 0:
                popcorn_step_dist_p = popcorn_step_dist_n = 0
            else:
                popcorn_step_dist_p = torch.exp(
                    torch.normal(self.mma_attr.pop_lognorm_mean, self.mma_attr.pop_lognorm_sigma,
                                 generator=self.popcorn_noise_gen,
                                 size=weights.shape, dtype=torch.float32, device=self.device)
                )
                popcorn_step_dist_n = torch.exp(
                    torch.normal(self.mma_attr.pop_lognorm_mean, self.mma_attr.pop_lognorm_sigma,
                                 generator=self.popcorn_noise_gen,
                                 size=weights.shape, dtype=torch.float32, device=self.device)
                )
            wp, self.pop_stats_p = apply_popcorn_noise(wp, popcorn_step_dist_p, self.pop_fraction,
                                                       generator=self.popcorn_noise_gen)
            wn, self.pop_stats_n = apply_popcorn_noise(wn, popcorn_step_dist_n, self.pop_fraction,
                                                       generator=self.popcorn_noise_gen)

            # clip differential weights so they stay positive
            wp = torch.clamp(wp, min=0)
            wn = torch.clamp(wn, min=0)

            iflash_weights = wp - wn

            # Apply linear model variation
            iflash_weights = apply_linear_noise(iflash_weights, self.linear_beta0, self.linear_beta1)

            return iflash_weights

        return (apply_noise(self.weights, self.mc_w_rand_p, self.mc_w_rand_n),
                apply_noise(self.biases, self.mc_b_rand_p, self.mc_b_rand_n).sum(1))

    def adc(self, adc_in):
        """adc_in shape: (8, num_inputs, out_dim, ...)"""
        # Combine all the output dimensions into one to avoid dealing with shape differences in
        # compute_adc_output
        adc_in_reshaped = adc_in.reshape(adc_in.shape[0], adc_in.shape[1], -1)
        self.out_channels = adc_in_reshaped.shape[2]
        if self.lazy_randomization_requested:
            self.lazy_randomization_requested = False
            self.randomize_adc_parameters(self.random_state)

        self.adc_noise = self.get_adc_noise()
        res = compute_adc_output(
            adc_in=adc_in_reshaped,
            adc_noise=self.adc_noise,
            simple_offset=self.simple_offset,
            sar_bases=self.sar_bases,
            group_size=self.num_inputs,
            out_channels=self.out_channels)
        # Restore output dimensions
        return res.reshape(*adc_in.shape)

    def dot(self, uint8_input, dot_op=F.linear):
        """
        uint8_input shape:
          Linear: (batch_size, ∗, in_features)
          Conv2d: (batch_size, in_channels, h, w)
        """

        # Ensure that inputs are in the correct range
        uint8_input = torch.clamp(uint8_input.round(), 0, 255).to(dtype=torch.uint8, device=self.device)
        # Split into bits, the new shape is (bit, batch_size, ...)
        x_bits = uint8_input.unsqueeze(0) * self.bit_scales // 128
        # Combine "bit" and "batch_size" dimensions into one to get the shape `dot_op` expects.
        x_bits_reshaped = x_bits.flatten(0, 1)
        # adc_in: (8, num_inputs, input_size).(input_size, out_channels) --> (8, num_inputs, out_channels)
        adc_in = dot_op(x_bits_reshaped.type(self.dtype), self.iflash_weights, self.iflash_biases)
        # Split the combined dimension back to "bit" and "batch_size" dimensions.
        adc_in_reshaped = adc_in.unflatten(0, (8, uint8_input.shape[0]))
        adc_out = self.adc(adc_in_reshaped)
        acc = (adc_out / self.bit_scales).sum(0).round()
        res = torch.clamp(acc, -256, 255)

        if self.debug_container is not None:
            self.debug_container.iflash_pfsr_weight = self.iflash_weights
            self.debug_container.iflash_pfsr_bias = self.iflash_biases
            self.debug_container.adc_in = adc_in
            self.debug_container.accumulator_out = res

        return res

    def register_forward_debug(self):
        """Register a forward debug container with this node."""
        self.debug_container = SimpleMMADebugContainer()
        return self.debug_container

    def unregister_forward_debug(self):
        """Unregister a forward debug container with this node."""
        self.debug_container = None
        return None

    def set_adc_noise_generator(self, generator):
        self.adc_noise_gen = generator

    def set_popcorn_noise_generator(self, generator):
        self.popcorn_noise_gen = generator

    def set_adc_offset_and_inl_generator(self, generator):
        self.adc_offset_and_inl_gen = generator

    def set_chip_level_adc_offset_and_inl(self, generator):
        random_normal = torch.normal(
            0, 1, (2, 1), dtype=self.dtype, device=self.device,
            generator=generator
        )
        self.mma_attr.simple_offset = self.mma_base_attr.simple_offset * random_normal[0, 0]
        self.mma_attr.simple_inl = self.mma_base_attr.simple_inl * random_normal[1, 0]

    def set_weight_programming_randomization(self, generator):
        def noise(flash):
            return torch.normal(0, 1, size=flash.shape, dtype=self.dtype, device=flash.device, generator=generator)

        self.mc_w_rand_p = noise(self.weights)
        self.mc_w_rand_n = noise(self.weights)
        self.mc_b_rand_p = noise(self.biases)
        self.mc_b_rand_n = noise(self.biases)

    def set_env_variables(self, temp_delta, decay_rate, decay_hours):
        self.mma_attr.decay_rate = decay_rate
        self.mma_attr.decay_hours = decay_hours
        self.mma_attr.temp_delta = temp_delta


def linear_charge_decay(flash, decay_rate, decay_hours):
    return flash * (1 - decay_rate * decay_hours)


def exponential_temp_change(flash, temp_delta):
    if temp_delta == 0:
        return flash
    # This code is sensitive to the floating point precision of flash
    # If flash is 16-bit, exponent will be in 32-bit and then converted back into 16-bit in the return
    exponent = (-5.6064e6 * 1e-7 * torch.abs(flash)).float()
    return flash * ((2.0928e-2 * torch.exp(exponent) * temp_delta) + 1).type(flash.dtype)


def apply_popcorn_noise(flash, popcorn_step_dist, pop_fraction, generator=None):
    if pop_fraction == 0:
        return flash, 0
    else:
        # Binary mask of the effected flash cells
        flash_mask = generate_flash_mask(flash, pop_fraction, generator=generator)
        dtype = flash.dtype
        # popcorn_step_dist is float-32 and thus if flash is 16-bit, it needs to be converted back to 16-bit
        # popcorn_step_dist must be 32-bit because random numbers can be outside the 16-bit range
        flash = flash - flash * popcorn_step_dist * flash_mask
        flash = torch.clamp(flash, min=0).type(dtype)
        return flash, torch.sum(popcorn_step_dist*flash_mask).item()/torch.numel(flash_mask)


def apply_weight_var_flatmc(flash, sigma, sigma_lsb, mc_w_rand, pFSR=None):
    """flash input is expected to be already scaled by pFSR, and its units are 100nA"""
    if pFSR is None:
        raise ValueError("pFSR not defined for weight var sigma specified in lsb")
    if sigma == 0 and sigma_lsb == 0:
        return flash
    else:
        _mask = torch.ones_like(flash)
        _mask[flash == 0] = 0
        if mc_w_rand is None:
            mc_w_rand = torch.normal(0, 1, size=flash.shape, dtype=flash.dtype, device=flash.device)
        flash = (flash + flash * sigma * mc_w_rand * 8 / pFSR
                 + _mask * sigma_lsb * (1.5625 / 100.) * pFSR * 0.5 * mc_w_rand)
        return torch.clamp(flash, min=0)


def generate_flash_mask(flash, pop_fraction, generator=None):
    return torch.empty_like(flash).uniform_(generator=generator) > pop_fraction


def apply_linear_noise(flash, beta0, beta1):
    """Apply linear combination to input flash tensor

    Parameters
    -------
    flash : torch.tensor
        Weights values.
    beta0 : float
        Intercept parameter.
    beta1 : float
        Slope parameter.

    Returns
    -------
    torch.tensor
        Weight values after linear combination is applied.
    """
    return beta0 + beta1*flash


@torch.jit.script
def compute_adc_output(adc_in, adc_noise, simple_offset, sar_bases, group_size: int, out_channels: int):
    """
    Computes ADC output with SAR modelling.

    Parameters
    ----------
    adc_in : Float (8, num_inputs, out_channels)
        ADC input values
    adc_noise : Float (8, 8, group_size, out_channels)
        ADC noise for each cycle
    simple_offset : Float (1, group_size, out_channels)
        ADC input offsets
    sar_bases : Float (8, group_size, out_channels)?
    group_size : int
        Number of random parameter groups (e.g. offsets, noises, etc.)
    out_channels : int
        Number of output channels.

    Returns
    -------
    tensor (8, num_inputs, out_channels)
        ADC output values
    """
    # A data type to hold bits collected by an ADC. Will have to change if ADCs are more than 8 bits.
    adc_out_type = torch.uint8
    # The current approximation of ADC input signal. It gets improved every cycle.
    iref = torch.zeros_like(adc_in)
    adc_in_with_offset = adc_in + simple_offset
    # Accumulated ADC output bits
    adc_out = torch.zeros_like(adc_in, dtype=adc_out_type)
    for cycle in range(8):
        # 1 if the ADC input is above the current approximation (iref), 0 otherwise.
        # This gives up the next bit of ADC output.
        bit_values = torch.ge(adc_in_with_offset + adc_noise[cycle], iref)
        # By how much the reference current changes when the bit is on/off
        # In an ideal and unscaled case this would be a power of 2, 128 on the first
        # cycle, 64 on the second, etc.
        bases = sar_bases[cycle].reshape(1, group_size, out_channels)
        # Improve the input approximation (add/subtract the base if the input is above/below the reference).
        iref = iref + torch.where(bit_values, bases, -bases)
        # Push the bit to the collected output bits.
        adc_out = adc_out << 1
        adc_out = adc_out.bitwise_or(bit_values)
    # Shift the values from [0,255] to [-128,127]. ADCs is bipolar, but collected bits are stored as uint8.
    return (adc_out.to(dtype=adc_in.dtype) - 128)
