# noqa-flake8-docstrings
import torch
import torch.nn.functional as F
import copy
import logging
from math import sqrt
import dataclasses

from munc.bcm.debug_containers import ACMSignOffMMADebugContainer


logger = logging.getLogger(__name__)

FACTORY_NAME = 'munc_acm_signoff'


@dataclasses.dataclass
class NoNoiseACMSignoffAttributes:
    # Zero noise case: set attributes to yield a noiseless scenario. This
    # reduces ACM-Signoff to BCM digital ('munc_digital')
    # MMA.mod_weights_torch() parameters
    linear_beta0: float = 0.0  # Constant linear offset
    linear_beta1: float = 1.0  # Constant linear slope
    per_tile_sigma_weight_prop: float = 0.0  # Per tile proportional noise - TODO:  Implement me!
    sigma_weight_add: float = 0.0  # Additive noise
    sigma_weight_prop: float = 0.0  # Proportional noise
    sigma_weight_sqrt_prop: float = 0.0  # weight ** 1/2 proportional noise - TODO:  Implement me!
    # MMA.dot() parameters
    linear_gamma0: float = 0.0  # Constant linear dot product offset
    linear_gamma1: float = 1.0  # Constant linear dot product slope
    sigma_dot: float = 0.0  # Proportional dot product noise


class ACMSignoffAttributes(NoNoiseACMSignoffAttributes):
    def __init__(self):
        """This implements ACM-S v0.4

        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14089978091/ACM+v0.1
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14148632823/ACM-S+v0.3
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14192476436/ACM-S+v0.4
        Using AI-SE 3Q21 Data collection
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14085128531/3Q21+Data+Collection+Results+Work-In-Progress

        """
        super().__init__(
            # MMA.mod_weights_torch() parameters
            linear_beta0=0.0,
            linear_beta1=0.95,
            sigma_weight_prop=0.16,
            sigma_weight_add=0.0,
            # MMA.dot() parameters
            linear_gamma0=-0.12,
            linear_gamma1=0.96,
            sigma_dot=2.48
        )

    @classmethod
    def no_noise(cls):
        """Zero noise case : set attributes to yield a noiseless scenario.

        This reduces ACM-Signoff to BCM digital ('munc_digital').

        """
        return NoNoiseACMSignoffAttributes()

    @classmethod
    def v0p4(cls):
        return ACMSignoffAttributes()

    @classmethod
    def v0p4_no_std(cls):
        """ACM-S v0.4 parameters, No-STD."""
        return disable_random_noise(cls.v0p4())

    @classmethod
    def v0p5(cls):
        """ACM-S v0.5 parameters based on a Robust Linear Model.

        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14219018251/ACM-S+v0.5#ACM-S-v0.5-Weights-Parameters%3A-Robust-Linear-Model-Weights-Parameters

        Using AI-SE 4Q21 Data collection
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14141489386/4Q21+Data+Collection?focusedTaskId=225

        """
        return NoNoiseACMSignoffAttributes(
            # MMA.mod_weights_torch() parameters
            linear_beta0=0.0,
            linear_beta1=0.96,
            sigma_weight_add=1.85,
            sigma_weight_prop=0.0,
            # MMA.dot() parameters
            linear_gamma0=-0.12,
            linear_gamma1=0.96,
            sigma_dot=2.48,
        )

    @classmethod
    def v0p5_no_std(cls):
        """ACM-S v0.5 parameters, No-STD."""
        return disable_random_noise(cls.v0p5())

    @classmethod
    def v0p8(cls):
        """ACM-S v0.8 parameters.

        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14340227098/ACM-S+v0.8
        """
        sigma_weight_add = 0.637
        sigma_weight_prop = 0.073
        return NoNoiseACMSignoffAttributes(
            # MMA.mod_weights_torch() parameters
            linear_beta0=0.0,
            linear_beta1=1.0,
            per_tile_sigma_weight_prop=0.055,
            sigma_weight_add=sigma_weight_add,
            sigma_weight_prop=sigma_weight_prop,
            sigma_weight_sqrt_prop=sqrt(2 * sigma_weight_add * sigma_weight_prop),
            # MMA.dot() parameters
            linear_gamma0=0.0,
            linear_gamma1=1.0,
            sigma_dot=2.32
        )

    @classmethod
    def v0p8_no_std(cls):
        """ACM-S v0.8 parameters, No-STD."""
        return disable_random_noise(cls.v0p8())


def disable_random_noise(attrs):
    """Return a copy of ACMSignoffAttributes `attrs` with all the random noise sigmas set to zero."""
    return dataclasses.replace(attrs, sigma_weight_prop=0.0, sigma_weight_add=0.0, sigma_dot=0.0,
                               sigma_weight_sqrt_prop=0.0, per_tile_sigma_weight_prop=0.0)


class PytorchACMSignoffMMA:
    """An 'ACM-Signoff' model of the MMA in Boreas B

    Models digital artifacts:
        - quantization
        - clipping
        - multi-cycle
    and the empirical noise artifacts:
        - Robust regression weights error / Proportional weights error models
        - Robust regression dot products error model

    ACM-S releases with model description can be found here:
    - ACM-S v0-Prototype:
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/13961855252/ACM-S+v0.0+prototype+model+and+description
    - ACM-S v0.1 (Parameters fitted from hardware)
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14089978091/ACM+v0.1
    - ACM-S v0.2 (Incorporation of ACM-S into RMTR flow)
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14131921009/ACM-S+v0.2
    - ACM-S v0.3 (Add SNR-RN50 support and parameters from hardware)
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14148632823/ACM-S+v0.3
    - ACM-S v0.4 (Analyze SNR-RN50 on non-first silicon parts, provide model selection
        metrics on weights models)
    - ACM-S v0.5 (Convert weights model from proportional to robust linear model
        trained on 4Q2021 data)
    - ACM-S v0.8 https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14340227098/ACM-S+v0.8

    TODO: model weights to be randomized during class initialization instead
    of inside the self.randomize function to follow the Monte Carlo simulations
    use case explained here:
    https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14129365022/Monte+Carlo+Simulation+for+ACM-Signoff

    """
    _ALLOWED_FIXABLE_NOISE = (None, )

    def __init__(self, weights, biases, mma_attr=None, pFSR=2.0, iFSR=2.0,
                 name=None, seed=None, weight_scale=128):
        self.name = name

        # We need to know the device of the weights to cast the inputs to the
        # correct device when running multi-gpu
        self.device = weights.device
        self.dtype = weights.dtype
        if not weights.is_cuda and self.dtype is not torch.float32:
            self.dtype = torch.float32
            weights = weights.type(self.dtype)
            biases = biases.type(self.dtype)

        self.iFSR = iFSR
        self.pFSR = pFSR
        self.mma_attr = mma_attr or ACMSignoffAttributes()

        # Note: Concatenate and rescale weights and biases to [-128,128] scale
        # (originally assumed in [-1, 1] scale). The second scaling by
        # pFSR/iFSR happens within self.mod_weights_torch()
        self.weights = weight_scale * weights
        self.biases = weight_scale * biases

        self.pows1 = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=self.dtype, device=self.device)
        pows2 = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=self.device)
        self.pows2 = pows2.reshape(8, *((1,) * weights.dim()))

        # copied from simplemodel.py
        # internal state to manage and monitor the randomization of the noises
        self.mma_base_attr = copy.deepcopy(self.mma_attr)
        self.dot_gen = None
        self.flash_gen = None

        self.fit_parameters = [f.name for f in dataclasses.fields(ACMSignoffAttributes)]

        # Set default parameters from ACMSignoffAttributes()
        self.update_fit_parameters(reset_to_mma_attr=True)
        self.randomize()
        self.debug_container = None

    def randomize(self, fix_vals=None, random_state=None):
        fix_vals = fix_vals or {}

        if not all(noise_src in self._ALLOWED_FIXABLE_NOISE
                   for noise_src in fix_vals.keys()):
            raise KeyError("You are requesting a noise source to be fixed that"
                           " cannot be fixed. Allowed values are "
                           f"{self._ALLOWED_FIXABLE_NOISE}")

        self.weight_noisy = self.mod_weights_torch(self.weights)
        self.bias_noisy = self.mod_weights_torch(self.biases).sum(1)
        logger.debug('ACM-Signoff: Applied randomization to weights')

    def update_fit_parameters(self, new_params_lookup=None,
                              reset_to_mma_attr=False):
        """Update fit parameters or reset to defaults.

        Parameters:
        -----------
        new_params_lookup : dict
            dictionary with new parameter values
        reset_to_mma : bool
            if True, resets all parameters to what is in self.mma_attr

        """
        for param in self.fit_parameters:
            if (new_params_lookup is not None and param in new_params_lookup
                    and new_params_lookup[param] is not None):
                setattr(self, param, new_params_lookup[param])
            elif reset_to_mma_attr:
                setattr(self, param, getattr(self.mma_attr, param))

            logger.debug((f'ACMSignoff {self.name}: {param} is now '
                          + f'{getattr(self, param)}'))

    def normal(self, mean, std, size=None):
        return (torch.normal(mean=mean, std=std, generator=self.flash_gen) if size is None
                else torch.normal(mean=mean, std=std, size=size, generator=self.flash_gen,
                                  device=self.device, dtype=self.dtype))

    def mod_weights_torch(self, weight):
        """Apply noise models to flash weights. The model consists of a
        Gaussian model with non-constant variance, aka, proportional weights
        error model.

        Sequence of weight noise models applied:
            1) Apply linear tranformation of weights based on beta0 and beta1.
                This defines the mean of the noise Gaussian model
            2) Draw a random Gaussian model given the proportional weights error
                model using the mean of 1) and sigma_weight_prop or
                sigma_weight_add as part of the Gaussian standard deviation

        Reference:
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/13961855252/ACM-S+v0.0+prototype+model+and+description#Weights%3A--Proportional-Weights-Error-Model
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/14219018251/ACM-S+v0.5#Weights-Model-Equations

        Returns
        -------
        torch.tensor
            Weights after noise models are applied.

        """
        assert self.linear_beta0 == 0, f'Only linear_beta0=0 is currently supported, given: {self.linear_beta0}'

        # Compute weights mean, check for the identity transformation case to avoid unneeded computations.
        tile_prop_error = self.linear_beta1 * self.normal(mean=1.0, std=self.per_tile_sigma_weight_prop, size=(1,))
        weight_mean = (weight if self.linear_beta0 == 0 and tile_prop_error == 1
                       else torch.where(weight != 0,
                                        apply_linear_transform(tensor=weight,
                                                               beta0=self.linear_beta0,
                                                               beta1=tile_prop_error),
                                        weight))

        def gen_noise(std):
            size = None if isinstance(std, torch.Tensor) else weight_mean.shape
            return torch.where(weight_mean != 0, self.normal(mean=0.0, std=std, size=size), weight_mean)

        sigma_weight_sqrt_prop = self.sigma_weight_sqrt_prop
        # Do not compute abs_weight unless we need it.
        abs_weight = weight_mean.abs() if self.sigma_weight_prop > 0 or sigma_weight_sqrt_prop > 0 else None
        # Add different types of noise to weights.
        weight_noisy = (weight_mean
                        + (gen_noise(self.sigma_weight_prop * abs_weight) if self.sigma_weight_prop > 0 else 0.0)
                        + (gen_noise(sigma_weight_sqrt_prop * abs_weight.sqrt()) if sigma_weight_sqrt_prop > 0 else 0.0)
                        + (gen_noise(self.sigma_weight_add) if self.sigma_weight_add > 0 else 0.0))

        # At this point, weight_noisy may exceed the [-128, 127] range. This is how noise behaves in hardware.

        # Rescale weights back to 128*pFSR/iFSR scale, ie the transformed scale
        # of wnb (Following bcm_digital wnb definition during initialization)
        weight_noisy = weight_noisy * self.pFSR/self.iFSR

        return weight_noisy

    def register_forward_debug(self):
        """Register a forward debug container with this node."""
        self.debug_container = ACMSignOffMMADebugContainer()
        return self.debug_container

    def unregister_forward_debug(self):
        """Unregister a forward debug container with this node."""
        self.debug_container = None
        return None

    def dot(self, uint8_input, dot_op=F.linear):
        """
        uint8_input shape: (batch_size * num_windows, window_size)
        self.wnb shape: (window_size, out_channels)

        Reference:
        https://mythic-ai.atlassian.net/wiki/spaces/AISE/pages/13961855252/ACM-S+v0.0+prototype+model+and+description#Dot-Products%3A-Linear-Model

        """
        # Ensure that inputs are in the correct range
        uint8_input = torch.clamp(uint8_input.round(), 0, 255).to(dtype=torch.uint8, device=self.device)
        # Split into bits, the new shape is (bit, batch_size, ...)
        x_bits = uint8_input.unsqueeze(0) * self.pows2 // 128
        # Combine "bit" and "batch_size" dimensions into one to get the shape `dot_op` expects.
        x_bits_reshaped = x_bits.flatten(0, 1)
        dot = dot_op(x_bits_reshaped.type(self.dtype), self.weight_noisy, self.bias_noisy).round()
        # Split the combined dimensions back to "bit" and "batch_size" dimensions.
        dot_reshaped = dot.unflatten(0, (8, uint8_input.shape[0]))
        dot_clipped = torch.clamp(dot_reshaped, -128, 127)
        # Sum results from individual bits.
        accumulator = torch.matmul(dot_clipped.permute(*range(1, dot_clipped.dim()), 0), self.pows1)
        accumulator_clipped = torch.clamp((accumulator / 128).round(), -256, 255)

        dot_mean = apply_linear_transform(tensor=accumulator_clipped,
                                          beta0=self.linear_gamma0,
                                          beta1=self.linear_gamma1)

        dot_noisy = dot_mean if self.sigma_dot == 0 else self.normal(mean=dot_mean, std=self.sigma_dot)

        dot_noisy_rounded_clipped = torch.clamp(dot_noisy.round(), -256, 255)

        if self.debug_container is not None:
            self.debug_container.noisy_weight = self.weight_noisy
            self.debug_container.noisy_bias = self.bias_noisy
            self.debug_container.accumulator_out = accumulator_clipped
            self.debug_container.adc_input = dot

        return dot_noisy_rounded_clipped

    # The following functions conform to API of /munc/munc/bcm/bcm_models/simplemodel.py
    # and monte carlo calls.
    # Used to set the random number generator for different noise sources
    # during monte carlo runs
    # simplemodel.py has 5 generators, while acmmodel has only two,
    # some of the funcions will be deactivated, and their arguments will
    # be changed to (self, *args, **kwargs) so that they don't produce
    # errors if their declaration is changed in randomize_attrs.py
    def set_adc_noise_generator(self, *args, **kwargs):
        pass

    def set_popcorn_noise_generator(self, *args, **kwargs):
        pass

    # Not sure what the difference is between this and the below method:
    # probably one works by randomizing individual ADCs while the other
    # generates a chip level offset and slope
    # We need to activate the individual adcs
    def set_adc_offset_and_inl_generator(self, generator):
        self.dot_gen = generator

    def set_chip_level_adc_offset_and_inl(self, *args, **kwargs):
        pass

    # different implementation from simplemodel.py
    # This function only sets the generator for the weights which is used
    # in mod_weights_torch()
    def set_weight_programming_randomization(self, generator):
        self.flash_gen = generator

    def set_env_variables(self, *args, **kwargs):
        pass


def apply_linear_transform(tensor, beta0, beta1):
    """Apply linear combination to input tensor

    Parameters
    -------
    tensor : torch.tensor
        Weights or dot product values values.
    beta0 : float
        Intercept parameter.
    beta1 : float
        Slope parameter.

    Returns
    -------
    torch.tensor
        tensor values after linear combination is applied.

    """
    return beta0 + beta1*tensor
