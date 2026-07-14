# noqa-flake8-docstrings
from functools import partial
import torch
import torch.nn.functional as F
import logging
import copy
from munc.bcm.registry import SimpleAttributes
from .acmsignoffmodel import ACMSignoffAttributes
from munc.bcm.debug_containers import TrainingACMMMADebugContainer
from enum import Enum, IntEnum
from inspect import signature

logger = logging.getLogger(__name__)

FACTORY_NAME = 'munc_tacm'


class Accuracy(Enum):
    """Accuracy levels."""

    IGNORE = 0
    MOCKUP = 1
    FULLMULTICYCLE = 2
    BCMSIMPLE = 3
    ACMS = 4


class Quantization(IntEnum):
    """Quantization levels."""
    OFF = 0
    QUANTIZE = 1
    CLIP = 2


# TODO: move attribute definitions and their default values here from PytorchTrainingACM.
class PytorchTrainingACMAttributes(SimpleAttributes, ACMSignoffAttributes):
    """Configuration attributes of PytorchTrainingACM."""

    def __init__(self, submodel_type="training"):
        """See `tacm_submodel_types` for a list of the supported model types."""
        SimpleAttributes.__init__(self)
        ACMSignoffAttributes.__init__(self)
        self.tacm_submodel = submodel_type


tacm_submodel_types = {
    # weights_accuracy_level, multicycle_accuracy_level, adc_accuracy_level
    'quantized': (Accuracy.IGNORE,) * 3,
    'digital': (Accuracy.IGNORE, Accuracy.FULLMULTICYCLE, Accuracy.IGNORE),
    'training': (Accuracy.MOCKUP,) * 3,
    'multicycle_training': (Accuracy.MOCKUP, Accuracy.FULLMULTICYCLE, Accuracy.MOCKUP),
    # bcm-simple:
    'full': (Accuracy.BCMSIMPLE, Accuracy.FULLMULTICYCLE, Accuracy.BCMSIMPLE),
    'weight_noise_only': (Accuracy.BCMSIMPLE, Accuracy.IGNORE, Accuracy.IGNORE),
    'sarcycle_adc_noise': (Accuracy.IGNORE, Accuracy.FULLMULTICYCLE, Accuracy.BCMSIMPLE),
    'digital_with_weight_noise': (Accuracy.BCMSIMPLE, Accuracy.FULLMULTICYCLE, Accuracy.IGNORE),
    # acm-signoff:
    'acms': (Accuracy.ACMS, Accuracy.FULLMULTICYCLE, Accuracy.ACMS),
    'acms_weights_only': (Accuracy.ACMS, Accuracy.IGNORE, Accuracy.IGNORE),
    'acms_dot_only': (Accuracy.IGNORE, Accuracy.IGNORE, Accuracy.ACMS),
    'acms_no_multicycle': (Accuracy.ACMS, Accuracy.IGNORE, Accuracy.ACMS),
}
"""Supported PytorchTrainingACM named configuration. A dictionary: name -> accuracy settings"""


class PytorchTrainingACM:
    """Fast approximations to the full Analog Compute model.

    The full Analog Compute Model (ACM) has three sources of imperfections on
    top of quantized vector-matrix multiplication:
    - weight imperfections
    - multicycle effects
    - ADC imperfections.

    For the weight and dot-product noise, we have three levels of accuracy modelling.
    - "ignore" : do not model
    - "mockup" : just throw a Gaussian noise
    - "full accuracy" can correspond to full-multicycle, bcmsimple, and acm-signoff

    For example, the accuracy levels of
    [ignore, ignore, ignore]               <=> quantized multiplication, aka "munc_fp"
    [bcmsimple, fullmulticycle, bcmsimple] <=> "munc_simple"
    [ignore, fullmulticycle, ignore]       <=> effectively, "munc_digital"
                                               (nit: the difference is up to floor() vs round();
                                                "munc_digital" is not a zero-noise limit of "munc_simple")
    [mockup, mockup, mockup]               <=> a fast training model, mockup of all imperfections;
                                               5x-10x faster than [bcmsimple, fullmulticycle, bcmsimple]
    [bcmsimple, fullmulticycle, mockup]    <=> a training model thats fair wrt multicycle, but mockups the SAR cycle
                                               2x-3x faster than [bcmsimple, fullmulticycle, bcmsimple]


    [acms, fullmullticycle, acms]          <=> same as acm-signoff
    """
    _ALLOWED_FIXABLE_PARAMS = (None, )

    def __init__(self, weights, biases, mma_attr=None, pFSR=2.0, iFSR=2.0, name=None,
                 seed=None, weight_scale=128.0):
        self.name = name
        self.mma_attr = PytorchTrainingACMAttributes() if mma_attr is None else copy.deepcopy(mma_attr)
        self.device = weights.device
        # dtype - FP16 training is 20% faster than FP32 for Simplemodel
        self.dtype = weights.dtype if weights.is_cuda else torch.float32

        self.out_channels = weights.shape[0]
        self.iFSR = iFSR
        self.configure_accuracy_params()

        self.weights_dim = weights.dim()

        # A function to generate weights with imperfections and noise.
        self.weights_factory = WeightsFactory(weights, biases,
                                              pFSR=pFSR, weights_accuracy_level=self.weights_accuracy_level,
                                              device=self.device, dtype=self.dtype, mma_attr=self.mma_attr,
                                              weight_scale=weight_scale)

        # Setup model parts according to selected accuracy levels. A full dot model comprises 3
        # configurable parts: expand_inputs - unpacking of input bits for multicycle processing (or some other input
        # preprocessing), mcd - modelling of multicycle processing effects, and sar - modelling of ADC SAR.
        if self.adc_accuracy_level == Accuracy.BCMSIMPLE:
            self.expand_inputs = ExpandInputsForMulticycleProcessing(self)
            # When the multicycle processing is enabled the bias is added each per bit computation, i.e.
            # it gets multiplied by 255 automatically.
            self.bias_multiplier = 1
            # adc_sar models MCD, nothing to do here
            self.mcd = NOP(self)
            self.sar = ADCSAR(self)
        else:
            if self.multicycle_accuracy_level == Accuracy.FULLMULTICYCLE:
                self.expand_inputs = ExpandInputsForMulticycleProcessing(self)
                self.bias_multiplier = 1
                self.mcd = MCDFull(self)
            elif self.multicycle_accuracy_level == Accuracy.MOCKUP:
                self.expand_inputs = NOP(self)
                # When the multicycle processing is disabled the bias needs to be multiplied by 255,
                # because that's what a chip does during the multicycle processing.
                self.bias_multiplier = 255
                self.mcd = MCDMock(self)
            else:
                assert self.multicycle_accuracy_level == Accuracy.IGNORE
                self.expand_inputs = NOP(self)
                self.bias_multiplier = 255
                self.mcd = NOP(self)

            if self.adc_accuracy_level == Accuracy.MOCKUP:
                self.sar = ADCNoiseMock(self)
            elif self.adc_accuracy_level == Accuracy.ACMS:
                self.sar = ADCNoiseAcms(self)
            else:
                self.sar = NOP(self)

        self.set_quantization_levels(self.input_quantization_level, self.output_quantization_level)
        self.push_mma_attrs()
        self.randomize()
        self.debug_container = None

    def set_quantization_levels(self, input_quantization_level, output_quantization_level):
        input_quantizers = {Quantization.OFF: identity, Quantization.QUANTIZE: _quantize_input,
                            Quantization.CLIP: _clip_input}
        self.input_quantizer = input_quantizers[input_quantization_level]
        output_quantizers = {Quantization.OFF: identity, Quantization.QUANTIZE: _quantize_output,
                             Quantization.CLIP: _clip_output}
        self.output_quantizer = output_quantizers[output_quantization_level]
        self.input_quantization_level = input_quantization_level
        self.output_quantization_level = output_quantization_level

    def dot(self, uint8_input, dot_op=F.linear):
        """
        Dot product + imperfections according to the accuracy settings.

        inputs shape: (num_inputs, input_size)
        wnb shape: (out_channels, input_size)
        """
        uint8_input = self.input_quantizer(uint8_input)

        # Prepare input for multicycle processing if necessary.
        mc_input = self.expand_inputs.run(uint8_input)

        adc_in = dot_op(mc_input.to(dtype=self.dtype), self.iflash_weights, self.iflash_biases * self.bias_multiplier)
        adc_in = adc_in / self.iFSR

        # Multicycle processing effects
        mcd_output = self.mcd.run(adc_in)

        # ADC/SAR effects
        sar_output = self.sar.run(mcd_output)
        output = self.output_quantizer(sar_output)

        if self.debug_container is not None:
            self.debug_container.iflash_weight = self.iflash_weights
            self.debug_container.iflash_bias = self.iflash_biases
            self.debug_container.adc_input = adc_in
            self.debug_container.mcd_output = mcd_output
            self.debug_container.sar_output = sar_output
            self.debug_container.output = output

        return output

    def register_forward_debug(self):
        """Register a forward debug container with this node."""
        self.debug_container = TrainingACMMMADebugContainer()
        return self.debug_container

    def unregister_forward_debug(self):
        """Unregister a forward debug container with this node."""
        self.debug_container = None
        return None

    def randomize(self, fix_vals=None, random_state=None):
        fix_vals = {} if fix_vals is None else fix_vals

        def fix_noise_sources(obj):
            """Back door to tweak the parameters."""
            for key, val in fix_vals.items():
                if key in self._ALLOWED_FIXABLE_PARAMS:
                    setattr(self, key, val)
                else:
                    raise KeyError("You are requesting a noise source to be fixed that cannot be fixed. "
                                   f"Allowed values are {self._ALLOWED_FIXABLE_PARAMS}")
        # Push new attr values to the model parts.
        # TODO: do not push unrelated attributes.
        for obj in [self, self.expand_inputs, self.mcd, self.sar]:
            fix_noise_sources(obj)
        # Randomize weights
        self.iflash_weights, self.iflash_biases = self.weights_factory.generate_iflash_wnb()
        # Randomize ADC by calling randomize on each model part.
        for obj in [self.expand_inputs, self.mcd, self.sar]:
            obj.randomize()

    def fetch_mma_attrs(self, objects, attr_names, convert_value):
        """
        Set instance variables `attr_names` of `objects` to corresponding values from `self.mma_attr`.

        Keys are prefixed by 'tacm_'. Fetched values are passed through `convert_value`.
        """
        for attr_name in attr_names:
            if hasattr(self.mma_attr, "tacm_" + attr_name):
                val = convert_value(vars(self.mma_attr)["tacm_" + attr_name])
                # TODO: do not push unrelated attributes.
                for o in objects:
                    setattr(o, attr_name, val)

    def configure_accuracy_params(self):
        """Reset accuracy level weight noise, multicycle, adc.

        Each is modelled ignore/mockup/fullmulticycle[/..more]

        This is a temporary template
        TODO replace according to what will be in .yaml
        """
        # Default accuracy levels
        self.weights_accuracy_level = Accuracy.MOCKUP
        self.multicycle_accuracy_level = Accuracy.MOCKUP
        self.adc_accuracy_level = Accuracy.MOCKUP
        self.input_quantization_level = Quantization.QUANTIZE
        self.output_quantization_level = Quantization.QUANTIZE

        # Override by a keyword
        def set_levels(weights_accuracy_level, multicycle_accuracy_level, adc_accuracy_level,
                       input_quantization_level=Quantization.QUANTIZE,
                       output_quantization_level=Quantization.QUANTIZE):
            self.weights_accuracy_level = weights_accuracy_level
            self.multicycle_accuracy_level = multicycle_accuracy_level
            self.adc_accuracy_level = adc_accuracy_level
            self.input_quantization_level = input_quantization_level
            self.output_quantization_level = output_quantization_level

        try:
            set_levels(*tacm_submodel_types[self.mma_attr.tacm_submodel])
        except KeyError:
            raise Exception(f"Unknown ACM accuracy keyword {self.mma_attr.tacm_submodel}")

        # Override by setting accuracy and quantization levels directly.
        # TODO: use level names in configs instead of interger values.
        self.fetch_mma_attrs([self], ['weights_accuracy_level', 'multicycle_accuracy_level', 'adc_accuracy_level'],
                             Accuracy)
        self.fetch_mma_attrs([self], ['input_quantization_level', 'output_quantization_level'], Quantization)

        # verify accuracy levels
        if self.adc_accuracy_level == Accuracy.BCMSIMPLE:
            assert self.multicycle_accuracy_level == Accuracy.FULLMULTICYCLE, \
                "Can only simulate SAR cycle if Multicycle is fully calculated"

    def push_mma_attrs(self):
        """Fetch relevant configuration parameters from self.mma_attrs and push them to the model and its parts."""
        self.fetch_mma_attrs(
            [self, self.expand_inputs, self.mcd, self.sar, self.weights_factory],
            ['fitting_factor_adcnoise', 'fitting_factor_adcimperfections', 'fitting_factor_multicycle'],
            lambda x: x)

    def __str__(self):
        str_name = (" / " + self.name) if self.name else ""
        return f"{FACTORY_NAME}  mma{str_name}"

    def dump_attrs(self):
        return vars(self.mma_attr)


class Compose:
    """
    Compose a sequence of functions.
    Compose(f,g) is the same as lambda *args: f(*g(*args)), but can be pickled (if f ang g can).
    """

    def __init__(self, *funcs):
        self.funcs = list(funcs)
        self.funcs.reverse()

    def __call__(self, *x):
        for f in self.funcs:
            x = f(*x)
        return x


def _wnb_signs_wrapper(add_noise, abs_wnb, wnb_signs):
    "A helper function for _add_wnb_signs. It has to be at the top level to be pickleable."
    return add_noise(abs_wnb), wnb_signs


def _add_wnb_signs(add_noise):
    """Convert a noise source that works on absolute weight values to one that works on absolute values and signs.

    WeightFactory expects noise sources to have (abs_wnb, wnb_signs) -> (abs_wnb, wnb_signs) signature,
    because in some cases weight signs may be needed. But in the most cases only absolute weight values are used,
    so it is convenient to write a noise source as abs_wnb -> abs_wnb function and to use a wrapper to implement
    the required signature.
    """
    return (partial(_wnb_signs_wrapper, add_noise) if len(signature(add_noise).parameters) == 1
            else add_noise)


class WeightsFactory():
    """A class to generate distorted and noisy weights and biases."""

    def __init__(self, weights, biases, pFSR, weights_accuracy_level, device, dtype, mma_attr,
                 weight_scale):
        """
        Initialize.

        Parameters
        ----------
        acm : model
            an owner
        weights : tensor
        biases : tensor
        pFSR : number
        weights_accuracy_level : Accuracy
            defines how to model weights/biases:
            IGNORE - no noise added
            MOCKUP - temperature effects and flash variations are added
            FULL - temperature effects flash variations, and popcorn noise are added
        """
        weights = weights.to(dtype=dtype)
        biases = biases.to(dtype=dtype)
        self.device = device
        self.mma_attr = mma_attr
        # Weights are -1..1.  Biases are -nsplit..nsplit
        self.weights = weights
        self.biases = biases
        self.pFSR = pFSR
        self.weight_scale = weight_scale

        def compose(*noise_sources):
            return Compose(*map(_add_wnb_signs, noise_sources))

        # self.add_noise is a function (abs_weights, weight_signs) -> (abs_weights, weight_signs)
        if weights_accuracy_level == Accuracy.IGNORE:
            self.add_noise = _identity2  # No noise
        elif weights_accuracy_level == Accuracy.MOCKUP:
            self.add_noise = compose(self.apply_wnb_flash_variations,
                                     self.apply_wnb_temp_effects)
        elif weights_accuracy_level == Accuracy.BCMSIMPLE:
            self.add_noise = compose(self.apply_popcorn_wnb_noise,
                                     self.apply_wnb_flash_variations,
                                     self.apply_wnb_temp_effects)
        else:
            assert weights_accuracy_level == Accuracy.ACMS
            self.add_noise = self.apply_acms_weight_noise

    def generate_iflash_wnb(self,
                            temp_delta=None,
                            mc_mult_sigma_lsb=None,
                            pop_fraction=None,
                            mc_mult=None,
                            linear_beta0=None,
                            linear_beta1=None,
                            sigma_proportional_w=None,
                            sigma_additive_w=None,
                            mma_attr=None):
        """
        Generate distorted and noisy weights and biases using provided parameters.

        If a parameter is not specified, its value from the previous call is used.
        """
        def _with_default_value(val, default_obj, default_attr_name):
            return getattr(default_obj, default_attr_name) if hasattr(default_obj, default_attr_name) and val is None \
                else val

        # Take default parameter values from mma_attr.
        self.mma_attr = _with_default_value(mma_attr, self, "mma_attr")
        # Names from SimpleAttributes
        self.temp_delta = _with_default_value(temp_delta, self.mma_attr, "temp_delta")
        self.mc_mult = _with_default_value(mc_mult, self.mma_attr, "mc_mult")
        self.mc_mult_sigma_lsb = _with_default_value(mc_mult_sigma_lsb, self.mma_attr, "mc_mult_sigma_lsb")
        self.pop_fraction = _with_default_value(pop_fraction, self.mma_attr, "pop_fraction")
        # Names from ACMSignoffAttributes. Note name changes
        self.linear_beta0 = _with_default_value(linear_beta0, self.mma_attr, "linear_beta0")
        self.linear_beta1 = _with_default_value(linear_beta1, self.mma_attr, "linear_beta1")
        self.sigma_additive_w = _with_default_value(sigma_additive_w, self.mma_attr, "add_sigma")
        self.sigma_proportional_w = _with_default_value(sigma_proportional_w, self.mma_attr, "prop_sigma")

        def _apply_noise(weights):
            # program flash (w=1 corresponds to 100nA flash current)
            flash = weights * self.pFSR

            flash_abs = flash.abs()
            flash_signs = flash.sign()
            flash_abs, flash_signs = self.add_noise(flash_abs, flash_signs)
            flash = flash_abs.copysign(flash_signs)
            return flash

        return _apply_noise(self.weights), _apply_noise(self.biases).sum(1)

    def apply_wnb_temp_effects(self, abs_wnb):
        # Modify flash values due to temperature effects
        return exponential_temp_change(abs_wnb, self.temp_delta)

    def apply_wnb_flash_variations(self, abs_wnb):
        # Apply Flash MonteCarlo variation
        return apply_weight_var_flatmc(abs_wnb, self.mc_mult, self.mc_mult_sigma_lsb, pFSR=self.pFSR)

    def apply_popcorn_wnb_noise(self, abs_wnb):
        # Apply pop-corn noise
        mma_attr = self.mma_attr
        popcorn_step_dist = (0 if mma_attr.pop_fraction == 0 or mma_attr.pop_lognorm_sigma == 0
                             else torch.exp(torch.normal(mma_attr.pop_lognorm_mean,
                                                         mma_attr.pop_lognorm_sigma,
                                                         size=abs_wnb.shape,
                                                         dtype=torch.float32,
                                                         device=self.device)))

        abs_wnb = apply_popcorn_noise(abs_wnb, popcorn_step_dist, self.pop_fraction)
        return abs_wnb

    def apply_acms_weight_noise(self, abs_wnb, wnb_signs):
        """ACMS-style statistical noises.

        Masks out 0-weights, and has both additive and proportional noises.
        """
        wnb = abs_wnb.copysign(wnb_signs)

        # zero weights are not programmed
        mask = torch.ones_like(wnb)
        mask[wnb == 0] = 0
        # match ACMS scale before applying noises.
        wnb_scaled = wnb * self.weight_scale / self.pFSR

        wnb_noisy = _linear_transform(wnb_scaled, self.linear_beta0, self.linear_beta1)
        if self.sigma_additive_w and self.sigma_additive_w > 0:
            wnb_noisy = torch.normal(
                mean=wnb_noisy,
                std=self.sigma_additive_w,
            )
        if self.sigma_proportional_w and self.sigma_proportional_w > 0:
            wnb_noisy = torch.normal(
                mean=wnb_noisy,
                std=(self.linear_beta1 * wnb_scaled * self.sigma_proportional_w).abs(),
            )

        # At this point, wnb_noisy may exceed the [-128, 127] range. This is how noise behaves in hardware.
        wnb_processed = wnb_noisy * mask * self.pFSR / 128.

        return wnb_processed.abs(), wnb_processed.sign()


class ComputationPart:
    """
    Conceptually a computation part is a function from an input to an output.

    E.g. additive noise or a dot product with fixed weights. It is implemented as a class for now, because some parts
    need to be parametrized or have a state.
    `run` implements a function application. `randomize` is called to randomize parameters of the part (e.g. apply
    non-idealities to weights).
    """

    def __init__(self, acm):
        """
        Initialize.

        Parameters
        ----------
        acm : an object that provides mma_attrs, device, and dtype
            a model owning this part (currently an instance of `PytorchTrainingACM`)
        """
        self.acm = acm

    def run(self, val):
        """Apply effects of this part to input `val`. The default implementation is identity."""
        return val

    def randomize(self):
        """
        Randomize parameters of the part.

        An owner calls this method on every model part. The default implementation does nothing.
        """
        pass


class NOP(ComputationPart):
    """A 'do nothing' model part."""


class ExpandInputsForMulticycleProcessing(ComputationPart):
    """Unpack inputs into a per-bit form for multicycle processing."""

    def __init__(self, acm):
        super().__init__(acm)
        pows2 = torch.logspace(0, 7, 8, 2, device=acm.device, dtype=torch.uint8)
        self.bit_scales = pows2.reshape(8, *((1,) * acm.weights_dim))

    def run(self, uint8_input):
        # (1, num_inputs, input_size) * (8, 1, 1)--> (8, num_inputs, input_size)
        input_bits = uint8_input.to(dtype=torch.uint8, device=self.acm.device).unsqueeze(0) * self.bit_scales // 128
        # Combine "bit" and "batch_size" dimensions into one to get the shape `dot_op` expects.
        return input_bits.flatten(0, 1)


class MCDMock(ComputationPart):
    """Model MCD as Gaussian noise. Input is not expected to be unpacked into bits."""

    def __init__(self, acm):
        super().__init__(acm)
        self.fitting_factor_multicycle = 3.

    def run(self, z):
        """Mockup multicycle effects."""
        return z + self.fitting_factor_multicycle * torch.randn(size=z.shape, dtype=z.dtype, device=z.device)


class MCDFull(ComputationPart):
    """Model MCD by clipping per bit inputs. Input is expected to be unpacked into bits."""

    def __init__(self, acm):
        super().__init__(acm)
        # Noiseless SAR cycle scaling parameters
        self.bit_scales = torch.logspace(0, 7, 8, 1/2, device=acm.device, dtype=acm.dtype)

    def run(self, input):
        # Split the combined dimension back to "bit" and "batch_size" dimensions.
        input = input.unflatten(0, (8, input.shape[0] // 8))
        # NB below floor(), not round() unlike BCM-digital. This version is the zero-noise limit of BCM-simple.
        input = (input * 128).floor()
        dot_clipped = torch.clamp(input, -128, 127)
        output = torch.matmul(dot_clipped.permute(*range(1, dot_clipped.dim()), 0), self.bit_scales)
        return output


class ADCSAR(ComputationPart):
    """
    Accurately models ADC SAR.

    Input is expected to be unpacked into bits. This class applies MCD, so it should be used with mcd=NOP() in
    the model.
    """

    def __init__(self, acm):
        super().__init__(acm)
        self.num_inputs = 1
        # Noiseless SAR cycle scaling parameters
        pows2 = torch.logspace(0, 7, 8, 2, device=acm.device, dtype=torch.uint8)
        self.bit_scales = pows2.reshape(8, *((1,) * acm.weights_dim))
        self.out_channels = None

    def run(self, adc_in):
        """Run SAR cycle. adc_in shape: (8, num_inputs, out_dim, ...)."""
        # Split the combined dimension back to "bit" and "batch_size" dimensions.
        adc_in = adc_in.unflatten(0, (8, adc_in.shape[0] // 8))
        # Combine all the output dimensions into one to avoid dealing with shape differences in
        # compute_bcms_adc_output
        adc_in_reshaped = adc_in.reshape(adc_in.shape[0], adc_in.shape[1], -1)
        self.out_channels = adc_in_reshaped.shape[2]
        if self.lazy_randomization_requested:
            self.lazy_randomization_requested = False
            self.randomize_adc_parameters()

        res = compute_bcms_adc_output(
            adc_in=adc_in_reshaped,
            adc_noise=self.get_adc_noise(),
            simple_offset=self.simple_offset,
            sar_bases=self.sar_bases,
            group_size=self.num_inputs,
            out_channels=self.out_channels)
        # Restore output dimensions
        res = res.reshape(*adc_in.shape)

        return (res / self.bit_scales).sum(0)

    # Keep this as a function (i.e. do not rename or inline), because tests override this function to make noise
    # reproducible.
    def get_adc_noise(self):
        scaled_simple_noise = 10e6 * self.acm.mma_attr.simple_noise / self.acm.iFSR
        return scaled_simple_noise * torch.normal(0, 1, (8, 8, self.num_inputs, self.out_channels),
                                                  dtype=self.acm.dtype, device=self.acm.device)

    # Keep this as a function (i.e. do not rename or inline), because tests override this function to make offsets
    # reproducible.
    def get_adc_offset_and_inl(self):
        random_normal = torch.normal(
            0, 1, (1, self.num_inputs, 2*self.out_channels), dtype=self.acm.dtype, device=self.acm.device)
        # offset shape: (1, 1, num_inputs, out_channels)
        scaled_simple_offset = 10e6 * self.acm.mma_attr.simple_offset / self.acm.iFSR
        offset = scaled_simple_offset * random_normal[:, :, 0::2]
        inl = self.acm.mma_attr.simple_inl * random_normal[:, :, 1::2]
        return offset, inl

    def randomize(self):
        if self.out_channels is None:
            self.lazy_randomization_requested = True
        else:
            self.randomize_adc_parameters()

    def randomize_adc_parameters(self):
        """Generate Per-ace adc imperfections."""
        self.simple_offset, self.simple_inl = self.get_adc_offset_and_inl()
        # pows2 shape: (8, 1, num_inputs, out_channels)
        pows2 = torch.cat([
            (2 + self.simple_inl) ** x
            for x in torch.arange(2, 10, 1, dtype=torch.float32, device=self.acm.device)]
        )
        # This is equivalent of 'sar_scale_factor' in bcm repo
        self.sar_bases = (2. / pows2).type(self.acm.dtype)


@torch.jit.script
def compute_bcms_adc_output(adc_in, adc_noise, simple_offset, sar_bases, group_size: int, out_channels: int):
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


class ADCNoiseMock(ComputationPart):
    """Model ADC/SAR imperfections by adding Gaussian noise, does not run SAR."""

    def __init__(self, acm):
        super().__init__(acm)
        # Defaults for mockup at "mockup" accuracies:
        self.fitting_factor_adcnoise = 4.
        self.fitting_factor_adcimperfections = 3.

    def run(self, z):
        adc_noise = torch.tensor([10e6 * self.acm.mma_attr.simple_noise], device=z.device, dtype=z.dtype)

        sigma = torch.sqrt(
            (self.fitting_factor_adcnoise * adc_noise)**2
            + (self.fitting_factor_adcimperfections * self.adc_imperfections)**2
        )
        return z + sigma * torch.randn(size=z.shape, dtype=z.dtype, device=z.device)

    def randomize(self):
        """Per-ace adc imperfections."""
        self.adc_imperfections = torch.randn(size=(1,), dtype=self.acm.dtype, device=self.acm.device)


class ADCNoiseAcms(ComputationPart):
    """Model ADC/SAR imperfections by adding Gaussian noise, does not run SAR."""

    def __init__(self, acm):
        super().__init__(acm)
        self.linear_gamma0 = self.acm.mma_attr.linear_gamma0
        self.linear_gamma1 = self.acm.mma_attr.linear_gamma1
        self.sigma_dot = self.acm.mma_attr.sigma_dot

    def run(self, z):
        z_noisy = _linear_transform(z, self.linear_gamma0, self.linear_gamma1)

        if self.sigma_dot > 0:
            z_noisy = torch.normal(mean=z_noisy, std=self.sigma_dot)

        return z_noisy


def exponential_temp_change(flash, temp_delta):
    if temp_delta == 0 or temp_delta is None:
        return flash
    # This code is sensitive to the floating point precision of flash
    # Do not use a 16-bit floating point value for flash otherwise this will introduce additional noise
    exponent = (-5.6064e6 * 1e-7 * torch.abs(flash)).float()
    return flash * ((2.0928e-2 * torch.exp(exponent) * temp_delta) + 1).type(flash.dtype)


def apply_weight_var_flatmc(flash, sigma, sigma_lsb, pFSR=None):
    """
    Flash input is expected to be already scaled by pFSR, and its units are 100nA.

    From BCM-Simple
    """
    if pFSR is None:
        raise ValueError("pFSR not defined for weight var sigma specified in lsb")
    if sigma == 0 and sigma_lsb == 0:
        return flash
    else:
        _mask = torch.ones_like(flash)
        _mask[flash == 0] = 0

        mc_w_rand = torch.normal(0, 1, size=flash.shape, dtype=flash.dtype, device=flash.device)
        flash = (flash + flash * sigma * mc_w_rand * 8 / pFSR
                 + _mask * sigma_lsb * (1.5625 / 100.) * pFSR * 0.5 * mc_w_rand)
        return torch.clamp(flash, min=0)


def apply_popcorn_noise(flash, popcorn_step_dist, pop_fraction):
    """
    Popcorn weight noise as in bcm_simple.

    TODO(ES) Verify that it is not all-0.
    """
    if pop_fraction == 0:
        return flash
    else:
        # Binary mask of the effected flash cells
        flash_mask = torch.empty_like(flash).uniform_() > pop_fraction
        dtype = flash.dtype
        # This math should happen in 32-bit if popcorn_step_dist can be large
        flash = flash - flash * popcorn_step_dist * flash_mask
        flash = torch.clamp(flash, min=0).type(dtype)
        return flash


def identity(x):
    return x


def _identity2(x, y):
    return x, y


def _clip_input(input):
    return torch.clamp(input, 0, 255)


def _quantize_input(input):
    return _clip_input(input.round())


def _clip_output(output):
    return torch.clamp(output, -256, 255)


def _quantize_output(output):
    return _clip_output(output.round())


def enable_quantization(mma, level):
    if hasattr(mma, 'set_quantization_levels'):
        mma.set_quantization_levels(level, level)


def _linear_transform(tensor, beta0, beta1):
    return beta0 + beta1 * tensor if beta0 != 0 or beta1 != 1 else tensor
