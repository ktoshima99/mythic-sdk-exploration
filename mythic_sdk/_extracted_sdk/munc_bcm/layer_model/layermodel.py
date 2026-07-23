# noqa-flake8-docstrings
"""
Layer-CM core implementation.

The Layer-CM class definitions.
"""

from collections import namedtuple
from enum import Enum
from functools import partial
import logging
from munc.bcm import bcm_utils
from munc._constants import ONNXType

logger = logging.getLogger(__name__)


class RandomizeMode(Enum):
    """Model parameters randomization modes."""

    ALL = 1
    "Apply noise and non-idealities to all the parameters."


ScaleFactors = namedtuple('ScaleFactors', ['iFSR', 'pFSR', 'dsf', 'multiplier', 'shift', 'wsf', 'remainder',  # MMA
                                           'sum_csf'  # SALU
                                           ])


class LayerConfig():
    """A layer configuration base class. A layer configuration comprises a layer type and layer parameters."""

    def __init__(self, layer_type, layer_activation='relu'):
        """
        Initialize an instance.

        Parameters
        ----------
        layer_type : str
            A layer type (one of ONNXType.MYTHIC_CONV, ONNXType.MYTHIC_LINEAR)
        layer_activation : str
            The type of activation of the layer.
        """
        self.layer_type = layer_type
        self.layer_activation = layer_activation


class LinearLayerConfig(LayerConfig):
    """Configuration parameters of a linear layer."""

    def __init__(self, layer_activation='relu'):
        super().__init__(ONNXType.MYTHIC_LINEAR, layer_activation=layer_activation)

    def make_dot(self, weight_shape):
        """
        Return a wrapper on top of `ace_dot` that compute standard dot.

        LayerCM uses the function to compute product over input.
        """
        return bcm_utils.simple_dot_product


def compute_conv_kernel_shape(weight_shape):
    """Compute the shape of a Conv2D kernel based on the shape of weight."""
    kernel_shape = weight_shape[2:]

    if isinstance(kernel_shape, int):
        return (kernel_shape, kernel_shape)
    elif len(kernel_shape) > 2:
        raise ValueError('Kernel size should not have more than 2 elements, 1 per each spatial dim')
    else:
        return tuple(kernel_shape)


class ConvLayerConfig(LayerConfig):
    """Configuration parameters of a Conv layer."""

    def __init__(self, pads, strides, group=1, layer_activation='relu'):
        super().__init__(ONNXType.MYTHIC_CONV, layer_activation=layer_activation)
        self.pads = pads
        self.strides = strides
        self.group = group

    def make_dot(self, weight_shape):
        """
        Return a function to compute dot product over folded input using bcm_utils.torch_conv2d.

        LayerCM uses the function to compute product over input.
        """
        return partial(
            bcm_utils.acm_conv2d,
            kernel_size=compute_conv_kernel_shape(weight_shape),
            pad=self.pads,
            stride=self.strides,
            group=self.group)


class LayerCM:
    """Pytorch model of the effect of implementing a BCM layer on the Mythic chip.

    This will have support for:
            - input splitting
            - output splitting
            - parallelization
            - diagonal packing
            - bit-spreading
    All the actual noise modeling is still delegated to the underlying ACE noise model class.
    """

    def __init__(self, make_acm, make_layer_mapping, name=None, layer_config=LinearLayerConfig()):
        """
        Initialize the structure of the ACE's that make up the layer.

        Does not involve weights, biases, FSRs, etc. These will be set dynamically.

        Parameters
        ----------
        make_layer_mapping : (weights_shape, input_shape, config) -> LayerToACEsMapping
            a function that provides a mapping of the layer to ACEs.
        make_acm : (weights, biases, scale_factors, ace_activation) -> PytorchTrainingACM
            a function to be used to create ACM instances.
        name : str
           The name of the original node
        layer_config : LayerConfig
            a layer configuration that comprises a layer type, parameters, and a way to compute dot.
        """
        self.name = name
        self.make_layer_mapping = make_layer_mapping
        self.make_acm = make_acm
        self.layer_config = layer_config

        self.weights = None
        self.biases = None
        self.scale_factors = None
        self.randomize_mode = None
        self.acms = None
        self.input_shape = None
        self.weights_shape = None
        self.mapping = None
        self.dot_products = None

    def set_parameters(self, weights=None, biases=None, scale_factors=None, input_shape=None,
                       randomize_mode=RandomizeMode.ALL):
        """Set or change some parameters of the model. If a parameter value is omitted, its previous value is used."""
        def if_specified(value, default):
            return value if value is not None else default
        new_weights = if_specified(weights, self.weights)
        new_biases = if_specified(biases, self.biases)
        new_scale_factors = if_specified(scale_factors, self.scale_factors)
        new_input_shape = if_specified(input_shape, self.input_shape)
        self.randomize_mode = if_specified(randomize_mode, self.randomize_mode)

        assert new_weights is not None
        self.initialize_mapping(new_weights.shape, new_input_shape)

        need_to_randomize = True
        # If new weights, biases, or scale factors are provided, create a new set of noise models.
        if self.acms is None or any(p is not None for p in [weights, biases, scale_factors]):
            per_ace_weights, per_ace_biases, activation = self._get_per_ace_parameters(new_weights, new_biases,
                                                                                       new_scale_factors)
            self.acms = list(map(lambda weights, biases: self.make_acm(weights, biases, new_scale_factors, activation),
                                 per_ace_weights, per_ace_biases))
            need_to_randomize = False

        # Store the new parameter values, so the next time the method is called we know what's different.
        self.weights = new_weights
        self.biases = new_biases
        self.scale_factors = new_scale_factors
        self.input_shape = new_input_shape

        # TODO: remove need_to_randomize uglyness. Currently ACM constructors call `randomize`. If it was called here
        # uncoditionally, it would be called twice and it would affect performance, because `randomize` is expensive.
        if need_to_randomize:
            # apply weight noise and randomize the noise model params
            self.randomize(self.randomize_mode)

    def _get_per_ace_parameters(self, weights, biases, scale_factors):
        per_ace_weights = self.mapping.weights(weights, scale_factors)
        per_ace_biases = self.mapping.biases(biases, scale_factors)
        activation = self.mapping.required_ace_activation() or self.layer_config.layer_activation
        return per_ace_weights, per_ace_biases, activation

    def initialize_mapping(self, weights_shape, input_shape):
        """Initialize layer to ACEs mapping used by this LayerCM instance."""
        # If there is no mapping yet or it is based on different tensor shapes, create a new mapping
        # from the layer to ACEs.
        if (None in [self.mapping, self.weights_shape, self.input_shape]
           or ([self.weights_shape, self.input_shape] != [weights_shape, input_shape])):
            assert None not in [weights_shape, input_shape]
            self.mapping = self.make_layer_mapping(weights_shape=weights_shape, input_shape=input_shape)
            self.acms = None
            self.weights_shape = weights_shape
            self.input_shape = input_shape
            self.dot_products = map(self.layer_config.make_dot, self.mapping.weight_shapes())

    def randomize(self, randomize_mode=RandomizeMode.ALL):
        """
        Randomize all the ACE noise models.

        Includes applying noise to weights and biases The randomize_mode argument allows fine control of what kind of
        randomization to do.
        """
        for acm in self.acms:
            acm.randomize()

    def get_current_mapping(self):
        """Return the current layer to ACEs mapping.

        The mapping corresponds to parameters set by the last `set_parameters` call.
        """
        return self.mapping

    def compute_output(self, inputs):
        """
        Compute layer output.

        This function distributes parts of `inputs' to the layers ACE's, collects results from the ACEs, and
        combines them together into a layer output.
        """
        # Split the input tensor and push the sub-tensors down to the dot() functions of the ACE models.
        per_ace_inputs = self.mapping.inputs(inputs, self.scale_factors)

        def get_ace_output(acm, input, dot_product):
            return dot_product(input, acm.compute_output)

        ace_results = list(map(get_ace_output, self.acms, per_ace_inputs, self.dot_products))

        # Compute and return layer output from the ACEs results.
        # TODO: Layer activation should be applied here. It currently in InputSplitter.output.
        return self.mapping.output(ace_results, self.scale_factors)

    # TODO: This method does not have anything ONNX specific and has a lot in common with compute_output,
    # the two could be replaced with generate_output(...) parametrized with get_ace_output. _get_per_ace_parameters
    # would become public and called from layermodel2onnx. It's unclear if it's worth the effort though.
    def generate_onnx(self, input, result_format, make_onnx_dot):
        """
        Generate an ONNX subgraph representing this layer.

        Layer parameters should be set before this method is used. See `set_parameters`.

        Parameters
        ----------
        input : string
            The input (edge) name of this layer.
        result_format : ResultFormatONNX
            A context: the current node, model, etc.

        Returns
        -------
        string
            The output (edge) name of a generated subgraph representing this layer.
        """

        per_ace_inputs = self.mapping.inputs(input, self.scale_factors, result_format)
        per_ace_weights, per_ace_biases, activation = self._get_per_ace_parameters(self.weights, self.biases,
                                                                                   self.scale_factors)
        make_onnx_dot = partial(make_onnx_dot, activation=activation, scale_factors=self.scale_factors)
        ace_node_outputs = list(map(make_onnx_dot, per_ace_weights, per_ace_biases, per_ace_inputs))

        # Compute and return layer output from the ACEs results.
        # TODO: Layer activation should be applied here. It currently in InputSplitter.output.
        output = self.mapping.output(ace_node_outputs, self.scale_factors, result_format)
        return output
