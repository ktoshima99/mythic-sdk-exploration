# noqa-flake8-docstrings
"""Simple mappers that are based on slicing weights and inputs along a specified dimension."""

from collections import namedtuple
import logging
from munc.bcm.layer_model.mapper import LayerToACEsMappingWithResultFormatDispatch
from munc._constants import BiasSplittingMethod
import torch
from munc._session_tools import balance_split_bias, get_num_input_splits
import math
from munc import _node_utils
from munc._constants import MYTHICType, ONNXType
from munc import _session_tools as tools

logger = logging.getLogger(__name__)


def _replace_tuple_elt(tpl, index, new_value):
    lst = list(tpl)
    lst[index] = new_value
    return tuple(lst)


class DimSplitter(LayerToACEsMappingWithResultFormatDispatch):
    """
    An abstract layer to ACEs mapping that slices weights and inputs along a specified dimension.

    It maps each slice to an ACE. A user (subclass) must implement methods `output` and `biases` of
    `LayerToACEsMapping`.
    """

    def __init__(self, weight_shape, input_shape, weight_split_dim, input_split_dim, split_sizes,
                 onnx_slice_node_name, onnx_slice_node_mythic_type):
        """Initialize an instance.

        Parameters
        ----------
            weight_shape : tensor
            input_shape : tensor
            weight_split_dim : int or None
                along which dimension to split weights. If None, weights are replicated to each ACE.
            input_split_dim : int of None
                along which dimension to split inputs. If None, inputs are replicated to each ACE.
            split_sizes : list of int
                slice sizes.
            onnx_slice_node_name : string
                A name for ONNX slice nodes.
            onnx_slice_node_mythic_type : A MYTHICType
                A mythic node type for ONNX slice nodes.

        """
        self.num_splits = len(split_sizes)
        self.split_sizes = split_sizes
        self.weight_shape = weight_shape
        self.input_shape = input_shape
        self.weight_split_dim = weight_split_dim
        self.input_split_dim = input_split_dim
        split_starts = []
        start = 0
        for split_size in self.split_sizes:
            split_starts.append(start)
            start += split_size
        self.split_starts = split_starts
        self.onnx_slice_node_name = onnx_slice_node_name
        self.onnx_slice_node_mythic_type = onnx_slice_node_mythic_type

    def num_aces(self):
        return self.num_splits

    def weight_shapes(self):
        return self._slice_shapes(self.weight_shape, self.weight_split_dim)

    def input_shapes(self):
        return self._slice_shapes(self.input_shape, self.input_split_dim)

    def weights(self, layer_weights, scale_factors):
        return self._slice_data(layer_weights, self.weight_split_dim)

    def _inputs_value(self, layer_inputs, scale_factors):
        return self._slice_data(layer_inputs, self.input_split_dim)

    def _inputs_onnx(self, input, scale_factors, result_format):
        model = result_format.model
        if self.input_split_dim is None:
            return [input] * self.num_splits
        elif self.num_splits > 1:
            def narrow(start, size):
                output = model.get_new_edge_name()
                node_slice = result_format.add_node([input], [output], ONNXType.SLICE, self.onnx_slice_node_name,
                                                    mythic_type=self.onnx_slice_node_mythic_type)
                params = _node_utils.NodeData()
                params.starts = [start]
                params.ends = [start + size]
                params.axes = [self.input_split_dim]
                params.steps = [1]
                _node_utils.create_node_params(model, node_slice, params)
                return output

            return list(map(narrow, self.split_starts, self.split_sizes))
        else:
            return [input]

    def _slice_shapes(self, shape, dim):
        if dim is None:
            return (shape,) * self.num_splits
        else:
            return tuple(_replace_tuple_elt(shape, dim, split_size) for split_size in self.split_sizes)

    def _slice_data(self, tensor, dim):
        if dim is None:
            return [tensor] * self.num_splits
        else:
            return [tensor.narrow(dim, start, size) for (start, size)
                    in zip(self.split_starts, self.split_sizes)]

    def required_ace_activation(self):
        return None


class InputSplitter(DimSplitter):
    """A layer to ACEs mapping that splits inputs and weights to make them fit the ACE input size."""

    def __init__(self, weight_shape, input_shape, layer_activation, split_sizes, use_salu_model=True,
                 ace_activation_for_salu_sum='hardtanh', bias_splitting_method=BiasSplittingMethod.BALANCED):
        super().__init__(weight_shape=weight_shape, input_shape=input_shape, weight_split_dim=1, input_split_dim=1,
                         split_sizes=split_sizes, onnx_slice_node_name='Slice (Input Split)',
                         onnx_slice_node_mythic_type=MYTHICType.SPLIT_INPUTS_SLICE)
        self.activation = layer_activation
        self.use_salu_model = use_salu_model
        self.ace_activation_for_salu_sum = ace_activation_for_salu_sum
        self.bias_splitting_method = bias_splitting_method

    def biases(self, layer_biases, scale_factors):
        # This is what balance_split_bias expects.
        biases_1d = layer_biases.squeeze(1)
        # This bias splitting happens in floating point, because we want to quantize the weights after doing input
        # splitting.
        assert self.bias_splitting_method == BiasSplittingMethod.BALANCED, \
            f"Bias splitting method {self.bias_splitting_method} is not implemented"
        # TODO: Consider using split_bias here to align all bias splitting code paths.
        split_biases = balance_split_bias(biases_1d, num_bias_splits=self.num_splits, use_fp=True)
        # Convert back to the shape used by ACM and unpack to one tensor per ACM.
        return split_biases.unsqueeze(2).unbind(1)

    def _output_value(self, ace_results, scale_factors):
        # TODO: Move layer activation from sum_mma_split_inputs to LayerCM.
        return _sum_mma_split_inputs(tuple(ace_results), salu_model=self.use_salu_model,
                                     scale_factor=scale_factors.sum_csf, activation=self.activation)

    def _output_onnx(self, ace_results, scale_factors, result_format):
        model = result_format.model
        if self.num_splits > 1:
            assert self.use_salu_model
            # Create sum
            output = model.get_new_edge_name()
            node_sum = result_format.add_node(ace_results, [output], ONNXType.SUM, 'Sum (Input Splitting)',
                                              mythic_type=MYTHICType.SPLIT_INPUTS_ADD)

            # This will split the inputs for nodes using __csf_sum as the scaling factor in SUM
            _node_utils.create_attribute_with_value(node_sum, '__max_output', 1.0 / scale_factors.sum_csf)

            # Calculate the multiplier and shift values for the SUM node
            multiplier, shift, _ = tools.compute_multiplier_and_shift(scale_factors.sum_csf, number_of_bits=8)
            _node_utils.create_attribute_with_value(node_sum, '__multiplier', multiplier)
            _node_utils.create_attribute_with_value(node_sum, '__shift', shift)

            assert self.activation, "Compiler graph MMA nodes must have an activation"
            _node_utils.create_attribute_with_value(node_sum, '__activation', self.activation)

            return output
        else:
            assert not self.use_salu_model
            return ace_results[0]

    def required_ace_activation(self):
        return self.ace_activation_for_salu_sum if self.use_salu_model or self.num_splits > 1 else None


def _sum_mma_split_inputs(inputs, salu_model=True, scale_factor=None, activation=None):
    raise NotImplementedError()


InputSplitterConfig = namedtuple('InputSplitterConfig',
                                 ['hw_config', 'layer_config',
                                  'splits', 'splitting_method', 'use_salu_model',
                                  'ace_activation_for_salu_sum'])
"""
    An InputSplitter configuration, `make_split_input_mapping` takes it as parameter `config`.

    hw_config : HWConfig
        A hardware configuration.
    layer_config : LayerConfig
        A layer configuration.
    splits : list of int or None
        The size of each input slice. If it is not specified, slices are created automatically using `splitting_method'.
    splitting_method : str or None
        The method of input splitting to apply: "balanced", "greedy", "equal". It does not need to be specified if
        `splits` are specified.
    use_salu_model : bool
        If true, a SALU model will be used for summing ACE outputs.
    ace_activation_for_salu_sum : str or None
        An ACE activation to apply before summing ACE outputs in SALU. The hardware always uses 'hardtanh'.
"""


def make_split_input_mapping_helper(weights_shape, input_shape, config, max_ace_input_length):
    """Return a layer to ACEs mapping that splits inputs and weights to make them fit the ACE input size."""
    if config.splitting_method is None and config.splits is None:
        raise ValueError("Either input_splits or splitting_method must be specified")
    elif config.splitting_method is not None and config.splits is not None:
        raise ValueError("input_splits is mutually exclusive with splitting_method.")

    if config.splits:
        split_sizes = config.splits
    else:
        _, split_sizes = get_num_input_splits(weights_shape, max_ace_input_length,
                                              method=config.splitting_method)

    # If splits are explicitly provided, use the balanced splitting method for bias.
    bias_splitting_method = BiasSplittingMethod.BALANCED if config.splits is not None else config.splitting_method

    return InputSplitter(weights_shape, input_shape, layer_activation=config.layer_config.layer_activation,
                         split_sizes=split_sizes, use_salu_model=config.use_salu_model,
                         ace_activation_for_salu_sum=config.ace_activation_for_salu_sum,
                         bias_splitting_method=bias_splitting_method)


def make_split_input_mapping(weights_shape, input_shape, config):
    """
    Return a layer to ACEs mapping that splits inputs and weights to make them fit the ACE input size.

    This version of input splitting is for unsigned inputs.
    """
    return make_split_input_mapping_helper(weights_shape, input_shape, config, config.hw_config.max_inputs)


def make_split_input_mapping_signed(weights_shape, input_shape, config):
    """
    Return a layer to ACEs mapping that splits inputs and weights to make them fit the ACE input size.

    This version of input splitting is for signed inputs. In this version only a half of max_inputs can be used
    because a convolution with signed inputs requires duplication of weights.
    """
    # Need at least two inputs, because a signed input requires duplication of weights.
    assert config.hw_config.max_inputs > 1
    return make_split_input_mapping_helper(weights_shape, input_shape, config, config.hw_config.max_inputs // 2)


class OutputSplitter(DimSplitter):
    """A layer to ACEs mapping that splits inputs and weights to make them fit the ACE output size."""

    def __init__(self, weight_shape, input_shape, split_sizes):
        super().__init__(weight_shape=weight_shape, input_shape=input_shape, input_split_dim=None,
                         weight_split_dim=0, split_sizes=split_sizes, onnx_slice_node_name='Slice (Output Split)',
                         onnx_slice_node_mythic_type=MYTHICType.SPLIT_OUTPUTS_SLICE)

    def biases(self, layer_biases, scale_factors):
        return self._slice_data(layer_biases, 0)

    def _output_value(self, ace_results, scale_factors):
        return torch.cat(tuple(ace_results), 1)

    def _output_onnx(self, ace_results, scale_factors, result_format):
        model = result_format.model
        if self.num_splits > 1:
            output = model.get_new_edge_name()
            node = result_format.add_node(ace_results, [output], ONNXType.CONCAT, 'Concat (Output Split)')
            _node_utils.create_attribute_with_value(node, 'axis', 1)
            return output
        else:
            return ace_results[0]


OutputSplitterConfig = namedtuple('OutputSplitterConfig', ['hw_config', 'layer_config', 'splits'])
"""
    An OutputSplitter configuration, `make_split_output_mapping` takes it as parameter `config`.
    hw_config : HWConfig
        A hardware configuration.
    layer_config : LayerConfig
        A layer configuration.
    splits : list of int or None
        The size of each output slice.
"""


def _get_output_splits(layer_config, weights_shape, max_outputs):
    def chop(n, chunk_size):
        num_chunks = math.ceil(n / chunk_size)
        last_chunk_size = n % chunk_size or chunk_size
        return [chunk_size] * (num_chunks - 1) + [last_chunk_size]

    output_size = weights_shape[0]
    return chop(output_size, max_outputs)


def make_split_output_mapping(weights_shape, input_shape, config):
    """Return a layer to ACEs mapping that splits inputs and weights to make them fit the ACE output size."""
    if config.splits:
        split_sizes = config.splits
    else:
        split_sizes = _get_output_splits(config.layer_config, weights_shape, config.hw_config.num_of_adcs)

    return OutputSplitter(weights_shape, input_shape, split_sizes=split_sizes)


class ParallelizingMapping(DimSplitter):
    """A layer to ACEs mapping that splits samples into several groups to run them in parallel on multiple ACEs."""

    def __init__(self, weight_shape, input_shape, split_sizes):
        super().__init__(weight_shape=weight_shape, input_shape=input_shape, input_split_dim=0,
                         weight_split_dim=None, split_sizes=split_sizes, onnx_slice_node_name='Slice (Parallelization)',
                         onnx_slice_node_mythic_type=MYTHICType.SPLIT_PARALLELIZATION_SLICE)

    def biases(self, layer_biases, scale_factors):
        return [layer_biases] * self.num_splits

    def _output_value(self, ace_results, scale_factors):
        return torch.cat(tuple(ace_results), 0)

    def _output_onnx(self, ace_results, scale_factors, result_format):
        # TODO: Implement me
        raise AssertionError("Implement me")


ParallelizingMappingConfig = namedtuple('ParallelizingMappingConfig', ['hw_config', 'layer_config', 'splits'])
"""
    An ParallelizingMapping configuration, `make_parallelizing_mapping` takes it as parameter `config`.
    hw_config : HWConfig
        A hardware configuration.
    layer_config : LayerConfig
        A layer configuration.
    splits : list of int or None
        The size of each input group. The length of this list sets the number of times the ACEs for this layer will
        be replicated for performance.
"""


def make_parallelizing_mapping(weights_shape, input_shape, config):
    """Return a layer to ACEs mapping that splits samples into groups to run them in parallel on multiple ACEs."""
    # parallelization = config.parallelization
    # slice_size = math.ceil(input_shape[0] / parallelization)
    # last_slice_size = input_shape[0] % slice_size or slice_size
    # split_sizes = ([slice_size] * (parallelization - 1)) + [last_slice_size]
    return ParallelizingMapping(weights_shape, input_shape, split_sizes=config.splits)
