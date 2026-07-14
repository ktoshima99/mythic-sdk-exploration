# noqa-flake8-docstrings
from collections import namedtuple
import torch
from munc.bcm.layer_model.mapper import LayerToACEsMappingWithResultFormatDispatch, PassThroughMapping
from munc import _node_utils


def _multiply_element_in_list(data, multiplier, target_index):
    return [element * multiplier if target_index == index else element for index, element in enumerate(data)]


SignedMappingConfig = namedtuple('SignedMappingConfig', ['duplicate_weight'])
"""
A SignedMapping configuration. `make_signed_mapping` takes an instance
of this tuple as parameter `config`.

duplicate_weight : bool
    if false the mapping becomes a pass-through and can only be used to generate ONNX.
"""


def make_signed_mapping(weights_shape, input_shape, config):
    """Return a signed mapping."""
    mapper = SignedMapper if config.duplicate_weight else SignedMapperWithoutWeightDuplication
    return mapper(weights_shape, input_shape)


class SignedMapper(LayerToACEsMappingWithResultFormatDispatch):
    """Emulate the way the Boreas-B HW implements signed convolution."""

    def __init__(self, weight_shape, input_shape):
        """Initialize an instance.

        Parameters
        ----------
            weight_shape : tensor
            input_shape : tensor
        """
        self.weight_shape = weight_shape
        self.input_shape = input_shape

        # Shapes are doubled in the second input dimension
        self.dimension_to_modify = 1

    def num_aces(self):
        """Return the number of ACEs (tiles) by the mapping."""
        return 1

    def weight_shapes(self):
        """Return weight shape for each tile (as a list)."""
        return [_multiply_element_in_list(self.weight_shape, 2, self.dimension_to_modify)]

    def input_shapes(self):
        """Return input shape for each tile (as a list)."""
        return [_multiply_element_in_list(self.input_shape, 2, self.dimension_to_modify)]

    def weights(self, layer_weights, scale_factors):
        """Compute per tile weights from `layer_weights`, return a list of tensors."""
        return [torch.cat([layer_weights, -layer_weights], dim=self.dimension_to_modify)]

    def biases(self, layer_biases, scale_factors):
        """Compute per tile biases from `layer_biases`, return a list of tensors."""
        return [layer_biases]

    def _inputs_value(self, layer_inputs, scale_factors):
        """Compute per tile inputs from `layer_inputs`, return a list of tensors.

        Duplicate the inputs across the perferred dimension, with only the
        positive inputs in the first set and only the
        negative inputs in the second set.
        The negative inputs are negated so that the input to the ACE is always positive
        """
        pos_inputs = torch.clamp(layer_inputs, min=0)
        neg_inputs = torch.clamp(-layer_inputs, min=0)
        out_inputs = torch.cat([pos_inputs, neg_inputs], dim=self.dimension_to_modify)
        return [out_inputs]

    def _output_value(self, ace_results, scale_factors):
        """Compute layer output from the tile outputs. Return output of the first (and only) ACE."""
        return ace_results[0]

    def _inputs_onnx(self, layer_inputs, scale_factors, result_format):
        # TODO: Implement me. Probably we do not need to do anything here. Or we can add a subgraph that mimics
        # what _inputs_value does.
        return [layer_inputs]

    def _output_onnx(self, ace_results, scale_factors, result_format):
        tile_output_name = ace_results[0]
        _mark_tile_as_singed(result_format.model, tile_output_name)
        return tile_output_name

    def required_ace_activation(self):
        return None


class SignedMapperWithoutWeightDuplication(PassThroughMapping):
    """Emulate the way the Boreas-B HW implements signed convolution."""

    def __init__(self, weights_shape, input_shape):
        super().__init__(weights_shape, input_shape)

    def _output_value(self, ace_results, scale_factors):
        raise AssertionError("This class can only be used to generate ONNX.")

    def _output_onnx(self, ace_results, scale_factors, result_format):
        tile_output_name = ace_results[0]
        _mark_tile_as_singed(result_format.model, tile_output_name)
        return tile_output_name


def _mark_tile_as_singed(model, tile_output_name):
    tile_node = model.get_node_with_output_name(tile_output_name)
    _node_utils.mark_node_as_signed(tile_node)
