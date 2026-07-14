# noqa-flake8-docstrings
"""The definition of a layer of ACEs mapping, mapper composition functions, and a passthrough mapping."""
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from itertools import chain
import logging

logger = logging.getLogger(__name__)


class LayerToACEsMapping(ABC):
    """
    A mapping of weights, inputs, etc of a layer to weights, inputs, etc of ACEs that implement the layer.

    This class defines an interface. Its implementations may provide different mappings from a layer to ACEs.
    """

    @abstractmethod
    def num_aces(self):
        """Return the number of ACEs (tiles) by the mapping."""
        pass

    @abstractmethod
    def weight_shapes(self):
        """Return weight shape for each tile (as a list)."""
        pass

    @abstractmethod
    def input_shapes(self):
        """Return input shape for each tile (as a list)."""
        pass

    @abstractmethod
    def weights(self, layer_weights, scale_factors):
        """Compute per tile weights from `layer_weights`, return a list of tensors."""
        pass

    @abstractmethod
    def biases(self, layer_biases, scale_factors):
        """Compute per tile biases from `layer_biases`, return a list of tensors."""
        pass

    @abstractmethod
    def inputs(self, layer_inputs, scale_factors, result_format):
        """
        Compute per tile inputs from `layer_inputs`, return a list of objects.

        Parameters
        ----------
        layer_inputs : list of objects
            A list of per tile values required for a given 'result_format`.
        scale_factors : layermodel.ScaleFactors
        result_format : object
            specifies a result format, e.g. per tile input tensors or
            an ONNX subgraph to compute them.

        Returns
        -------
        list of objects
            A list of objects representing tile inputs. The type of objects depends
            on `result_format` (e.g. tensor if the result_format computes values or
            an edge name if it generates an ONNX graph).
        """
        pass

    @abstractmethod
    def output(self, ace_results, scale_factors, result_format):
        """
        Compute layer output from the tile outputs.

        Parameters
        ----------
        layer_inputs : list of objects
            A list of per tile values required for a given 'result_format`.
        scale_factors : layermodel.ScaleFactors
        result_format : object
            specifies a result format, e.g. per tile input tensors or
            an ONNX subgraph to compute them.

        Returns
        -------
        list of objects
            A list of objects representing tile inputs. The type of objects depends
            on `result_format` (e.g. tensor if the result format is values or
            an edge name if it is an ONNX graph).
        """
        pass

    @abstractmethod
    def required_ace_activation(self):
        """Return an activation if this mapping needs a specific one, None otherwise."""
        pass


class _ResultFormatValues:
    """This result format is to compute values.
    If mapping's method `inputs`/`output` is called with an instance of this class as
    `result_format` the call will be delegated to mapper's `_inputs_value`/`_output_value`.
    This class should be used together with `LayerToACEsMappingWithResultFormatDispatch`.
    """

    def inputs(self, mapping, layer_inputs, scale_factors):
        return mapping._inputs_value(layer_inputs, scale_factors)

    def output(self, mapping, ace_results, scale_factors):
        return mapping._output_value(ace_results, scale_factors)


RESULT_FORMAT_VALUES = _ResultFormatValues()


# Making this class a mixin is an option, but it is unclear if it would improve anything. -- Ilya
class LayerToACEsMappingWithResultFormatDispatch(LayerToACEsMapping):
    """Dispatch methods `inputs` and `output` to "result format"-specific methods.

    This class provides default implementations of methods `inputs` and `output` that delegate work
    either to methods that compute values or to methods that generate ONNX depending on `result_format`.
    See `ResultFormatONNX` and `_ResultFormatValues`.
    """

    def inputs(self, layer_inputs, scale_factors, result_format=RESULT_FORMAT_VALUES):
        return result_format.inputs(self, layer_inputs, scale_factors)

    def output(self, ace_results, scale_factors, result_format=RESULT_FORMAT_VALUES):
        return result_format.output(self, ace_results, scale_factors)

    @abstractmethod
    def _inputs_value(self, layer_inputs, scale_factors):
        """Compute per tile inputs from `layer_inputs`, return a list of tensors."""
        pass

    @abstractmethod
    def _output_value(self, ace_results, scale_factors):
        """Compute layer output from the tile outputs."""
        pass

    @abstractmethod
    def _inputs_onnx(self, layer_inputs, scale_factors, result_format):
        """Generate an ONNX subgraph that computes per tile inputs from `layer_inputs`.

        Parameters
        ----------
        layer_inputs : string
            A layer input (edge) name.
        scale_factors : ScaleFactors
        result_format : ResultFormatONNX

        Returns
        -------
        list of strings
            a list of output (edge) names that provide per tile inputs.
        """
        pass

    @abstractmethod
    def _output_onnx(self, ace_results, scale_factors, result_format):
        """Generate an ONNX subgraph that computes layer output from the tile outputs.
        Parameters
        ----------
        ace_results : list of strings
            tile outputs to be combined to produce layer output
        scale_factors : ScaleFactors
        result_format : ResultFormatONNX

        Returns
        -------
        string
            the name of an output that provides this layer result.
        """
        pass


class PassThroughMapping(LayerToACEsMappingWithResultFormatDispatch):
    """A layer to ACEs mapping that uses one ACE and maps the layer directly to it."""

    def __init__(self, weights_shape, input_shape):
        self.weights_shape = weights_shape
        self.input_shape = input_shape

    def num_aces(self):
        return 1

    def weight_shapes(self):
        return [self.weights_shape]

    def input_shapes(self):
        return [self.input_shape]

    def weights(self, layer_weights, scale_factors):
        return [layer_weights]

    def biases(self, layer_biases, scale_factors):
        return [layer_biases]

    def _inputs_value(self, layer_inputs, scale_factors):
        return [layer_inputs]

    def _output_value(self, ace_results, scale_factors):
        # Return output of the first (and only) ACE.
        return ace_results[0]

    def required_ace_activation(self):
        return None

    def _inputs_onnx(self, layer_inputs, scale_factors, result_format):
        return [layer_inputs]

    def _output_onnx(self, ace_results, scale_factors, result_format):
        return ace_results[0]


PassThroughMappingConfig = namedtuple('PassThroughMappingConfig', [])
"""
A PassThroughMapping configuration. `make_pass_through_mapping` takes an instance
of this tuple as parameter `config`.
"""


def make_pass_through_mapping(weights_shape, input_shape, config=None):
    """
    Return a pass-through mapping.

    This function is useful because it has the same signature as all the other layer mapper and can be
    composed with some of them for testing purposes.
    """
    return PassThroughMapping(weights_shape, input_shape)


class ComposedMapping(LayerToACEsMapping):
    """A layer to ACEs mapping that is a composion of two mappings."""

    def __init__(self, first_mapping, second_mappings):
        """
        Initialize an instance.

        Parameters
        ----------
        first_mapping : LayerToACEsMapping
            A mapping that is applied to a layer first and splits it into chunks.
        second_mappings : list of LayerToACEsMapping
            A mapping for each chunk that maps it into ACEs.
        """
        self.first_mapping = first_mapping
        self.second_mappings = second_mappings
        self.total_num_aces = sum([m.num_aces() for m in second_mappings])

    def num_aces(self):
        return self.total_num_aces

    def weight_shapes(self):
        # Concatenate shapes from all the second mappings into a flat list.
        return list(chain.from_iterable(m.weight_shapes() for m in self.second_mappings))

    def input_shapes(self):
        # Concatenate shapes from all the second mappings into a flat list.
        return list(chain.from_iterable(m.input_shapes() for m in self.second_mappings))

    def _map_twice(self, map_values, layer_values):
        """
        Pass `layer_values` though the two mappings.

        For example, if `layer_values` are layer weights, the first mapping is used to
        to slice the weights, and then the second mappings are used to slice each slice into
        smaller slices.
        Parameters
        ----------
        map_values : (mapping, parent_values) -> list of per chunk values.
            A function that splits `parent_values` into chunks using `mapping`.
            `parent_values` may be either weights, biases, or inputs.
            `mapping` is a layer to ACEs mapping.
        layer_values : tensor
            Either weights, biases, or inputs of a layer.

        Returns:
            list of tensors
                a chunk of `layer_values` for each ACE.
        """
        # Apply the first mapping.
        first_values = map_values(self.first_mapping, layer_values)
        # Apply the second mapping to the results of the first mapping.
        second_values = map(map_values, self.second_mappings, first_values)
        # Flatten.
        return list(chain.from_iterable(second_values))

    def weights(self, layer_weights, scale_factors):
        def map_weights(mapping, parent_weights):
            return mapping.weights(parent_weights, scale_factors)
        return self._map_twice(map_weights, layer_weights)

    def biases(self, layer_biases, scale_factors):
        def map_biases(mapping, parent_biases):
            return mapping.biases(parent_biases, scale_factors)
        return self._map_twice(map_biases, layer_biases)

    def inputs(self, layer_inputs, scale_factors, result_format=RESULT_FORMAT_VALUES):
        def map_inputs(mapping, parent_inputs):
            return mapping.inputs(parent_inputs, scale_factors, result_format)
        return self._map_twice(map_inputs, layer_inputs)

    def output(self, ace_results, scale_factors, result_format=RESULT_FORMAT_VALUES):
        # Give each second mapping results from its ACEs, and ask for its output.
        start = 0
        results = []
        for mapping in self.second_mappings:
            end = start + mapping.num_aces()
            results.append(mapping.output(ace_results[start:end], scale_factors, result_format))
            start = end
        # Then use the first mapping to combine all the outputs produced by the second mappings.
        return self.first_mapping.output(results, scale_factors, result_format)

    def required_ace_activation(self):
        # Check that all the mappers, that require an activation, require the same one, and
        # return it.
        result = self.first_mapping.required_ace_activation()
        for mapping in self.second_mappings:
            activation = mapping.required_ace_activation()
            if activation:
                assert not result or result == activation, \
                    f"Two different ACE activations requested {result} and {activation}"
                result = activation
        return result


def _make_composed_mapping(make_second_mapping, make_first_mapping, weights_shape, input_shape):
    """
    Make a mapping that is a composition of mappings created by `make_first_mapping` and `make_second_mapping`.

    A partial application of this function to a pair of mappers produces a mapper that has the standard signature.

    """
    first_mapping = make_first_mapping(weights_shape, input_shape)
    second_mappings = list(map(make_second_mapping, first_mapping.weight_shapes(),
                               first_mapping.input_shapes()))
    return ComposedMapping(first_mapping, second_mappings)


def compose_mappers(*mappers):
    """Compose ACEs `mappers`."""
    def compose2(make_second_mapping, make_first_mapping):
        return partial(_make_composed_mapping, make_second_mapping, make_first_mapping)
    mappers = list(mappers)
    res = None
    for m in mappers:
        res = m if res is None else compose2(m, res)
    return res
