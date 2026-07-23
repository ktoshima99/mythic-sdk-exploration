# noqa-flake8-docstrings
from functools import lru_cache
from munc._constants import ONNXType, MYTHICType
from munc import _node_utils, _constants
import numpy as np
import onnx.helper
from itertools import permutations


class ResultFormatONNX:
    """This result format is to generate ONNX.
    If mapping's method `inputs`/`output` is called with an instance of this class as
    `result_format` the call will be delegated to mapper's `_inputs_onnx`/`_output_onnx`.
    This class should be used together with `LayerToACEsMappingWithResultFormatDispatch`.
    """

    def __init__(self, model, layer_node):
        """Initialize an instance.

        Parameters
        ----------
        model : ONNXModel
        layer_node : ONNXNode
            a node that represents the current layer.
        """
        self.model = model
        self.layer_node = layer_node
        self.nodes = []

    def inputs(self, mapping, layer_inputs, scale_factors):
        return mapping._inputs_onnx(layer_inputs, scale_factors, self)

    def output(self, mapping, ace_results, scale_factors):
        return mapping._output_onnx(ace_results, scale_factors, self)

    def add_node(self, inputs, outputs, op_type, name, mythic_type=None):
        node = onnx.helper.make_node(op_type=op_type, inputs=inputs, outputs=outputs,
                                     name=self.model.get_new_node_name(name), doc_string='')

        if mythic_type is not None:
            _node_utils.create_attribute_with_value(node, _node_utils.MYTHIC_TYPE_ATTRIBUTE_NAME, mythic_type)

        self.nodes.append(node)
        return node

    def add_sum_node(self, inputs, outputs, name):
        return self.add_node(
            inputs,
            outputs,
            op_type=ONNXType.SUM,
            name=name,
            mythic_type=MYTHICType.SPLIT_INPUTS_ADD
        )


def make_onnx_dot(weight, bias, input, scale_factors, activation, result_format, quantize_weight, hw_config):
    """
    Create an ONNX node for an ACE computation corresponding to this layer (or a part of it).

    Parameters
    ----------
    weight : tensor
        ACE weight
    bias : tensor
        ACE bias
    input : string
        An input (edge) name for the node.
    scale_factors : layermodel.ScaleFactors
    activation : string
        ACE activation to use
    result_format : ResultFormatONNX
        A context: the current node, model, etc.
    quantize_weight : bool
        Round the weights to the nearest integer and clip them according
        to the hw_config.
    """
    model = result_format.model
    layer_node = result_format.layer_node

    # TODO: There is a big overlap between scaling code here and in prepare_mma. Refactor to use the same function in
    # both places.
    scale_factor = scale_factors.wsf * scale_factors.remainder

    # Scale, quantize, and clip weight and bias.
    weight = 128 * scale_factor * weight.numpy()
    # ACM and LayerCM use biases shaped as [N, 1]. In an ONNX node bias shape is expected to be [N].
    bias = 128 * scale_factor * bias.squeeze(1).numpy()
    if quantize_weight:
        weight = np.round(weight)
        bias = np.round(bias)
    weight = np.clip(weight, -128, 127)
    num_bias_splits = hw_config.bias_rows
    bias = np.clip(bias, -128 * num_bias_splits, 127 * num_bias_splits)

    weight_name = model.make_initializer_np(weight)
    bias_name = model.make_initializer_np(bias)
    output = model.get_new_edge_name()
    node_mm = result_format.add_node([input, weight_name, bias_name], [output], layer_node.op_type,
                                     f"{layer_node.name}_layer_cm")
    _node_utils.create_attribute_with_value(node_mm, _constants.ONNX_ATTR_SPLIT_ORIGIN, layer_node.name)

    for attribute in layer_node.attribute:
        node_mm.attribute.append(attribute)

    def set_attr(name, value):
        if _node_utils.get_attribute_value(node_mm, name) is not None:
            _node_utils.set_attribute_value(node_mm, name, value)
        else:
            _node_utils.create_attribute_with_value(node_mm, name, value)

    # Attach attributes to new MM
    scale_factors_to_save = ('iFSR', 'pFSR', 'multiplier', 'shift', 'wsf', 'remainder')
    for factor_name in scale_factors_to_save:
        attr_name = '__' + factor_name
        value = getattr(scale_factors, factor_name)
        set_attr(attr_name, value.item() if isinstance(value, np.ndarray) else value)
    set_attr('__activation', activation)

    return output


def _sort_onnx_nodes(nodes):
    """Sort a list of ONNX nodes by rank.

    Nodes with inputs outside of this list or without inputs will have rank 0.

    Returns
    -------
    list of ONNXNodes
        a fresh list of nodes sorted by rank.
    """
    output2node = {output: node for node in nodes for output in node.output}

    @lru_cache(maxsize=None)
    def node_rank(output):
        """Return the rank of a node identified by `output`."""
        node = output2node.get(output)
        return max(map(node_rank, node.input), default=-1) + 1 if node else 0

    return sorted(nodes, key=lambda node: node_rank(node.output[0]))


def _default_attribute_checker(attrs1, attrs2, node1, node2):
    assert attrs1 == attrs2


def check_onnx_graphs_match(model1, model2, attribute_checker=_default_attribute_checker, check_initializers=True,
                            check_input_names=True, outputs1=None, outputs2=None):
    """Compare ONNX graphs `model1` and `model2`, raise an error if the graphs are not isomorphic.

    The graphs must have identical outputs, but do not have to use the same node and edge names.
    Currently the function assumes all ops as noncommutative (i.e. the order of node inputs is fixed).

    Parameters
    ----------
    model1 : ONNXModel
        A model to compare
    model2 : ONNXModel
        A model to compare
    attribute_checker : function of (attrs1, attrs2, node1, node2), optional
        A function that is called to check that two nodes have the same attributes.
        `attrs1 == attrs2` by default.
    check_initializers : bool, optional
        Whether to compare initializer values. Defaults to True.
    check_input_names : bool, optional
        Whether to compare model input names. Defaults to True.
    outputs1 : list of str, optional
        A list of model1 output names. outputs1 are mapped to outputs2 in order.
        Defaults to `sorted(model1.get_output_names())`.
        If neither outputs1 nor outputs2 is provided, the function will check that
        outputs of both models have the same names.
    outputs2 : list of str, optional
        A list of model2 output names. outputs1 are mapped to outputs2 in order.
        Defaults to `sorted(model2.get_output_names())`.

    Returns
    -------
        dict, dict
        An edge mapping and a node name mapping from model1 to model2.

    Raises
    ------
        AssertError
        if the models are not isomorphic.
    """

    def map_edges(outputs1, outputs2, edge_map):
        """Create an edge mapping corresponding to a `model1` to `model2` isomorphism.

        `outputs1` and `outputs2` provide a starting point. The function assumes `outputs1[i]` corresponds to
        `outputs2[i]`. Because of it a node in `model1` that provides `outputs1[i]` corresponds to a node in
        `model2` that provides `outputs2[i]`, and inputs of these nodes have to match.

        Parameters
        ----------
        outputs1 : a list of some model1 node output names
        outputs2 : a list of some model2 node output names
        edge_map : dict
            a mapping from edges of `model1` to edges of `model2`. The function adds elements to the mapping.
        """
        for output1, output2 in zip(outputs1, outputs2):
            node1 = model1.get_node_with_output_name(output1)
            node2 = model2.get_node_with_output_name(output2)
            if node1 and node2 and output1 not in edge_map:
                edge_map[output1] = output2
                map_edges(node1.input, node2.input, edge_map)

    def node_attributes(node):
        return {attr.name: _node_utils.get_attribute_value(node, attr.name) for attr in node.attribute}

    outputs_provided = outputs1 is not None or outputs2 is not None
    if outputs1 is None:
        outputs1 = sorted(model1.get_output_names())
    if outputs2 is None:
        outputs2 = sorted(model2.get_output_names())

    assert outputs_provided or outputs1 == outputs2

    # Find a model1 to model2 edge mapping that give us an isomorphism (if the graphs are isomorphic).
    edge_map = {}
    map_edges(outputs1, outputs2, edge_map)

    def check_inputs(input1, input2):
        """Check `input1` of `model1` and `input2` of `model2` are the same modulo ismorphism."""
        if model1.get_node_with_output_name(input1):
            # The inputs are node outputs
            assert edge_map[input1] == input2
        else:
            # Check optional inputs. They can be either '' or np.ndarray([]).
            if input1 == '' or input2 == '':
                init1 = np.array([]) if input1 == '' else model1.get_initializer_np(input1)
                init2 = np.array([]) if input2 == '' else model2.get_initializer_np(input2)
                assert np.array_equal(init1, init2)
            else:
                # Check for initializers before checking for inputs, because all initializers are in inputs too.
                init1 = model1.get_initializer_np(input1)
                if init1 is not None:
                    # The inputs are initializers
                    if check_initializers:
                        assert np.allclose(init1, model2.get_initializer_np(input2))
                else:
                    # The inputs are model inputs
                    if check_input_names:
                        assert (input1 == input2 and input1 in model1.get_input_names()
                                and input2 in model2.get_input_names())

    def check_outputs(output1, output2):
        """Check `output1` of `model1` and `output2` of `model2` are the same modulo ismorphism."""
        assert edge_map[output1] == output2

    def check_nodes(node1, node2):
        """Check `node1` of `model1` and `node2` of `model2` are the same modulo ismorphism."""
        assert node1.op_type == node2.op_type
        assert len(node1.input) == len(node2.input) and len(node1.output) == len(node2.output)
        attribute_checker(node_attributes(node1), node_attributes(node2), node1, node2)
        for input1, input2 in zip(node1.input, node2.input):
            check_inputs(input1, input2)
        for output1, output2 in zip(node1.output, node2.output):
            check_outputs(output1, output2)

    nodes1 = model1.get_nodes()
    nodes2 = model2.get_nodes()
    assert len(nodes1) == len(nodes2)

    node_map = {}
    # For each node in model1 check that it is the same as a corresponding node of model2.
    for node1 in nodes1:
        output1 = node1.output[0]
        # If the models are not identical, a model1 edge may not have a corresponding model2 edge.
        assert output1 in edge_map
        output2 = edge_map[output1]
        node2 = model2.get_node_with_output_name(output2)
        check_nodes(node1, node2)
        node_map[node1.name] = node2.name
    return edge_map, node_map


def check_onnx_graphs_match_with_output_auto_mapping(model1, model2, attribute_checker=_default_attribute_checker,
                                                     check_initializers=True, check_input_names=True):
    """Compare ONNX graphs `model1` and `model2`, raise an error if the graphs are not isomorphic.

    This function tries all mappings of model outputs. See `check_onnx_graphs_match_with_output` for a description of
    the parameters.
    """
    outputs1 = sorted(model1.get_output_names())
    outputs2 = sorted(model2.get_output_names())

    def run_checker(outputs1, outputs2):
        return check_onnx_graphs_match(model1, model2, attribute_checker=attribute_checker,
                                       check_initializers=check_initializers, check_input_names=check_input_names,
                                       outputs1=outputs1, outputs2=outputs2)

    # Try all inputs1 to inputs2 mappings. It is very inefficient, but works fine, because the number of outputs
    # in our practical cases is small.
    for outputs2_perm in permutations(tuple(outputs2)):
        try:
            return run_checker(outputs1, outputs2_perm)
        except AssertionError:
            pass

    # Fail with the default mapping
    run_checker(outputs1, outputs2)
