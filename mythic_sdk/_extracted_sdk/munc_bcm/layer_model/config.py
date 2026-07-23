# noqa-flake8-docstrings
"""
Layer-CM configuration.

Layer-CM configuration functions, a list of layer mappers LayerCM uses, their configuration classes, and corresponding
ONNX node attributes.
"""

from functools import partial
import logging
from munc.bcm.layer_model.mapper import compose_mappers, make_pass_through_mapping
from munc.bcm.layer_model.signed_conv_mapper import SignedMappingConfig, make_signed_mapping
from munc.bcm.layer_model.splitters import (InputSplitterConfig, make_split_input_mapping,
                                            make_split_input_mapping_signed)
from munc import _node_utils

logger = logging.getLogger(__name__)


def _attr(attr_name, field, default):
    """Create a parser that sets 'field` to a value of attribute `attr_name` (`default` if the attribute is missing)."""
    def parser(node, layer_config, hw_config, config_class):
        return {field: _node_utils.get_attribute_value(node, attr_name, default)}
    return parser


def _hw_config():
    """Create a parser that puts a hardware configuration to field 'hw_config` if `config_class` defines it."""
    def parser(node, layer_config, hw_config, config_class):
        return {'hw_config': hw_config} if 'hw_config' in config_class._fields else {}
    return parser


def _layer_config():
    """Create a parser that puts a hardware configuration to field 'layer_config` if `config_class` defines it."""
    def parser(node, layer_config, hw_config, config_class):
        return {'layer_config': layer_config} if 'layer_config' in config_class._fields else {}
    return parser


def _mapped_attr(attr_name, field, val_map, default):
    """
    Create a parser that sets 'field` to a mapped value of attribute `attr_name`.

    Dict `mal_map` is used to map attribute values to field values. If the attribute is not specified,
    the field is set to `default`.
    """
    def parser(node, layer_config, hw_config, config_class):
        attr_val = _node_utils.get_attribute_value(node, attr_name)
        return {field: val_map.get(attr_val, default)}
    return parser


def _has_one_of_attrs(attr_names, field):
    """Create a parser that sets `field` to true if `node` has one of `attr_names` attributes, otherwise to false."""
    def parser(node, layer_config, hw_config, config_class):
        return {field: any(_node_utils.is_attribute(node, attr) for attr in attr_names)}
    return parser


def _node_satisfies(pred, field):
    """Create a parser that sets `field` to true if `node` satisfies predicate `pred`, otherwise set it to false."""
    def parser(node, layer_config, hw_config, config_class):
        return {field: pred(node)}
    return parser


DEFAULT_PARSERS = [
    _layer_config(),
    _hw_config()
]


def _input_splitting_enabled(node):
    return _node_utils.get_attribute_value(node, '__input_splitter_enabled', True)


COMMON_INPUT_SPLITTER_PARAMETER_PARSERS = [
    *DEFAULT_PARSERS,
    _attr('__input_splitter_splitting_method', 'splitting_method', 'balanced'),
    _attr('__input_splitter_splits', 'splits', None),
    _attr('__input_splitter_use_salu_model', 'use_salu_model', True),
    _attr('__input_splitter_ace_activation_for_salu_sum', 'ace_activation_for_salu_sum', 'hardtanh')
]
"""
Map ONNX node attribute names to InputSplitterConfig field names and default values.
This is used to build an input splitter configuration.
"""

UNSIGNED_INPUT_SPLITTER_PARAMETER_PARSERS = [
    *COMMON_INPUT_SPLITTER_PARAMETER_PARSERS,
    _node_satisfies(lambda node: _input_splitting_enabled(node)
                    and not _node_utils.is_node_signed(node)
                    and not _node_utils.is_digital_onchip(node), 'enabled')
]
"""Enable an input splitted unless its node is a signed conv."""

SIGNED_INPUT_SPLITTER_PARAMETER_PARSERS = [
    *COMMON_INPUT_SPLITTER_PARAMETER_PARSERS,
    _node_satisfies(lambda node: _input_splitting_enabled(node)
                    and _node_utils.is_node_signed(node)
                    and not _node_utils.is_digital_onchip(node), 'enabled')
]
"""Enable an input splitted only if its node is a signed conv."""

SIGNED_CONV_PARAMETER_PARSERS = [
    *DEFAULT_PARSERS,
    _attr('__layer_cm_for_training', 'duplicate_weight', True),
    _node_satisfies(lambda node: _node_utils.is_node_signed(node), 'enabled')
]
"""Enable a signed conv mapper for signed conv nodes only."""


LAYER_MAPPERS = [
    # (make_mapping, config_class, attribute_definitions)
    # The two input splitter below are mutually exclusive.
    (make_split_input_mapping, InputSplitterConfig, UNSIGNED_INPUT_SPLITTER_PARAMETER_PARSERS),
    (make_split_input_mapping_signed, InputSplitterConfig, SIGNED_INPUT_SPLITTER_PARAMETER_PARSERS),
    (make_signed_mapping, SignedMappingConfig, SIGNED_CONV_PARAMETER_PARSERS)
]
"""
A list of layer mappers LayerCM uses, their configuration classes, and corresponding ONNX node attributes.
Each list element is a tuple (make_mapping, config_class, attribute_definitions):
    make_layer_mapping : (weights_shape, input_shape, config) -> LayerToACEsMapping
        a function that provides a mapping of the layer to ACEs.
    config_class : class
        a namedtuple class that defines a configuration of `make_layer_mapping` and can be used as a value of
        its `config`.
    attribute_definitions : dict
        A map from ONNX node attribute names to (field_name, default_value) pairs. It defines how to create
        a mapper configuration from node attributes.
"""


def get_mapper_configurations(node, hw_config, layer_config, layer_mappers=LAYER_MAPPERS):
    """
    Return configurations of layer mappers to be used by a LayerCM instance that models 'node'.

    Parameters
    ----------
    node : ONNX node
        A node to get configuration parameters from.
    hw_config : HWConfig
        Hardware parameters.
    layer_config : LayerConfig
        A layer configuration to use.
    layer_mappers : list
        A list of layer mappers to configure. Defaults to `LAYER_MAPPERS`. See `LAYER_MAPPERS`.

    Returns
    -------
    dict
        A dictionary that contains a configuration (tuple) for each enabled mapper.
    """
    res = {}
    for (make_mapping, config_class, attribute_definitions) in layer_mappers:
        # Fetch field values from node attributes and store them into a dict.
        parameters_dict = {}
        for parser in attribute_definitions:
            parameters_dict.update(parser(node, layer_config, hw_config, config_class))
        enabled = parameters_dict.pop('enabled', True)
        if enabled:
            config = config_class(**parameters_dict)
            res[make_mapping] = config
    return res


def create_layer_mapper(mapper_configurations, layer_mappers=LAYER_MAPPERS):
    """
    Create a layer mapper that is a composition of configured `layer_mappers`.

    This function configures all the `layer_mappers` that have configurations in `mapper_configurations`
    (by partially applying `make_layer_mapping` functions to their configurations). Then it composes the
    configured mappers and returns the result.

    Parameters
    ----------
    mapper_configurations : dict
        A dictionary that contains a configuration (tuple) for each enabled mapper.
    layer_mappers : list
        A list of layer mappers to configure. Defaults to `LAYER_MAPPERS`. See `LAYER_MAPPERS`.

    Returns
    -------
    function
        A layer mapper that is a composition of the configured layer mappers.

    LayerCM configuration example
    -----------------------------
    .. code-block:: python
        activation = _node_utils.get_attribute_value(node, _constants.ONNX_ATTR_ACTIVATION)
        layer_config = LinearLayerConfig(layer_activation=activation)

        # Fetch mapper configurations.
        mapper_configs = get_mapper_configurations(node, hw_config, layer_config)

        # Create a layer to ACEs mapper using the configurations.
        make_layer_mapping = create_layer_mapper(mapper_configs)

        layer_cm = LayerCM(make_acm, make_layer_mapping, layer_config=layer_config)
    """

    def configure_mapper(make_mapping):
        return partial(make_mapping, config=mapper_configurations[make_mapping])

    enabled_mappers = [make_mapping for (make_mapping, _, _) in layer_mappers if make_mapping in mapper_configurations]
    configured_mappers = [configure_mapper(desc) for desc in enabled_mappers]
    return compose_mappers(*configured_mappers) or make_pass_through_mapping
