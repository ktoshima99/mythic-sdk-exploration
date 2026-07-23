import logging
from math import modf

from munc import _constants, _node_utils, _base_op
from munc._constants import ONNXType, HardwareType
from munc._pattern_detector import is_relu6
from munc._session_tools import is_op_type_supported_on_chip

logger = logging.getLogger(__name__)


class MarkUnsupportedOpsOffChip(_base_op.BaseOp):
    """Mark nodes as off-chip if unsupported on-chip."""

    def _get_info(self):
        return {
            'node_count': 0,
            'run_msg': 'Marking unsupported off-chip layers...',
            'requires_stat_collection': _constants.STATS_NOT_REQUIRED
        }

    def _run(self, nodes):
        if self.model.hwconfig is None:
            raise ValueError("Hardware configuration is not set. Please set the hardware configuration before running.")
        for node in self.model.get_nodes():
            if not is_op_type_supported_on_chip(node.op_type, self.model.hwconfig.name):
                if is_relu6(self.model, node):
                    logger.info(f"Assuming Clip node {node.name} refers to a ReLU-6 activation.")
                    continue
                _node_utils.mark_off_chip(node)
            elif not _node_utils.is_off_chip(node) and node.op_type in checker_functions:
                check_fn = checker_functions.get(node.op_type)
                mark_off_chip = check_fn(self.model, node)
                if mark_off_chip:
                    _node_utils.mark_off_chip(node)
            else:
                continue


class _IncompatibleAttrValue(ValueError):
    pass


def _make_check_one_of(*expected_values):
    def check(node, attribute, actual_value):
        if actual_value not in expected_values:
            raise _IncompatibleAttrValue(f"Node {node.name} not supported on-chip since {attribute} "
                                         f"has value {actual_value}. Expected one of {expected_values}.")
    return check


def _check_attributes_are_valid(model, node, expected_attributes):
    node_params = _node_utils.get_node_params(model, node)
    try:
        for attribute, expected in expected_attributes.items():
            actual_value = getattr(node_params, attribute)
            check = expected if callable(expected) else _make_check_one_of(expected)
            check(node, attribute, actual_value)
        return False
    except _IncompatibleAttrValue as err:
        logger.info(str(err) + ' Marking off-chip.')
        return True


def _check_dilations(node, attribute, actual_value):
    if actual_value is not None and any(x != 1 for x in actual_value):
        raise _IncompatibleAttrValue(f"Node {node.name} not supported on-chip since is uses "
                                     f"non-default dilations ({actual_value}). All the dilations are expected to be 1.")


def _check_conv(model, node):
    # Only Conv3D operations
    input_shape = model.get_edge_shape(node.input[0])
    if len(input_shape) != 4:
        logger.info(f"Node {node.name} not supported on-chip since input shape != 4. Marking off-chip.")
        return True

    mark_off_chip = _check_attributes_are_valid(model, node,
                                                {'auto_pad': 'NOTSET',
                                                 'dilations': _check_dilations})
    return mark_off_chip


def _check_gemm(model, node):
    mark_off_chip = _check_attributes_are_valid(
        model, node, {'alpha': 1.0, 'beta': 1.0, 'transA': 0.0, 'transB': 1.0})
    return mark_off_chip


def _check_max_pool(model, node):
    # Max pool can have two outputs but is not supported
    if node.output == 2:
        logger.info(f"Node {node.name} has two outputs which is unsupported on-chip. Marking off-chip.")
        return True

    # Autopad, storage order, ceil mode, and dilations are new attributes. Defaults are ok.
    mark_off_chip = _check_attributes_are_valid(
        model, node, {'auto_pad': 'NOTSET',
                      'storage_order': 0,
                      'ceil_mode': 0,
                      'dilations': _check_dilations})
    return mark_off_chip


def _check_slice(model, node):
    # Only step = 1 is supported
    if len(node.input) == 5:
        steps = model.get_initializer_np(node.input[4])
        if steps != 1:
            logger.info(f"{node.name} not supported on-chip since step != 1. Marking off-chip.")
            return True
    return False


def _check_resize(model, node):
    def is_int(x):
        return modf(x)[0] == 0

    def check_resize_scales(node, attribute, actual_value):
        if any(x == 0 or not is_int(x) for x in actual_value):
            raise _IncompatibleAttrValue(f"Node {node.name} not supported on-chip since not all scales ({actual_value})"
                                         f"are positive integers.")

    if model.hwconfig.name == HardwareType.BOREAS:
        attributes = {
            'coordinate_transformation_mode': 'asymmetric',
            'exclude_outside': 0,
            'mode': 'nearest',
            'nearest_mode': _make_check_one_of('floor', 'round_prefer_floor'),
            'scales': check_resize_scales
        }
    elif model.hwconfig.name == HardwareType.DENALI:
        attributes = {
            'coordinate_transformation_mode': _make_check_one_of('asymmetric', 'half_pixel'),
            'exclude_outside': 0,
            'mode': _make_check_one_of('nearest', 'linear'),
            'nearest_mode': _make_check_one_of('floor', 'round_prefer_floor'),
            'scales': check_resize_scales
        }
    mark_off_chip = _check_attributes_are_valid(model, node, attributes)
    return mark_off_chip


checker_functions = {
    ONNXType.CONV: _check_conv,
    ONNXType.GEMM: _check_gemm,
    ONNXType.MAXPOOL: _check_max_pool,
    ONNXType.SLICE: _check_slice,
    ONNXType.RESIZE: _check_resize,
}
