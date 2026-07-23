import munc
from munc._constants import ONNXType
from munc import _constants, _base_op
import logging

logger = logging.getLogger(__name__)


class MarkDepthwiseConvsAsDigital(_base_op.BaseOp):
    """
    Mark depthwise convs with a __digital_onchip attribute.

    In Onnx, depthwise convolutions are Conv nodes where group == [output channels] and [input channels] == 1.
    On a mythic chip, they are processed in the SALU, i.e. digitally. This op marks these nodes for easier
    identification.
    """

    def _get_info(self):
        def is_unmarked_conv_with_group_not_equal_to_one(node):
            return (node.op_type == ONNXType.CONV
                    and not munc.graph_utils.is_attribute(node, _constants.DIGITAL_ATTRIBUTE_NAME)
                    and munc.graph_utils.get_attribute_value(node, 'group', 1) != 1)
        return {
            'node_count': 1,
            'pattern': is_unmarked_conv_with_group_not_equal_to_one,
            'run_msg': 'Marking depthwise convolution nodes for digital processing...',
            'off_chip': _constants.OFFCHIP_IGNORE,
            'requires_stat_collection': _constants.STATS_NOT_REQUIRED
        }

    def _run(self, nodes):
        node = nodes[0]
        weight_root_edge = self.model.get_root_edge(node.input[1])
        weight = self.model.get_initializer_np(weight_root_edge)
        group = munc.graph_utils.get_attribute_value(node, 'group', 1)
        if weight.shape[0] == group and weight.shape[1] == 1:
            # This is a depthwise convolution
            munc.graph_utils.create_attribute_with_value(node, _constants.DIGITAL_ATTRIBUTE_NAME, '')
        else:
            logger.warning(f"Unexpected conv type: Node {node.name} has group=={group} but is not depthwise.")
