from munc._constants import ONNXType
from munc import _constants, _node_utils, _base_op
from munc.bcm import bcm_utils

BCM_TYPES = [ONNXType.BCMCONV2D, ONNXType.BCMLINEAR]


class SwitchBCM(_base_op.BaseOp):
    """Switch between BCM classes and/or attributes.

    This Op simply changes the following BCM nodes attributes/classes with the new given class/attribute strings:

    - ONNXType.BCMCONV2D
    - ONNXType.BCMLINEAR

    MUNC uses these to set the logic to use with the BCM. Valid BCM classes are munc_digital, munc_fp, munc_simple, and
    munc_tacm.
    """

    def __init__(self, bcm_class_str=None, bcm_attr_str=None):
        self._bcm_class_str = bcm_class_str
        self._bcm_attr_str = bcm_attr_str
        self._node_types = BCM_TYPES

    def _get_info(self):
        return {
            'node_count': 1,
            'pattern': [{
                'op_type': self._node_types
            }],
            'run_msg': f'Switching BCM model to {self._bcm_class_str}:{self._bcm_attr_str}...',
            'off_chip': _constants.OFFCHIP_UNDEFINED,
            'requires_stat_collection': _constants.STATS_NOT_REQUIRED
        }

    def _run(self, nodes):
        # Node auxiliary
        node = nodes[0]

        if _node_utils.is_digital_onchip(node):
            bcm_class_str = bcm_utils.int8model.FACTORY_NAME
            bcm_attr_str = bcm_utils.int8model.FACTORY_NAME
        else:
            bcm_class_str = self._bcm_class_str
            bcm_attr_str = self._bcm_attr_str

        # Do the conversion
        if bcm_class_str is not None:
            _node_utils.set_attribute_value(node, bcm_utils.MMA_CLASS_ATTRIBUTE_NAME, bcm_class_str)
            if _node_utils.is_attribute(node, bcm_utils.MMA_ATTR_ATTRIBUTE_NAME):
                _node_utils.remove_attribute(node, bcm_utils.MMA_ATTR_ATTRIBUTE_NAME)

        if bcm_attr_str is not None:
            try:
                _node_utils.set_attribute_value(node, bcm_utils.MMA_ATTR_ATTRIBUTE_NAME, bcm_attr_str)
            except AttributeError:
                _node_utils.create_attribute_with_value(node, bcm_utils.MMA_ATTR_ATTRIBUTE_NAME, bcm_attr_str)
