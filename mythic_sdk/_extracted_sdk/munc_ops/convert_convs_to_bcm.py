# noqa-flake8-docstrings
from munc._constants import ONNXType
from munc import _constants, _node_utils, _base_op
from munc.bcm import bcm_utils

ONNX_TO_BCM_OP_TYPE = {ONNXType.CONV: ONNXType.BCMCONV2D,
                       ONNXType.GEMM: ONNXType.BCMLINEAR}


class ConvertConvsToBCM(_base_op.BaseOp):
    """
    Converts Conv/Linear layer to BCM layer.

    This Op converts convolutions or linear layers to BCM layers. Depending on the parameters passed, it uses
    trainable/non-trainable BCM layer.
    """

    def __init__(self, bcm_class_str="munc_fp", bcm_attr_str=None,
                 acm_hardware_name=None, acm_noise_name=None):
        """
        Initialize the object.

        Parameters
        ----------
        bcm_class_str : str
            Attribute value for bcm_utils.MMA_CLASS_ATTRIBUTE_NAME attribute.
        bcm_attr_str : str
            Attribute value for bcm_utils.MMA_ATTR_ATTRIBUTE_NAME attribute.
        """
        self._bcm_class_str = bcm_class_str
        self._bcm_attr_str = bcm_attr_str

        self._node_mapping = ONNX_TO_BCM_OP_TYPE

    def _get_info(self):
        return {
            'node_count': 1,
            'pattern': [{
                'op_type': list(self._node_mapping.keys())
            }],
            'run_msg': 'Converting convolutions to ACM model...',
            'off_chip': _constants.OFFCHIP_IGNORE,
            'requires_stat_collection': _constants.STATS_NOT_REQUIRED
        }

    def _run(self, nodes):
        # Node auxiliary
        node = nodes[0]

        # Do the conversion
        node.op_type = self._node_mapping[node.op_type]

        # Remove existing BCM attributes if they are already present, such as from the conversion by ConvertBCMToConv
        possible_bcm_attributes = [bcm_utils.MMA_CLASS_ATTRIBUTE_NAME, bcm_utils.MMA_ATTR_ATTRIBUTE_NAME]
        for possible_bcm_attribute in possible_bcm_attributes:
            _node_utils.remove_attribute(node, possible_bcm_attribute)

        if _node_utils.is_digital_onchip(node):
            bcm_class_str = bcm_utils.int8model.FACTORY_NAME
            bcm_attr_str = bcm_utils.int8model.FACTORY_NAME
        else:
            bcm_class_str = self._bcm_class_str
            bcm_attr_str = self._bcm_attr_str

        # Else we use the legacy BCM models
        _node_utils.create_attribute_with_value(node, bcm_utils.MMA_CLASS_ATTRIBUTE_NAME, bcm_class_str)
        if bcm_attr_str is not None:
            _node_utils.create_attribute_with_value(node, bcm_utils.MMA_ATTR_ATTRIBUTE_NAME, bcm_attr_str)
