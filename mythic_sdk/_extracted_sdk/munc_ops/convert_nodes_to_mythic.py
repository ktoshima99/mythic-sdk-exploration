from munc._constants import ONNXType
from munc import _constants, _node_utils, _base_op


class ConvertNodesToMythic(_base_op.BaseOp):
    """Rename onnx nodes so they are mapped to custom layers.

    Changes the name of onnx nodes so that a custom o2t layer can be used for
    _pytorch training.

    Attaches the name of the hw_config used by session so that noise values and
    hw parameters (e.g. bit depth) can be referenced by the mythic node

    Attributes
    ----------
    DEFAULT_MYTHIC_NODE_MAP : dict
        Default onnx node to _pytorch layer conversion dictionary

    Note
    ----
    This converter can be used as a general purpose node type mapping, it isn't
    limited to the mythic node. A future refactor could consolidate other converters
    to use the logic here
    """

    DEFAULT_MYTHIC_NODE_MAP = {
        ONNXType.CONV: ONNXType.MYTHIC_CONV,
        ONNXType.GEMM: ONNXType.MYTHIC_LINEAR,
        ONNXType.MUL: ONNXType.MYTHIC_QUANTIZED_MUL,
        ONNXType.SOFTMAX: ONNXType.MYTHIC_SOFTMAX,
        ONNXType.MATMUL: ONNXType.MYTHIC_MATMUL,
    }

    def __init__(self, mythic_node_map=None, trainable_dsfs=_constants.DEFAULT_DSF_PARAMETER_GROUP):
        """Initialize the mapping of onnx nodes to _pytorch layers.

        Parameters
        ----------
        mythic_node_map : dict
            A dictionary specifying the onnx node type (key) and the target
            custom _pytorch o2t layer to be run in its place (value)
        """
        self.mythic_node_map = mythic_node_map or self.DEFAULT_MYTHIC_NODE_MAP
        self.trainable_dsfs = trainable_dsfs

    def _get_info(self):
        keys = list(self.mythic_node_map.keys())

        return {
            'node_count': 1,
            'pattern': [{'op_type': keys}],
            'run_msg': 'Converting convs and gemms to Mythic nodes and quantize on-chip mul operations...',
            'off_chip': _constants.OFFCHIP_IGNORE,
            'requires_stat_collection': _constants.STATS_NOT_REQUIRED
        }

    def _run(self, nodes):
        node = nodes[0]

        if node.op_type in self.mythic_node_map:
            node.op_type = self.mythic_node_map[node.op_type]
            if self.trainable_dsfs:
                _node_utils.create_attribute_with_value(node, '__trainable_dsf', self.trainable_dsfs)
            else:
                _node_utils.remove_attribute(node, '__trainable_dsf')
