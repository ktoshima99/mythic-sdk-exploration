# noqa-flake8-docstrings
from munc._constants import ONNXType
from munc import _constants, _base_op

BCM_TO_ONNX_OP_TYPE = {
    ONNXType.BCMCONV2D: ONNXType.CONV,
    ONNXType.BCMLINEAR: ONNXType.GEMM,
    ONNXType.ACMCONV2D: ONNXType.CONV,
    ONNXType.ACMLINEAR: ONNXType.GEMM,
}


class ConvertBCMToConvs(_base_op.BaseOp):

    def _get_info(self):
        return {
            'node_count': 1,
            'pattern': [{
                'op_type': list(BCM_TO_ONNX_OP_TYPE.keys())
            }],
            'run_msg': 'Converting BCM model to convolutions...',
            'off_chip': _constants.OFFCHIP_UNDEFINED,
            'requires_stat_collection': _constants.STATS_NOT_REQUIRED
        }

    def _run(self, nodes):
        # Node auxiliary
        node = nodes[0]

        # Do the conversion
        if node.op_type in BCM_TO_ONNX_OP_TYPE:
            node.op_type = BCM_TO_ONNX_OP_TYPE[node.op_type]
