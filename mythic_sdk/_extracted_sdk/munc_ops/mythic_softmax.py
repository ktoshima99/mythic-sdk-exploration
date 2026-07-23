import torch
from munc._o2t_module import O2TModule
from munc._pytorch.layers import quantize_and_clip
from munc.hw_specs import get_hw_config
from munc._constants import SOFTMAX_SCALE_IN_ATTR, SOFTMAX_SCALE_OUT_ATTR


class MythicSoftmax(O2TModule):
    """Performs quantized softmax according to the ONNX v13 spec.

    This quantized version has two scaling factors: one for the input tensor and one for the output tensor.
    The input scaling factor compensates for the quantization of the input tensor (returns input
    tensor to the original range). The output scaling factor is used to quantize the output tensor.

    TODO: For now, neither factor is trainable or quantized.
    """
    arg_check = (1, 1)
    attr_prefix = ""
    attr_defs = [(SOFTMAX_SCALE_IN_ATTR, None, float),
                 (SOFTMAX_SCALE_OUT_ATTR, None, float),
                 ('__trainable_dsf', 0, int),
                 ('axis', -1, int),]

    def __init__(self, node, o2t):
        super().__init__(node, o2t)
        self.hw_config = get_hw_config(node)
        self.clip_min, self.clip_max = self.hw_config.SOFTMAX_clip

    def _layer_op(self, x):
        """Scaled softmax according to ONNX v11 spec."""

        # Apply softmax along the second dimension
        x = x * self.softmax_scale_in
        x = torch.softmax(x, dim=self.axis)
        x = x * self.softmax_scale_out
        x = quantize_and_clip(x, self.clip_min, self.clip_max)
        return x

    def forward(self, x):
        result = self._layer_op(x)
        return result
