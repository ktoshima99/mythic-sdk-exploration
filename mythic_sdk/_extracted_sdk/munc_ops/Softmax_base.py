"""
ONNX Softmax (opset 13).

Documentation: https://onnx.ai/onnx/operators/onnx__Softmax.html

onnx.defs schema since_version=13 (domain='').
Inputs:
  - [required] input: T
Outputs:
  - [required] output: T
Attributes:
  - [optional] axis: INT = -1
"""
from math import prod
import torch
from munc._o2t_module import o2t_func_module, use_node_opset


@o2t_func_module([('axis', -1, int)], attr_prefix="_attr_", init=(use_node_opset,))
def Softmax(self, x):
    if self.opset <= 11:
        shape = x.shape
        # Coerce to a 2D vector
        first_dim = prod(shape[:self._attr_axis])
        second_dim = prod(shape[self._attr_axis:])
        x = x.reshape(first_dim, second_dim)
        x = torch.softmax(x, dim=1)
        x = x.reshape(shape)
        return x
    else:
        return torch.softmax(x, dim=self._attr_axis)


Softmax._supported_opsets = True
