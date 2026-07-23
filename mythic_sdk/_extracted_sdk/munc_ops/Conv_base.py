"""
ONNX Conv (opset 20).

Documentation: https://onnx.ai/onnx/operators/onnx__Conv.html

onnx.defs schema since_version=11 (domain='').
Inputs:
  - [required] X: T
  - [required] W: T
  - [optional] B: T
Outputs:
  - [required] Y: T
Attributes:
  - [optional] auto_pad: STRING = 'NOTSET'
  - [optional] dilations: INTS
  - [optional] group: INT = 1
  - [optional] kernel_shape: INTS
  - [optional] pads: INTS
  - [optional] strides: INTS
"""
import logging
import torch.nn.functional as F
from munc._o2t_module import O2TModule
from munc._value_converters import optional
from munc import _node_utils
from munc._constants import QAT_TAG
from munc._pytorch.layers import quantize_and_clip, quantize


logger = logging.getLogger()


def _get_padding_attribute(pads):
    """Get padding attributes and warn if given paddings are asymmetric (different starting and end).

    If asymmetric, symmetric paddings will be used instead.
    """
    if isinstance(pads, int):
        return pads
    if len(pads) == 2:
        pad_attribute = pads[0]
        if pads[0] != pads[1]:
            logger.warning(f'Start and end paddings must be the same in Conv1D. Got pads = {pads}. Using symmetric '
                           f'padding {2 * pad_attribute} instead.')
    elif len(pads) == 4:
        pad_attribute = (pads[0], pads[1])
        if pads[0] != pads[2] or pads[1] != pads[3]:
            logger.warning(f'Start and end paddings must be the same in Conv2D. Got pads = {pads}. Using symmetric '
                           f'padding {2 * pad_attribute} instead.')
    elif len(pads) == 6:
        pad_attribute = (pads[0], pads[1], pads[2])
        if pads[0] != pads[2] or pads[1] != pads[3] or pads[2] != pads[4]:
            logger.warning(f'Start and end paddings must be the same in Conv3D. Got pads = {pads}. Using symmetric '
                           f'padding {2 * pad_attribute} instead.')
    else:
        raise ValueError(
            f'Length of padding attribute for Conv node is {len(pads)};'
            'expected 2 for Conv1D, 4 for Conv2D, or 6 for Conv3D')
    return pad_attribute


def _int_or_tuple(x):
    return x if isinstance(x, int) else tuple(x)


class Conv(O2TModule):
    attr_defs = [('auto_pad', 'NOTSET'),
                 ('dilations', 1, _int_or_tuple),
                 ('group', 1),
                 ('kernel_shape', None, optional(_int_or_tuple)),
                 ('pads', 0, _get_padding_attribute),
                 ('strides', 1, _int_or_tuple),
                 ]
    attr_prefix = '_attr_'
    arg_check = (2, 3)

    def __init__(self, node, o2t):
        super().__init__(node, o2t)
        self.qat = _node_utils.is_qat(node)
        if self.qat:
            self.weight_min, self.weight_max, self.bias_min, self.bias_max = _node_utils.get_attribute_value(
                node, QAT_TAG)

        # What does 'VALID' mean exactly? 'NOTSET' means explicit padding, that's what we support.
        # I left valid here, because I do not understand it and removing it may break something. -- Ilya, July 2022.
        assert self._attr_auto_pad in ['NOTSET', 'VALID']

        # Deduce kernel shape (per onnx spec)
        if self._attr_kernel_shape is None:
            edge_weight = node.input[1]
            edge_root_weight = o2t._model.get_root_edge(edge_weight)
            weight = o2t._model.get_initializer_np(edge_root_weight)
            self._attr_kernel_shape = weight.shape[2:]

        if isinstance(self._attr_dilations, tuple) and len(self._attr_dilations) == 2 \
           and self._attr_dilations[0] != self._attr_dilations[1]:
            raise Exception(f'Dilation must be same in both directions, got {self._attr_dilations}')

    def _get_conv_layer(self, input_dims):
        """Return the appropiate Convolution layer for the input data baseed on it's dimensions.

        Parameters
        ----------
        input_dims : int
            Number of dimensions in the input data.

        Returns
        -------
        callable
            The appropiate torch convolution layer.

        Raises
        ------
        NotImplementedError
            If the number of dimensions do not correspond to Conv 1D, 2D, or 3D.
        """
        if input_dims == 5:
            return F.conv3d
        if input_dims == 4:
            return F.conv2d
        elif input_dims == 3:
            return F.conv1d
        else:
            raise NotImplementedError("Only Conv1D, Conv2D, and Conv3D are supported in TorchNet.")

    def forward(self, x, y, z=None):
        # if z (bias) has been split into multiple rows, we must combine them to use Pytorch Conv2d op.
        if z is not None and z.ndim > 1:
            z = z.sum(axis=1)

        input_data_dimensions = len(x.size())
        conv_layer = self._get_conv_layer(input_data_dimensions)

        # Quantize weights and biases if off-chip and QAT enabled
        if self.qat:
            y = quantize_and_clip(y, self.weight_min, self.weight_max)
            if z is not None:
                z = quantize_and_clip(z, self.bias_min, self.bias_max)

        result = conv_layer(
            x, y, bias=z, stride=self._attr_strides, padding=self._attr_pads, dilation=self._attr_dilations,
            groups=self._attr_group)
        result = quantize(result) if self.qat else result
        return result
