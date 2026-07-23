from functools import partial

import torch.nn.functional as F
import numpy as np

from munc._value_converters import optional, typed_list
from munc._o2t_ops.mythic_mma import MythicMMA


def _int_or_int_vector(x):
    return x.astype(int) if isinstance(x, np.ndarray) else int(x)


def _float_or_float_vector(x):
    return x.astype(float) if isinstance(x, np.ndarray) else float(x)


class MythicConv2d(MythicMMA):
    attr_prefix = ""
    attr_defs = [('__iFSR', None, _float_or_float_vector),
                 ('__pFSR', None, int),
                 ('__multiplier', None, _int_or_int_vector),
                 ('__shift', None, _int_or_int_vector),
                 ('__trainable_dsf', 0, int),
                 ('__activation', None, str),
                 ('__activation_clip', None, optional(typed_list(int))),
                 ('__weight_scale', 1.0, float),
                 ('dilations', (1, 1), optional(typed_list(int))),
                 ('kernel_shape', None, optional(typed_list(int))),
                 ('pads', (0, 0, 0, 0), optional(typed_list(int))),
                 ('strides', (1, 1), optional(typed_list(int))),
                 ('group', 1, int),
                 ('auto_pad', "NOTSET", str)]

    def _init_mma_func(self):
        if self.pads is not None:
            pads = (self.pads[0], self.pads[1])
            if self.pads[0] != self.pads[2] or self.pads[1] != self.pads[3]:
                raise ValueError('Start and end paddings must be the same')
        self.mma_func = partial(F.conv2d, stride=self.strides, padding=pads,
                                dilation=self.dilations, groups=self.group)
