# noqa-flake8-docstrings
import logging
import numpy as np
import torch

from munc.bcm import bcm_utils
from munc import _node_utils
from munc.bcm.ace_digital_datapath import ace_digital_datapath_factory
from munc.bcm.salu_datapath import SALUDatapathInt8


logger = logging.getLogger(__name__)


class BCMMMAOp(torch.nn.Module):
    """A super class of Linear and Conv layers to be run with the Boreas Compute Model.

    The subclasses need to implement two methods `torch_mma_op` and `np_mma_op`.
    """

    def __init__(self, node, o2t):
        super().__init__()

        self.iFSR, self.pFSR, self.dsf_mult, self.dsf_shft, self.activation = bcm_utils._get_hw_attrs_from_node(node)
        self.mma_class, self.mma_attr, self.is_torch, self.fp16 = bcm_utils._get_mma_class_from_node(node)
        self.is_digital_onchip = _node_utils.is_digital_onchip(node)
        self.digital_datapath = ace_digital_datapath_factory(fp_mode=False)

        self._mma = None
        self.name = node.name

        self.signed_input = _node_utils.is_node_signed(node)
        self.duplicate_weight = True
        self.debug_container = None

    @property
    def layer(self):
        """Attribute needed to comply with api."""
        return self

    @property
    def mma(self):
        return self._mma

    def layer_gen(self, weight, bias):
        if self.duplicate_weight and self.signed_input:
            weight = torch.cat([weight, -weight], dim=1)

        weight, bias = bcm_utils.clean_weights_and_bias(weight, bias, str(self), to_numpy=not self.is_torch,
                                                        to_fp16=self.fp16)
        self.out_channels = weight.shape[0]
        self._mma = self.mma_class(weight, bias, mma_attr=self.mma_attr, iFSR=self.iFSR, pFSR=self.pFSR, name=self.name)

        logger.debug(f"[{self}] -- Initializing {self.mma}")
        logger.debug(f"\t[name] -- {self.name}")
        logger.debug(f"\t[pFSR] -- {self.pFSR}")
        logger.debug(f"\t[iFSR] -- {self.iFSR}")
        logger.debug(f"\t[dsf_mult] -- {self.dsf_mult}")
        logger.debug(f"\t[dsf_shft] -- {self.dsf_shft}")
        logger.debug(f"\t[activation] -- {self.activation}")
        bcm_utils.dump_attrs(self.mma_attr)

        if not hasattr(self.mma, 'randomize'):
            logger.warning("Warning, you are using an MMA which is not randomized per image")

    def torch_layer_op(self, torch_X):
        """Should we call randomize from inside dot function?"""
        self.mma.randomize()
        return self.torch_mma_op(torch_X)

    def np_layer_op(self, torch_X):
        """BCM operations called during each forward pass

        For every image, the BCM MMA is re-randomized. Additionally, a new
        random state is passed in to ensure the noise calculations receive a
        difference random seed for each worker
        """
        if hasattr(self.mma, 'randomize'):
            # a new random state needs to be passed in to make noise multiprocessing safe
            self.mma.randomize(random_state=np.random.RandomState())
        X = bcm_utils.to_numpy(torch_X)
        Y = np.array(self.np_mma_op(X), dtype=np.float32)
        return torch.from_numpy(Y).to(torch_X.device)

    def forward(self, torch_X, weights, biases=None):

        if self._mma is None:
            self.layer_gen(weights, biases)

        if self.duplicate_weight and self.signed_input:
            pos_inputs = torch.clamp(torch_X, min=0)
            neg_inputs = torch.clamp(-torch_X, min=0)
            torch_X = torch.cat([pos_inputs, neg_inputs], dim=1)

        if self.is_torch:
            y = self.torch_layer_op(torch_X)
        else:
            y = self.np_layer_op(torch_X)

        y = self.digital_datapath.compute(y, self.dsf_mult, self.dsf_shft, self.activation)
        y = y.type(torch_X.dtype)

        return y

    def register_forward_debug(self):
        """Register a forward debug container with this node."""
        if hasattr(self._mma, 'register_forward_debug'):
            self.debug_container = self._mma.register_forward_debug()
        return self.debug_container

    def unregister_forward_debug(self):
        """Unregister a forward debug container with this node."""
        if hasattr(self._mma, 'register_forward_debug'):
            self.debug_container = self._mma.unregister_forward_debug()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_mma']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mma = None


class BCMLinear(BCMMMAOp):
    """Linear layer to be run with the Boreas Compute Model.

    Linear layer dot product is replaced with an MMA dot product from the
    boreas compute model.
    """

    def __init__(self, node, o2t):
        super().__init__(node, o2t)

        if len(node.input) != 3:
            raise ValueError("GEMM (or Linear) Node must have 3 inputs")

        # Check and fetch attributes
        alpha, beta, transA, transB = _node_utils.fetch_attributes(
            node, ['alpha', 'beta', 'transA', 'transB'], [1.0, 1.0, 0, 0]
        )
        if not (alpha == 1.0 and beta == 1.0 and not transA and transB):
            raise ValueError(f'The combination of attributes alpha={alpha}, beta={beta}, '
                             f'transA={transA}, transB={transB} is not supported.'
                             ' The only support combination of values is 1, 1, False, True.')

    def torch_mma_op(self, x):
        return self.mma.dot(x)

    def np_mma_op(self, X):
        return [self.mma.dot(x) for x in X]


class BCMConv2d(BCMMMAOp):
    def __init__(self, node, o2t):
        super().__init__(node, o2t)

        dilations, group, kernel_shape, pads, strides, auto_pad = _node_utils.fetch_attributes(
            node, ['dilations', 'group', 'kernel_shape', 'pads', 'strides', "auto_pad"], [1, 1, None, 0, 1, "NOTSET"])

        # Deduce kernel shape (per onnx spec)
        if kernel_shape is None:
            edge_weight = node.input[1]
            edge_root_weight = o2t._model.get_root_edge(edge_weight)
            weight = o2t._model.get_initializer_np(edge_root_weight)
            kernel_shape = weight.shape[2:]

        if isinstance(pads, int):
            self.padding = [int(pads), int(pads)]
        else:
            self.padding = (pads[0], pads[1])
            if pads[0] != pads[2] or pads[1] != pads[3]:
                raise ValueError('Start and end paddings must be the same')

        self.kernel_shape = [int(shape) for shape in kernel_shape]

        if isinstance(strides, int):
            self.strides = [int(strides), int(strides)]
        else:
            self.strides = [int(stride) for stride in strides]

        if isinstance(dilations, int):
            self.dilations = [int(dilations), int(dilations)]
        else:
            self.dilations = [int(dialation) for dialation in dilations]

        self.group = group

    def torch_mma_op(self, x):
        return bcm_utils.acm_conv2d(x,
                                    self.mma.dot,
                                    self.kernel_shape,
                                    self.padding,
                                    self.strides,
                                    self.group)

    def np_mma_op(self, X):
        padding = self.padding + self.padding
        return [bcm_utils.np_conv2d(x, self.mma.dot, self.kernel_shape, padding, self.strides, self.out_channels)
                for x in X]


class BCMSum(torch.nn.Module):
    def __init__(self, node, o2t):
        super().__init__()

        if len(node.input) == 0:
            raise ValueError("Sum node must have at least one input")

        # Check and fetch attributes
        _ = _node_utils.fetch_attributes(node, [], [])

        *_, self.mul_output, self.shift_output, self.activation = bcm_utils._get_hw_attrs_from_node(node)
        self.name = node.name
        self.debug_container = None

    # Create layer
    def _layer_op(self, *args):
        # Start from the first one
        acc = args[0]

        # Accumulate
        for i in range(1, len(args)):
            acc = acc + args[i]

        acc = SALUDatapathInt8.scale_and_bitshift(acc, self.mul_output, self.shift_output)
        acc = SALUDatapathInt8().apply_activation(acc, self.activation)

        # Return the result
        return acc

    def forward(self, *args):
        result = self._layer_op(*args)
        return result


class BCMAdd(torch.nn.Module):
    def __init__(self, node, o2t):
        super().__init__()

        # We expect two inputs to add; raise error if different
        if len(node.input) != 2:
            raise ValueError("Operation needs two inputs. "
                             f"Received {len(node.input)}")

        # We don't need to retrieve any attributes from node
        _node_utils.fetch_attributes(node, [], [])
        self.name = node.name

        (self.mul_input1, self.shift_input1, self.mul_input2, self.shift_input2,
         self.activation, self.mul_output, self.shift_output) = bcm_utils._get_hw_attrs_from_node_add(node)
        assert self.mul_input1 == 1 and self.shift_input1 == 0 and self.mul_input2 == 1 and self.shift_input2 == 0, \
            "The compiler does not support ADD input multipliers"
        self.debug_container = None

    # Create layer
    def _layer_op(self, x, y):
        x = SALUDatapathInt8.scale_and_bitshift(x, self.mul_input1, self.shift_input1)
        y = SALUDatapathInt8.scale_and_bitshift(y, self.mul_input2, self.shift_input2)
        z = x + y
        z = SALUDatapathInt8.scale_and_bitshift(z, self.mul_output, self.shift_output)
        z = SALUDatapathInt8().apply_activation(z, self.activation)
        return z

    def forward(self, x, y):
        result = self._layer_op(x, y)
        return result


class BCMMul(torch.nn.Module):
    def __init__(self, node, o2t):
        super().__init__()

        # We expect two inputs to mult; raise error if different
        if len(node.input) != 2:
            raise ValueError("Operation needs two inputs. "
                             f"Received {len(node.input)}")

        # We don't need to retrieve any attributes from node
        _node_utils.fetch_attributes(node, [], [])
        self.name = node.name

        (self.mul_output, self.shift_output) = bcm_utils._get_hw_attrs_from_node_mult(node)
        self.debug_container = None

    def forward(self, x, y):
        # `y` is a constant operand represented as a multiplier and a shift. It's not used in the computation.
        return SALUDatapathInt8.scale_and_bitshift(x, self.mul_output, self.shift_output)
