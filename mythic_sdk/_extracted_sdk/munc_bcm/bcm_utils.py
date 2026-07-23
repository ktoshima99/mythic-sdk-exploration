# noqa-flake8-docstrings
import os
import logging
import numpy as np
import torch
from functools import partial
from torch import multiprocessing
import torch.nn.functional as F

from munc import _node_utils
from munc.bcm.registry import MMA, MMAAttributeFactory
from munc.bcm.bcm_models import digitalmodel, simplemodel, fpmodel, int8model
from munc.bcm.bcm_models import trainingacm
from munc.bcm.bcm_models import acmsignoffmodel


logger = logging.getLogger(__name__)

MMA_CLASS_ATTRIBUTE_NAME = '__mma_class'
MMA_ATTR_ATTRIBUTE_NAME = '__mma_attr'


class ParallelInference:
    """Run BCM validation on multiple cores

    Note
    ----
    *before* importing numpy you must set omp_num_threads to 1. If you don't,
    multithreadding in numpy causes issues with the GIL to slow down the program

    ~~~
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    ~~~

    Side effects
    ------------
    - sets model to .eval
    - maps model to cpu device
    - restricts number of torch threads to 1
    - enables torch share_memory on the model

    Example
    -------
    ~~~
    with inference.get_inference_ctx() as inference:
        with bcm_utils.ParallelInference(inference, workers=5) as pmodel:
            for x, _ in loader_test:
                outputs = pmodel(x)
    ~~~
    """
    def __init__(self, inference, workers):

        if os.getenv("OMP_NUM_THREADS") != "1":
            err_message = "To use multiprocessing the environment variable 'OMP_NUM_THREADS' must be set to 1"
            raise OSError(err_message)

        self.inference = inference
        self.workers = workers

        self.inference._o2t.eval()
        self.inference._o2t.cpu()
        self.inference._o2t.share_memory()

        ctx = multiprocessing.get_context("spawn")
        self.pool = ctx.Pool(workers)

    def __call__(self, x):
        x = torch.chunk(x, self.workers)
        y = self.pool.map(self.inference.predict, x)
        return torch.cat(y)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.pool.close()
        self.pool.join()


@torch.no_grad()
def clean_weights_and_bias(weight_tensor, bias_tensor, layer_name, to_numpy=True, to_fp16=False):
    weights = _convert_munc_weights_to_bcm(weight_tensor, to_numpy)
    bias = _convert_munc_bias_to_bcm(bias_tensor)
    weights, bias = _assert_weights_in_mma_range(layer_name, weights, bias)

    if to_numpy:
        weights = weights.cpu().detach().numpy()
        bias = bias.cpu().detach().numpy()
    elif to_fp16:
        # Only convert to FP16 for Torch models
        weights = weights.type(torch.float16)
        bias = bias.type(torch.float16)

    return weights, bias


def _convert_munc_weights_to_bcm(weight, flatten=True):
    """Normalize weights to from int8 to fp

    If weights from a convolution layer, reorder weights to be 2x2 matrix. i.e.
    unravel the individual filters
    """
    weight = weight.reshape(weight.shape[0], -1) if flatten else weight
    return torch.mul(weight, 1./128)


def _convert_munc_bias_to_bcm(bias):
    """Normalize biases to from int8 to fp
    """
    return torch.mul(bias, 1./128)


def _assert_weights_in_mma_range(op_name, weight, bias):
    """Helper function to ensure weights are in range"""
    if torch.abs(weight).max() > 1:
        _max = weight.max()
        _min = weight.min()
        logger.error(f"[{op_name}] Weight value in range {_min:.3f}..{_max:.3f}. Clipping to range -1..1")
        weight = torch.clamp(weight, -1, 1)
    if torch.abs(bias).max() > 1:
        _max = bias.max()
        _min = bias.min()
        logger.error(f"[{op_name}] Bias value in range {_min:.3f}..{_max:.3f}. Clipping to range -1..1")
        bias = torch.clamp(bias, -1, 1)
    return weight, bias


def _get_hw_attrs_from_node_add(node):
    mul_input1 = _node_utils.get_attribute_value(node, "__input1_multiplier", 1)
    shift_input1 = _node_utils.get_attribute_value(node, "__input1_shift", 0)
    mul_input2 = _node_utils.get_attribute_value(node, "__input2_multiplier", 1)
    shift_input2 = _node_utils.get_attribute_value(node, "__input2_shift", 0)
    activation = _node_utils.get_attribute_value(node, "__activation")
    mul_output = _node_utils.get_attribute_value(node, "__multiplier")
    shift_output = _node_utils.get_attribute_value(node, "__shift")
    return mul_input1, shift_input1, mul_input2, shift_input2, activation, mul_output, shift_output


def _get_hw_attrs_from_node_mult(node):
    mul_output = _node_utils.get_attribute_value(node, "__multiplier")
    shift_output = _node_utils.get_attribute_value(node, "__shift")
    return mul_output, shift_output


def _get_hw_attrs_from_node(node):
    iFSR = _node_utils.get_attribute_value(node, '__iFSR')
    pFSR = _node_utils.get_attribute_value(node, '__pFSR')

    try:
        dsf_mult = _node_utils.get_attribute_value(node, '__multiplier')
    except:  # noqa: E722
        dsf_mult = _node_utils.get_attribute_value(node, '__act_multiplier')

    try:
        dsf_shft = _node_utils.get_attribute_value(node, '__shift')
    except:  # noqa: E722
        dsf_shft = _node_utils.get_attribute_value(node, '__act_shift')

    activation = _node_utils.get_attribute_value(node, "__activation")

    return iFSR, pFSR, dsf_mult, dsf_shft, activation


def _get_mma_class_from_node(node):
    """Retrieve the BCM MMA using the onnx node string
    """
    # Get class with name
    mma_class_str = _node_utils.get_attribute_value(node, MMA_CLASS_ATTRIBUTE_NAME)
    is_torch = 'munc' in mma_class_str

    try:
        mma_class = MMA[mma_class_str]
    except KeyError:
        raise KeyError(f"Unable to find MMA of type {mma_class_str}")
    else:
        logger.debug(f"Using MMA: {mma_class_str}")

    mma_attr_str = _node_utils.get_attribute_value(node, MMA_ATTR_ATTRIBUTE_NAME)
    if mma_attr_str is None:
        mma_attr = None
    else:
        try:
            mma_attr = MMAAttributeFactory.from_string(mma_attr_str)
        except KeyError:
            logger.info(f"Unable to find attributes of type '{mma_attr_str}' using defaults")
            mma_attr = None
        else:
            logger.debug(f"Using attributes: {mma_attr_str}")

    fp16_value = _node_utils.get_attribute_value(node, '__fp16')
    fp16 = fp16_value is not None

    return mma_class, mma_attr, is_torch, fp16


def to_numpy(tensor):
    # We need to cast as a contiguous array to use the stride trick for convolution
    return np.ascontiguousarray(tensor.cpu().detach().numpy(), dtype=np.float32)


def dump_attrs(attrs):
    """TEMP. This will be available in future release of bcm"""
    if attrs is not None:
        logger.debug("Attributes used:")
        for key, val in attrs.__dict__.items():
            logger.debug(f"\t{key:.<20}{val}")


def img2col(input_data, filter_h, filter_w, pad_pairs, stride=(1, 1)):
    """
    Stolen from:
    https://stackoverflow.com/questions/50292750/python-the-implementation-of-im2col-which-takes-the-advantages-of-6-dimensional
    """
    img = np.pad(input_data, pad_pairs, 'constant')
    stride_h, stride_w = stride
    C, H, W = img.shape
    CC, HH, WW = img.strides
    out_h = (H - filter_h) // stride_h + 1
    out_w = (W - filter_w) // stride_w + 1
    col = np.lib.stride_tricks.as_strided(
        img,
        (out_h, out_w, C, filter_h, filter_w),
        (stride_h * HH, stride_w * WW, CC, HH, WW)
    ).astype(np.float32)
    # output should be C * filter_h * filter_w
    col = col.reshape(np.multiply.reduceat(col.shape, (0, 2)))
    return col


def np_conv2d(img, dot_op, kernel_size, pad, stride, out_channels):
    """
    https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    """
    d_x, h_x, w_x = img.shape
    h_filter, w_filter = kernel_size
    h_out = (h_x - h_filter + 2*pad[0]) // stride[0] + 1
    w_out = (w_x - w_filter + 2*pad[1]) // stride[1] + 1

    pad_pairs = [(0, 0), (pad[0], pad[1]), (pad[2], pad[3])]
    img_windows = img2col(img, h_filter, w_filter, pad_pairs=pad_pairs, stride=stride)
    out = np.empty(shape=(len(img_windows), out_channels), dtype=np.float32)
    for i, window in enumerate(img_windows):
        out[i] = dot_op(window)
    out = np.transpose(out)
    return out.reshape(out_channels, h_out, w_out)


def acm_conv2d(img, dot, kernel_size, pad, stride, group):
    dot_op = partial(F.conv2d, stride=stride, padding=pad, groups=group)
    return dot(img, dot_op)


def simple_dot_product(input, ace_dot):
    """Call `ace_dot` with `input`.

    A wrapper on top of `ace_dot` that computes standard dot and can be used to configure LayerCM to implement a linear
    layer.

    Parameters
    ----------
    input : tensor
        A layer input
    ace_dot : function
        An ACM.dot method.

    Returns
    -------
    tensor
        ace_dot(input)
    """
    return ace_dot(input)


MMA.register_builder(digitalmodel.FACTORY_NAME, digitalmodel.PytorchDigitalMMA)
MMAAttributeFactory.register_builder(digitalmodel.FACTORY_NAME, digitalmodel.DigitalAttributes())
MMA.register_builder(simplemodel.FACTORY_NAME, simplemodel.PytorchSimpleMMA)
MMAAttributeFactory.register_builder(simplemodel.FACTORY_NAME, simplemodel.SimpleAttributes())
MMA.register_builder(fpmodel.FACTORY_NAME, fpmodel.PytorchFloatingPointMMA)
MMAAttributeFactory.register_builder(fpmodel.FACTORY_NAME, fpmodel.FloatingPointAttributes())
MMA.register_builder(trainingacm.FACTORY_NAME, trainingacm.PytorchTrainingACM)
MMAAttributeFactory.register_builder(trainingacm.FACTORY_NAME, trainingacm.PytorchTrainingACMAttributes())
MMA.register_builder(acmsignoffmodel.FACTORY_NAME, acmsignoffmodel.PytorchACMSignoffMMA)
MMAAttributeFactory.register_builder(acmsignoffmodel.FACTORY_NAME, acmsignoffmodel.ACMSignoffAttributes())
MMA.register_builder(int8model.FACTORY_NAME, int8model.PytorchInt8MMA)
MMAAttributeFactory.register_builder(int8model.FACTORY_NAME, int8model.PytorchInt8MMA)
