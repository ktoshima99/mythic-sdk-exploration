import torch
import torch.nn as nn
from munc._constants import ONNXType, ACTIVATION_SHIFT, LRELU_SLOPE, SWISH_ALPHA
from munc._node_utils import get_attribute_value
from munc._session_tools import compute_multiplier_and_shift_torch


class FloorSTE(torch.autograd.Function):
    """Floor autograd function with STE backward pass."""

    @staticmethod
    def forward(ctx, x):
        """Run forward pass where data is floored."""
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        return grad_output


class Quantize(torch.autograd.Function):
    """Quantization autograd function."""

    @staticmethod
    def forward(ctx, x, num_fractional_bits=0):
        """Run forward pass where data is rounded."""
        if num_fractional_bits == 0:
            return x.round()
        else:
            scale = (2 ** num_fractional_bits)
            return (x * scale).round() / scale

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        return grad_output, None


quantize = Quantize.apply
floor_ste = FloorSTE.apply


class ShiftedLeakyReLU(nn.Module):
    """Implementation of Shifted Leaky ReLU."""

    def __init__(self, node):
        super().__init__()
        shift = get_attribute_value(node, ACTIVATION_SHIFT) or 0.0
        slope = get_attribute_value(node, LRELU_SLOPE) or 0.01
        self.leaky_relu = nn.LeakyReLU(negative_slope=slope)

        self.register_buffer("_shift", torch.tensor(float(shift), dtype=torch.float32))

    def forward(self, x):
        """Return shifted Leaky ReLU."""
        return self.leaky_relu(x) + self._shift.to(dtype=x.dtype)


class ShiftedSwish(nn.Module):
    """Implementation of Shifted Swish."""

    def __init__(self, node):
        super().__init__()
        shift = get_attribute_value(node, ACTIVATION_SHIFT)
        if shift is None:
            shift = 0.0
        alpha = get_attribute_value(node, SWISH_ALPHA)
        if alpha is None:
            alpha = 1.0
        self.sigmoid = nn.Sigmoid()

        self.register_buffer("_alpha", torch.tensor(float(alpha), dtype=torch.float32))
        self.register_buffer("_shift", torch.tensor(float(shift), dtype=torch.float32))

    def forward(self, x):
        """Return shifted Swish."""
        a = self._alpha.to(x)
        s = self._shift.to(x)
        return x * self.sigmoid(a * x) + s


ACTIVATIONS = {"hardtanh": None,  # lambda node: torch.nn.Identity(),
               ONNXType.RELU.lower(): None,  # lambda node: torch.relu,
               ONNXType.LEAKY_RELU.lower(): ShiftedLeakyReLU,
               ONNXType.SWISH.lower(): ShiftedSwish
               }


def retrieve_activation_function(node):
    """Initialize the activation class and return a callable activation function."""
    activation_name = get_attribute_value(node, '__activation')
    if (activation_name is not None) and (activation_name not in ACTIVATIONS):
        raise KeyError(f"Activation {activation_name} is not a supported activation on-chip.")

    activation_obj = ACTIVATIONS.get(activation_name)
    if activation_obj is None:
        return None
    else:
        return activation_obj(node)


def clip(x, min_, max_):
    """Clip a given value by a min and max."""
    return torch.clamp(x, min_, max_)


def quantize_and_clip(X, min_, max_):
    """Quantize (round) and clip a given value by a min and max."""
    return clip(quantize(X), min_, max_)


class QuantizeDSF(torch.autograd.Function):
    """Quantizate a DSF as multiplier / (2 ** shift)."""

    @staticmethod
    def forward(ctx, x, hwconfig):
        """Run forward pass where data is rounded."""
        with torch.no_grad():
            mult, shift = compute_multiplier_and_shift_torch(x, hwconfig.ds_max_mult, hwconfig.ds_max_shift)
        return mult / (2 ** shift)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        return grad_output, None


quantize_dsf = QuantizeDSF.apply


class QuantizeFSR(torch.autograd.Function):
    """Quantizate a FSR to the nearest available value."""

    @staticmethod
    def forward(ctx, fsrs, available_fsrs):
        """Quantize each element of 'x' to the closest value in `available_fsrs`."""
        with torch.no_grad():
            # Compute the pairwise differences
            diff = torch.abs((fsrs.unsqueeze(1) if fsrs.dim() > 0 else fsrs) - available_fsrs.unsqueeze(0))
            # Find the indices of the nearest quantization values
            nearest_indices = torch.argmin(diff, dim=1)
            # Return the corresponding quantized values
            return available_fsrs[nearest_indices]

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        return grad_output, None


quantize_fsr = QuantizeFSR.apply
