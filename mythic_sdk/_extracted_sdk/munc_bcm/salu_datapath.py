# noqa-flake8-docstrings
import torch
import torch.nn.functional as F


class SALUDatapathFP(object):
    """Base class for computing SALU datapath math."""

    @staticmethod
    def apply_activation(x, activation):
        """Apply the desired activation function.

        Parameters
        ----------
        x : torch.tensor
            Activation values.
        activation : str
            Activation name, e.g. relu, hardtanh.

        Returns
        -------
        torch.tensor
            Activations that have been applied.
        """
        if activation is None:
            y = x
        elif activation.lower() == 'relu':
            # y = F.relu(x)
            # This makes the activation function more like BCM
            y = x.clamp(0.0, 1.0)
        elif activation.lower() == 'hardtanh':
            y = F.hardtanh(x, min_val=-1, max_val=1)
        else:
            raise NotImplementedError(f"{activation} has not been implemented in apply_activation")

        return y


class SALUDatapathInt8(object):
    """Base class for computing SALU datapath math.

    For the activation and scale_and_shift operations, BoreasA and BoreasB are basically the same in the SALU
    except that the range of the multiplier and the shift are larger. The accumulators are large enough to not clip.

    For now, the value of differentiating the models is low, so we will compute both models the same.

    ..note::
        1. The SUM and ADD layers do not have a HW_config attached. So, in the future if we want to
        differentiate the BoreasA/BoreasB behavior on a particular layer, we will have to add that
        setting to layer.
        2. Since the primary difference in the BoreasA/BoreasB is the range of the multiplier and shift
        we could add a range check in the scale_and_bitshift() function.  We don't for two reasons.
        2a. We would not want to take the performance penalty. Instead we control the factors when they are created.
        2b. We don't currently know which HW_config a layer has.

    """

    def __init__(self):
        self.clip_after_relu = True

    @staticmethod
    def scale_and_bitshift(inputs, scaling_factor, shift_factor):
        """Scale and bitshift in int8."""
        out = torch.floor(torch.floor(inputs) * scaling_factor / 2 ** shift_factor)
        return out

    def apply_activation(self, y, activation):
        """Apply the activation in int8."""
        if activation == "relu":
            y = torch.clamp(y, 0, 255 if self.clip_after_relu else None)
        elif activation == "hardtanh":
            y = torch.clamp(y, -128, 127)
        elif activation == "hardsigmoid":
            y = torch.clamp(y, -127, 127) + 128
        elif activation == "swish":
            y = torch.clamp(y, 0, 255)
        else:
            raise Exception(f"Unknown activation: {activation}")
        return y
