# noqa-flake8-docstrings
import torch
import torch.nn.functional as F


class ACEDigitalDatapathFP(object):
    """Base class for computing ACE datapath math."""

    def __init__(self):
        # This makes the activation function more like BCM
        self.clip_after_relu = True

    def compute(self, input, multiplier, shift, activation):
        """Compute datapath math."""
        pass

    def apply_activation(self, x, activation):
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
            y = x.clamp(0.0, 1.0) if self.clip_after_relu else F.relu(x)
        elif activation.lower() == 'hardtanh':
            y = F.hardtanh(x, min_val=-1, max_val=1)
        else:
            raise NotImplementedError(f"{activation} has not been implemented in apply_activation")

        return y


class ACEDigitalDatapathBoreasFP(ACEDigitalDatapathFP):
    """Class for computing Floating Point version of ACE datapath."""

    @classmethod
    def is_registrar_for(cls, fp_mode):
        """Register the subclass.

        BoreasA and BoreasB FP mode are the same
        """
        return fp_mode

    def compute(self, input, multiplier, shift, activation):
        """
        Compute datapath math.

        :param input: tensor data in the range [-255, 255] or [0, 255]
        :param multiplier: integer multiplier
        :param shift: integer shift value
        :param activation: String activation value
        :return: data scaled to [-1, 1] or [0, 1]
        """
        out = input * float(multiplier / (255.0 * 2 ** shift))
        out = self.apply_activation(out, activation)  # Note: this operates in the [-1, 1] range

        return out


class ACEDigitalDatapathInt8(object):
    """Base class for computing ACE datapath math."""

    def __init__(self):
        pass

    def compute(self, input, multiplier, shift, activation):
        """Compute datapath math."""
        pass

    def scale_and_bitshift(self, inputs, scaling_factor, shift_factor):
        """Scale and bitshift in int8."""
        out = torch.floor(torch.floor(inputs) * scaling_factor / 2 ** shift_factor)
        return out

    def apply_activation(self, y, activation):
        """Apply the activation in int8."""
        if activation == "relu":
            y = torch.clamp(y, 0, 255)
        elif activation == "hardtanh":
            y = torch.clamp(y, -128, 127)
        elif activation == "hardsigmoid":
            y = torch.clamp(y, -127, 127) + 128
        elif activation == "swish":
            y = torch.clamp(y, 0, 255)
        else:
            raise Exception(f"Unknown activation: {activation}")
        return y

    def is_boreasA(hw_config):
        """Determine if the hwConfig is boreasA.

        This is temporary until the HW_config can explicitly tell what family it is
        """
        return hw_config is None or hw_config.accum_clip is not None


class ACEDigitalDatapathBoreasAInt8(ACEDigitalDatapathInt8):
    """Class for computing BoreasA version of ACE datapath."""

    @classmethod
    def is_registrar_for(cls, fp_mode):
        """Register the subclass."""
        return False

    def wrap_to_10bits(self, inputs):
        """Compute 10bit wrapping."""
        inputs = inputs.int()
        x = inputs & 1023
        x = (x & 511) - (x & 512)
        return x.float()

    def compute(self, input, multiplier, shift, activation):
        """
        Compute datapath math.

        :param input: tensor data in the range [-255, 255] or [0, 255]
        :param integer multiplier
        :param shift: integer shift value
        :param activation: String activation value
        :return: data scaled to [-128, 127] or [0, 255]
        """
        out = self.scale_and_bitshift(input, multiplier, shift)
        out = self.wrap_to_10bits(out)
        out = torch.clamp(out, -256, 255)
        out = self.apply_activation(out, activation)

        return out


class ACEDigitalDatapathBoreasBInt8(ACEDigitalDatapathInt8):
    """Class for computing BoreasA version of ACE datapath."""

    @classmethod
    def is_registrar_for(cls, fp_mode):
        """Register the subclass."""
        return not fp_mode

    def compute(self, input, multiplier, shift, activation):
        """
        Compute datapath math.

        :param input: tensor data in the range [-255, 255] or [0, 255]
        :param integer multiplier
        :param shift: integer shift value
        :param activation: String activation value
        :return: data scaled to [-128, 127] or [0, 255]
        """
        out = self.scale_and_bitshift(input, multiplier, shift)
        out = self.apply_activation(out, activation)
        return out


def ace_digital_datapath_factory(fp_mode):
    """ACE Datapath instance creator.

    :param hw_config: None or HWConfig instance
    :param fp_mode: Boolean
    :return: class instance of the correct type
    """
    for cls in ACEDigitalDatapathFP.__subclasses__():
        if cls.is_registrar_for(fp_mode):
            return cls()

    for cls in ACEDigitalDatapathInt8.__subclasses__():
        if cls.is_registrar_for(fp_mode):
            return cls()

    raise Exception(f"Unknown fp_mode: {fp_mode}")
