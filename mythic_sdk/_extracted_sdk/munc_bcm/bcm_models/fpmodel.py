# noqa-flake8-docstrings
import torch
import torch.nn.functional as F

FACTORY_NAME = 'munc_fp'


class FloatingPointAttributes:
    def __init__(self):
        pass


class PytorchFloatingPointMMA:
    """A quantized model of the mma

    Same functionality as the BCM "Numpy" model

    Models digital-only artifacts
        - quantization
    """
    _ALLOWED_FIXABLE_NOISE = (None, )

    def __init__(self, weights, biases, mma_attr=None, pFSR=2.0, iFSR=2.0, name=None, seed=None, weight_scale=128):
        # We need to know the device of the weights to cast the inputs to the correct device when running multi-gpu
        self.device = weights.device
        self.dtype = weights.dtype
        if not weights.is_cuda and self.dtype is not torch.float32:
            self.dtype = torch.float32
            weights = weights.type(self.dtype)
            biases = biases.type(self.dtype)

        self.iFSR = iFSR
        self.pFSR = pFSR
        self.mma_attr = mma_attr or FloatingPointAttributes()
        self.weights = (pFSR / iFSR) * weight_scale * weights
        self.biases = (pFSR / iFSR) * weight_scale * biases.sum(1) * 255

    def randomize(self, fix_vals=None, random_state=None):
        fix_vals = fix_vals or {}

        if not all(noise_src in self._ALLOWED_FIXABLE_NOISE for noise_src in fix_vals.keys()):
            raise KeyError("You are requesting a noise source to be fixed that cannot be fixed. "
                           f"Allowed values are {self._ALLOWED_FIXABLE_NOISE}")

    def dot(self, uint8_input, dot_op=F.linear):
        uint8_input = uint8_input.to(device=self.device, dtype=self.dtype)
        accumulator = dot_op(uint8_input, self.weights, self.biases)
        return torch.clamp((accumulator / 128).round(), -256, 255)
