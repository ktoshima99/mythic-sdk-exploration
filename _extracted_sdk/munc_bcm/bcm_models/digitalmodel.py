# noqa-flake8-docstrings
import torch
import torch.nn.functional as F

FACTORY_NAME = 'munc_digital'


class DigitalAttributes:
    def __init__(self):
        pass


class PytorchDigitalMMA:
    """A 'Digital' model of the mma

    Models digital-only artifacts
        - quantization
        - clipping
        - multi-cycle
    """
    _ALLOWED_FIXABLE_NOISE = (None, )

    def __init__(self, weights, biases, mma_attr=None, pFSR=2.0, iFSR=2.0, name=None, seed=None,
                 weight_scale=128):
        # We need to know the device of the weights to cast the inputs to the correct device when running multi-gpu
        self.device = weights.device
        self.dtype = weights.dtype
        if not weights.is_cuda and self.dtype is not torch.float32:
            self.dtype = torch.float32
            weights = weights.type(self.dtype)
            biases = biases.type(self.dtype)

        self.iFSR = iFSR
        self.pFSR = pFSR
        self.mma_attr = mma_attr or DigitalAttributes()
        self.weights = (pFSR / iFSR) * weight_scale * weights
        self.biases = (pFSR / iFSR) * weight_scale * biases.sum(1)

        self.pows1 = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=self.dtype, device=self.device)
        pows2 = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=self.device)
        self.pows2 = pows2.reshape(8, *((1,) * weights.dim()))

    def randomize(self, fix_vals=None, random_state=None):
        fix_vals = fix_vals or {}

        if not all(noise_src in self._ALLOWED_FIXABLE_NOISE for noise_src in fix_vals.keys()):
            raise KeyError("You are requesting a noise source to be fixed that cannot be fixed. "
                           f"Allowed values are {self._ALLOWED_FIXABLE_NOISE}")

    def dot(self, uint8_input, dot_op=F.linear):
        """
        uint8_input shape:
          Linear: (batch_size, ∗, in_features)
          Conv2d: (batch_size, in_channels, h, w)
        """
        # Ensure that inputs are in the correct range
        uint8_input = torch.clamp(uint8_input.round(), 0, 255).to(dtype=torch.uint8, device=self.device)
        # Split into bits, the new shape is (bit, batch_size, ...)
        x_bits = uint8_input.unsqueeze(0) * self.pows2 // 128
        # Combine "bit" and "batch_size" dimensions into one to get the shape `dot_op` expects.
        x_bits_reshaped = x_bits.flatten(0, 1)
        dot = dot_op(x_bits_reshaped.type(self.dtype), self.weights, self.biases).round()
        # Split the combined dimension back to "bit" and "batch_size" dimensions.
        dot_reshaped = dot.unflatten(0, (8, uint8_input.shape[0]))
        dot_clipped = torch.clamp(dot_reshaped, -128, 127)
        # Sum results from individual bits.
        accumulator = torch.matmul(dot_clipped.permute(*range(1, dot_clipped.dim()), 0), self.pows1)
        return torch.clamp((accumulator / 128).round(), -256, 255)
