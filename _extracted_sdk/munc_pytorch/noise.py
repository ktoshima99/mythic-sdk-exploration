import torch


class WeightNoise(torch.autograd.Function):
    """Apply weight programming noise.

    A multiplicative noise is attached to each weight value to represent
    inaccuracies during programming. AKA "monte carlo" noise

    ```
                        additive weight noise       multiplicative weight noise
                                |                    |
                                V                    V
    weights <= weights + N(0, std_1) + weight * N(0, std_2)
    ```

    Parameters
    ----------
    weight : torch.Tensor
        The weight values _before_ they have been scaled by pFSR, in range [-128, 127]
    mult_sigma : float
        The percentage noise associated with a weight value being programmed

    Returns
    -------
    Noisy weights
        mis-programed weight values
    """

    @staticmethod
    def forward(ctx, weight, additive_noise, mult_sigma):
        """Run forward pass."""
        weight = weight + normal_like(0, additive_noise, weight) + weight*normal_like(0, mult_sigma, weight)
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        """Run backward pass."""
        return grad_output, None, None


def weight_noise(weight, additive_noise, mult_sigma, ste=True):
    """Apply weight programming noise.

    Parameters
    ----------
    ste : bool, optional
        If true, STE is used during a backward pass.

    See WeightNoise for a description of other parameters.
    """
    if ste:
        return WeightNoise.apply(weight, additive_noise, mult_sigma)
    else:
        return WeightNoise.forward(None, weight, additive_noise, mult_sigma)


class TempShift(torch.autograd.Function):
    """Apply temperature shift to weights or biases.

    A global temperature is calculated as part of the computational graph and
    passed in. A random local temperature is then drawn in units of C from a
    uniform distribution between [global_temp-local_temp_shift,
    global_temp+temp_shift]. This temperature shift applies to all weights in
    the given kernel. If local temperature is 0 then the network will only
    experience the global temperature shift

    Derivation
    ----------
    new_flash_w = flash_w * ((2.0928e-2 * EXP(-5.6064e6 * |flash_w|) * temp_delta) + 1)
    flash_w = 200e-9 * w / 128 * pFSR / 2
    new_w = w * ((2.0928e-2 * EXP(-0.00876 * |w| * pFSR/2)) * temp_delta) + 1)

    Parameters
    ----------
    weight : torch.Tensor
        The weight values _before_ they have been scaled by pFSR, in range [-128, 127]
    pfsr: float
        The pFSR value that weights will be scaled by.
    global_temp : float
        global temperature shift in deg C
    local_temp_range : float
        max temperature shift to be applied Ain deg C.

    Returns
    -------
    Shifted weights.
        Weight values have been shifted, but have not been multiplied by
        pFSR. This scaling is expected to happen at the end of the noise
        op, where all scaling is accounted for
    """

    @staticmethod
    def forward(ctx, weight, pfsr, global_temp, local_temp_range):
        """Run forward pass."""
        temp_delta = uniform(global_temp - local_temp_range, global_temp + local_temp_range, device=weight.device,
                             dtype=weight.dtype)
        # abs_weight = torch.abs(weight)
        # systematic_temp_shift = weight * temp_delta * 2.0928e-2 * torch.exp(-0.00876 * pfsr/2 * abs_weight)
        # NOTE:
        # The additional term `weight*temp_delta*.005` is not representative of the hw
        # This is included to make the mythic node match the old retraining model
        # This leads to better accuracy, but is not actually correct
        # This is a temporary fix until we better understand the implementation
        return weight + weight*temp_delta*.005  # + systematic_temp_shift

    @staticmethod
    def backward(ctx, grad_output):
        """Run backward pass."""
        return grad_output, None, None, None


class ADCNonLinearity(torch.autograd.Function):
    """Nonlinearity due to 3rd order distortion.

    ADCs exhibit a 3rd order distortion.

    Derivation
    ----------
    X = X + eta * X**3
    eta ~ Normal(inl_mean, inl_sigma)

    Parameters
    ----------
    X : torch.Tensor
        Activations to be subjected to ADC non linearity
    nl_noise_perc : float
    nl_shift_perc : float
    """

    @staticmethod
    def forward(ctx, X, nl_shift_perc, nl_noise_perc):
        """Run forward pass."""
        maximum_9bit = 255
        ifsr_reference = 10
        nl_noise_coeff = nl_noise_perc / (maximum_9bit * ifsr_reference)**2
        nl_shift_coeff = nl_shift_perc / (maximum_9bit * ifsr_reference)**2
        eta = normal(nl_shift_coeff, nl_noise_coeff, device=X.device, dtype=X.dtype)
        return X + eta * torch.pow(X, 3)

    @staticmethod
    def backward(ctx, grad_output):
        """Run backward pass."""
        return grad_output, None, None


class ADCNoise(torch.autograd.Function):
    """A noise associated with ADC thermal effects.

    Thermal noise in the ADC generates errors in the activations. The ADC noise
    has been calibrated at 3 LSB at iFSR=10. Because the ADC is run 8 times
    during multi-cycle, we modify the noise here to be the shifted noise added
    in quadrature, so that we can approximate multi-cycle ADC noise

    Derivation
    ----------
    err^2 = sum_{b=0}^9 (sigma/2^b)^2
          = sigma^2 sum (1/2^b)^2
          = sigma^2 * .58

    sigma = noise_at_10_ifsr * 10 [iFSR] / 2
          = noise_at_10_ifsr * 5 [half_iFSR]

    Parameters
    ----------
    X : torch.Tensor
        Activations to be subjected to ADC noise
    nlsb : float
        LSBs of noise at the anchor iFSR
    """

    @staticmethod
    def forward(ctx, X, noise_at_ifsr10):
        """Run forward pass."""
        rand_sigma = noise_at_ifsr10 * 5.0 * 0.58
        return X + normal_like(0, rand_sigma, X)

    @staticmethod
    def backward(ctx, grad_output):
        """Run backward pass."""
        return grad_output, None


@torch.no_grad()
def normal_like(mean, std, tensor):
    """Return normal-like random tensor."""
    return mean + std * torch.randn(size=tensor.shape, device=tensor.device, dtype=tensor.dtype)


@torch.no_grad()
def normal(mean, std, device, dtype):
    """Return normal random tensor of size (1,)."""
    return mean + std * torch.randn(size=(1,), device=device, dtype=dtype)


@torch.no_grad()
def uniform(low, high, device, dtype):
    """Return uniform random tensor of size 1."""
    return (high - low) * torch.rand(1, device=device, dtype=dtype) + low


temp_shift = TempShift.apply
adc_noise = ADCNoise.apply
adc_nl = ADCNonLinearity.apply
