"""Batch norm helper functions.

This module provides functions for doing the batchnorm folding.
"""

import torch

from munc._session_tools import apply_momentum


BN_ATTRIBUTE_PARAMETERS = {
    '__bn_enable': ('enable', True),
    '__bn_post': ('post', True),
    '__bn_stats': ('stats', True),
    '__bn_affine': ('affine', True),
    '__bn_momentum': ('momentum', 0.9),
    '__bn_eps': ('eps', 1e-5),
    '__bn_renorm': ('renorm', True),
    '__bn_max_r': ('max_r', 3.0),
    '__bn_max_d': ('max_d', 5.0)
}
"""Map ONNX node attribute names to namedtuple names and default values.

This is used to build a namedtuple container that organizes attributes from an ONNX model.

Attributes
----------
enable : bool
    Turns on/off the batchnorm in the layer
post : bool
    Uses batchnorm stats after the dot product
stats : bool
    Normalizes using the batchnorm stats
affine : bool
    Applies a scale and shift after normalizing with the stats
momentum : float
    Smooths out changes in the batchnorm stats
eps : float
    Stability constant to avoid division by zero when variance is close to 0
renorm : bool
    Uses the running average stats during training
max_r : float
    Clips the scaling factor of the running average correction with renorm
max_d : float
    Clips the shift factor of the running average correction with renorm
"""


def do_batch_norm(inputs, weight, bias, batch_scale, batch_offset, running_mean, running_var,
                  training,
                  parameters,
                  dot_product=None):
    """Perform the batchnorm operation and produce the folded weights.

    Note that this will update the running_mean and running_var in-place if stats=True and training=True.

    Parameters
    ----------
    inputs : torch.tensor
        The inputs to compute the batchnorm stats from.
        Unless post=True, then we compute the stats from dot_product(inputs, weight, bias)
    weight : torch.tensor
        Weight tensor can be from conv2d or linear layer.
    bias : torch.tensor
        Bias tensor can be from conv2d or linear layer.
    batch_scale : torch.tensor
        The learned batch norm scale factors (i.e. gamma).
    batch_offset : torch.tensor
        The learned batch norm offset factors (i.e. beta).
    running_mean : torch.tensor
        The current running means
    running_var : torch.tensor
        The current running variances.
    training : bool
        A flag that indicates whether this is training or inference.
    parameters : namedtuple
        A named tuple that contains all the parameters associated with batchnorm
    dot_product : lambda, optional
        A function that performs the dot product, e.g. F.conv2d, F.linear, by default None

    Returns
    -------
    torch.tensor, torch.tensor
        The folded weights and biases
    """
    # There isn't a clean way to do this - these are differentiable parameters in torch
    # but we need to ensure they are detached and thus not updated under the following conditions
    # NOTE: This detaches the running mean and var stats in-place
    running_mean.detach_()
    running_var.detach_()
    # affine is the scale and offset that happens after normalization
    # If disabled, the outputs are only normalized to a unit Normal distribution
    #   normalized = (inputs - mean) / std
    # If enabled, the outputs are scaled and shifted by gamma and beta
    #   rescaled = normalized * gamma + beta
    if not parameters.affine:
        # Don't back-propogate through the batchnorm stats
        # NOTE: This detatches the batch scale parameters in-place
        batch_scale.detach_()
        batch_offset.detach_()
    if parameters.stats and training:
        # The post flag determines which side of the dot product the stats are calculated on
        # If disabled, the stats (mean and variance) are calculated directly on the inputs
        # If enabled, the stats (mean and variance) are calculated on the output of the dot product
        if parameters.post:
            values = dot_product(inputs, weight, bias)
        else:
            values = inputs.clone()
        stats = compute_batchnorm_stats(
            values,
            running_mean,
            running_var,
            eps=parameters.eps,
            renormalization=parameters.renorm,
            momentum=parameters.momentum,
            max_r=parameters.max_r,
            max_d=parameters.max_d
        )
        mean, stdev, r_mean, r_var = stats

        # We need to do a copy_ to ensure the tensor is not replaced
        # NOTE: This changes the running stat arguments in-place
        running_mean.copy_(r_mean)
        running_var.copy_(r_var)
    else:
        # We get here if self.training is False or self.bn_stats is False
        mean = running_mean
        stdev = torch.sqrt(running_var + parameters.eps)

    # Compute the folded weights
    folded_weight, folded_bias = fold_batch_norm(
        weight, bias,
        batch_scale, batch_offset,
        mean, stdev,
        post=parameters.post
    )

    return folded_weight, folded_bias


def compute_batchnorm_stats(values, running_mean, running_var, eps=1e-5,
                            renormalization=True, track_running_stats=True,
                            momentum=0.9, max_r=3, max_d=5):
    """Compute the batch norm stats.

    The method computes the batch norm stats and handles the different variations of batchnorm.

    Parameters
    ----------
    values : torch.tensor
        A tensor of activations.
    running_mean : torch.tensor
        The current running means
    running_var : torch.tensor
        The current running variances.
    eps : float, optional
        The stability factor used to prevent divide by zero, by default 1e-5
    renormalization : bool, optional
        Enable batch renormalization, by default True
    track_running_stats : bool, optional
        Enable tracking the running stats, by default True
    momentum : float, optional
        Set the momentum value for the running stats, by default 0.9
    max_r : int, optional
        Ratio clipping factor in batch renormalization, by default 3
    max_d : int, optional
        Offset clipping factor in batch renormalization, by default 5

    Returns
    -------
    (torch.tensor, torch.tensor, torch.tensor, torch.tensor)
        A tuple of 4 elements containing the mean of the batch, variance of the batch,
        running mean and running standard deviation.

    Raises
    ------
    ValueError
        The values have dimension that don't match either conv2d or linear features.
    """
    if len(values.shape) == 4:
        # This is data for conv2d
        var, mean = torch.var_mean(values, dim=[0, 2, 3])
    elif len(values.shape) == 2:
        var, mean = torch.var_mean(values, dim=0)
    else:
        raise ValueError("Batchnorm stats only supported for 2D feature maps or fully connected layers.  "
                         "Received values with dimension: {0}".format(values.shape))
    stdev = torch.sqrt(var + eps)

    if track_running_stats:
        with torch.no_grad():
            running_mean = apply_momentum(mean, running_mean, momentum)
            running_var = apply_momentum(var, running_var, momentum)

        if renormalization:
            with torch.no_grad():
                running_stdev = torch.sqrt(running_var + eps)
                ratio_scale = stdev / running_stdev
                if max_r is not None:
                    ratio_scale = torch.clamp(ratio_scale, 1/max_r, max_r)
                ratio_offset = (mean - running_mean)/running_stdev
                if max_d is not None:
                    ratio_offset = torch.clamp(ratio_offset, -max_d, max_d)

            # Update the mean and stdev
            stdev = stdev / ratio_scale
            mean = mean - ratio_offset * stdev      # Need to use the new stdev

    return mean, stdev, running_mean, running_var


def fold_batch_norm(weight, bias, batch_scale, batch_offset, mean, stdev, post=True):
    """Fold a batch norm layer into a neighbouring layer.

    Folds a batch norm into a conv2d or linear layer.

    Parameters
    ----------
    weight : torch.tensor
        Weight tensor can be from conv2d or linear layer.
    bias : torch.tensor
        Bias tensor can be from conv2d or linear layer.
    batch_scale : torch.tensor
        The learned batch norm scale factors (i.e. gamma).
    batch_offset : torch.tensor
        The learned batch norm offset factors (i.e. beta).
    mean : torch.tensor
        The mean of the activations in the batch norm layer.
    stdev : torch.tensor
        The standard deviation of the activations in the batch norm layer plus an epsilon.
        stdev must not be zero.
    post : bool
        Flag determines whether the batch norm is post or pre the conv2d or linear layer.

    Returns
    -------
    (torch.tensor, torch.tensor)
        Returns a tuple of folded weights and bias tensors.

    Raises
    ------
    NotImplementedError
        If the shape of the weight tensor does not have 2 or 4 dimensions, then the weight
        tensor must be from a layer that is not supported.  The weight and bias tensors
        must be from either a conv2d or a linear layer.
    """
    if post:
        return fold_batch_norm_post(weight, bias, batch_scale, batch_offset, mean, stdev)
    else:
        return fold_batch_norm_pre(weight, bias, batch_scale, batch_offset, mean, stdev)


def fold_batch_norm_pre(weight, bias, batch_scale, batch_offset, mean, stdev):
    """Fold a batch norm layer into the following layer.

    Folds a batch norm into a conv2d or linear layer.  This function expects
    the batch norm to be before the conv2d or linear layer.

    Parameters
    ----------
    weight : torch.tensor
        Weight tensor can be from conv2d or linear layer.
    bias : torch.tensor
        Bias tensor can be from conv2d or linear layer.
    batch_scale : torch.tensor
        The learned batch norm scale factors (i.e. gamma).
    batch_offset : torch.tensor
        The learned batch norm offset factors (i.e. beta).
    mean : torch.tensor
        The mean of the activations in the batch norm layer.
    stdev : torch.tensor
        The standard deviation of the activations in the batch norm layer plus an epsilon.
        stdev must not be zero.

    Returns
    -------
    (torch.tensor, torch.tensor)
        Returns a tuple of folded weights and bias tensors.

    Raises
    ------
    NotImplementedError
        If the shape of the weight tensor does not have 2 or 4 dimensions, then the weight
        tensor must be from a layer that is not supported.  The weight and bias tensors
        must be from either a conv2d or a linear layer.
    """
    # Combine the batch norm scaling and batch stdev into a single scale
    scale = batch_scale/stdev
    # Combine the batch norm offset and batch mean into a single offset
    offset = batch_offset - mean * scale
    # Fold the combined scale and offset into the layer weights
    if len(weight.shape) == 4:
        # weight is N_OUT x N_IN x K x K where
        #   N_OUT is the number of output channels
        #   N_IN is the number of input channels
        #   K is the size of the kernel (non-symmetric kernels are supported as well)
        #
        # We need to multiply the scale along the N_IN dimension.  We do that by
        # 1) re-arranging the dimensions so that we have N_OUT x K x K x N_IN
        # 2) multiplying by scale (multiplies with outer dimension)
        # 3) re-arranging the dimensions so that we get N_OUT x N_IN x K x K again
        new_weight = (weight.permute(0, 2, 3, 1) * scale).permute(0, 3, 1, 2)
        weight_sum = weight.sum(dim=[2, 3])
        new_bias = fold_input_norm_common(weight_sum, bias, offset)
    elif len(weight.shape) == 2:
        new_weight = weight * scale
        new_bias = fold_input_norm_common(weight, bias, offset)
    else:
        raise NotImplementedError(f"Batch norm folding with weights of shape {weight.shape} not implemented")

    return new_weight, new_bias


def fold_batch_norm_post(weight, bias, batch_scale, batch_offset, mean, stdev):
    """Folds a batch norm layer following a conv2d or linear layer.

    This function expects the batch norm to be after the conv2d or linear layer.

    Parameters
    ----------
    weight : torch.tensor
        Weight tensor can be from conv2d or linear layer.
    bias : torch.tensor
        Bias tensor can be from conv2d or linear layer.
    batch_scale : torch.tensor
        The learned batch norm scale factors (i.e. gamma).
    batch_offset : torch.tensor
        The learned batch norm offset factors (i.e. beta).
    mean : torch.tensor
        The mean of the activations in the batch norm layer.
    stdev : torch.tensor
        The standard deviation of the activations in the batch norm layer plus an epsilon.
        stdev must not be zero.

    Returns
    -------
    (torch.tensor, torch.tensor)
        Returns a tuple of folded weights and bias tensors.
    """
    # Combine the batch norm scaling and batch stdev into a single scale
    scale = batch_scale/stdev
    # Combine the batch norm offset and batch mean into a single offset
    offset = batch_offset - scale * mean
    # Fold the combined scale and offset into the weights and biases
    return fold_output_norm(weight, bias, scale, offset)


def fold_input_norm_common(weight, bias, offset):
    """Compute the bias of the folded weights into another layer.

    A utility function to fold a batch norm layer into another layer
    when it is before the other layer.

    Parameters
    ----------
    weight : torch.tensor
        Weight tensor must be from a linear layer.
    bias : torch.tensor
        Bias tensor must be from a linear layer.
    offset : torch.tensor
        A tensor of offset factors to be applied to the input that combines the
        batch stats (mean & stdev) and affine transform (gamma & beta)
        in batch norm.

    Returns
    -------
    torch.tensor
        Returns the new bias tensor.
    """
    if offset is not None:
        # Multiply the offset through the weight tensor to compute
        # the change in bias which is in the output space of the layer.
        new_bias = torch.matmul(weight, offset)
        if bias is not None:
            new_bias = new_bias + bias
    elif bias is not None:
        new_bias = bias.clone()
    else:
        new_bias = None

    return new_bias


def fold_output_norm(weight, bias, scale, offset):
    """Folds combiend scale and offset into a layer.

    A utility function to fold a batch norm layer into either a conv2d or
    linear layer when it is after the layer.

    Parameters
    ----------
    weight : torch.tensor
        Weight tensor must be from a linear layer.
    bias : torch.tensor
        Bias tensor must be from a linear layer.  Can be None.
    scale : torch.tensor
        A tensor of scale factors to be applied to the input that combines the
        batch stats (mean & stdev) and affine transform (gamma & beta)
        in batch norm.
    offset : torch.tensor
        A tensor of offset factors to be applied to the input that combines the
        batch stats (mean & stdev) and affine transform (gamma & beta)
        in batch norm.

    Returns
    -------
    (torch.tensor, torch.tensor)
        Returns a tuple of folded weights and bias tensors.
    """
    if bias is not None:
        new_bias = bias * scale
        if offset is not None:
            new_bias = new_bias + offset
    else:
        new_bias = offset.clone()

    # This works for both conv2d and linear layers
    #
    # For linear layers weight is N_OUT x N_IN where
    #   N_OUT is the number of output features
    #   N_IN is the number of input features
    # For conv2d layers weight is N_OUT x N_IN x K x K where
    #   K is the size of the kernel (does not need to be symmetric)
    #
    # We need to multiply the scale along the N_OUT dimension.  We do that by
    # 1) transposing weight so that we have ... x N_IN x N_OUT
    # 2) multiplying by scale (multiplies with outer dimension)
    # 3) transposing weight so that get N_OUT x N_IN x ... again
    new_weight = (weight.T * scale).T
    return new_weight, new_bias
