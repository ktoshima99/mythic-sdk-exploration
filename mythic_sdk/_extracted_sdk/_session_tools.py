from functools import partial
import logging
import math
from itertools import islice, product

import torch
import numpy as np
from torch.ao.quantization.observer import HistogramObserver

from munc import _utils, _node_utils
from munc._constants import (DEBUG_DIR, SUPPORTED_ON_CHIP_NODES_BOREAS,
                             SUPPORTED_ON_CHIP_NODES_DENALI, MODELType,
                             MYTHICType, ONNXType, HardwareType,
                             STATS_DEBUG_PREFIX, BiasSplittingMethod)

from munc.viz.debug_plots import plot_range_through_network, PLOTMode

logger = logging.getLogger(__name__)
N_PAD = 8


def _check_non_zeros_in_input_splits(input_sizes):
    there_is_at_least_one_zero = any([input_size == 0 for input_size in input_sizes])
    if there_is_at_least_one_zero:
        raise Exception('There is at least one input size that is zero.')


def balanced_input_split(weight_shape, max_size, input_block_size=N_PAD):
    """Split data evenly (balanced) among tiles.

    Parameters
    ----------
    weight_shape : Tuple[int]
        The shape of layer weights.
    max_size : int
        The maximum number of weights in a dot product.
    input_block_size : int, optional
        The block size to use for splitting the input, by default N_PAD.

    Returns
    -------
    list
        Input sizes per tile.
    """
    input_channels = weight_shape[1]
    block_size = input_block_size
    kernel_size = np.prod(weight_shape[2:])

    def round_to_block_size(x, up):
        round_func = math.ceil if up else math.trunc
        return round_func(x / block_size) * block_size

    max_channels_per_tile = round_to_block_size(max_size / kernel_size, up=False)
    assert max_channels_per_tile > 0, f"Can't fit {block_size} channels with kernel size {kernel_size} to a tile."
    num_tiles = math.ceil(input_channels / max_channels_per_tile)
    channels_per_tile = round_to_block_size(input_channels / num_tiles, up=True)
    channels_in_last_tile = input_channels - (num_tiles - 1) * channels_per_tile
    input_splits = [channels_per_tile] * (num_tiles - 1) + [channels_in_last_tile]
    return input_splits


def greedy_input_split(data_size, n_inputs):
    """Fits as much data as possible to each tile.

    Data left over goes to the last tile.

    Parameters
    ----------
    data_size : int
        Usually the weight.shape[1].
    n_inputs : int

    Returns
    -------
    list
        Input sizes per tile.
    """
    data_remaining = data_size
    input_sizes = []
    while (data_remaining > n_inputs):
        input_sizes.append(n_inputs)
        data_remaining -= n_inputs
    input_sizes.append(data_remaining)
    _check_non_zeros_in_input_splits(input_sizes)

    return input_sizes


def compute_multiplier_and_shift_torch(digital_scale, max_multiplier, max_shift):
    """Find a multiplier and a shift such that `multiplier / 2 ** shift` is the best approximation of `digital_scale`.

    Parameters
    ----------
    digital_scale : torch.Tensor
        A digital scale to convert to a multiplier and shift.
    max_multiplier : int
        The maximum value for the multiplier.
    max_shift : int
        The maximum value for the shift.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Digital scale multiplier and shift.
    """
    assert (max_multiplier & (max_multiplier + 1)) == 0, "max_multiplier + 1 must be a power of 2"
    # The largest feasible divisor (shift) results in the smallest error (1/2*(2**shift)),
    # A corresponding multiplier would be digital_scale / 2**shift and it can't be larger than max_multiplier.
    # That sets a limit on the shift, i.e. shift = log2((max_multiplier + 0.5) / digital_scale). +0.5 here
    # covers the case where digital_scale is between max_multiplier / 2**shift and (max_multiplier + 1) / 2**shift,
    # and is closer to the former.
    # An alternative torch.frexp-based implementation didn't seem to improve performance or reduce complexity.
    shift = torch.clamp(torch.trunc(torch.log2((max_multiplier + 0.5) / (digital_scale + 1e-10))), 0, max_shift)
    multiplier = torch.round(digital_scale * 2 ** shift)
    return multiplier, shift


def calculate_digital_scale_factors(digital_scale, max_multiplier, max_shift, normilize=True):
    """Find a multiplier and a shift such that `multiplier / 2 ** shift` is the best approximation of `digital_scale`.

    Parameters
    ----------
    digital_scale : float
        A digital scale to convert to a multiplier and shift.
    max_multiplier : int
        The maximum value for the multiplier.
    max_shift : int
        The maximum value for the shift.
    normilize : bool, optional
        If True, the multiplier and divisor will be normalized, by default True.

    Returns
    -------
    int, int
        Digital scale multiplier and divisor.
    """
    tensor_input = isinstance(digital_scale, np.ndarray) and digital_scale.ndim > 0
    digital_scale = torch.as_tensor(digital_scale, device='cpu')
    multiplier, shift = compute_multiplier_and_shift_torch(digital_scale, max_multiplier, max_shift)
    multiplier = multiplier.to(dtype=torch.int64)
    divider = (2 ** shift).to(dtype=torch.int64)
    if normilize:
        gcd = torch.gcd(multiplier, divider)
        multiplier = multiplier // gcd
        divider = divider // gcd
    if tensor_input:
        return multiplier.numpy(), divider.numpy()
    else:
        return multiplier.item(), divider.item()


def get_node_digital_scale(node):
    """Return the digital scale factor of a (MythicConv, MythicLinear) node."""
    return node.attrs["__multiplier"] / (2 ** node.attrs["__shift"])


def set_digital_scale_attrs(node, dsf, hwconfig):
    """Set digital scale multipler and divisor node attributes.

    Parameters
    ----------
    node : ONNXNode
        A node to modify.
    dsf : float
        A digital scale value
    hwconfig : HWConfig
        A hardware configuration to use.
    """
    new_multiplier, new_divider = calculate_digital_scale_factors(np.array(dsf, dtype=float), hwconfig.ds_max_mult,
                                                                  hwconfig.ds_max_shift)
    if "__multiplier" in node.attrs:
        del node.attrs["__multiplier"]
    if "__shift" in node.attrs:
        del node.attrs["__shift"]
    node.attrs["__multiplier"] = new_multiplier
    node.attrs["__shift"] = np.log2(new_divider).astype(int)


def compute_multiplier_and_shift(scale_factor, number_of_bits=8):
    """Decompose a scale factor into an integer multiply and shift.

    Parameters
    ----------
    scale_factor : float
        The scale factor to convert to multiply and shift
    number_of_bits : int, optional
        Number of bits for the multiply and shift

    Returns
    -------
    (int, int)
        The multiply and shift factors
    """
    multiplier, divisor = calculate_digital_scale_factors(np.array(scale_factor), 2**number_of_bits - 1,
                                                          number_of_bits - 1)
    shift = int(np.round(np.log2(divisor)))
    multiplier = int(multiplier)
    actual_scaling_factor = float(multiplier) / float(2**shift)
    return multiplier, shift, actual_scaling_factor


def calculate_half_pfsr_ifsr_and_digital_scale(half_pFSR_arr, half_iFSR_arr, composite_scale):
    """Select the best pFSR, iFSR pair to implement a composite scale.

    Despite the function name, the 3rd return value can't always be used as a digital scale factor. When it is smaller
    than 1, it must be used a weight scale factor to avoid accumulator clipping (ASF must be <= CSF). When it is larger
    than 1, it can't be used a weight scale and must be used as DSF to avoid weight clipping.

    Parameters
    ----------
    half_pFSR_arr : list of int
        A list of pFSR/2 values to choose from.
    half_iFSR_arr : list of int
        A list of iFSR/2 values to choose from.
    composite_scale : int

    Returns
    -------
    float, float, float
        pFSR/2, iFSR/2, composite_scale / (pFSR / iFSR)
    """
    # Determine FSR
    half_pfsr = half_pFSR_arr[0]
    half_ifsr = half_iFSR_arr[0]
    half_pFSR_cost = -1  # choose larger FSR, everything else being equal
    cost_clip = 1000  # strongly choose no clip, everything else being equal

    def score(half_fsr):
        """Return a score of a iFSR, pFSR pair. The lower the better."""
        half_pfsr, half_ifsr = half_fsr
        fsr_ratio = half_pfsr / half_ifsr
        delta = np.abs(np.log(fsr_ratio / composite_scale))
        # "clip" here probably means clipping of weights, but it does not happen in reality,
        # because mythic node does not implement csf/fsr as wsf if it is > 1, it implements it as dsf. -- Ilya
        clip = fsr_ratio < composite_scale
        score = delta + half_pFSR_cost * half_pfsr + cost_clip * clip
        return score

    # Find the best pair (one with the mimimum score).
    half_pfsr, half_ifsr = min(product(half_pFSR_arr, half_iFSR_arr), key=score)
    results = (half_pfsr, half_ifsr, composite_scale / half_pfsr * half_ifsr)
    # Coerce results to composite_scale.dtype
    return list(map(lambda x: np.array(x, dtype=composite_scale.dtype), results))


def calculate_from_lookup_table(composite_scale, table):
    """Calculate the half pFSR, half iFSR, digital scale, weight scale and the remainder using the lookup table.

    Parameters
    ----------
    composite_scale : int
    table : dict
        Lookup table.

    Returns
    -------
    (int, int, int, float, float)
        Tuple containing the half pFSR, half iFSR, digital scale, weight scale and the remainder.
    """
    # Maximum acceptable WSF value
    max_WSF = np.max(table["WSF"])

    # Find proposed CSF values such that CSF_proposed = pFSR*DSF*WSF/iFSR
    CSF_proposed = []
    for h_pFSR, h_iFSR, DSF, WSF in zip(table["half_pFSR"], table["half_iFSR"], table["DSF"], table["WSF"]):
        CSF_proposed.append(h_pFSR * DSF * WSF / h_iFSR)

    log_remainders = [np.log(composite_scale / CSF_) for CSF_ in CSF_proposed]

    # Find the lowest abs(negative) and lowest positive log(remainder).
    # These correspond to the ratios Composite_Scale / CSF that are closest to 1
    negative_log, positive_log = -float('inf'), float('inf')
    for log_remainder in log_remainders:
        if negative_log < log_remainder < 0:
            negative_log = log_remainder
        elif 0 <= log_remainder < positive_log:
            positive_log = log_remainder

    # If the composite scale is higher than any available CSF value
    if negative_log == -float("inf") and positive_log != float("inf"):
        i_CSF = log_remainders.index(positive_log)

    # If the composite scale is lower than any available CSF value
    elif negative_log != -float("inf") and positive_log == float("inf"):
        i_CSF = log_remainders.index(negative_log)

    # If the composite scale is between two CSF values
    else:
        i_CSF_pos = log_remainders.index(positive_log)
        i_CSF_neg = log_remainders.index(negative_log)

        # ... and is closer to the smaller value
        if (positive_log + negative_log) < 0:

            # the remainder is > 1,
            # we need to make sure that remainder*WSF < max_WSF
            if np.exp(positive_log) * table["WSF"][i_CSF_pos] < max_WSF:
                i_CSF = i_CSF_pos
            else:
                i_CSF = i_CSF_neg
        else:
            i_CSF = i_CSF_neg

    # Enforce scale data types
    half_pfsr = np.array(table["half_pFSR"][i_CSF], dtype=composite_scale.dtype)
    half_ifsr = np.array(table["half_iFSR"][i_CSF], dtype=composite_scale.dtype)
    digital_scale = np.array(table["DSF"][i_CSF], dtype=composite_scale.dtype)
    weight_scale = np.array(table["WSF"][i_CSF], dtype=composite_scale.dtype)
    remainder = np.array(np.exp(log_remainders[i_CSF]), dtype=composite_scale.dtype)

    # Return results
    return half_pfsr, half_ifsr, digital_scale, weight_scale, remainder


def calculate_csf_range_without_wsf_from_lookup_table(table):
    """Calculate the minumum/maximum CSF possible from the table without WSF.

    NOTE: Since WSF is WSF*remainder, we can't find min_wsf and max_wsf from the table. These values returned can be
    used to limit the range of WSF values to ensure WSF <= 1.

    Parameters
    ----------
    table : dict
        Lookup table.

    Returns
    -------
    float, float
        The minimum and maximum CSF.
    """
    min_dsf = np.min(table["DSF"])
    max_dsf = np.max(table["DSF"])
    min_half_pfsr = np.min(table["half_pFSR"])
    max_half_pfsr = np.max(table["half_pFSR"])
    min_half_ifsr = np.min(table["half_iFSR"])
    max_half_ifsr = np.max(table["half_iFSR"])

    min_csf = (min_half_pfsr * min_dsf) / max_half_ifsr
    max_csf = (max_half_pfsr * max_dsf) / min_half_ifsr

    return min_csf, max_csf


def quantize_weight(weight, num_fractional_bits=0):
    """Return the quantized (rounded) weight (the type of the array is unchanged)."""
    scale = (2 ** num_fractional_bits)
    return np.round(weight * scale) / scale


def clip_weight(weight, min_, max_):
    """Clip weight with the provided minimum and maximum.

    Parameters
    ----------
    weight : numpy.ndarray or torch.Tensor
    min_ : int or float
    max_ : int or float

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Cliped weight.
    """
    weight[weight > max_] = max_
    weight[weight < min_] = min_
    return weight


def scale_weight(weight, scale):
    """Return the scaled (weight/scale) weight."""
    weight_scaled = weight / scale
    return weight_scaled


def scale_weight_with_min_max(weight, min_, max_):
    """Return the scaled weights.

    The scaling parameter is calculated using the provided minimum and maximum.

    Parameters
    ----------
    weight : numpy.ndarray
    min_ : int or float
    max_ : int or float

    Returns
    -------
    numpy.ndarray, float
        The scaled weights and the used scaling parameter.
    """
    # Scale is not given, calculate it
    weight_lims = np.abs([np.min(weight), np.max(weight)])
    Ns = np.abs([min_ + 0.5 * np.sign(min_), max_ + 0.5 * np.sign(max_)])
    scales = np.divide(weight_lims, Ns)
    scale = np.max(scales)

    # Normalize
    weight_scaled = scale_weight(weight, scale)

    # Return the weight and scale
    return weight_scaled, scale


def is_op_type_supported_on_chip(op_type, hardware_type):
    """Return True if Operator type is supported on-chip.

    It does not check for valid attributes.

    Parameters
    ----------
    op_type : str
        ONNX operator type.
    hardware_type : str
        Hardware type. Currently either HardwareType.BOREAS or HardwareType.DENALI.

    Returns
    -------
    bool

    Raises
    ------
    Exception
        If the op_type argument is not a string.
    """
    logger.debug(
        "The Op is not guaranteed to be supported on-chip even if it returns True.")
    if not isinstance(op_type, str):
        raise Exception("Op_type input is expected to be a string.")
    if hardware_type == HardwareType.BOREAS:
        on_chip_nodes = SUPPORTED_ON_CHIP_NODES_BOREAS
    elif hardware_type == HardwareType.DENALI:
        on_chip_nodes = SUPPORTED_ON_CHIP_NODES_DENALI
    else:
        raise ValueError("Unsupported hardware type. Supported types are Boreas and Denali.")
    return op_type in on_chip_nodes


@_utils.deprecated
def _infer_model_type(model):
    # Infer model type from counts of ops
    nodes_matrix_multiply = model.get_nodes_with_mythic_type(MYTHICType.MATRIXMULTIPLY)
    nodes_composite_scale = model.get_nodes_with_mythic_type(MYTHICType.COMPOSITE_SCALE)
    nodes_convgemm = model.get_nodes_with_op_type([ONNXType.CONV, ONNXType.GEMM])

    ifsr_attribute_exists = any(_node_utils.is_attribute(n, "__iFSR") for n in nodes_matrix_multiply)
    any_convgemm_onchip = ((len(nodes_convgemm) > 0) and (not all(_node_utils.is_off_chip(n) for n in nodes_convgemm)))

    if len(nodes_matrix_multiply) == 0:
        return MODELType.ORIGINAL
    elif not ifsr_attribute_exists or len(nodes_composite_scale) >= 1:
        return MODELType.PTM
    elif any_convgemm_onchip:
        return MODELType.COMPILER
    else:
        return MODELType.BCM


def get_model_type(model):
    """Retrieve or infer the model type from the graphs metadata.

    Parameters
    ----------
    model : munc.ONNXModel

    Returns
    -------
    str
    """
    model_type = model.get_meta_data('__type')

    # If no model type, check to see if it is an original model. If not, the
    # model is likely old and we need to infer the model type based on the node
    # types
    if not model_type:
        nodes_matrix_multiply = model.get_nodes_with_mythic_type(MYTHICType.MATRIXMULTIPLY)
        if len(nodes_matrix_multiply) == 0:
            model_type = MODELType.ORIGINAL
        else:
            model_type = _infer_model_type(model)

    return model_type


def _collect_debugging_point(prefix, model, stats, n_samples, saving_dir=DEBUG_DIR, save_only=False):
    def _store_stats_in_nodes(model, stats):
        for edge in stats._stats:
            node = model.get_node_with_output_name(edge)
            if node is not None:
                # For the available stats and their meaning, see StatsCollector.__init__().
                for attr in ['clip_min', 'clip_max', 'min', 'max', 'mean', 'std']:
                    if attr in stats._stats[edge]:
                        attr_name = STATS_DEBUG_PREFIX + attr
                        _node_utils.set_attribute_value(node, attr_name, stats._stats[edge][attr], create=True)
                if model.hwconfig is not None:
                    target_range = model.hwconfig.target_range_fcn(model, edge)
                    if target_range is not None:
                        min_, max_ = target_range
                        _node_utils.set_attribute_value(node, STATS_DEBUG_PREFIX + 'target_min', min_, create=True)
                        _node_utils.set_attribute_value(node, STATS_DEBUG_PREFIX + 'target_max', max_, create=True)

    def _remove_stats_from_nodes(model):
        for node in model.get_nodes():
            for attr in ['clip_min', 'clip_max', 'min', 'max', 'mean', 'std', 'target_min', 'target_max']:
                attr_name = STATS_DEBUG_PREFIX + attr
                _node_utils.remove_attribute(node, attr_name)

    logger.info(f"Writing model debugging data to {saving_dir / prefix}")
    saving_dir.mkdir(parents=True, exist_ok=True)
    if not save_only:
        if n_samples != 0:
            stats.collect(n_samples_min=n_samples)
        _store_stats_in_nodes(model, stats)
        torch.save(stats._stats, saving_dir / f"{prefix}_stats.pt")
        plot_range_through_network(model, stats._stats, saving_dir / f"{prefix}_plot.html")
        if get_model_type(model) == MODELType.MYTHIC:
            plot_range_through_network(
                model, stats._stats, saving_dir / f"{prefix}_clipping_plot.html", plot_mode=PLOTMode.CLIPPING)
        _dump_input_ratios(saving_dir / f"{prefix}_node_input_ratios.txt", model, stats)
    model.save(saving_dir / f'{prefix}_model.onnx')
    if not save_only:
        _remove_stats_from_nodes(model)  # Ensure we don't get stale stats


def _dump_node_input_ratios(node, stats, stream):
    """Print the input range of each node input to a stream.

    Output format is
      Node name
          edge1: [min1, max1], range1/min_range
          edge2: [min2, max2], range2/min_range
          ...
    """
    def get_edge_range(edge_data):
        return max(np.abs(edge_data[1:]))

    def node_label(node):
        return f'{node.name}, {node.op_type}, {_node_utils.get_mythic_type(node)}'

    stats = stats.get_reference_to_stats()
    per_edge_data = []
    for edge in node.input:
        edge_stat = stats.get(edge)
        if edge_stat and 'clip_min' in edge_stat:
            per_edge_data.append((edge, edge_stat['clip_min'], edge_stat['clip_max']))
    per_edge_data = sorted(per_edge_data, key=get_edge_range)

    if len(per_edge_data) > 1:
        min_range = get_edge_range(per_edge_data[0])
        stream.write(f'Input ranges of {node_label(node)} (edge: [min, max], range/min_range\n')
        for edge_data in per_edge_data:
            edge_range = get_edge_range(edge_data)
            edge, min_val, max_val = edge_data
            stream.write(f'\t{edge_data[0]}: {edge_data[1:]}, {edge_range / min_range}\n')


def _dump_input_ratios(file, model, stats):
    with open(file, mode='w') as stream:
        for node in model.get_nodes():
            _dump_node_input_ratios(node, stats, stream)
        stream.write('\n')


def split_bias(bias_tensor, max_abs_weight, num_bias_splits=6, method=None, use_fp=True):
    """Split the biases into smaller values.

    Weights and biases passed here are expected to be in floating point rather than int8 or similar format.

    Parameters
    ----------
    bias_tensor : torch.tensor
        Bias tensor for layer.  Floating point biases.
    max_abs_weight : float
        Maximum absolute weight value in float.
    num_bias_splits : int, optional
        Maximum number of splits to use, by default 6.
    method : str, optional
        The method of bias splitting to use, "balanced", "overflow", or None

    Returns
    -------
    torch.tensor
        Tensor of split biases, e.g. N_OUT x N_SPLITS where N_OUT and N_SPLITS are
        the number of output channels and bias splits respectively.
    """
    assert bias_tensor is not None, "bias_tensor must not be None."
    if method is None or method.lower() == BiasSplittingMethod.OVERFLOW:
        split_biases = overflow_split_bias(bias_tensor,
                                           num_bias_splits=num_bias_splits,
                                           max_abs_weight=max_abs_weight,
                                           use_fp=use_fp)
    elif method.lower() == BiasSplittingMethod.BALANCED:
        split_biases = balance_split_bias(bias_tensor,
                                          num_bias_splits=num_bias_splits,
                                          use_fp=use_fp)
    else:
        raise NotImplementedError(f"bias splitting method {method} is not implemented.")

    return split_biases


def balance_split_bias(bias_tensor, num_bias_splits=6, use_fp=True):
    """Split the biases into smaller values using balanced method.

    This method is the same as the BCM bias splitting method when use_fp=False.
    Otherwise, when use_fp=True, it is an "ideal" balanced bias split done in floating point.

    Parameters
    ----------
    bias_tensor : torch.tensor
        Bias tensor for layer.
    num_bias_splits : int, optional
        Maximum number of splits to use, by default 6.
    use_fp : bool, optional
        If True, do balanced split in floating point.
        If False, the balanced splitting is done with biases as integers

    Returns
    -------
    torch.tensor
        Tensor of split biases, e.g. N_OUT x N_SPLITS where N_OUT and N_SPLITS are
        the number of output channels and bias splits respectively.
    """
    assert bias_tensor is not None, "bias_tensor must not be None."
    if use_fp:
        return bias_tensor.unsqueeze(1).repeat(1, num_bias_splits) / num_bias_splits

    # split 1d bias into 2d array (bias.shape, bias_rows), and duplicate its values along bias_rows dimension
    split_bias = bias_tensor.unsqueeze(1).repeat(1, num_bias_splits)
    # scale down each duplicated value by bias_rows, so that they sum to the original values along the bias_rows dim
    split_bias = split_bias * 1./num_bias_splits
    # Add a quantization correction to the split biases to spread out the rounding error
    rounding_errors = torch.arange(0.5, num_bias_splits, dtype=bias_tensor.dtype, device=bias_tensor.device)
    rounding_errors = (rounding_errors.flip(dims=[0]) - 0.5 * num_bias_splits) / num_bias_splits
    split_bias = split_bias + rounding_errors
    # quantize the split values
    split_bias = split_bias.round()
    return split_bias


def overflow_split_bias(bias_tensor, max_abs_weight, num_bias_splits=6, use_fp=True):
    """Split the biases into smaller values using overflow method.

    Parameters
    ----------
    bias_tensor : torch.tensor
        Bias tensor for layer.
    max_abs_weight : float
        Maximum absolute weight value in float.
    num_bias_splits : int, optional
        Maximum number of splits to use, by default 6.
    use_fp : bool, optional
        Toggle FP vs Integer bias splitting

    Returns
    -------
    torch.tensor
        Tensor of split biases, e.g. N_OUT x N_SPLITS where N_OUT and N_SPLITS are
        the number of output channels and bias splits respectively.

    Raises
    ------
    ZeroDivisionError
        When max_abs_weight is zero.
    ValueError
        When max_abs_weight is negative.
    """
    assert bias_tensor is not None, "bias_tensor must not be None."
    bias_shape = (bias_tensor.shape[0], num_bias_splits)  # bias can have 6 rows
    original = torch.zeros(bias_shape, device=bias_tensor.device, dtype=bias_tensor.dtype)
    if abs(max_abs_weight) <= 0:
        raise ZeroDivisionError("max_abs_weight must be positive (non-zero)")
    if max_abs_weight < 0:
        raise ValueError("max_abs_weight must be positive")

    # We know what the max weight now, so let's split the biases with that value in mind
    # Compute the number of splits given the bias and the max_abs_weight
    #  We want the number of splits to be dynamic according to how big the bias is
    #  e.g. consider max_abs_weight = 1, then
    #    (1) if 0 < bias.abs() <= 1 --> we should not split the bias
    #    (2) if 1 < bias.abs() <= 2 --> we should split the bias across 2
    #    (3) if 2 < bias.abs() <= 3 --> we should split the bias across 3
    #        etc.
    # This approach ensures that when biases are split across more than one bias,
    # the split bias values wil be >0.5*max_abs_weight.  Unsplit biases can be any value
    # between -max_abs_weight and max_abs_weight.
    norm_abs_bias = bias_tensor.abs() / max_abs_weight
    # Find the minimum number of biases needed where the bias is limited to max_abs_weight
    bias_count = norm_abs_bias.ceil()
    # Clip the bias count if it exceeds the desired number of splits
    num_of_splits = bias_count.clamp(1, num_bias_splits).unsqueeze(1)
    # Split the biases evenly across the minimum number of required splits
    bias_repeat = bias_tensor.unsqueeze(1).repeat(1, num_bias_splits)
    if use_fp:
        bias_repeat = bias_repeat / num_of_splits
    else:
        # These shift factors are computed to ensure that when we split the biases evenly across num_of_splits
        # NOTE: This spreads the overflow amoung the split biases before actually splitting them
        # e.g. support bias is 167 then this is split across two biases evenly as [83.5, 83.5]
        #      but in order to split this in integer, we need round the first up and the second down.
        #      bias_split_shifts will compute [0.5, 0] which when added to the split biases gives us
        #      [84, 83.5].  We can now round down all split biases to get the correctly split result
        #      in integer as [84, 83].
        #
        #      Consider splitting the bias 601.  The overflow method will split this across a total of 5
        #      rows (out of a maximum of 6) producing
        #        [120.2, 120.2, 120.2, 120.2, 120.2, 0]
        #      To round this exactly, we need to round up one to 121 and the rest to 120.  We do this by
        #      computing bias_split_shifts as
        #        [0.8, 0.6, 0.4, 0.2, 0.0, 0.0]
        #      and adding to the original split
        #        [121.0, 120.8, 120.6, 120.4, 120.2, 0.0]
        #      and then rounding down to get
        #        [121, 120, 120, 120, 120, 0]
        #      This is fast on GPU and guaranteed to have a rounding error less than 1-bit
        num_of_splits_numpy = num_of_splits.squeeze(1).cpu().numpy().tolist()
        bias_split_shifts = torch.stack(
            [torch.linspace(
                1 - num_bias_splits/split,
                1 - 1/split,
                num_bias_splits,
                dtype=bias_tensor.dtype,
                device=bias_tensor.device
            ).clamp(0, 1).flip(0)
                for split in num_of_splits_numpy]
        )
        # Round down with floor
        bias_repeat = (bias_repeat / num_of_splits + bias_split_shifts).floor()
    # Compute indices for each bias and its split biases
    bias_index = torch.arange(num_bias_splits, dtype=bias_tensor.dtype, device=bias_tensor.device).unsqueeze(1)
    # Repeat the split bias indices across the total number of biases
    split_counts = bias_index.repeat(1, bias_shape[0]).T
    # Repeat the minimum number of require splits across the number of splits
    split_thresholds = num_of_splits.repeat(1, num_bias_splits)
    # Mask out (zero) all bias splits that have an index greater than the minimum
    # number of required bias splits
    split_biases = torch.where(split_counts < split_thresholds, bias_repeat, original)

    return split_biases


def get_num_input_splits(weight_shape, max_size, method=None):
    """Get the number of input splits for a given input and weight shape and max_size.

    MUNC supported two different approaches to input splitting, we will use their algorithms here.

    Parameters
    ----------
    weight_shape : tuple
        Shape of the weight tensor
    max_size : int
        The maximum length of the MMA dot product allowed
    method : str, optional
        The method of input splitting to apply: "balanced", "greedy", "equal"

    Returns
    -------
    list
        A list of the size of each split
    """
    num_input_channels = weight_shape[1]
    kernel_size = np.prod(weight_shape[2:])
    max_channels = max_size // kernel_size
    num_splits = int(np.ceil(num_input_channels / max_channels))
    if method is None or method.lower() == "balanced":
        # This is the balanced method in split_inputs.py
        # It aligns the split to 8 channel blocks
        split_sizes = balanced_input_split(weight_shape, max_size)
    elif method.lower() == "greedy":
        n_inputs = int(np.floor(max_size / kernel_size))
        split_sizes = greedy_input_split(num_input_channels, n_inputs)
    elif method.lower() == "equal":
        # This method is an "equal" split that ignores channel block alignment
        split_sizes = [int(num_input_channels / num_splits)]*num_splits
    else:
        raise NotImplementedError(f"Input splitting method {method} is not implemented")
    return num_splits, split_sizes


def get_weight_scale(weight, bias, use_sigma=False, correct_mean=True, n_sigma=3, pctl=0.997, num_bias_splits=6,
                     protect_biases=False, protect_filters=False, hw_weight_min=-128, hw_weight_max=127,
                     use_histogram=False):
    """Find the optimal weight scale using sigma, percentile, or histogram heuristics.

    To enable histogram-based scaing set use_histogram=True, to use sigma heuristic set use_sigma=True.
    Otherwise, percentile will be used. If pctl=1, then the max value will be used.

    Parameters
    ----------
    weight : torch.tensor
        Weight tensor for the layer.
    bias : [type]
        Bias tensor for the layer.
    use_sigma : bool, optional
        Enable to compute bound with sigma heuristic, by default False.
    correct_mean : bool, optional
        Enable to compute sigma heuristic with mean offset, by default True.
    n_sigma : int, optional
        The scale factor for the sigma heuristic, by default 3.
    pctl : float, optional
        The percentile used if use_sigma is False, by default 0.997.
    num_bias_splits : int, optional
        The number of bias splits used to get the weight bound, by default 6.
    protect_biases : bool, optional
        Ensures that the biases are not clipped aggressively by treating them equally with weights, by default False.
        The number of weights could be so much larger than the number biases that the traditional approach
        of computing 3 sigma over the combined weights + biases can result in significant clipping of the
        biases.  By turning this on, it treats the weights and biases as equals regardless of the number
        of each.
    protect_filters : bool, optional
        Ensures that clipping does not destroy filters, by default False.
        If set to true, a weight bound is computed for each filter separately and then the maximum of the per-filter
        maximums is returned.
    use_histogram : bool, optional
        If True, use the histogram of the weights to compute the optimal scaling factor.

    Returns
    -------
    float
        A scalor value containing the weight bound.
    """
    def choose_scale_factor(values, pctl=pctl, use_sigma=use_sigma, use_histogram=use_histogram):
        if use_histogram:
            hist = HistogramObserver(qscheme=torch.per_tensor_symmetric, quant_min=hw_weight_min,
                                     quant_max=hw_weight_max, dtype=torch.qint32)
            hist(values)
            qparams = hist.calculate_qparams()
            scale_factor = (1 / qparams[0]).numpy()
            return scale_factor
        elif use_sigma:
            # The weights might not be zero mean
            max_weight = compute_n_sigma_bound(values, sigma=n_sigma, correct_mean=correct_mean, abs_mean=True)
        elif pctl < 1:
            kw = int(pctl * torch.numel(values))
            max_weight = torch.abs(values.view(-1)).kthvalue(k=kw)[0]
        else:
            max_weight = torch.abs(values.view(-1)).max()
        return hw_weight_max / max_weight

    with torch.no_grad():
        flat_weight = weight.view(weight.shape[0], -1)

        # Split each bias into six parts, if bias is None, set it to zero
        bias_for_split = bias if bias is not None else torch.zeros(flat_weight.shape[0],
                                                                   dtype=flat_weight.dtype,
                                                                   device=flat_weight.device)
        max_abs_weight = torch.max(flat_weight.abs()).item()
        split_biases = split_bias(bias_for_split, num_bias_splits=num_bias_splits, max_abs_weight=max_abs_weight,
                                  method=BiasSplittingMethod.BALANCED)

        # Find the scaling factor for the union of weights and split biases
        if protect_biases:
            # Disable percentile and sigma clipping for bias scale computations. At this point each filter bias is
            # divided into equal parts, so we want to keep all bias values.
            choose_bias_scale_factor = partial(choose_scale_factor, pctl=1.0, use_sigma=False, use_histogram=False)
            if protect_filters:
                weight_scale = min(map(choose_scale_factor, list(flat_weight)))
                bias_scale = min(map(choose_bias_scale_factor, list(split_biases)))
            else:
                weight_scale = choose_scale_factor(flat_weight)
                bias_scale = choose_bias_scale_factor(split_biases)
            weight_scale = min(weight_scale.item(), bias_scale.item())
        else:
            wnb = torch.cat([flat_weight, split_biases], dim=1)
            weight_scale = choose_scale_factor(wnb).item()

    return weight_scale


def compute_n_sigma_bound(x, sigma=1.0, correct_mean=True, abs_mean=False, dim=None):
    """Approximate the bounds of the range of values with 2nd order stats.

    Parameters
    ----------
    x : torch.tensor
        A tensor of values to compute the sigma bound.
    sigma : float, optional
        The sigma multiple to use when computing the bound, by default 1.0
    correct_mean : bool, optional
        A flag whether to include the mean in the sigma bound computation, by default True
    abs_mean : bool, optional
        A flag whether to take the absolute value of the mean, by default False
    dim : int, optional
        The dimension to compute the statistics over for channel equalization, by default None

    Returns
    -------
    torch.tensor
        A tensor containing the bound(s) of the values provided.
    """
    if dim is None:
        # This causes problems if we pass None or [] as dim in backward
        x_var, x_mean = torch.var_mean(x)
    else:
        # These are torch specific arguments
        x_var, x_mean = torch.var_mean(x, dim=dim)
    x_std = torch.sqrt(x_var)

    bound = x_std*sigma
    if correct_mean:
        if abs_mean:
            x_mean = x_mean.abs()
        bound = bound + x_mean
    return bound


def apply_momentum(new_value, prev_value=None, momentum=0.9):
    """Average a value with its previous value with momentum.

    Parameters
    ----------
    new_value : torch.tensor, numpy.ndarray, or float
        Most recent value to by averaged by momentum.
    prev_value : torch.tensor, numpy.ndarray, or float, optional
        Previously averaged value, by default None.
    momentum : float, optional
        The momentum scaling factor to apply to the previous value, by default 0.9.

    Returns
    -------
    torch.tensor, numpy.ndarray, or float
        The averaged value.
    """
    if momentum is None:
        return new_value
    if prev_value is not None:
        smoothed_value = momentum * prev_value + (1.0-momentum) * new_value
    else:
        # prev_value can be None on the first call to this function
        smoothed_value = new_value
    return smoothed_value


def get_mma_layer_data(torch_model, node, dataloader, num_batches):
    """Get a torch module, weights, biases, device, and captured inputs of a Mythic MMA layer."""
    def capture_edge_data(torch_model, dataloader, edges, num_batches):
        """Capture edge values during execution of a model.

        Parameters
        ----------
        torch_model : TorchNet
            a TorchNet model to run.
        dataloader : Dataloader
            a dataloader to take model input data from.
        edges : List[str]
            Edges to capture values for.
        num_batches : int
            The number of batches to capture.

        Returns
        -------
        Dict[str, List[torch.Tensor]]
            A tensor of captured data for each batch for each edge.
        """
        torch_model.eval()
        res = {k: [] for k in edges}

        def on_set_edge_value(edge, value):
            if edge in res:
                res[edge].append(value.detach().cpu())

        with torch_model.set_edge_hook(on_set_edge_value):
            for batch in islice(dataloader, num_batches):
                with torch.no_grad():
                    torch_model(*batch)
        return res

    input_data = capture_edge_data(torch_model, dataloader, {node.input[0]}, num_batches)
    layer = torch_model.get_layer(node.name)
    weight = torch_model.get_initializer_value(node.input[1])
    bias = (torch_model.get_initializer_value(node.input[2]) if _node_utils.input_exists(node, 2)
            else torch.zeros(weight.shape[1], dtype=weight.dtype, device=weight.device))
    device = torch_model.device
    return layer, weight, bias, device, input_data
