# type: ignore
import onnx
from onnxscript import BOOL, FLOAT, INT8, INT64
from onnxscript import opset19 as op  # TODO: streamline this with optimizer
from onnxscript import script, values


@script(values.Opset("com.videantis", 1))
def QDQLayer(x: FLOAT, max_exponents: INT8, skip: BOOL, n_bits: INT64, fraction_bits: INT64) -> FLOAT:
    """Quantize and then immediately dequantize the input tensor x.

    Args:
        x (FLOAT): Input tensor to be quantized
        max_exponents (INT8): max exponents to be applied
        skip (BOOL): Whether to skip this node or not (useful when calibrating quantization)
        n_bits (INT64): overall number of bits to use
        fraction_bits (INT64): number of fraction bits (out of total n_bits) to use

    Returns:
        FLOAT: Result tensor which is result of quantization and dequantization of input tensor
    """
    if skip:
        result = x
    else:
        max_exponents = op.Cast(max_exponents, to=onnx.TensorProto.FLOAT)
        scale = op.Pow(2.0, -max_exponents)
        resolution = op.Pow(2.0, fraction_bits)

        scaled = x * scale * resolution
        quantized = op.Round(scaled)
        quant_max = op.Pow(2.0, n_bits - 1)
        clipped = op.Clip(quantized, -quant_max, quant_max - 1)
        dequantized = clipped / (scale * resolution)
        result = dequantized
    return result
