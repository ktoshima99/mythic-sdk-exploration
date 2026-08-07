"""Run the v-NN (digital / v-MP) flow on the FULL BEVFormer-Tiny graph.

The shipped vnnsdk_scripts/bevformer entry point sets TRANSFORMER_PART_ONLY=True, so its
profiling JSON covers only the transformer (16.5 G MACs) and excludes the ResNet-50
backbone. This script forces the full graph, which requires two workarounds documented in
doc/reverse-engineering/05_all_digital_ppa.md §7.

Usage (inside the compilerd container; see §3 of that document):
    /mythic/pyvnnsdk-env/bin/python run_full_digital.py <result_directory>

Output: <result_directory>/BevformerTiny_proflings.json plus .vidi/.vido/.vidir artifacts.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/mythic/vnnsdk/scripts")

import vnnmap.network as netmod  # noqa: E402

_TENSOR_TYPE_STATIC = 1  # TensorType.static in vnnmap/network_schema.capnp
_orig_add_tensor = netmod.CapnprotoNetwork.add_tensor


def _add_tensor_without_dynamic_data(
    self,
    name,
    tensor_type,
    data,
    fixed_point_data,
    max_exponents,
    adjusted_max_exponents,
    shape,
    n_bits,
    quant_axis,
):
    """Drop calibration payloads on non-static tensors before capnp serialization.

    network_schema.capnp documents Tensor.data as carrying weight data for
    TensorType.static only. The full BEVFormer graph has dynamic activations above
    512 MB (e.g. x_0__1 [1,64,450,4800] = 552,960,000 B), which overflow a single
    Cap'n Proto blob and abort the export with "text blob too big".
    """
    if data is not None and int(tensor_type) != _TENSOR_TYPE_STATIC:
        data = None
    return _orig_add_tensor(
        self,
        name,
        tensor_type,
        data,
        fixed_point_data,
        max_exponents,
        adjusted_max_exponents,
        shape,
        n_bits,
        quant_axis,
    )


netmod.CapnprotoNetwork.add_tensor = _add_tensor_without_dynamic_data

import onnx.inliner as _inliner  # noqa: E402
import bevformer.bevformer_tiny as bevformer_tiny_module  # noqa: E402
from bevformer.bevformer_tiny import BevformerTiny  # noqa: E402
from mythic_utils import run_vnn_flow  # noqa: E402
from vnnort import configure_logging  # noqa: E402

_orig_inline = _inliner.inline_selected_functions


def _dedupe_then_inline(model, function_names):
    """Remove duplicate local functions before inlining.

    initialize_onnx() appends every cached com.videantis.dynamic_functions entry on top of
    the functions the onnxscript graph already carries. On the full graph this yields
    duplicate implementation ids (e.g. ResnetBlock0::ResnetBlock) and the ONNX checker
    rejects the model.
    """
    seen, keep = set(), []
    for function in model.functions:
        key = (function.domain, function.name)
        if key in seen:
            continue
        seen.add(key)
        keep.append(function)
    del model.functions[:]
    model.functions.extend(keep)
    return _orig_inline(model, function_names)


bevformer_tiny_module.inline_selected_functions = _dedupe_then_inline

BevformerTiny.TRANSFORMER_PART_ONLY = False

if __name__ == "__main__":
    result_directory = Path(sys.argv[1])
    result_directory.mkdir(parents=True, exist_ok=True)
    configure_logging()
    model = BevformerTiny(result_directory)
    run_vnn_flow(
        model,
        result_directory,
        system_config=Path("/mythic/vnnsdk/scripts/system_configs/bevformer.cfg"),
        skip_validation=True,
        advanced=True,
    )
