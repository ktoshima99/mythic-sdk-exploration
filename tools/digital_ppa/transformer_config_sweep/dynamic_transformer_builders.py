"""Source-generated, variable-layer-count encoder/decoder builders for BEVFormer-Tiny.

bevformer.modeling.encoder.build_bevformer_tiny_encoder (N=3) and
bevformer.modeling.decoder.build_bevformer_tiny_decoder (N=6) hard-unroll their layer
stack in Python -- config["...']['num_layers'] is never read. These functions generate
the @script-decorated graph function's Python source as a string and exec() it, so the
generated graph body contains no loop construct at all: it looks exactly like the
hand-written N=3/N=6 version, just emitted programmatically for arbitrary N. This avoids
any dependency on whether onnxscript's restricted Python subset supports indexing into a
closed-over list of distinct sub-layer callables (each with its own baked-in random
weights) from within a native `for` loop in the graph body.

The generated @script function is nested inside a `_factory(layer_0_0, layer_0_1, ...)`
def and called with the actual built layer callables as arguments, so `layer_0_0` etc.
are real Python closure variables of the returned OnnxFunction -- exactly like the
hand-written version, where they are locals of the enclosing `build_bevformer_tiny_*`
function. This matters because onnxscript's @script decorator resolves free variables
via `inspect.getclosurevars(f).nonlocals`, which is only populated for true closures.

Usage (see run_transformer_config_point.py):
    import bevformer.modeling.bevformer as bevformer_module
    bevformer_module.build_bevformer_tiny_encoder = lambda config: build_bevformer_tiny_encoder_n(config, N_ENC)
    bevformer_module.build_bevformer_tiny_decoder = lambda config: build_bevformer_tiny_decoder_n(config, N_DEC)
This must patch `bevformer.modeling.bevformer`'s own namespace (the names are bound there
at its import time), not `bevformer.modeling.encoder`/`decoder` themselves.
"""

import itertools
import linecache
import textwrap

from onnxscript import FLOAT
from onnxscript import opset20 as op
from onnxscript import script

from bevformer.modeling.decoder import (
    build_decoder_deformable_attention,
    build_multi_head_self_attention,
    build_reg_branch,
)
from bevformer.modeling.encoder import (
    build_spatial_cross_attention,
    build_temporal_self_attention,
)
from vnnort.optimizer.functions import Shortcut

_ENCODER_SIGNATURE = (
    "def BevformerTinyEncoder(\n"
    '    bev_query: FLOAT["B", "C", 1, "N_q"],\n'
    '    bev_value_stacked: FLOAT["1", "C", "Q", "N_q"],\n'
    '    img_features: FLOAT["1", "C", "N_cam", "H_feat*W_feat"],\n'
    '    bev_mask: FLOAT["B", "1", "N_cam", "N_q"],\n'
    '    ref_pts_cam: FLOAT["1", "N_cam", "N_q", "-1"],\n'
    '    count_normalizer: FLOAT["B", "1", "1", "N_q"],\n'
    ') -> FLOAT["B", "C", 1, "N_q"]:\n'
)

_DECODER_SIGNATURE = (
    "def BevformerTinyDecoder(\n"
    '    obj_queries: FLOAT["B", "C", 1, "N_obj"],\n'
    '    bev_features: FLOAT["B", "C", 1, "N_bev"],\n'
    '    ref_points: FLOAT["B", 3, 1, "N_obj"],\n'
    "):\n"
)

_source_counter = itertools.count()


def _build_and_call_factory(decorator_and_def, body_lines, func_name, param_names, param_values, tag):
    """Wrap `decorator_and_def` + body in a `_factory(*param_names)` def, exec it under a
    linecache-registered synthetic filename (so @script's inspect.getsource() works), and
    call the factory with `param_values` so the sub-layer callables become real closures.
    """
    inner_src = decorator_and_def + "\n".join(body_lines)
    inner_src_indented = textwrap.indent(inner_src, "    ")
    src = f"def _factory({', '.join(param_names)}):\n" + inner_src_indented + f"\n    return {func_name}\n"

    filename = f"<generated_{tag}_{next(_source_counter)}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(keepends=True), filename)
    # onnxscript's @script decorator does `inspect.getmodule(f).__dict__` to build its name
    # resolution environment; inspect.getmodule() looks up sys.modules[f.__module__], which
    # requires `__name__` to be present in the exec() globals and registered in sys.modules.
    # Point it at this module (already imported/registered) -- its globals (op/script/FLOAT/
    # Shortcut) are exactly what a hand-written BevformerTiny*.py file would expose too.
    ns = {"op": op, "script": script, "FLOAT": FLOAT, "Shortcut": Shortcut, "__name__": __name__}
    exec(compile(src, filename, "exec"), ns)  # noqa: S102
    return ns["_factory"](*param_values)


def build_bevformer_tiny_encoder_n(config, num_layers):
    """Variable-N version of bevformer.modeling.encoder.build_bevformer_tiny_encoder.

    Build order mirrors the hardcoded original exactly: for each layer index i,
    build (temporal_self_attention, spatial_cross_attention) as an interleaved pair.
    """
    layer_pairs = [
        (build_temporal_self_attention(config), build_spatial_cross_attention(config)) for _ in range(num_layers)
    ]

    param_names, param_values = [], []
    for i, (temporal, spatial) in enumerate(layer_pairs):
        param_names += [f"layer_{i}_0", f"layer_{i}_1"]
        param_values += [temporal, spatial]

    body_lines = ["    out = bev_query"]
    for i in range(num_layers):
        body_lines.append(f"    out = layer_{i}_0(out, bev_value_stacked)")
        body_lines.append(f"    out = layer_{i}_1(out, img_features, bev_mask, ref_pts_cam, count_normalizer)")
    body_lines.append("    return out")

    return _build_and_call_factory(
        "@script(default_opset=op)\n" + _ENCODER_SIGNATURE,
        body_lines,
        "BevformerTinyEncoder",
        param_names,
        param_values,
        f"encoder_n{num_layers}",
    )


def build_bevformer_tiny_decoder_n(config, num_layers):
    """Variable-N version of bevformer.modeling.decoder.build_bevformer_tiny_decoder.

    Build order mirrors the hardcoded original exactly: all N (self_attn, cross_attn)
    pairs first, then all N reg_branch_i. The graph body is the same loop-carried
    residual-refinement chain, with `out`/`ref` reassigned each iteration instead of
    being renamed out_0/out_1/... -- an equally valid pattern already used throughout
    the hand-written sub-builders (e.g. `out` is reassigned repeatedly within
    TemporalSelfAttention/SpatialCrossAttention).
    """
    attn_pairs = [
        (build_multi_head_self_attention(config), build_decoder_deformable_attention(config))
        for _ in range(num_layers)
    ]
    reg_branches = [build_reg_branch(config) for _ in range(num_layers)]

    param_names, param_values = [], []
    for i, (self_attn, cross_attn) in enumerate(attn_pairs):
        param_names += [f"layer_{i}_0", f"layer_{i}_1"]
        param_values += [self_attn, cross_attn]
    for i, reg_branch in enumerate(reg_branches):
        param_names.append(f"reg_branch_{i}")
        param_values.append(reg_branch)

    # Variable naming (out_i / ref_points_i per iteration, rather than reassigning a single
    # `out`/`ref`) intentionally mirrors decoder.py's hand-written naming exactly, so the
    # generated graph is byte-identical (not just structurally equivalent) to the hand-written
    # one at N=6 -- this is what verify_dynamic_layers.py checks.
    body_lines = ["    out = obj_queries"]
    for i in range(num_layers):
        prev_out = "out" if i == 0 else f"out_{i - 1}"
        prev_ref = "ref_points" if i == 0 else f"ref_points_{i - 1}"
        body_lines.append(f"    out_{i} = layer_{i}_0({prev_out})")
        body_lines.append(f"    out_{i} = layer_{i}_1(out_{i}, bev_features, {prev_ref})")
        body_lines.append(
            f"    ref_points_{i} = op.Gather(reg_branch_{i}(out_{i}), op.Constant(value_ints=[0, 1, 4]), axis=1)"
        )
        body_lines.append(f'    ref_points_{i} = Shortcut(ref_points_{i}, {prev_ref}, mode="addition")')
    body_lines.append(f"    return out_{num_layers - 1}, ref_points_{num_layers - 1}")

    return _build_and_call_factory(
        "@script(default_opset=op)\n" + _DECODER_SIGNATURE,
        body_lines,
        "BevformerTinyDecoder",
        param_names,
        param_values,
        f"decoder_n{num_layers}",
    )
