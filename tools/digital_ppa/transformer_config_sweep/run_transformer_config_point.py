"""Run the digital (v-MP) PPA flow on BEVFormer-Tiny's Transformer-only graph
(TRANSFORMER_PART_ONLY=True, the shipped default -- see the plan's "scope decision" section
for why the full backbone+transformer graph is NOT used here) for one point in
(num_layers_enc, num_layers_dec, bev_h, bev_w, embed_dims) config space.

Mirrors tools/digital_ppa/run_full_digital.py's pipeline and its two defensive monkeypatches
(CapnprotoNetwork.add_tensor, inline_selected_functions dedup), but does NOT flip
TRANSFORMER_PART_ONLY, and additionally installs the dynamic-layer-count encoder/decoder
builders from dynamic_transformer_builders.py and overrides bev_h_/bev_w_/embed_dims on the
config module before constructing BevformerTiny.

Usage (inside the compilerd container):
    /mythic/pyvnnsdk-env/bin/python run_transformer_config_point.py \\
        --tag baseline --out /work/sweep_out/baseline \\
        --num-layers-enc 3 --num-layers-dec 6 --bev-h 50 --bev-w 50 --embed-dims 256

Writes <out>/result.json and also prints one JSON line to stdout (last line) so an outer
orchestrator can capture the result from subprocess stdout without a second file read.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mythic/vnnsdk/scripts")

import vnnmap.network as netmod  # noqa: E402

_TENSOR_TYPE_STATIC = 1  # TensorType.static in vnnmap/network_schema.capnp
_orig_add_tensor = netmod.CapnprotoNetwork.add_tensor


def _add_tensor_without_dynamic_data(
    self, name, tensor_type, data, fixed_point_data, max_exponents, adjusted_max_exponents, shape, n_bits, quant_axis
):
    """Drop calibration payloads on non-static tensors before capnp serialization.

    Copied verbatim from run_full_digital.py: guards against the >512 MB Cap'n Proto blob
    limit. Kept on defensively here too, since swept bev_h_/bev_w_/embed_dims could in
    principle grow a dynamic activation tensor past that limit (see plan risks section).
    """
    if data is not None and int(tensor_type) != _TENSOR_TYPE_STATIC:
        data = None
    return _orig_add_tensor(self, name, tensor_type, data, fixed_point_data, max_exponents, adjusted_max_exponents, shape, n_bits, quant_axis)


netmod.CapnprotoNetwork.add_tensor = _add_tensor_without_dynamic_data

import onnx.inliner as _inliner  # noqa: E402
import bevformer.bevformer_tiny as bevformer_tiny_module  # noqa: E402
import bevformer.bevformer_tiny_config as config  # noqa: E402
import bevformer.modeling.bevformer as bevformer_module  # noqa: E402
from bevformer.bevformer_tiny import BevformerTiny  # noqa: E402
from vnnmap import explore_model  # noqa: E402
from vnnort import configure_logging  # noqa: E402
from vnnort.quantizer.quantization_config import QuantizationConfig  # noqa: E402
from vnnort.utils.onnx_utils.meta_fields import _get_onnx_meta_field  # noqa: E402

from dynamic_transformer_builders import (  # noqa: E402
    build_bevformer_tiny_decoder_n,
    build_bevformer_tiny_encoder_n,
)
from power_component_breakdown import compute_breakdown  # noqa: E402

_orig_inline = _inliner.inline_selected_functions


def _dedupe_then_inline(model, function_names):
    """Remove duplicate local functions before inlining. Copied verbatim from
    run_full_digital.py: the dynamic-N builders call the same sub-layer @script functions
    repeatedly, structurally similar to the pattern that produced duplicate function ids on
    the full graph -- kept on defensively, it is a no-op when there is nothing to dedupe.
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

# TRANSFORMER_PART_ONLY intentionally left at its shipped default (True) -- do not set False.
BevformerTiny.TRANSFORMER_PART_ONLY = True

BEVFORMER_SYSTEM_CONFIG = Path("/mythic/vnnsdk/scripts/system_configs/bevformer.cfg")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-layers-enc", required=True, type=int)
    parser.add_argument("--num-layers-dec", required=True, type=int)
    parser.add_argument("--bev-h", required=True, type=int)
    parser.add_argument("--bev-w", required=True, type=int)
    parser.add_argument("--embed-dims", required=True, type=int)
    return parser.parse_args()


def apply_config_overrides(args):
    """Mutate the bevformer_tiny_config module + BevformerTiny class attrs in place.

    Must run BEFORE BevformerTiny(...) is constructed -- that is when initialize_onnx()
    fires and reads these values.
    """
    config.bev_h_ = args.bev_h
    config.bev_w_ = args.bev_w
    config._dim_ = args.embed_dims
    # _pos_dim_/_ffn_dim_ are derived once at import time (bevformer_tiny_config.py:40-41)
    # and do NOT auto-update when _dim_ is reassigned -- recompute explicitly.
    config._pos_dim_ = config._dim_ // 2
    config._ffn_dim_ = config._dim_ * 2

    # Cosmetic only: the graph builders never read config.model[...]["num_layers"] (they
    # hard-unroll a fixed layer count in Python -- see dynamic_transformer_builders.py's
    # docstring). Set it anyway so introspection of `config.model` stays truthful.
    transformer_cfg = config.model["pts_bbox_head"]["transformer"]
    transformer_cfg["encoder"]["num_layers"] = args.num_layers_enc
    transformer_cfg["decoder"]["num_layers"] = args.num_layers_dec

    # BevformerTiny.BEV_H/BEV_W/EMBED_DIMS are snapshotted from `config` at class-definition
    # time (bevformer_tiny.py:55-57) and do NOT auto-update from mutating `config` above.
    BevformerTiny.BEV_H = config.bev_h_
    BevformerTiny.BEV_W = config.bev_w_
    BevformerTiny.EMBED_DIMS = config._dim_


def install_dynamic_layer_builders(args):
    """Monkeypatch bevformer.modeling.bevformer's OWN namespace (not .encoder/.decoder --
    see dynamic_transformer_builders.py's module docstring for why that's the correct
    target), so build_bevformer_tiny_transformer() picks up the variable-N builders.
    """
    bevformer_module.build_bevformer_tiny_encoder = lambda cfg: build_bevformer_tiny_encoder_n(
        cfg, args.num_layers_enc
    )
    bevformer_module.build_bevformer_tiny_decoder = lambda cfg: build_bevformer_tiny_decoder_n(
        cfg, args.num_layers_dec
    )


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_logging()

    apply_config_overrides(args)
    install_dynamic_layer_builders(args)

    result = {"tag": args.tag, "params": vars(args) | {"out": str(args.out)}, "error": None}
    try:
        model = BevformerTiny(args.out)
        model.optimize()
        model.save(args.out / (model.model_name + ".vido.onnx"))
        model.quantize(QuantizationConfig(calibration_dataset_size=1))
        metrics, _layerwise_df = explore_model(model, system_config=BEVFORMER_SYSTEM_CONFIG, advanced=True)

        vidir_path = Path(_get_onnx_meta_field(model._model_repr, "model_directory")) / (model.model_name + ".vidir")
        power_breakdown = compute_breakdown(vidir_path, args.out / "power_breakdown")

        result.update(
            {
                "cycles": {
                    "mac_cycles": metrics["MAC Cycles"],
                    "non_mac_cycles": metrics["Non MAC Cycles"],
                    "exposed_dma_cycles": metrics["Exposed DMA Cycles"],
                    "total_cycles": metrics["Total Cycles"],
                },
                "performance": {
                    "effective_fps": metrics["Effective FPS"],
                    "effective_latency_ms": metrics["Effective Latency (ms)"],
                    "efficiency_pct": metrics["Efficiency (%)"],
                },
                "power_aggregate": {
                    "power_eff_fps_mW": metrics["Power@eff. fps (mW)"],
                    "power_30fps_mW": metrics["Power@30fps (mW)"],
                },
                "power_components": power_breakdown,
                "model_stats": {
                    "macs_bn": metrics["MACs (bn)"],
                    "model_size_MB": metrics["Model Size (MB)"],
                },
                "memory": {
                    "max_ddr_kB": metrics["Max DDR (kB)"],
                    "max_ocr_kB": metrics["Max OCR (kB)"],
                    "ddr_read_MB": metrics["DDR Read (MB)"],
                    "ddr_write_MB": metrics["DDR Write (MB)"],
                },
                "vidir_path": str(vidir_path),
            }
        )
    except Exception as exc:  # noqa: BLE001 - one failing point must not abort an outer sweep
        result["error"] = str(exc)[:2000]

    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
