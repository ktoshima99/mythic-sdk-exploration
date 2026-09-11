"""Outer orchestrator for the BEVFormer-Tiny Transformer-config PPA sweep.

Each sweep point runs as its own fresh subprocess (run_transformer_config_point.py), never
in-process, because:
  1. initialize_onnx() walks a process-global onnxscript.values.Opset.cache
     (bevformer_tiny.py:168-176) and appends every cached function def into whatever model
     is built next -- building multiple different model-config points in one process risks
     stale/wrong-shape cached function defs leaking across points (a silent correctness bug).
  2. The config-module/class-attribute overrides applied per point are global singleton
     state; a fresh subprocess makes cleanup trivially correct (process exit) instead of
     requiring careful save/restore between points.

Usage (inside the compilerd container):
    /mythic/pyvnnsdk-env/bin/python sweep_transformer_config.py /work/sweep_out
Writes <out_root>/sweep_transformer_config.json (list of all per-point result dicts).
"""

import json
import subprocess
import sys
from pathlib import Path

PYTHON = "/mythic/pyvnnsdk-env/bin/python"
POINT_SCRIPT = Path(__file__).resolve().parent / "run_transformer_config_point.py"

BASELINE = dict(tag="baseline", num_layers_enc=3, num_layers_dec=6, bev_h=50, bev_w=50, embed_dims=256)

# Independent 1-D sweeps around BASELINE -- one parameter group varied at a time, everything
# else held at baseline. Small initial scope per plan (~3-4 points/axis).
INPUT_SIZE_CASES = [
    dict(tag="bev_25", bev_h=25, bev_w=25),
    dict(tag="bev_75", bev_h=75, bev_w=75),
    dict(tag="embed_128", embed_dims=128),  # divisible by num_heads=8
    dict(tag="embed_384", embed_dims=384),
]
LAYER_COUNT_CASES = [
    dict(tag="enc_1", num_layers_enc=1),
    dict(tag="enc_5", num_layers_enc=5),
    dict(tag="dec_2", num_layers_dec=2),
    dict(tag="dec_10", num_layers_dec=10),
]


def run_one(case, out_root, timeout_s=1800):
    params = {**BASELINE, **case}
    tag = params["tag"]
    out_dir = out_root / tag
    args = [
        PYTHON,
        str(POINT_SCRIPT),
        "--tag",
        tag,
        "--out",
        str(out_dir),
        "--num-layers-enc",
        str(params["num_layers_enc"]),
        "--num-layers-dec",
        str(params["num_layers_dec"]),
        "--bev-h",
        str(params["bev_h"]),
        "--bev-w",
        str(params["bev_w"]),
        "--embed-dims",
        str(params["embed_dims"]),
    ]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            return {"tag": tag, "params": params, "error": proc.stderr[-2000:]}
        # run_transformer_config_point.py's own stdout may include SDK log noise on earlier
        # lines (e.g. onnxruntime device-discovery warnings) -- the result row is always the
        # single JSON object it prints as its last stdout line.
        last_line = proc.stdout.strip().splitlines()[-1]
        return json.loads(last_line)
    except Exception as exc:  # noqa: BLE001 - one failing point must not abort the sweep
        return {"tag": tag, "params": params, "error": str(exc)[:2000]}


def main():
    out_root = Path(sys.argv[1]).absolute()
    out_root.mkdir(parents=True, exist_ok=True)

    # Run BASELINE once; INPUT_SIZE_CASES/LAYER_COUNT_CASES both use it as their "center"
    # point without re-running it.
    cases = [BASELINE, *INPUT_SIZE_CASES, *LAYER_COUNT_CASES]

    results = []
    for case in cases:
        row = run_one(case, out_root)
        results.append(row)
        print(json.dumps(row), flush=True)

    (out_root / "sweep_transformer_config.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_root / 'sweep_transformer_config.json'}")


if __name__ == "__main__":
    main()
