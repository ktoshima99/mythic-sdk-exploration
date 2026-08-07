"""Sweep vnnmap [sys] parameters against an already-quantized .vidir.

Usage (inside the compilerd container; see doc/reverse-engineering/05_all_digital_ppa.md §3):
    /mythic/pyvnnsdk-env/bin/python sweep_system_config.py <model.vidir> [cases.json]

cases.json is a list of dicts; each dict may override any key of DEFAULT_CASE and should
carry a "tag" naming the case (used for the output subdirectory). Without cases.json the
built-in CASES list is used. Results are written to <vidir_dir>/sweep/sweep.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/mythic/vnnsdk/scripts")
from vnnmap.run_vnnmap import run_vnnmap  # noqa: E402

CFG_TEMPLATE = """[sys]
mCluster={mcluster}
pCluster={pcluster}
nMPs={nmps}
OCRAM0={ocram0}
OCRAM1={ocram1}
DDR={ddr}
nomemFirstGraphInput=true
nomemGraphOutput=false
frequency={frequency}
nBatch=1
xTile={xtile}
"""

# Mirrors vnnsdk_scripts/system_configs/bevformer.cfg as shipped in v26.05.x
DEFAULT_CASE = dict(
    mcluster=1,
    pcluster=12,
    nmps=288,
    ocram0=33_554_432,
    ocram1=1_048_576,
    ddr=107_374_182_400,
    frequency=2_000_000_000,
    xtile=2,
)

CASES = [
    dict(tag="baseline"),
    dict(tag="nmps576", nmps=576),
    dict(tag="nmps1152", nmps=1152),
    dict(tag="xtile1", xtile=1),
    dict(tag="xtile4", xtile=4),
    dict(tag="ocram0_128M", ocram0=134_217_728, ocram1=16_777_216),
    dict(tag="ocram0_512M", ocram0=536_870_912, ocram1=16_777_216),
    dict(tag="ocram0_512M_f4G", ocram0=536_870_912, ocram1=16_777_216, frequency=4_000_000_000),
]

METRIC_KEYS = [
    "Effective FPS",
    "Effective Latency (ms)",
    "Efficiency (%)",
    "Power@eff. fps (mW)",
    "Power@30fps (mW)",
    "MACs (bn)",
    "Model Size (MB)",
    "MAC Cycles",
    "Non MAC Cycles",
    "Exposed DMA Cycles",
    "Total Cycles",
    "Max DDR (kB)",
    "Max OCR (kB)",
    "DDR Read (MB)",
    "DDR Write (MB)",
]


def main() -> None:
    vidir = Path(sys.argv[1]).absolute()
    cases = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else CASES
    work = vidir.parent / "sweep"
    work.mkdir(parents=True, exist_ok=True)

    results = []
    for index, case in enumerate(cases):
        params = {**DEFAULT_CASE, **{k: v for k, v in case.items() if k != "tag"}}
        tag = case.get("tag", f"case{index}")
        cfg = work / f"{tag}.cfg"
        cfg.write_text(CFG_TEMPLATE.format(**params))
        try:
            metrics, _ = run_vnnmap(
                vidir, work / tag, generate_vci=False, system_config=cfg, advanced=True
            )
            row = {"tag": tag, **params, **{k: metrics[k] for k in METRIC_KEYS}}
        except Exception as exc:  # noqa: BLE001 - one failing case must not abort the sweep
            row = {"tag": tag, **params, "error": str(exc)[:300]}
        results.append(row)
        print(json.dumps(row), flush=True)

    (work / "sweep.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {work / 'sweep.json'}")


if __name__ == "__main__":
    main()
