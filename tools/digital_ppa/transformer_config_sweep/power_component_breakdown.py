"""Automate the per-component digital power breakdown for a compiled BEVFormer(-family)
.vidir, by zeroing groups of `pow*Pj` [sys] cfg coefficients one at a time and reading the
delta in vnnmap's already-parsed "Power@eff. fps (mW)" metric.

This replaces the manual/ad-hoc technique described in
doc/reverse-engineering/05_all_digital_ppa.md §5 / 06_hybrid_digital_and_structural_analysis.md
§2.3 (hand-edit [sys] cfg, rerun vnnmap, read the delta by hand) with a reusable script.
Reuses the [sys] CFG_TEMPLATE / DEFAULT_SYS_PARAMS idiom from
tools/digital_ppa/sweep_system_config.py -- this only varies [sys] cfg against one
already-exported .vidir, so it is safe to loop in-process (no onnxscript/model-building
state is touched, unlike the model-config sweep axis in run_transformer_config_point.py).

Usage (inside the compilerd container):
    /mythic/pyvnnsdk-env/bin/python power_component_breakdown.py <model.vidir> [out_dir]
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

# Mirrors vnnsdk_scripts/system_configs/bevformer.cfg as shipped in v26.05.x (same values
# as tools/digital_ppa/sweep_system_config.py's DEFAULT_CASE).
DEFAULT_SYS_PARAMS = dict(
    mcluster=1,
    pcluster=12,
    nmps=288,
    ocram0=33_554_432,
    ocram1=1_048_576,
    ddr=107_374_182_400,
    frequency=2_000_000_000,
    xtile=2,
)

# pow*Pj [sys] cfg coefficient groups (doc/reverse-engineering/05_all_digital_ppa.md §5.1),
# bucketed into the same 6 components used in 06_hybrid_digital_and_structural_analysis.md
# §2.3's table so results stay directly comparable.
POW_GROUPS = {
    "mac_unit": ["powMacUnitPjPerCycle"],
    "non_mac_unit": ["powNonMacUnitPjPerCycle"],
    "dmem_imem": [
        "powDmemReadPj",
        "powDmemWritePj",
        "powDmem1ReadPj",
        "powDmem1WritePj",
        "powDmem2ReadPj",
        "powDmem2WritePj",
        "powDmem3ReadPj",
        "powDmem3WritePj",
        "powImemReadPj",
        "powImemWritePj",
    ],
    "ocram": ["powOcrMReadPj", "powOcrMWritePj", "powOcrCReadPj", "powOcrCWritePj"],
    "ddr": ["powDdrReadPj", "powDdrWritePj"],
    "bus_noc": [
        "powBusExtDdrIfPj",
        "powBusMatrixMPj",
        "powBusMatrixCPj",
        "powBusMatrixCIntPj",
        "powBusMatrixCExtPj",
        "powNocPj",
        "powNocIntraPj",
        "powNocInterPj",
    ],
}


def compute_breakdown(vidir_path, work_dir, sys_params=None):
    """Return per-component power contribution for the given .vidir.

    Args:
        vidir_path: Path to a compiled .vidir (already quantized/exported).
        work_dir: Scratch directory for per-case vnnmap outputs and generated cfg files.
        sys_params: [sys] cfg params (hardware config only -- NOT the pow*Pj coefficients).
            Defaults to DEFAULT_SYS_PARAMS (matches the shipped bevformer.cfg).

    Returns:
        dict with baseline power and, per component, the delta/fraction/30fps-scaled delta.
    """
    vidir_path = Path(vidir_path).absolute()
    work_dir = Path(work_dir).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)
    params = {**DEFAULT_SYS_PARAMS, **(sys_params or {})}

    baseline_cfg = work_dir / "baseline.cfg"
    baseline_cfg.write_text(CFG_TEMPLATE.format(**params))  # no pow*Pj overrides = tool defaults
    baseline_metrics, _ = run_vnnmap(vidir_path, work_dir / "baseline", generate_vci=False, system_config=baseline_cfg, advanced=True)
    base_power = baseline_metrics["Power@eff. fps (mW)"]
    base_power_30fps = baseline_metrics["Power@30fps (mW)"]

    components = {}
    for name, keys in POW_GROUPS.items():
        cfg_path = work_dir / f"{name}.cfg"
        cfg_path.write_text(CFG_TEMPLATE.format(**params) + "".join(f"{k}=0\n" for k in keys))
        metrics, _ = run_vnnmap(vidir_path, work_dir / name, generate_vci=False, system_config=cfg_path, advanced=True)
        zeroed_power = metrics["Power@eff. fps (mW)"]
        delta = base_power - zeroed_power
        components[name] = {
            "power_eff_fps_zeroed_mW": zeroed_power,
            "delta_mW": delta,
            "fraction_of_total": delta / base_power if base_power else None,
        }

    scale = base_power_30fps / base_power if base_power else None
    for c in components.values():
        c["delta_mW_at_30fps"] = c["delta_mW"] * scale if scale is not None else None

    return {
        "baseline_power_eff_fps_mW": base_power,
        "baseline_power_30fps_mW": base_power_30fps,
        "components": components,
    }


if __name__ == "__main__":
    vidir = Path(sys.argv[1]).absolute()
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else vidir.parent / "power_breakdown"
    result = compute_breakdown(vidir, out_dir)
    print(json.dumps(result, indent=2))
    (out_dir / "power_breakdown.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_dir / 'power_breakdown.json'}")
