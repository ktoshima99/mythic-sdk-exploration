#!/usr/bin/env python3
"""Emit the per-component / per-op-type power breakdown that power_estimator.py computes
internally (OpEnergy: ace_active, ace_sleep, sram, accessor, control, noc) but never
prints for the grand total in its own CLI path (calc_power() only ever logs the sum
".total" per operation type, and its debug-only full breakdown print is gated behind a
call to calc_energy_interconnect() that crashes when no packet log is given).

This script calls M2000_Power's internal calc_energy_*() methods directly instead of
calc_power(), so it works standalone without a running compiler container -- power
estimation is a pure parse of the compiled final.l0.pb, no funcsim re-run needed
(compare tools/perf_breakdown/perf_breakdown.sh, which needs the compiler image because
perf_analysis.py's HDF5 trace path requires the in-container source; power_estimator.py
has no such dependency once mythic_pkg is on sys.path).

Usage: power_breakdown.py <ppa_artifact.tar.gz> <num_aces> <inf_rate> [out_dir]
  num_aces must match the artifact's --amp-arch (m2048 -> 48, m2072 -> 72).
  inf_rate is the target frames/second used for the reported W figures.

See doc/reverse-engineering/07_ppa_improvement_challenges.md §3-10.
"""
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MYTHIC_PKG_DIR = REPO_ROOT / "mythic_sdk/v26.05.0/_extracted_compiler/mythic_pkg"
POWER_ESTIMATOR_DIR = MYTHIC_PKG_DIR / "m2000_power_estimator"


def main():
    if len(sys.argv) < 4:
        sys.exit(f"usage: {sys.argv[0]} <ppa_artifact.tar.gz> <num_aces> <inf_rate> [out_dir]")
    artifact_path = Path(sys.argv[1]).resolve()
    num_aces = int(sys.argv[2])
    inf_rate = int(sys.argv[3])
    out_dir = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else Path(tempfile.mkdtemp(prefix="power_breakdown_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    members = ["artifacts/firmware/final.l0.pb", "artifacts/ppa/packet_log.json", "artifacts/ppa/event_log.json"]
    with tarfile.open(artifact_path) as tf:
        tf.extractall(out_dir, members=[m for m in tf.getmembers() if m.name in members])

    # mythic_pkg's on-disk directory name doesn't match the "mythic" import root the
    # sources use ("from mythic.irs.l0.ir_pb2 import ..."); alias it via a symlink.
    pylibs = out_dir / "pylibs"
    pylibs.mkdir(exist_ok=True)
    mythic_link = pylibs / "mythic"
    if not mythic_link.exists():
        mythic_link.symlink_to(MYTHIC_PKG_DIR)
    sys.path.insert(0, str(pylibs))
    sys.path.insert(0, str(POWER_ESTIMATOR_DIR))

    import logging
    logging.basicConfig(level=logging.ERROR)  # suppress per-op debug spam; we print our own summary
    from power_estimator import M2000_Power, Operation, OpEnergy  # noqa: E402

    m = M2000_Power(
        l0_pb_path=out_dir / "artifacts/firmware/final.l0.pb",
        inf_rate=inf_rate,
        digital_process_node=5,
        num_aces=num_aces,
        num_pcie_lanes=4,
        pcie_activity_factor=0.3,
        num_d2d_lanes=12,
        num_die_in_system=1,
        packet_log_path=out_dir / "artifacts/ppa/packet_log.json",
        event_log_path=out_dir / "artifacts/ppa/event_log.json",
    )

    per_op = {
        Operation.ACE: m.calc_energy_ace(),
        Operation.COPY: m.calc_energy_copy(),
        Operation.SIMD: m.calc_energy_simd(),
        Operation.PAD: m.calc_energy_pad(),
        Operation.INFEED: m.calc_energy_infeed(),
        Operation.OUTFEED: m.calc_energy_outfeed(),
    }
    grand = OpEnergy()
    for e in per_op.values():
        grand += e
    interconnect_power = m.calc_energy_interconnect() * inf_rate
    functional_unit_power = grand.total * inf_rate
    total_power = functional_unit_power + interconnect_power

    print(f"=== num_aces={num_aces}, inf_rate={inf_rate}fps, artifact={artifact_path.name} ===\n")
    header = (f"{'Op type':10s} {'ACE_active(W)':>14s} {'ACE_sleep(W)':>13s} {'SRAM(W)':>10s} "
              f"{'Accessor(W)':>12s} {'Control(W)':>11s} {'NOC(W)':>8s} {'Total(W)':>10s}")
    print(header)
    for op, e in per_op.items():
        print(f"{op.name:10s} {e.ace_active*inf_rate:14.5f} {e.ace_sleep*inf_rate:13.5f} {e.sram*inf_rate:10.5f} "
              f"{e.accessor*inf_rate:12.5f} {e.control*inf_rate:11.5f} {e.noc*inf_rate:8.5f} {e.total*inf_rate:10.5f}")
    print("-" * len(header))
    print(f"{'TOTAL':10s} {grand.ace_active*inf_rate:14.5f} {grand.ace_sleep*inf_rate:13.5f} {grand.sram*inf_rate:10.5f} "
          f"{grand.accessor*inf_rate:12.5f} {grand.control*inf_rate:11.5f} {grand.noc*inf_rate:8.5f} {grand.total*inf_rate:10.5f}")

    print(f"\nInterconnect (NOC, from packet log): {interconnect_power:.4f} W")
    print(f"\nTOTAL analog power (functional unit + interconnect): {total_power:.4f} W")
    print(f"  Functional Unit: {functional_unit_power:.4f} W ({100*functional_unit_power/total_power:.1f}%)")
    print(f"    - ACE:      {grand.ace*inf_rate:.4f} W ({100*grand.ace*inf_rate/total_power:.1f}%)")
    print(f"    - SRAM:     {grand.sram*inf_rate:.4f} W ({100*grand.sram*inf_rate/total_power:.1f}%)")
    print(f"    - Accessor: {grand.accessor*inf_rate:.4f} W ({100*grand.accessor*inf_rate/total_power:.1f}%)")
    print(f"    - Control:  {grand.control*inf_rate:.4f} W ({100*grand.control*inf_rate/total_power:.1f}%)")
    print(f"  Interconnect (NOC): {interconnect_power:.4f} W ({100*interconnect_power/total_power:.1f}%)")


if __name__ == "__main__":
    main()
