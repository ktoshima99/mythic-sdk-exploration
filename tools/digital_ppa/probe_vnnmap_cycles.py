"""gdb script: read vnnmap's internal cycle accumulators and re-derive fps/latency.

Confirms the formulas documented in doc/reverse-engineering/05_all_digital_ppa.md
§2.2-§2.4 by breaking at vNetwork::printFullProfile, dumping the vSummaryProfile
accumulators, and independently recomputing them from the layer array.

Usage (inside the compilerd container; see 05_all_digital_ppa.md §2.5 / §3.1):

    VN=/mythic/pyvnnsdk-env/lib/python3.12/site-packages/vnnmap/vnnmap
    gdb -q -batch -x /work/probe_vnnmap_cycles.py --args $VN \
        --csv_dir=./csv --output_prefix=P \
        --network=/work/out_full/BevformerTiny.vidir \
        --system_cfg=/work/system_configs/bevformer.cfg --explore --edma

Every PROBE-RECOMP value should equal its PROBE-STORED counterpart. PROBE-DERIVED
should reproduce the `eff. fps` / `eff. latency` / `efficiency` line that vnnmap
prints, and will NOT match a recomputation from the printed `Total` (§2.4).
"""

import struct

import gdb

gdb.execute("set pagination off")
gdb.execute("set confirm off")

# vNetwork field offsets (§2.2)
NET_LAYERS_BEGIN = 0x38
NET_LAYERS_END = 0x40
NET_NMPS = 0x120
NET_NMACS = 0x124
NET_NBATCH = 0x130
NET_FREQUENCY = 0x138
NET_MACS_BN = 0xD38  # double, total MACs in billions
NET_PARAMS = 0xD40
NET_PARAM_BYTES = 0xD48
NET_IDEAL_CYCLES = 0xD50  # double, vSummaryProfile
NET_PROC_CYCLES = 0xD58  # double, vSummaryProfile -> latency denominator
NET_DMA_CYCLES = 0xD60  # int64,  vSummaryProfile -> latency denominator

# vLayer field offsets
LAY_MACS_BN = 0x228  # double
LAY_PARAMS = 0x230
LAY_PARAM_BYTES = 0x234
LAY_SKIP = 0x1B8  # nonzero -> layer excluded from the summary
LAY_PROFILES = 0x240  # std::vector<vProfile>
LAY_MAX_CYCLES = 0x598  # int64, max over tiles (vLayer::calcMaxCycles)

SIZEOF_VPROFILE = 0x770
PROF_MAC_CYCLES = 0x720
PROF_NONMAC_CYCLES = 0x728

MAC_FLOOR = 1.1  # rodata 0x245c00
EFF_MACS_PER_CYCLE_PER_MP = 64  # rodata 0x245c10 == 1/64; note: ignores nMACs


class PrintFullProfileBP(gdb.Breakpoint):
    def stop(self):
        inf = gdb.selected_inferior()
        net = int(gdb.parse_and_eval("$rdi"))  # vNetwork* (ASLR-safe)

        def raw(addr, n):
            return bytes(inf.read_memory(addr, n))

        def u32(addr):
            return struct.unpack("<I", raw(addr, 4))[0]

        def i64(addr):
            return struct.unpack("<q", raw(addr, 8))[0]

        def u64(addr):
            return struct.unpack("<Q", raw(addr, 8))[0]

        def f64(addr):
            return struct.unpack("<d", raw(addr, 8))[0]

        n_mps = u32(net + NET_NMPS)
        n_macs = u32(net + NET_NMACS)
        n_batch = u32(net + NET_NBATCH)
        freq = i64(net + NET_FREQUENCY)
        peak_macs_per_cycle = 2 * n_mps * n_macs

        begin, end = u64(net + NET_LAYERS_BEGIN), u64(net + NET_LAYERS_END)
        n_ptrs = (end - begin) // 8

        ideal = actual = actual_unfloored = 0.0
        mac_cycles = nonmac_cycles = 0
        n_layers = n_skipped = n_floored = 0

        for i in range(n_ptrs):
            layer = u64(begin + 8 * i)
            if raw(layer + LAY_SKIP, 1)[0] != 0:
                n_skipped += 1
                continue
            n_layers += 1

            layer_ideal = f64(layer + LAY_MACS_BN) * 1e9 / peak_macs_per_cycle
            layer_actual = float(i64(layer + LAY_MAX_CYCLES))
            ideal += layer_ideal
            actual_unfloored += layer_actual
            if layer_ideal > layer_actual:
                n_floored += 1
                layer_actual = MAC_FLOOR * layer_ideal
            actual += layer_actual

            pbeg, pend = u64(layer + LAY_PROFILES), u64(layer + LAY_PROFILES + 8)
            for j in range((pend - pbeg) // SIZEOF_VPROFILE):
                prof = pbeg + j * SIZEOF_VPROFILE
                mac_cycles += i64(prof + PROF_MAC_CYCLES)
                nonmac_cycles += i64(prof + PROF_NONMAC_CYCLES)

        stored_ideal = f64(net + NET_IDEAL_CYCLES)
        stored_proc = f64(net + NET_PROC_CYCLES)
        stored_dma = i64(net + NET_DMA_CYCLES)
        macs_bn = f64(net + NET_MACS_BN)

        print(
            "PROBE-CFG      nMPs=%d nMACs=%d nBatch=%d frequency=%d peakMACs/cycle=%d"
            % (n_mps, n_macs, n_batch, freq, peak_macs_per_cycle)
        )
        print(
            "PROBE-LAYERS   ptrs=%d counted=%d skipped=%d mac_floor_applied=%d"
            % (n_ptrs, n_layers, n_skipped, n_floored)
        )
        print(
            "PROBE-RECOMP   ideal=%.4f proc=%.4f proc_unfloored=%.4f floor_delta=%.4f"
            % (ideal, actual, actual_unfloored, actual - actual_unfloored)
        )
        print(
            "PROBE-STORED   ideal=%.4f proc=%.4f dma=%d macs_bn=%.6f"
            % (stored_ideal, stored_proc, stored_dma, macs_bn)
        )
        print(
            "PROBE-PROFSUM  MAC=%d nonMAC=%d sum=%d  (per-vProfile totals, all layers)"
            % (mac_cycles, nonmac_cycles, mac_cycles + nonmac_cycles)
        )

        cycles = stored_proc + stored_dma
        print(
            "PROBE-DERIVED  cycles=%.3f eff_fps=%.4f eff_latency_ms=%.6f efficiency=%.4f%%"
            % (
                cycles,
                n_batch * freq / cycles,
                1000.0 * cycles / freq,
                macs_bn * 1e9 * 100 / (cycles * EFF_MACS_PER_CYCLE_PER_MP * n_mps),
            )
        )
        print(
            "PROBE-HIDDEN   cycles=%.3f max_fps=%.4f min_latency_ms=%.6f  (DMAs hidden)"
            % (stored_proc, n_batch * freq / stored_proc, 1000.0 * stored_proc / freq)
        )
        return True


PrintFullProfileBP("vNetwork::printFullProfile")
gdb.execute("run")
gdb.execute("quit")
