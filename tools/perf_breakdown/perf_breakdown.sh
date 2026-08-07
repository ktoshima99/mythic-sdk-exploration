#!/usr/bin/env bash
# Emit every pre-max latency term of perf_analysis.py for one perf_trace_dump.h5.
#
# perf_analysis.py reports maximum_bottleneck(_accesses)_ns = max(...) over 3-4 terms,
# but only the terms of the hard-coded SRAM_METHOD ("accesses") reach the report table.
# This script runs the in-container perf_analysis.py twice against an already-generated
# trace: once unmodified (accesses method) and once with SRAM_METHOD patched to "bytes",
# and forces the DEBUG-gated per-tile breakdown into the report. No funcsim re-run.
#
# Usage: perf_breakdown.sh <perf_trace_dump.h5> <num_aces> [out_dir]
#   num_aces must match the artifact's --amp-arch (m2048 -> 48, m2072 -> 72).
#
# See doc/reverse-engineering/02_ppa_estimation.md §3.9.
set -euo pipefail

H5=$(realpath "${1:?usage: perf_breakdown.sh <perf_trace_dump.h5> <num_aces> [out_dir]}")
NUM_ACES="${2:?num_aces required (48 for m2048, 72 for m2072)}"
OUT_DIR=$(realpath -m "${3:-./perf_breakdown_out}")
IMAGE="${MYTHIC_COMPILER_IMAGE:-gcr.io/mythic-devops/compilerd-bin:v26.05.2}"

mkdir -p "$OUT_DIR/scripts"

# Pull perf_analysis.py out of the compiler image and generate the two patched variants.
docker run --rm --entrypoint bash -v "$OUT_DIR/scripts:/out" "$IMAGE" \
    -c 'cp /mythic/funcsim/bin/perf_analysis.py /out/perf_analysis_orig.py'

python3 - "$OUT_DIR/scripts" <<'PATCH'
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
src = d.joinpath("perf_analysis_orig.py").read_text()

# Force the two per-tile metric blocks, which are gated on the logger level purely for
# runtime cost. Leaving log-level at INFO keeps the per-timestep debug spew suppressed.
gate = "        if logging.getLogger().isEnabledFor(logging.DEBUG):"
assert src.count(gate) == 2, f"expected 2 per-tile gates, found {src.count(gate)}"
src = src.replace(gate, "        if True:  # patched: emit per-tile metrics at INFO")

for method in ("accesses", "bytes"):
    out = src.replace('SRAM_METHOD = "accesses"', f'SRAM_METHOD = "{method}"', 1)
    assert 'SRAM_METHOD = "%s"' % method in out
    d.joinpath(f"perf_analysis_{method}.py").write_text(out)
PATCH

for METHOD in accesses bytes; do
    echo "=== SRAM_METHOD=$METHOD ==="
    docker run --rm --entrypoint bash \
        -v "$(dirname "$H5")":/data:ro \
        -v "$OUT_DIR/scripts":/pa:ro \
        "$IMAGE" -c "source /mythic/pysim-env/bin/activate && \
            python /pa/perf_analysis_$METHOD.py \
                --hdf5-path /data/$(basename "$H5") \
                --num-aces $NUM_ACES \
                --report-level DETAILED --report-style JSON" \
        > "$OUT_DIR/report_$METHOD.json" 2>&1  # perf_analysis.py reports via logging -> stderr
    echo "  -> $OUT_DIR/report_$METHOD.json"
done

echo
echo "=== pre-max bottleneck terms ==="
python3 - "$OUT_DIR" <<'SHOW'
import json
import pathlib
import re
import sys

d = pathlib.Path(sys.argv[1])
# perf_analysis.py logs a preamble before the JSON array; slice from the first '['.
for method in ("accesses", "bytes"):
    raw = d.joinpath(f"report_{method}.json").read_text()
    rows = json.loads(raw[raw.index("["):raw.rindex("]") + 1])
    print(f"[{method}]")
    for r in rows:
        if re.search(r"Critical Path ACE|Maximum SRAM|Maximum SIMD|"
                     r"Analog NPU Total Estimated Processing", r["description"]):
            print(f"  {r['description']:<62} {r['metric']}")
SHOW
