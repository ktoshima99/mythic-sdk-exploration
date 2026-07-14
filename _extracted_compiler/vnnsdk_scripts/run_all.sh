#!/usr/bin/env bash
# Run all Mythic postprocessing scripts against extracted artifact directories.
#
# Usage:
#   run_all.sh <mythic_artifacts> <result_directory>
#
# <mythic_artifacts> must contain these extracted artifact directories:
#   artifacts-v-yolov8-pose-2026-04-24T213935+0000
#   artifacts-v-yolopx-2026-05-01T090146
#   artifacts-v-resnet50_imagenet-2026-04-24T190830
#   artifacts-v-yolov8-2026-05-18T184643+0000
#   bevformer

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <mythic_artifacts> <result_directory>" >&2
    exit 1
fi

mythic_artifacts="$1"
result_directory="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
onnx_subpath="compiler_ready_artifact/reference/compiler_ready_artifact_off_chip_2.onnx"

YOLOV8_POSE_DIR="artifacts-v-yolov8-pose-2026-04-24T213935+0000"
YOLOPX_DIR="artifacts-v-yolopx-2026-05-01T090146"
RESNET50_DIR="artifacts-v-resnet50_imagenet-2026-04-24T190830"
YOLOV8_DIR="artifacts-v-yolov8-2026-05-18T184643+0000"
BEVFORMER_DIR="bevformer"

if [[ ! -d "$mythic_artifacts" ]]; then
    echo "Error: '$mythic_artifacts' is not a directory." >&2
    exit 1
fi

if ! python -c "import vnnort" 2>/dev/null; then
    echo "Error: cannot 'import vnnort' with current python — activate the right environment." >&2
    exit 1
fi

run_one() {
    local script="$1"
    local artifact_subdir="$2"  # empty when the script builds its own ONNX (e.g. bevformer)
    local result_subdir="$3"

    echo "=== Running $script ==="
    echo "  result_directory: $result_directory/$result_subdir"

    if [[ -n "$artifact_subdir" ]]; then
        local onnx_path="$mythic_artifacts/$artifact_subdir/$onnx_subpath"
        if [[ ! -f "$onnx_path" ]]; then
            echo "Error: expected ONNX not found at '$onnx_path'." >&2
            exit 1
        fi
        echo "  source_onnx:      $onnx_path"
        python "$script_dir/$script" \
            --source_onnx="$onnx_path" \
            --result_directory="$result_directory/$result_subdir"
    else
        python "$script_dir/$script" \
            --result_directory="$result_directory/$result_subdir"
    fi
}

mkdir -p "$result_directory"

run_one yolov8pose_postprocessing.py  "$YOLOV8_POSE_DIR" MythicYoloV8PosePostprocessing
run_one yolopx_postprocessing.py      "$YOLOPX_DIR"      MythicYoloPXPostprocessing
run_one resnet50_postprocessing.py    "$RESNET50_DIR"    MythicResnet50Postprocessing
run_one yolov8_postprocessing.py      "$YOLOV8_DIR"      MythicYoloV8Postprocessing
run_one bevformer_postprocessing.py   "$BEVFORMER_DIR"   MythicBevformerPostprocessing
