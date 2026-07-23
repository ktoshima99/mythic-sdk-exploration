#!/bin/bash

VERSION_FILE="${VERSION_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/archive/SDK-VERSION}"
VERSION="${VERSION:-$(cat "${VERSION_FILE}" 2>/dev/null || true)}"
VERSION="${VERSION//[[:space:]]/}"
VERSION="${VERSION#v}"

if [[ -z "${VERSION}" ]]; then
    echo "Error: could not determine SDK version. Provide one of:" >&2
    echo "  VERSION env var (e.g. VERSION=v26.05.0)" >&2
    echo "  VERSION_FILE env var pointing to a version file" >&2
    echo "  A file at ${VERSION_FILE}" >&2
    exit 1
fi

# Host paths. The defaults are set for Collaboration Chamber compatibility. These
# can be overridden by setting the corresponding environment variables or at script
# invocation time. For example:
# MYTHIC_WORKSPACE=/path/to/workspace DATASET_DIR=/path/to/datasets ./run_mythic_sdk_container.sh
MYTHIC_WORKSPACE="${MYTHIC_WORKSPACE:-$HOME}"
MYTHIC_TMPDIR="${MYTHIC_TMPDIR:-/tmp/data}"
DATASET_DIR="${DATASET_DIR:-/projects/tonbomythic3/datasets}"
TRAINING_MODELS_HOST_DIR="${TRAINING_MODELS_HOST_DIR:-/projects/tonbomythic3/tools/sdk_${VERSION}_containers/archive/models/training}"

# Docker container paths. Not designed to be overridden.
MODEL_ZOO_DIR="/root/mythic_sdk/v${VERSION}/mythic-model-zoo"
MODEL_ZOO_DATASETS_DIR="${MODEL_ZOO_DIR}/datasets"
TRAINING_MODELS_DIR="/root/mythic_sdk/v${VERSION}/models/training"

mkdir -p "${MYTHIC_TMPDIR}"
mkdir -p "${MYTHIC_WORKSPACE}"

docker run \
 --shm-size 128m \
 --rm -it \
 --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
 --mount type=bind,src="${MYTHIC_WORKSPACE}",dst="${MYTHIC_WORKSPACE}" \
 --mount type=bind,src="${MYTHIC_TMPDIR}",dst="${MYTHIC_TMPDIR}" \
 --mount type=bind,src="${DATASET_DIR}",dst="${MODEL_ZOO_DATASETS_DIR}" \
 --mount type=bind,src="${TRAINING_MODELS_HOST_DIR}",dst="${TRAINING_MODELS_DIR}" \
 -e MYTHIC_ROOT="${MYTHIC_TMPDIR}" \
 -e WANDB_MODE=offline \
 -e HF_HUB_OFFLINE=1 \
 -e HF_EVALUATE_OFFLINE=1 \
 gcr.io/mythic-devops/mythic-sdk-ubuntu-24.04:m2000-v${VERSION} /bin/bash
