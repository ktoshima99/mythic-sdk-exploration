#!/bin/bash
# Do not source this script otherwise it will exit in the archive directory

set -euo pipefail

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

# This can be overridden by setting the environment variable or at script invocation time.
# Example:
# ARCHIVE_DIR=/path/to/installation/archives ./load_and_tag_docker_images.sh
#
# ARCHIVE_DIR - the directory containing the unzipped contents of the Mythic Installer zip files.
ARCHIVE_DIR="${ARCHIVE_DIR:-/projects/tonbomythic3/tools/sdk_${VERSION}_containers/archive}"

cd "${ARCHIVE_DIR}"

echo "Loading compiler container ..."
./install_compiler.sh -n

echo "Loading sdk container ..."
./install_sdk_docker_image.sh -n
