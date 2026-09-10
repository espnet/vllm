#!/usr/bin/env bash
# Build the ESPnet audio-LM image for x86_64/amd64, ARM64/aarch64, or both.
#
# There is one Dockerfile, not one per architecture. The base image
# vllm/vllm-openai:v0.28.0 is a real multi-arch manifest list, verified against
# the Docker Hub registry API on 2026-09-03 (see the Dockerfile header for the
# digests and how they were checked), so the only per-architecture input is
# which base digest to pin. This script supplies that explicitly, which keeps
# the two architectures visible without duplicating a 130-line Dockerfile that
# would then drift.
#
# Pinning by digest rather than by tag means a rebuild months from now uses the
# same bytes, even if the v0.28.0 tag is ever re-pushed.
#
#   ./build.sh --arch amd64                  single-arch, digest-pinned, --load
#   ./build.sh --arch arm64                  same for aarch64
#   ./build.sh --arch both                   multi-platform, OCI layout output
#   ./build.sh --arch amd64 --dry-run        print the command, run nothing
#   ./build.sh --arch arm64 --tag my:tag
#
# Serving the three models this image supports (exact names — the served name
# must match what the client sends and what config.json declares):
#   bagpiper, opuslm, opuslm_dialogue
# See README.md next to this file for run commands.
#
# See README.md for recorded validation and PUBLISHING.md for CI setup.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

# Verified 2026-09-03 against registry-1.docker.io for vllm/vllm-openai:v0.28.0.
# Index: sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
BASE_REPO="vllm/vllm-openai"
BASE_DIGEST_INDEX="sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
BASE_DIGEST_AMD64="sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
BASE_DIGEST_ARM64="sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce"

ARCH=""
TAG=""
DRY_RUN=0
OCI_DEST=""

usage() {
    sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)     ARCH="${2:-}"; shift 2 ;;
        --tag)      TAG="${2:-}"; shift 2 ;;
        --oci-dest) OCI_DEST="${2:-}"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage 2 ;;
    esac
done

case "$ARCH" in
    amd64|x86_64)    ARCH=amd64 ;;
    arm64|aarch64)   ARCH=arm64 ;;
    both)            ARCH=both ;;
    "") echo "ERROR: --arch is required (amd64 | arm64 | both)" >&2; usage 2 ;;
    *)  echo "ERROR: unsupported --arch: $ARCH (want amd64 | arm64 | both)" >&2; exit 2 ;;
esac

[[ -f "$DOCKERFILE" ]] || { echo "ERROR: Dockerfile not found at $DOCKERFILE" >&2; exit 1; }

# The build context must be the repo root: the Dockerfile COPYs the fork source.
# Dockerfile.dockerignore next to the Dockerfile filters .git and build output,
# and BuildKit prefers it over the repo-root .dockerignore.
[[ -f "${REPO_ROOT}/pyproject.toml" ]] || {
    echo "ERROR: ${REPO_ROOT} does not look like the repo root" >&2; exit 1; }

run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}

if [[ "$ARCH" == both ]]; then
    # A multi-platform result cannot be --load'ed into the classic docker image
    # store; buildx needs --push or a filesystem output. Pushing is off the
    # table here, so write an OCI layout instead.
    TAG="${TAG:-espnet-vllm:v0.28.0-multiarch}"
    # Default outside the repo so a build never litters the tracked tree.
    OCI_DEST="${OCI_DEST:-${PWD}/espnet-vllm-oci}"
    echo "[build] arch      : linux/amd64,linux/arm64 (multi-platform)"
    echo "[build] base      : ${BASE_REPO}@${BASE_DIGEST_INDEX} (multi-arch index)"
    echo "[build] tag       : ${TAG}"
    echo "[build] output    : OCI layout at ${OCI_DEST}"
    echo "[build] note      : requires docker buildx with a container driver"
    echo "                    (docker buildx create --use) and QEMU for the"
    echo "                    non-native architecture."
    run docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --build-arg "BASE_IMAGE=${BASE_REPO}@${BASE_DIGEST_INDEX}" \
        -f "$DOCKERFILE" \
        -t "$TAG" \
        --output "type=oci,dest=${OCI_DEST}.tar" \
        "$REPO_ROOT"
else
    if [[ "$ARCH" == amd64 ]]; then
        DIGEST="$BASE_DIGEST_AMD64"; PLATFORM=linux/amd64
    else
        DIGEST="$BASE_DIGEST_ARM64"; PLATFORM=linux/arm64
    fi
    TAG="${TAG:-espnet-vllm:v0.28.0-${ARCH}}"
    echo "[build] arch      : ${PLATFORM}"
    echo "[build] base      : ${BASE_REPO}@${DIGEST}"
    echo "[build] tag       : ${TAG}"
    echo "[build] context   : ${REPO_ROOT}"
    if [[ "$ARCH" != "$(uname -m | sed -e s/x86_64/amd64/ -e s/aarch64/arm64/)" ]]; then
        echo "[build] note      : cross-architecture build; needs QEMU binfmt"
        echo "                    (docker run --privileged --rm tonistiigi/binfmt --install all)"
    fi
    run docker buildx build \
        --platform "$PLATFORM" \
        --build-arg "BASE_IMAGE=${BASE_REPO}@${DIGEST}" \
        -f "$DOCKERFILE" \
        -t "$TAG" \
        --load \
        "$REPO_ROOT"
fi
