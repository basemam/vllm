#!/bin/bash
#SBATCH -A amd-rccl-guest
#SBATCH -p amd-arad-burst
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=04:00:00
#SBATCH --job-name=build-vllm-main
#SBATCH --exclusive
#SBATCH --exclude=useocpm2m-097-[008,015,019-020,025-026,028,033,039-043,045,049,069,071,077-080,082-084,089,094-095,100,114,115,121-123,125,135-137,139-140,142,144,154]
#SBATCH --output=/shared_inference/%u/model_blog_logs/slurm-%j.out
#SBATCH --error=/shared_inference/%u/model_blog_logs/slurm-%j.err
# ---------------------------------------------------------------------------
# Build + push mad/docker/vllm_main.dsv4.ubuntu.amd.Dockerfile on a compute
# node: vLLM (VLLM_REF) + MoRI + vllm-router from source.
#
#   cd mad/scripts/vllm_dissag
#   sbatch --qos=low build_vllm_main_image.slurm
#
# Pins are Dockerfile ARGs; override with e.g. VLLM_REF=<sha> sbatch ...
# ---------------------------------------------------------------------------
set -euo pipefail

VLLM_REF="${VLLM_REF:-fff6f12c4ec0c51f80220121ea83d1ce8de16c7e}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai-rocm:base-nightly-rocm72-83c078d955cd6e0c63d3b12abe669c8de936a370}"
MORI_REF="${MORI_REF:-v1.2.4}"
MAX_JOBS="${MAX_JOBS:-64}"
TAG="${TAG:-rocm/pytorch-private:vllm-main-${VLLM_REF:0:9}-mori-${MORI_REF}-router}"
PUSH="${PUSH:-1}"

ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" rev-parse --show-toplevel)"
DOCKERFILE="${ROOT}/mad/docker/vllm_main.dsv4.ubuntu.amd.Dockerfile"
test -f "${DOCKERFILE}"

echo "[build] host=$(hostname) date=$(date -u +%FT%TZ)"
echo "[build] tree=$(git -C "${ROOT}" rev-parse --short HEAD) $(git -C "${ROOT}" rev-parse --abbrev-ref HEAD)"
echo "[build] VLLM_REF=${VLLM_REF} MORI_REF=${MORI_REF} BASE_IMAGE=${BASE_IMAGE} MAX_JOBS=${MAX_JOBS}"
echo "[build] TAG=${TAG}"
df -h / /var/lib/docker 2>/dev/null || true

docker pull "${BASE_IMAGE}"
docker image inspect --format '[build] base digest {{index .RepoDigests 0}}' "${BASE_IMAGE}"

t0=$(date +%s)
DOCKER_BUILDKIT=1 docker build \
    --progress plain \
    --ulimit nofile=1048576:1048576 \
    -f "${DOCKERFILE}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg VLLM_REF="${VLLM_REF}" \
    --build-arg MORI_REF="${MORI_REF}" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    -t "${TAG}" \
    "${ROOT}/mad/docker"
echo "[build] built in $(( $(date +%s) - t0 ))s"

docker run --rm --entrypoint cat "${TAG}" /app/versions.txt

if [ "${PUSH}" = "1" ]; then
    docker push "${TAG}"
    docker image inspect --format '[build] pushed {{index .RepoDigests 0}}' "${TAG}"
fi
echo "[build] OK total $(( $(date +%s) - t0 ))s"
