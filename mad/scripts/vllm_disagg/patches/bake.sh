#!/bin/bash
# Bake the eight DSV4 source patches into an installed vLLM tree.
# Image-build only (MAD-private docker/vllm_disagg_inference.dsv4.v0290.ubuntu.amd.Dockerfile
# fetches this directory to /opt/dsv4_patches and runs this script).
# Order matches the old connectors/moriio.sh runtime apply list.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VLLM_DIR="${1:-}"
if [[ -z "${VLLM_DIR}" ]]; then
    VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
fi
if [[ -z "${VLLM_DIR}" || ! -d "${VLLM_DIR}" ]]; then
    echo "Error: [dsv4-bake] cannot locate vLLM install dir" >&2
    exit 1
fi

echo "[dsv4-bake] vllm=${VLLM_DIR}"

# Same order as the retired moriio.sh apply functions. attn_xfer must follow
# the transfer gate (it rewrites wait_for_save after the gate anchors land).
# Trim/attn-xfer stay env-gated inside the patched sources:
#   MORI_TRIM_DISPATCH=0     opt out of the ITL trim
#   DSV4_TRANSFER_ATTN=1     WRITE .attn (product HMA=0)
patches=(
    apply_mori_combine_original_topk_fix.py
    apply_mori_trim_dispatch.py
    apply_moriio_dsv4_attn_backend_fix.py
    apply_moriio_dsv4_skip_noncontiguous_register_fix.py
    apply_moriio_dsv4_mixed_block_size_fix.py
    apply_moriio_dsv4_transfer_gate_fix.py
    apply_moriio_dsv4_attn_transfer_fix.py
    apply_moriio_dsv4_rdma_wait_fix.py
)

for p in "${patches[@]}"; do
    echo "[dsv4-bake] ${p}"
    python3 "${HERE}/${p}" "${VLLM_DIR}"
done

_mori="${VLLM_DIR}/model_executor/layers/fused_moe/prepare_finalize/mori.py"
if ! grep -q '_original_topk_ids' "${_mori}"; then
    echo "Error: [dsv4-bake] combine original-topk did not land" >&2
    exit 1
fi
if ! grep -q '_mori_trim_rows' "${_mori}"; then
    echo "Error: [dsv4-bake] trim dispatch did not land" >&2
    exit 1
fi

mkdir -p /app
{
    echo "DSV4_PATCHES_BAKED=combine,trim,attn_backend,storage_span,mixed_bs,gate,attn_xfer,rdma_wait"
    echo "DSV4_PATCHES_DIR=/opt/dsv4_patches"
} >> /app/versions.txt

echo "[dsv4-bake] done"
