#!/bin/bash
# Bake the four DSV4 patches into an installed vLLM tree at 83c078d.
# Image-build only (MAD-private docker/vllm_disagg_inference.dsv4.nightly83c078d.ubuntu.amd.Dockerfile
# fetches this directory to /opt/dsv4_patches and runs this script).
#
#   0001-swa-mla-drop-bounded-replay.patch   vllm-project/vllm#63411
#   0002-mori-dispatch-trim.patch            vllm-project/vllm#63412
#   apply_moriio_dsv4_attn_backend_fix.py    MoRIIO connector accepts fp8_ds_mla
#   apply_moriio_dsv4_mixed_block_size_fix.py MoRIIO connector, 256/64 page mix
#
# The two .patch files are the PR commits (vllm/ only, -p2 against the
# installed package). The trim is unconditional; there is no MORI_TRIM_DISPATCH.
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

diffs=(
    0001-swa-mla-drop-bounded-replay.patch
    0002-mori-dispatch-trim.patch
)
for d in "${diffs[@]}"; do
    if (cd "${VLLM_DIR}" && git apply -p2 -R --check "${HERE}/${d}" 2>/dev/null); then
        echo "[dsv4-bake] ${d}: already applied"
    else
        echo "[dsv4-bake] ${d}"
        (cd "${VLLM_DIR}" && git apply -p2 --verbose "${HERE}/${d}")
    fi
done

# apply_*.py names below are also what connectors/moriio.sh re-runs at boot.
patches=(
    apply_moriio_dsv4_attn_backend_fix.py
    apply_moriio_dsv4_mixed_block_size_fix.py
)
for p in "${patches[@]}"; do
    echo "[dsv4-bake] ${p}"
    python3 "${HERE}/${p}" "${VLLM_DIR}"
done

_need() {
    if ! grep -q -- "$2" "${VLLM_DIR}/$1"; then
        echo "Error: [dsv4-bake] $3 did not land ($1)" >&2
        exit 1
    fi
}
_conn=distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py
_need v1/core/kv_cache_planning.py '"bounded_replay")' bounded_replay
_need model_executor/layers/fused_moe/prepare_finalize/mori.py _num_live_dispatch_rows trim
_need "${_conn}" DSV4-ATTN-FIX attn_backend
_need "${_conn}" DSV4-MIXED-BS mixed_bs

mkdir -p /app
{
    echo "DSV4_PATCHES_BAKED=bounded_replay,trim,attn_backend,mixed_bs"
    echo "DSV4_PATCHES_DIR=/opt/dsv4_patches"
} >> /app/versions.txt

echo "[dsv4-bake] done"
