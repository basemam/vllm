# DSV4 vLLM patches (image bake only)

Branch `v0.29.0-mad` of [basemam/vllm](https://github.com/basemam/vllm), cut from
upstream `v0.29.0` (`98dff2a`). The vLLM source on this branch is unmodified;
these eight scripts rewrite the **installed** vLLM tree at docker build time.

The Dockerfile lives in MAD-private:
`docker/vllm_disagg_inference.dsv4.v0290.ubuntu.amd.Dockerfile`. It fetches this
directory (build args `PATCHES_REPO`, `PATCHES_REF`, `PATCHES_PATH`), copies it to
`/opt/dsv4_patches/` and runs `bake.sh`. Nothing applies them at job start;
MAD-private `scripts/vllm_dissag/connectors/moriio.sh` only checks that the image
carries them.

| Script | Why |
|--------|-----|
| `apply_mori_combine_original_topk_fix.py` | `combine()` uses original top-k (EP32 garbage otherwise) |
| `apply_mori_trim_dispatch.py` | Slice MoE dispatch to live rows (~308 → ~27 ms ITL). Runtime opt-out: `MORI_TRIM_DISPATCH=0` |
| `apply_moriio_dsv4_attn_backend_fix.py` | MoRIIO accepts `fp8_ds_mla` / `ROCM_FLASHMLA_SPARSE_DSV4` |
| `apply_moriio_dsv4_skip_noncontiguous_register_fix.py` | Storage span; no `.contiguous()` |
| `apply_moriio_dsv4_mixed_block_size_fix.py` | SWA page 64 + MLA page 256 |
| `apply_moriio_dsv4_transfer_gate_fix.py` | `wait_for_save` matches `block_size` |
| `apply_moriio_dsv4_attn_transfer_fix.py` | WRITE `.attn`. Runtime gate: `DSV4_TRANSFER_ATTN=1` |
| `apply_moriio_dsv4_rdma_wait_fix.py` | CQE wait logs |

`bake.sh` runs them in that order against `import vllm`. The scripts target
v0.29.0 only: on v0.30.0 and later `main`, the combine fix is upstream (#56073)
and several anchors have moved.

Offline test for the trim patch (no GPU):

```bash
python3 mad/scripts/vllm_disagg/tests/test_mori_trim.py
```
