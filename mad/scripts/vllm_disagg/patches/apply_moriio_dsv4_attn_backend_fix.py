#!/usr/bin/env python3
"""Give the MoRIIO connector DSV4's own attention backend (runtime patch).

MoRIIOConnectorWorker.__init__ resolves a backend only to read get_name() for
handshake metadata, via get_attn_backend(..., use_mla=True) with use_sparse
left False. On ROCm that tries ROCM_AITER_MLA / TRITON_MLA /
ROCM_AITER_TRITON_MLA, all of which reject kv_cache_dtype=fp8_ds_mla -- and
DSV4 remaps every fp8* cache dtype to fp8_ds_mla. So stock vLLM main cannot
boot DSV4 Flash or Pro under MoRIIO (job 475880):

    No valid attention backend found for rocm with
    AttentionSelectorConfig(head_size=512, kv_cache_dtype=fp8_ds_mla, ...)

The model itself runs DeepseekV4ROCMAiterMLASparseBackend, which accepts
fp8_ds_mla; the connector is pointed at the same class.

Idempotent. A missing anchor is a hard error, so a vLLM bump that moves the
call fails the boot loudly instead of reverting to the selector failure.

Usage: apply_moriio_dsv4_attn_backend_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "DSV4-ATTN-FIX"

OLD = """        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            use_mla=self.use_mla,
        )
"""

NEW = """        # DSV4-ATTN-FIX: the generic MLA selector rejects fp8_ds_mla.
        _arch = tuple(getattr(self.model_config, "architectures", None) or ())
        if str(self.cache_config.cache_dtype) == "fp8_ds_mla" or any(
            "DeepseekV4" in a for a in _arch
        ):
            from vllm.models.deepseek_v4.amd.rocm import (
                DeepseekV4ROCMAiterMLASparseBackend,
            )

            backend = DeepseekV4ROCMAiterMLASparseBackend
            logger.info(
                "[dsv4-attn] MoRIIO uses %s (cache_dtype=%s arch=%s)",
                backend.get_name(),
                self.cache_config.cache_dtype,
                _arch,
            )
        else:
            backend = get_attn_backend(
                self.model_config.get_head_size(),
                self.model_config.dtype,
                self.cache_config.cache_dtype,
                use_mla=self.use_mla,
            )
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-attn] ERROR: {path} not found.", file=sys.stderr)
        return 1

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-attn] already patched: {path}")
        return 0
    if src.count(OLD) != 1:
        print(
            f"[dsv4-attn] ERROR: expected 1 get_attn_backend() anchor in {path}, "
            f"found {src.count(OLD)}.",
            file=sys.stderr,
        )
        for i, line in enumerate(src.splitlines(), 1):
            if "get_attn_backend" in line:
                print(f"[dsv4-attn]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    tmp = path + ".dsv4tmp"
    with open(tmp, "w") as f:
        f.write(src.replace(OLD, NEW, 1))
    os.replace(tmp, path)
    print(f"[dsv4-attn] APPLIED: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
