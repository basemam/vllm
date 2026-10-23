#!/usr/bin/env python3
"""Let MoRIIO register DSV4's mixed per-layer block sizes (runtime patch).

MoRIIOConnectorWorker.register_kv_caches allows per-layer block_size only when
the hybrid KV cache manager is on (#58585) and raises otherwise. DSV4 runs
with it off and still mixes page sizes: sparse MLA `.attn` pages are 256
tokens, the SWA and indexer caches are 64. The connector already keeps
per-layer geometry (block_lens, transfer offsets), so the HMA-off raise is
dropped and the existing per-layer log line runs for both cases. The
num_blocks uniformity check, which the peer protocol does rely on, stays.

Idempotent. A missing anchor is a hard error.

Usage: apply_moriio_dsv4_mixed_block_size_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "DSV4-MIXED-BS"

OLD = """                if not hma_enabled:
                    raise ValueError(
                        "MoRIIO KV cache block size mismatch for layer "
                        f"{layer_name}: {geometry.block_size} != "
                        f"{self.block_size}"
                    )
"""

NEW = """                # DSV4-MIXED-BS: DSV4 mixes 256-token MLA and 64-token SWA /
                # indexer pages with HMA off; per-layer geometry covers it.
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-mixed-bs] ERROR: {path} not found.", file=sys.stderr)
        return 1

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-mixed-bs] already patched: {path}")
        return 0
    if src.count(OLD) != 1:
        print(
            f"[dsv4-mixed-bs] ERROR: expected 1 block_size-mismatch anchor in "
            f"{path}, found {src.count(OLD)}.",
            file=sys.stderr,
        )
        for i, line in enumerate(src.splitlines(), 1):
            if "block size mismatch" in line or "geometry.block_size" in line:
                print(f"[dsv4-mixed-bs]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    tmp = path + ".dsv4tmp"
    with open(tmp, "w") as f:
        f.write(src.replace(OLD, NEW, 1))
    os.replace(tmp, path)
    print(f"[dsv4-mixed-bs] APPLIED: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
