#!/usr/bin/env python3
"""Let MoRIIO register DSV4's mixed per-layer block sizes (runtime patch).

With the hybrid KV cache manager off, MoRIIOConnectorWorker.register_kv_caches
requires every layer to share the first layer's block_size. DSV4 does not:
sparse MLA `.attn` pages are 256 tokens, the SWA and indexer caches are 64.
The connector already keeps per-layer geometry (block_lens, transfer offsets),
so only the uniformity check is in the way; it is downgraded to a log line.
The num_blocks uniformity check, which the peer protocol does rely on, stays.

Idempotent. A missing anchor is a hard error.

Usage: apply_moriio_dsv4_mixed_block_size_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "DSV4-MIXED-BS"

OLD = """            elif geometry.block_size != self.block_size:
                raise ValueError(
                    "MoRIIO KV cache block size mismatch for layer "
                    f"{layer_name}: {geometry.block_size} != {self.block_size}"
                )
"""

NEW = """            elif geometry.block_size != self.block_size:
                # DSV4-MIXED-BS: DSV4 mixes 256-token MLA and 64-token SWA /
                # indexer pages; per-layer geometry already covers it.
                logger.info(
                    "[dsv4-mixed-bs] layer %s block_size=%d (global %d); "
                    "registering with per-layer geometry",
                    layer_name,
                    geometry.block_size,
                    self.block_size,
                )
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
