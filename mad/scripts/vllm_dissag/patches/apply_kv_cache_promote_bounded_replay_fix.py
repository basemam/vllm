#!/usr/bin/env python3
"""Drop SWA-only `bounded_replay` when promoting SWA specs (runtime patch).

With the hybrid KV cache manager off, _promote_local_kv_cache_specs rebuilds
every sliding-window spec as its full-attention counterpart via replace_as(),
which copies each field of the source spec. SlidingWindowMLASpec carries
`bounded_replay`, MLAAttentionSpec does not, so DSV4 dies at KV planning
(job 476800):

    TypeError: MLAAttentionSpec.__init__() got an unexpected keyword
    argument 'bounded_replay'

A promoted spec keeps every block, so window replay bounds are moot, the same
reason `extra_retained_tokens` is already dropped there.

Idempotent. A missing anchor is a hard error.

Usage: apply_kv_cache_promote_bounded_replay_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "v1/core/kv_cache_planning.py"
MARKER = '"bounded_replay")'

OLD = """                drop=("extra_retained_tokens",),
"""

NEW = """                drop=("extra_retained_tokens", "bounded_replay"),
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[kv-promote] ERROR: {path} not found.", file=sys.stderr)
        return 1

    src = open(path).read()
    if MARKER in src:
        print(f"[kv-promote] already patched: {path}")
        return 0
    if src.count(OLD) != 1:
        print(
            f"[kv-promote] ERROR: expected 1 promotion drop= anchor in {path}, "
            f"found {src.count(OLD)}.",
            file=sys.stderr,
        )
        for i, line in enumerate(src.splitlines(), 1):
            if "replace_as(" in line or "drop=" in line:
                print(f"[kv-promote]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    tmp = path + ".dsv4tmp"
    with open(tmp, "w") as f:
        f.write(src.replace(OLD, NEW, 1))
    os.replace(tmp, path)
    print(f"[kv-promote] APPLIED: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
