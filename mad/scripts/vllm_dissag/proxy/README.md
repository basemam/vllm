# Python PD proxy (`moriio_pd_proxy.py`)

Default proxy for the DSV4 launcher (`PROXY_TYPE=moriio_toy` or `moriio_pd`).
`PROXY_TYPE=vllm_router` uses the router baked into the image instead.

Dependencies (Quart, Hypercorn, aiohttp, pyzmq, msgpack) are in the image;
outside it, `pip install -r requirements.txt`. `check_deps()` fails fast with
that line if anything is missing.

## Launch (from `connector_start_proxy`)

- `--moriio-dp-size` = prefill DP (`xP×8`), `--decode-dp-size` = decode DP
  (`yD×8`). Round-robin world is `min(P, D)`.
- `--dp-size-local 8`, `--expect-prefill 1 --expect-decode 1`, one
  `--prefill`/`--decode` per HTTP master.
- HTTP `:10001`, ZMQ discovery `:36367`. `--max-concurrency`
  (`PROXY_MAX_CONCURRENCY`, default 512); extra POSTs wait on the listen backlog.
- `PYTHONUNBUFFERED=1`, or the `tee`'d log looks frozen.

## Topology

WideEP children run `--headless`: they never bind `:20005` and never
ZMQ-register. There is one HTTP frontend per role (the prefill and decode
masters); handshake `:8405` and notify stay on that master.

`remote_hosts` repeats the master IP once per pod slot so `rank // dp_local`
still indexes the list, but WRITE never dials a headless `:8405`. Ranks 8+ are
selected with `X-data-parallel-rank` on the master API, which DP-RPCs to the
child. `PROXY_ROUTE_DP` (default 8) pins routing to the master's local ranks.

Never put `--kv-transfer-config` on headless children. To route ranks 8+
directly, set `MORIIO_CHILD_HTTP=1` (each child is a full HTTP + MoRIIO server)
with `PROXY_ROUTE_DP=0`.

`/ready` is 200 once each HTTP master has ZMQ-registered. Engine readiness
(weights + graph capture) is gated separately in `connector_wait_workers_ready`.

## Requests

`min_tokens` is clamped per leg: prefill runs with `max_tokens=1`, so its
`min_tokens` is clamped to 1; decode clamps to its own cap. Bodies without
`min_tokens` are untouched.

Tests: `python3 test_moriio_pd_proxy.py` (stdlib unittest).
