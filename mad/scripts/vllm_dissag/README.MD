# DSV4 PD-disagg launcher (MoRIIO + MoRI EP)

Runs DeepSeek-V4-Flash-FP8 and DeepSeek-V4-Pro-FP8 as prefill/decode
disaggregated serving on SLURM: MoRIIO KV transfer between roles, `-tp 1` +
DP + MoRI expert parallelism inside each role. The image is stock upstream
vLLM. The scripts in `patches/` are applied at boot on every node, before
`vllm serve`, until they are baked into the image:

| patch | why |
|---|---|
| `apply_moriio_dsv4_attn_backend_fix.py` | MoRIIO's generic MLA backend lookup rejects DSV4's `fp8_ds_mla` cache; point it at `DeepseekV4ROCMAiterMLASparseBackend` |
| `apply_moriio_dsv4_mixed_block_size_fix.py` | DSV4 mixes 256-token MLA and 64-token SWA/indexer pages; MoRIIO's uniform `block_size` check would reject the register |
| `apply_kv_cache_promote_bounded_replay_fix.py` | with HMA off, KV planning rebuilds `SlidingWindowMLASpec` as `MLAAttentionSpec` and forwards the SWA-only `bounded_replay` field, which the target rejects |
| `apply_mori_trim_dispatch.py` | MoRI `dispatch()` returns a `world_size × max_num_tokens` receive buffer and the expert GEMMs run over all of it; trim it to this step's real token bound from `DPMetadata` (`MORI_TRIM_DISPATCH=0` in the container restores stock) |

A patch whose anchor no longer matches aborts the boot. Both models also run
with `--disable-hybrid-kv-cache-manager` (HMA off, `models.yaml`): MoRIIO's
HMA register path requires one `block_len` across all layers, which DSV4 does
not have. They also pass `--no-async-scheduling`, which main would otherwise
enable and which has hung PD decode before, and run the V1 model runner
(`VLLM_USE_V2_MODEL_RUNNER=0`): under Model Runner V2 a decode rank's first
real forward deadlocked the MoRI EP all2all against its peers' dummy batches.

## Image

`mad/docker/vllm_main.dsv4.ubuntu.amd.Dockerfile` builds, on top of
`vllm/vllm-openai-rocm:base-nightly-<sha>`:

- vLLM from source (`VLLM_REF`, default upstream main `bf6c80c7e`)
- MoRI from source (`MORI_REF=v1.2.3.post1`)
- vllm-router from source, installed at `/usr/local/bin/vllm-router`

Pins are recorded in `/app/versions.txt` (`DSV4_PATCHES=none`). Build on a
compute node, never on a login node or WSL:

```bash
cd mad/scripts/vllm_dissag
sbatch --qos=low build_vllm_main_image.slurm
# or by hand on an allocated node:
docker build --ulimit nofile=1048576:1048576 \
  -f mad/docker/vllm_main.dsv4.ubuntu.amd.Dockerfile \
  --build-arg BASE_IMAGE=vllm/vllm-openai-rocm:base-nightly-77154ca6116ae34cbfe314beef1016bde9b63989 \
  -t rocm/pytorch-private:vllm-main-bf6c80c7e-mori-v1.2.3.post1-router mad/docker
```

## Launching

Submit from a login node with `run_wideep_bench.sh BENCH MODEL TOPO`:

```bash
QOS=low LOG_WAIT_TIMEOUT_SECONDS=1800 ./run_wideep_bench.sh validate dsv4fls ep16 --time 04:00:00
QOS=low ./run_wideep_bench.sh smoke dsv4pro 4p4d --time 04:00:00 --dry-run
```

| arg | values |
|---|---|
| BENCH | `smoke` (concurrency sweep), `niah`, `mrcr`, `validate` (curl + NIAH + sweep), `accuracy` |
| MODEL | `dsv4fls`, `dsv4pro` |
| TOPO | `1p1d`/`ep8`, `2p2d`/`ep16`, `3p3d`/`ep24`, `4p4d`/`ep32`, asymmetric `1p2d`/`2p1d`/`2p3d`/`3p2d` |

Flags: `--image main|<tag>`, `--time`, `--exclude`, `--nodelist`, `--name`,
`--after JOBID`, `--dry-run`. The proxy is the vllm-router baked into the
image (`PROXY_TYPE=vllm_router`, the default).

Flash NIAH primes the answer with `1.`; Pro is never primed. Never pass
`--enforce-eager`; `DSV4_EAGER=1` runs decode with `cudagraph_mode: NONE`.

## Files

| file | role |
|---|---|
| `run_wideep_bench.sh` | submit wrapper: derives every knob from BENCH/MODEL/TOPO |
| `run_xPyD_models.slurm` | per-node `docker run` of `vllm_disagg.sh` |
| `vllm_disagg.sh` | in-container driver: roles, models.yaml, workers, proxy, bench |
| `connectors/moriio.sh`, `connectors/moriio.env` | MoRIIO serve argv, platform env, vllm-router start, curl |
| `parallelism.sh` | DP role helpers |
| `models.yaml` | per-model serve flags + env |
| `benchmark_*.sh`, `benchmark_*.py` | sweep / NIAH / MRCR / validate / accuracy |
| `niah_score.py`, `mrcr_lib.py`, `fetch_mrcr.py` | NIAH scoring, MRCR loader, login-side MRCR stager |
| `benchmark_parser.py`, `parse_to_csv.py` | result tables and madengine `perf.csv` |
| `socket_barrier.py`, `socket_wait.py`, `tcp_listen.sh` | cross-node rendezvous |
| `build_vllm_main_image.slurm` | image build + push on a compute node |
| `tests/run_all.sh` | offline suites (no cluster, no GPU) |
