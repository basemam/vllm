#!/bin/bash
# Offline argv + env assertions for the DSV4 MoRIIO + MoRI-EP launcher: checks the
# emitted `vllm serve` flags/env and the submit-side forwards. No cluster / no GPUs.
# Exits 0 if all assertions hold.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM="$DIR/run_xPyD_models.slurm"
pass=0; fail=0

# emit argv for a cell
_argv() { # connector wide_ep ep_backend model model_path
  env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=1 yD=1 CONNECTOR="$1" WIDE_EP="$2" EP_BACKEND="$3" \
    MODEL_NAME="$4" MODEL_PATH="$5" MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f'
}

_has()   { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasnot(){ grep -qF -- "$2" <<<"$1" && { printf "  FAIL  %s (unexpected: %s)\n" "$3" "$2"; fail=$((fail+1)); } || { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); }; }
_count() { local n; n="$(grep -cF -- "$2" <<<"$1")"; [[ "$n" == "$3" ]] && { printf "  PASS  %s (=%s)\n" "$4" "$n"; pass=$((pass+1)); } || { printf "  FAIL  %s (got %s want %s)\n" "$4" "$n" "$3"; fail=$((fail+1)); }; }

echo "=== connector platform env files carry the RDMA-fix env ==="
# The ROCm-7.2.3 GPU-RDMA env now lives in per-connector .env files; the slurm
# sources connectors/<CONNECTOR>.env and forwards each var via docker -e.
S="$(cat "$SLURM")"
_has "$S" 'CONNECTOR_ENV_FILE="${SCRIPT_DIR}/connectors/${CONNECTOR}.env"' "slurm sources connector .env"
_has "$S" '${CONNECTOR_ENV_ARGS}' "slurm forwards CONNECTOR_ENV_ARGS in docker run"
for cf in moriio; do
  F="$DIR/connectors/${cf}.env"
  if [[ -f "$F" ]]; then
    E="$(cat "$F")"
    _has "$E" "PYTORCH_ALLOC_CONF=expandable_segments:False" "${cf}.env: PYTORCH_ALLOC_CONF"
    _has "$E" "PYTORCH_HIP_ALLOC_CONF=expandable_segments:False" "${cf}.env: PYTORCH_HIP_ALLOC_CONF"
    _has "$E" "HSA_ENABLE_IPC_MODE_LEGACY=0" "${cf}.env: IPC_MODE_LEGACY=0"
    _has "$E" "MORI_GPU_ARCHS=gfx942" "${cf}.env: MORI_GPU_ARCHS"
  else
    printf "  FAIL  connectors/%s.env missing\n" "$cf"; fail=$((fail+1))
  fi
done
# parse check: the slurm's KEY=${KEY:-VAL} loop yields correct -e args (+ override wins)
_parse() { # $1=connector ; reads its .env with same logic as the slurm
  local A="" l k v
  while IFS= read -r l; do
    [[ "$l" =~ ^[[:space:]]*# || -z "${l// }" ]] && continue
    k="${l%%=*}"; v="${l#*=}"; A+=" -e ${k}=${!k:-$v}"
  done < "$DIR/connectors/$1.env"
  printf '%s' "$A"
}
_has "$(_parse moriio)" "-e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False" "parse yields HIP_ALLOC -e arg"
_has "$(PYTORCH_HIP_ALLOC_CONF=expandable_segments:True _parse moriio)" "-e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True" "submit-time override wins"

# Per-shape warmup must stay opt-in: a default-on gate would change measured TPOT.
echo ""
echo "=== per-shape warmup is opt-in, not default-on ==="
B="$(cat "$DIR/benchmark_xPyD.sh")"
_has    "$B" '${SHAPE_WARMUP:-0}' "benchmark_xPyD.sh: warmup gate defaults OFF"
_hasnot "$B" '${SHAPE_WARMUP:-1}' "benchmark_xPyD.sh: gate is not default-on"
_has "$(cat "$SLURM")" '${SHAPE_WARMUP:+-e SHAPE_WARMUP=' "slurm forwards SHAPE_WARMUP override"
_has "$(cat "$SLURM")" '${USE_INDUCTOR_GRAPH_PARTITION:+-e USE_INDUCTOR_GRAPH_PARTITION=' "slurm forwards IGP override"

echo ""
echo "=== slurm forwards DSV4 MoRI-EP / NIAH knobs ==="
_has "$S" '${DSV4_EAGER:+-e DSV4_EAGER=' "slurm forwards DSV4_EAGER"

# The image is stock upstream vLLM: nothing is patched at build or at boot,
# and no submit-side knob pretends otherwise.
echo ""
echo "=== stock upstream image, no patches ==="
M="$(cat "$DIR/connectors/moriio.sh")"
W="$(cat "$DIR/run_wideep_bench.sh")"
_DF="$(cat "$DIR/../../docker/vllm_main.dsv4.ubuntu.amd.Dockerfile")"
_hasnot "$_DF" "bake.sh" "main Dockerfile does not bake patches"
_hasnot "$_DF" "PATCHES_REPO" "main Dockerfile fetches no patch scripts"
_hasnot "$_DF" "SKIP_RUNTIME_PATCH" "main Dockerfile sets no SKIP_RUNTIME_PATCH"
_has "$_DF" 'echo "DSV4_PATCHES=none"' "main Dockerfile stamps DSV4_PATCHES=none"
_has "$_DF" '! grep -q _mori_trim_rows' "main Dockerfile asserts mori.py is untrimmed"
for k in apply_ SKIP_RUNTIME_PATCH MORI_TRIM DSV4_ENABLE_HMA DSV4_TRANSFER_ATTN DSV4_STORAGE_UPSTREAM_SPAN ROUTER_BOOT_INSTALL; do
  _hasnot "$M" "$k" "moriio.sh has no $k"
  _hasnot "$W" "$k" "wrapper has no $k"
  _hasnot "$S" "$k" "slurm has no $k"
done
_YML="$(python3 - "$DIR/models.yaml" <<'PY'
import sys, yaml
y = yaml.safe_load(open(sys.argv[1])) or {}
print("models: " + ",".join(sorted(y)))
hits = [m for m, c in sorted(y.items())
        if isinstance(c, dict) and set(c.get("env") or {}) & {"MORI_TRIM_DISPATCH", "DSV4_ENABLE_HMA"}]
print("patch knobs: " + (", ".join(hits) if hits else "NONE"))
PY
)"
_has "$_YML" "models: DeepSeek-V4-Flash-FP8,DeepSeek-V4-Pro-FP8" "models.yaml holds only DSV4"
_has "$_YML" "patch knobs: NONE" "models.yaml pins no patch knobs"
_MAIN_PLAN="$(cd "$DIR" && ./run_wideep_bench.sh smoke dsv4fls 2p2d --dry-run 2>&1 || true)"
_has "$_MAIN_PLAN" "vllm-main-bf6c80c7e-mori-v1.2.3.post1-router" "default image is the main tag"
_hasnot "$(ls "$DIR"/apply_*.py 2>/dev/null || true)" "apply_" "cookbook root has no apply_*.py"

# The container-side init section. models.yaml is loaded INSIDE the container by
# vllm_disagg.sh with a bare `import yaml`, twice, and it resolves both the
# per-model env block and the per-role serve flags -- so PyYAML is a critical
# dependency of the serve path, not a reporting nicety. Images are a moving
# target, so the init step must keep covering it. pandas is the reporting half
# (benchmark_parser.py, the only ITL/TTFT/TPOT reporter).
echo ""
echo "=== container creation barrier is bounded (437656) ==="
B="$(cat "$DIR/socket_barrier.py")"
V="$(cat "$DIR/vllm_disagg.sh")"
_has "$B" '"--timeout"' "barrier accepts a deadline"
_has "$B" 'never opened their port' "barrier names the peers that never showed"
_has "$B" 'default=0' "barrier default stays wait-forever for the other callers"
_has "$B" 'failed to bind local barrier port' "barrier bind failure is fatal"
_has "$V" 'CONTAINER_BARRIER_TIMEOUT_SECONDS:-1800' "launcher bounds the barrier"
_has "$V" 'container creation barrier failed' "launcher aborts when the barrier times out"
_has "$V" 'tcp_listen_kill' "launcher clears the barrier port without ss"

echo ""
echo "=== container init ensures the python deps the serve path imports ==="
_has "$V" "Initialization — container-side boot setup" "vllm_disagg has an init section"
_has "$V" "_ensure_py_deps" "init defines the dep check"
_has "$V" "_init_container_env" "init has an umbrella hook for future steps"
_has "$V" 'PY_BOOT_DEPS-yaml:PyYAML pandas' "init default covers BOTH PyYAML and pandas"
_has "$V" "import os, yaml, shlex" "models.yaml is still parsed with a bare import yaml"
_has "$V" 'PyYAML absent' "a failed PyYAML install says the serve path is affected"
_has "$V" 'pandas absent is harmless' "a failed pandas install says reporting only"
# Nothing in init may abort: a compute node can be offline from PyPI, and no
# reporting library is worth losing a multi-hour serving cell.
_INIT="$(python3 - "$DIR/vllm_disagg.sh" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r'^_ensure_py_deps\(\) \{.*?^\}', s, re.S | re.M)
body = m.group(0) if m else ""
print("FOUND" if body else "MISSING",
      "ABORTS" if re.search(r'\b(exit 1|exit 2)\b', body) else "NONFATAL")
PY
)"
_has "$_INIT" "FOUND NONFATAL" "init dep step is non-fatal (no exit on failure)"

_has "$S" '${NIAH_HALT_ON_FAIL:+-e NIAH_HALT_ON_FAIL=' "slurm forwards NIAH_HALT_ON_FAIL"
_has "$S" '${NIAH_LIST_PRIME:+-e NIAH_LIST_PRIME=' "slurm forwards NIAH_LIST_PRIME"
_has "$S" '${MRCR_NEEDLES:+-e MRCR_NEEDLES=' "slurm forwards MRCR_NEEDLES"
_has "$S" 'mrcr)         BENCHMARK_SCRIPT_FILE="benchmark_mrcr.sh"' "slurm maps mrcr -> benchmark_mrcr.sh"
_has "$S" 'validate)     BENCHMARK_SCRIPT_FILE="benchmark_validate.sh"' "slurm maps validate -> benchmark_validate.sh"
_has "$S" 'accuracy)     BENCHMARK_SCRIPT_FILE="benchmark_accuracy.sh"' "slurm maps accuracy -> benchmark_accuracy.sh"
_has "$S" '-e ACC_THINKING=$ACC_THINKING' "slurm forwards ACC_THINKING"
_hasnot "$S" '[gpu-free]' "slurm has no pre-docker VRAM wait (boot matches the PR branch)"
_has "$S" '${NIAH_STOP:+-e NIAH_STOP=' "slurm forwards NIAH_STOP"
_has "$S" '${NIAH_LOGPROBS:+-e NIAH_LOGPROBS=' "slurm forwards NIAH_LOGPROBS"
_has "$S" '${NIAH_TERSE:+-e NIAH_TERSE=' "slurm forwards NIAH_TERSE"
_has "$S" '${NIAH_STOP_BLANK:+-e NIAH_STOP_BLANK=' "slurm forwards NIAH_STOP_BLANK"
_has "$S" '${NIAH_PAIR_SLEEP_S:+-e NIAH_PAIR_SLEEP_S=' "slurm forwards NIAH_PAIR_SLEEP_S"
_has "$S" '${NIAH_WARMUP_SCORE_SLEEP_S:+-e NIAH_WARMUP_SCORE_SLEEP_S=' "slurm forwards NIAH_WARMUP_SCORE_SLEEP_S"
_has "$S" '${CONTAINER_BARRIER_TIMEOUT_SECONDS:+-e CONTAINER_BARRIER_TIMEOUT_SECONDS=' "slurm forwards CONTAINER_BARRIER_TIMEOUT_SECONDS"
_has "$S" '${PROXY_ROUTE_DP:+-e PROXY_ROUTE_DP=' "slurm forwards PROXY_ROUTE_DP"
W="$(cat "$DIR/run_wideep_bench.sh")"
_has "$W" 'unset NIAH_LIST_PRIME' "wrapper mrcr unsets NIAH_LIST_PRIME"
_has "$W" 'NIAH_LIST_PRIME="1."' "wrapper Flash NIAH defaults list-prime 1."
_has "$W" 'Pro: never prime' "wrapper Pro NIAH forces the unprimed stem"
_has "$W" 'BENCHMARK_SCRIPT=mrcr' "wrapper mrcr sets BENCHMARK_SCRIPT"
M="$(cat "$DIR/mrcr_lib.py")"
_hasnot "$M" 'import pandas' "mrcr_lib does not import pandas"
_hasnot "$M" 'import tiktoken' "mrcr_lib does not import tiktoken"
_hasnot "$M" 'huggingface_hub' "mrcr_lib does not import huggingface_hub"
_hasnot "$M" 'read_parquet' "mrcr_lib does not read parquet at job time"
_has "$M" '.jsonl' "mrcr_lib stages/reads JSONL"
_has "$(cat "$DIR/fetch_mrcr.py")" 'huggingface_hub' "login stager is the huggingface_hub caller"
_has "$W" 'BENCHMARK_SCRIPT=validate' "wrapper validate sets BENCHMARK_SCRIPT"
_has "$W" 'BENCHMARK_SCRIPT=accuracy' "wrapper accuracy sets BENCHMARK_SCRIPT"
_has "$(cat "$DIR/benchmark_accuracy.sh")" '/ready' "accuracy waits on proxy /ready"

echo ""
echo "=== moriio + wideEP (DeepSeek-V4-Flash-FP8) ==="
C="$(_argv moriio 1 mori DeepSeek-V4-Flash-FP8 /m/DSV4F)"
_has    "$C" "--enable-expert-parallel" "DSV4 has --enable-expert-parallel"
_has    "$C" "--data-parallel-size" "DSV4 has --data-parallel-size"
_has    "$C" "deepseek_v4" "DSV4 tokenizer-mode deepseek_v4"
_has    "$C" "--block-size" "DSV4 has --block-size"
_has    "$C" "256" "DSV4 block-size 256 present"
_count  "$C" "--compilation-config" 1 "DSV4 exactly one --compilation-config"
_hasnot "$C" "--tensor-parallel-size" "DSV4 no --tensor-parallel-size (uses -tp 1)"

echo ""
echo "=== NIAH harness defaults (the wrapper always sets method/maxtok) ==="
NSH="$(cat "${DIR}/benchmark_niah.sh")"
NPY="$(cat "${DIR}/benchmark_niah.py")"
_has "$NPY" 'os.environ.get("NIAH_METHOD", "product")' "niah.py defaults to the model-agnostic product stem"
_has "$NSH" 'NIAH_MAXTOK="${NIAH_MAXTOK:-2048}"' "niah.sh keeps develop's 2048 answer budget"
_has "$NSH" '/v1/models' "niah.sh falls back to the /v1/models readiness probe"
_has "$NSH" 'WARN: neither /ready' "niah.sh readiness failure is non-fatal"
_has "$NSH" 'parse_to_csv.py' "niah.sh still emits madengine perf.csv rows"
_has "$NPY" 'no scored answer' "niah.py exits 1 when every request is NO-RESULT"
_has "$NSH" 'every request is NO-RESULT' "niah.sh documents harness NO-RESULT as niah_rc"
_has "$V" 'exit "${_BENCH_RC:-0}"' "launcher returns the benchmark script status"
_has "$(cat "$DIR/benchmark_xPyD.sh")" 'sweep_rc=$rc' "xPyD tracks cell failures past tee"
_TCP_CHECK="$(python3 - <<'PY'
line = "  46: 010310AC:9C4C 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 57245 1 ffff8803c23ec000 100 0 0 10 -1"
f = line.split()
print("inode="+f[9], "uid="+f[7], "ptr="+f[11])
PY
)"
_has "$_TCP_CHECK" "inode=57245" "awk \$10 on /proc/net/tcp is inode (not uid)"
_has "$_TCP_CHECK" "uid=0" "awk \$8 on /proc/net/tcp is uid"
_has "$(cat "$DIR/tcp_listen.sh")" 'print $10' "tcp_listen_inodes prints awk inode field \$10"
_has "$W" 'NIAH_METHOD="${NIAH_METHOD:-hybrid}"' "wrapper sets DSV4 method explicitly"
_has "$W" 'NIAH_MAXTOK="${NIAH_MAXTOK:-512}"' "wrapper sets DSV4 maxtok explicitly"

echo ""
echo "======================================================"
echo "  argv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
