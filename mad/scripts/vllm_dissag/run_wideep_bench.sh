#!/bin/bash
# DSV4 WideEP PD-disagg submit — derive every dependent knob from BENCH + MODEL + TOPO.
# =============================================================================
# Login node only. Do not docker build here.
#
#   ./run_wideep_bench.sh smoke dsv4fls 1p1d
#   ./run_wideep_bench.sh validate dsv4fls ep16
#   ./run_wideep_bench.sh validate dsv4pro ep32 --time 04:00:00
#   ./run_wideep_bench.sh niah dsv4fls 2p2d --dry-run
#   MRCR_PER_BIN=8 ./run_wideep_bench.sh mrcr dsv4fls 2p2d --time 04:00:00
#
# Changing EP16 → EP24/EP32 is TOPO only. Do not set xP/yD/-N/PROXY_ROUTE_DP by hand.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Upstream vLLM main + MoRI v1.2.3.post1 + vllm-router, unpatched; patches/ is
# applied at boot.
# Dockerfile: mad/docker/vllm_main.dsv4.ubuntu.amd.Dockerfile.
IMG_MAIN="rocm/pytorch-private:vllm-main-bf6c80c7e-mori-v1.2.3.post1-router"

# Live exclude (override with EXCLUDE_NODES= or --exclude). Empty + --nodelist
# means "only that pool".
DEFAULT_EXCLUDE='useocpm2m-097-[008,015,019-020,025-026,028,033,038,040-043,045,049,069,071,077-080,082-084,089,094-095,100,114,115,121-123,125,135-136,139-140,142,144,154]'

usage() {
    cat <<'EOF'
Usage: ./run_wideep_bench.sh BENCH MODEL TOPO [flags]

  BENCH   niah | smoke | mrcr | validate | accuracy
  MODEL   dsv4fls | dsv4pro | DeepSeek-V4-Flash-FP8 | DeepSeek-V4-Pro-FP8
  TOPO    1p1d | 2p2d | 3p3d | 4p4d | ep8 | ep16 | ep24 | ep32
          asymmetric: 1p2d | 2p1d | 2p3d | 3p2d  (driver derives per-role DP)

Flags:
  --dry-run              print sbatch, do not submit
  --image main|<tag>     image (default: main = vLLM main + MoRI + router)
  --exclude LIST         slurm --exclude (ignored if --nodelist is set)
  --nodelist LIST        pin to a node pool (no --exclude)
  --name NAME            slurm job name override
  --time HH:MM:SS        walltime override
  --after JOBID          sbatch --dependency=afterany:JOBID (queue behind a live job)

Env still wins for rare knobs (NIAH_SEEDS, PARTITION, DSV4_EAGER, ...).
  DSV4_EAGER=1  one-shot: decode cudagraph NONE +quant_fp8 (never --enforce-eager).
  HOLD_AFTER_READY=1  skip proxy/router and bench; keep P+D up for a manual proxy.
EOF
    exit 2
}

BENCH="${1:-}"
MODEL_IN="${2:-}"
TOPO_IN="${3:-}"
[[ -n "$BENCH" && -n "$MODEL_IN" && -n "$TOPO_IN" ]] || usage
shift 3

DRY_RUN=0
IMAGE_ARG=""
EXCLUDE_ARG="${EXCLUDE_NODES:-$DEFAULT_EXCLUDE}"
NODELIST_ARG="${NODELIST:-}"
JOB_NAME_ARG=""
TIME_ARG=""
AFTER_JOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --image) IMAGE_ARG="${2:-}"; shift 2 ;;
        --exclude) EXCLUDE_ARG="${2:-}"; shift 2 ;;
        --nodelist) NODELIST_ARG="${2:-}"; shift 2 ;;
        --name) JOB_NAME_ARG="${2:-}"; shift 2 ;;
        --time) TIME_ARG="${2:-}"; shift 2 ;;
        --after) AFTER_JOB="${2:-}"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Error: unknown flag '$1'" >&2; usage ;;
    esac
done

# --- MODEL ---
case "$MODEL_IN" in
    dsv4fls|ds4fls|dsv4|dsv4flash|DeepSeek-V4-Flash-FP8) MODEL_NAME="DeepSeek-V4-Flash-FP8"; SHORT=dsv4fls ;;
    dsv4pro|dsv4-pro|DeepSeek-V4-Pro-FP8) MODEL_NAME="DeepSeek-V4-Pro-FP8"; SHORT=dsv4pro ;;
    *) echo "Error: unknown MODEL '$MODEL_IN'" >&2; usage ;;
esac

# --- TOPO → xP yD N EP ---
# vllm_disagg.sh computes PREFILL_DP_SIZE=xP*8 and DECODE_DP_SIZE=yD*8
# independently, and the proxy routes on min(P,D) capped by PROXY_ROUTE_DP.
case "$TOPO_IN" in
    1p1d|ep8)  xP=1; yD=1; TOPO=1p1d ;;
    2p2d|ep16) xP=2; yD=2; TOPO=2p2d ;;
    3p3d|ep24) xP=3; yD=3; TOPO=3p3d ;;
    4p4d|ep32) xP=4; yD=4; TOPO=4p4d ;;
    2p3d)      xP=2; yD=3; TOPO=2p3d ;;
    3p2d)      xP=3; yD=2; TOPO=3p2d ;;
    1p2d)      xP=1; yD=2; TOPO=1p2d ;;
    2p1d)      xP=2; yD=1; TOPO=2p1d ;;
    *) echo "Error: unknown TOPO '$TOPO_IN' (2p2d/3p3d/4p4d, asymmetric 1p2d/2p1d/2p3d/3p2d, or ep8/ep16/ep24/ep32)" >&2; usage ;;
esac
N=$((xP + yD))
# Per-role EP, display only — the real DP world is derived in vllm_disagg.sh.
EP_P=$((xP * 8))
EP_D=$((yD * 8))
if [[ "$EP_P" == "$EP_D" ]]; then EP="$EP_P"; else EP="P${EP_P}/D${EP_D}"; fi

# Pro 4P/4D: con=1 is eight serial 1024-token decodes at ~3 s/tok and overruns the wall.
smoke_con_defaults() {
    if [[ "$SHORT" == "dsv4pro" && "$TOPO" == "4p4d" ]]; then
        BENCHMARK_CON="${BENCHMARK_CON:-8 16 32}"
        STEP_SEC_PER_TOK="${STEP_SEC_PER_TOK:-3}"
        STEP_TIMEOUT="${STEP_TIMEOUT:-7200}"
    else
        BENCHMARK_CON="${BENCHMARK_CON:-1 8 16 32}"
    fi
}

# --- BENCH ---
case "$BENCH" in
    niah)
        BENCHMARK_SCRIPT=niah
        NIAH_WORDS="${NIAH_WORDS:-2000,8000,16000,20000,28000,35000}"
        NIAH_SEEDS="${NIAH_SEEDS:-0}"
        NIAH_PAIR_SLEEP_S="${NIAH_PAIR_SLEEP_S:-30}"
        NIAH_WARMUP_SCORE_SLEEP_S="${NIAH_WARMUP_SCORE_SLEEP_S:-5}"
        BENCHMARK_COMBINATIONS=""
        ;;
    smoke)
        BENCHMARK_SCRIPT=sweep
        BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024}"
        NIAH_WORDS=""
        smoke_con_defaults
        ;;
    validate)
        # One allocation: curl (moriio.sh) then NIAH then smoke.
        BENCHMARK_SCRIPT=validate
        NIAH_WORDS="${NIAH_WORDS:-2000,8000,16000,20000,28000,35000}"
        NIAH_SEEDS="${NIAH_SEEDS:-0}"
        NIAH_PAIR_SLEEP_S="${NIAH_PAIR_SLEEP_S:-30}"
        NIAH_WARMUP_SCORE_SLEEP_S="${NIAH_WARMUP_SCORE_SLEEP_S:-5}"
        BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024}"
        smoke_con_defaults
        ;;
    mrcr)
        # OpenAI MRCR, 2-needle, bins that fit 40k. No prime.
        BENCHMARK_SCRIPT=mrcr
        MRCR_NEEDLES="${MRCR_NEEDLES:-2}"
        MRCR_BINS="${MRCR_BINS:-8192,16384,32768}"
        MRCR_PER_BIN="${MRCR_PER_BIN:-8}"
        MRCR_MAXTOK="${MRCR_MAXTOK:-1024}"
        MRCR_TIMEOUT="${MRCR_TIMEOUT:-1800}"
        MRCR_MAX_CTX="${MRCR_MAX_CTX:-40960}"
        MRCR_THINKING="${MRCR_THINKING:-chat}"
        MRCR_HALT_ON_FAIL="${MRCR_HALT_ON_FAIL:-0}"
        BENCHMARK_COMBINATIONS=""
        # A leftover NIAH_LIST_PRIME from a previous submit would zero MMR.
        unset NIAH_LIST_PRIME || true
        ;;
    accuracy)
        # 20-prompt correctness suite (benchmark_accuracy.py). curl still runs
        # first from moriio.sh; no NIAH, no sweep.
        BENCHMARK_SCRIPT=accuracy
        ACC_THINKING="${ACC_THINKING:-chat}"
        ACC_TIMEOUT="${ACC_TIMEOUT:-900}"
        NIAH_WORDS=""
        BENCHMARK_COMBINATIONS=""
        unset NIAH_LIST_PRIME || true
        ;;
    *) echo "Error: unknown BENCH '$BENCH' (niah|smoke|mrcr|validate|accuracy)" >&2; usage ;;
esac

# Walltime. Asking for more than the assoc MaxWall does NOT fail at submit —
# slurm queues the job forever as AssocMaxWallDurationPerJobLimit.
if [[ "${QOS:-}" == "low" ]]; then
    ASSOC_MAX_WALL="${ASSOC_MAX_WALL:-04:00:00}"
else
    ASSOC_MAX_WALL="${ASSOC_MAX_WALL:-08:00:00}"
fi
if [[ -z "$TIME_ARG" ]]; then
    case "$BENCH" in
        mrcr|accuracy) TIME_ARG=04:00:00 ;;
        *)             TIME_ARG=08:00:00 ;;
    esac
    if [[ "$TIME_ARG" > "$ASSOC_MAX_WALL" ]]; then
        echo "note: default TIME=$TIME_ARG exceeds assoc MaxWall; clamping to $ASSOC_MAX_WALL" >&2
        TIME_ARG="$ASSOC_MAX_WALL"
    fi
elif [[ "$TIME_ARG" > "$ASSOC_MAX_WALL" ]]; then
    echo "WARN: --time $TIME_ARG exceeds assoc MaxWall $ASSOC_MAX_WALL; slurm will hold this job as AssocMaxWallDurationPerJobLimit" >&2
fi

# Rank pin: head local DP is always 8. Without this, ranks 8+ (kv=null) get
# routed while children are headless. MORIIO_CHILD_HTTP=1 gives each child a
# real connector — leave PROXY_ROUTE_DP=0 so ranks 8+ POST to that pod.
if [[ "${MORIIO_CHILD_HTTP:-0}" == "1" ]]; then
    PROXY_ROUTE_DP="${PROXY_ROUTE_DP:-0}"
    PROXY_HANDSHAKE_PER_POD="${PROXY_HANDSHAKE_PER_POD:-1}"
else
    PROXY_ROUTE_DP="${PROXY_ROUTE_DP:-8}"
fi
PROXY_MAX_CONCURRENCY="${PROXY_MAX_CONCURRENCY:-512}"
# vllm_router KV-notify wants --moriio-dp-size; the toy proxy has no such flag.
if [[ "${PROXY_TYPE:-moriio_toy}" == "vllm_router" ]]; then
    ROUTER_SKIP_MORIIO_DP_SIZE="${ROUTER_SKIP_MORIIO_DP_SIZE:-0}"
else
    ROUTER_SKIP_MORIIO_DP_SIZE="${ROUTER_SKIP_MORIIO_DP_SIZE:-1}"
fi
# 36367 is in ip_local_port_range; a DP worker can steal it (218629).
MORI_PROXY_PING_PORT="${MORI_PROXY_PING_PORT:-36367}"

# --- IMAGE ---
case "${IMAGE_ARG}" in
    "")             DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-$IMG_MAIN}"; IMG_TAG=main ;;
    main|bf6c80c7e) DOCKER_IMAGE_NAME="$IMG_MAIN"; IMG_TAG=main ;;
    *)              DOCKER_IMAGE_NAME="$IMAGE_ARG"; IMG_TAG=custom ;;
esac
[[ "$DOCKER_IMAGE_NAME" != "$IMG_MAIN" ]] && IMG_TAG=custom

JOB_NAME="${JOB_NAME_ARG:-${SHORT}-${IMG_TAG}-${BENCH}-${TOPO}}"
# amd-rccl + amd-rccl-guest is rejected (Invalid account or account/partition).
PARTITION="${PARTITION:-amd-arad-burst}"
ACCOUNT="${ACCOUNT:-amd-rccl-guest}"
QOS="${QOS:-}"

# --kv-cache-dtype fp8 remaps to fp8_ds_mla, which ROCm MLA backends reject;
# fp8_e4m3 is the ROCm recipe. AITER_MLA must stay 1 (V4 sparse needs it).
EXTRA_ENV=(KV_CACHE_DTYPE=fp8_e4m3 VLLM_ROCM_USE_AITER_MLA=1)
# Pro weights are 125 GiB/GPU; a 32 GiB heap leaves no room for the profile dummy.
if [[ "$SHORT" == "dsv4pro" ]]; then
    EXTRA_ENV+=(MORI_SHMEM_HEAP_SIZE=17179869184)
else
    EXTRA_ENV+=(MORI_SHMEM_HEAP_SIZE=34359738368)
fi
# DSV4_EAGER=1: decode cudagraph NONE with +quant_fp8. NEVER --enforce-eager
# (AITER aiter_tensor_t crash). Submit-time wins yaml.
EXTRA_ENV+=(DSV4_EAGER="${DSV4_EAGER:-0}")
if [[ "${DSV4_EAGER:-0}" == "1" ]]; then
    EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE=NONE)
elif [[ -n "${DECODE_CUDAGRAPH_MODE:-}" ]]; then
    EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE="${DECODE_CUDAGRAPH_MODE}")
fi
EXTRA_ENV+=(VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}")
EXTRA_ENV+=(PROXY_LOG_LEVEL="${PROXY_LOG_LEVEL:-DEBUG}")
EXTRA_ENV+=(CURL_SUITE=short)
EXTRA_ENV+=(MORI_PROXY_PING_PORT="$MORI_PROXY_PING_PORT")

if [[ "$BENCH" == "niah" || "$BENCH" == "validate" ]]; then
    # hybrid = system+user through the DeepSeek template plus a restatement
    # after the haystack. product = bare stem (64 tokens). MAXTOK 512 is a
    # precision cap: a bigger budget buys recall and fabrication together.
    NIAH_METHOD="${NIAH_METHOD:-hybrid}"
    if [[ "$NIAH_METHOD" == "product" ]]; then
        NIAH_MAXTOK="${NIAH_MAXTOK:-64}"
    else
        NIAH_MAXTOK="${NIAH_MAXTOK:-512}"
    fi
    NIAH_WARMUP=0
    NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}"
    # validate must finish the ladder so smoke still runs.
    if [[ "$BENCH" == "validate" ]]; then
        NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-0}"
    else
        NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-1}"
    fi
    if [[ "$SHORT" == "dsv4fls" ]]; then
        # Flash: prime the stem with "1." so token 1 is given, not sampled.
        # Unprimed Flash answers `0`. NIAH_LIST_PRIME=off restores it.
        case "${NIAH_LIST_PRIME:-}" in
            off|OFF|0) NIAH_LIST_PRIME="" ;;
            "") NIAH_LIST_PRIME="1." ;;
        esac
    else
        # Pro: never prime. Force empty so a leftover Flash `1.` cannot leak in.
        NIAH_LIST_PRIME=""
    fi
    EXTRA_ENV+=(NIAH_METHOD="$NIAH_METHOD")
    EXTRA_ENV+=(NIAH_DS_WRAP=0)
    EXTRA_ENV+=(NIAH_TERSE="${NIAH_TERSE:-0}")
    EXTRA_ENV+=(NIAH_STOP_BLANK="${NIAH_STOP_BLANK:-0}")
    EXTRA_ENV+=(NIAH_MIN_TOKENS="${NIAH_MIN_TOKENS:-0}")
    EXTRA_ENV+=(NIAH_LIST_PRIME="${NIAH_LIST_PRIME}")
    EXTRA_ENV+=(NIAH_STOP="${NIAH_STOP:-}")
    EXTRA_ENV+=(NIAH_LOGPROBS="${NIAH_LOGPROBS:-0}")
fi

# HOLD_AFTER_READY=1: skip proxy/router + bench; keep P+D up for a manual proxy.
if [[ "${HOLD_AFTER_READY:-0}" == "1" ]]; then
    EXTRA_ENV+=(HOLD_AFTER_READY=1)
fi

if [[ "$BENCH" == "mrcr" ]]; then
    EXTRA_ENV+=(MRCR_NEEDLES="$MRCR_NEEDLES" MRCR_BINS="$MRCR_BINS" MRCR_PER_BIN="$MRCR_PER_BIN")
    EXTRA_ENV+=(MRCR_MAXTOK="$MRCR_MAXTOK" MRCR_TIMEOUT="$MRCR_TIMEOUT" MRCR_MAX_CTX="$MRCR_MAX_CTX")
    EXTRA_ENV+=(MRCR_THINKING=chat MRCR_HALT_ON_FAIL="$MRCR_HALT_ON_FAIL")
    [[ -n "${MRCR_DATA_DIR:-}" ]] && EXTRA_ENV+=(MRCR_DATA_DIR="$MRCR_DATA_DIR")
fi
if [[ "$BENCH" == "accuracy" ]]; then
    EXTRA_ENV+=(ACC_THINKING="$ACC_THINKING" ACC_TIMEOUT="$ACC_TIMEOUT")
    [[ -n "${ACC_MAXTOK:-}" ]] && EXTRA_ENV+=(ACC_MAXTOK="$ACC_MAXTOK")
fi

# EP16 NFS+JIT boot can exceed an hour. A hung boot under a 4h wall should
# pass LOG_WAIT_TIMEOUT_SECONDS=1800 to give up early.
LOG_WAIT_TIMEOUT_SECONDS="${LOG_WAIT_TIMEOUT_SECONDS:-10800}"
EXTRA_ENV+=(LOG_WAIT_TIMEOUT_SECONDS="$LOG_WAIT_TIMEOUT_SECONDS")

# --nodelist replaces --exclude so the pool is not filtered out.
SBATCH_LOC=()
if [[ -n "$NODELIST_ARG" ]]; then
    SBATCH_LOC+=(--nodelist="$NODELIST_ARG")
elif [[ -n "$EXCLUDE_ARG" ]]; then
    SBATCH_LOC+=(--exclude="$EXCLUDE_ARG")
fi

if [[ ! -f run_xPyD_models.slurm ]]; then
    echo "Error: run_xPyD_models.slurm missing in $SCRIPT_DIR" >&2
    exit 1
fi

unset SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_JOB_NODELIST || true

echo "=== WideEP bench plan ==="
echo "BENCH=$BENCH  MODEL=$MODEL_NAME  TOPO=$TOPO  EP=$EP  xP=$xP yD=$yD  N=$N"
echo "IMAGE=$DOCKER_IMAGE_NAME"
echo "PROXY_TYPE=${PROXY_TYPE:-moriio_toy}  PROXY_ROUTE_DP=$PROXY_ROUTE_DP  SKIP_MORIIO_DP=${ROUTER_SKIP_MORIIO_DP_SIZE}  PING=${MORI_PROXY_PING_PORT}  CONC=${PROXY_MAX_CONCURRENCY}  WIDE_EP=1"
echo "TIME=$TIME_ARG"
[[ "$BENCH" == "smoke" || "$BENCH" == "validate" ]] && echo "SMOKE CON=${BENCHMARK_CON:-default}  COMBOS=$BENCHMARK_COMBINATIONS  STEP_SEC_PER_TOK=${STEP_SEC_PER_TOK:-}  STEP_TIMEOUT=${STEP_TIMEOUT:-}"
[[ "$BENCH" == "niah" || "$BENCH" == "validate" ]] && echo "NIAH_METHOD=${NIAH_METHOD:-}  NIAH_WORDS=$NIAH_WORDS  NIAH_SEEDS=$NIAH_SEEDS  HALT=$NIAH_HALT_ON_FAIL  MAXTOK=${NIAH_MAXTOK:-}  WARMUP=${NIAH_WARMUP:-}  TIMEOUT=${NIAH_TIMEOUT:-}  WRAP=0  TERSE=${NIAH_TERSE:-0}  STOP_BLANK=${NIAH_STOP_BLANK:-0}  MINTOK=${NIAH_MIN_TOKENS:-0}  PRIME=${NIAH_LIST_PRIME:-}  STOP=${NIAH_STOP:-}  LOGPROBS=${NIAH_LOGPROBS:-0}"
[[ "$BENCH" == "mrcr" ]] && echo "MRCR needles=${MRCR_NEEDLES}  bins=${MRCR_BINS}  per_bin=${MRCR_PER_BIN}  MAXTOK=${MRCR_MAXTOK}  TIMEOUT=${MRCR_TIMEOUT}  MAX_CTX=${MRCR_MAX_CTX}  THINKING=chat  PRIME=OFF  DATA_DIR=${MRCR_DATA_DIR:-/shared_inference/bbarakat/datasets/openai_mrcr}"
[[ "$BENCH" == "accuracy" ]] && echo "ACCURACY prompts=20  THINKING=${ACC_THINKING}  MAXTOK=${ACC_MAXTOK:-auto}  TIMEOUT=${ACC_TIMEOUT}"
echo "EXTRA ${EXTRA_ENV[*]}"
echo "JOB_NAME=$JOB_NAME  ACCOUNT=$ACCOUNT  PARTITION=$PARTITION  QOS=${QOS:-default}  ${SBATCH_LOC[*]:-any-node}"
[[ -n "$AFTER_JOB" ]] && echo "DEPENDENCY=afterany:${AFTER_JOB}"
echo "========================="

SBATCH_EXTRA=()
if [[ -n "$AFTER_JOB" ]]; then
    SBATCH_EXTRA+=(--dependency="afterany:${AFTER_JOB}")
fi

SBATCH_CMD=(sbatch -A "$ACCOUNT")
if [[ -n "$QOS" ]]; then
    SBATCH_CMD+=(--qos="$QOS")
fi
SBATCH_CMD+=(-p "$PARTITION" -N "$N" -n "$N" --ntasks-per-node=1 --gres=gpu:8 --time="$TIME_ARG"
    "${SBATCH_LOC[@]}" "${SBATCH_EXTRA[@]}"
    --job-name="$JOB_NAME" --export=ALL
    run_xPyD_models.slurm)
echo "SBATCH: ${SBATCH_CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] not submitting"
    exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: SLURM_JOB_ID is set (${SLURM_JOB_ID}). Submit from a login node, not inside a job." >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch not found. Run this on a login node, not WSL." >&2
    exit 1
fi

export DOCKER_IMAGE_NAME PYTHONUNBUFFERED=1
export PROXY_TYPE="${PROXY_TYPE:-moriio_toy}"
export CONNECTOR=moriio WIDE_EP=1
export xP yD MODEL_NAME BENCHMARK_SCRIPT
export PROXY_ROUTE_DP PROXY_MAX_CONCURRENCY
export MORIIO_CHILD_HTTP="${MORIIO_CHILD_HTTP:-0}"
export PROXY_HANDSHAKE_PER_POD="${PROXY_HANDSHAKE_PER_POD:-}"
export ROUTER_SKIP_MORIIO_DP_SIZE MORI_PROXY_PING_PORT
if [[ "$BENCH" == "smoke" || "$BENCH" == "validate" ]]; then
    export BENCHMARK_COMBINATIONS BENCHMARK_CON
    [[ -n "${STEP_SEC_PER_TOK:-}" ]] && export STEP_SEC_PER_TOK
    [[ -n "${STEP_TIMEOUT:-}" ]] && export STEP_TIMEOUT
fi
if [[ "$BENCH" == "niah" || "$BENCH" == "validate" ]]; then
    export NIAH_WORDS NIAH_SEEDS NIAH_HALT_ON_FAIL
    export NIAH_PAIR_SLEEP_S NIAH_WARMUP_SCORE_SLEEP_S
    export NIAH_METHOD NIAH_MAXTOK NIAH_WARMUP NIAH_TIMEOUT
fi
if [[ "$BENCH" == "mrcr" ]]; then
    export MRCR_NEEDLES MRCR_BINS MRCR_PER_BIN MRCR_MAXTOK MRCR_TIMEOUT
    export MRCR_MAX_CTX MRCR_THINKING MRCR_HALT_ON_FAIL
fi
export "${EXTRA_ENV[@]}"

"${SBATCH_CMD[@]}"
