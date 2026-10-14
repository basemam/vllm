#!/bin/bash
# Offline test for vllm_disagg.sh _wait_log_signal_or_fail: a worker that dies
# at boot (on any node) ends the readiness wait at once instead of at timeout.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRV="$DIR/../vllm_disagg.sh"
pass=0; fail=0
ok()  { echo "  PASS  $1"; pass=$((pass + 1)); }
bad() { echo "  FAIL  $1"; fail=$((fail + 1)); }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FN="$TMP/fn.sh"
sed -n '/^_BOOT_FATAL_SIGNALS=/,/^}/p' "$DRV" > "$FN"
sed -i "s#/run_logs/#${TMP}/run_logs/#g" "$FN"
grep -q '_wait_log_signal_or_fail()' "$FN" && ok "extracted the function" || { bad "extract"; exit 1; }

J=900001
L="$TMP/run_logs/$J"
reset() { rm -rf "$L"; mkdir -p "$L"; : > "$L/prefill_NODE0.log"; : > "$L/decode_NODE4.log"; }
call() {
    OUT="$(SLURM_JOB_ID=$J NODE_RANK=0 bash -c "source '$FN'; _wait_log_signal_or_fail '$L/decode_NODE4.log' 'decode master' 'Application startup complete.' $1 1" 2>&1)"
    RC=$?
}

echo "=== ready signal present -> returns 0 ==="
reset; echo "INFO:     Application startup complete." >> "$L/decode_NODE4.log"
call 30
[ "$RC" = "0" ] && grep -q '^Ready: decode master' <<<"$OUT" && ok "ready" || bad "rc=$RC out=$OUT"

echo "=== master segfault (459188 shape) -> exits 1 at once ==="
reset; printf '!!!!!!! Segfault encountered !!!!!!!\nRuntimeError: Engine core initialization failed. See root cause above.\n' >> "$L/decode_NODE4.log"
t0=$(date +%s); call 600; dt=$(( $(date +%s) - t0 ))
[ "$RC" = "1" ] && ok "rc=1" || bad "rc=$RC"
[ "$dt" -lt 5 ] && ok "returned in ${dt}s, not at timeout" || bad "took ${dt}s"
grep -q 'worker died during boot while waiting for decode master' <<<"$OUT" && ok "names the wait" || bad "message: $OUT"
grep -q 'decode_NODE4.log' <<<"$OUT" && ok "names the log" || bad "log not named"
grep -q 'worker died during boot' "$L/proxy_NODE0.log" && ok "written to proxy_NODE0.log" || bad "not in proxy log"

echo "=== child crash on another node -> also exits 1 ==="
reset; echo "(EngineCore_DP9 pid=1) ERROR EngineCore failed to start." > "$L/decode_NODE6.log"
call 600
[ "$RC" = "1" ] && grep -q 'decode_NODE6.log' <<<"$OUT" && ok "child log caught" || bad "rc=$RC out=$OUT"

echo "=== healthy but slow -> still times out normally ==="
reset; echo "loading weights" >> "$L/decode_NODE4.log"
call 2
[ "$RC" = "1" ] && grep -q '^Timeout (2s)' <<<"$OUT" && ok "timeout path unchanged" || bad "rc=$RC out=$OUT"

echo ""
echo "======================================================"
echo "  test_wait_fatal: ${pass} passed, ${fail} failed"
echo "======================================================"
[ "$fail" = "0" ]
