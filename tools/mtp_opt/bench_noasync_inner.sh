#!/usr/bin/env bash
# =============================================================================
# No-Async-Scheduling Impact Benchmark — Qwen3.5-35B-A3B MTP1
# TP=2, ISL=34K, OSL=300, BS=1, BF16
#
# Tests whether --no-async-scheduling makes O4+O5 CPU optimizations more
# impactful (hypothesis: async scheduling hides CPU savings under GPU compute;
# customer uses --no-async-scheduling so CPU is serial).
#
# Configs tested (2x2 matrix):
#   1. mtp1_base_async       — MTP1 vanilla, async ON  (default)
#   2. mtp1_base_noasync     — MTP1 vanilla, async OFF
#   3. mtp1_o4o5_async       — MTP1 + O4+O5, async ON
#   4. mtp1_o4o5_noasync     — MTP1 + O4+O5, async OFF
# =============================================================================
set -uo pipefail

RESULTS_BASE="${RESULTS_DIR:-/workspace/results}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HOME HOME="/workspace"
export XDG_CACHE_HOME="/workspace/.cache"
export FLASHINFER_WORKSPACE_DIR="/workspace/.cache/flashinfer"
export VLLM_CACHE_DIR="/workspace/.cache/vllm"
export TRITON_CACHE_DIR="/workspace/.cache/triton"
export TORCH_HOME="/workspace/.cache/torch"
mkdir -p "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_DIR}" "${VLLM_CACHE_DIR}" \
         "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${HF_HOME}" "${RESULTS_BASE}"

ISL=34000
OSL=300
BS=1
MAX_MODEL_LEN=40960
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
SERVE_PROMPTS="${SERVE_PROMPTS:-15}"
SERVE_WARMUP=5
TIMEOUT="${TIMEOUT:-1800}"
MODEL="Qwen/Qwen3.5-35B-A3B"

LOG="${RESULTS_BASE}/master.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " No-Async-Scheduling Impact Bench — Qwen3.5-35B-A3B"
echo " TP=2  ISL=${ISL}  OSL=${OSL}  BS=${BS}  BF16"
echo " Start: $(date)"
echo "============================================================"

# ── Phase 0: Install vLLM ────────────────────────────────────────────────────
echo ""
echo ">>> [Phase 0] Install vLLM"
rm -rf /workspace/.local 2>/dev/null || true
pip install --quiet --upgrade vllm 2>&1 | tail -5
pip uninstall flash-attn -y 2>/dev/null || true
export PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):${PATH}"

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo "  vLLM: ${VLLM_VER}  PyTorch: ${TORCH_VER}  CUDA: ${CUDA_VER}"

cat > "${RESULTS_BASE}/env.txt" <<SW
vllm=${VLLM_VER}
pytorch=${TORCH_VER}
cuda=${CUDA_VER}
isl=${ISL} osl=${OSL} bs=${BS} tp=2
workload=noasync_bench
timestamp=$(date -Iseconds)
SW

nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
    --format=csv,noheader 2>/dev/null | tee "${RESULTS_BASE}/gpu_info.txt"

# ── NUMA GPU pair ────────────────────────────────────────────────────────────
echo ""
echo ">>> Detecting NUMA-optimal GPU pair"
NUMA_GPUS=$(python3 - <<'PYEOF'
import subprocess, collections
result = subprocess.check_output(
    ["nvidia-smi","--query-gpu=index,gpu_bus_id","--format=csv,noheader"], text=True)
gpu_numa = {}
for line in result.strip().split("\n"):
    p = line.strip().split(", ")
    if len(p) != 2: continue
    idx, bus = int(p[0]), p[1].lower()
    try: numa = int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read().strip())
    except: numa = -1
    gpu_numa[idx] = numa
groups = collections.defaultdict(list)
for g, n in gpu_numa.items(): groups[n].append(g)
pair = None
for n in sorted(groups):
    if len(groups[n]) >= 2: pair = sorted(groups[n])[:2]; break
if pair is None: pair = [0, 1]
effective = gpu_numa.get(pair[0], 0)
if effective < 0: effective = 0
print(f"NUMA_GPU_PAIR={pair[0]},{pair[1]}")
print(f"NUMA_NODE={effective}")
PYEOF
)
GPU_PAIR=$(echo "${NUMA_GPUS}" | grep "^NUMA_GPU_PAIR=" | cut -d= -f2)
GPU_PAIR="${GPU_PAIR:-0,1}"
echo ">>> Using GPUs: ${GPU_PAIR}"

MTP1_SPEC='--speculative-config {"method":"mtp","num_speculative_tokens":1}'

# ── Helper: run serve benchmark ──────────────────────────────────────────────
run_serve() {
    local tag="$1"; shift
    local extra_args="$*"
    local port=8000
    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  BENCH: ${tag}"
    echo "  extra: ${extra_args}"
    echo "  START: $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${GPU_PAIR}" \
    vllm serve "${MODEL}" \
        --port ${port} \
        --dtype bfloat16 \
        --tensor-parallel-size 2 \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image":0,"video":0}' \
        ${extra_args} \
        > "${outdir}/server.log" 2>&1 &
    local SRV_PID=$!

    local ELAPSED=0
    sleep 5
    until curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        if ! kill -0 ${SRV_PID} 2>/dev/null; then
            echo "  ERROR: server died"
            tail -30 "${outdir}/server.log"
            return 1
        fi
        [ ${ELAPSED} -ge ${TIMEOUT} ] && echo "  TIMEOUT" && kill ${SRV_PID} 2>/dev/null && return 1
        sleep 10; ELAPSED=$((ELAPSED+10))
        [ $((ELAPSED % 60)) -eq 0 ] && echo "  [${ELAPSED}s] waiting..."
    done
    echo "  Server ready after ${ELAPSED}s"

    # Warmup
    echo "  Warmup (${SERVE_WARMUP} prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --base-url "http://localhost:${port}" --model "${MODEL}" \
        --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
        --num-prompts ${SERVE_WARMUP} --max-concurrency ${BS} --request-rate inf \
        --ignore-eos > /dev/null 2>&1 || true

    # Main bench
    echo "  Benchmark (${SERVE_PROMPTS} prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --base-url "http://localhost:${port}" --model "${MODEL}" \
        --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
        --num-prompts ${SERVE_PROMPTS} --max-concurrency ${BS} --request-rate inf \
        --ignore-eos --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "50,90,99" --save-result \
        --result-dir "${outdir}" --result-filename "serve.json" \
        2>&1 | tee "${outdir}/serve.txt"

    # Spec decode acceptance metrics
    echo "  --- Spec decode acceptance ---"
    grep "SpecDecoding metrics" "${outdir}/server.log" | tail -3

    # Shutdown
    kill -TERM ${SRV_PID} 2>/dev/null || true
    for i in $(seq 1 60); do
        kill -0 ${SRV_PID} 2>/dev/null || break; sleep 1
    done
    kill -9 ${SRV_PID} 2>/dev/null || true
    wait ${SRV_PID} 2>/dev/null || true
    sleep 2
    echo "  END: $(date)"
}

# ── Helper: reinstall clean vLLM ─────────────────────────────────────────────
reinstall_vllm() {
    echo ""
    echo ">>> Reinstalling clean vLLM..."
    pip install --quiet --upgrade --force-reinstall vllm 2>&1 | tail -3
    pip uninstall flash-attn -y 2>/dev/null || true
    export PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):${PATH}"
}

# ── Helper: apply O4+O5 ─────────────────────────────────────────────────────
apply_o4o5() {
    echo ">>> Applying O4+O5 patches..."
    python3 /workspace/patch_mtp1_opts.py --only o4
    python3 /workspace/patch_mtp1_opts.py --only o5
}

# ==========================================================================
# Phase 1: Baseline — no patches, async ON (default)
# ==========================================================================
echo ""
echo "============================================================"
echo " Phase 1: mtp1_base_async (no patches, async ON)"
echo "============================================================"

run_serve "mtp1_base_async" "${MTP1_SPEC}"

# ==========================================================================
# Phase 2: Baseline — no patches, async OFF
# ==========================================================================
echo ""
echo "============================================================"
echo " Phase 2: mtp1_base_noasync (no patches, async OFF)"
echo "============================================================"

run_serve "mtp1_base_noasync" "${MTP1_SPEC} --no-async-scheduling"

# ==========================================================================
# Phase 3: O4+O5 — async ON
# ==========================================================================
echo ""
echo "============================================================"
echo " Phase 3: mtp1_o4o5_async (O4+O5, async ON)"
echo "============================================================"

apply_o4o5

run_serve "mtp1_o4o5_async" "${MTP1_SPEC}"

# ==========================================================================
# Phase 4: O4+O5 — async OFF
# ==========================================================================
echo ""
echo "============================================================"
echo " Phase 4: mtp1_o4o5_noasync (O4+O5, async OFF)"
echo "============================================================"
# Same O4+O5 patches already applied, just change scheduling mode

run_serve "mtp1_o4o5_noasync" "${MTP1_SPEC} --no-async-scheduling"

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"

python3 - <<'PYEOF'
import json, os
from pathlib import Path

RD = os.environ.get("RESULTS_DIR", os.environ.get("RESULTS_BASE", "/workspace/results"))
order = ["mtp1_base_async", "mtp1_base_noasync", "mtp1_o4o5_async", "mtp1_o4o5_noasync"]
W = 120

print(f"\n{'=' * W}")
print(f"  No-Async-Scheduling Impact — Qwen3.5-35B-A3B TP=2 BF16 ISL=34K OSL=300 MTP1")
print(f"{'=' * W}")
print()

results = {}
for tag in order:
    sf = Path(RD) / tag / "serve.json"
    if not sf.exists():
        continue
    d = json.load(open(sf))
    results[tag] = d

header = (f"  {'Config':<25} {'TPOT mean':>12} {'±std':>7} {'TPOT P50':>9} "
          f"{'TPOT P99':>9} {'ITL P50':>8} {'TTFT P50':>9} {'E2E mean':>10} "
          f"{'Accept':>7}")
print(header)
print(f"  {'-'*23} {'-'*12} {'-'*7} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*10} {'-'*7}")

for tag in order:
    if tag not in results:
        continue
    d = results[tag]
    tpot_mean = d.get("mean_tpot_ms", 0)
    if tpot_mean == 0:
        continue
    tpot_std = d.get("std_tpot_ms", 0)
    tpot_p50 = d.get("p50_tpot_ms", 0)
    tpot_p99 = d.get("p99_tpot_ms", 0)
    itl_p50 = d.get("p50_itl_ms", 0)
    ttft_p50 = d.get("p50_ttft_ms", 0)
    e2e_mean = d.get("mean_e2el_ms", 0)

    accept = ""
    ar = d.get("acceptance_rate", None)
    if ar is not None:
        accept = f"{ar*100:.0f}%"

    print(f"  {tag:<25} {tpot_mean:>9.3f}ms {tpot_std:>6.3f} {tpot_p50:>8.2f}ms "
          f"{tpot_p99:>8.2f}ms {itl_p50:>7.2f}ms {ttft_p50:>8.1f}ms {e2e_mean:>9.1f}ms "
          f"{accept:>7}")

# ── Analysis: O4+O5 impact under async vs no-async ──────────────────────────
print(f"\n  {'─' * 80}")
print(f"  Analysis: O4+O5 savings under different scheduling modes")
print(f"  {'─' * 80}")

for mode, base_tag, opt_tag in [
    ("async ON ", "mtp1_base_async",   "mtp1_o4o5_async"),
    ("async OFF", "mtp1_base_noasync", "mtp1_o4o5_noasync"),
]:
    if base_tag in results and opt_tag in results:
        base_tpot = results[base_tag].get("mean_tpot_ms", 0)
        opt_tpot = results[opt_tag].get("mean_tpot_ms", 0)
        base_itl = results[base_tag].get("p50_itl_ms", 0)
        opt_itl = results[opt_tag].get("p50_itl_ms", 0)
        if base_tpot > 0 and opt_tpot > 0:
            tpot_delta = opt_tpot - base_tpot
            tpot_pct = tpot_delta / base_tpot * 100
            itl_delta = opt_itl - base_itl
            print(f"  {mode}:  TPOT {base_tpot:.3f} → {opt_tpot:.3f}ms  "
                  f"(delta={tpot_delta:+.3f}ms, {tpot_pct:+.1f}%)  "
                  f"ITL {base_itl:.2f} → {opt_itl:.2f}ms (delta={itl_delta:+.2f}ms)")

# ── Analysis: async vs no-async baseline impact ─────────────────────────────
print()
for patches, async_tag, noasync_tag in [
    ("no patches", "mtp1_base_async",  "mtp1_base_noasync"),
    ("O4+O5     ", "mtp1_o4o5_async",  "mtp1_o4o5_noasync"),
]:
    if async_tag in results and noasync_tag in results:
        a_tpot = results[async_tag].get("mean_tpot_ms", 0)
        n_tpot = results[noasync_tag].get("mean_tpot_ms", 0)
        a_itl = results[async_tag].get("p50_itl_ms", 0)
        n_itl = results[noasync_tag].get("p50_itl_ms", 0)
        if a_tpot > 0 and n_tpot > 0:
            tpot_delta = n_tpot - a_tpot
            tpot_pct = tpot_delta / a_tpot * 100
            print(f"  {patches}:  async→noasync  TPOT {a_tpot:.3f} → {n_tpot:.3f}ms  "
                  f"(delta={tpot_delta:+.3f}ms, {tpot_pct:+.1f}%)  "
                  f"ITL {a_itl:.2f} → {n_itl:.2f}ms")

print(f"\n{'=' * W}")
print(f"  Results: {RD}")
print(f"{'=' * W}")
PYEOF

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
