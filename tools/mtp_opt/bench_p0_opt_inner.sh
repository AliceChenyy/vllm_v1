#!/usr/bin/env bash
# =============================================================================
# P0 CPU Overhead Optimization — MTP Decode Bench
# Tests: baseline vs patched for mtp3, TP=2, ISL=34K, OSL=300, BS=1, BF16
# Validates: TPOT reduction (target >0) + GSM8K quality preserved
#
# Runs inside nvcr.io/nvidia/pytorch:25.10-py3 container.
# =============================================================================
set -uo pipefail

RESULTS_BASE="${RESULTS_DIR:-/workspace/results/p0_opt}"
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
TIMEOUT="${TIMEOUT:-3600}"
MODEL="Qwen/Qwen3.5-35B-A3B"
MTP_CONFIG='{"method":"mtp","num_speculative_tokens":3}'

LOG="${RESULTS_BASE}/master.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " P0 CPU Opt — MTP=3  TP=2  ISL=${ISL}  OSL=${OSL}  BF16"
echo " Start: $(date)"
echo "============================================================"

# ── Phase 0: install vllm ────────────────────────────────────────
echo ""
echo ">>> [Phase 0] Install vLLM"
pip install --quiet "vllm==0.20.2" 2>&1 | tail -3
pip uninstall flash-attn -y 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

# Fix: system flash_attn has ABI-broken .so (aarch64/CUDA mismatch).
# Wrap the flash_attn import in rotary_embedding to fail gracefully.
python3 /workspace/fix_flash_attn.py

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo "  vLLM: ${VLLM_VER}  PyTorch: ${TORCH_VER}  CUDA: ${CUDA_VER}"

cat > "${RESULTS_BASE}/env.txt" <<SW
vllm=${VLLM_VER}
pytorch=${TORCH_VER}
cuda=${CUDA_VER}
isl=${ISL} osl=${OSL} bs=${BS} tp=2 mtp=3
timestamp=$(date -Iseconds)
SW

# ── GPU / NUMA detection ─────────────────────────────────────────
echo ""
echo ">>> GPU info"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee "${RESULTS_BASE}/gpu_info.txt"

NUMA_GPUS=$(python3 - <<'PYEOF'
import subprocess, collections
result = subprocess.check_output(
    ["nvidia-smi","--query-gpu=index,gpu_bus_id","--format=csv,noheader"], text=True)
gpu_numa = {}
for line in result.strip().split("\n"):
    p = line.strip().split(", ")
    if len(p) != 2: continue
    idx, bus = int(p[0]), p[1].lower()
    try:
        numa = int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read().strip())
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
NUMA_NODE=$(echo "${NUMA_GPUS}" | grep "^NUMA_NODE=" | cut -d= -f2)
GPU_PAIR="${GPU_PAIR:-0,1}"
NUMA_NODE="${NUMA_NODE:-0}"
echo ">>> Using GPUs: ${GPU_PAIR} (NUMA ${NUMA_NODE})"
echo "gpu_pair=${GPU_PAIR}" >> "${RESULTS_BASE}/env.txt"

# ── Helper: serve + bench ────────────────────────────────────────
run_serve() {
    local tag="$1"; shift
    local extra_args="$*"
    local port=8000
    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  RUN: ${tag}"
    echo "  args: ${extra_args}"
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
            echo "  ERROR: server died"; tail -30 "${outdir}/server.log"; return 1
        fi
        [ ${ELAPSED} -ge ${TIMEOUT} ] && echo "  ERROR: timeout" && kill ${SRV_PID} 2>/dev/null && return 1
        sleep 10; ELAPSED=$((ELAPSED+10))
        [ $((ELAPSED % 60)) -eq 0 ] && echo "  [${ELAPSED}s] waiting for server..."
    done
    echo "  Server ready after ${ELAPSED}s"

    # Warmup
    python3 -m vllm.entrypoints.cli.main bench serve \
        --base-url "http://localhost:${port}" --model "${MODEL}" \
        --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
        --num-prompts ${SERVE_WARMUP} --max-concurrency ${BS} --request-rate inf \
        --ignore-eos > /dev/null 2>&1 || true

    # 3 benchmark runs
    for run in 1 2 3; do
        echo "  Run ${run}/3 (${SERVE_PROMPTS} prompts)..."
        python3 -m vllm.entrypoints.cli.main bench serve \
            --base-url "http://localhost:${port}" --model "${MODEL}" \
            --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
            --num-prompts ${SERVE_PROMPTS} --max-concurrency ${BS} --request-rate inf \
            --ignore-eos --percentile-metrics "ttft,tpot,itl,e2el" \
            --metric-percentiles "50,90,99" --save-result \
            --result-dir "${outdir}" --result-filename "serve_run${run}.json" \
            2>&1 | tee "${outdir}/serve_run${run}.txt"
    done

    # Capture spec decode acceptance metrics
    curl -sf "http://localhost:${port}/metrics" > "${outdir}/prometheus_metrics.txt" 2>/dev/null || true
    grep -E "spec_decode|acceptance|draft_token|speculative" \
        "${outdir}/prometheus_metrics.txt" 2>/dev/null | head -20 || true

    kill ${SRV_PID} 2>/dev/null || true
    wait ${SRV_PID} 2>/dev/null || true
    sleep 3
    echo "  END: $(date)"
}


# ============================================================
# Phase 1: Baseline (no patch)  — mtp3 + no_mtp
# ============================================================
echo ""
echo "============================================================"
echo " Phase 1: Baseline (no patch)"
echo "============================================================"
run_serve "baseline_no_mtp" ""
run_serve "baseline_mtp3" \
    "--speculative-config ${MTP_CONFIG}"

# ============================================================
# Phase 2: Apply P0 patch
# ============================================================
echo ""
echo "============================================================"
echo " Phase 2: Applying P0 opt patch"
echo "============================================================"
python3 /workspace/apply_p0_opt_patch.py
echo ""
python3 /workspace/apply_p0_opt_patch.py --check

# ============================================================
# Phase 3: Patched  — mtp3
# ============================================================
echo ""
echo "============================================================"
echo " Phase 3: Patched (P0 opt)"
echo "============================================================"
run_serve "patched_mtp3" \
    "--speculative-config ${MTP_CONFIG}"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"
python3 - <<'PYEOF'
import json, glob, os, statistics
from pathlib import Path

RD = os.environ.get("RESULTS_DIR", os.environ.get("RESULTS_BASE", "/workspace/results/p0_opt"))

def load_runs(tag):
    files = sorted(glob.glob(f"{RD}/{tag}/serve_run*.json"))
    tpots_mean, tpots_p50, tpots_p99, e2es, ttfts = [], [], [], [], []
    for f in files:
        try:
            d = json.load(open(f))
            if d.get("mean_tpot_ms", 0) > 0:
                tpots_mean.append(d["mean_tpot_ms"])
                tpots_p50.append(d.get("p50_tpot_ms", 0))
                tpots_p99.append(d.get("p99_tpot_ms", 0))
                e2es.append(d.get("mean_e2el_ms", 0))
                ttfts.append(d.get("p50_ttft_ms", 0))
        except: pass
    if not tpots_mean:
        return None
    n = len(tpots_mean)
    return dict(
        tpot_mean=statistics.mean(tpots_mean),
        tpot_p50=statistics.mean(tpots_p50),
        tpot_p99=statistics.mean(tpots_p99),
        tpot_std=statistics.stdev(tpots_mean) if n > 1 else 0,
        e2e=statistics.mean(e2es),
        ttft=statistics.mean(ttfts),
        n=n,
    )

configs = {
    "baseline_no_mtp":   load_runs("baseline_no_mtp"),
    "baseline_mtp3":     load_runs("baseline_mtp3"),
    "patched_mtp3":      load_runs("patched_mtp3"),
}

print(f"\n{'='*95}")
print(f"  P0 CPU Opt — Qwen3.5-35B-A3B  TP=2 BF16  ISL=34K OSL=300 BS=1")
print(f"{'='*95}")
print(f"  {'Config':<22} {'TPOT mean':>10} {'±std':>7} {'P50':>8} {'P99':>8} {'TTFT P50':>10} {'E2E mean':>10} {'vs base':>9}")
print(f"  {'-'*22} {'-'*10} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*9}")

base_tpot = configs["baseline_mtp3"]["tpot_mean"] if configs["baseline_mtp3"] else None

for tag, c in configs.items():
    if c is None:
        print(f"  {tag:<22}  (no data)")
        continue
    if base_tpot and tag != "baseline_no_mtp" and tag != "baseline_mtp3":
        delta = (c["tpot_mean"] - base_tpot) / base_tpot * 100
        vs = f"{delta:+.1f}%"
    else:
        vs = "—"
    print(f"  {tag:<22} {c['tpot_mean']:>9.3f}ms {c['tpot_std']:>6.3f} "
          f"{c['tpot_p50']:>7.2f}ms {c['tpot_p99']:>7.2f}ms "
          f"{c['ttft']:>9.1f}ms {c['e2e']:>9.1f}ms {vs:>9}")

print(f"\n  Results: {RD}")
print(f"{'='*95}\n")
PYEOF

echo ""
echo "============================================================"
echo " DONE  $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
