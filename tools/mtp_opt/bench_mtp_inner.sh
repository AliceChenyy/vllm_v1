#!/usr/bin/env bash
# =============================================================================
# MTP (Multi-Token Prediction) TPOT Optimization — Qwen3.5-35B-A3B on B200
# TP=2, ISL=34K, OSL=300, BS=1, BF16
#
# Tests:
#   1. no_mtp       — baseline without MTP (OSL=300)
#   2. mtp1         — MTP num_speculative_tokens=1
#   3. mtp2         — MTP num_speculative_tokens=2
#   4. mtp1_nopc    — MTP-1 with prefix caching disabled
# =============================================================================
set -uo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
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

# ── Workload config ─────────────────────────────────────────────────────────
ISL=34000
OSL=300
BS=1
MAX_MODEL_LEN=40960
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
SERVE_PROMPTS="${SERVE_PROMPTS:-15}"
SERVE_WARMUP=5
TIMEOUT="${TIMEOUT:-1800}"

MODEL="Qwen/Qwen3.5-35B-A3B"
RUN_CONFIGS="${RUN_CONFIGS:-all}"

LOG="${RESULTS_BASE}/master.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " MTP TPOT Optimization — Qwen3.5-35B-A3B B200"
echo " TP=2  ISL=${ISL}  OSL=${OSL}  BS=${BS}  BF16"
echo " Configs: ${RUN_CONFIGS}"
echo " Start: $(date)"
echo "============================================================"

# =============================================================================
# Phase 0: Environment setup
# =============================================================================
echo ""
echo ">>> [Phase 0] Environment setup"

pip install --quiet --upgrade vllm 2>&1 | tail -5
pip uninstall flash-attn -y 2>/dev/null || true
echo "Removed flash-attn to avoid symbol conflicts"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
FI_VER=$(python3 -c "import flashinfer; print(flashinfer.__version__)" 2>/dev/null || echo "N/A")

echo "  vLLM: ${VLLM_VER}  PyTorch: ${TORCH_VER}  CUDA: ${CUDA_VER}  FlashInfer: ${FI_VER}"

cat > "${RESULTS_BASE}/env.txt" <<SW
vllm=${VLLM_VER}
pytorch=${TORCH_VER}
cuda=${CUDA_VER}
flashinfer=${FI_VER}
timestamp=$(date -Iseconds)
isl=${ISL} osl=${OSL} bs=${BS} tp=2
workload=mtp_optimization
SW

echo ""
echo ">>> GPU info"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
    --format=csv,noheader 2>/dev/null | tee "${RESULTS_BASE}/gpu_info.txt"

# =============================================================================
# NUMA-optimal GPU pair selection (reused from ttft_tpot_tp2)
# =============================================================================
echo ""
echo ">>> Detecting NUMA-optimal GPU pair for TP=2"

NUMA_GPUS=$(python3 - <<'PYEOF'
import subprocess, collections

result = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,gpu_bus_id", "--format=csv,noheader"], text=True)

gpu_numa = {}
for line in result.strip().split("\n"):
    parts = line.strip().split(", ")
    if len(parts) != 2:
        continue
    idx = int(parts[0])
    bus_id = parts[1].lower()
    pci_path = f"/sys/bus/pci/devices/{bus_id}/numa_node"
    try:
        with open(pci_path) as f:
            numa = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        numa = -1
    gpu_numa[idx] = numa

numa_groups = collections.defaultdict(list)
for gpu, numa in gpu_numa.items():
    numa_groups[numa].append(gpu)

best_pair = None
for numa in sorted(numa_groups.keys()):
    gpus = sorted(numa_groups[numa])
    if len(gpus) >= 2:
        best_pair = gpus[:2]
        break

if best_pair is None:
    best_pair = [0, 1]

effective_numa = gpu_numa.get(best_pair[0], 0)
if effective_numa < 0:
    effective_numa = 0

print(f"NUMA_GPU_PAIR={best_pair[0]},{best_pair[1]}")
print(f"NUMA_NODE={effective_numa}")
PYEOF
)

GPU_PAIR=$(echo "${NUMA_GPUS}" | grep "^NUMA_GPU_PAIR=" | cut -d= -f2)
NUMA_NODE=$(echo "${NUMA_GPUS}" | grep "^NUMA_NODE=" | cut -d= -f2)
GPU_PAIR="${GPU_PAIR:-0,1}"
NUMA_NODE="${NUMA_NODE:-0}"

echo ">>> Using GPUs: ${GPU_PAIR} (NUMA node ${NUMA_NODE})"
echo "gpu_pair=${GPU_PAIR}" >> "${RESULTS_BASE}/env.txt"
echo "numa_node=${NUMA_NODE}" >> "${RESULTS_BASE}/env.txt"

# =============================================================================
# Helper: run serve benchmark
# =============================================================================
run_serve() {
    local tag="$1"
    shift
    local extra_args="$*"
    local port=8000

    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  TEST: ${tag}"
    echo "  extra_args: ${extra_args}"
    echo "  START: $(date)"
    echo "=========================================="

    # Launch server
    CUDA_VISIBLE_DEVICES="${GPU_PAIR}" \
    vllm serve "${MODEL}" \
        --port ${port} \
        --dtype bfloat16 \
        --tensor-parallel-size 2 \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image": 0, "video": 0}' \
        ${extra_args} \
        > "${outdir}/server.log" 2>&1 &
    local SRV_PID=$!

    # Wait for server ready
    local ELAPSED=0
    sleep 5
    until curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        if ! kill -0 ${SRV_PID} 2>/dev/null; then
            echo "  ERROR: server died"
            echo "  --- Last 50 lines of server log ---"
            tail -50 "${outdir}/server.log"
            echo "  ---"
            return 1
        fi
        if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
            echo "  ERROR: timeout ${TIMEOUT}s"
            tail -30 "${outdir}/server.log"
            kill ${SRV_PID} 2>/dev/null || true
            return 1
        fi
        sleep 10; ELAPSED=$((ELAPSED + 10))
        [ $((ELAPSED % 60)) -eq 0 ] && echo "  [${ELAPSED}s] waiting..."
    done
    echo "  Server ready after ${ELAPSED}s"

    # Print server config for verification
    echo "  --- Server startup config ---"
    grep -E "speculative|MTP|multi.token|spec_decode|num_speculative" "${outdir}/server.log" 2>/dev/null | head -10 || echo "  (no speculative config lines found)"
    echo "  ---"

    # Warmup
    echo "  Warmup (${SERVE_WARMUP} prompts, OSL=${OSL})..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --base-url "http://localhost:${port}" \
        --model "${MODEL}" \
        --dataset-name random \
        --random-input-len ${ISL} \
        --random-output-len ${OSL} \
        --num-prompts ${SERVE_WARMUP} \
        --max-concurrency ${BS} \
        --request-rate inf \
        --ignore-eos \
        > /dev/null 2>&1 || true

    # Main benchmark — 3 runs for stability
    for run in 1 2 3; do
        echo "  Run ${run}/3 (${SERVE_PROMPTS} prompts)..."
        python3 -m vllm.entrypoints.cli.main bench serve \
            --base-url "http://localhost:${port}" \
            --model "${MODEL}" \
            --dataset-name random \
            --random-input-len ${ISL} \
            --random-output-len ${OSL} \
            --num-prompts ${SERVE_PROMPTS} \
            --max-concurrency ${BS} \
            --request-rate inf \
            --ignore-eos \
            --percentile-metrics "ttft,tpot,itl,e2el" \
            --metric-percentiles "50,90,99" \
            --save-result \
            --result-dir "${outdir}" \
            --result-filename "serve_run${run}.json" \
            2>&1 | tee "${outdir}/serve_run${run}.txt"
    done

    # Collect server metrics & acceptance rate
    curl -sf "http://localhost:${port}/metrics" > "${outdir}/prometheus_metrics.txt" 2>/dev/null || true

    # Extract spec decode metrics if MTP is enabled
    if grep -q "speculative" <<< "${extra_args}"; then
        echo "  --- Speculative decode metrics ---"
        grep -E "spec_decode|acceptance|draft_token|speculative" "${outdir}/prometheus_metrics.txt" 2>/dev/null | head -20 || echo "  (no spec decode metrics found)"
        # Also try vLLM's internal metrics endpoint
        curl -sf "http://localhost:${port}/v1/completions" -H "Content-Type: application/json" \
            -d '{"model":"'"${MODEL}"'","prompt":"Hello","max_tokens":1}' > /dev/null 2>&1 || true
        echo "  ---"
    fi

    kill ${SRV_PID} 2>/dev/null || true
    wait ${SRV_PID} 2>/dev/null || true
    sleep 5
    echo "  END: $(date)"
}

should_run() {
    local config="$1"
    [ "${RUN_CONFIGS}" = "all" ] && return 0
    echo " ${RUN_CONFIGS} " | grep -q " ${config} "
}

# =============================================================================
# Phase 1: No-MTP Baseline (OSL=300)
# =============================================================================
if should_run "no_mtp"; then
    echo ""
    echo "============================================================"
    echo " Phase 1: No-MTP Baseline (OSL=${OSL})"
    echo "============================================================"
    run_serve "no_mtp" ""
fi

# =============================================================================
# Phase 2: MTP-1 (num_speculative_tokens=1)
# =============================================================================
if should_run "mtp1"; then
    echo ""
    echo "============================================================"
    echo " Phase 2: MTP-1 (num_speculative_tokens=1)"
    echo "============================================================"
    run_serve "mtp1" \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
fi

# =============================================================================
# Phase 3: MTP-2 (num_speculative_tokens=2)
# =============================================================================
if should_run "mtp2"; then
    echo ""
    echo "============================================================"
    echo " Phase 3: MTP-2 (num_speculative_tokens=2)"
    echo "============================================================"
    run_serve "mtp2" \
        --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
fi

# =============================================================================
# Phase 3b: MTP-3 (num_speculative_tokens=3)
# =============================================================================
if should_run "mtp3"; then
    echo ""
    echo "============================================================"
    echo " Phase 3b: MTP-3 (num_speculative_tokens=3)"
    echo "============================================================"
    run_serve "mtp3" \
        --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
fi

# =============================================================================
# Phase 4: MTP-1 + disable prefix caching
# =============================================================================
if should_run "mtp1_nopc"; then
    echo ""
    echo "============================================================"
    echo " Phase 4: MTP-1 + no prefix caching"
    echo "============================================================"
    run_serve "mtp1_nopc" \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
        --no-enable-prefix-caching
fi

# =============================================================================
# Phase 5: MTP-1 + higher gpu-mem-util (more KV cache room)
# =============================================================================
if should_run "mtp1_highmem"; then
    echo ""
    echo "============================================================"
    echo " Phase 5: MTP-1 + gpu-mem-util=0.95"
    echo "============================================================"
    run_serve "mtp1_highmem" \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
        --gpu-memory-utilization 0.95
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"

python3 - <<'PYEOF'
import json, glob, os, statistics
from pathlib import Path

RD = os.environ.get("RESULTS_DIR", os.environ.get("RESULTS_BASE", "/workspace/results"))

print(f"\n{'='*90}")
print(f"  MTP TPOT Optimization — Qwen3.5-35B-A3B B200 TP=2 BF16")
print(f"  ISL=34K  OSL=300  BS=1")
print(f"{'='*90}")

configs = {}
for d in sorted(Path(RD).iterdir()):
    if not d.is_dir():
        continue
    tag = d.name
    runs = sorted(glob.glob(str(d / "serve_run*.json")))
    tpots_mean, tpots_p50, tpots_p99, ttfts_p50, e2es = [], [], [], [], []
    for f in runs:
        try:
            data = json.load(open(f))
            m = data.get("mean_tpot_ms", 0)
            if m > 0:
                tpots_mean.append(m)
                tpots_p50.append(data.get("p50_tpot_ms", 0))
                tpots_p99.append(data.get("p99_tpot_ms", 0))
                ttfts_p50.append(data.get("p50_ttft_ms", 0))
                e2es.append(data.get("mean_e2el_ms", 0))
        except:
            pass
    if tpots_mean:
        configs[tag] = {
            "tpot_mean": statistics.mean(tpots_mean),
            "tpot_p50": statistics.mean(tpots_p50),
            "tpot_p99": statistics.mean(tpots_p99),
            "ttft_p50": statistics.mean(ttfts_p50),
            "e2e_mean": statistics.mean(e2es) if any(e2es) else 0,
            "tpot_stdev": statistics.stdev(tpots_mean) if len(tpots_mean) > 1 else 0,
            "n_runs": len(tpots_mean),
        }

if configs:
    baseline_tpot = configs.get("no_mtp", {}).get("tpot_mean", 0)
    baseline_e2e = configs.get("no_mtp", {}).get("e2e_mean", 0)

    print(f"\n  {'Config':<18} {'TPOT mean':>10} {'±stdev':>8} {'TPOT P50':>9} {'TPOT P99':>9} {'TTFT P50':>9} {'E2E mean':>10} {'vs base':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*8}")

    for tag in ["no_mtp", "mtp1", "mtp2", "mtp3", "mtp1_nopc", "mtp1_highmem"]:
        if tag not in configs:
            continue
        c = configs[tag]
        if baseline_tpot > 0 and tag != "no_mtp":
            delta = (c["tpot_mean"] - baseline_tpot) / baseline_tpot * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "—"

        print(f"  {tag:<18} {c['tpot_mean']:>9.3f}ms {c['tpot_stdev']:>7.3f} {c['tpot_p50']:>8.2f}ms {c['tpot_p99']:>8.2f}ms {c['ttft_p50']:>8.1f}ms {c['e2e_mean']:>9.1f}ms {delta_str:>8}")

    # E2E comparison (most important for user: total request latency)
    if baseline_e2e > 0:
        print(f"\n  E2E Latency Comparison (what the user sees):")
        for tag in sorted(configs.keys(), key=lambda x: configs[x].get("e2e_mean", 9999)):
            c = configs[tag]
            if c["e2e_mean"] > 0:
                delta_e2e = (c["e2e_mean"] - baseline_e2e) / baseline_e2e * 100
                print(f"    {tag:<18} E2E={c['e2e_mean']:.1f}ms  ({delta_e2e:+.1f}% vs no_mtp)")

    # Previous reference
    print(f"\n  Previous results (vLLM 0.20.1, OSL=32, no MTP):")
    print(f"    TP=2 BF16: TPOT mean=3.20ms, TTFT P50=446ms, E2E=547ms")
    print(f"    Note: OSL increased 32→300, so E2E is expected ~10x higher in decode phase")

print(f"\n{'='*90}")
print(f"  Results: {RD}")
print(f"{'='*90}\n")
PYEOF

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
