#!/usr/bin/env bash
# =============================================================================
# CUDA Graph Minimal Capture Benchmark — Qwen3.5-35B-A3B MTP1
# TP=2, ISL=34K, OSL=300, BS=1, BF16
#
# Configs tested:
#   1. mtp1_base          — MTP1 default graph sizes (~51 sizes)
#   2. mtp1_minimal_graph — MTP1 with VLLM_MTP_MINIMAL_GRAPHS=1 (only BS=1)
#
# Measures: server startup time, TPOT, ITL, acceptance rate
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
echo " CUDA Graph Minimal Capture Bench — Qwen3.5-35B-A3B"
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
workload=cudagraph_mtp_bench
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

MTP1_BASE='--speculative-config {"method":"mtp","num_speculative_tokens":1}'

# ── Helper: run serve benchmark ──────────────────────────────────────────────
run_serve() {
    local tag="$1"; shift
    local extra_env="$1"; shift
    local extra_args="$*"
    local port=8000
    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  BENCH: ${tag}"
    echo "  extra_env: ${extra_env}"
    echo "  extra_args: ${extra_args}"
    echo "  START: $(date)"
    echo "=========================================="

    # Record startup time
    local START_TS=$(date +%s%N)

    env ${extra_env} \
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

    local END_TS=$(date +%s%N)
    local STARTUP_MS=$(( (END_TS - START_TS) / 1000000 ))
    echo "  Server ready after ${ELAPSED}s (startup: ${STARTUP_MS}ms)"
    echo "${STARTUP_MS}" > "${outdir}/startup_ms.txt"

    # Count captured CUDA graphs from server log
    local GRAPH_COUNT=$(grep -c "CG Capture" "${outdir}/server.log" 2>/dev/null || echo "0")
    local PROPOSER_GRAPHS=$(grep "O6.*minimal CUDA graphs" "${outdir}/server.log" 2>/dev/null || echo "N/A")
    echo "  CUDA graphs captured: ${GRAPH_COUNT}"
    echo "  Proposer graph info: ${PROPOSER_GRAPHS}"
    echo "${GRAPH_COUNT}" > "${outdir}/graph_count.txt"

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

# ── Phase 1: Baseline MTP1 (default graph sizes, no patches) ─────────────────
echo ""
echo "============================================================"
echo " Phase 1: Baseline MTP1 (default CUDA graph sizes)"
echo "============================================================"

run_serve "mtp1_base" "" "${MTP1_BASE}"

# ── Phase 2: Apply O6 patch and bench with minimal graphs ────────────────────
echo ""
echo "============================================================"
echo " Phase 2: Apply O6 patch (minimal CUDA graphs for proposer)"
echo "============================================================"
python3 /workspace/patch_cudagraph_mtp.py

run_serve "mtp1_minimal_graph" "VLLM_MTP_MINIMAL_GRAPHS=1" "${MTP1_BASE}"

# ── Phase 3: Minimal graphs + O2 (FULL cudagraph) ────────────────────────────
echo ""
echo "============================================================"
echo " Phase 3: O6 + O2 (minimal graphs + FULL cudagraph)"
echo "============================================================"
# Apply O2 on top (if not already applied by O6 patch)
python3 /workspace/patch_mtp1_opts.py --only o2 2>/dev/null || true

run_serve "mtp1_minimal_full" "VLLM_MTP_MINIMAL_GRAPHS=1" "${MTP1_BASE}"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"

python3 - <<'PYEOF'
import json, os
from pathlib import Path

RD = os.environ.get("RESULTS_DIR", os.environ.get("RESULTS_BASE", "/workspace/results"))
order = ["mtp1_base", "mtp1_minimal_graph", "mtp1_minimal_full"]
W = 110

print(f"\n{'=' * W}")
print(f"  CUDA Graph Minimal Capture Results — Qwen3.5-35B-A3B TP=2 BF16 ISL=34K OSL=300")
print(f"{'=' * W}")
print()

header = (f"  {'Config':<22} {'Startup':>10} {'Graphs':>7} {'TPOT mean':>12} "
          f"{'TPOT P50':>9} {'TPOT P99':>9} {'ITL P50':>8} {'TTFT P50':>9} "
          f"{'Accept':>7}")
print(header)
print(f"  {'-'*20} {'-'*10} {'-'*7} {'-'*12} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*7}")

base_startup = None
base_tpot = None

for tag in order:
    sf = Path(RD) / tag / "serve.json"
    startup_f = Path(RD) / tag / "startup_ms.txt"
    graph_f = Path(RD) / tag / "graph_count.txt"

    if not sf.exists():
        continue

    d = json.load(open(sf))
    tpot_mean = d.get("mean_tpot_ms", 0)
    tpot_p50 = d.get("p50_tpot_ms", 0)
    tpot_p99 = d.get("p99_tpot_ms", 0)
    itl_p50 = d.get("p50_itl_ms", 0)
    ttft_p50 = d.get("p50_ttft_ms", 0)

    startup_ms = int(startup_f.read_text().strip()) if startup_f.exists() else 0
    graph_count = graph_f.read_text().strip() if graph_f.exists() else "?"

    accept = ""
    ar = d.get("acceptance_rate", None)
    if ar is not None:
        accept = f"{ar*100:.0f}%"

    startup_str = f"{startup_ms/1000:.1f}s"
    if base_startup is not None and base_startup > 0:
        delta = (startup_ms - base_startup) / base_startup * 100
        startup_str += f" ({delta:+.0f}%)"
    if tag == "mtp1_base":
        base_startup = startup_ms
        base_tpot = tpot_mean

    tpot_delta = ""
    if base_tpot is not None and base_tpot > 0 and tag != "mtp1_base":
        pct = (tpot_mean - base_tpot) / base_tpot * 100
        tpot_delta = f" ({pct:+.1f}%)"

    print(f"  {tag:<22} {startup_str:>10} {graph_count:>7} {tpot_mean:>9.3f}ms{tpot_delta:>6} "
          f"{tpot_p50:>8.2f}ms {tpot_p99:>8.2f}ms {itl_p50:>7.2f}ms {ttft_p50:>8.1f}ms "
          f"{accept:>7}")

print(f"\n{'=' * W}")
print(f"  Results: {RD}")
print(f"{'=' * W}")
PYEOF

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
