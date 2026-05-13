#!/usr/bin/env bash
# =============================================================================
# MTP1 propose_draft Deep Profiling — Qwen3.5-35B-A3B
# TP=2, ISL=34K, OSL=300, BS=1, BF16, MTP1
#
# Profiles the INTERNAL sub-components of propose_draft to identify
# where the +1.01ms overhead comes from.
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
SERVE_RUNS="${SERVE_RUNS:-1}"
TIMEOUT="${TIMEOUT:-3600}"
MODEL="Qwen/Qwen3.5-35B-A3B"

LOG="${RESULTS_BASE}/master.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " MTP1 propose_draft Deep Profiling — Qwen3.5-35B-A3B"
echo " TP=2  ISL=${ISL}  OSL=${OSL}  BS=${BS}  BF16  MTP1"
echo " Start: $(date)"
echo "============================================================"

# ── Phase 0: Install & patch ────────────────────────────────────────────────
echo ""
echo ">>> [Phase 0] Install vLLM & apply propose_draft profiling patch"
rm -rf /workspace/.local 2>/dev/null || true
pip install --quiet --upgrade vllm 2>&1 | tail -5
pip uninstall flash-attn -y 2>/dev/null || true
export PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):${PATH}"

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo "  vLLM: ${VLLM_VER}  PyTorch: ${TORCH_VER}  CUDA: ${CUDA_VER}"

# Apply propose_draft profiling patch
python3 /workspace/patch_propose_draft_profile.py
echo ""

cat > "${RESULTS_BASE}/env.txt" <<SW
vllm=${VLLM_VER}
pytorch=${TORCH_VER}
cuda=${CUDA_VER}
isl=${ISL} osl=${OSL} bs=${BS} tp=2
workload=propose_draft_profile
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
NUMA_NODE=$(echo "${NUMA_GPUS}" | grep "^NUMA_NODE=" | cut -d= -f2)
GPU_PAIR="${GPU_PAIR:-0,1}"
echo ">>> Using GPUs: ${GPU_PAIR} (NUMA ${NUMA_NODE:-0})"

# ── Profile MTP1 propose_draft ──────────────────────────────────────────────
run_propose_profile() {
    local tag="$1"; shift
    local use_cuda_events="$1"; shift
    local extra_args="$*"
    local port=8000
    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  PROPOSE PROFILE: ${tag}"
    echo "  CUDA events: ${use_cuda_events}"
    echo "  extra: ${extra_args}"
    echo "  START: $(date)"
    echo "=========================================="

    # Enable propose_draft profiling
    export VLLM_PROPOSE_PROFILE=1
    export VLLM_PROPOSE_PROFILE_WARMUP=20
    export VLLM_PROPOSE_PROFILE_DUMP_INTERVAL=50
    export VLLM_PROPOSE_PROFILE_OUT="${outdir}/propose_profile.json"
    export VLLM_PROPOSE_PROFILE_CUDA_EVENTS="${use_cuda_events}"

    CUDA_VISIBLE_DEVICES="${GPU_PAIR}" \
    vllm serve "${MODEL}" \
        --port ${port} \
        --dtype bfloat16 \
        --tensor-parallel-size 2 \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image":0,"video":0}' \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
        ${extra_args} \
        > "${outdir}/server.log" 2>&1 &
    local SRV_PID=$!

    # Wait for ready
    local ELAPSED=0
    sleep 5
    until curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        if ! kill -0 ${SRV_PID} 2>/dev/null; then
            echo "  ERROR: server died"
            tail -80 "${outdir}/server.log"
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
    for run in $(seq 1 ${SERVE_RUNS}); do
        echo "  Run ${run}/${SERVE_RUNS} (${SERVE_PROMPTS} prompts)..."
        if ! curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "  ERROR: server not responding before run ${run}, skipping"
            break
        fi
        python3 -m vllm.entrypoints.cli.main bench serve \
            --base-url "http://localhost:${port}" --model "${MODEL}" \
            --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
            --num-prompts ${SERVE_PROMPTS} --max-concurrency ${BS} --request-rate inf \
            --ignore-eos --percentile-metrics "ttft,tpot,itl,e2el" \
            --metric-percentiles "50,90,99" --save-result \
            --result-dir "${outdir}" --result-filename "serve_run${run}.json" \
            2>&1 | tee "${outdir}/serve_run${run}.txt"
    done

    # Graceful shutdown — SIGUSR1 to children for profile dump
    echo "  Shutting down (triggers profile dump)..."
    local child_pids
    child_pids=$(pgrep -P ${SRV_PID} -f python 2>/dev/null || true)
    if [ -n "${child_pids}" ]; then
        echo "  Sending SIGUSR1 to engine children: ${child_pids}"
        for cpid in ${child_pids}; do
            kill -USR1 ${cpid} 2>/dev/null || true
        done
        sleep 2
    fi
    kill -TERM ${SRV_PID} 2>/dev/null || true
    for i in $(seq 1 60); do
        kill -0 ${SRV_PID} 2>/dev/null || break
        sleep 1
    done
    kill -9 ${SRV_PID} 2>/dev/null || true
    wait ${SRV_PID} 2>/dev/null || true
    sleep 2

    # Extract profile from server log
    echo ""
    echo "  --- Propose Profile Results ---"
    sed -n '/propose_draft Deep Profile/,/^=\{50,\}/p' "${outdir}/server.log" 2>/dev/null | head -150 \
        || echo "  (profile not found in log)"

    if [ -f "${outdir}/propose_profile.json" ]; then
        echo "  Profile JSON: ${outdir}/propose_profile.json"
        python3 -c "
import json
d = json.load(open('${outdir}/propose_profile.json'))
print(f'  Steps: {d[\"num_steps\"]}, Regions: {len(d.get(\"stats\",{}))}')
stats = d.get('stats', {})
print()
print('  Key timings (mean):')
for k in ['propose_total_cpu_us', 'set_inputs_cpu_us', 'build_attn_cpu_us',
           'build_model_inputs_cpu_us', 'model_forward_cpu_us',
           'compute_logits_cpu_us', 'sample_cpu_us', 'remaining_cpu_us']:
    if k in stats:
        ms = stats[k]['mean_us'] / 1000
        print(f'    {k:<35} {ms:.3f}ms')
" 2>/dev/null
    fi
    echo "  END: $(date)"
}

# ── Phase 1: CPU-only profiling (no sync overhead from CUDA events) ─────────
echo ""
echo "============================================================"
echo " Phase 1: propose_draft CPU profiling (no CUDA events)"
echo "============================================================"
run_propose_profile "propose_cpu_only" "0"

# ── Phase 2: CPU + CUDA event profiling ─────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2: propose_draft with CUDA event timing"
echo "============================================================"
run_propose_profile "propose_cuda_events" "1"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " PROPOSE_DRAFT PROFILING SUMMARY"
echo "============================================================"

python3 - <<'PYEOF'
import json, os
from pathlib import Path

RD = os.environ.get("RESULTS_DIR", os.environ.get("RESULTS_BASE", "/workspace/results"))

for tag in ["propose_cpu_only", "propose_cuda_events"]:
    pf = Path(RD) / tag / "propose_profile.json"
    if not pf.exists():
        continue

    data = json.load(open(pf))
    stats = data.get("stats", {})

    print(f"\n{'=' * 100}")
    print(f"  {tag} ({data['num_steps']} steps, cuda_events={data.get('cuda_events', False)})")
    print(f"{'=' * 100}")

    # Serve results
    for run in range(1, 5):
        sf = Path(RD) / tag / f"serve_run{run}.json"
        if not sf.exists():
            break
        d = json.load(open(sf))
        tpot = d.get('mean_tpot_ms', 0)
        if tpot == 0:
            continue
        print(f"  TPOT mean={tpot:.3f}ms  P50={d.get('p50_tpot_ms',0):.2f}ms")

    print(f"\n  propose_draft breakdown:")
    total = stats.get("propose_total_cpu_us", {}).get("mean_us", 1)

    regions = [
        ("propose_total_cpu_us", "TOTAL"),
        ("set_inputs_cpu_us", "  set_inputs_first_pass"),
        ("build_attn_cpu_us", "  build_attn_metadata"),
        ("build_model_inputs_cpu_us", "  build_model_inputs"),
        ("model_forward_cpu_us", "  model_forward"),
        ("compute_logits_cpu_us", "  compute_logits"),
        ("sample_cpu_us", "  _greedy_sample"),
        ("remaining_cpu_us", "  remaining"),
    ]

    for key, label in regions:
        if key not in stats:
            continue
        s = stats[key]
        ms = s["mean_us"] / 1000
        pct = s["mean_us"] / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {label:<35} {ms:>7.3f}ms  {pct:>5.1f}%  {bar}")

    # CUDA event timings
    cuda_keys = [k for k in stats if "cuda" in k]
    if cuda_keys:
        print(f"\n  CUDA event timings:")
        for k in sorted(cuda_keys):
            s = stats[k]
            print(f"    {k:<35} {s['mean_us']/1000:>7.3f}ms")

print()
PYEOF

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
