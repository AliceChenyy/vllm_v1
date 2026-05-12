#!/usr/bin/env bash
# =============================================================================
# MTP1 Decode Step Profiling — Qwen3.5-35B-A3B
# TP=2, ISL=34K, OSL=300, BS=1, BF16
#
# Profiles both no_mtp and mtp1 to identify the 2.52ms MTP overhead source.
# Uses CUDA-synced timestamps for accurate GPU timing.
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
SERVE_PROMPTS="${SERVE_PROMPTS:-10}"
SERVE_WARMUP=5
SERVE_RUNS="${SERVE_RUNS:-1}"
TIMEOUT="${TIMEOUT:-3600}"
MODEL="Qwen/Qwen3.5-35B-A3B"

LOG="${RESULTS_BASE}/master.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " MTP1 Decode Step Profiling — Qwen3.5-35B-A3B"
echo " TP=2  ISL=${ISL}  OSL=${OSL}  BS=${BS}  BF16"
echo " Start: $(date)"
echo "============================================================"

# ── Phase 0: Install & patch ────────────────────────────────────────────────
echo ""
echo ">>> [Phase 0] Install vLLM & apply profiling patch"
# Nuke entire .local to avoid stale numpy/torch/vllm breaking NGC container
rm -rf /workspace/.local 2>/dev/null || true
pip install --quiet --upgrade vllm 2>&1 | tail -5
pip uninstall flash-attn -y 2>/dev/null || true
# Ensure vllm CLI is on PATH
export PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):${PATH}"

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo "  vLLM: ${VLLM_VER}  PyTorch: ${TORCH_VER}  CUDA: ${CUDA_VER}"

# Apply profiling patch
python3 /workspace/profile_mtp1_patch.py
echo ""

cat > "${RESULTS_BASE}/env.txt" <<SW
vllm=${VLLM_VER}
pytorch=${TORCH_VER}
cuda=${CUDA_VER}
isl=${ISL} osl=${OSL} bs=${BS} tp=2
workload=mtp1_profile
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

# ── Helper: run profiled serve ───────────────────────────────────────────────
run_profiled() {
    local tag="$1"; shift
    local extra_args="$*"
    local port=8000
    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  PROFILE: ${tag}"
    echo "  extra: ${extra_args}"
    echo "  START: $(date)"
    echo "=========================================="

    # Enable MTP profiling
    export VLLM_MTP_PROFILE=1
    export VLLM_MTP_PROFILE_WARMUP=15
    export VLLM_MTP_PROFILE_OUT="${outdir}/mtp_profile.json"

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

    # Graceful shutdown — send SIGUSR1 to all vllm python children to trigger
    # profile dump in engine core subprocess, THEN SIGTERM the parent
    echo "  Shutting down (triggers profile dump)..."
    # Send SIGUSR1 to all child python processes (engine core workers)
    local child_pids
    child_pids=$(pgrep -P ${SRV_PID} -f python 2>/dev/null || true)
    if [ -n "${child_pids}" ]; then
        echo "  Sending SIGUSR1 to engine children: ${child_pids}"
        for cpid in ${child_pids}; do
            kill -USR1 ${cpid} 2>/dev/null || true
        done
        sleep 2  # let dump complete
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
    echo "  --- Profile Results ---"
    sed -n '/vLLM MTP Profile/,/^=\{50,\}/p' "${outdir}/server.log" 2>/dev/null | head -150 \
        || echo "  (profile not found in log)"

    if [ -f "${outdir}/mtp_profile.json" ]; then
        echo "  Profile JSON: ${outdir}/mtp_profile.json"
        python3 -c "import json; d=json.load(open('${outdir}/mtp_profile.json')); print(f'  Steps: {d[\"num_steps\"]}, Regions: {len(d.get(\"stats\",{}))}')" 2>/dev/null
    fi
    echo "  END: $(date)"
}

# ── Phase 1: Profile no_mtp baseline ─────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 1: Profile no_mtp baseline"
echo "============================================================"
run_profiled "profile_no_mtp" ""

# ── Phase 2: Profile MTP-1 ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2: Profile MTP-1"
echo "============================================================"
run_profiled "profile_mtp1" \
    "--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1}"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " COMPARISON SUMMARY"
echo "============================================================"

python3 - <<'PYEOF'
import json, os
from pathlib import Path

RD = os.environ.get("RESULTS_DIR", os.environ.get("RESULTS_BASE", "/workspace/results"))

for tag in ["profile_no_mtp", "profile_mtp1"]:
    print(f"\n{'=' * 100}")
    print(f"  {tag}")
    print(f"{'=' * 100}")

    # Serve results
    for run in range(1, 5):
        sf = Path(RD) / tag / f"serve_run{run}.json"
        if not sf.exists():
            break
        d = json.load(open(sf))
        tpot = d.get('mean_tpot_ms', 0)
        if tpot == 0:
            print(f"  Run{run}: FAILED (all zeros)")
            continue
        print(f"  Run{run}: TPOT mean={tpot:.3f}ms  "
              f"P50={d.get('p50_tpot_ms',0):.2f}ms  "
              f"ITL P50={d.get('p50_itl_ms',0):.2f}ms  "
              f"TTFT={d.get('p50_ttft_ms',0):.1f}ms  "
              f"E2E={d.get('mean_e2el_ms',0):.1f}ms")

    # Profile results
    pf = Path(RD) / tag / "mtp_profile.json"
    if pf.exists():
        data = json.load(open(pf))
        stats = data.get("stats", {})
        print(f"\n  Profiled {data['num_steps']} steps:")

        key_regions = [
            "step_total_us", "em_total", "st_total",
            "em.update_states", "em.prepare_inputs",
            "em.model_forward", "em.compute_logits",
            "st.rejection_sample", "st.update_states_after",
            "st.propose_draft", "st.copy_draft_to_cpu",
            "st.bookkeeping", "st.finalize_kv",
            "draft.propose_total", "draft.set_inputs",
            "draft.build_attn", "draft.build_model_inputs",
            "draft.sample",
        ]
        for r in key_regions:
            if r in stats:
                s = stats[r]
                ms = s["mean_us"] / 1000
                print(f"    {r:<35} {ms:>7.3f}ms  (med={s['median_us']/1000:.3f}ms)")

# Delta analysis
no_mtp_pf = Path(RD) / "profile_no_mtp" / "mtp_profile.json"
mtp1_pf = Path(RD) / "profile_mtp1" / "mtp_profile.json"
if no_mtp_pf.exists() and mtp1_pf.exists():
    ns = json.load(open(no_mtp_pf)).get("stats", {})
    ms = json.load(open(mtp1_pf)).get("stats", {})
    print(f"\n{'=' * 100}")
    print(f"  MTP1 OVERHEAD BREAKDOWN (delta vs no_mtp)")
    print(f"{'=' * 100}")
    for r in ["step_total_us", "em_total", "st_total",
              "em.update_states", "em.prepare_inputs",
              "em.model_forward", "em.compute_logits",
              "st.rejection_sample", "st.propose_draft",
              "st.copy_draft_to_cpu", "st.bookkeeping", "st.finalize_kv",
              "draft.propose_total", "draft.set_inputs",
              "draft.build_attn", "draft.sample"]:
        nv = ns.get(r, {}).get("mean_us", 0)
        mv = ms.get(r, {}).get("mean_us", 0)
        delta = mv - nv
        print(f"    {r:<35} no_mtp={nv/1000:.3f}ms  mtp1={mv/1000:.3f}ms  delta={delta/1000:+.3f}ms")

print()
PYEOF

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
