#!/usr/bin/env bash
# =============================================================================
# CPU Profile — MTP=3 Decode Step Breakdown on B200
# TP=2, ISL=34K, OSL=300, BS=1, BF16
#
# Uses runtime monkey-patching (cpu_profile_patch.py) — no vLLM source mods.
# Runs inside nvcr.io/nvidia/pytorch:25.10-py3 container.
# =============================================================================
set -uo pipefail

RESULTS_BASE="${RESULTS_DIR:-/workspace/results/cpu_profile}"
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
SERVE_WARMUP=3
TIMEOUT="${TIMEOUT:-3600}"
MODEL="Qwen/Qwen3.5-35B-A3B"
MTP_CONFIG='{"method":"mtp","num_speculative_tokens":3}'

LOG="${RESULTS_BASE}/master.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " CPU Profile — MTP=3 Decode Step Breakdown"
echo " TP=2  ISL=${ISL}  OSL=${OSL}  BS=${BS}  BF16"
echo " Start: $(date)"
echo "============================================================"

# ── Phase 0: install vLLM ────────────────────────────────────────────────────
echo ""
echo ">>> [Phase 0] Install vLLM"
pip install --quiet "vllm>=0.20.1" 2>&1 | tail -5
pip uninstall flash-attn -y 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

# Fix flash_attn import crash on aarch64
python3 /workspace/fix_flash_attn.py 2>/dev/null || true

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
FI_VER=$(python3 -c "import flashinfer; print(flashinfer.__version__)" 2>/dev/null || echo "N/A")
echo "  vLLM: ${VLLM_VER}  PyTorch: ${TORCH_VER}  CUDA: ${CUDA_VER}  FlashInfer: ${FI_VER}"

cat > "${RESULTS_BASE}/env.txt" <<SW
vllm=${VLLM_VER}
pytorch=${TORCH_VER}
cuda=${CUDA_VER}
flashinfer=${FI_VER}
isl=${ISL} osl=${OSL} bs=${BS} tp=2 mtp=3
workload=cpu_profile
timestamp=$(date -Iseconds)
SW

echo ""
echo ">>> GPU info"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
    --format=csv,noheader 2>/dev/null | tee "${RESULTS_BASE}/gpu_info.txt"

# ── NUMA-optimal GPU pair ────────────────────────────────────────────────────
echo ""
echo ">>> Detecting NUMA-optimal GPU pair for TP=2"
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

# ── Helper: serve + bench with CPU profiling ─────────────────────────────────
run_profiled_serve() {
    local tag="$1"; shift
    local extra_args="$*"
    local port=8000
    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  PROFILE: ${tag}"
    echo "  args: ${extra_args}"
    echo "  START: $(date)"
    echo "=========================================="

    # Enable CPU profiling (patch was applied to vLLM source in Phase 1)
    export VLLM_CPU_PROFILE=1
    export VLLM_CPU_PROFILE_WARMUP=10
    export VLLM_CPU_PROFILE_OUT="${outdir}/cpu_profile.json"

    # Launch server (patched vLLM auto-activates profiling via import hook)
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
        [ ${ELAPSED} -ge ${TIMEOUT} ] && echo "  ERROR: timeout" && kill ${SRV_PID} 2>/dev/null && return 1
        sleep 10; ELAPSED=$((ELAPSED+10))
        [ $((ELAPSED % 60)) -eq 0 ] && echo "  [${ELAPSED}s] waiting for server..."
    done
    echo "  Server ready after ${ELAPSED}s"

    # Warmup
    echo "  Warmup (${SERVE_WARMUP} prompts)..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --base-url "http://localhost:${port}" --model "${MODEL}" \
        --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
        --num-prompts ${SERVE_WARMUP} --max-concurrency ${BS} --request-rate inf \
        --ignore-eos > /dev/null 2>&1 || true

    # Main benchmark
    echo "  Main bench (${SERVE_PROMPTS} prompts, OSL=${OSL})..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --base-url "http://localhost:${port}" --model "${MODEL}" \
        --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
        --num-prompts ${SERVE_PROMPTS} --max-concurrency ${BS} --request-rate inf \
        --ignore-eos --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "50,90,99" --save-result \
        --result-dir "${outdir}" --result-filename "serve.json" \
        2>&1 | tee "${outdir}/serve.txt"

    # Graceful shutdown — triggers atexit dump of profiling data
    echo "  Shutting down server (triggers profile dump)..."
    kill -TERM ${SRV_PID} 2>/dev/null || true
    for i in $(seq 1 60); do
        kill -0 ${SRV_PID} 2>/dev/null || break
        sleep 1
    done
    kill -9 ${SRV_PID} 2>/dev/null || true
    wait ${SRV_PID} 2>/dev/null || true
    sleep 2

    # Show results
    echo ""
    echo "  --- CPU Profile Results ---"
    if [ -f "${outdir}/cpu_profile.json" ]; then
        echo "  Profile JSON found: ${outdir}/cpu_profile.json"
        python3 -c "
import json, sys
data = json.load(open('${outdir}/cpu_profile.json'))
print(f'  Steps profiled: {data[\"num_steps\"]}')
print(f'  Regions: {len(data.get(\"stats\", {}))}')
" 2>/dev/null || echo "  (parse failed)"
    else
        echo "  WARNING: cpu_profile.json not found — checking server log"
    fi

    # Extract profile dump from server log (atexit prints to stdout)
    echo ""
    echo "  --- Profile Summary (from server log) ---"
    # The profiler prints between === markers
    sed -n '/vLLM CPU Profile/,/^=\{50,\}/p' "${outdir}/server.log" 2>/dev/null | head -120 || echo "  (not found in log)"
    echo "  ---"
    echo "  END: $(date)"
}

# ── Phase 1: Profile no-MTP baseline ─────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 1: Profile no-MTP baseline"
echo "============================================================"
run_profiled_serve "profile_no_mtp" ""

# ── Phase 2: Profile MTP=3 ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2: Profile MTP=3"
echo "============================================================"
run_profiled_serve "profile_mtp3" \
    "--speculative-config ${MTP_CONFIG}"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " FINAL SUMMARY"
echo "============================================================"

python3 - <<'PYEOF'
import json, os
from pathlib import Path

RD = os.environ.get("RESULTS_DIR", "/workspace/results/cpu_profile")

for tag in ["profile_no_mtp", "profile_mtp3"]:
    print(f"\n{'='*100}")
    print(f"  {tag}")
    print(f"{'='*100}")

    serve_file = Path(RD) / tag / "serve.json"
    if serve_file.exists():
        d = json.load(open(serve_file))
        print(f"  TPOT mean={d.get('mean_tpot_ms',0):.3f}ms  "
              f"P50={d.get('p50_tpot_ms',0):.2f}ms  "
              f"TTFT P50={d.get('p50_ttft_ms',0):.1f}ms  "
              f"E2E={d.get('mean_e2el_ms',0):.1f}ms")
    else:
        print("  (no serve.json)")

    prof_file = Path(RD) / tag / "cpu_profile.json"
    if prof_file.exists():
        data = json.load(open(prof_file))
        stats = data.get("stats", {})
        print(f"  Profiled steps: {data['num_steps']}")

        order = [
            "step_total_us", "",
            "schedule", "execute_model", "sample_tokens", "update_from_output", "",
            "em.update_states", "em.prepare_inputs", "",
            "pi.block_table_commit", "pi.req_indices_cumsum",
            "pi.positions_and_token_indices", "pi.index_select_tokens",
            "pi.attn_metadata", "pi.prev_positions_and_sync",
            "pi.gpu_copies", "pi.spec_decode_section", "",
            "pi.sd.dict_iter", "pi.sd.calc_spec_decode_metadata", "pi.sd.remaining", "",
            "csm.cumsum_arange_1", "csm.np_repeat_logits",
            "csm.bonus_and_draft_cumsum", "csm.np_repeat_target",
            "csm.cpu_to_gpu_copies", "csm.draft_token_ids_gpu",
        ]
        print(f"\n  {'Region':<45} {'Mean(us)':>9} {'Med(us)':>9} {'P90(us)':>9} {'Stdev':>8}")
        print(f"  {'-'*45} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")
        for region in order:
            if region == "":
                print()
                continue
            if region not in stats:
                continue
            s = stats[region]
            indent = "    " if "." in region else "  "
            name = f"{indent}{region}"
            print(f"  {name:<45} {s['mean_us']:>9.1f} {s['median_us']:>9.1f} "
                  f"{s['p90_us']:>9.1f} {s['stdev_us']:>8.1f}")

        # % breakdown
        if "step_total_us" in stats:
            total = stats["step_total_us"]["mean_us"]
            print(f"\n  Step total: {total:.0f}us = {total/1000:.2f}ms")
            for r in ["schedule", "execute_model", "sample_tokens", "update_from_output"]:
                if r in stats:
                    print(f"    {r:<35} {stats[r]['mean_us']:>8.1f}us  "
                          f"{stats[r]['mean_us']/total*100:>5.1f}%")
    else:
        print("  (no cpu_profile.json)")

print(f"\n{'='*100}")
print(f"  Results: {RD}")
print(f"{'='*100}\n")
PYEOF

echo ""
echo "============================================================"
echo " DONE  $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
