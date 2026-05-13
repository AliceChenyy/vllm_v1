#!/usr/bin/env bash
# =============================================================================
# P0 CPU Overhead — nsys offline profiling
# Configs: no_mtp, mtp3_patched (P0 patch applied)
# Captures GPU kernel timeline + inter-kernel gaps via cudaProfilerApi markers
# =============================================================================
set -uo pipefail

RESULTS_BASE="${RESULTS_DIR:-/workspace/results/p0_nsys}"
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
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
WARMUP="${WARMUP:-2}"
BENCH_PROMPTS="${BENCH_PROMPTS:-4}"

LOG="${RESULTS_BASE}/master.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " P0 Nsys Profile  ISL=${ISL}  OSL=${OSL}  BF16  TP=2"
echo " warmup=${WARMUP}  bench_prompts=${BENCH_PROMPTS}"
echo " Start: $(date)"
echo "============================================================"

# ── Phase 0: install vllm ────────────────────────────────────────
echo ""
echo ">>> [Phase 0] Install vLLM"
pip install --quiet "vllm==0.20.2" 2>&1 | tail -3
pip uninstall flash-attn -y 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
python3 /workspace/fix_flash_attn.py

VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
echo "  vLLM: ${VLLM_VER}  PyTorch: ${TORCH_VER}"

# ── Find nsys ─────────────────────────────────────────────────────
NSYS_BIN=$(which nsys 2>/dev/null \
    || ls /usr/local/cuda/bin/nsys 2>/dev/null \
    || ls /opt/nvidia/nsight-systems-cli/*/bin/nsys 2>/dev/null | tail -1 \
    || echo "")
if [ -z "${NSYS_BIN}" ]; then
    echo "ERROR: nsys not found in PATH or common locations"; exit 1
fi
NSYS_VER=$("${NSYS_BIN}" --version 2>&1 | head -1)
echo "  nsys: ${NSYS_VER}"
echo "  nsys path: ${NSYS_BIN}"

# ── GPU pair ─────────────────────────────────────────────────────
GPU_PAIR=$(python3 - <<'PYEOF'
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
print(f"{pair[0]},{pair[1]}")
PYEOF
)
GPU_PAIR="${GPU_PAIR:-0,1}"
echo "  GPUs: ${GPU_PAIR}"

# ── Apply P0 patch ────────────────────────────────────────────────
echo ""
echo ">>> [Phase 1] Apply P0 patch"
python3 /workspace/apply_p0_opt_patch.py
python3 /workspace/apply_p0_opt_patch.py --check

# ── Helper ────────────────────────────────────────────────────────
run_nsys() {
    local tag="$1"; shift
    local py_extra="$*"
    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  NSYS PROFILE: ${tag}"
    echo "  py_args: ${py_extra}"
    echo "  START: $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${GPU_PAIR}" \
    "${NSYS_BIN}" profile \
        --capture-range=cudaProfilerApi \
        --trace=cuda,nvtx \
        --output="${outdir}/nsys_profile" \
        --force-overwrite=true \
        --export=sqlite \
        -- \
        python3 /workspace/offline_mtp_profile.py \
            --isl ${ISL} --osl ${OSL} \
            --warmup ${WARMUP} --bench ${BENCH_PROMPTS} \
            ${py_extra} \
        2>&1 | tee "${outdir}/profile_run.log"

    local rc=${PIPESTATUS[0]}
    echo "  nsys exit: ${rc}"

    # Generate text stats from the .nsys-rep
    local rep="${outdir}/nsys_profile.nsys-rep"
    if [ -f "${rep}" ]; then
        echo "  Generating nsys stats reports..."
        "${NSYS_BIN}" stats "${rep}" \
            --report gpukernsum,cudaapisum,nvtxsum \
            --format csv \
            --output "${outdir}/" \
            2>&1 | tee "${outdir}/nsys_stats.log" || true

        # Also dump a human-readable summary
        "${NSYS_BIN}" stats "${rep}" \
            --report gpukernsum \
            --format column \
            2>&1 | head -60 | tee "${outdir}/gpukernsum.txt" || true
    else
        echo "  WARN: ${rep} not found — profile may have failed"
        echo "  Last 20 lines of log:"
        tail -20 "${outdir}/profile_run.log" || true
    fi

    echo "  END: $(date)"
}

# ── Profile both configs ──────────────────────────────────────────
run_nsys "no_mtp"       "--no-mtp"
run_nsys "mtp3_patched" ""

# ── Gap analysis ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " GAP ANALYSIS"
echo "============================================================"
python3 /workspace/analyze_nsys_gaps.py \
    --results-dir "${RESULTS_BASE}" \
    2>&1 | tee "${RESULTS_BASE}/gap_analysis.txt" \
    || echo "  (analysis script failed — check SQLite availability)"

echo ""
echo "============================================================"
echo " DONE  $(date)"
echo " Results: ${RESULTS_BASE}"
echo " nsys files:"
find "${RESULTS_BASE}" -name "*.nsys-rep" -o -name "*.sqlite" 2>/dev/null | sort
echo "============================================================"
