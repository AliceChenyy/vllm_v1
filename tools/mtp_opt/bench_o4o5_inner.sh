#!/usr/bin/env bash
# =============================================================================
# O4+O5 Optimization Benchmark — Qwen3.5-35B-A3B MTP1
# TP=2, ISL=34K, OSL=300, BS=1, BF16
#
# Configs tested:
#   1. no_mtp           — baseline (no spec decode)
#   2. mtp1_base        — MTP1 vanilla (no patches)
#   3. mtp1_o4o5        — MTP1 + O4 (metadata fast path) + O5 (greedy fast)
#   4. mtp1_all         — MTP1 + O1 + O4 + O5 (all applicable patches)
#
# Also runs GSM8K correctness check for patched configs.
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
echo " O4+O5 Optimization Bench — Qwen3.5-35B-A3B"
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
workload=o4o5_bench
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
MTP1_O1='--speculative-config {"method":"mtp","num_speculative_tokens":1,"use_local_argmax_reduction":true}'

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
    if echo "${extra_args}" | grep -q "mtp"; then
        echo "  --- Spec decode acceptance ---"
        grep "SpecDecoding metrics" "${outdir}/server.log" | tail -3
    fi

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

# ── Phase 1: Baseline (no patches) ──────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 1: Baselines (no patches)"
echo "============================================================"

run_serve "no_mtp" ""
run_serve "mtp1_base" "${MTP1_BASE}"

# ── Phase 2: Apply O4+O5 and bench ──────────────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 2: Apply O4+O5 patches"
echo "============================================================"
python3 /workspace/patch_mtp1_opts.py --only o4
python3 /workspace/patch_mtp1_opts.py --only o5

run_serve "mtp1_o4o5" "${MTP1_BASE}"

# ── Phase 3: Apply O1 on top (all patches) ──────────────────────────────────
echo ""
echo "============================================================"
echo " Phase 3: Apply O1 (all = O1+O4+O5)"
echo "============================================================"
python3 /workspace/patch_mtp1_opts.py --only o1

run_serve "mtp1_all_v2" "${MTP1_O1}"

# ── GSM8K Correctness ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " GSM8K Correctness Check"
echo "============================================================"

GSM8K_QUESTIONS='[
  {"prompt":"Janet has 16 chickens. 3 more than twice as many chickens as Janet has are roosters. How many chickens and roosters are there in total?","answer":51},
  {"prompt":"A store sells apples for $2 each and oranges for $3 each. If a customer buys 4 apples and 5 oranges, how much do they pay?","answer":23},
  {"prompt":"A train travels 60 miles per hour. How many miles does it travel in 2.5 hours?","answer":150},
  {"prompt":"A rectangle has a length of 8 and a width of 5. What is its perimeter?","answer":26},
  {"prompt":"If you have 100 coins and give away 37, how many do you have left?","answer":63},
  {"prompt":"A baker makes 12 cookies per batch. How many cookies in 7 batches?","answer":84},
  {"prompt":"What is 15% of 200?","answer":30},
  {"prompt":"A car uses 5 gallons of gas per 100 miles. How many gallons for 350 miles?","answer":17.5},
  {"prompt":"What is the sum of the first 5 positive integers?","answer":15},
  {"prompt":"If a shirt costs $45 and is 20% off, what is the sale price?","answer":36}
]'

for config_tag in "mtp1_o4o5" "mtp1_base"; do
    echo ""
    echo "  --- GSM8K: ${config_tag} ---"
    outdir="${RESULTS_BASE}/${config_tag}"
    port=8000

    # Reinstall if needed
    if [ "${config_tag}" = "mtp1_base" ]; then
        pip install --quiet --upgrade --force-reinstall vllm 2>&1 | tail -3
        pip uninstall flash-attn -y 2>/dev/null || true
        export PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):${PATH}"
        local_args="${MTP1_BASE}"
    else
        local_args="${MTP1_BASE}"
    fi

    CUDA_VISIBLE_DEVICES="${GPU_PAIR}" \
    vllm serve "${MODEL}" \
        --port ${port} --dtype bfloat16 --tensor-parallel-size 2 \
        --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
        ${local_args} \
        > "${outdir}/gsm8k_server.log" 2>&1 &
    SRV_PID=$!

    ELAPSED=0; sleep 5
    until curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        if ! kill -0 ${SRV_PID} 2>/dev/null; then echo "  GSM8K: server died"; break; fi
        [ ${ELAPSED} -ge ${TIMEOUT} ] && break
        sleep 10; ELAPSED=$((ELAPSED+10))
    done

    if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
        python3 -c "
import json, requests, re
questions = json.loads('${GSM8K_QUESTIONS}')
correct = 0
for i, q in enumerate(questions):
    r = requests.post('http://localhost:${port}/v1/completions',
        json={'model':'${MODEL}','prompt':f\"Answer with just a number: {q['prompt']}\",
              'max_tokens':32,'temperature':0})
    text = r.json()['choices'][0]['text'].strip()
    nums = re.findall(r'-?[\d,.]+', text)
    got = float(nums[0].replace(',','')) if nums else None
    ok = got == q['answer']
    if ok: correct += 1
    status = 'OK' if ok else f'WRONG(got={got})'
    print(f'  Q{i+1}: {status}')
print(f'  Score: {correct}/{len(questions)} ({100*correct//len(questions)}%)')
" 2>&1 | tee "${outdir}/gsm8k.txt"
    fi

    kill -TERM ${SRV_PID} 2>/dev/null || true
    for i in $(seq 1 30); do kill -0 ${SRV_PID} 2>/dev/null || break; sleep 1; done
    kill -9 ${SRV_PID} 2>/dev/null || true
    wait ${SRV_PID} 2>/dev/null || true
    sleep 2
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"

python3 - <<'PYEOF'
import json, os
from pathlib import Path

RD = os.environ.get("RESULTS_DIR", os.environ.get("RESULTS_BASE", "/workspace/results"))
order = ["no_mtp", "mtp1_base", "mtp1_o4o5", "mtp1_all_v2"]
W = 100

print(f"\n{'=' * W}")
print(f"  O4+O5 Optimization Results — Qwen3.5-35B-A3B TP=2 BF16 ISL=34K OSL=300")
print(f"{'=' * W}")
print()

base_tpot = None
header = (f"  {'Config':<20} {'TPOT mean':>12} {'±std':>7} {'TPOT P50':>9} "
          f"{'TPOT P99':>9} {'ITL P50':>8} {'TTFT P50':>9} {'E2E mean':>10} "
          f"{'Accept':>7} {'vs base':>8}")
print(header)
print(f"  {'-'*18} {'-'*12} {'-'*7} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*10} {'-'*7} {'-'*8}")

for tag in order:
    sf = Path(RD) / tag / "serve.json"
    if not sf.exists():
        continue
    d = json.load(open(sf))
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

    delta = ""
    if tag == "no_mtp":
        delta = "—"
    elif base_tpot is not None and base_tpot > 0:
        pct = (tpot_mean - base_tpot) / base_tpot * 100
        delta = f"{pct:+.1f}%"

    if tag == "no_mtp":
        base_tpot = tpot_mean

    print(f"  {tag:<20} {tpot_mean:>9.3f}ms {tpot_std:>6.3f} {tpot_p50:>8.2f}ms "
          f"{tpot_p99:>8.2f}ms {itl_p50:>7.2f}ms {ttft_p50:>8.1f}ms {e2e_mean:>9.1f}ms "
          f"{accept:>7} {delta:>8}")

# MTP overhead analysis
no_mtp_f = Path(RD) / "no_mtp" / "serve.json"
if no_mtp_f.exists():
    no_mtp_itl = json.load(open(no_mtp_f)).get("p50_itl_ms", 0)
    if no_mtp_itl > 0:
        print(f"\n  MTP overhead analysis (ITL delta vs no_mtp baseline ITL={no_mtp_itl:.2f}ms):")
        for tag in order[1:]:
            sf = Path(RD) / tag / "serve.json"
            if sf.exists():
                itl = json.load(open(sf)).get("p50_itl_ms", 0)
                if itl > 0:
                    delta = itl - no_mtp_itl
                    print(f"    {tag:<20} ITL={itl:.2f}ms  delta={delta:+.2f}ms  (MTP overhead per step)")

print(f"\n{'=' * W}")
print(f"  Results: {RD}")
print(f"{'=' * W}")
PYEOF

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
