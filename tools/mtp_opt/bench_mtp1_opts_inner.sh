#!/usr/bin/env bash
# =============================================================================
# MTP1 Optimization Benchmark — Qwen3.5-35B-A3B
# TP=2, ISL=34K, OSL=300, BS=1, BF16
#
# Configs tested:
#   1. no_mtp           — baseline (no spec decode)
#   2. mtp1_base        — MTP1 vanilla
#   3. mtp1_o1          — MTP1 + use_local_argmax_reduction
#   4. mtp1_o1_o2       — MTP1 + local_argmax + FULL CUDA graph
#   5. mtp1_o2          — MTP1 + FULL CUDA graph only
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
RUN_CONFIGS="${RUN_CONFIGS:-all}"

LOG="${RESULTS_BASE}/master.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " MTP1 Optimization Bench — Qwen3.5-35B-A3B"
echo " TP=2  ISL=${ISL}  OSL=${OSL}  BS=${BS}  BF16"
echo " Configs: ${RUN_CONFIGS}"
echo " Start: $(date)"
echo "============================================================"

# ── Phase 0: Install vLLM ────────────────────────────────────────────────────
echo ""
echo ">>> [Phase 0] Install vLLM"
# Nuke stale .local to avoid numpy/torch conflicts with NGC container
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
workload=mtp1_opts
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

# ── Helper: serve + bench ────────────────────────────────────────────────────
run_serve() {
    local tag="$1"; shift
    local extra_args="$*"
    local port=8000
    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  TEST: ${tag}"
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

    # Main bench — 3 runs
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

    # Spec decode metrics
    if grep -q "speculative" <<< "${extra_args}"; then
        curl -sf "http://localhost:${port}/metrics" > "${outdir}/prometheus_metrics.txt" 2>/dev/null || true
        echo "  --- Spec decode acceptance ---"
        grep "acceptance" "${outdir}/server.log" 2>/dev/null | tail -3 || true
    fi

    kill ${SRV_PID} 2>/dev/null || true
    wait ${SRV_PID} 2>/dev/null || true
    sleep 5
    echo "  END: $(date)"
}

should_run() {
    [ "${RUN_CONFIGS}" = "all" ] && return 0
    echo " ${RUN_CONFIGS} " | grep -q " $1 "
}

MTP1_BASE='--speculative-config {"method":"mtp","num_speculative_tokens":1}'
MTP1_O1='--speculative-config {"method":"mtp","num_speculative_tokens":1,"use_local_argmax_reduction":true}'

# =============================================================================
# Phase 1: Baseline no_mtp
# =============================================================================
if should_run "no_mtp"; then
    run_serve "no_mtp" ""
fi

# =============================================================================
# Phase 2: MTP1 baseline (before patches)
# =============================================================================
if should_run "mtp1_base"; then
    run_serve "mtp1_base" "${MTP1_BASE}"
fi

# =============================================================================
# Phase 3: Apply O1 only, test
# =============================================================================
if should_run "mtp1_o1"; then
    echo ""
    echo ">>> Applying O1 patch (use_local_argmax_reduction)..."
    python3 /workspace/patch_mtp1_opts.py --only o1

    run_serve "mtp1_o1" "${MTP1_O1}"
fi

# =============================================================================
# Phase 4: Apply O2 (FULL cudagraph) + O3 (greedy fast path), test O1+O2+O3
# =============================================================================
if should_run "mtp1_all"; then
    echo ""
    echo ">>> Applying O2 patch (FULL CUDA graph)..."
    python3 /workspace/patch_mtp1_opts.py --only o2
    echo ">>> Applying O3 patch (greedy rejection fast path)..."
    python3 /workspace/patch_mtp1_opts.py --only o3

    run_serve "mtp1_all" "${MTP1_O1}"
fi

# =============================================================================
# Phase 5: O3 only (greedy fast path, no local_argmax)
# =============================================================================
if should_run "mtp1_o3"; then
    # O3 already applied from Phase 4 (if ran); apply if not
    python3 /workspace/patch_mtp1_opts.py --only o3 2>/dev/null
    run_serve "mtp1_o3" "${MTP1_BASE}"
fi

# =============================================================================
# GSM8K correctness (quick 10-question check for best config)
# =============================================================================
if should_run "gsm8k"; then
    echo ""
    echo "============================================================"
    echo " GSM8K Correctness Check"
    echo "============================================================"

    for config_tag in "mtp1_o1_o2" "mtp1_base"; do
        echo ""
        echo "  --- GSM8K: ${config_tag} ---"
        local_args="${MTP1_O1}"
        [ "${config_tag}" = "mtp1_base" ] && local_args="${MTP1_BASE}"

        outdir="${RESULTS_BASE}/gsm8k_${config_tag}"
        mkdir -p "${outdir}"

        CUDA_VISIBLE_DEVICES="${GPU_PAIR}" \
        vllm serve "${MODEL}" --port 8000 \
            --dtype bfloat16 --tensor-parallel-size 2 \
            --max-model-len ${MAX_MODEL_LEN} \
            --gpu-memory-utilization ${GPU_MEM_UTIL} \
            --trust-remote-code \
            --limit-mm-per-prompt '{"image":0,"video":0}' \
            ${local_args} \
            > "${outdir}/server.log" 2>&1 &
        SRV_PID=$!

        ELAPSED=0; sleep 5
        until curl -sf "http://localhost:8000/v1/models" > /dev/null 2>&1; do
            [ ${ELAPSED} -ge ${TIMEOUT} ] && break
            sleep 10; ELAPSED=$((ELAPSED+10))
        done

        if curl -sf "http://localhost:8000/v1/models" > /dev/null 2>&1; then
            python3 -c "
import requests, re

QUESTIONS = [
    'Janet buys 3 duck eggs a day. She eats 2 for breakfast and bakes muffins with the rest. She sells muffins for \$2 each. How much does she make per day?',
    'A class has 30 students. 40% are girls. How many boys are there?',
    'Tom has 5 apples and buys 3 more. He gives 2 to his friend. How many does he have?',
    'A train travels 60 mph for 2.5 hours. How far does it go?',
    'Sarah has \$100. She spends 30% on books and 20% on food. How much is left?',
    'A rectangle has length 8 and width 5. What is its perimeter?',
    'If 3x + 7 = 22, what is x?',
    'A store has a 25% off sale. A shirt costs \$40. What is the sale price?',
    'There are 12 eggs in a dozen. How many eggs in 3.5 dozen?',
    'A car uses 5 gallons per 100 miles. How many gallons for 350 miles?',
]
ANSWERS = [2, 18, 6, 150, 50, 26, 5, 30, 42, 17.5]

correct = 0
for i, (q, expected) in enumerate(zip(QUESTIONS, ANSWERS)):
    try:
        resp = requests.post('http://localhost:8000/v1/completions',
            json={'model': '${MODEL}', 'prompt': f'Q: {q}\nA: Let me solve step by step.\n',
                  'max_tokens': 256, 'temperature': 0},
            timeout=60)
        text = resp.json()['choices'][0]['text']
        # Extract last number
        nums = re.findall(r'[-+]?\d*\.?\d+', text.split('####')[-1] if '####' in text else text[-100:])
        if nums:
            got = float(nums[-1])
            ok = abs(got - expected) < 0.1
            correct += ok
            status = 'OK' if ok else f'WRONG(got={got})'
        else:
            status = 'NO_NUMBER'
    except Exception as e:
        status = f'ERROR({e})'
    print(f'  Q{i+1}: {status}')
print(f'  Score: {correct}/10 ({correct*10}%)')
" 2>&1 | tee "${outdir}/gsm8k.txt"
        fi

        kill ${SRV_PID} 2>/dev/null || true
        wait ${SRV_PID} 2>/dev/null || true
        sleep 5
    done
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

print(f"\n{'='*100}")
print(f"  MTP1 Optimization Results — Qwen3.5-35B-A3B TP=2 BF16 ISL=34K OSL=300")
print(f"{'='*100}")

configs = {}
for d in sorted(Path(RD).iterdir()):
    if not d.is_dir() or d.name.startswith("gsm8k"):
        continue
    tag = d.name
    runs = sorted(glob.glob(str(d / "serve_run*.json")))
    tpots_mean, tpots_p50, tpots_p99, itls_p50, ttfts_p50, e2es = [], [], [], [], [], []
    acc_rate = None
    for f in runs:
        try:
            data = json.load(open(f))
            m = data.get("mean_tpot_ms", 0)
            if m > 0:
                tpots_mean.append(m)
                tpots_p50.append(data.get("p50_tpot_ms", 0))
                tpots_p99.append(data.get("p99_tpot_ms", 0))
                itls_p50.append(data.get("p50_itl_ms", 0))
                ttfts_p50.append(data.get("p50_ttft_ms", 0))
                e2es.append(data.get("mean_e2el_ms", 0))
                if data.get("spec_decode_acceptance_rate"):
                    acc_rate = data["spec_decode_acceptance_rate"]
        except:
            pass
    if tpots_mean:
        configs[tag] = {
            "tpot_mean": statistics.mean(tpots_mean),
            "tpot_std": statistics.stdev(tpots_mean) if len(tpots_mean) > 1 else 0,
            "tpot_p50": statistics.mean(tpots_p50),
            "tpot_p99": statistics.mean(tpots_p99),
            "itl_p50": statistics.mean(itls_p50),
            "ttft_p50": statistics.mean(ttfts_p50),
            "e2e_mean": statistics.mean(e2es),
            "acc_rate": acc_rate,
        }

if configs:
    base_tpot = configs.get("no_mtp", {}).get("tpot_mean", 0)
    base_e2e = configs.get("no_mtp", {}).get("e2e_mean", 0)

    print(f"\n  {'Config':<18} {'TPOT mean':>10} {'±std':>7} {'TPOT P50':>9} {'TPOT P99':>9}"
          f" {'ITL P50':>8} {'TTFT P50':>9} {'E2E mean':>10} {'Accept':>7} {'vs base':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*7} {'-'*9} {'-'*9}"
          f" {'-'*8} {'-'*9} {'-'*10} {'-'*7} {'-'*8}")

    order = ["no_mtp", "mtp1_base", "mtp1_o1", "mtp1_o3", "mtp1_all"]
    for tag in order:
        if tag not in configs:
            continue
        c = configs[tag]
        if base_tpot > 0 and tag != "no_mtp":
            delta = (c["tpot_mean"] - base_tpot) / base_tpot * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "—"
        acc_str = f"{c['acc_rate']:.0f}%" if c.get("acc_rate") else "—"
        print(f"  {tag:<18} {c['tpot_mean']:>9.3f}ms {c['tpot_std']:>6.3f}"
              f" {c['tpot_p50']:>8.2f}ms {c['tpot_p99']:>8.2f}ms"
              f" {c['itl_p50']:>7.2f}ms {c['ttft_p50']:>8.1f}ms"
              f" {c['e2e_mean']:>9.1f}ms {acc_str:>7} {delta_str:>8}")

    # ITL breakdown
    base_itl = configs.get("no_mtp", {}).get("itl_p50", 0)
    if base_itl > 0:
        print(f"\n  MTP overhead analysis (ITL delta vs no_mtp baseline ITL={base_itl:.2f}ms):")
        for tag in order[1:]:
            if tag not in configs:
                continue
            c = configs[tag]
            itl_delta = c["itl_p50"] - base_itl
            print(f"    {tag:<18} ITL={c['itl_p50']:.2f}ms  delta={itl_delta:+.2f}ms"
                  f"  (MTP overhead per step)")

print(f"\n{'='*100}")
print(f"  Results: {RD}")
print(f"{'='*100}\n")
PYEOF

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
