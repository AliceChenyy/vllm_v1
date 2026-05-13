#!/usr/bin/env bash
# =============================================================================
# Strategy D: Deferred Draft Benchmark — Qwen3.5-35B-A3B MTP
# TP=2, ISL=34K, OSL=300, BS=1, BF16
#
# Tests the overlap of draft GPU work with CPU bookkeeping via separate stream.
# Compares with baseline under both async and no-async scheduling modes.
#
# Configs tested:
#   1. mtp_base_async       — Baseline MTP, async ON
#   2. mtp_deferred_async   — Strategy D, async ON
#   3. mtp_base_noasync     — Baseline MTP, async OFF (customer config)
#   4. mtp_deferred_noasync — Strategy D, async OFF (customer config)
# =============================================================================
set -uo pipefail

MTP_TOKENS="${MTP_TOKENS:-1}"  # 1 or 3
TP_SIZE="${TP_SIZE:-2}"
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
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
MODEL="Qwen/Qwen3.5-35B-A3B"

LOG="${RESULTS_BASE}/master.log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo " Strategy D: Deferred Draft Bench — Qwen3.5-35B-A3B"
echo " TP=${TP_SIZE}  ISL=${ISL}  OSL=${OSL}  BS=${BS}  BF16  MTP${MTP_TOKENS}"
echo " Start: $(date)"
echo "============================================================"

# ── Phase 0: Install vLLM ────────────────────────────────────────────────────
echo ""
echo ">>> [Phase 0] Install vLLM"
rm -rf /workspace/.local 2>/dev/null || true
rm -rf /workspace/.cache/flashinfer 2>/dev/null || true
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
isl=${ISL} osl=${OSL} bs=${BS} tp=${TP_SIZE} mtp=${MTP_TOKENS}
workload=deferred_draft
timestamp=$(date -Iseconds)
SW

nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
    --format=csv,noheader 2>/dev/null | tee "${RESULTS_BASE}/gpu_info.txt"

# ── Verify patch compatibility ──────────────────────────────────────────────
echo ""
echo ">>> Verifying patch compatibility..."
python3 /workspace/patch_deferred_draft.py --verify || {
    echo "FATAL: Patch incompatible with this vLLM version"
    exit 1
}

# ── Select GPUs ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Detecting NUMA-optimal GPU set (TP=${TP_SIZE})"
GPU_SET=$(python3 - "${TP_SIZE}" <<'PYEOF'
import subprocess, collections, sys
tp = int(sys.argv[1])
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
selected = None
for n in sorted(groups):
    if len(groups[n]) >= tp: selected = sorted(groups[n])[:tp]; break
if selected is None: selected = sorted(gpu_numa.keys())[:tp]
print(",".join(str(g) for g in selected))
PYEOF
)
GPU_SET="${GPU_SET:-0}"
echo ">>> Using GPUs: ${GPU_SET} (TP=${TP_SIZE})"

MTP_SPEC="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"

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

    CUDA_VISIBLE_DEVICES="${GPU_SET}" \
    vllm serve "${MODEL}" \
        --port ${port} \
        --dtype bfloat16 \
        --tensor-parallel-size ${TP_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image":0,"video":0}' \
        ${EXTRA_VLLM_ARGS} \
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

    # Shutdown — kill server and all child processes (EngineCore, workers)
    kill -TERM ${SRV_PID} 2>/dev/null || true
    for i in $(seq 1 30); do
        kill -0 ${SRV_PID} 2>/dev/null || break; sleep 1
    done
    kill -9 ${SRV_PID} 2>/dev/null || true
    wait ${SRV_PID} 2>/dev/null || true
    # Kill any orphaned vllm/engine processes still holding GPU memory
    pkill -9 -f "vllm.v1.engine.core" 2>/dev/null || true
    pkill -9 -f "multiproc_worker" 2>/dev/null || true
    sleep 3
    # Verify GPU memory is freed
    local gpu_used
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_SET%%,*}" 2>/dev/null | head -1)
    if [ -n "${gpu_used}" ] && [ "${gpu_used}" -gt 5000 ]; then
        echo "  WARNING: GPU still using ${gpu_used}MiB after shutdown, forcing cleanup..."
        pkill -9 -f "python.*vllm" 2>/dev/null || true
        sleep 5
    fi
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

# ── Helper: apply Strategy D patch via vllm startup hook ─────────────────────
# We inject the patch via VLLM_PLUGINS env var or by wrapping the vllm serve cmd.
# Simplest approach: use a wrapper script that imports and patches before serve.

write_serve_wrapper() {
    # Strategy D: inject patch into the installed vLLM source file directly.
    # This ensures worker subprocesses also get the patch.
    # We append a self-applying hook at the end of gpu_model_runner.py.
    local runner_py
    runner_py=$(python3 -c "import vllm.v1.worker.gpu_model_runner as m; print(m.__file__)" 2>&1 | grep '\.py$' | tail -1)
    echo "  Patching ${runner_py}..."

    # Backup original
    cp "${runner_py}" "${runner_py}.orig"

    # Append patch loader at module level — runs when module is imported by worker
    cat >> "${runner_py}" <<'PATCH_HOOK_EOF'

# === Strategy D: deferred draft on separate CUDA stream ===
def _apply_strategy_d_patch():
    import torch
    _draft_streams = {}

    def _get_draft_stream(runner):
        rid = id(runner)
        if rid not in _draft_streams:
            _draft_streams[rid] = torch.cuda.Stream(device=runner.device)
        return _draft_streams[rid]

    _orig_sample_tokens = GPUModelRunner.sample_tokens

    def patched_sample_tokens(self, grammar_output):
        orig_propose = self.propose_draft_token_ids
        orig_copy = self._copy_draft_token_ids_to_cpu
        draft_stream = _get_draft_stream(self)
        draft_event = [None]

        def propose_on_draft_stream(*args, **kwargs):
            default_stream = torch.cuda.current_stream(self.device)
            draft_stream.wait_stream(default_stream)
            with torch.cuda.stream(draft_stream):
                result = orig_propose(*args, **kwargs)
            draft_event[0] = draft_stream.record_event()
            return result

        def copy_with_draft_wait(*args, **kwargs):
            if draft_event[0] is not None and hasattr(self, 'draft_token_ids_copy_stream'):
                if self.draft_token_ids_copy_stream is not None:
                    self.draft_token_ids_copy_stream.wait_event(draft_event[0])
            return orig_copy(*args, **kwargs)

        self.propose_draft_token_ids = propose_on_draft_stream
        self._copy_draft_token_ids_to_cpu = copy_with_draft_wait
        try:
            result = _orig_sample_tokens(self, grammar_output)
        finally:
            self.propose_draft_token_ids = orig_propose
            self._copy_draft_token_ids_to_cpu = orig_copy

        if draft_event[0] is not None:
            default_stream = torch.cuda.current_stream(self.device)
            default_stream.wait_event(draft_event[0])

        return result

    GPUModelRunner.sample_tokens = patched_sample_tokens

_apply_strategy_d_patch()
del _apply_strategy_d_patch
# === End Strategy D patch ===
PATCH_HOOK_EOF

    echo "  [OK] Strategy D patch injected into ${runner_py}"
}

restore_vllm() {
    # Restore original gpu_model_runner.py from backup
    local runner_py
    runner_py=$(python3 -c "import vllm.v1.worker.gpu_model_runner as m; print(m.__file__)" 2>&1 | grep '\.py$' | tail -1)
    if [ -n "${runner_py}" ] && [ -f "${runner_py}.orig" ]; then
        cp "${runner_py}.orig" "${runner_py}"
        echo "  [OK] Restored original ${runner_py}"
    fi
}

run_serve_patched() {
    local tag="$1"; shift
    local extra_args="$*"
    local port=8000
    local outdir="${RESULTS_BASE}/${tag}"
    mkdir -p "${outdir}"

    echo ""
    echo "=========================================="
    echo "  BENCH: ${tag} (Strategy D patched)"
    echo "  extra: ${extra_args}"
    echo "  START: $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${GPU_SET}" \
    vllm serve "${MODEL}" \
        --port ${port} \
        --dtype bfloat16 \
        --tensor-parallel-size ${TP_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code \
        --limit-mm-per-prompt '{"image":0,"video":0}' \
        ${EXTRA_VLLM_ARGS} \
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

    echo "  --- Spec decode acceptance ---"
    grep "SpecDecoding metrics" "${outdir}/server.log" | tail -3

    # Shutdown — kill server and all child processes (EngineCore, workers)
    kill -TERM ${SRV_PID} 2>/dev/null || true
    for i in $(seq 1 30); do
        kill -0 ${SRV_PID} 2>/dev/null || break; sleep 1
    done
    kill -9 ${SRV_PID} 2>/dev/null || true
    wait ${SRV_PID} 2>/dev/null || true
    pkill -9 -f "vllm.v1.engine.core" 2>/dev/null || true
    pkill -9 -f "multiproc_worker" 2>/dev/null || true
    sleep 3
    local gpu_used
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_SET%%,*}" 2>/dev/null | head -1)
    if [ -n "${gpu_used}" ] && [ "${gpu_used}" -gt 5000 ]; then
        echo "  WARNING: GPU still using ${gpu_used}MiB after shutdown, forcing cleanup..."
        pkill -9 -f "python.*vllm" 2>/dev/null || true
        sleep 5
    fi
    echo "  END: $(date)"
}

# ==========================================================================
# Phase 1: Baseline — async ON
# ==========================================================================
echo ""
echo "============================================================"
echo " Phase 1: mtp_base_async (baseline, async ON)"
echo "============================================================"
run_serve "mtp${MTP_TOKENS}_base_async" "${MTP_SPEC}"

# ==========================================================================
# Phase 2: Strategy D — async ON
# ==========================================================================
echo ""
echo "============================================================"
echo " Phase 2: mtp_deferred_async (Strategy D, async ON)"
echo "============================================================"
write_serve_wrapper  # inject patch into vllm source
run_serve_patched "mtp${MTP_TOKENS}_deferred_async" "${MTP_SPEC}"

# ==========================================================================
# Phase 3: Baseline — async OFF (customer config)
# ==========================================================================
echo ""
echo "============================================================"
echo " Phase 3: mtp_base_noasync (baseline, async OFF)"
echo "============================================================"
restore_vllm  # restore original vllm source
run_serve "mtp${MTP_TOKENS}_base_noasync" "${MTP_SPEC} --no-async-scheduling"

# ==========================================================================
# Phase 4: Strategy D — async OFF (customer config)
# ==========================================================================
echo ""
echo "============================================================"
echo " Phase 4: mtp_deferred_noasync (Strategy D, async OFF)"
echo "============================================================"
write_serve_wrapper  # re-inject patch
run_serve_patched "mtp${MTP_TOKENS}_deferred_noasync" "${MTP_SPEC} --no-async-scheduling"

# ==========================================================================
# GSM8K correctness check (on deferred_noasync to validate patch)
# ==========================================================================
echo ""
echo "============================================================"
echo " Phase 5: GSM8K Correctness — Strategy D"
echo "============================================================"

# Start patched server for GSM8K (patch already in source from Phase 4)
CUDA_VISIBLE_DEVICES="${GPU_SET}" \
vllm serve "${MODEL}" \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    ${MTP_SPEC} \
    > "${RESULTS_BASE}/gsm8k_server.log" 2>&1 &
GSM_PID=$!

ELAPSED=0
sleep 5
until curl -sf "http://localhost:8000/v1/models" > /dev/null 2>&1; do
    if ! kill -0 ${GSM_PID} 2>/dev/null; then
        echo "  ERROR: GSM8K server died"
        tail -20 "${RESULTS_BASE}/gsm8k_server.log"
        break
    fi
    [ ${ELAPSED} -ge ${TIMEOUT} ] && echo "  TIMEOUT" && break
    sleep 10; ELAPSED=$((ELAPSED+10))
done

if curl -sf "http://localhost:8000/v1/models" > /dev/null 2>&1; then
    echo "  GSM8K server ready"
    python3 - <<'PYEOF'
import json, re, requests

QUESTIONS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make to buy the wallet?", "5"),
    ("Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", "42"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "624"),
    ("Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in red. Blue flowers are only 25% of the red flowers. How many flowers does Mark have in his garden?", "35"),
    ("Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?", "48"),
    ("Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?", "16"),
    ("Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did she pay for the shoes?", "41"),
    ("Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?", "990"),
]

correct = 0
total = len(QUESTIONS)
for i, (q, expected) in enumerate(QUESTIONS):
    try:
        resp = requests.post("http://localhost:8000/v1/completions",
            json={"model": "Qwen/Qwen3.5-35B-A3B", "prompt": f"Solve: {q}\nAnswer (number only): ",
                  "max_tokens": 256, "temperature": 0}, timeout=120)
        text = resp.json()["choices"][0]["text"].strip()
        numbers = re.findall(r'-?\d+\.?\d*', text)
        answer = numbers[-1] if numbers else ""
        answer = answer.rstrip('0').rstrip('.') if '.' in answer else answer
        ok = answer == expected
        if ok: correct += 1
        print(f"  Q{i+1}: {'OK' if ok else f'WRONG(got={answer})'}")
    except Exception as e:
        print(f"  Q{i+1}: ERROR({e})")
print(f"  Score: {correct}/{total} ({correct*100//total}%)")
PYEOF
fi

kill -TERM ${GSM_PID} 2>/dev/null || true
wait ${GSM_PID} 2>/dev/null || true
pkill -9 -f "vllm.v1.engine.core" 2>/dev/null || true
pkill -9 -f "multiproc_worker" 2>/dev/null || true

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
MTP = os.environ.get("MTP_TOKENS", "1")
order = [
    f"mtp{MTP}_base_async", f"mtp{MTP}_deferred_async",
    f"mtp{MTP}_base_noasync", f"mtp{MTP}_deferred_noasync",
]
W = 130

print(f"\n{'=' * W}")
TP = os.environ.get("TP_SIZE", "2")
print(f"  Strategy D: Deferred Draft — Qwen3.5-35B-A3B TP={TP} BF16 ISL=34K OSL=300 MTP{MTP}")
print(f"{'=' * W}")
print()

results = {}
for tag in order:
    sf = Path(RD) / tag / "serve.json"
    if not sf.exists():
        continue
    results[tag] = json.load(open(sf))

header = (f"  {'Config':<30} {'TPOT mean':>12} {'±std':>7} {'TPOT P50':>9} "
          f"{'TPOT P99':>9} {'ITL P50':>8} {'TTFT P50':>9} {'E2E mean':>10}")
print(header)
print(f"  {'-'*28} {'-'*12} {'-'*7} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*10}")

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
    print(f"  {tag:<30} {tpot_mean:>9.3f}ms {tpot_std:>6.3f} {tpot_p50:>8.2f}ms "
          f"{tpot_p99:>8.2f}ms {itl_p50:>7.2f}ms {ttft_p50:>8.1f}ms {e2e_mean:>9.1f}ms")

# Analysis
print(f"\n  {'─' * 90}")
print(f"  Analysis: Strategy D savings")
print(f"  {'─' * 90}")

for mode, base_tag, opt_tag in [
    ("async ON ", f"mtp{MTP}_base_async",   f"mtp{MTP}_deferred_async"),
    ("async OFF", f"mtp{MTP}_base_noasync", f"mtp{MTP}_deferred_noasync"),
]:
    if base_tag in results and opt_tag in results:
        base_tpot = results[base_tag].get("mean_tpot_ms", 0)
        opt_tpot = results[opt_tag].get("mean_tpot_ms", 0)
        if base_tpot > 0 and opt_tpot > 0:
            tpot_delta = opt_tpot - base_tpot
            tpot_pct = tpot_delta / base_tpot * 100
            base_itl = results[base_tag].get("p50_itl_ms", 0)
            opt_itl = results[opt_tag].get("p50_itl_ms", 0)
            print(f"  {mode}:  TPOT {base_tpot:.3f} → {opt_tpot:.3f}ms  "
                  f"(delta={tpot_delta:+.3f}ms, {tpot_pct:+.1f}%)  "
                  f"ITL P50 {base_itl:.2f} → {opt_itl:.2f}ms")

print(f"\n{'=' * W}")
print(f"  Results: {RD}")
print(f"{'=' * W}")
PYEOF

echo ""
echo "============================================================"
echo " DONE — $(date)"
echo " Results: ${RESULTS_BASE}"
echo "============================================================"
