#!/usr/bin/env bash
# Quick Strategy D validation — baseline vs patched, 5 prompts each
set -uo pipefail

export HOME="/workspace"
export HF_HOME="/workspace/.cache/huggingface"
export XDG_CACHE_HOME="/workspace/.cache"
export FLASHINFER_WORKSPACE_DIR="/workspace/.cache/flashinfer"
export VLLM_CACHE_DIR="/workspace/.cache/vllm"
export TRITON_CACHE_DIR="/workspace/.cache/triton"
export TORCH_HOME="/workspace/.cache/torch"
mkdir -p "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_DIR}" "${VLLM_CACHE_DIR}" \
         "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${HF_HOME}"

RESULTS="/workspace/results/quick_deferred_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS}"
LOG="${RESULTS}/test.log"
exec > >(tee -a "${LOG}") 2>&1

MODEL="Qwen/Qwen3.5-35B-A3B"
ISL=34000; OSL=300; PROMPTS=5; WARMUP=3
TP="${TP_SIZE:-1}"
EXTRA="${EXTRA_VLLM_ARGS:-}"
MTP_SPEC='--speculative-config {"method":"mtp","num_speculative_tokens":1}'

echo "=== Quick Strategy D Test === $(date)"
echo "TP=${TP} ISL=${ISL} OSL=${OSL} ${EXTRA}"

# Install
rm -rf /workspace/.local /workspace/.cache/flashinfer 2>/dev/null || true
pip install --quiet --upgrade vllm 2>&1 | tail -3
pip uninstall flash-attn -y 2>/dev/null || true
export PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):${PATH}"
echo "vLLM: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# Get runner path
RUNNER=$(python3 -c "import vllm.v1.worker.gpu_model_runner as m; print(m.__file__)" 2>&1 | grep '\.py$' | tail -1)
echo "Runner: ${RUNNER}"
cp "${RUNNER}" "${RUNNER}.orig"

run_bench() {
    local tag="$1"; shift
    local args="$*"
    echo ""
    echo ">>> ${tag} — $(date)"
    mkdir -p "${RESULTS}/${tag}"

    CUDA_VISIBLE_DEVICES=0 vllm serve "${MODEL}" \
        --port 8000 --dtype bfloat16 --tensor-parallel-size ${TP} \
        --max-model-len 40960 --gpu-memory-utilization 0.92 --trust-remote-code \
        --limit-mm-per-prompt '{"image":0,"video":0}' \
        ${EXTRA} ${args} \
        > "${RESULTS}/${tag}/server.log" 2>&1 &
    local PID=$!

    local ELAPSED=0
    sleep 5
    until curl -sf "http://localhost:8000/v1/models" > /dev/null 2>&1; do
        kill -0 ${PID} 2>/dev/null || { echo "  DIED"; tail -20 "${RESULTS}/${tag}/server.log"; return 1; }
        [ ${ELAPSED} -ge 1800 ] && { echo "  TIMEOUT"; kill ${PID} 2>/dev/null; return 1; }
        sleep 10; ELAPSED=$((ELAPSED+10))
        [ $((ELAPSED % 120)) -eq 0 ] && echo "  [${ELAPSED}s] waiting..."
    done
    echo "  Ready after ${ELAPSED}s"

    echo "  Warmup ${WARMUP}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --base-url http://localhost:8000 --model "${MODEL}" \
        --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
        --num-prompts ${WARMUP} --max-concurrency 1 --request-rate inf \
        --ignore-eos > /dev/null 2>&1 || true

    echo "  Bench ${PROMPTS}..."
    python3 -m vllm.entrypoints.cli.main bench serve \
        --base-url http://localhost:8000 --model "${MODEL}" \
        --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
        --num-prompts ${PROMPTS} --max-concurrency 1 --request-rate inf \
        --ignore-eos --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "50,90,99" --save-result \
        --result-dir "${RESULTS}/${tag}" --result-filename "serve.json" \
        2>&1 | tee "${RESULTS}/${tag}/serve.txt"

    kill -TERM ${PID} 2>/dev/null; sleep 5
    kill -9 ${PID} 2>/dev/null; wait ${PID} 2>/dev/null || true
    # Kill orphaned EngineCore/worker processes still holding GPU memory
    pkill -9 -f "vllm.v1.engine.core" 2>/dev/null || true
    pkill -9 -f "multiproc_worker" 2>/dev/null || true
    sleep 3
    # Verify GPU memory is freed before next run
    local gpu_used
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
    if [ -n "${gpu_used}" ] && [ "${gpu_used}" -gt 5000 ]; then
        echo "  WARNING: GPU still using ${gpu_used}MiB, forcing cleanup..."
        pkill -9 -f "python.*vllm" 2>/dev/null || true
        sleep 5
    fi
}

inject_patch() {
    echo ">>> Injecting Strategy D patch..."
    cp "${RUNNER}.orig" "${RUNNER}"
    cat >> "${RUNNER}" <<'PATCH_EOF'

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
PATCH_EOF
    echo "  [OK] Patch injected"
}

restore_original() {
    echo ">>> Restoring original vLLM..."
    cp "${RUNNER}.orig" "${RUNNER}"
    echo "  [OK] Restored"
}

# === Phase 1: Baseline ===
run_bench "baseline" "${MTP_SPEC}"

# === Phase 2: Strategy D ===
inject_patch
run_bench "deferred" "${MTP_SPEC}"

# === Summary ===
echo ""
echo "=== SUMMARY ==="
python3 - <<'PYEOF'
import json, os
from pathlib import Path
RD = os.environ.get("RESULTS", "/workspace/results")
# Find latest results dir
for tag in ["baseline", "deferred"]:
    sf = Path(RD) / tag / "serve.json"
    if sf.exists():
        d = json.load(open(sf))
        tpot = d.get("mean_tpot_ms", 0)
        itl = d.get("p50_itl_ms", 0)
        ttft = d.get("p50_ttft_ms", 0)
        print(f"  {tag:<15} TPOT={tpot:.3f}ms  ITL_P50={itl:.2f}ms  TTFT_P50={ttft:.1f}ms")
    else:
        print(f"  {tag:<15} NO RESULT")
PYEOF

restore_original
echo "=== DONE === $(date)"
