#!/usr/bin/env python3
"""
Offline vLLM MTP decode profiler — nsys capture via cudaProfilerApi markers.

Usage:
  nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx \
    --export=sqlite -o output -- \
    python3 offline_mtp_profile.py [--no-mtp] [--warmup N] [--bench N]
"""
import argparse
import time
import torch
import torch.cuda.nvtx as nvtx

parser = argparse.ArgumentParser()
parser.add_argument("--no-mtp", action="store_true")
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--bench", type=int, default=4)
parser.add_argument("--isl", type=int, default=34000)
parser.add_argument("--osl", type=int, default=300)
args = parser.parse_args()

from vllm import LLM, SamplingParams

tag = "no_mtp" if args.no_mtp else "mtp3"
spec_cfg = None if args.no_mtp else {"method": "mtp", "num_speculative_tokens": 3}

print(f"=== Offline MTP Profile [{tag}] ===", flush=True)
print(f"    ISL={args.isl} OSL={args.osl} warmup={args.warmup} bench={args.bench}", flush=True)

llm = LLM(
    model="Qwen/Qwen3.5-35B-A3B",
    dtype="bfloat16",
    tensor_parallel_size=2,
    max_model_len=40960,
    gpu_memory_utilization=0.92,
    speculative_config=spec_cfg,
    trust_remote_code=True,
)

params = SamplingParams(max_tokens=args.osl, temperature=0, ignore_eos=True)
# ~2 tokens per "word " → args.isl // 2 repetitions ≈ isl tokens
prompt = "hello world " * (args.isl // 2)

# ── Warmup (NOT profiled) ──────────────────────────────────────────────────
print(f"Warmup ({args.warmup} prompts)...", flush=True)
for i in range(args.warmup):
    llm.generate([prompt], params)
    print(f"  warmup {i+1}/{args.warmup} done", flush=True)

torch.cuda.synchronize()
print("Starting profiled capture...", flush=True)

# ── Profiled capture ───────────────────────────────────────────────────────
torch.cuda.profiler.start()

for i in range(args.bench):
    nvtx.range_push(f"bench_{i}")
    t0 = time.perf_counter()
    out = llm.generate([prompt], params)
    t1 = time.perf_counter()
    nvtx.range_pop()
    n_tok = len(out[0].outputs[0].token_ids)
    print(f"  bench {i+1}/{args.bench}: {n_tok} tokens  {t1-t0:.3f}s  "
          f"({n_tok/(t1-t0):.1f} tok/s)", flush=True)

torch.cuda.synchronize()
torch.cuda.profiler.stop()
print(f"Profile capture done ({args.bench} prompts).", flush=True)
