#!/usr/bin/env python3
"""
Deep profiler for MTP1 propose_draft sub-components.

Instruments the INTERNAL operations of the propose_draft call to identify
exactly where the +1.01ms overhead comes from. Profiles:

  1. set_inputs_first_pass       — preparing inputs/positions for draft model
  2. build_per_group_and_layer_attn_metadata — building attention metadata
  3. build_model_inputs_first_pass — assembling model forward inputs
  4. execute_model / CUDA graph   — actual draft model forward pass
  5. _greedy_sample / compute_logits — sampling draft tokens from logits
  6. Miscellaneous overhead       — everything else (propose() orchestration)

Features:
  - CPU timing via time.perf_counter() for all sub-components
  - Optional CUDA event timing for GPU-bound ops (model_forward, sample)
  - Accumulates stats, dumps table every N steps (default 50)
  - Clear breakdown with percentages relative to propose_total

Usage:
  python patch_propose_draft_profile.py           # apply patch
  python patch_propose_draft_profile.py --check   # verify
  python patch_propose_draft_profile.py --revert  # print revert info

After patching, enable with:
  VLLM_PROPOSE_PROFILE=1                  # enable profiling
  VLLM_PROPOSE_PROFILE_CUDA_EVENTS=1      # also use CUDA events (optional)
  VLLM_PROPOSE_PROFILE_WARMUP=20          # warmup steps to skip (default 20)
  VLLM_PROPOSE_PROFILE_DUMP_INTERVAL=50   # dump every N steps (default 50)
  VLLM_PROPOSE_PROFILE_OUT=/tmp/propose_profile.json  # output path
"""
import argparse
import sys
import textwrap
from pathlib import Path


def find_vllm_root() -> Path:
    import vllm
    return Path(vllm.__file__).parent


PROFILER_CODE = textwrap.dedent(r'''
"""vLLM MTP1 propose_draft deep sub-component profiler."""
import atexit, json, os, signal, statistics, time, torch
from functools import wraps

ENABLED = os.environ.get("VLLM_PROPOSE_PROFILE", "0") == "1"
USE_CUDA_EVENTS = os.environ.get("VLLM_PROPOSE_PROFILE_CUDA_EVENTS", "0") == "1"
WARMUP = int(os.environ.get("VLLM_PROPOSE_PROFILE_WARMUP", "20"))
DUMP_INTERVAL = int(os.environ.get("VLLM_PROPOSE_PROFILE_DUMP_INTERVAL", "50"))
OUT_PATH = os.environ.get("VLLM_PROPOSE_PROFILE_OUT",
                          "/tmp/propose_profile.json")

_records: list[dict[str, float]] = []
_step_n = 0
_on = False
_patched = False
_device = None

# CUDA event pool (reused to avoid allocation overhead)
_event_pool: list[tuple] = []  # list of (start, end) event pairs
_event_idx = 0
_cuda_results: dict[str, float] = {}


def _get_event_pair():
    """Get a reusable CUDA event pair."""
    global _event_idx
    if _event_idx >= len(_event_pool):
        pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        _event_pool.append(pair)
    p = _event_pool[_event_idx]
    _event_idx += 1
    return p


def _reset_events():
    global _event_idx
    _event_idx = 0


# ── CPU timing helpers ───────────────────────────────────────────────────

def _ns():
    return time.perf_counter_ns() if _on else 0


def _elapsed_us(t0):
    if _on and t0:
        return (time.perf_counter_ns() - t0) / 1e3
    return 0.0


# ── Per-step state ───────────────────────────────────────────────────────

_cur: dict[str, float] = {}


def propose_step_begin():
    """Called at the start of each propose() call."""
    global _step_n, _on, _cur
    _step_n += 1
    if _step_n <= WARMUP:
        _on = False
        return
    _on = True
    _reset_events()
    _cur = {}
    if _device is not None and USE_CUDA_EVENTS:
        torch.cuda.synchronize(_device)
    _cur["__t0"] = time.perf_counter_ns()


def propose_step_end():
    """Called at the end of each propose() call."""
    global _on
    if not _on:
        return

    # Sync to get accurate final time
    if _device is not None:
        torch.cuda.synchronize(_device)

    _cur["propose_total_cpu_us"] = (
        time.perf_counter_ns() - _cur.pop("__t0")) / 1e3

    # Collect CUDA event timings
    if USE_CUDA_EVENTS and _cuda_results:
        _cur.update(_cuda_results)
        _cuda_results.clear()

    # Compute "remaining" = total - sum of known sub-components
    known_cpu = sum(
        _cur.get(k, 0) for k in [
            "set_inputs_cpu_us",
            "build_attn_cpu_us",
            "build_model_inputs_cpu_us",
            "model_forward_cpu_us",
            "compute_logits_cpu_us",
            "sample_cpu_us",
        ]
    )
    _cur["remaining_cpu_us"] = _cur.get("propose_total_cpu_us", 0) - known_cpu

    _records.append(_cur.copy())
    _on = False

    # Auto-dump periodically
    if len(_records) > 0 and len(_records) % DUMP_INTERVAL == 0:
        _dump_table()
        _dump_json()


def _compute_stats(records):
    """Compute statistics for all regions across records."""
    if not records:
        return {}

    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    stats = {}
    for key in sorted(all_keys):
        if key.startswith("__"):
            continue
        vals = [r[key] for r in records if key in r]
        if not vals:
            continue
        sv = sorted(vals)
        n = len(sv)
        stats[key] = {
            "count": n,
            "mean_us": statistics.mean(vals),
            "median_us": statistics.median(vals),
            "p90_us": sv[int(n * 0.9)] if n >= 10 else sv[-1],
            "p99_us": sv[int(n * 0.99)] if n >= 100 else sv[-1],
            "min_us": min(vals),
            "max_us": max(vals),
            "stdev_us": statistics.stdev(vals) if n > 1 else 0,
        }
    return stats


def _dump_table():
    """Print a clear breakdown table to stdout."""
    if not _records:
        print("[propose_profile] No steps recorded.")
        return

    skip = min(5, len(_records) // 4)
    records = _records[skip:] or _records
    stats = _compute_stats(records)

    W = 130
    print(f"\n{'=' * W}")
    print(f"  MTP1 propose_draft Deep Profile -- {len(records)} steps "
          f"(skipped {_step_n - len(records)} warmup/initial)")
    print(f"{'=' * W}")

    # CPU timing breakdown
    cpu_regions = [
        ("propose_total_cpu_us", "TOTAL (CPU wall)"),
        ("set_inputs_cpu_us", "  set_inputs_first_pass"),
        ("build_attn_cpu_us", "  build_attn_metadata"),
        ("build_model_inputs_cpu_us", "  build_model_inputs"),
        ("model_forward_cpu_us", "  model_forward (CUDA graph/eager)"),
        ("compute_logits_cpu_us", "  compute_logits"),
        ("sample_cpu_us", "  _greedy_sample"),
        ("remaining_cpu_us", "  remaining / orchestration"),
    ]

    print(f"\n  --- CPU Timing (time.perf_counter) ---")
    hdr = (f"  {'Component':<45} {'Mean':>9} {'Med':>9} {'P90':>9} "
           f"{'Min':>9} {'Max':>9} {'Std':>8} {'N':>5}  {'%total':>6}")
    print(hdr)
    print(f"  {'-' * 45} {'-' * 9} {'-' * 9} {'-' * 9} "
          f"{'-' * 9} {'-' * 9} {'-' * 8} {'-' * 5}  {'-' * 6}")

    total_mean = stats.get("propose_total_cpu_us", {}).get("mean_us", 1)
    for key, label in cpu_regions:
        if key not in stats:
            continue
        s = stats[key]
        pct = s["mean_us"] / total_mean * 100 if total_mean > 0 else 0
        print(f"  {label:<45} {s['mean_us']:>8.1f}u {s['median_us']:>8.1f}u "
              f"{s['p90_us']:>8.1f}u {s['min_us']:>8.1f}u "
              f"{s['max_us']:>8.1f}u {s['stdev_us']:>7.1f} {s['count']:>5}  "
              f"{pct:>5.1f}%")

    # CUDA event timing (if available)
    cuda_regions = [
        ("model_forward_cuda_ms", "  model_forward (GPU kernel)"),
        ("compute_logits_cuda_ms", "  compute_logits (GPU kernel)"),
        ("sample_cuda_ms", "  sample (GPU kernel)"),
    ]
    has_cuda = any(k in stats for k, _ in cuda_regions)
    if has_cuda:
        print(f"\n  --- CUDA Event Timing (GPU kernel) ---")
        hdr = (f"  {'Component':<45} {'Mean':>9} {'Med':>9} {'P90':>9} "
               f"{'Min':>9} {'Max':>9} {'Std':>8} {'N':>5}")
        print(hdr)
        print(f"  {'-' * 45} {'-' * 9} {'-' * 9} {'-' * 9} "
               f"{'-' * 9} {'-' * 9} {'-' * 8} {'-' * 5}")
        for key, label in cuda_regions:
            if key not in stats:
                continue
            s = stats[key]
            # These are in ms from CUDA events, convert to us for display
            mean_us = s["mean_us"]  # stored as us already
            print(f"  {label:<45} {s['mean_us']:>8.1f}u {s['median_us']:>8.1f}u "
                  f"{s['p90_us']:>8.1f}u {s['min_us']:>8.1f}u "
                  f"{s['max_us']:>8.1f}u {s['stdev_us']:>7.1f} {s['count']:>5}")

    # Percentage bar chart
    print(f"\n  === Breakdown of propose_total "
          f"({total_mean:.0f}us = {total_mean / 1000:.3f}ms) ===")
    for key, label in cpu_regions[1:]:  # skip total
        if key not in stats:
            continue
        s = stats[key]
        pct = s["mean_us"] / total_mean * 100 if total_mean > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {label:<43} {s['mean_us']:>7.0f}us  {pct:>5.1f}%  {bar}")

    # Additional sub-timings
    extra_regions = [
        ("set_inputs.positions_cpu_us", "    positions computation"),
        ("set_inputs.block_table_cpu_us", "    block table update"),
        ("set_inputs.gpu_copies_cpu_us", "    GPU tensor copies"),
        ("build_attn.metadata_cpu_us", "    metadata construction"),
        ("model_forward.dispatch_cpu_us", "    CUDA graph dispatch"),
        ("model_forward.launch_cpu_us", "    kernel launch"),
    ]
    has_extra = any(k in stats for k, _ in extra_regions)
    if has_extra:
        print(f"\n  --- Sub-component Detail ---")
        for key, label in extra_regions:
            if key not in stats:
                continue
            s = stats[key]
            print(f"    {label:<43} {s['mean_us']:>7.0f}us")

    print(f"{'=' * W}\n")


def _dump_json():
    """Dump stats and raw records to JSON."""
    if not _records:
        return
    try:
        skip = min(5, len(_records) // 4)
        records = _records[skip:] or _records
        stats = _compute_stats(records)
        with open(OUT_PATH, "w") as f:
            json.dump({
                "num_steps": len(records),
                "total_steps": _step_n,
                "warmup": WARMUP,
                "cuda_events": USE_CUDA_EVENTS,
                "stats": stats,
                "raw_last_20": records[-20:],
            }, f, indent=2, default=str)
        print(f"[propose_profile] Saved {len(records)} steps -> {OUT_PATH}")
    except Exception as e:
        print(f"[propose_profile] JSON save failed: {e}")


def dump():
    """Final dump on exit."""
    _dump_table()
    _dump_json()


def patch_proposer():
    """Monkey-patch the MTP proposer for deep sub-component profiling."""
    global _patched, _device
    if _patched or not ENABLED:
        return
    _patched = True

    # Import the proposer class
    try:
        from vllm.v1.spec_decode.llm_base_proposer import (
            SpecDecodeBaseProposer as ProposerCls,
        )
    except ImportError:
        try:
            from vllm.v1.spec_decode.llm_base_proposer import (
                LlmBaseProposer as ProposerCls,
            )
        except ImportError:
            print("[propose_profile] ERROR: Could not import proposer class")
            return

    print(f"[propose_profile] Patching {ProposerCls.__name__} for deep profiling")

    # ── Patch propose() — the outer orchestrator ─────────────────────
    _orig_propose = ProposerCls.propose

    @wraps(_orig_propose)
    def _prof_propose(self, *a, **kw):
        global _device
        if _device is None and hasattr(self, 'device'):
            _device = self.device
        elif _device is None:
            # Try to get device from model
            try:
                _device = next(self.model.parameters()).device
            except Exception:
                pass

        propose_step_begin()
        result = _orig_propose(self, *a, **kw)
        propose_step_end()
        return result

    ProposerCls.propose = _prof_propose

    # ── Patch set_inputs_first_pass ──────────────────────────────────
    if hasattr(ProposerCls, 'set_inputs_first_pass'):
        _orig = ProposerCls.set_inputs_first_pass

        @wraps(_orig)
        def _prof(self, *a, **kw):
            t0 = _ns()
            r = _orig(self, *a, **kw)
            _cur["set_inputs_cpu_us"] = _elapsed_us(t0)
            return r

        ProposerCls.set_inputs_first_pass = _prof

    # ── Patch build_per_group_and_layer_attn_metadata ────────────────
    if hasattr(ProposerCls, 'build_per_group_and_layer_attn_metadata'):
        _orig = ProposerCls.build_per_group_and_layer_attn_metadata

        @wraps(_orig)
        def _prof(self, *a, **kw):
            t0 = _ns()
            r = _orig(self, *a, **kw)
            _cur["build_attn_cpu_us"] = _elapsed_us(t0)
            return r

        ProposerCls.build_per_group_and_layer_attn_metadata = _prof

    # ── Patch build_model_inputs_first_pass ──────────────────────────
    if hasattr(ProposerCls, 'build_model_inputs_first_pass'):
        _orig = ProposerCls.build_model_inputs_first_pass

        @wraps(_orig)
        def _prof(self, *a, **kw):
            t0 = _ns()
            r = _orig(self, *a, **kw)
            _cur["build_model_inputs_cpu_us"] = _elapsed_us(t0)
            return r

        ProposerCls.build_model_inputs_first_pass = _prof

    # ── Patch model forward (execute_model or _model_forward) ────────
    # The proposer calls self.model_runner._model_forward() or
    # self.execute_model_part(). We need to find the right method.
    # In the propose() flow, model forward is typically called via
    # self.model_runner or directly. Let's patch the CUDAGraph dispatch
    # or the execute_model_part if it exists.
    if hasattr(ProposerCls, 'execute_model_part'):
        _orig = ProposerCls.execute_model_part

        @wraps(_orig)
        def _prof(self, *a, **kw):
            t0 = _ns()
            if USE_CUDA_EVENTS and _on and _device is not None:
                start_evt, end_evt = _get_event_pair()
                start_evt.record()
            r = _orig(self, *a, **kw)
            if _on:
                if _device is not None:
                    torch.cuda.synchronize(_device)
                _cur["model_forward_cpu_us"] = _elapsed_us(t0)
                if USE_CUDA_EVENTS and _device is not None:
                    end_evt.record()
                    torch.cuda.synchronize(_device)
                    _cuda_results["model_forward_cuda_us"] = (
                        start_evt.elapsed_time(end_evt) * 1000  # ms -> us
                    )
            return r

        ProposerCls.execute_model_part = _prof
    else:
        # Fallback: try patching the model_runner's _model_forward through
        # the proposer. We'll do this at the GPUModelRunner level instead.
        pass

    # ── Patch _greedy_sample ─────────────────────────────────────────
    if hasattr(ProposerCls, '_greedy_sample'):
        _orig = ProposerCls._greedy_sample

        @wraps(_orig)
        def _prof(self, *a, **kw):
            t0 = _ns()
            if USE_CUDA_EVENTS and _on and _device is not None:
                start_evt, end_evt = _get_event_pair()
                start_evt.record()
            r = _orig(self, *a, **kw)
            if _on:
                if _device is not None:
                    torch.cuda.synchronize(_device)
                _cur["sample_cpu_us"] = _elapsed_us(t0)
                if USE_CUDA_EVENTS and _device is not None:
                    end_evt.record()
                    torch.cuda.synchronize(_device)
                    _cuda_results["sample_cuda_us"] = (
                        start_evt.elapsed_time(end_evt) * 1000
                    )
            return r

        ProposerCls._greedy_sample = _prof

    # ── Patch compute_logits on the draft model ──────────────────────
    # compute_logits is on the model object, not the proposer.
    # We patch it lazily on first propose() call.
    _logits_patched = [False]
    _orig_propose_inner = ProposerCls.propose

    @wraps(_orig_propose_inner)
    def _patch_logits_once(self, *a, **kw):
        if not _logits_patched[0]:
            _logits_patched[0] = True
            model = getattr(self, 'model', None)
            if model is None:
                model = getattr(self, 'draft_model', None)
            if model is not None and hasattr(model, 'compute_logits'):
                _orig_cl = model.compute_logits

                @wraps(_orig_cl)
                def _prof_cl(*cl_a, **cl_kw):
                    t0 = _ns()
                    if USE_CUDA_EVENTS and _on and _device is not None:
                        se, ee = _get_event_pair()
                        se.record()
                    r = _orig_cl(*cl_a, **cl_kw)
                    if _on:
                        if _device is not None:
                            torch.cuda.synchronize(_device)
                        _cur["compute_logits_cpu_us"] = _elapsed_us(t0)
                        if USE_CUDA_EVENTS and _device is not None:
                            ee.record()
                            torch.cuda.synchronize(_device)
                            _cuda_results["compute_logits_cuda_us"] = (
                                se.elapsed_time(ee) * 1000
                            )
                    return r

                model.compute_logits = _prof_cl
                print(f"[propose_profile] Patched compute_logits on "
                      f"{type(model).__name__}")

            # Remove this wrapper now that patching is done
            ProposerCls.propose = _prof_propose

        return _prof_propose(self, *a, **kw)

    ProposerCls.propose = _patch_logits_once

    # ── Also try to patch the model_runner level model_forward ───────
    # This catches cases where execute_model_part doesn't exist
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        if not hasattr(ProposerCls, 'execute_model_part'):
            # The proposer calls self.model_runner._model_forward
            # We'll patch it specifically for draft calls
            if hasattr(GPUModelRunner, '_model_forward'):
                _orig_mf = GPUModelRunner._model_forward
                _mf_in_propose = [False]

                @wraps(_orig_mf)
                def _prof_mf(self_runner, **kw):
                    t0 = _ns()
                    if USE_CUDA_EVENTS and _on and _device is not None:
                        se, ee = _get_event_pair()
                        se.record()
                    r = _orig_mf(self_runner, **kw)
                    # Only record if we're inside a propose call
                    if _on and "model_forward_cpu_us" not in _cur:
                        if _device is not None:
                            torch.cuda.synchronize(_device)
                        _cur["model_forward_cpu_us"] = _elapsed_us(t0)
                        if USE_CUDA_EVENTS and _device is not None:
                            ee.record()
                            torch.cuda.synchronize(_device)
                            _cuda_results["model_forward_cuda_us"] = (
                                se.elapsed_time(ee) * 1000
                            )
                    return r

                # Don't patch here -- it would interfere with the main model.
                # Instead, rely on execute_model_part or propose-level timing.
                pass
    except ImportError:
        pass

    # ── Register cleanup ─────────────────────────────────────────────
    atexit.register(dump)

    # Signal handler for graceful shutdown
    def _sig_dump(signum, frame):
        print(f"[propose_profile] Signal {signum}, dumping...")
        dump()

    try:
        signal.signal(signal.SIGUSR1, _sig_dump)
    except (OSError, ValueError):
        pass

    print(f"[propose_profile] Patched {ProposerCls.__name__}. "
          f"warmup={WARMUP}, dump_every={DUMP_INTERVAL}, "
          f"cuda_events={USE_CUDA_EVENTS}, out={OUT_PATH}")


# NOTE: patch_proposer() is called from gpu_model_runner.py at module load,
# NOT at import time (to avoid circular imports).
''')


def apply(root: Path) -> None:
    print(f"vLLM root: {root}")

    # Write the profiler module
    prof_path = root / "v1" / "_propose_profile.py"
    prof_path.write_text(PROFILER_CODE)
    print(f"  [OK] Wrote {prof_path}")

    # Add hook to gpu_model_runner.py (end-of-module, after class definition)
    mr_path = root / "v1" / "worker" / "gpu_model_runner.py"
    mr_src = mr_path.read_text()

    tail_marker = "# Propose-draft profiling hook (end-of-module)"
    if tail_marker in mr_src:
        print("  [SKIP] gpu_model_runner: propose profile hook already present")
    else:
        patch_code = (
            f"\n\n{tail_marker}\n"
            "from vllm.v1._propose_profile import patch_proposer as _propose_patch\n"
            "_propose_patch()\n"
            "del _propose_patch\n"
        )
        mr_src += patch_code
        mr_path.write_text(mr_src)
        print("  [OK] gpu_model_runner: added _propose_profile hook at end of module")

    _invalidate_pycache(mr_path)
    _invalidate_pycache(prof_path)

    # Fix flash_attn import if needed
    _fix_rotary(root)

    print(f"\n  Patch applied. Enable with: VLLM_PROPOSE_PROFILE=1")
    print(f"  For CUDA event timing: VLLM_PROPOSE_PROFILE_CUDA_EVENTS=1")


def _fix_rotary(root: Path):
    p = root / "model_executor/layers/rotary_embedding/common.py"
    if not p.exists():
        return
    txt = p.read_text()
    old = ('        if find_spec("flash_attn") is not None:\n'
           '            from flash_attn.ops.triton.rotary import apply_rotary\n'
           '\n'
           '            self.apply_rotary_emb_flash_attn = apply_rotary')
    new = ('        if find_spec("flash_attn") is not None:\n'
           '            try:\n'
           '                from flash_attn.ops.triton.rotary import apply_rotary\n'
           '                self.apply_rotary_emb_flash_attn = apply_rotary\n'
           '            except (ImportError, OSError):\n'
           '                pass')
    if old in txt:
        txt = txt.replace(old, new, 1)
        p.write_text(txt)
        print("  [OK] rotary_embedding: guarded flash_attn import")


def _invalidate_pycache(py_file: Path):
    """Remove .pyc for the patched file so Python re-compiles it."""
    stem = py_file.stem
    cache_dir = py_file.parent / "__pycache__"
    if cache_dir.exists():
        for pyc in cache_dir.glob(f"{stem}*.pyc"):
            pyc.unlink()
            print(f"  [OK] Removed stale cache: {pyc.name}")


def check(root: Path) -> bool:
    ok = True
    p = root / "v1" / "_propose_profile.py"
    if p.exists() and "propose_profile" in p.read_text().lower():
        print("  [OK] _propose_profile.py exists")
    else:
        print("  [MISSING] _propose_profile.py")
        ok = False

    mr = root / "v1" / "worker" / "gpu_model_runner.py"
    if "Propose-draft profiling hook" in mr.read_text():
        print("  [OK] gpu_model_runner propose profile hook")
    else:
        print("  [MISSING] gpu_model_runner propose profile hook")
        ok = False
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply propose_draft deep profiling patch to vLLM")
    parser.add_argument("--check", action="store_true",
                        help="Verify patch is applied")
    parser.add_argument("--revert", action="store_true",
                        help="Print revert instructions")
    args = parser.parse_args()

    root = find_vllm_root()
    if args.check:
        sys.exit(0 if check(root) else 1)
    elif args.revert:
        print("To revert: pip install --force-reinstall vllm")
        sys.exit(0)
    else:
        apply(root)
        print("\nVerifying...")
        check(root)
