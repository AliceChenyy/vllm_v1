#!/usr/bin/env python3
"""
Enhanced CPU profiler for MTP1 decode step breakdown.
Instruments both execute_model AND sample_tokens sub-components
to identify the exact source of MTP1's 2.52ms per-step overhead.

Usage:
  python profile_mtp1_patch.py           # apply patch
  python profile_mtp1_patch.py --check   # verify

After patching, set VLLM_MTP_PROFILE=1 to enable.
Output: /tmp/vllm_mtp_profile.json (or VLLM_MTP_PROFILE_OUT)
"""
import argparse, sys, textwrap
from pathlib import Path


def find_vllm_root() -> Path:
    import vllm
    return Path(vllm.__file__).parent


PROFILER_CODE = textwrap.dedent(r'''
"""vLLM MTP1 decode-step profiler with sub-component breakdown."""
import atexit, json, os, statistics, time, torch
from functools import wraps

ENABLED = os.environ.get("VLLM_MTP_PROFILE", "0") == "1"
WARMUP = int(os.environ.get("VLLM_MTP_PROFILE_WARMUP", "15"))
OUT_PATH = os.environ.get("VLLM_MTP_PROFILE_OUT", "/tmp/vllm_mtp_profile.json")

_records: list[dict[str, float]] = []
_cur: dict[str, float] = {}
_step_n = 0
_on = False
_patched = False
_device = None
AUTO_DUMP_INTERVAL = int(os.environ.get("VLLM_MTP_PROFILE_DUMP_INTERVAL", "50"))


def _ns():
    return time.perf_counter_ns() if _on else 0


def _rec(name, t0):
    if _on and t0:
        _cur[name] = (time.perf_counter_ns() - t0) / 1e3  # us


def _sync_rec(name, t0):
    """Record with CUDA sync to get accurate GPU time."""
    if _on and t0 and _device is not None:
        torch.cuda.synchronize(_device)
        _cur[name] = (time.perf_counter_ns() - t0) / 1e3


def step_begin():
    global _step_n, _on, _cur
    _step_n += 1
    if _step_n <= WARMUP:
        _on = False
        return
    _on = True
    if _device is not None:
        torch.cuda.synchronize(_device)
    _cur = {"__t0": time.perf_counter_ns()}


def step_end():
    global _on
    if not _on:
        return
    if _device is not None:
        torch.cuda.synchronize(_device)
    _cur["step_total_us"] = (time.perf_counter_ns() - _cur.pop("__t0")) / 1e3
    _records.append(_cur)
    _on = False
    # Auto-dump periodically so data survives process crashes
    if len(_records) > 0 and len(_records) % AUTO_DUMP_INTERVAL == 0:
        _auto_dump()


def _auto_dump():
    """Periodic dump — writes JSON silently so data survives crashes."""
    if not _records:
        return
    try:
        skip = min(3, len(_records) // 4)
        records = _records[skip:] or _records
        all_regions = set()
        for r in records:
            all_regions.update(r.keys())
        stats = {}
        for region in sorted(all_regions):
            vals = [r[region] for r in records if region in r]
            if not vals:
                continue
            sv = sorted(vals)
            n = len(sv)
            stats[region] = dict(
                count=n, mean_us=statistics.mean(vals),
                median_us=statistics.median(vals),
                p90_us=sv[int(n * 0.9)] if n >= 10 else sv[-1],
                p99_us=sv[int(n * 0.99)] if n >= 100 else sv[-1],
                min_us=min(vals), max_us=max(vals),
                stdev_us=statistics.stdev(vals) if n > 1 else 0,
            )
        with open(OUT_PATH, "w") as f:
            json.dump({"num_steps": len(records), "total_steps": _step_n,
                       "auto_dump": True, "stats": stats,
                       "raw": records[-20:]}, f, indent=2, default=str)
        print(f"[mtp_profile] Auto-dump: {len(records)} steps -> {OUT_PATH}")
    except Exception as e:
        print(f"[mtp_profile] Auto-dump failed: {e}")


def dump():
    if not _records:
        print("[mtp_profile] No steps recorded.")
        return
    skip = min(3, len(_records) // 4)
    records = _records[skip:] or _records

    all_regions = set()
    for r in records:
        all_regions.update(r.keys())

    stats = {}
    for region in sorted(all_regions):
        vals = [r[region] for r in records if region in r]
        if not vals:
            continue
        sv = sorted(vals)
        n = len(sv)
        stats[region] = dict(
            count=n, mean_us=statistics.mean(vals),
            median_us=statistics.median(vals),
            p90_us=sv[int(n * 0.9)] if n >= 10 else sv[-1],
            p99_us=sv[int(n * 0.99)] if n >= 100 else sv[-1],
            min_us=min(vals), max_us=max(vals),
            stdev_us=statistics.stdev(vals) if n > 1 else 0,
        )

    W = 120
    print(f"\n{'=' * W}")
    print(f"  vLLM MTP Profile — {len(records)} steps "
          f"(skipped {_step_n - len(records)} warmup/initial)")
    print(f"{'=' * W}")

    groups = [
        ("Step total", ["step_total_us"]),
        ("execute_model (target)", [
            "em_total",
            "em.update_states",
            "em.prepare_inputs",
            "em.model_forward",
            "em.compute_logits",
            "em.remaining",
        ]),
        ("sample_tokens", [
            "st_total",
            "st.rejection_sample",
            "st.update_states_after",
            "st.propose_draft",
            "st.copy_draft_to_cpu",
            "st.bookkeeping",
            "st.finalize_kv",
            "st.remaining",
        ]),
        ("prepare_inputs breakdown", [
            "pi.block_table_commit",
            "pi.cumsum_positions",
            "pi.gpu_copies",
            "pi.spec_decode_metadata",
        ]),
        ("propose_draft breakdown", [
            "draft.propose_total",
            "draft.set_inputs",
            "draft.build_attn",
            "draft.build_model_inputs",
            "draft.model_forward",
            "draft.sample",
            "draft.remaining",
        ]),
    ]

    for gname, names in groups:
        present = [r for r in names if r in stats]
        if not present:
            continue
        print(f"\n  --- {gname} ---")
        hdr = (f"  {'Region':<40} {'Mean':>9} {'Med':>9} {'P90':>9} "
               f"{'Min':>9} {'Max':>9} {'Std':>8} {'N':>5}")
        print(hdr)
        print(f"  {'-' * 40} {'-' * 9} {'-' * 9} {'-' * 9} "
              f"{'-' * 9} {'-' * 9} {'-' * 8} {'-' * 5}")
        for r in names:
            if r not in stats:
                continue
            s = stats[r]
            print(f"  {r:<40} {s['mean_us']:>8.1f}u {s['median_us']:>8.1f}u "
                  f"{s['p90_us']:>8.1f}u {s['min_us']:>8.1f}u "
                  f"{s['max_us']:>8.1f}u {s['stdev_us']:>7.1f} {s['count']:>5}")

    # Percentage breakdown
    if "step_total_us" in stats:
        total = stats["step_total_us"]["mean_us"]
        print(f"\n  === % of step_total ({total:.0f}us = {total / 1000:.2f}ms) ===")
        for r in ["em_total", "st_total"]:
            if r in stats:
                pct = stats[r]["mean_us"] / total * 100
                bar = "#" * int(pct / 2)
                print(f"    {r:<36} {stats[r]['mean_us']:>8.0f}us  "
                      f"{pct:>5.1f}%  {bar}")

    if "em_total" in stats:
        em = stats["em_total"]["mean_us"]
        print(f"\n  === % of execute_model ({em:.0f}us = {em / 1000:.2f}ms) ===")
        for r in ["em.update_states", "em.prepare_inputs", "em.model_forward",
                   "em.compute_logits", "em.remaining"]:
            if r in stats:
                pct = stats[r]["mean_us"] / em * 100
                bar = "#" * int(pct / 2)
                print(f"    {r:<36} {stats[r]['mean_us']:>8.0f}us  "
                      f"{pct:>5.1f}%  {bar}")

    if "st_total" in stats:
        st = stats["st_total"]["mean_us"]
        print(f"\n  === % of sample_tokens ({st:.0f}us = {st / 1000:.2f}ms) ===")
        for r in ["st.rejection_sample", "st.update_states_after",
                   "st.propose_draft", "st.bookkeeping",
                   "st.finalize_kv", "st.remaining"]:
            if r in stats:
                pct = stats[r]["mean_us"] / st * 100
                bar = "#" * int(pct / 2)
                print(f"    {r:<36} {stats[r]['mean_us']:>8.0f}us  "
                      f"{pct:>5.1f}%  {bar}")

    if "st.propose_draft" in stats:
        dr = stats["st.propose_draft"]["mean_us"]
        print(f"\n  === % of propose_draft ({dr:.0f}us = {dr / 1000:.2f}ms) ===")
        for r in ["draft.propose_total", "draft.set_inputs", "draft.build_attn",
                   "draft.build_model_inputs", "draft.model_forward",
                   "draft.sample", "draft.remaining"]:
            if r in stats:
                pct = stats[r]["mean_us"] / dr * 100
                bar = "#" * int(pct / 2)
                print(f"    {r:<36} {stats[r]['mean_us']:>8.0f}us  "
                      f"{pct:>5.1f}%  {bar}")

    try:
        with open(OUT_PATH, "w") as f:
            json.dump({"num_steps": len(records), "total_steps": _step_n,
                       "stats": stats, "raw": records[:50]}, f, indent=2,
                      default=str)
        print(f"\n  Saved: {OUT_PATH}")
    except Exception as e:
        print(f"\n  Save failed: {e}")
    print(f"{'=' * W}\n")


def patch_model_runner():
    """Monkey-patch GPUModelRunner for MTP1 profiling."""
    global _patched, _device
    if _patched or not ENABLED:
        return
    _patched = True

    import numpy as np
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # ── Patch execute_model ───────────────────────────────────────────
    _orig_em = GPUModelRunner.execute_model

    @wraps(_orig_em)
    def _prof_em(self, scheduler_output, *a, **kw):
        global _device
        if _device is None:
            _device = self.device
        step_begin()
        t0 = _ns()
        result = _orig_em(self, scheduler_output, *a, **kw)
        _sync_rec("em_total", t0)
        return result

    GPUModelRunner.execute_model = _prof_em

    # ── Patch _update_states ──────────────────────────────────────────
    _orig_us = GPUModelRunner._update_states

    @wraps(_orig_us)
    def _prof_us(self, *a, **kw):
        t0 = _ns()
        r = _orig_us(self, *a, **kw)
        _rec("em.update_states", t0)
        return r

    GPUModelRunner._update_states = _prof_us

    # ── Patch _prepare_inputs ─────────────────────────────────────────
    _orig_pi = GPUModelRunner._prepare_inputs

    @wraps(_orig_pi)
    def _prof_pi(self, *a, **kw):
        t0 = _ns()
        r = _orig_pi(self, *a, **kw)
        _rec("em.prepare_inputs", t0)
        return r

    GPUModelRunner._prepare_inputs = _prof_pi

    # ── Patch _model_forward ──────────────────────────────────────────
    _orig_mf = GPUModelRunner._model_forward

    @wraps(_orig_mf)
    def _prof_mf(self, **kw):
        t0 = _ns()
        r = _orig_mf(self, **kw)
        _sync_rec("em.model_forward", t0)
        return r

    GPUModelRunner._model_forward = _prof_mf

    # ── Patch sample_tokens ───────────────────────────────────────────
    _orig_st = GPUModelRunner.sample_tokens

    @wraps(_orig_st)
    def _prof_st(self, grammar_output=None):
        t_all = _ns()
        # We need fine-grained timing, so inline the original logic
        # with timing points. But that's fragile — instead, we patch
        # the sub-methods and wrap the whole thing.
        result = _orig_st(self, grammar_output)
        _sync_rec("st_total", t_all)
        step_end()
        return result

    GPUModelRunner.sample_tokens = _prof_st

    # ── Patch _sample (rejection sampling) ────────────────────────────
    _orig_sample = GPUModelRunner._sample

    @wraps(_orig_sample)
    def _prof_sample(self, logits, spec_decode_metadata):
        t0 = _ns()
        r = _orig_sample(self, logits, spec_decode_metadata)
        _sync_rec("st.rejection_sample", t0)
        return r

    GPUModelRunner._sample = _prof_sample

    # ── Patch _update_states_after_model_execute ──────────────────────
    _orig_usame = GPUModelRunner._update_states_after_model_execute

    @wraps(_orig_usame)
    def _prof_usame(self, *a, **kw):
        t0 = _ns()
        r = _orig_usame(self, *a, **kw)
        _rec("st.update_states_after", t0)
        return r

    GPUModelRunner._update_states_after_model_execute = _prof_usame

    # ── Patch propose_draft_token_ids ─────────────────────────────────
    _orig_propose = GPUModelRunner.propose_draft_token_ids

    @wraps(_orig_propose)
    def _prof_propose(self, *a, **kw):
        t0 = _ns()
        r = _orig_propose(self, *a, **kw)
        _sync_rec("st.propose_draft", t0)
        return r

    GPUModelRunner.propose_draft_token_ids = _prof_propose

    # ── Patch _bookkeeping_sync ───────────────────────────────────────
    _orig_bk = GPUModelRunner._bookkeeping_sync

    @wraps(_orig_bk)
    def _prof_bk(self, *a, **kw):
        t0 = _ns()
        r = _orig_bk(self, *a, **kw)
        _sync_rec("st.bookkeeping", t0)
        return r

    GPUModelRunner._bookkeeping_sync = _prof_bk

    # ── Patch _copy_draft_token_ids_to_cpu ────────────────────────────
    if hasattr(GPUModelRunner, '_copy_draft_token_ids_to_cpu'):
        _orig_cdtc = GPUModelRunner._copy_draft_token_ids_to_cpu
        @wraps(_orig_cdtc)
        def _prof_cdtc(self, *a, **kw):
            t0 = _ns()
            r = _orig_cdtc(self, *a, **kw)
            _rec("st.copy_draft_to_cpu", t0)
            return r
        GPUModelRunner._copy_draft_token_ids_to_cpu = _prof_cdtc

    # ── Patch finalize_kv_connector (static-like, no self) ──────────
    if hasattr(GPUModelRunner, 'finalize_kv_connector'):
        _orig_fkv = GPUModelRunner.finalize_kv_connector
        @wraps(_orig_fkv)
        def _prof_fkv(*a, **kw):
            t0 = _ns()
            r = _orig_fkv(*a, **kw)
            _rec("st.finalize_kv", t0)
            return r
        GPUModelRunner.finalize_kv_connector = staticmethod(_prof_fkv)

    # ── Patch compute_logits ──────────────────────────────────────────
    # compute_logits is called on the model, not the runner
    # We'll patch it via the runner's model attribute at first call
    _orig_em_inner = GPUModelRunner.execute_model
    _logits_patched = [False]

    @wraps(_orig_em_inner)
    def _patch_logits_once(self, *a, **kw):
        if not _logits_patched[0] and hasattr(self, 'model') and hasattr(self.model, 'compute_logits'):
            _logits_patched[0] = True
            _orig_cl = self.model.compute_logits

            @wraps(_orig_cl)
            def _prof_cl(*cl_a, **cl_kw):
                t0 = _ns()
                r = _orig_cl(*cl_a, **cl_kw)
                _sync_rec("em.compute_logits", t0)
                return r

            self.model.compute_logits = _prof_cl
            # Restore original execute_model (remove this wrapper)
            GPUModelRunner.execute_model = _prof_em
        return _prof_em(self, *a, **kw)

    GPUModelRunner.execute_model = _patch_logits_once

    # ── Patch the proposer's sub-components for draft breakdown ──
    # In vLLM 0.20.2, the base class is SpecDecodeBaseProposer (was LlmBaseProposer)
    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer as _ProposerCls
    except ImportError:
        try:
            from vllm.v1.spec_decode.llm_base_proposer import LlmBaseProposer as _ProposerCls
        except ImportError:
            _ProposerCls = None
            print("[mtp_profile] Could not import proposer class")

    if _ProposerCls is not None:
        print(f"[mtp_profile] Patching {_ProposerCls.__name__} sub-components")

        # Patch set_inputs_first_pass
        if hasattr(_ProposerCls, 'set_inputs_first_pass'):
            _orig_sifp = _ProposerCls.set_inputs_first_pass
            @wraps(_orig_sifp)
            def _prof_sifp(self, *a, **kw):
                t0 = _ns()
                r = _orig_sifp(self, *a, **kw)
                _rec("draft.set_inputs", t0)
                return r
            _ProposerCls.set_inputs_first_pass = _prof_sifp

        # Patch build_per_group_and_layer_attn_metadata
        if hasattr(_ProposerCls, 'build_per_group_and_layer_attn_metadata'):
            _orig_bam = _ProposerCls.build_per_group_and_layer_attn_metadata
            @wraps(_orig_bam)
            def _prof_bam(self, *a, **kw):
                t0 = _ns()
                r = _orig_bam(self, *a, **kw)
                _rec("draft.build_attn", t0)
                return r
            _ProposerCls.build_per_group_and_layer_attn_metadata = _prof_bam

        # Patch _greedy_sample
        if hasattr(_ProposerCls, '_greedy_sample'):
            _orig_gs = _ProposerCls._greedy_sample
            @wraps(_orig_gs)
            def _prof_gs(self, *a, **kw):
                t0 = _ns()
                r = _orig_gs(self, *a, **kw)
                _sync_rec("draft.sample", t0)
                return r
            _ProposerCls._greedy_sample = _prof_gs

        # Patch build_model_inputs_first_pass
        if hasattr(_ProposerCls, 'build_model_inputs_first_pass'):
            _orig_bmifp = _ProposerCls.build_model_inputs_first_pass
            @wraps(_orig_bmifp)
            def _prof_bmifp(self, *a, **kw):
                t0 = _ns()
                r = _orig_bmifp(self, *a, **kw)
                _rec("draft.build_model_inputs", t0)
                return r
            _ProposerCls.build_model_inputs_first_pass = _prof_bmifp

        # Patch propose (the outer orchestrator)
        if hasattr(_ProposerCls, 'propose'):
            _orig_prop = _ProposerCls.propose
            @wraps(_orig_prop)
            def _prof_prop(self, *a, **kw):
                t0 = _ns()
                r = _orig_prop(self, *a, **kw)
                _sync_rec("draft.propose_total", t0)
                return r
            _ProposerCls.propose = _prof_prop

    atexit.register(dump)

    # Register signal handlers so child processes dump on SIGTERM/SIGUSR1
    import signal
    def _sig_dump(signum, frame):
        print(f"[mtp_profile] Signal {signum} received, dumping...")
        dump()
    try:
        signal.signal(signal.SIGUSR1, _sig_dump)
    except (OSError, ValueError):
        pass  # can't set signal handler in non-main thread

    print(f"[mtp_profile] Patched. warmup={WARMUP}, auto_dump_every={AUTO_DUMP_INTERVAL}, out={OUT_PATH}")


# NOTE: patch_model_runner() must NOT be called at module level here
# because this module is imported FROM gpu_model_runner.py (circular import).
# Instead, gpu_model_runner.py calls patch_model_runner() at the END of its
# module body, after GPUModelRunner is fully defined.
''')


def apply(root: Path) -> None:
    print(f"vLLM root: {root}")

    prof_path = root / "v1" / "_mtp_profile.py"
    prof_path.write_text(PROFILER_CODE)
    print(f"  [OK] Wrote {prof_path}")

    mr_path = root / "v1" / "worker" / "gpu_model_runner.py"
    mr_src = mr_path.read_text()

    # Add patch call at END of module (after GPUModelRunner is defined)
    # to avoid circular import
    tail_marker = "# MTP profiling hook (end-of-module)"
    if tail_marker in mr_src:
        print("  [SKIP] gpu_model_runner: patch hook already present")
    else:
        patch_code = (
            f"\n\n{tail_marker}\n"
            "from vllm.v1._mtp_profile import patch_model_runner as _mtp_patch\n"
            "_mtp_patch()\n"
            "del _mtp_patch\n"
        )
        mr_src += patch_code
        mr_path.write_text(mr_src)
        print("  [OK] gpu_model_runner: added _mtp_profile hook at end of module")

    # Fix flash_attn import
    _fix_rotary(root)
    print("\n  Patch applied. Enable with: VLLM_MTP_PROFILE=1")


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


def check(root: Path) -> bool:
    ok = True
    p = root / "v1" / "_mtp_profile.py"
    if p.exists() and "mtp_profile" in p.read_text().lower():
        print("  [OK] _mtp_profile.py exists")
    else:
        print("  [MISSING] _mtp_profile.py")
        ok = False

    mr = root / "v1" / "worker" / "gpu_model_runner.py"
    if "MTP profiling hook" in mr.read_text():
        print("  [OK] gpu_model_runner hook")
    else:
        print("  [MISSING] gpu_model_runner hook")
        ok = False
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = find_vllm_root()
    if args.check:
        sys.exit(0 if check(root) else 1)
    else:
        apply(root)
        print("\nVerifying...")
        check(root)
