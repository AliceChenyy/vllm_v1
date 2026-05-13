#!/usr/bin/env python3
"""
Apply CPU profiling patch to installed vLLM.
Writes _cpu_profile.py into the vllm package and adds one import line
to gpu_model_runner.py. Works across worker processes (TP>1).

Usage:
  python apply_cpu_profile_patch.py           # apply
  python apply_cpu_profile_patch.py --check   # verify
  python apply_cpu_profile_patch.py --revert  # undo (reinstall vllm)

After patching, set VLLM_CPU_PROFILE=1 to enable.
"""
import argparse, sys, textwrap
from pathlib import Path


def find_vllm_root() -> Path:
    import vllm
    return Path(vllm.__file__).parent


# ─── The profiler + monkey-patch module ───────────────────────────────────────
PROFILER_CODE = textwrap.dedent(r'''
"""vLLM CPU decode-step profiler. Auto-patches GPUModelRunner on import."""
import atexit, json, os, statistics, time
from functools import wraps

ENABLED = os.environ.get("VLLM_CPU_PROFILE", "0") == "1"
WARMUP = int(os.environ.get("VLLM_CPU_PROFILE_WARMUP", "10"))
OUT_PATH = os.environ.get("VLLM_CPU_PROFILE_OUT", "/tmp/vllm_cpu_profile.json")

_step_records: list[dict[str, float]] = []
_cur: dict[str, float] = {}
_step_n = 0
_on = False
_patched = False

def _t():
    return time.perf_counter_ns() if _on else 0

def _rec(name, t0):
    if _on and t0:
        _cur[name] = (time.perf_counter_ns() - t0) / 1000  # us

def step_begin():
    global _step_n, _on, _cur
    _step_n += 1
    if _step_n <= WARMUP:
        _on = False; return
    _on = True
    _cur = {"__t0": time.perf_counter_ns()}

def step_end():
    global _on
    if not _on: return
    _cur["step_total_us"] = (time.perf_counter_ns() - _cur["__t0"]) / 1000
    _step_records.append(_cur)
    _on = False


def dump():
    if not _step_records:
        print("[cpu_profile] No steps recorded."); return
    skip = min(2, len(_step_records) // 4)
    records = _step_records[skip:] or _step_records
    all_regions = set()
    for r in records:
        all_regions.update(k for k in r if not k.startswith("__"))

    stats = {}
    for region in sorted(all_regions):
        vals = [r[region] for r in records if region in r]
        if not vals: continue
        sv = sorted(vals)
        stats[region] = dict(
            count=len(vals), mean_us=statistics.mean(vals),
            median_us=statistics.median(vals),
            p90_us=sv[int(len(sv)*0.9)] if len(sv)>=10 else sv[-1],
            p99_us=sv[int(len(sv)*0.99)] if len(sv)>=100 else sv[-1],
            min_us=min(vals), max_us=max(vals),
            stdev_us=statistics.stdev(vals) if len(vals)>1 else 0,
        )

    W = 110
    print(f"\n{'='*W}")
    print(f"  vLLM CPU Profile — {len(records)} steps "
          f"(skipped {_step_n - len(records)} warmup/initial)")
    print(f"{'='*W}")

    groups = [
        ("Top-level", [
            "step_total_us", "execute_model", "sample_tokens",
        ]),
        ("execute_model breakdown", [
            "em.update_states", "em.prepare_inputs",
            "em.determine_batch", "em.build_attn", "em.preprocess",
            "em.model_forward", "em.postprocess",
        ]),
        ("_prepare_inputs breakdown", [
            "pi.block_table_commit", "pi.req_indices_cumsum",
            "pi.positions_token_indices", "pi.index_select",
            "pi.attn_metadata", "pi.prev_pos_sync",
            "pi.gpu_copies", "pi.spec_decode",
        ]),
        ("spec_decode breakdown", [
            "pi.sd.dict_iter", "pi.sd.calc_metadata", "pi.sd.copy_draft_tokens",
        ]),
        ("_calc_spec_decode_metadata", [
            "csm.cumsum1", "csm.repeat_logits", "csm.bonus_draft_cumsum",
            "csm.repeat_target", "csm.to_gpu", "csm.draft_ids_gpu",
        ]),
    ]

    for gname, names in groups:
        present = [r for r in names if r in stats]
        if not present: continue
        print(f"\n  --- {gname} ---")
        hdr = f"  {'Region':<42} {'Mean':>8} {'Med':>8} {'P90':>8} {'Min':>8} {'Max':>8} {'Std':>7} {'N':>5}"
        print(hdr)
        print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*5}")
        for r in names:
            if r not in stats: continue
            s = stats[r]
            print(f"  {r:<42} {s['mean_us']:>8.1f} {s['median_us']:>8.1f} "
                  f"{s['p90_us']:>8.1f} {s['min_us']:>8.1f} {s['max_us']:>8.1f} "
                  f"{s['stdev_us']:>7.1f} {s['count']:>5}")

    # Percentage breakdown
    if "step_total_us" in stats:
        total = stats["step_total_us"]["mean_us"]
        print(f"\n  === % of step_total ({total:.0f}us = {total/1000:.2f}ms) ===")
        for r in ["execute_model", "sample_tokens"]:
            if r in stats:
                pct = stats[r]["mean_us"] / total * 100
                print(f"    {r:<38} {stats[r]['mean_us']:>8.1f}us  {pct:>5.1f}%  {'#'*int(pct/2)}")

    if "execute_model" in stats:
        em = stats["execute_model"]["mean_us"]
        print(f"\n  === % of execute_model ({em:.0f}us = {em/1000:.2f}ms) ===")
        for r in ["em.update_states", "em.prepare_inputs", "em.determine_batch",
                   "em.build_attn", "em.preprocess", "em.model_forward", "em.postprocess"]:
            if r in stats:
                pct = stats[r]["mean_us"] / em * 100
                print(f"    {r:<38} {stats[r]['mean_us']:>8.1f}us  {pct:>5.1f}%  {'#'*int(pct/2)}")

    if "em.prepare_inputs" in stats:
        pi = stats["em.prepare_inputs"]["mean_us"]
        print(f"\n  === % of prepare_inputs ({pi:.0f}us = {pi/1000:.2f}ms) ===")
        for r in ["pi.block_table_commit", "pi.req_indices_cumsum",
                   "pi.positions_token_indices", "pi.index_select",
                   "pi.attn_metadata", "pi.prev_pos_sync",
                   "pi.gpu_copies", "pi.spec_decode"]:
            if r in stats:
                pct = stats[r]["mean_us"] / pi * 100
                print(f"    {r:<38} {stats[r]['mean_us']:>8.1f}us  {pct:>5.1f}%  {'#'*int(pct/2)}")

    try:
        with open(OUT_PATH, "w") as f:
            json.dump({"num_steps": len(records), "total_steps": _step_n,
                        "stats": stats, "raw": records[:30]}, f, indent=2, default=str)
        print(f"\n  Saved: {OUT_PATH}")
    except Exception as e:
        print(f"\n  Save failed: {e}")
    print(f"{'='*W}\n")


def patch_model_runner():
    """Monkey-patch GPUModelRunner methods. Called once per process."""
    global _patched
    if _patched or not ENABLED:
        return
    _patched = True

    import numpy as np
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # ── Patch execute_model ───────────────────────────────────────────────
    _orig_em = GPUModelRunner.execute_model
    @wraps(_orig_em)
    def _prof_em(self, scheduler_output, *a, **kw):
        step_begin()
        t0 = _t()
        result = _orig_em(self, scheduler_output, *a, **kw)
        _rec("execute_model", t0)
        return result
    GPUModelRunner.execute_model = _prof_em

    # ── Patch sample_tokens ───────────────────────────────────────────────
    _orig_st = GPUModelRunner.sample_tokens
    @wraps(_orig_st)
    def _prof_st(self, *a, **kw):
        t0 = _t()
        result = _orig_st(self, *a, **kw)
        _rec("sample_tokens", t0)
        step_end()
        return result
    GPUModelRunner.sample_tokens = _prof_st

    # ── Patch _update_states ──────────────────────────────────────────────
    _orig_us = GPUModelRunner._update_states
    @wraps(_orig_us)
    def _prof_us(self, *a, **kw):
        t0 = _t()
        r = _orig_us(self, *a, **kw)
        _rec("em.update_states", t0)
        return r
    GPUModelRunner._update_states = _prof_us

    # ── Patch _prepare_inputs (full rewrite with sub-timing) ──────────────
    def _prof_pi(self, scheduler_output, num_scheduled_tokens):
        t_all = _t()
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # block_table commit
        t0 = _t()
        self.input_batch.block_table.commit_block_table(num_reqs)
        _rec("pi.block_table_commit", t0)

        # req_indices + cumsum
        t0 = _t()
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
        _rec("pi.req_indices_cumsum", t0)

        # positions + token_indices
        t0 = _t()
        positions_np = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[:cu_num_tokens[-1]]
        )
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)
        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        token_indices_tensor = torch.from_numpy(token_indices)
        _rec("pi.positions_token_indices", t0)

        # index_select
        t0 = _t()
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(), 0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            torch.index_select(
                self.input_batch.is_token_ids_tensor.flatten(), 0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                ns = num_scheduled_tokens[req_idx]
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += ns; continue
                if ns <= 0:
                    output_idx += ns; continue
                re = self.input_batch.req_prompt_embeds[req_idx]
                sp = self.input_batch.num_computed_tokens_cpu[req_idx]
                if sp >= re.shape[0]:
                    output_idx += ns; continue
                ep = min(sp + ns, re.shape[0])
                an = ep - sp
                if an > 0:
                    self.inputs_embeds.cpu[output_idx:output_idx+an].copy_(re[sp:ep])
                output_idx += ns
        _rec("pi.index_select", t0)

        # attn metadata
        t0 = _t()
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1:num_reqs+1] = cu_num_tokens
        self.query_start_loc.np[num_reqs+1:].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[:num_reqs+1]
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
        _rec("pi.attn_metadata", t0)

        # prev_positions + accepted_tokens sync
        t0 = _t()
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)
        num_tokens_list = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens_list, dtype=np.int32)
        self.discard_request_mask.np[:num_reqs] = (
            self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)
        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[np.where(new_mask, 0, prev_idx)]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        else:
            self.num_accepted_tokens.np.fill(1)
            self.num_accepted_tokens.gpu.fill_(1)
        _rec("pi.prev_pos_sync", t0)

        # GPU copies
        t0 = _t()
        if (self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index):
            self.prev_positions.copy_to_gpu(num_reqs)
            self.prev_num_draft_tokens.copy_to_gpu()
            cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                device=self.device, non_blocking=True)
            from vllm.v1.worker.gpu.input_batch import update_num_computed_tokens_for_batch_change
            update_num_computed_tokens_for_batch_change(
                self.num_computed_tokens,
                self.num_accepted_tokens.gpu[:num_reqs],
                self.prev_positions.gpu[:num_reqs],
                self.valid_sampled_token_count_gpu,
                self.prev_num_draft_tokens.gpu, cpu_values,
            )
        else:
            self.num_computed_tokens[:num_reqs].copy_(
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                non_blocking=True)

        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]
        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )
        self.seq_lens[:num_reqs] = self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        self.seq_lens[num_reqs:].fill_(0)
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs, self.query_start_loc.gpu[:num_reqs+1],
            self.positions[:total_num_scheduled_tokens],
        )
        self._prepare_input_ids(
            scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens,
        )
        if self.uses_mrope:
            self.mrope_positions.gpu[:,:total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:,:total_num_scheduled_tokens], non_blocking=True)
        elif self.uses_xdrope_dim > 0:
            self.xdrope_positions.gpu[:,:total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:,:total_num_scheduled_tokens], non_blocking=True)
        if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
            drift = self.num_computed_tokens[req_indices_gpu].to(torch.int64
            ) - self.input_batch.num_computed_tokens_cpu_tensor[req_indices].to(
                device=self.device, dtype=torch.int64, non_blocking=True)
            target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
            target.gpu[:,:total_num_scheduled_tokens] += drift
        _rec("pi.gpu_copies", t0)

        # spec decode
        t0 = _t()
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            td = _t()
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                dl = len(draft_token_ids)
                num_draft_tokens[req_idx] = dl
                if self.input_batch.num_computed_tokens_cpu[req_idx] >= self.input_batch.num_prompt_tokens[req_idx]:
                    num_decode_draft_tokens[req_idx] = dl
            _rec("pi.sd.dict_iter", td)

            tc = _t()
            spec_decode_metadata = self._calc_spec_decode_metadata(num_draft_tokens, cu_num_tokens)
            _rec("pi.sd.calc_metadata", tc)

            tr = _t()
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
            _rec("pi.sd.copy_draft_tokens", tr)

        if self.lora_config:
            assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
            self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
        _rec("pi.spec_decode", t0)

        _rec("em.prepare_inputs", t_all)
        return (logits_indices, spec_decode_metadata)

    GPUModelRunner._prepare_inputs = _prof_pi

    # ── Patch _calc_spec_decode_metadata ──────────────────────────────────
    def _prof_csm(self, num_draft_tokens, cu_num_scheduled_tokens):
        t0 = _t()
        num_sampled_tokens = num_draft_tokens + 1
        cu_num_sampled_tokens = self._get_cumsum_and_arange(
            num_sampled_tokens, self._arange_scratch, cumsum_dtype=np.int32)
        _rec("csm.cumsum1", t0)

        t0 = _t()
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        logits_indices += self._arange_scratch[:cu_num_sampled_tokens[-1]]
        _rec("csm.repeat_logits", t0)

        t0 = _t()
        bonus_logits_indices = cu_num_sampled_tokens - 1
        cu_num_draft_tokens = self._get_cumsum_and_arange(
            num_draft_tokens, self._arange_scratch, cumsum_dtype=np.int32)
        _rec("csm.bonus_draft_cumsum", t0)

        t0 = _t()
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        target_logits_indices += self._arange_scratch[:cu_num_draft_tokens[-1]]
        _rec("csm.repeat_target", t0)

        t0 = _t()
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(self.device, non_blocking=True)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device, non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(self.device, non_blocking=True)
        _rec("csm.to_gpu", t0)

        t0 = _t()
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        _rec("csm.draft_ids_gpu", t0)

        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
    GPUModelRunner._calc_spec_decode_metadata = _prof_csm

    atexit.register(dump)
    print(f"[cpu_profile] Patched GPUModelRunner. warmup={WARMUP}, out={OUT_PATH}")


# Auto-patch when this module is imported
if ENABLED:
    patch_model_runner()
''')


def apply(root: Path) -> None:
    print(f"vLLM root: {root}")

    # 1. Write profiler module
    prof_path = root / "v1" / "_cpu_profile.py"
    prof_path.write_text(PROFILER_CODE)
    print(f"  [OK] Wrote {prof_path}")

    # 2. Add import to gpu_model_runner.py (at top, after first import block)
    mr_path = root / "v1" / "worker" / "gpu_model_runner.py"
    mr_src = mr_path.read_text()

    import_line = "import vllm.v1._cpu_profile  # CPU profiling hook\n"
    marker = "import vllm.v1._cpu_profile"

    if marker in mr_src:
        print("  [SKIP] gpu_model_runner: import already present")
    else:
        # Insert after the first "import" block — find a safe anchor
        anchor = "from vllm.v1.sample.sampler import Sampler"
        if anchor in mr_src:
            mr_src = mr_src.replace(anchor, anchor + "\n" + import_line, 1)
            mr_path.write_text(mr_src)
            print("  [OK] gpu_model_runner: added _cpu_profile import")
        else:
            print("  [WARN] gpu_model_runner: anchor not found, trying fallback")
            # Fallback: insert after first blank line after imports
            lines = mr_src.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("logger = "):
                    lines.insert(i, import_line.rstrip())
                    mr_path.write_text("\n".join(lines))
                    print("  [OK] gpu_model_runner: added import (fallback)")
                    break

    # 3. Fix flash_attn import (aarch64 container issue)
    _fix_rotary(root)

    print("\n  Patch applied. Enable with: VLLM_CPU_PROFILE=1")


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
    else:
        print("  [SKIP] rotary_embedding: already patched or different version")


def check(root: Path) -> bool:
    ok = True
    # Check profiler module
    p = root / "v1" / "_cpu_profile.py"
    if p.exists() and "patch_model_runner" in p.read_text():
        print("  [OK] _cpu_profile.py exists")
    else:
        print("  [MISSING] _cpu_profile.py")
        ok = False

    # Check import in gpu_model_runner
    mr = root / "v1" / "worker" / "gpu_model_runner.py"
    if "import vllm.v1._cpu_profile" in mr.read_text():
        print("  [OK] gpu_model_runner import")
    else:
        print("  [MISSING] gpu_model_runner import")
        ok = False

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    root = find_vllm_root()
    if args.check:
        sys.exit(0 if check(root) else 1)
    elif args.revert:
        print("To revert: pip install --force-reinstall vllm")
    else:
        apply(root)
        print("\nVerifying...")
        check(root)
