#!/usr/bin/env python3
"""
Runtime CPU Profiling for vLLM MTP Decode Path
================================================
Monkey-patches vLLM methods at import time with perf_counter_ns timing.
No source-file modification needed — works with any pip-installed vLLM 0.20.x.

Profiled regions per decode step:
  - schedule()
  - execute_model()  (top-level, includes _update_states + _prepare_inputs + model_forward)
    - _update_states()
    - _prepare_inputs()  (with sub-regions: token_indexing, attn_metadata, spec_decode_meta)
      - _calc_spec_decode_metadata()
    - model_forward (from end of prepare_inputs to end of execute_model)
  - sample_tokens()  (sampling + draft token proposal)
  - update_from_output()

Usage:
  # Must be imported BEFORE vllm starts serving
  import cpu_profile_patch  # auto-patches on import

  # Or call explicitly:
  cpu_profile_patch.install()

Environment:
  VLLM_CPU_PROFILE=1         Enable collection (default: 0)
  VLLM_CPU_PROFILE_WARMUP=10 Skip first N steps (default: 10)
  VLLM_CPU_PROFILE_OUT=/path  JSON output (default: /tmp/vllm_cpu_profile.json)
"""
import atexit
import json
import os
import statistics
import time
from collections import defaultdict
from functools import wraps

# ─── Configuration ────────────────────────────────────────────────────────────
ENABLED = os.environ.get("VLLM_CPU_PROFILE", "0") == "1"
WARMUP = int(os.environ.get("VLLM_CPU_PROFILE_WARMUP", "10"))
OUT_PATH = os.environ.get("VLLM_CPU_PROFILE_OUT", "/tmp/vllm_cpu_profile.json")

# ─── Storage ──────────────────────────────────────────────────────────────────
_step_records: list[dict[str, int]] = []
_current_step: dict[str, int] = {}
_step_count = 0
_collecting = False
_installed = False


def _step_begin():
    global _step_count, _collecting, _current_step
    _step_count += 1
    if _step_count <= WARMUP:
        _collecting = False
        return
    _collecting = True
    _current_step = {"__step_begin_ns": time.perf_counter_ns()}


def _step_end():
    global _collecting
    if not _collecting:
        return
    _current_step["__step_end_ns"] = time.perf_counter_ns()
    _current_step["step_total_us"] = (
        _current_step["__step_end_ns"] - _current_step["__step_begin_ns"]
    ) / 1000
    _step_records.append(_current_step)
    _collecting = False


def _record(name: str, start_ns: int):
    if _collecting and start_ns > 0:
        _current_step[name] = (time.perf_counter_ns() - start_ns) / 1000  # us


def _t():
    return time.perf_counter_ns() if _collecting else 0


# ─── Dump & Analysis ─────────────────────────────────────────────────────────
def dump():
    if not _step_records:
        print("[cpu_profile] No steps recorded.")
        return

    # Skip first 2 recorded steps (may be noisy from cache warm)
    skip = min(2, len(_step_records) // 4)
    records = _step_records[skip:]
    if not records:
        records = _step_records

    # Collect all region names
    all_regions = set()
    for r in records:
        all_regions.update(k for k in r if not k.startswith("__"))

    stats = {}
    for region in sorted(all_regions):
        vals = [r[region] for r in records if region in r]
        if not vals:
            continue
        stats[region] = {
            "count": len(vals),
            "mean_us": statistics.mean(vals),
            "median_us": statistics.median(vals),
            "p90_us": sorted(vals)[int(len(vals) * 0.9)] if len(vals) >= 10 else max(vals),
            "p99_us": sorted(vals)[int(len(vals) * 0.99)] if len(vals) >= 100 else max(vals),
            "min_us": min(vals),
            "max_us": max(vals),
            "stdev_us": statistics.stdev(vals) if len(vals) > 1 else 0,
        }

    # ── Print summary ─────────────────────────────────────────────────────
    W = 110
    print(f"\n{'=' * W}")
    print(f"  vLLM CPU Profile — {len(records)} decode steps "
          f"(skipped {_step_count - len(records)} warmup/initial)")
    print(f"{'=' * W}")

    # Ordered display groups
    groups = [
        ("Top-level (per engine step)", [
            "step_total_us", "schedule", "execute_model",
            "sample_tokens", "update_from_output",
        ]),
        ("execute_model breakdown", [
            "em.update_states", "em.prepare_inputs", "em.batch_exec_pad",
            "em.build_attn_meta", "em.preprocess", "em.model_forward",
            "em.postprocess_to_state",
        ]),
        ("_prepare_inputs breakdown", [
            "pi.block_table_commit", "pi.req_indices_cumsum",
            "pi.positions_and_token_indices", "pi.index_select_tokens",
            "pi.attn_metadata", "pi.prev_positions_and_sync",
            "pi.gpu_copies", "pi.spec_decode_section",
        ]),
        ("spec_decode_section breakdown", [
            "pi.sd.dict_iter", "pi.sd.calc_spec_decode_metadata",
            "pi.sd.remaining",
        ]),
        ("_calc_spec_decode_metadata breakdown", [
            "csm.cumsum_arange_1", "csm.np_repeat_logits",
            "csm.bonus_and_draft_cumsum", "csm.np_repeat_target",
            "csm.cpu_to_gpu_copies", "csm.draft_token_ids_gpu",
        ]),
    ]

    for group_name, region_names in groups:
        present = [r for r in region_names if r in stats]
        if not present:
            continue
        print(f"\n  --- {group_name} ---")
        print(f"  {'Region':<45} {'Mean(us)':>9} {'Med(us)':>9} {'P90(us)':>9} "
              f"{'Min(us)':>9} {'Max(us)':>9} {'Stdev':>8} {'N':>5}")
        print(f"  {'-'*45} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*5}")
        for region in region_names:
            if region not in stats:
                continue
            s = stats[region]
            print(f"  {region:<45} {s['mean_us']:>9.1f} {s['median_us']:>9.1f} "
                  f"{s['p90_us']:>9.1f} {s['min_us']:>9.1f} {s['max_us']:>9.1f} "
                  f"{s['stdev_us']:>8.1f} {s['count']:>5}")

    # Any remaining regions not in groups
    shown = {r for _, names in groups for r in names}
    remaining = sorted(r for r in stats if r not in shown)
    if remaining:
        print(f"\n  --- Other ---")
        for region in remaining:
            s = stats[region]
            print(f"  {region:<45} {s['mean_us']:>9.1f} {s['median_us']:>9.1f} "
                  f"{s['p90_us']:>9.1f}")

    # Overhead breakdown as % of step_total
    if "step_total_us" in stats:
        total_us = stats["step_total_us"]["mean_us"]
        print(f"\n  === Overhead breakdown (step_total mean={total_us:.0f}us = {total_us/1000:.2f}ms) ===")
        for region in ["schedule", "execute_model", "sample_tokens", "update_from_output"]:
            if region in stats:
                pct = stats[region]["mean_us"] / total_us * 100
                bar = "#" * int(pct / 2)
                print(f"    {region:<40} {stats[region]['mean_us']:>8.1f}us  {pct:>5.1f}%  {bar}")

        print()
        if "execute_model" in stats:
            em_us = stats["execute_model"]["mean_us"]
            print(f"  === execute_model breakdown ({em_us:.0f}us = {em_us/1000:.2f}ms) ===")
            for region in ["em.update_states", "em.prepare_inputs", "em.batch_exec_pad",
                           "em.build_attn_meta", "em.preprocess", "em.model_forward",
                           "em.postprocess_to_state"]:
                if region in stats:
                    pct = stats[region]["mean_us"] / em_us * 100
                    bar = "#" * int(pct / 2)
                    print(f"    {region:<40} {stats[region]['mean_us']:>8.1f}us  {pct:>5.1f}%  {bar}")

        if "em.prepare_inputs" in stats:
            pi_us = stats["em.prepare_inputs"]["mean_us"]
            print(f"\n  === _prepare_inputs breakdown ({pi_us:.0f}us = {pi_us/1000:.2f}ms) ===")
            for region in ["pi.block_table_commit", "pi.req_indices_cumsum",
                           "pi.positions_and_token_indices", "pi.index_select_tokens",
                           "pi.attn_metadata", "pi.prev_positions_and_sync",
                           "pi.gpu_copies", "pi.spec_decode_section"]:
                if region in stats:
                    pct = stats[region]["mean_us"] / pi_us * 100
                    bar = "#" * int(pct / 2)
                    print(f"    {region:<40} {stats[region]['mean_us']:>8.1f}us  {pct:>5.1f}%  {bar}")

    # Save raw data
    output = {
        "num_steps": len(records),
        "total_steps_including_warmup": _step_count,
        "stats": stats,
        "raw_steps": records[:30],
    }
    try:
        with open(OUT_PATH, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Raw data saved to: {OUT_PATH}")
    except Exception as e:
        print(f"\n  Failed to save raw data: {e}")

    print(f"{'=' * W}\n")


# ─── Monkey-Patching ─────────────────────────────────────────────────────────
def install():
    """Install runtime profiling hooks into vLLM. Call after vLLM is imported."""
    global _installed
    if _installed or not ENABLED:
        return
    _installed = True

    import numpy as np

    # ── 1. Patch EngineCore.step (top-level step envelope) ────────────────
    from vllm.v1.engine.core import EngineCore

    _orig_step = EngineCore.step
    @wraps(_orig_step)
    def _profiled_step(self):
        _step_begin()
        t0 = _t()
        # schedule
        ts = _t()
        if not self.scheduler.has_requests():
            _step_end()
            return {}, False
        scheduler_output = self.scheduler.schedule()
        _record("schedule", ts)

        # execute_model (GPU forward launch)
        te = _t()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)
        _record("execute_model_plus_sample", te)

        # update_from_output
        tu = _t()
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        _record("update_from_output", tu)

        _step_end()
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    EngineCore.step = _profiled_step

    # ── 2. Patch GPUModelRunner.execute_model ─────────────────────────────
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _orig_execute_model = GPUModelRunner.execute_model
    @wraps(_orig_execute_model)
    def _profiled_execute_model(self, scheduler_output, *args, **kwargs):
        t0 = _t()
        result = _orig_execute_model(self, scheduler_output, *args, **kwargs)
        _record("execute_model", t0)
        return result

    GPUModelRunner.execute_model = _profiled_execute_model

    # ── 3. Patch _update_states ───────────────────────────────────────────
    _orig_update_states = GPUModelRunner._update_states
    @wraps(_orig_update_states)
    def _profiled_update_states(self, *args, **kwargs):
        t0 = _t()
        result = _orig_update_states(self, *args, **kwargs)
        _record("em.update_states", t0)
        return result

    GPUModelRunner._update_states = _profiled_update_states

    # ── 4. Patch _prepare_inputs with sub-region instrumentation ──────────
    _orig_prepare_inputs = GPUModelRunner._prepare_inputs
    def _profiled_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        t_all = _t()

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # --- block_table commit ---
        t0 = _t()
        self.input_batch.block_table.commit_block_table(num_reqs)
        _record("pi.block_table_commit", t0)

        # --- req_indices + cumsum ---
        t0 = _t()
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        _record("pi.req_indices_cumsum", t0)

        # --- positions + token_indices ---
        t0 = _t()
        positions_np = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[: cu_num_tokens[-1]]
        )
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)
        _record("pi.positions_and_token_indices", t0)

        # --- index_select tokens ---
        t0 = _t()
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids, 0, token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue
                if num_sched <= 0:
                    output_idx += num_sched
                    continue
                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos
                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])
                output_idx += num_sched
        _record("pi.index_select_tokens", t0)

        # --- attention metadata ---
        t0 = _t()
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
        _record("pi.attn_metadata", t0)

        # --- prev_positions + accepted_tokens sync ---
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
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
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
        _record("pi.prev_positions_and_sync", t0)

        # --- GPU copies (num_computed_tokens, positions, seq_lens, slot_mapping) ---
        t0 = _t()
        if (
            self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):
            self.prev_positions.copy_to_gpu(num_reqs)
            self.prev_num_draft_tokens.copy_to_gpu()
            cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                device=self.device, non_blocking=True
            )
            from vllm.v1.worker.gpu.input_batch import update_num_computed_tokens_for_batch_change
            update_num_computed_tokens_for_batch_change(
                self.num_computed_tokens,
                self.num_accepted_tokens.gpu[:num_reqs],
                self.prev_positions.gpu[:num_reqs],
                self.valid_sampled_token_count_gpu,
                self.prev_num_draft_tokens.gpu,
                cpu_values,
            )
        else:
            self.num_computed_tokens[:num_reqs].copy_(
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )

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
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)

        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        self._prepare_input_ids(
            scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens,
        )

        if self.uses_mrope:
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
            drift = self.num_computed_tokens[req_indices_gpu].to(
                torch.int64
            ) - self.input_batch.num_computed_tokens_cpu_tensor[req_indices].to(
                device=self.device, dtype=torch.int64, non_blocking=True
            )
            target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
            target.gpu[:, :total_num_scheduled_tokens] += drift
        _record("pi.gpu_copies", t0)

        # --- spec decode section ---
        t0 = _t()
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # dict iteration
            td = _t()
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (req_id, draft_token_ids) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                draft_len = len(draft_token_ids)
                num_draft_tokens[req_idx] = draft_len
                if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = draft_len
            _record("pi.sd.dict_iter", td)

            tc = _t()
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            _record("pi.sd.calc_spec_decode_metadata", tc)

            tr = _t()
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
            _record("pi.sd.remaining", tr)

        if self.lora_config:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.set_active_loras(
                self.input_batch, num_scheduled_tokens, num_sampled_tokens
            )
        _record("pi.spec_decode_section", t0)

        _record("em.prepare_inputs", t_all)
        return (logits_indices, spec_decode_metadata)

    GPUModelRunner._prepare_inputs = _profiled_prepare_inputs

    # ── 5. Patch _calc_spec_decode_metadata with sub-regions ──────────────
    _orig_calc_sdm = GPUModelRunner._calc_spec_decode_metadata
    def _profiled_calc_sdm(self, num_draft_tokens, cu_num_scheduled_tokens):
        t_all = _t()

        # Step 1: cumsum + arange for sampled tokens
        t0 = _t()
        num_sampled_tokens = num_draft_tokens + 1
        cu_num_sampled_tokens = self._get_cumsum_and_arange(
            num_sampled_tokens, self._arange_scratch, cumsum_dtype=np.int32
        )
        _record("csm.cumsum_arange_1", t0)

        # Step 2: np.repeat for logits_indices
        t0 = _t()
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        logits_indices += self._arange_scratch[: cu_num_sampled_tokens[-1]]
        _record("csm.np_repeat_logits", t0)

        # Step 3: bonus + draft cumsum
        t0 = _t()
        bonus_logits_indices = cu_num_sampled_tokens - 1
        cu_num_draft_tokens = self._get_cumsum_and_arange(
            num_draft_tokens, self._arange_scratch, cumsum_dtype=np.int32
        )
        _record("csm.bonus_and_draft_cumsum", t0)

        # Step 4: np.repeat for target_logits_indices
        t0 = _t()
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        target_logits_indices += self._arange_scratch[: cu_num_draft_tokens[-1]]
        _record("csm.np_repeat_target", t0)

        # Step 5: CPU → GPU copies
        t0 = _t()
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )
        _record("csm.cpu_to_gpu_copies", t0)

        # Step 6: draft token ids on GPU
        t0 = _t()
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        _record("csm.draft_token_ids_gpu", t0)

        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata as SDM
        return SDM(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    GPUModelRunner._calc_spec_decode_metadata = _profiled_calc_sdm

    # ── 6. Patch sample_tokens ────────────────────────────────────────────
    _orig_sample_tokens = GPUModelRunner.sample_tokens
    @wraps(_orig_sample_tokens)
    def _profiled_sample_tokens(self, *args, **kwargs):
        t0 = _t()
        result = _orig_sample_tokens(self, *args, **kwargs)
        _record("sample_tokens", t0)
        return result

    GPUModelRunner.sample_tokens = _profiled_sample_tokens

    # Register dump at exit
    atexit.register(dump)
    print(f"[cpu_profile] Installed. warmup={WARMUP}, output={OUT_PATH}")


# Need torch imported for the patched _prepare_inputs
import torch
import numpy as np

# Auto-install on import
if ENABLED:
    install()
