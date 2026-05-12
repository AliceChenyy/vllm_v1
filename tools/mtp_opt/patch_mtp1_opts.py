#!/usr/bin/env python3
"""
MTP1 optimization patches for Qwen3.5-35B-A3B on vLLM.

Patch O1: use_local_argmax_reduction — vocab-parallel argmax for draft token
           sampling. Avoids full logits all-gather (O(vocab_size) → O(2*tp_size)).

Patch O2: FULL CUDA graph for MTP proposer — switch from PIECEWISE to FULL
           graph capture for the draft model forward. MTP1 with BS=1 always
           processes exactly 1 token, so shapes are fixed.

Patch O3: Greedy rejection fast path — for all_greedy + no_penalties + no_logprobs,
           skip the full sampler pipeline for bonus tokens and avoid unnecessary
           clone/logits_processors on target logits.

Usage:
  python patch_mtp1_opts.py                    # apply all patches
  python patch_mtp1_opts.py --only o1          # only local argmax
  python patch_mtp1_opts.py --only o2          # only FULL cudagraph
  python patch_mtp1_opts.py --only o3          # only greedy rejection fast path
  python patch_mtp1_opts.py --check            # verify
  python patch_mtp1_opts.py --revert           # revert (reinstall vllm)

After patching:
  O1 is enabled via: --speculative-config '{"method":"mtp","num_speculative_tokens":1,"use_local_argmax_reduction":true}'
  O2 is always active when MTP uses CUDA graphs (default).
  O3 is always active for greedy requests without penalties/logprobs.
"""
import argparse
import sys
from pathlib import Path


def find_vllm_root() -> Path:
    import vllm
    return Path(vllm.__file__).parent


# =============================================================================
# Patch O1: Add get_top_tokens() to Qwen3_5MTP and Qwen3_5MoeMTP
# =============================================================================
def apply_o1(root: Path) -> bool:
    """Add get_top_tokens() to Qwen3.5 MTP model classes."""
    mtp_path = root / "model_executor" / "models" / "qwen3_5_mtp.py"
    if not mtp_path.exists():
        print(f"  [ERROR] {mtp_path} not found")
        return False

    src = mtp_path.read_text()

    marker = "# O1_PATCH: get_top_tokens"
    if marker in src:
        print("  [SKIP] O1 already applied")
        return True

    # Insert get_top_tokens() method into Qwen3_5MTP class, right after
    # compute_logits method
    old_compute_logits = """\
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)"""

    new_compute_logits = """\
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # O1_PATCH: get_top_tokens
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax over its vocab shard, then only
        (value, index) pairs are gathered and reduced.
        Communication: O(batch * 2 * tp_size) vs O(batch * vocab_size).
        \"\"\"
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)"""

    if old_compute_logits not in src:
        print("  [ERROR] O1: compute_logits anchor not found — vLLM version mismatch?")
        return False

    src = src.replace(old_compute_logits, new_compute_logits, 1)
    mtp_path.write_text(src)
    print("  [OK] O1: Added get_top_tokens() to Qwen3_5MTP")

    # Invalidate bytecache
    _invalidate_pycache(mtp_path)
    return True


# =============================================================================
# Patch O2: FULL CUDA graph for MTP proposer
# =============================================================================
def apply_o2(root: Path) -> bool:
    """Switch MTP proposer from PIECEWISE to FULL CUDA graph."""
    proposer_path = root / "v1" / "spec_decode" / "llm_base_proposer.py"
    if not proposer_path.exists():
        print(f"  [ERROR] {proposer_path} not found")
        return False

    src = proposer_path.read_text()

    marker = "# O2_PATCH: FULL cudagraph for proposer"
    if marker in src:
        print("  [SKIP] O2 already applied")
        return True

    # Replace the hardcoded PIECEWISE with FULL for MTP
    old_code = """\
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        \"\"\"Initialize cudagraph dispatcher keys for eagle.

        Eagle only supports PIECEWISE cudagraphs (via mixed_mode).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        \"\"\"
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)"""

    new_code = """\
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        \"\"\"Initialize cudagraph dispatcher keys for eagle/MTP.

        Eagle uses PIECEWISE cudagraphs (via mixed_mode).
        MTP with num_speculative_tokens=1 can use FULL cudagraphs since the
        proposer always processes a fixed-shape batch (1 token per request).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        \"\"\"
        # O2_PATCH: FULL cudagraph for proposer
        if self.speculative_config.enforce_eager:
            eagle_cudagraph_mode = CUDAGraphMode.NONE
        elif (
            cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            # Use FULL for MTP (fixed-shape decode-only proposer).
            # PIECEWISE for EAGLE/DFlash (variable shapes, tree attention).
            if (
                self.speculative_config.method == "mtp"
                and self.num_speculative_tokens == 1
                and cudagraph_mode.has_full_cudagraphs()
            ):
                eagle_cudagraph_mode = CUDAGraphMode.FULL
                import logging
                logging.getLogger(__name__).info(
                    "MTP proposer: using FULL CUDA graph "
                    "(num_speculative_tokens=1, fixed batch shape)")
            else:
                eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)"""

    if old_code not in src:
        print("  [ERROR] O2: initialize_cudagraph_keys anchor not found")
        return False

    src = src.replace(old_code, new_code, 1)
    proposer_path.write_text(src)
    print("  [OK] O2: MTP proposer → FULL CUDA graph (mtp1 only)")

    _invalidate_pycache(proposer_path)
    return True


# =============================================================================
# Patch O3: Greedy rejection sampler fast path
# =============================================================================
def apply_o3(root: Path) -> bool:
    """Add greedy fast path to RejectionSampler.forward().

    For all_greedy + no_penalties + no_logprobs (typical MTP decode):
    - Bonus token: argmax instead of full sampler pipeline
    - Target logits: skip clone + logits_processors + sampling_constraints
    Saves ~200-400us per step.
    """
    rs_path = root / "v1" / "sample" / "rejection_sampler.py"
    if not rs_path.exists():
        print(f"  [ERROR] {rs_path} not found")
        return False

    src = rs_path.read_text()

    marker = "# O3_PATCH: greedy fast path"
    if marker in src:
        print("  [SKIP] O3 already applied")
        return True

    # We insert a fast path at the beginning of forward(), before the
    # existing bonus sampling logic. For the fast path conditions:
    #   all_greedy + no_penalties + no logprobs + no logits_processors
    # we can skip the full sampler pipeline entirely.

    old_forward_start = """\
    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        \"\"\"
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        \"\"\"
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids"""

    new_forward_start = """\
    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        \"\"\"
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        \"\"\"
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O3_PATCH: greedy fast path
        # For all_greedy + no_penalties + no_logprobs, skip full sampler
        # pipeline. Just argmax bonus + greedy rejection kernel.
        if (
            sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logitsprocs.non_argmax_invariant
            and not self.synthetic_mode
        ):
            return self._greedy_fast_path(metadata, logits, sampling_metadata)

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids"""

    if old_forward_start not in src:
        print("  [ERROR] O3: forward() anchor not found in rejection_sampler.py")
        return False

    src = src.replace(old_forward_start, new_forward_start, 1)

    # Now insert the _greedy_fast_path method before the existing forward method
    # Find a good insertion point - right before the forward method
    fast_path_method = '''
    def _greedy_fast_path(
        self,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """Optimized rejection sampling for greedy decoding without penalties.

        Skips:
        - Full sampler pipeline for bonus token (just argmax)
        - .clone() of target logits
        - apply_logits_processors (no-op for greedy)
        - apply_sampling_constraints (no-op for greedy)
        - uniform_probs generation (not needed for greedy)
        """
        # Bonus token: simple argmax (no need for full sampler pipeline)
        bonus_logits = logits[metadata.bonus_logits_indices]
        bonus_token_ids = bonus_logits.to(torch.float32).argmax(
            dim=-1, keepdim=True
        ).to(torch.int32)

        # Target logits: skip clone and processors (all are no-ops for greedy)
        target_logits = logits[metadata.target_logits_indices].to(torch.float32)

        # Run greedy rejection kernel directly
        batch_size = len(metadata.num_draft_tokens)
        output_token_ids = torch.full(
            (batch_size, metadata.max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=target_logits.device,
        )
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            metadata.cu_num_draft_tokens,
            metadata.draft_token_ids,
            target_argmax,
            bonus_token_ids,
            None,  # is_greedy (None = all greedy)
            metadata.max_spec_len,
            None,  # uniform_probs (not needed)
            None,  # synthetic_conditional_rates
            SYNTHETIC_MODE=False,
        )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )

'''

    # Insert before the forward method
    insert_anchor = "    def forward(\n        self,\n        metadata: SpecDecodeMetadata,"
    if insert_anchor in src:
        src = src.replace(insert_anchor, fast_path_method + insert_anchor, 1)
        rs_path.write_text(src)
        print("  [OK] O3: Added _greedy_fast_path to RejectionSampler")
        _invalidate_pycache(rs_path)
        return True
    else:
        print("  [ERROR] O3: insertion anchor not found")
        return False


# =============================================================================
# Patch O4: Fast path for _prepare_inputs spec decode metadata (MTP1)
# =============================================================================
def apply_o4(root: Path) -> bool:
    """Add fast path to _calc_spec_decode_metadata for single-request MTP1.

    For BS=1 + num_draft_tokens=1 (common MTP1 decode), all metadata indices
    are trivially computable with scalar arithmetic. Skips:
    - 2x _get_cumsum_and_arange (numpy cumsum + arange)
    - 2x np.repeat
    - 2x numpy addition
    - 5x torch.from_numpy().to(device) copies
    Replaces with pre-allocated scalar GPU tensors.
    Target savings: ~0.15-0.20ms per step.
    """
    mr_path = root / "v1" / "worker" / "gpu_model_runner.py"
    if not mr_path.exists():
        print(f"  [ERROR] {mr_path} not found")
        return False

    src = mr_path.read_text()

    marker = "# O4_PATCH: MTP1 fast path"
    if marker in src:
        print("  [SKIP] O4 already applied")
        return True

    # Insert fast path at the beginning of _calc_spec_decode_metadata
    old_calc = '''\
    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]'''

    new_calc = '''\
    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # O4_PATCH: MTP1 fast path
        # For single-request with exactly 1 draft token, skip all numpy ops
        batch_size = len(num_draft_tokens)
        if batch_size == 1 and num_draft_tokens[0] == 1:
            return self._calc_spec_decode_metadata_mtp1_bs1(
                cu_num_scheduled_tokens)

        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]'''

    if old_calc not in src:
        print("  [ERROR] O4: _calc_spec_decode_metadata anchor not found")
        return False

    src = src.replace(old_calc, new_calc, 1)

    # Now insert the fast path method. Find a good insertion point —
    # right after _calc_spec_decode_metadata ends (before _prepare_kv_sharing)
    fast_path_method = '''
    def _calc_spec_decode_metadata_mtp1_bs1(
        self,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> "SpecDecodeMetadata":
        """O4_PATCH: Ultra-fast metadata for BS=1, num_draft_tokens=1.

        All indices are trivially computable:
          total_scheduled = cu_num_scheduled_tokens[-1]  (e.g. 2)
          logits_indices = [total-2, total-1]
          target_logits_indices = [0]
          bonus_logits_indices = [1]
          cu_num_draft_tokens = [1]
          cu_num_sampled_tokens = [2]
          draft_token_ids = input_ids[total-1]
        """
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

        T = int(cu_num_scheduled_tokens[-1])
        device = self.device

        # Use pre-allocated buffers if available, else create them
        if not hasattr(self, '_o4_cache'):
            self._o4_cache = {
                'cu_num_draft': torch.ones(1, dtype=torch.int32, device=device),
                'cu_num_sampled': torch.tensor([2], dtype=torch.int32,
                                               device=device),
                'target_idx': torch.zeros(1, dtype=torch.int64, device=device),
                'bonus_idx': torch.ones(1, dtype=torch.int64, device=device),
                'logits_buf': torch.zeros(2, dtype=torch.int64, device=device),
            }

        cache = self._o4_cache
        # Only logits_indices changes per step (depends on T)
        cache['logits_buf'][0] = T - 2
        cache['logits_buf'][1] = T - 1

        # draft_token_ids = input_ids[T-1] (the draft token to verify)
        draft_token_ids = self.input_ids.gpu[T - 1:T]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=[1],
            cu_num_draft_tokens=cache['cu_num_draft'],
            cu_num_sampled_tokens=cache['cu_num_sampled'],
            target_logits_indices=cache['target_idx'],
            bonus_logits_indices=cache['bonus_idx'],
            logits_indices=cache['logits_buf'],
        )

'''

    insert_anchor = "    def _prepare_kv_sharing_fast_prefill("
    if insert_anchor in src:
        src = src.replace(insert_anchor, fast_path_method + insert_anchor, 1)
    else:
        # Fallback: insert after _calc_spec_decode_metadata
        # Find the return statement of _calc_spec_decode_metadata
        return_anchor = "        return SpecDecodeMetadata(\n            draft_token_ids=draft_token_ids,"
        if return_anchor in src:
            # Find end of the method (next method def or end)
            idx = src.index(return_anchor)
            # Find the closing paren
            close_idx = src.index("\n        )\n", idx)
            src = src[:close_idx + 10] + fast_path_method + src[close_idx + 10:]
        else:
            print("  [ERROR] O4: Could not find insertion point")
            return False

    mr_path.write_text(src)
    print("  [OK] O4: Added _calc_spec_decode_metadata_mtp1_bs1 fast path")
    _invalidate_pycache(mr_path)

    # Also optimize the dict iteration in _prepare_inputs for BS=1
    old_dict_iter = '''\
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                draft_len = len(draft_token_ids)
                num_draft_tokens[req_idx] = draft_len
                if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = draft_len'''

    new_dict_iter = '''\
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # O4_PATCH: Reuse pre-allocated arrays when possible
            if not hasattr(self, '_o4_ndt_buf') or len(self._o4_ndt_buf) < num_reqs:
                self._o4_ndt_buf = np.zeros(max(num_reqs, 8), dtype=np.int32)
                self._o4_nddt_buf = np.full(max(num_reqs, 8), -1, dtype=np.int32)
            num_draft_tokens = self._o4_ndt_buf[:num_reqs]
            num_draft_tokens[:] = 0
            num_decode_draft_tokens = self._o4_nddt_buf[:num_reqs]
            num_decode_draft_tokens[:] = -1
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                draft_len = len(draft_token_ids)
                num_draft_tokens[req_idx] = draft_len
                if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = draft_len'''

    if old_dict_iter in src:
        src = src.replace(old_dict_iter, new_dict_iter, 1)
        mr_path.write_text(src)
        print("  [OK] O4: Pre-allocated numpy arrays for spec decode metadata")
    else:
        print("  [WARN] O4: dict iteration anchor not found (may differ in version)")

    return True


# =============================================================================
# Patch O5: Improved greedy rejection fast path for MTP1 BS=1
# =============================================================================
def apply_o5(root: Path) -> bool:
    """Ultra-fast greedy rejection for MTP1 BS=1.

    For BS=1 + MTP1 + greedy + no penalties + no logprobs:
    - Single argmax on 2 logits rows (target + bonus positions)
    - Compare target argmax with draft token
    - Fill output directly without Triton kernel
    Target savings: ~0.1-0.15ms per step.
    """
    rs_path = root / "v1" / "sample" / "rejection_sampler.py"
    if not rs_path.exists():
        print(f"  [ERROR] {rs_path} not found")
        return False

    src = rs_path.read_text()

    marker = "# O5_PATCH: MTP1 BS=1 greedy"
    if marker in src:
        print("  [SKIP] O5 already applied")
        return True

    # Insert fast path at the beginning of forward()
    old_forward = '''\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices'''

    # Check if O3 was already applied (it modifies the same area)
    if "O3_PATCH" in src:
        # O3 adds a fast path before the bonus_logits_indices line
        old_forward = '''\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O3_PATCH: greedy fast path'''

        new_forward = '''\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O5_PATCH: MTP1 BS=1 greedy — ultra-fast path
        # For single request with 1 draft token, greedy, no penalties:
        # just argmax 2 logits rows and compare with draft token.
        if (
            metadata.max_spec_len == 1
            and len(metadata.num_draft_tokens) == 1
            and sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
            and not self.synthetic_mode
        ):
            logits_2 = logits[metadata.logits_indices]  # [2, vocab]
            argmax_2 = logits_2.to(torch.float32).argmax(dim=-1)  # [2]
            draft_id = metadata.draft_token_ids[0]
            accepted = (argmax_2[0] == draft_id)
            out = torch.full(
                (1, 2), PLACEHOLDER_TOKEN_ID,
                dtype=torch.int32, device=logits.device)
            if accepted:
                out[0, 0] = draft_id
                out[0, 1] = argmax_2[1].to(torch.int32)
            else:
                out[0, 0] = argmax_2[0].to(torch.int32)
            return SamplerOutput(
                sampled_token_ids=out, logprobs_tensors=None)

        # O3_PATCH: greedy fast path'''
    else:
        new_forward = '''\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O5_PATCH: MTP1 BS=1 greedy — ultra-fast path
        if (
            metadata.max_spec_len == 1
            and len(metadata.num_draft_tokens) == 1
            and sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
            and not getattr(self, 'synthetic_mode', False)
        ):
            logits_2 = logits[metadata.logits_indices]  # [2, vocab]
            argmax_2 = logits_2.to(torch.float32).argmax(dim=-1)  # [2]
            draft_id = metadata.draft_token_ids[0]
            accepted = (argmax_2[0] == draft_id)
            out = torch.full(
                (1, 2), PLACEHOLDER_TOKEN_ID,
                dtype=torch.int32, device=logits.device)
            if accepted:
                out[0, 0] = draft_id
                out[0, 1] = argmax_2[1].to(torch.int32)
            else:
                out[0, 0] = argmax_2[0].to(torch.int32)
            return SamplerOutput(
                sampled_token_ids=out, logprobs_tensors=None)

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices'''

    if old_forward not in src:
        print("  [ERROR] O5: forward() anchor not found in rejection_sampler.py")
        return False

    src = src.replace(old_forward, new_forward, 1)
    rs_path.write_text(src)
    print("  [OK] O5: Added MTP1 BS=1 greedy ultra-fast path to RejectionSampler")
    _invalidate_pycache(rs_path)
    return True


# =============================================================================
# Fix flash_attn import (reused from previous patches)
# =============================================================================
def fix_rotary(root: Path):
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


def _invalidate_pycache(py_file: Path):
    """Remove .pyc for the patched file so Python re-compiles it."""
    stem = py_file.stem
    cache_dir = py_file.parent / "__pycache__"
    if cache_dir.exists():
        for pyc in cache_dir.glob(f"{stem}*.pyc"):
            pyc.unlink()
            print(f"  [OK] Removed stale cache: {pyc.name}")


# =============================================================================
# Check
# =============================================================================
def check(root: Path) -> bool:
    ok = True

    mtp_path = root / "model_executor" / "models" / "qwen3_5_mtp.py"
    if mtp_path.exists() and "get_top_tokens" in mtp_path.read_text():
        print("  [OK] O1: get_top_tokens() in qwen3_5_mtp.py")
    else:
        print("  [MISSING] O1: get_top_tokens() not in qwen3_5_mtp.py")
        ok = False

    proposer_path = root / "v1" / "spec_decode" / "llm_base_proposer.py"
    if proposer_path.exists() and "O2_PATCH" in proposer_path.read_text():
        print("  [OK] O2: FULL cudagraph patch in llm_base_proposer.py")
    else:
        print("  [MISSING] O2: FULL cudagraph patch not in llm_base_proposer.py")
        ok = False

    rs_path = root / "v1" / "sample" / "rejection_sampler.py"
    rs_src = rs_path.read_text() if rs_path.exists() else ""
    if "O3_PATCH" in rs_src:
        print("  [OK] O3: greedy fast path in rejection_sampler.py")
    else:
        print("  [MISSING] O3: greedy fast path not in rejection_sampler.py")
        ok = False

    mr_path = root / "v1" / "worker" / "gpu_model_runner.py"
    mr_src = mr_path.read_text() if mr_path.exists() else ""
    if "O4_PATCH" in mr_src:
        print("  [OK] O4: MTP1 BS=1 metadata fast path in gpu_model_runner.py")
    else:
        print("  [MISSING] O4: MTP1 BS=1 metadata fast path")
        ok = False

    if "O5_PATCH" in rs_src:
        print("  [OK] O5: MTP1 BS=1 greedy ultra-fast path in rejection_sampler.py")
    else:
        print("  [MISSING] O5: MTP1 BS=1 greedy ultra-fast path")
        ok = False

    return ok


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply MTP1 optimization patches to vLLM")
    parser.add_argument("--check", action="store_true",
                        help="Verify patches are applied")
    parser.add_argument("--revert", action="store_true",
                        help="Print revert instructions")
    parser.add_argument("--only", choices=["o1", "o2", "o3", "o4", "o5"],
                        help="Apply only one patch")
    args = parser.parse_args()

    root = find_vllm_root()
    print(f"vLLM root: {root}")

    if args.check:
        sys.exit(0 if check(root) else 1)
    elif args.revert:
        print("To revert: pip install --force-reinstall vllm")
        sys.exit(0)

    results = {}
    if args.only is None or args.only == "o1":
        results["o1"] = apply_o1(root)
    if args.only is None or args.only == "o2":
        results["o2"] = apply_o2(root)
    if args.only is None or args.only == "o3":
        results["o3"] = apply_o3(root)
    if args.only is None or args.only == "o4":
        results["o4"] = apply_o4(root)
    if args.only is None or args.only == "o5":
        results["o5"] = apply_o5(root)

    fix_rotary(root)

    print("\n--- Summary ---")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    print("\nVerifying...")
    check(root)

    if results.get("o1"):
        print("\nO1 usage:")
        print('  --speculative-config \'{"method":"mtp",'
              '"num_speculative_tokens":1,'
              '"use_local_argmax_reduction":true}\'')
    if results.get("o2"):
        print("\nO2: Active automatically for MTP num_speculative_tokens=1")
    if results.get("o3"):
        print("\nO3: Active automatically for greedy requests without penalties")
