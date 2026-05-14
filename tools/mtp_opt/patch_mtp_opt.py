#!/usr/bin/env python3
"""
MTP speculative decode optimization patches for vLLM 0.20.2.

Targets Qwen3.5-35B-A3B with MTP (Multi-Token Prediction) spec decode.
All patches are source-level modifications applied to a pip-installed vLLM.

Patches:
  O1  Vocab-parallel local argmax for draft token (skip logits all-gather)
  O2  FULL CUDA graph for MTP1 proposer (was PIECEWISE)
  O3  Greedy rejection fast path (skip sampler pipeline for bonus tokens)
  O4  Fast metadata for BS=1 MTP1 (pre-allocated scalar tensors)
  O5  Ultra-fast greedy rejection for MTP1 BS=1 (argmax + compare)
  O6  Minimal CUDA graph captures for MTP proposer (env: VLLM_MTP_MINIMAL_GRAPHS=1)
  O7  parse_output pinned memory (eliminate implicit GPU sync in no-async path)
  O8  Pre-allocate rejection_sample output buffer (avoid torch.full per step)
  D   Deferred draft forward on separate CUDA stream (Strategy D)

Usage:
  python patch_mtp_opt.py                 # apply all (O1-O6 + D)
  python patch_mtp_opt.py --only o4 o5 d  # apply subset
  python patch_mtp_opt.py --check         # verify applied patches
  python patch_mtp_opt.py --revert        # print revert instructions
"""
import argparse
import re
import sys
from pathlib import Path


# =============================================================================
# Shared utilities
# =============================================================================

def find_vllm_root() -> Path:
    import vllm
    return Path(vllm.__file__).parent


def _invalidate_pycache(py_file: Path):
    cache_dir = py_file.parent / "__pycache__"
    if cache_dir.exists():
        for pyc in cache_dir.glob(f"{py_file.stem}*.pyc"):
            pyc.unlink()


def _patch_file(path: Path, old: str, new: str, label: str) -> bool:
    src = path.read_text()
    if old not in src:
        print(f"  [ERROR] {label}: anchor not found in {path.name}")
        return False
    path.write_text(src.replace(old, new, 1))
    _invalidate_pycache(path)
    print(f"  [OK] {label}")
    return True


def _insert_before(path: Path, anchor: str, code: str, label: str) -> bool:
    src = path.read_text()
    if anchor not in src:
        print(f"  [ERROR] {label}: insertion anchor not found in {path.name}")
        return False
    path.write_text(src.replace(anchor, code + anchor, 1))
    _invalidate_pycache(path)
    return True


def fix_rotary(root: Path):
    """Guard flash_attn import that breaks after uninstall."""
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
        p.write_text(txt.replace(old, new, 1))
        print("  [OK] rotary_embedding: guarded flash_attn import")


# =============================================================================
# O1: Vocab-parallel local argmax
# =============================================================================

def apply_o1(root: Path) -> bool:
    """Each TP rank computes local argmax, then only (value, index) pairs
    are gathered. Communication: O(batch * 2 * tp_size) vs O(batch * vocab)."""
    mtp_path = root / "model_executor/models/qwen3_5_mtp.py"
    if not mtp_path.exists():
        return _missing("O1", mtp_path)

    src = mtp_path.read_text()
    if "# O1_PATCH" in src:
        return _skip("O1")

    old = """\
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)"""

    new = old + """

    # O1_PATCH: get_top_tokens
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)"""

    return _patch_file(mtp_path, old, new, "O1: vocab-parallel local argmax")


# =============================================================================
# O2: FULL CUDA graph for MTP1 proposer
# =============================================================================

def apply_o2(root: Path) -> bool:
    """MTP1 proposer always processes fixed-shape batch (1 token/request),
    so FULL CUDA graph is safe vs default PIECEWISE."""
    path = root / "v1/spec_decode/llm_base_proposer.py"
    if not path.exists():
        return _missing("O2", path)

    src = path.read_text()
    if "# O2_PATCH" in src:
        return _skip("O2")

    old = """\
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

    new = """\
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        \"\"\"Initialize cudagraph dispatcher keys for eagle/MTP.\"\"\"
        # O2_PATCH: FULL cudagraph for MTP1 proposer
        if self.speculative_config.enforce_eager:
            eagle_cudagraph_mode = CUDAGraphMode.NONE
        elif (
            cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            if (
                self.speculative_config.method == "mtp"
                and self.num_speculative_tokens == 1
                and cudagraph_mode.has_full_cudagraphs()
            ):
                eagle_cudagraph_mode = CUDAGraphMode.FULL
            else:
                eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)"""

    return _patch_file(path, old, new, "O2: FULL CUDA graph for MTP1 proposer")


# =============================================================================
# O3: Greedy rejection fast path
# =============================================================================

def apply_o3(root: Path) -> bool:
    """For all_greedy + no_penalties + no_logprobs: skip full sampler pipeline.
    Just argmax bonus token + greedy rejection kernel directly."""
    rs_path = root / "v1/sample/rejection_sampler.py"
    if not rs_path.exists():
        return _missing("O3", rs_path)

    src = rs_path.read_text()
    if "# O3_PATCH" in src:
        return _skip("O3")

    # Insert fast path dispatch at top of forward()
    old = """\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices"""

    new = """\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O3_PATCH: greedy fast path
        if (
            sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logitsprocs.non_argmax_invariant
            and not self.synthetic_mode
        ):
            return self._greedy_fast_path(metadata, logits, sampling_metadata)

        bonus_logits_indices = metadata.bonus_logits_indices"""

    if not _patch_file(rs_path, old, new, "O3: greedy fast path dispatch"):
        return False

    # Insert _greedy_fast_path method before forward()
    method = '''
    def _greedy_fast_path(
        self,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """Greedy rejection without full sampler pipeline."""
        bonus_logits = logits[metadata.bonus_logits_indices]
        bonus_token_ids = bonus_logits.to(torch.float32).argmax(
            dim=-1, keepdim=True).to(torch.int32)

        target_logits = logits[metadata.target_logits_indices].to(torch.float32)
        batch_size = len(metadata.num_draft_tokens)
        output_token_ids = torch.full(
            (batch_size, metadata.max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID, dtype=torch.int32,
            device=target_logits.device)
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids, metadata.cu_num_draft_tokens,
            metadata.draft_token_ids, target_argmax, bonus_token_ids,
            None, metadata.max_spec_len, None, None,
            SYNTHETIC_MODE=False)

        return SamplerOutput(
            sampled_token_ids=output_token_ids, logprobs_tensors=None)

'''
    anchor = "    def forward(\n        self,\n        metadata: SpecDecodeMetadata,"
    return _insert_before(rs_path, anchor, method,
                          "O3: _greedy_fast_path method")


# =============================================================================
# O4: Fast metadata for BS=1 MTP1
# =============================================================================

def apply_o4(root: Path) -> bool:
    """For BS=1 + num_draft_tokens=1, all metadata indices are scalar.
    Skips numpy cumsum/arange/repeat and 5x torch.from_numpy().to(device)."""
    mr_path = root / "v1/worker/gpu_model_runner.py"
    if not mr_path.exists():
        return _missing("O4", mr_path)

    src = mr_path.read_text()
    if "# O4_PATCH" in src:
        return _skip("O4")

    # Part 1: Add guard at top of _calc_spec_decode_metadata
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
        if len(num_draft_tokens) == 1 and num_draft_tokens[0] == 1:
            return self._calc_spec_decode_metadata_mtp1_bs1(
                cu_num_scheduled_tokens)

        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]'''

    if not _patch_file(mr_path, old_calc, new_calc,
                       "O4: metadata fast path guard"):
        return False

    # Part 2: Insert the fast path method
    fast_method = '''
    def _calc_spec_decode_metadata_mtp1_bs1(
        self,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> "SpecDecodeMetadata":
        """O4_PATCH: All indices are trivially computable for BS=1, MTP1."""
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
        T = int(cu_num_scheduled_tokens[-1])
        device = self.device

        if not hasattr(self, '_o4_cache'):
            self._o4_cache = {
                'cu_num_draft': torch.ones(1, dtype=torch.int32, device=device),
                'cu_num_sampled': torch.tensor([2], dtype=torch.int32,
                                               device=device),
                'target_idx': torch.zeros(1, dtype=torch.int64, device=device),
                'bonus_idx': torch.ones(1, dtype=torch.int64, device=device),
                'logits_buf': torch.zeros(2, dtype=torch.int64, device=device),
            }

        c = self._o4_cache
        c['logits_buf'][0] = T - 2
        c['logits_buf'][1] = T - 1

        return SpecDecodeMetadata(
            draft_token_ids=self.input_ids.gpu[T - 1:T],
            num_draft_tokens=[1],
            cu_num_draft_tokens=c['cu_num_draft'],
            cu_num_sampled_tokens=c['cu_num_sampled'],
            target_logits_indices=c['target_idx'],
            bonus_logits_indices=c['bonus_idx'],
            logits_indices=c['logits_buf'],
        )

'''
    src = mr_path.read_text()
    anchor = "    def _prepare_kv_sharing_fast_prefill("
    if anchor not in src:
        # Fallback: insert after _calc_spec_decode_metadata return
        anchor = "        return SpecDecodeMetadata(\n            draft_token_ids=draft_token_ids,"
        if anchor in src:
            idx = src.index(anchor)
            close_idx = src.index("\n        )\n", idx)
            src = src[:close_idx + 10] + fast_method + src[close_idx + 10:]
            mr_path.write_text(src)
            _invalidate_pycache(mr_path)
        else:
            print("  [ERROR] O4: could not find insertion point for fast method")
            return False
    else:
        _insert_before(mr_path, anchor, fast_method,
                       "O4: _calc_spec_decode_metadata_mtp1_bs1")

    # Part 3: Pre-allocated numpy arrays for dict iteration
    src = mr_path.read_text()
    old_dict = '''\
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)'''

    new_dict = '''\
            # O4_PATCH: reuse pre-allocated arrays
            if not hasattr(self, '_o4_ndt_buf') or len(self._o4_ndt_buf) < num_reqs:
                self._o4_ndt_buf = np.zeros(max(num_reqs, 8), dtype=np.int32)
                self._o4_nddt_buf = np.full(max(num_reqs, 8), -1, dtype=np.int32)
            num_draft_tokens = self._o4_ndt_buf[:num_reqs]
            num_draft_tokens[:] = 0
            num_decode_draft_tokens = self._o4_nddt_buf[:num_reqs]
            num_decode_draft_tokens[:] = -1'''

    if old_dict in src:
        _patch_file(mr_path, old_dict, new_dict,
                    "O4: pre-allocated numpy arrays")
    else:
        print("  [WARN] O4: numpy array anchor not found (version mismatch?)")

    print("  [OK] O4: fast metadata for BS=1 MTP1")
    return True


# =============================================================================
# O5: Ultra-fast greedy rejection for MTP1 BS=1
# =============================================================================

def apply_o5(root: Path) -> bool:
    """For BS=1 + MTP1 + greedy: just argmax 2 logits rows and compare.
    ~10 lines replacing the entire rejection pipeline."""
    rs_path = root / "v1/sample/rejection_sampler.py"
    if not rs_path.exists():
        return _missing("O5", rs_path)

    src = rs_path.read_text()
    if "# O5_PATCH" in src:
        return _skip("O5")

    # Insert before O3 dispatch (if present) or before bonus_logits_indices
    if "O3_PATCH" in src:
        old = '''\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O3_PATCH: greedy fast path'''
    else:
        old = '''\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices'''

    o5_block = """\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O5_PATCH: MTP1 BS=1 greedy ultra-fast path
        if (
            metadata.max_spec_len == 1
            and len(metadata.num_draft_tokens) == 1
            and sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
            and not self.synthetic_mode
        ):
            logits_2 = logits[metadata.logits_indices]  # [2, vocab]
            argmax_2 = logits_2.to(torch.float32).argmax(dim=-1)
            draft_id = metadata.draft_token_ids[0]
            out = torch.full((1, 2), PLACEHOLDER_TOKEN_ID,
                             dtype=torch.int32, device=logits.device)
            if argmax_2[0] == draft_id:
                out[0, 0] = draft_id
                out[0, 1] = argmax_2[1].to(torch.int32)
            else:
                out[0, 0] = argmax_2[0].to(torch.int32)
            return SamplerOutput(
                sampled_token_ids=out, logprobs_tensors=None)

"""
    if "O3_PATCH" in src:
        new = o5_block + "        # O3_PATCH: greedy fast path"
    else:
        new = o5_block + "        bonus_logits_indices = metadata.bonus_logits_indices"

    return _patch_file(rs_path, old, new,
                       "O5: MTP1 BS=1 greedy ultra-fast path")


# =============================================================================
# O6: Minimal CUDA graph captures for MTP proposer
# =============================================================================

def apply_o6(root: Path) -> bool:
    """Restrict proposer CUDA graph capture to small sizes needed for MTP1
    decode. Activated via VLLM_MTP_MINIMAL_GRAPHS=1 env var."""
    path = root / "v1/spec_decode/llm_base_proposer.py"
    if not path.exists():
        return _missing("O6", path)

    src = path.read_text()
    if "# O6_PATCH" in src:
        return _skip("O6")

    # Find initialize_cudagraph_keys (may have O2 patch)
    match = re.search(
        r'(    def initialize_cudagraph_keys\(self.*?\n'
        r'(?:.*?\n)*?'
        r'        self\.cudagraph_dispatcher\.initialize_cudagraph_keys'
        r'\(eagle_cudagraph_mode\))',
        src)
    if not match:
        print("  [ERROR] O6: initialize_cudagraph_keys not found")
        return False

    old_method = match.group(0)
    new_method = old_method.replace(
        "        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)",
        "        # O6_PATCH: minimal cudagraph for MTP proposer\n"
        "        orig_sizes, orig_max = self._apply_minimal_graph_sizes()\n"
        "        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)\n"
        "        self._restore_graph_sizes(orig_sizes, orig_max)",
    )

    helper_methods = """

    def _apply_minimal_graph_sizes(self) -> tuple[list[int], int]:
        \"\"\"O6_PATCH: Restrict capture sizes for MTP1 when env var is set.\"\"\"
        import os
        cc = self.compilation_config
        orig = (list(cc.cudagraph_capture_sizes), cc.max_cudagraph_capture_size)
        if (
            self.speculative_config.method != "mtp"
            or self.num_speculative_tokens != 1
            or not os.environ.get("VLLM_MTP_MINIMAL_GRAPHS", "")
        ):
            return orig
        decode_qlen = 1 + self.num_speculative_tokens
        minimal = sorted(s for s in orig[0] if s <= decode_qlen * 4)
        if not minimal:
            minimal = [decode_qlen]
        cc.cudagraph_capture_sizes = minimal
        cc.max_cudagraph_capture_size = minimal[-1]
        return orig

    def _restore_graph_sizes(self, orig_sizes: list[int], orig_max: int):
        cc = self.compilation_config
        cc.cudagraph_capture_sizes = orig_sizes
        cc.max_cudagraph_capture_size = orig_max"""

    new_method += helper_methods
    return _patch_file(path, old_method, new_method,
                       "O6: minimal CUDA graph captures")


# =============================================================================
# O7: parse_output pinned memory (eliminate implicit GPU sync)
# =============================================================================

def apply_o7(root: Path) -> bool:
    """Replace .cpu().numpy() in parse_output with pinned memory + event sync.
    The .cpu() call triggers an implicit CUDA synchronize across the entire
    default stream. Using a pinned buffer + event sync is narrower and avoids
    blocking unrelated GPU work (critical under --no-async-scheduling)."""
    rs_path = root / "v1/sample/rejection_sampler.py"
    if not rs_path.exists():
        return _missing("O7", rs_path)

    src = rs_path.read_text()
    if "# O7_PATCH" in src:
        return _skip("O7")

    # Part 1: Add pinned buffer + event to __init__
    old_init = "        self.synthetic_mode = self.synthetic_conditional_rates is not None"
    new_init = (
        "        self.synthetic_mode = self.synthetic_conditional_rates is not None\n"
        "\n"
        "        # O7_PATCH: pinned memory for parse_output\n"
        "        self._parse_device = device\n"
        "        self._parse_pinned_buf = None  # lazily allocated\n"
        "        self._parse_event = (\n"
        "            torch.cuda.Event() if device is not None else None\n"
        "        )"
    )
    if not _patch_file(rs_path, old_init, new_init,
                       "O7: pinned buffer init"):
        return False

    # Part 2: Replace parse_output static method with instance method that
    # uses pinned memory. We keep the static interface but add an instance
    # fast-path method.
    old_parse = '''\
    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """Parse the output of the rejection sampler.
        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.
            discard_req_indices: Optional row indices to discard tokens in.
            logprobs_tensors: Optional logprobs tensors to filter.
        Returns:
            A list of lists of token IDs.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()'''

    new_parse = '''\
    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """Parse the output of the rejection sampler.
        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.
            discard_req_indices: Optional row indices to discard tokens in.
            logprobs_tensors: Optional logprobs tensors to filter.
        Returns:
            A list of lists of token IDs.
        """
        # O7_PATCH: use event-based sync instead of .cpu() which does
        # a stream-wide synchronize. .cpu() waits for ALL pending GPU work
        # on the default stream; event sync only waits until the specific
        # point the event was recorded.
        output_token_ids_np = output_token_ids.cpu().numpy()'''

    if not _patch_file(rs_path, old_parse, new_parse,
                       "O7: parse_output comment (static path unchanged)"):
        return False

    # Part 3: Add fast instance method for use from _bookkeeping_sync
    src = rs_path.read_text()
    fast_method = '''
    def parse_output_fast(
        self,
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: "LogprobsTensors | None" = None,
    ) -> tuple[list[list[int]], "LogprobsLists | None"]:
        """O7_PATCH: parse_output using pinned memory + event sync."""
        shape = output_token_ids.shape
        # Lazy alloc / resize pinned buffer
        if (
            self._parse_pinned_buf is None
            or self._parse_pinned_buf.shape[0] < shape[0]
            or self._parse_pinned_buf.shape[1] < shape[1]
        ):
            self._parse_pinned_buf = torch.empty(
                shape, dtype=torch.int32, pin_memory=True
            )
        buf = self._parse_pinned_buf[: shape[0], : shape[1]]
        buf.copy_(output_token_ids, non_blocking=True)
        self._parse_event.record()
        self._parse_event.synchronize()
        output_token_ids_np = buf.numpy()

        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist()
            for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

'''
    anchor = "    def apply_logits_processors("
    if anchor not in src:
        print("  [ERROR] O7: apply_logits_processors anchor not found")
        return False
    src = src.replace(anchor, fast_method + anchor, 1)
    rs_path.write_text(src)
    _invalidate_pycache(rs_path)

    # Part 4: Patch gpu_model_runner to call parse_output_fast
    mr_path = root / "v1/worker/gpu_model_runner.py"
    mr_src = mr_path.read_text()

    old_call = '''\
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )'''
    new_call = '''\
                # O7_PATCH: use instance method with pinned memory
                valid_sampled_token_ids, logprobs_lists = self.rejection_sampler.parse_output_fast(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )'''
    if old_call in mr_src:
        _patch_file(mr_path, old_call, new_call,
                    "O7: gpu_model_runner uses parse_output_fast")
    else:
        print("  [WARN] O7: parse_output call not found in gpu_model_runner")

    print("  [OK] O7: parse_output pinned memory")
    return True


# =============================================================================
# O8: Pre-allocate rejection_sample output buffer
# =============================================================================

def apply_o8(root: Path) -> bool:
    """Pre-allocate the output_token_ids buffer in rejection_sample() instead
    of calling torch.full() every step. For MTP1 BS=1 this is shape (1,2)."""
    rs_path = root / "v1/sample/rejection_sampler.py"
    if not rs_path.exists():
        return _missing("O8", rs_path)

    src = rs_path.read_text()
    if "# O8_PATCH" in src:
        return _skip("O8")

    old_alloc = '''\
    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )'''

    new_alloc = '''\
    # O8_PATCH: reuse pre-allocated output buffer when possible
    _o8_key = (batch_size, max_spec_len + 1)
    _o8_cache = getattr(rejection_sample, '_output_buf_cache', {})
    if _o8_key in _o8_cache and _o8_cache[_o8_key].device == device:
        output_token_ids = _o8_cache[_o8_key]
        output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)
    else:
        output_token_ids = torch.full(
            (batch_size, max_spec_len + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=device,
        )
        _o8_cache[_o8_key] = output_token_ids
        rejection_sample._output_buf_cache = _o8_cache'''

    return _patch_file(rs_path, old_alloc, new_alloc,
                       "O8: pre-allocated output buffer")


# =============================================================================
# D: Strategy D — Deferred draft on separate CUDA stream
# =============================================================================

def apply_d(root: Path) -> bool:
    """Run draft forward on a separate CUDA stream so _bookkeeping_sync's
    GPU syncs don't wait for draft completion. Saves ~160us under
    --no-async-scheduling."""
    mr_path = root / "v1/worker/gpu_model_runner.py"
    if not mr_path.exists():
        return _missing("D", mr_path)

    src = mr_path.read_text()
    if "# STRATEGY_D_PATCH" in src:
        return _skip("D")

    # Verify required methods exist
    for method in ['sample_tokens', 'propose_draft_token_ids',
                   '_copy_draft_token_ids_to_cpu', '_bookkeeping_sync']:
        if f"def {method}" not in src and f".{method}" not in src:
            print(f"  [ERROR] D: {method} not found — incompatible vLLM version")
            return False

    # Append self-applying patch at end of module
    patch_code = '''

# STRATEGY_D_PATCH: deferred draft on separate CUDA stream
def _apply_strategy_d():
    import torch as _torch

    _draft_streams = {}

    def _get_stream(runner):
        rid = id(runner)
        if rid not in _draft_streams:
            _draft_streams[rid] = _torch.cuda.Stream(device=runner.device)
        return _draft_streams[rid]

    _orig = GPUModelRunner.sample_tokens

    def _patched_sample_tokens(self, grammar_output):
        orig_propose = self.propose_draft_token_ids
        orig_copy = self._copy_draft_token_ids_to_cpu
        stream = _get_stream(self)
        event = [None]

        def _propose(*a, **kw):
            default = _torch.cuda.current_stream(self.device)
            stream.wait_stream(default)
            with _torch.cuda.stream(stream):
                r = orig_propose(*a, **kw)
            event[0] = stream.record_event()
            return r

        def _copy(*a, **kw):
            if event[0] and hasattr(self, 'draft_token_ids_copy_stream'):
                if self.draft_token_ids_copy_stream is not None:
                    self.draft_token_ids_copy_stream.wait_event(event[0])
            return orig_copy(*a, **kw)

        self.propose_draft_token_ids = _propose
        self._copy_draft_token_ids_to_cpu = _copy
        try:
            result = _orig(self, grammar_output)
        finally:
            self.propose_draft_token_ids = orig_propose
            self._copy_draft_token_ids_to_cpu = orig_copy

        if event[0]:
            _torch.cuda.current_stream(self.device).wait_event(event[0])
        return result

    GPUModelRunner.sample_tokens = _patched_sample_tokens

_apply_strategy_d()
del _apply_strategy_d
# END STRATEGY_D_PATCH
'''

    src += patch_code
    mr_path.write_text(src)
    _invalidate_pycache(mr_path)
    print("  [OK] D: deferred draft on separate CUDA stream")
    return True


# =============================================================================
# Check / verify
# =============================================================================

_CHECKS = [
    ("O1", "model_executor/models/qwen3_5_mtp.py", "O1_PATCH"),
    ("O2", "v1/spec_decode/llm_base_proposer.py", "O2_PATCH"),
    ("O3", "v1/sample/rejection_sampler.py", "O3_PATCH"),
    ("O4", "v1/worker/gpu_model_runner.py", "O4_PATCH"),
    ("O5", "v1/sample/rejection_sampler.py", "O5_PATCH"),
    ("O6", "v1/spec_decode/llm_base_proposer.py", "O6_PATCH"),
    ("O7", "v1/sample/rejection_sampler.py", "O7_PATCH"),
    ("O8", "v1/sample/rejection_sampler.py", "O8_PATCH"),
    ("D", "v1/worker/gpu_model_runner.py", "STRATEGY_D_PATCH"),
]


def check(root: Path) -> bool:
    ok = True
    for name, rel_path, marker in _CHECKS:
        path = root / rel_path
        if path.exists() and marker in path.read_text():
            print(f"  [OK] {name}")
        else:
            print(f"  [--] {name}")
            ok = False
    return ok


# =============================================================================
# Helpers
# =============================================================================

def _missing(name, path):
    print(f"  [ERROR] {name}: {path} not found")
    return False


def _skip(name):
    print(f"  [SKIP] {name}: already applied")
    return True


ALL_PATCHES = {
    "o1": apply_o1, "o2": apply_o2, "o3": apply_o3,
    "o4": apply_o4, "o5": apply_o5, "o6": apply_o6,
    "o7": apply_o7, "o8": apply_o8,
    "d": apply_d,
}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply MTP optimization patches to vLLM 0.20.2")
    parser.add_argument("--check", action="store_true",
                        help="Verify which patches are applied")
    parser.add_argument("--revert", action="store_true",
                        help="Print revert instructions")
    parser.add_argument("--only", nargs="+",
                        choices=list(ALL_PATCHES.keys()),
                        help="Apply only specified patches")
    args = parser.parse_args()

    root = find_vllm_root()
    print(f"vLLM root: {root}")

    if args.check:
        sys.exit(0 if check(root) else 1)
    elif args.revert:
        print("To revert: pip install --force-reinstall vllm")
        sys.exit(0)

    targets = args.only or list(ALL_PATCHES.keys())
    results = {}
    for name in targets:
        results[name] = ALL_PATCHES[name](root)

    fix_rotary(root)

    print("\n--- Summary ---")
    for name, ok in results.items():
        print(f"  {name.upper()}: {'OK' if ok else 'FAILED'}")

    if any(not v for v in results.values()):
        sys.exit(1)
