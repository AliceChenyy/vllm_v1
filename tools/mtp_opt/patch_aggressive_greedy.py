#!/usr/bin/env python3
"""
Aggressive greedy rejection sampler bypass for MTP1 Qwen3.5-35B-A3B.

Patch O6: Completely replaces RejectionSampler.forward() for the common case:
  - Greedy decoding (temperature=0)
  - No logprobs requested
  - No penalties (no repetition/presence/frequency)
  - No non-argmax-invariant logits processors
  - Not synthetic mode

For MTP1, the rejection logic is trivially:
  1. Compute argmax of target logits at draft position
  2. Compare with draft token
  3. If match: accept draft, take bonus token (argmax of bonus logits)
  4. If no match: reject draft, use target argmax as output

What O6 skips vs vanilla forward():
  - self.sampler() call for bonus token (full sampler pipeline)
  - .clone() of target logits
  - apply_logits_processors() (no-op for greedy+no_penalties)
  - apply_sampling_constraints() (no-op for all_greedy)
  - rejection_sample() / Triton kernel overhead
  - generate_uniform_probs() (not needed for greedy)
  - sample_recovered_tokens() (not needed for greedy)
  - dataclasses.replace() for sampling_metadata
  - torch.full() allocation for output (reuses pre-allocated buffer)

What O6 does differently vs O5:
  - Pre-allocates output buffer (avoids torch.full per call)
  - Fuses argmax+compare into a single Triton kernel on raw logits
  - Avoids metadata.logits_indices gather (computes positions arithmetically)
  - Supports batch_size > 1 (not just BS=1)
  - Falls back cleanly to existing path for non-qualifying requests

Usage:
  python patch_aggressive_greedy.py                # apply O6
  python patch_aggressive_greedy.py --check        # verify
  python patch_aggressive_greedy.py --revert       # revert instructions

After patching:
  O6 is always active for greedy+no_logprobs+no_penalties MTP requests.
  Falls back to original forward() for non-greedy, logprobs, or penalties.
"""
import argparse
import sys
from pathlib import Path


def find_vllm_root() -> Path:
    import vllm
    return Path(vllm.__file__).parent


def _invalidate_pycache(py_file: Path):
    """Remove .pyc for the patched file so Python re-compiles it."""
    stem = py_file.stem
    cache_dir = py_file.parent / "__pycache__"
    if cache_dir.exists():
        for pyc in cache_dir.glob(f"{stem}*.pyc"):
            pyc.unlink()
            print(f"  [OK] Removed stale cache: {pyc.name}")


def apply_o6(root: Path) -> bool:
    """Aggressive greedy bypass for RejectionSampler.forward().

    Completely replaces forward() for greedy+no_logprobs+no_penalties.
    Uses a fused Triton kernel for argmax+compare on raw logits,
    pre-allocated output buffers, and zero unnecessary allocations.
    """
    rs_path = root / "v1" / "sample" / "rejection_sampler.py"
    if not rs_path.exists():
        print(f"  [ERROR] {rs_path} not found")
        return False

    src = rs_path.read_text()

    marker = "# O6_PATCH: aggressive greedy bypass"
    if marker in src:
        print("  [SKIP] O6 already applied")
        return True

    # --------------------------------------------------------------------------
    # Step 1: Add imports and the Triton kernel at the top of the file,
    #         right after existing imports.
    # --------------------------------------------------------------------------
    import_anchor = "from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates"
    if import_anchor not in src:
        print("  [ERROR] O6: import anchor not found")
        return False

    triton_kernel_code = '''

# O6_PATCH: aggressive greedy bypass — fused argmax+compare kernel
@triton.jit(do_not_specialize=["max_spec_len", "vocab_size"])
def _aggressive_greedy_mtp1_kernel(
    output_ptr,           # [batch_size, max_spec_len + 1]
    logits_ptr,           # [total_logits_rows, vocab_size]
    draft_token_ids_ptr,  # [num_draft_tokens_total]
    cu_num_draft_ptr,     # [batch_size]
    target_indices_ptr,   # [num_draft_tokens_total]
    bonus_indices_ptr,    # [batch_size]
    max_spec_len,
    vocab_size,
    BLOCK_V: tl.constexpr,
):
    """Fused argmax-compare-accept kernel for greedy MTP spec decode.

    For each request:
      1. For each draft position, compute argmax of target logits row
      2. Compare argmax with draft token
      3. If match: store draft token; if not: store target argmax, stop
      4. If all accepted: compute argmax of bonus logits row, store it
    """
    req_idx = tl.program_id(0)

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_ptr + req_idx)
    num_draft = end_idx - start_idx

    rejected = False
    for pos in range(num_draft):
        if not rejected:
            draft_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_row = tl.load(target_indices_ptr + start_idx + pos)

            # Compute argmax over target logits row
            best_val = float("-inf")
            best_idx = 0
            for v_start in range(0, vocab_size, BLOCK_V):
                offs = v_start + tl.arange(0, BLOCK_V)
                mask = offs < vocab_size
                vals = tl.load(
                    logits_ptr + target_row * vocab_size + offs,
                    mask=mask,
                    other=float("-inf"),
                ).to(tl.float32)
                local_max = tl.max(vals, axis=0)
                if local_max > best_val:
                    best_val = local_max
                    best_idx = v_start + tl.argmax(vals, axis=0)

            target_id = best_idx
            if draft_id == target_id:
                tl.store(
                    output_ptr + req_idx * (max_spec_len + 1) + pos,
                    draft_id,
                )
            else:
                rejected = True
                tl.store(
                    output_ptr + req_idx * (max_spec_len + 1) + pos,
                    target_id,
                )

    if not rejected:
        # All draft tokens accepted — compute bonus token (argmax of bonus row)
        bonus_row = tl.load(bonus_indices_ptr + req_idx)
        best_val = float("-inf")
        best_idx = 0
        for v_start in range(0, vocab_size, BLOCK_V):
            offs = v_start + tl.arange(0, BLOCK_V)
            mask = offs < vocab_size
            vals = tl.load(
                logits_ptr + bonus_row * vocab_size + offs,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            local_max = tl.max(vals, axis=0)
            if local_max > best_val:
                best_val = local_max
                best_idx = v_start + tl.argmax(vals, axis=0)

        tl.store(
            output_ptr + req_idx * (max_spec_len + 1) + num_draft,
            best_idx,
        )
'''

    src = src.replace(
        import_anchor,
        import_anchor + triton_kernel_code,
        1,
    )

    # --------------------------------------------------------------------------
    # Step 2: Add _aggressive_greedy_forward method + pre-alloc init to
    #         RejectionSampler class, and wire it into forward().
    # --------------------------------------------------------------------------

    # Find the forward method and add O6 fast path at the very top
    old_forward_assert = "        assert metadata.max_spec_len <= MAX_SPEC_LEN"

    # Check for O5 or O3 patches (they also modify this area)
    if "O5_PATCH" in src:
        # O5 is present — insert O6 before O5
        old_forward_block = """\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O5_PATCH:"""
        new_forward_block = """\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O6_PATCH: aggressive greedy bypass
        # For greedy + no_logprobs + no_penalties + no non-argmax processors:
        # completely bypass the sampler pipeline with a fused Triton kernel.
        if (
            sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logitsprocs.non_argmax_invariant
            and not self.synthetic_mode
            and not sampling_metadata.bad_words_token_ids
            and sampling_metadata.allowed_token_ids_mask is None
        ):
            return self._aggressive_greedy_forward(metadata, logits)

        # O5_PATCH:"""
        if old_forward_block not in src:
            print("  [ERROR] O6: O5 forward block anchor not found")
            return False
        src = src.replace(old_forward_block, new_forward_block, 1)

    elif "O3_PATCH" in src:
        old_forward_block = """\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O3_PATCH:"""
        new_forward_block = """\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O6_PATCH: aggressive greedy bypass
        if (
            sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logitsprocs.non_argmax_invariant
            and not self.synthetic_mode
            and not sampling_metadata.bad_words_token_ids
            and sampling_metadata.allowed_token_ids_mask is None
        ):
            return self._aggressive_greedy_forward(metadata, logits)

        # O3_PATCH:"""
        if old_forward_block not in src:
            print("  [ERROR] O6: O3 forward block anchor not found")
            return False
        src = src.replace(old_forward_block, new_forward_block, 1)

    else:
        # No O3/O5 — insert before bonus_logits_indices
        old_forward_block = """\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices"""
        new_forward_block = """\
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # O6_PATCH: aggressive greedy bypass
        if (
            sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logitsprocs.non_argmax_invariant
            and not self.synthetic_mode
            and not sampling_metadata.bad_words_token_ids
            and sampling_metadata.allowed_token_ids_mask is None
        ):
            return self._aggressive_greedy_forward(metadata, logits)

        bonus_logits_indices = metadata.bonus_logits_indices"""
        if old_forward_block not in src:
            print("  [ERROR] O6: vanilla forward block anchor not found")
            return False
        src = src.replace(old_forward_block, new_forward_block, 1)

    # --------------------------------------------------------------------------
    # Step 3: Insert the _aggressive_greedy_forward method into the class.
    #         Insert it right before the forward() method definition.
    # --------------------------------------------------------------------------
    aggressive_method = '''
    # O6_PATCH: aggressive greedy bypass
    _o6_output_buf: torch.Tensor | None = None
    _o6_max_batch: int = 0
    _o6_max_spec: int = 0

    def _aggressive_greedy_forward(
        self,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
    ) -> SamplerOutput:
        """Aggressive greedy bypass — fused argmax+compare Triton kernel.

        Skips ALL of:
        - self.sampler() for bonus tokens
        - logits .clone()
        - apply_logits_processors()
        - apply_sampling_constraints()
        - generate_uniform_probs()
        - rejection_sample() / Triton rejection kernel
        - sample_recovered_tokens()
        - dataclasses.replace() for sampling_metadata

        Uses a single fused Triton kernel that:
        1. Reads target logits rows directly (no gather into intermediate)
        2. Computes argmax in-kernel
        3. Compares with draft token
        4. Computes bonus argmax if all accepted
        5. Writes output directly
        """
        batch_size = len(metadata.num_draft_tokens)
        max_sl = metadata.max_spec_len
        vocab_size = logits.shape[-1]
        device = logits.device

        # Reuse pre-allocated output buffer when possible
        if (
            self._o6_output_buf is None
            or batch_size > self._o6_max_batch
            or max_sl > self._o6_max_spec
            or self._o6_output_buf.device != device
        ):
            self._o6_max_batch = max(batch_size, 8)
            self._o6_max_spec = max(max_sl, 4)
            self._o6_output_buf = torch.full(
                (self._o6_max_batch, self._o6_max_spec + 1),
                PLACEHOLDER_TOKEN_ID,
                dtype=torch.int32,
                device=device,
            )

        output = self._o6_output_buf[:batch_size, :max_sl + 1]
        output.fill_(PLACEHOLDER_TOKEN_ID)

        # Choose BLOCK_V for the Triton kernel (must be power of 2)
        # For Qwen3.5 vocab_size=151936, BLOCK_V=4096 gives ~37 iterations
        BLOCK_V = 4096

        _aggressive_greedy_mtp1_kernel[(batch_size,)](
            output,
            logits,
            metadata.draft_token_ids,
            metadata.cu_num_draft_tokens,
            metadata.target_logits_indices,
            metadata.bonus_logits_indices,
            max_sl,
            vocab_size,
            BLOCK_V=BLOCK_V,
        )

        return SamplerOutput(
            sampled_token_ids=output,
            logprobs_tensors=None,
        )

'''

    # Insert before forward() method definition
    insert_anchor = "    def forward(\n        self,\n        metadata: SpecDecodeMetadata,"
    if insert_anchor in src:
        src = src.replace(insert_anchor, aggressive_method + insert_anchor, 1)
    else:
        print("  [ERROR] O6: could not find forward() insertion point")
        return False

    rs_path.write_text(src)
    print("  [OK] O6: Added aggressive greedy bypass to RejectionSampler")
    _invalidate_pycache(rs_path)
    return True


def check(root: Path) -> bool:
    rs_path = root / "v1" / "sample" / "rejection_sampler.py"
    if not rs_path.exists():
        print("  [MISSING] rejection_sampler.py not found")
        return False

    src = rs_path.read_text()
    ok = True

    if "O6_PATCH" in src:
        print("  [OK] O6: aggressive greedy bypass in rejection_sampler.py")
    else:
        print("  [MISSING] O6: aggressive greedy bypass not found")
        ok = False

    if "_aggressive_greedy_mtp1_kernel" in src:
        print("  [OK] O6: fused Triton kernel present")
    else:
        print("  [MISSING] O6: fused Triton kernel not found")
        ok = False

    if "_aggressive_greedy_forward" in src:
        print("  [OK] O6: _aggressive_greedy_forward method present")
    else:
        print("  [MISSING] O6: _aggressive_greedy_forward method not found")
        ok = False

    if "_o6_output_buf" in src:
        print("  [OK] O6: pre-allocated output buffer")
    else:
        print("  [MISSING] O6: pre-allocated output buffer not found")
        ok = False

    return ok


# Also check that prior patches (O3-O5) are present for combined testing
def check_prior_patches(root: Path):
    rs_path = root / "v1" / "sample" / "rejection_sampler.py"
    src = rs_path.read_text() if rs_path.exists() else ""

    mr_path = root / "v1" / "worker" / "gpu_model_runner.py"
    mr_src = mr_path.read_text() if mr_path.exists() else ""

    for tag, name, text in [
        ("O3", "greedy fast path", src),
        ("O4", "metadata fast path", mr_src),
        ("O5", "MTP1 BS=1 greedy", src),
    ]:
        marker = f"{tag}_PATCH"
        if marker in text:
            print(f"  [OK] {tag}: {name}")
        else:
            print(f"  [INFO] {tag}: {name} — not applied (optional)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply aggressive greedy rejection bypass (O6) to vLLM")
    parser.add_argument("--check", action="store_true",
                        help="Verify O6 patch is applied")
    parser.add_argument("--revert", action="store_true",
                        help="Print revert instructions")
    args = parser.parse_args()

    root = find_vllm_root()
    print(f"vLLM root: {root}")

    if args.check:
        ok = check(root)
        print("\nPrior patches:")
        check_prior_patches(root)
        sys.exit(0 if ok else 1)
    elif args.revert:
        print("To revert: pip install --force-reinstall vllm")
        sys.exit(0)

    ok = apply_o6(root)

    print("\n--- Summary ---")
    print(f"  O6: {'OK' if ok else 'FAILED'}")

    if ok:
        print("\nVerifying...")
        check(root)
        print("\nPrior patches:")
        check_prior_patches(root)
        print("\nO6 is active automatically for greedy+no_logprobs+no_penalties.")
        print("Falls back to original path for non-greedy, logprobs, or penalty requests.")
