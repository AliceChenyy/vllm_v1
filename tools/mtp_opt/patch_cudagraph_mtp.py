#!/usr/bin/env python3
"""
Patch O6: Minimal CUDA graph capture for MTP1 proposer.

Problem: vLLM captures ~51 CUDA graph sizes for the proposer (same as target
model). For MTP1 decode, the proposer always runs BS=1, seq_len=1, so only
1 graph is ever used. The extra 50 graphs waste capture time and memory.

This patch overrides the proposer's CudagraphDispatcher to only capture
graphs for BS=1 (or the num_speculative_tokens+1 token count for MTP1=2).
Falls back to eager for any unexpected batch size.

Usage:
  python patch_cudagraph_mtp.py           # apply patch
  python patch_cudagraph_mtp.py --check   # verify
  python patch_cudagraph_mtp.py --revert  # print revert instructions
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
    """Override proposer's CUDA graph capture to only capture BS=1 for MTP1.

    The patch modifies initialize_cudagraph_keys() in llm_base_proposer.py
    to replace the compilation_config's cudagraph_capture_sizes with a minimal
    set [2] (= num_speculative_tokens + 1 = 2 for MTP1) before calling
    the dispatcher's initialize_cudagraph_keys, then restore original sizes.

    For MTP1, the proposer processes exactly 1 token per request (the draft
    token). With num_speculative_tokens=1, the total decode query len is 2,
    so capture sizes should be [2] (after adjust_cudagraph_sizes_for_spec_decode
    rounds up to multiples of 2). But for the first pass in propose(), the
    proposer can see larger batches (all tokens). The key insight: for the
    proposer's second+ passes (the loop in propose()), it always sees
    batch_size tokens where each request contributes 1 token. For the first
    pass, it reuses the target's graph sizes. So we only need to restrict
    the proposer's own dispatcher.

    Actually, re-reading the code more carefully:
    - The proposer's CudagraphDispatcher is used for both first pass and
      subsequent passes in propose().
    - First pass: num_tokens = total tokens (could be large during prefill)
    - Subsequent passes: batch_size (1 per request, decode only)
    - For BS=1 MTP1 decode: first pass has ~2 tokens, subsequent passes ~1

    So the minimal capture sizes should include small sizes that cover
    typical MTP1 decode workloads. For BS=1, we need size=1 and size=2.
    For larger BS, we still want reasonable coverage. The key optimization
    is eliminating the ~30+ large sizes that are never used by the proposer.

    Strategy: Cap the proposer's max_cudagraph_capture_size at
    max_num_seqs * (num_speculative_tokens + 1), and only keep sizes up to
    that cap. For typical configs (max_num_seqs=256, MTP1), this means
    max=512 which is similar to baseline. But for BS=1 benchmarks, we can
    also set an env var to force minimal sizes.

    Simplest effective approach: For MTP1, the proposer only needs graphs for
    sizes that are multiples of (num_speculative_tokens + 1). After
    adjust_cudagraph_sizes_for_spec_decode, all sizes are already multiples.
    We restrict to only [2] (the decode_query_len) when the proposer is MTP
    with num_speculative_tokens=1 and VLLM_MTP_MINIMAL_GRAPHS=1 is set.
    """
    proposer_path = root / "v1" / "spec_decode" / "llm_base_proposer.py"
    if not proposer_path.exists():
        print(f"  [ERROR] {proposer_path} not found")
        return False

    src = proposer_path.read_text()

    marker = "# O6_PATCH: minimal cudagraph for MTP proposer"
    if marker in src:
        print("  [SKIP] O6 already applied")
        return True

    # Replace initialize_cudagraph_keys to restrict capture sizes for MTP1
    old_code = """\
    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        \"\"\"Initialize cudagraph dispatcher keys for the drafter.

        Only supports PIECEWISE cudagraphs (via mixed_mode).
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

    # Check if O2 patch was already applied (it modifies this same method)
    if "O2_PATCH" in src:
        # O2 already modified initialize_cudagraph_keys — find the O2 version
        # We need to wrap whatever version exists
        old_code_o2 = None
        # Find the method boundaries
        import re
        match = re.search(
            r'(    def initialize_cudagraph_keys\(self.*?\n'
            r'(?:.*?\n)*?'
            r'        self\.cudagraph_dispatcher\.initialize_cudagraph_keys\(eagle_cudagraph_mode\))',
            src)
        if match:
            old_code = match.group(0)
        else:
            print("  [ERROR] O6: Could not find initialize_cudagraph_keys method")
            return False

    if old_code not in src:
        print("  [ERROR] O6: initialize_cudagraph_keys anchor not found")
        return False

    new_code = old_code + """

    def _apply_minimal_cudagraph_sizes(self) -> tuple[list[int], int]:
        # O6_PATCH: minimal cudagraph for MTP proposer
        \"\"\"Temporarily restrict cudagraph_capture_sizes for MTP1 proposer.

        Returns (original_sizes, original_max) to restore later.
        Only applied when:
        - method == "mtp" and num_speculative_tokens == 1
        - VLLM_MTP_MINIMAL_GRAPHS=1 env var is set

        For MTP1 decode, the proposer always processes exactly 1 token
        per request. The decode_query_len is 2 (= 1 + num_speculative_tokens).
        After adjust_cudagraph_sizes_for_spec_decode, all sizes are multiples
        of 2. We only need size=2 for the common BS=1 decode case.

        For larger batch sizes during decode, sizes up to
        max_num_seqs * decode_query_len could be needed, but in practice
        the proposer's first pass reuses target tokens (variable size)
        while subsequent passes use batch_size (fixed at 1 for BS=1).
        \"\"\"
        import os
        import logging
        _logger = logging.getLogger(__name__)

        cc = self.compilation_config
        orig_sizes = list(cc.cudagraph_capture_sizes)
        orig_max = cc.max_cudagraph_capture_size

        if (
            self.speculative_config.method != "mtp"
            or self.num_speculative_tokens != 1
        ):
            return orig_sizes, orig_max

        if not os.environ.get("VLLM_MTP_MINIMAL_GRAPHS", ""):
            return orig_sizes, orig_max

        decode_query_len = 1 + self.num_speculative_tokens  # = 2 for MTP1

        # Minimal set: just the decode_query_len (=2) for BS=1
        # Plus a few small multiples for slightly larger batches
        minimal_sizes = sorted(set(
            s for s in orig_sizes
            if s <= max(decode_query_len * 4, 8)  # keep sizes up to 8
        ))
        if not minimal_sizes:
            minimal_sizes = [decode_query_len]

        cc.cudagraph_capture_sizes = minimal_sizes
        cc.max_cudagraph_capture_size = minimal_sizes[-1]

        _logger.info(
            "O6: MTP1 proposer minimal CUDA graphs: %d sizes %s "
            "(was %d sizes, max=%d)",
            len(minimal_sizes), minimal_sizes,
            len(orig_sizes), orig_max,
        )

        return orig_sizes, orig_max

    def _restore_cudagraph_sizes(
        self, orig_sizes: list[int], orig_max: int
    ) -> None:
        \"\"\"Restore original cudagraph_capture_sizes after proposer init.\"\"\"
        cc = self.compilation_config
        cc.cudagraph_capture_sizes = orig_sizes
        cc.max_cudagraph_capture_size = orig_max"""

    # Now we need to modify the initialize_cudagraph_keys call to use the
    # minimal sizes. We wrap the dispatcher init with save/restore.
    final_code = new_code.replace(
        "        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)",
        "        # O6_PATCH: minimal cudagraph for MTP proposer\n"
        "        orig_sizes, orig_max = self._apply_minimal_cudagraph_sizes()\n"
        "        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)\n"
        "        self._restore_cudagraph_sizes(orig_sizes, orig_max)",
        1
    )

    src = src.replace(old_code, final_code, 1)
    proposer_path.write_text(src)
    print("  [OK] O6: Added minimal CUDA graph capture for MTP1 proposer")

    _invalidate_pycache(proposer_path)
    return True


def check(root: Path) -> bool:
    proposer_path = root / "v1" / "spec_decode" / "llm_base_proposer.py"
    if not proposer_path.exists():
        print("  [MISSING] llm_base_proposer.py not found")
        return False

    src = proposer_path.read_text()
    if "O6_PATCH" in src:
        print("  [OK] O6: minimal cudagraph patch in llm_base_proposer.py")
        return True
    else:
        print("  [MISSING] O6: minimal cudagraph patch not in llm_base_proposer.py")
        return False


# Also patch flash_attn rotary import guard (reused from patch_mtp1_opts.py)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply minimal CUDA graph patch for MTP1 proposer")
    parser.add_argument("--check", action="store_true",
                        help="Verify patch is applied")
    parser.add_argument("--revert", action="store_true",
                        help="Print revert instructions")
    args = parser.parse_args()

    root = find_vllm_root()
    print(f"vLLM root: {root}")

    if args.check:
        sys.exit(0 if check(root) else 1)
    elif args.revert:
        print("To revert: pip install --force-reinstall vllm")
        sys.exit(0)

    ok = apply_o6(root)
    fix_rotary(root)

    print("\n--- Summary ---")
    print(f"  O6: {'OK' if ok else 'FAILED'}")

    if ok:
        print("\nUsage:")
        print("  # Enable minimal graphs for MTP1 proposer:")
        print("  export VLLM_MTP_MINIMAL_GRAPHS=1")
        print("  # Then run vllm serve as usual with MTP1 spec decode")
        print("")
        print("  # Without the env var, the patch is a no-op (safe to deploy).")

    print("\nVerifying...")
    check(root)
