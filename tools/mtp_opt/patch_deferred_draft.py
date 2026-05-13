#!/usr/bin/env python3
"""
Strategy D: Deferred Draft on Separate CUDA Stream (vLLM 0.20.2)
================================================================

In vLLM 0.20.2, the sample_tokens() flow for EAGLE/MTP is:

  _sample()                         # rejection sampling (GPU, default stream)
  _update_states_after_model_execute()
  propose_draft_token_ids()         # draft forward (GPU, default stream)
    _copy_draft_token_ids_to_cpu()  # async D2H on copy_stream (already optimized)
  _bookkeeping_sync()               # CPU: logprobs, parse tokens, etc.
  eplb_step()
  return

The draft forward runs on the DEFAULT stream. When _bookkeeping_sync does
GPU->CPU syncs (e.g., sampled_token_ids.tolist()), it waits for ALL pending
work on the default stream — including draft forward kernels.

Fix: Run draft forward on a SEPARATE stream so _bookkeeping_sync's GPU syncs
only wait for rejection sampling (already done), not for draft forward.

New flow:
  _sample()                         # GPU, default stream
  _update_states()
  [launch draft on stream2]         # draft forward GPU, separate stream
    _copy_draft_to_cpu()            # async D2H (waits on stream2, not default)
  _bookkeeping_sync()               # CPU, syncs default stream only — overlaps with draft!
  [sync stream2 if needed]
  eplb_step()
  return

Usage:
  python3 patch_deferred_draft.py [--verify]
"""

import argparse
import inspect
import sys

import torch


def apply_patch(verify_only: bool = False):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # Verify we're on v0.20.2-compatible code
    assert hasattr(GPUModelRunner, 'sample_tokens'), \
        "GPUModelRunner has no sample_tokens — not v0.20.2?"
    assert hasattr(GPUModelRunner, 'propose_draft_token_ids'), \
        "GPUModelRunner has no propose_draft_token_ids"
    assert hasattr(GPUModelRunner, '_copy_draft_token_ids_to_cpu'), \
        "GPUModelRunner has no _copy_draft_token_ids_to_cpu"
    assert hasattr(GPUModelRunner, '_bookkeeping_sync'), \
        "GPUModelRunner has no _bookkeeping_sync"

    if verify_only:
        print(f"[INFO] GPUModelRunner module: {GPUModelRunner.__module__}")
        try:
            print(f"[INFO] sample_tokens defined in: "
                  f"{inspect.getfile(GPUModelRunner.sample_tokens)}")
        except (TypeError, OSError):
            pass
        draft_methods = [m for m in dir(GPUModelRunner)
                         if 'draft' in m.lower() or 'propose' in m.lower()
                         or 'spec' in m.lower()]
        print(f"[INFO] Draft/spec methods: {draft_methods}")
        print("[OK] Code structure verified, patch is compatible")
        return True

    # Save original sample_tokens
    GPUModelRunner._orig_sample_tokens = GPUModelRunner.sample_tokens

    # Lazy draft stream per runner instance
    _draft_streams = {}

    def _get_draft_stream(runner):
        rid = id(runner)
        if rid not in _draft_streams:
            _draft_streams[rid] = torch.cuda.Stream(device=runner.device)
        return _draft_streams[rid]

    def patched_sample_tokens(self, grammar_output):
        """Patched sample_tokens: run draft forward on separate CUDA stream.

        We monkey-patch self.propose_draft_token_ids temporarily so that
        when sample_tokens calls it, the GPU kernels go to a separate stream.
        Then _bookkeeping_sync's GPU syncs don't wait for draft.
        """
        # Save the original method
        orig_propose = self.propose_draft_token_ids

        draft_stream = _get_draft_stream(self)
        draft_event = [None]  # mutable container for closure

        def propose_on_draft_stream(*args, **kwargs):
            """Wrapper: run propose_draft_token_ids on separate stream."""
            default_stream = torch.cuda.current_stream(self.device)
            draft_stream.wait_stream(default_stream)

            with torch.cuda.stream(draft_stream):
                result = orig_propose(*args, **kwargs)

            # Record event so _copy_draft_token_ids_to_cpu can wait on it
            draft_event[0] = draft_stream.record_event()
            return result

        # Also patch _copy_draft_token_ids_to_cpu to wait on draft_stream
        orig_copy = self._copy_draft_token_ids_to_cpu

        def copy_with_draft_wait(*args, **kwargs):
            """Wrapper: make copy_stream wait on draft_stream instead of default."""
            if draft_event[0] is not None and hasattr(self, 'draft_token_ids_copy_stream'):
                # The copy stream should wait for draft_stream, not default_stream
                # The original _copy_draft_token_ids_to_cpu does:
                #   copy_stream.wait_stream(default_stream)
                # But draft ran on draft_stream, so we need:
                #   copy_stream.wait_stream(draft_stream)
                # We achieve this by recording on draft_stream and waiting
                self.draft_token_ids_copy_stream.wait_event(draft_event[0])
            return orig_copy(*args, **kwargs)

        # Temporarily replace methods
        self.propose_draft_token_ids = propose_on_draft_stream
        self._copy_draft_token_ids_to_cpu = copy_with_draft_wait

        try:
            result = self._orig_sample_tokens(grammar_output)
        finally:
            # Restore original methods
            self.propose_draft_token_ids = orig_propose
            self._copy_draft_token_ids_to_cpu = orig_copy

        # Ensure draft stream completes before we return
        # (draft D2H copy is already on copy_stream with proper sync,
        #  but we need to make sure draft_stream itself is done before
        #  the next step's execute_model uses the same GPU buffers)
        if draft_event[0] is not None:
            # Make default stream wait for draft to finish before next step
            default_stream = torch.cuda.current_stream(self.device)
            default_stream.wait_event(draft_event[0])

        return result

    GPUModelRunner.sample_tokens = patched_sample_tokens
    print("[OK] Strategy D patch applied: draft forward on separate CUDA stream")
    print("     _bookkeeping_sync GPU syncs no longer wait for draft forward")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Only verify compatibility, don't apply")
    args = parser.parse_args()

    try:
        apply_patch(verify_only=args.verify)
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)
