# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-spec W7 numerical-divergence instrumentation (research-only).

Env-gated (``VLLM_SELF_SPEC_MOE_NUM_DUMP`` = output dir). When active, the MoE
runner captures, for the FIRST decode forward on DP rank 0 at one chosen layer,
the per-token router logits, selected expert ids/weights, and the post-combine
MoE output, then writes them to ``{dir}/moe_dump_L{layer}.pt``. Diffing the dump
between config A (comm-free full-replica) and config C (EP all-to-all) proves the
bf16 reduce STRUCTURE is the divergence cause. Default off -> zero overhead.
"""
import os

import torch

import vllm.envs as envs

_DONE = False
_LOGITS_DONE = False
_PENDING: dict[str, object] = {}
# Minimum distinct router rows for a forward to count as the real prompt (DP
# dummy/coordination forwards have 1 distinct row).
_MIN_DISTINCT_ROWS = 8


def _armed() -> bool:
    """Dump only after the harness arms it (post-warmup), so the captured
    forward is the REAL prompt, not the engine's profiling/warmup forward.

    Armed when "{dir}/ARM" exists (the harness touches it after a warmup
    generate). If the dir is unset, never armed.
    """
    d = envs.VLLM_SELF_SPEC_MOE_NUM_DUMP
    return bool(d) and os.path.exists(os.path.join(d, "ARM"))


def dump_enabled() -> bool:
    return bool(envs.VLLM_SELF_SPEC_MOE_NUM_DUMP) and not _DONE and _armed()


def _is_rank0() -> bool:
    # DP rank 0 only (the worker that owns the dumped tokens locally).
    return os.environ.get("VLLM_DP_RANK", "0") == "0"


def record_topk(layer_id: int, topk_ids: torch.Tensor,
                topk_weights: torch.Tensor) -> None:
    """Capture (b): selected expert ids + weights, from _apply_quant_method."""
    if not dump_enabled() or not _is_rank0():
        return
    if layer_id != envs.VLLM_SELF_SPEC_MOE_DUMP_LAYER:
        return
    _PENDING["topk_ids"] = topk_ids.detach().to("cpu")
    _PENDING["topk_weights"] = topk_weights.detach().float().to("cpu")


def record_forward(layer_id: int, router_logits: torch.Tensor,
                   moe_output: torch.Tensor) -> None:
    """Capture (a) router logits + (c) post-combine MoE output, then write.

    Called at the end of MoERunner.forward. Writes once and latches _DONE so the
    dump is the first decode forward at the chosen layer (steady state, not
    warmup -- the harness clears prior steps before the measured generate).
    """
    global _DONE
    if not dump_enabled() or not _is_rank0():
        return
    if layer_id != envs.VLLM_SELF_SPEC_MOE_DUMP_LAYER:
        return
    # Skip DP dummy/coordination forwards (1 distinct router row) so the dump
    # latches on the REAL multi-token prompt prefill, not a padding forward.
    n_distinct = router_logits.detach().unique(dim=0).shape[0]
    if n_distinct < _MIN_DISTINCT_ROWS:
        return
    out_dir = envs.VLLM_SELF_SPEC_MOE_NUM_DUMP
    os.makedirs(out_dir, exist_ok=True)
    rec = {
        "layer_id": layer_id,
        "router_logits": router_logits.detach().float().to("cpu"),
        "moe_output": moe_output.detach().float().to("cpu"),
        "moe_output_dtype": str(moe_output.dtype),
        "topk_ids": _PENDING.get("topk_ids"),
        "topk_weights": _PENDING.get("topk_weights"),
        "fp32_accum": bool(envs.VLLM_SELF_SPEC_MOE_FP32_ACCUM),
        "n_tokens": int(moe_output.shape[0]),
    }
    path = os.path.join(out_dir, f"moe_dump_L{layer_id}.pt")
    torch.save(rec, path)
    _DONE = True
    _PENDING.clear()


def logits_dump_enabled() -> bool:
    # Only after the MoE dump latched (_DONE) so the captured logits belong to
    # the SAME forward as the MoE dump (the real prompt, not a dummy forward).
    return (
        bool(envs.VLLM_SELF_SPEC_MOE_NUM_DUMP)
        and _DONE
        and not _LOGITS_DONE
        and _armed()
    )


def record_logits(logits: torch.Tensor) -> None:
    """Capture (d): the final next-token logits of the first decode forward.

    Latched on the first single-row (decode batch of 1 prompt-token slice is
    larger; we take the first forward seen) call on DP rank 0. The argmax of
    these logits, diffed between configs A and C, gives the per-token argmax
    flips = the rejected draft tokens.
    """
    global _LOGITS_DONE
    if not logits_dump_enabled() or not _is_rank0():
        return
    out_dir = envs.VLLM_SELF_SPEC_MOE_NUM_DUMP
    os.makedirs(out_dir, exist_ok=True)
    rec = {
        "logits": logits.detach().float().to("cpu"),
        "argmax": logits.detach().argmax(dim=-1).to("cpu"),
        "fp32_accum": bool(envs.VLLM_SELF_SPEC_MOE_FP32_ACCUM),
        "shape": tuple(logits.shape),
    }
    torch.save(rec, os.path.join(out_dir, "logits_dump.pt"))
    _LOGITS_DONE = True
