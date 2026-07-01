# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-spec W7 micro-benchmark profiler (env-gated, timing-only).

Decomposes the speculative decode cycle into FIXABLE vs FUNDAMENTAL components
by recording CUDA-synchronized wall-clock (CPU+GPU) timings of three regions:

  - ``draft_chain``   : the whole ``SpecDecodeBaseProposer.propose()`` call
                        (the K-step draft chain, including CPU orchestration).
  - ``draft_forward`` : a SINGLE draft model forward inside the chain
                        (one ``self.model(**model_kwargs)`` decode-step call).
  - ``verify``        : the target verify forward (the ``_model_forward`` call
                        in ``GpuModelRunner.execute_model`` over the K+1
                        proposed tokens).

Everything is gated behind ``VLLM_SELF_SPEC_PROFILE``; when off, the context
managers are zero-overhead no-ops and there is no behavior change.

Timings are kept per region as a list of per-call elapsed seconds. The harness
(or any reader) calls :func:`get_profiler().summary()` to fetch steady-state
means after skipping warmup samples. Timing is process-local; in DP setups read
rank 0 (or aggregate consistently across ranks).
"""

from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import contextmanager

import torch

import vllm.envs as envs

# Default number of leading samples per region to drop as warmup. Decode steps
# accumulate fast (one draft_chain per decode step), so a generous warmup keeps
# CUDA-graph capture / first-touch allocations out of the steady-state mean.
_DEFAULT_WARMUP = 50


class SelfSpecProfiler:
    """Process-local accumulator of CUDA-synced region wall-clock samples."""

    def __init__(self) -> None:
        self.enabled: bool = bool(envs.VLLM_SELF_SPEC_PROFILE)
        # region label -> list[elapsed_seconds]
        self._samples: dict[str, list[float]] = {}
        # Because the GpuModelRunner (where timings accumulate) runs in a
        # separate worker process that is force-killed at engine shutdown, an
        # atexit dump is unreliable. Instead, when VLLM_SELF_SPEC_PROFILE_OUT is
        # set, the profiler flushes its summary to a per-PID file incrementally
        # (every _flush_every recorded samples and on a final atexit best-effort)
        # so the latest steady-state numbers survive a force-kill. The harness
        # reads the file with the most samples (the real model-running rank-0).
        self._out_dir: str = ""
        self._flush_every: int = 25
        self._since_flush: int = 0
        # Fine-grained per-step sub-region timers (step_*/step0_*/chain_setup)
        # are extra and perturb the chain total via additional CUDA syncs; only
        # record them when explicitly requested.
        self._fine: bool = bool(int(os.environ.get("VLLM_SELF_SPEC_PROFILE_FINE", "0")))
        if self.enabled:
            self._out_dir = os.environ.get(
                "VLLM_SELF_SPEC_PROFILE_OUT", ""
            ).strip()
            if self._out_dir:
                atexit.register(self._dump_on_exit, self._out_dir)

    def reset(self) -> None:
        self._samples = {}

    @staticmethod
    def _fine_only(label: str) -> bool:
        """Whether a region label is a fine-grained per-step sub-region."""
        return label.startswith(("step_", "step0_", "chain_setup"))

    def _record(self, label: str, elapsed: float) -> None:
        self._samples.setdefault(label, []).append(elapsed)
        if self._out_dir:
            self._since_flush += 1
            if self._since_flush >= self._flush_every:
                self._since_flush = 0
                self._dump_on_exit(self._out_dir)

    @contextmanager
    def region(self, label: str):
        """Time a region with a CUDA sync at both ends (full CPU+GPU time).

        No-op when profiling is disabled. The leading sync ensures prior GPU
        work has drained so it is not attributed to this region; the trailing
        sync captures the region's own GPU work in the elapsed wall time.
        """
        if not self.enabled:
            yield
            return
        if self._fine_only(label) and not self._fine:
            # Fine-grained per-step sub-region timers add many extra CUDA syncs
            # that perturb the enclosing draft_chain total. Gate them behind
            # VLLM_SELF_SPEC_PROFILE_FINE so the clean before/after measurement
            # (draft_chain / draft_forward / verify) is undisturbed by default.
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._record(label, time.perf_counter() - t0)

    @contextmanager
    def cpu_region(self, label: str):
        """Time a region WITHOUT any CUDA sync (pure host wall-clock).

        Used to attribute per-cycle CPU orchestration (rejection-sampler
        parse, bookkeeping loops, next-input build) without perturbing the
        stream: no sync is inserted, so any forced H<->D sync inside the region
        (e.g. ``.cpu()`` / ``.tolist()``) still shows up as the stall it is,
        but this timer adds none of its own. Gated behind
        ``VLLM_SELF_SPEC_PROFILE_FINE`` (like the other sub-region timers) so
        the clean draft_chain/verify measurement is undisturbed by default.
        """
        if not self.enabled or not self._fine:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._record(label, time.perf_counter() - t0)

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._samples.items()}

    def summary(self, warmup: int = _DEFAULT_WARMUP) -> dict[str, dict]:
        """Steady-state per-region stats after dropping ``warmup`` samples.

        Returns a dict label -> {n, mean_ms, std_ms, min_ms, max_ms} computed
        over the post-warmup tail. Regions with no post-warmup samples are
        reported with n=0.
        """
        out: dict[str, dict] = {}
        for label, xs in self._samples.items():
            tail = xs[warmup:] if len(xs) > warmup else []
            n = len(tail)
            if n == 0:
                out[label] = {
                    "n": 0,
                    "n_total": len(xs),
                    "mean_ms": None,
                    "std_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                }
                continue
            mean = sum(tail) / n
            if n > 1:
                var = sum((x - mean) ** 2 for x in tail) / (n - 1)
                std = var**0.5
            else:
                std = 0.0
            out[label] = {
                "n": n,
                "n_total": len(xs),
                "mean_ms": mean * 1e3,
                "std_ms": std * 1e3,
                "min_ms": min(tail) * 1e3,
                "max_ms": max(tail) * 1e3,
            }
        return out

    def _dump_on_exit(self, out_dir: str) -> None:
        try:
            if not any(self._samples.values()):
                return
            os.makedirs(out_dir, exist_ok=True)
            warmup = int(os.environ.get("VLLM_SELF_SPEC_PROFILE_WARMUP", "50"))
            payload = {
                "pid": os.getpid(),
                "dp_rank": os.environ.get("VLLM_DP_RANK", ""),
                "warmup": warmup,
                "counts": self.counts(),
                "summary": self.summary(warmup),
            }
            path = os.path.join(out_dir, f"self_spec_profile_{os.getpid()}.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)  # atomic; harness never reads a partial file
        except Exception:
            # Never let profiling teardown break process exit.
            pass


_PROFILER: SelfSpecProfiler | None = None


def get_profiler() -> SelfSpecProfiler:
    """Return the process-local profiler singleton (created on first use)."""
    global _PROFILER
    if _PROFILER is None:
        _PROFILER = SelfSpecProfiler()
    return _PROFILER
