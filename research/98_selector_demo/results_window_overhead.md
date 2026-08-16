# The window's short-context overhead: measured, attributed, and removed

Record of 2026-08-16, h103 GPU 7 / NUMA 1. Probe:
`scripts/probe_w98_window_fastpath.py` (X30); record:
`data/probe_window_fastpath/summary.json`.

`results_lever_mechanics.md` reported the KV window making the draft SLOWER at
short context -- cost ratios of 1.013-1.020 against the unwindowed draft,
where physics says the saving is exactly zero because there is no KV to trim.
That figure came from the FITTED cost model, whose own held-out error is 0.53%
median, so a ~1.5-2% claim needed a direct A/B before anything was built on
it. This is that A/B, and then the fix.

## What it was not

The obvious suspect was a device sync. `_apply_draft_kv_window` returns
`cad.replace(..., _seq_lens_cpu=None, ...)`, and the `seq_lens_cpu` property
does `self.seq_lens.to("cpu")` when that cache is empty -- an H<->D sync, and
the property is deprecated for exactly that reason. Per draft step, with K=4,
that would be five syncs an armed step and easily the whole effect.

**It is not the cause.** The FlashAttention metadata builder never reads
`seq_lens_cpu`; only chunked-local attention and flex attention do, and
neither is on this path. Nulling the cache costs nothing here.

## What it was

`_apply_draft_kv_window` itself: roughly fourteen small kernel launches plus a
dataclass `replace`, PER DRAFT STEP, so about five times that per armed step.
At batch 1 the draft chain is ~27 ms, and that bookkeeping is ~1% of it.

## The fix

When sinks + window already cover the longest sequence, every row's `dropped`
is zero, the gather is the identity, and the returned view equals the input
field for field -- which is also exactly what the window-DISABLED path passes
down, so it is a shape the stack already handles. The condition is

```
cad.max_seq_len <= (n_sink + n_last) * block_size
```

`max_seq_len` is a Python int, so the test never syncs.

## Result

Two interleaved repeats of three arms -- windowed slow path, windowed fast
path, and a no-window control -- at `target-matching/w1024/skip0`, which
isolates the window lever. Draft-chain mean, ms:

| regime | batch | no window | window slow | window fast | total overhead | fast-path gain | **residual** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 1 | 27.281 | 27.560 | 27.307 | **+1.02%** | -0.92% | **+0.10%** |
| R6 | 32 | 30.966 | 31.253 | 30.984 | **+0.93%** | -0.86% | **+0.06%** |
| R8 | 16 | 28.776 | 29.124 | 28.778 | **+1.21%** | -1.19% | **+0.01%** |
| R5 | 8 (14k) | 49.146 | 29.601 | 29.512 | **-39.77%** | -0.30% | -39.95% |

**The overhead is gone.** What was +0.93 to +1.21% is now +0.01 to +0.10% --
a windowed draft at short context is no longer distinguishable from an
unwindowed one.

Three checks make that trustworthy:

* **Acceptance is bit-identical in every arm and regime** (R1 4.953488, R5
  34.08, R6 158.511628, R8 79.255814, all arms). The fast path changes which
  tensor the attention metadata points at, so this is the property that had to
  hold and it holds exactly.
* **The negative control behaves.** At R5 the fast path took 0 of 551 calls,
  because at 14k the window genuinely trims. Its -0.30% is therefore pure
  run-to-run noise, which usefully calibrates the floor: the short-context
  gains are 3-4x that.
* **The fast path fired on every call** at R1/R6/R8 (515/515 in both repeats),
  so "no change" could not have been confused with "never taken".

## A correction to the earlier figure

The fitted model put the window's short-context overhead at **1.5-2.0%**; the
direct A/B measures **0.93-1.21%**. The fitted number was high, which is
within the cost model's own error budget but worth stating: `results_lever_
mechanics.md`'s window-overhead line should be read as ~1%, not ~2%.

The same control also gives the window's benefit at long context by direct
measurement rather than by fit: **-39.77% of draft-chain time at 14k**,
against the fitted 0.632 ratio (-36.8%). The fit was conservative.

## Shipped

`VLLM_SELF_SPEC_DRAFT_WINDOW_FASTPATH` now defaults **on**. It is
acceptance-neutral by construction and by measurement, so no scored acceptance
result changes.

**Cost records do not transfer.** Every Phase 98 cost campaign was measured
with this off, so their windowed short-context `d_hat` is ~1% high relative to
what this build produces. That is a small effect in the direction of the fit
being conservative, but a re-fit is the honest way to close it, and until then
cross-build cost comparisons should say which side of this change they sit on.
