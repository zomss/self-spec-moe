# G98-A — smoke gate result

Date: 2026-08-11

Verdict: **PASS, 5/5.** Every registered boot class initializes, keeps one
target-owned KV cache group, dispatches the PIECEWISE draft chain, and
generates. The preregistration's open assumption — that a quantized draft can
boot with shared target KV — is **verified**. The lattice stays at 30.

Non-scored. Grants no Round-1 or Round-2 authority.

## Result

| boot | quant | window | skip | KV blocks | chain | tokens |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | target-matching | off | 0 | 24,772 | PIECEWISE | 120 |
| A2 | target-matching | 512 | 0 | 24,772 | PIECEWISE | 120 |
| A3 | target-matching | off | 4 | 24,741 | PIECEWISE | 184 |
| A4 | **w4a16** | off | 0 | **22,190** | PIECEWISE | 157 |
| A5 | **w4a16** | 512 | 4 | **22,328** | PIECEWISE | 162 |

Every boot: one KV cache group, zero exceptions, every performed check true.

A5 is the strongest single result — quant, window and skip composed in one
boot, the first time all three axes have run together.

## What had to change to get here

The gate did its job by failing three times first, and each failure was a
different kind of obstacle. Recording them because the sequence is the finding:

1. **Operator bug.** `VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA` parses as a
   *string*, so the base environment's `"0"` is truthy and the contract read it
   as a declared replica. Phase 97 maps `"0" → ""`; the smoke runner did not.
2. **Policy wall.** The `minimal-b0` boot contract admitted only Phase 97's B0
   configuration — `SHARE_WEIGHTS=1`, window 0, no skipped layers. Phase 98's
   lattice was not merely untested but *structurally forbidden*. Fixed by
   registering the `w98-lattice` scope, bounded to the frozen lattice.
3. **Proof wall.** `validate_shared_weight_aliases` proves a draft *is*
   target-matching, which is false by design for skip and quant. Replaced per
   scope rather than disabled: **subset** for skip (every remaining parameter
   still aliases its twin, absences must be exactly the declared skip set) and
   **independence** for quant (no draft parameter may share target storage, so
   a half-aliased draft cannot pass unnoticed).
4. **Evidence wall.** The per-step record asserted the target and draft weight
   versions are equal. Under `w98-lattice` it now *records* what it found —
   `boot_scope`, `target_matching_weights`, and both version ids.

`minimal-b0` behaviour is unchanged throughout, and Phase 97's suite held at
its 38-failure baseline across every step.

## A false result that was reported and corrected

v3 reported **5/5 with the quant axis verified** while A4 and A5 generated
zero tokens. `run_boot` computed status from its checks alone, so a boot that
threw *after* its early checks passed was recorded as `pass` with an exception
attached. An exception now forces a fail, and the v3 record on disk was
corrected to 3/5 rather than left standing.

This is recorded because the failure mode is worth remembering: the aggregation
was optimistic by default, and the wrong answer looked exactly like the right
one.

## The number Round 2 needs

Shared-KV capacity is materially lower for the quantized boots, because a 5.7
GiB W4A16 draft is resident alongside the 15.27 GiB bf16 target:

| class | blocks | headroom over the 21,682-block Phase 97 floor |
| --- | --- | --- |
| target-matching | 24,772 | ~14% |
| skip-4 | 24,741 | ~14% |
| **quantized** | **22,190 / 22,328** | **~2.3% / ~3.0%** |

It clears, but with little room. Round 2's batch shapes and context lengths
should be chosen against the quantized figure, not the target-matching one.

## Known, not yet addressed

The same weight-equality assertion appears twice more on the **P4 capture**
path (`koff_runtime.py:1927` and `:2503`). Neither fires for a smoke gate that
enables no capture, but Round 2 uses the recorder for its KMAX streams and
will hit both. Changing what a *scored* capture may assert is a larger
decision than a passive record and is left for an explicit call.

## Artifacts

- `data/w98_g98a_authorization_v4.json`
- `data/g98_a_v4/g98a_result.json` and per-boot records
- consumed: `data/g98_a/finding.json`, `data/g98_a_v2/`, `data/g98_a_v3/`
  (the last corrected from its 5/5 misreport)
