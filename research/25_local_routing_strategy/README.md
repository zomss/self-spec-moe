# Phase 25: Local-Routing Strategy

Source: Phase 08/final-state (shared-expert anchor = the one untested local-routing
lever), Phase 18/22 (quant + acceptance method), Phase 24 (the comm-free local-routing
draft is beta-limited).

## Objective

The comm-free local-routing draft (best for comm-bound serving) is beta-limited by
expert coverage (~0.8 at 0.5E cache). Investigate two ways to raise it:

```
1. shared-expert anchor: does an always-local shared expert lift local-routing acc?
2. FP4 cache scaling:    does local routing reach the quant floor (0.92) as C -> E?
```

## Method

Rejection-sampling acceptance vs the bf16 full-routing target (Phase 18/22 method).
Local routing = router restricted to a per-layer top-C local cache (by request gate
mass), patched to match the real router exactly (softmax over all -> restrict
selection -> top-k -> optional norm; validated by C=E -> beta=1).

- `shared_anchor.py` (Qwen1.5-MoE-A2.7B): sweep routed cache C, `with_shared` vs
  `without_shared` (shared ablated in both), + measure shared mass fraction.
- `fp4_local_curve.py` (Qwen3-30B-A3B): sweep cache C, bf16 vs NVFP4 weights.

```bash
V=/data/smcho/self-spec-moe/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $V fp4_local_curve.py --local-files-only \
  --output-json data/qwen3_fp4_local_curve.json
CUDA_VISIBLE_DEVICES=1 $V shared_anchor.py --local-files-only \
  --output-json data/qwen15_shared_anchor.json
```

## Result (see `results_local_routing.md`)

Both levers work. **Shared-expert anchor** (Exp 1, Qwen1.5-MoE, 45% shared mass): lifts
local-routing acceptance by ~0.09-0.13, up to **2.5x at tiny cache** (0.067E: 0.38 vs
0.15) -- the always-local shared mass dilutes routed-coverage error, no replication
cost (lift scales with the shared fraction). **FP4 cache scaling** (Exp 2, Qwen3):
local routing interpolates coverage-limited -> quant floor; **NVFP4 -> 0.92 at full
coverage**, bf16 -> 1.0. The strongest comm-free draft combines both (shared anchor +
FP4 routed cache).

## Follow-up (done) -- anchor confirmed on 3 architectures

- `deepseek_shared_anchor.py`: DeepSeek-V2-Lite (softmax gating, ~48% shared) ->
  +0.25 to +0.58 lift.
- `deepseek_v3_shared_anchor.py`: Moonlight-16B-A3B (DeepSeek-V3 sigmoid+noaux_tc
  gating, ~65% shared) -> +0.13 to +0.54 lift.
- `alpha_shared_curve.py`: emulating a smaller fraction by scaling the shared output
  *failed* (intermediate scaling is OOD; only endpoints valid).

Anchor is robust across Qwen2-MoE / DeepSeek-V2 / DeepSeek-V3 and 2 gating types.

## Next artifact (hardware-gated)

All 3 runnable shared-expert MoEs are high-fraction (45-65%). The **small-fraction**
regime (DeepSeek-V3 1-shared ~11%, GLM-4.5) is frontier-scale only -> needs a
multi-GPU / rental testbed to confirm "small shared -> smaller lift" (a
mechanism-backed prediction). Then fold the raised comm-free-draft beta back into the
Phase 24 speedup.
