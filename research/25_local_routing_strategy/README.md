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

## Next artifact

Confirm the shared-anchor lift on a DeepSeek/GLM/Llama-4-class shared-expert model
(smaller shared fraction than Qwen1.5-MoE's 45% -> proportionally smaller lift), and
fold the comm-free-draft beta (now raised) back into the Phase 24 speedup.
