# Phase 23: Activation-Quant (Communication-Axis) Draft Acceptance

Source: Phase 22 (weight-only FP4 acceptance = 0.92; harness/method reused),
Phase 20 (comm-amortization model), the communication-bound positioning.

## Objective

The EP all-to-all moves **activations**, so quantizing the dispatched/combined
activations shrinks communication directly (FP8 -> 2x, FP4 -> 4x), independent of
the weight quantization. In speculative decoding the draft can use activation
precisions too lossy to serve directly, because the bf16 verify corrects it. This
fills in the two missing rows of the positioning table:

```
How much acceptance does activation quantization (the comm lever) cost,
on top of / instead of weight quantization?
```

## Method

`activation_quant_acceptance.py` monkeypatches `Qwen3MoeExperts.forward` to
fake-quantize, per-token, the dispatch input and each expert's combine output, under
FP8 / NVFP4 / MXFP4, combined with bf16 or NVFP4 weights. Acceptance = rejection
sampling vs the exact bf16 target at 122 positions (same metric as Phase 18/22). The
`w4a16` config reproduces Phase 22 as a built-in sanity check on the patched forward.

```bash
V=/data/smcho/self-spec-moe/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $V activation_quant_acceptance.py \
  --model Qwen/Qwen3-30B-A3B --local-files-only \
  --output-json data/qwen3_30b_actquant.json
```

## Result (see `results_activation_quant.md`)

**Activation quant buys the comm reduction almost for free.** FP8 activations cost
~zero (W4A8 0.915 = W4A16 0.913 within noise -> free 2x comm); NVFP4 activations cost
~0.014 (W4A4 0.899 -> 4x comm), both **memory-free**. MXFP4 activations are worse
(0.866) -- use NVFP4 on the wire. So the comm lever is cheap and is the first move on
a comm-bound system, with local routing (eliminate, spend HBM) as the finisher only
when needed.

## Next artifact

The acceptance (cost) side of every lever is now measured (Phases 22-23). The
remaining piece is the **benefit** side on a comm-bound link: f vs batch on the PCIe
/ inter-node all-to-all (P2P-disabled real run or Phase-20-style injection), and the
resulting end-to-end speedup of the activation-quant + local-routing draft.
