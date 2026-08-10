# Phase 97 P4 — B0 native-sampler retry authorization V4

Status: **APPROVE for one fresh GPU-4 OFF/K4/w512 value-screen attempt. V3 is
consumed and both zero-capture outputs are immutable. The runner now forces and
preflights the PyTorch-native sampler before output creation. P4a, action
admission, and performance claims remain unauthorized.**

Date: 2026-08-09.

## Decision

The V3 attempt passed Ninja discovery but stopped while loading a cached
FlashInfer sampling module linked to an unresolved `libcudart.so.13`. It
emitted no captures, adapted rounds, or score. The failure was unrelated to
shared KV, resource capacity, acceptance, or candidate value.

V4 keeps the frozen model, GPU, engine, actions, matrix, scorer, shared-KV
floor, and package versions unchanged. It changes only the sampling-backend
policy used by the retry launcher:

- every boot forces `VLLM_USE_FLASHINFER_SAMPLER=0`;
- a boot spec cannot enable the FlashInfer sampler;
- an isolated subprocess constructs `TopKTopPSampler` and proves that it binds
  `forward_native`;
- that subprocess completes before the create-new output directory is made;
- every boot child rechecks the policy before importing the engine; and
- the existing virtualenv/Ninja preflight remains active.

This matches the frozen workload rather than changing its sampling behavior.
Every request has a per-request seed, and the bound vLLM sampler source already
falls back to the native path when per-request generators are present. V4
prevents only the unseeded dummy warmup from entering the irrelevant
FlashInfer JIT path.

The fresh scope remains:

```text
physical GPU                         4 only
physical boots                       9
cells per boot                      48
required complete captures         432
shared-KV launch/capture floor   21,682 blocks
sampler backend          PyTorch native only
fallback GPU                         forbidden
prior output reuse or resume         forbidden
P4a or admission authority           false
```

## Exact authorized invocation

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v4.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v3
```

No other retry, output path, sampler policy, or GPU is authorized. Any failure
again stops without adaptation or scoring and consumes V4.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_run_authorization_v4.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v4.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_capture_runner.py \
  research/97_composition_runtime/tests/test_p4_b0_run_authorization_v2.py \
  research/97_composition_runtime/tests/test_p4_b0_run_authorization_v3.py \
  research/97_composition_runtime/tests/test_p4_b0_run_authorization_v4.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/97_composition_runtime/tests -p 'test_*.py' -v
```

The focused authorization/runner suite passes 47 tests. The complete Phase 97
suite passes 284 tests. Ruff 0.14.0 format and check pass all changed Python
files. The live environment preflight reports
`{"backend":"native","flashinfer_enabled":false}` without executing a model
or creating the V3 output directory.

## Next step

Run only the exact V4 invocation. A successful run must emit all 432 captures,
`adapted_rounds.jsonl`, and `score.json` before producing
`p4_b0_value_screen_result`. Even a positive score cannot authorize P4a or
admit w512.
