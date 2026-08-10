# Phase 97 P4 — B0 value-screen retry authorization V3

Status: **APPROVE for one fresh GPU-4 OFF/K4/w512 value-screen attempt. V2 is
consumed and its zero-capture output is immutable. The V3 launcher repair and
tool preflight pass, and only the create-new V2 output path is executable.
P4a, action admission, and performance claims remain unauthorized.**

Date: 2026-08-09.

## Decision

The V2 attempt stopped during the first engine's sampler profile because the
installed `.venv/bin/ninja` was absent from the child `PATH`. It emitted zero
captures and no score. The failed output is preserved and cannot be resumed,
overwritten, or used as evidence.

V3 keeps the frozen model, GPU, engine, actions, matrix, shared-KV floor, and
scorer unchanged. Its only executable repair is to:

- derive the executable directory from `sys.prefix/bin`;
- prepend it to each boot child's `PATH`;
- require `ninja` to resolve from that directory and pass `--version`; and
- complete that preflight before creating the new output directory.

The fresh scope remains:

```text
physical GPU                         4 only
physical boots                       9
cells per boot                      48
required complete captures         432
shared-KV launch/capture floor   21,682 blocks
fallback GPU                         forbidden
V1 output reuse or partial resume    forbidden
P4a or admission authority           false
```

## Exact authorized invocation

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v3.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v2
```

No other retry, output path, or GPU is authorized. A failure again stops
without scoring and consumes V3.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_run_authorization_v3.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v3.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_capture_runner.py \
  research/97_composition_runtime/tests/test_p4_b0_run_authorization_v3.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/97_composition_runtime/tests -p 'test_*.py' -v
```

The focused launcher/V3 suite passes 18 tests. The complete Phase 97 suite
passes 267 tests. Ruff check and format checks pass the changed Python files.
The validator confirms the zero-capture unscored V2 disposition, current
source hashes, Ninja preflight, nine boots, 432 captures, GPU 4 only, and an
absent create-new V2 output directory.

## Next step

Run only the exact V3 invocation. A successful run must emit all 432 captures,
`adapted_rounds.jsonl`, and `score.json` before producing
`p4_b0_value_screen_result`. Even a positive score cannot authorize P4a or
admit w512.
