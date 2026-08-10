# Phase 97 P0/P1 result — shared-KV schemas and fail-closed invariants

Status: **PASS for the CPU/schema layer. All 19 targeted tests pass. The
validator is not yet connected to engine startup, so this is not runtime
integration evidence.**

Date: 2026-08-08.

## Question

Can Phase 97 represent only the agreed `B0`/`B1` shared-target-KV design and
fail closed before runtime when a manifest, action, or runtime snapshot
introduces private KV, mismatched layer bindings, or action-specific cache
state?

## Artifacts

| artifact | purpose |
| --- | --- |
| `schemas/boot_class.schema.json` | strict `B0`/`B1` boot manifest with `kv.path=shared_target`, target ownership, one pool, layer-twin bindings, and shared-KV environment requirements |
| `schemas/action.schema.json` | `OFF` plus weight/window/skip/`K` actions; no KV-path action field |
| `schemas/shared_kv_runtime.schema.json` | serialized evidence for one pool, identical cache specs/storage, one binding, and invariant true slot mapping |
| `scripts/validate_shared_kv.py` | JSON-Schema and semantic validation plus live-object storage/slot identity checks |
| `tests/test_shared_kv_invariants.py` | positive `B0`/`B1` cases and fail-closed negative cases |

All JSON objects use Draft 2020-12 schemas with
`additionalProperties=false` at every controlled object boundary. Semantic
validation handles cross-document constraints that JSON Schema cannot express,
including path references, layer-map equality, graph admission, and exact
runtime alias coverage.

## Fail-closed coverage

The tests reject:

1. `VLLM_SELF_SPEC_SHARED_KV=0`;
2. any nonempty `VLLM_SELF_SPEC_DRAFT_KV_DTYPE`;
3. a private KV path or a second KV pool;
4. a `B1` manifest without target-matching recourse;
5. action-level KV-path fields;
6. unknown weight paths or incorrect quantized-weight version policy;
7. target/draft cache-spec mismatch;
8. a missing layer from the boot-declared alias union;
9. action-specific binding, pool, or true-slot-mapping changes;
10. actual draft tensors that do not alias the target storage; and
11. actual action slot mappings that are not the canonical target mapping.

Positive cases cover both `B0` and `B1`, the full boot/action/runtime package,
and direct live-object alias validation.

## Command and result

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/97_composition_runtime/tests \
  -p 'test_shared_kv_invariants.py' -v
```

Result:

```text
Ran 19 tests in 0.764s
OK
```

No GPU process, model boot, or performance profiler was run.

## Limitations

- The phase-local validator is not yet invoked by a boot manifest generator or
  vLLM startup.
- Runtime snapshots are a defined interface; the live engine does not yet emit
  them.
- Direct tensor tests use CPU dummy objects. The validator also supports
  tensor-like objects exposing `untyped_storage`, but a live vLLM binding test
  remains required after integration.
- These tests do not establish co-resident `B1` weight-path switching, graph
  compatibility, HBM capacity, or performance.

## Decision

The shared-KV schema/invariant portion of P0/P1 passes. The active next
artifact is a concrete `B0`/`B1` manifest and resource-ledger projection wired
through this validator. Engine switching work remains gated on that preflight.
