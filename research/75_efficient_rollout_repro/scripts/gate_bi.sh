#!/bin/bash
# Diagnostic: does batch-invariant (COMPILE_CONSISTENT=1) make greedy self-spec
# TOKEN-IDENTICAL to no-spec? If yes, the earlier E2 gate failure was FP argmax
# flips (batch-invariance off), not a plumbing bug -> tau is valid. Run BY PATH.
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/data/smcho/self-spec-moe/.venv/bin/python
W4="/data/smcho/ckpts/Qwen2.5-7B-Instruct-W4A16-INT4-sym"
source "$PHASE/scripts/env_e75.sh"
# Batch-invariant on BOTH arms directly. COMPILE_CONSISTENT only fires inside
# SpeculativeConfig (spec arm), so it would leave nospec on default kernels ->
# guaranteed mismatch. VLLM_BATCH_INVARIANT=1 makes nospec-decode and spec-verify
# use identical kernels, which is the precondition for bit-exact greedy spec.
export VLLM_BATCH_INVARIANT=1
"$PY" "$PHASE/scripts/check_lossless.py" --mode nospec --out "$PHASE/data/gate_bi_nospec.json"
"$PY" "$PHASE/scripts/check_lossless.py" --mode spec --draft "$W4" --gamma 5 --out "$PHASE/data/gate_bi_spec.json"
echo "=== COMPARE (batch-invariant ON) ==="
"$PY" "$PHASE/scripts/check_lossless.py" --compare "$PHASE/data/gate_bi_nospec.json" "$PHASE/data/gate_bi_spec.json"
echo "gate_bi DONE"
