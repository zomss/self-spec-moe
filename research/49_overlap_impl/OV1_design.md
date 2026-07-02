# OV1 design — free-running comm-free draft, verified-and-rewound

The lockstep cycle is relaxed, not replaced: lockstep is the degenerate schedule that
confirms/rewinds the draft after every K tokens and never runs ahead. OV1 lets the
draft chain keep producing tokens on a side stream while the verify runs, and moves
the confirm/rewind decision off the critical path.

## 1. Schedule

```
main stream : |== verify t (compute + A2A idle windows) ==|resolve| == verify t+1 ==|
draft stream: |== chain: b̂=d(K+1), e1..eK = d(K+2..2K+1) ==|        |== continue ===|
host        : launch verify async → launch K+1 draft steps on side stream →
              sync verify → rejection-sample → per-request hit/miss → assemble
```

Cycle time: `C_v + max(K·T_d, M_v)` (+ exposed re-draft for the miss cohort).

## 2. Why the dependency is breakable

The draft chain for cycle t+1 needs (i) the number of accepted tokens m, (ii) the
next committed token, (iii) draft-KV consistent with the committed prefix. The
most-likely outcome is m=K (P=beta^K≈0.90 at K=2, beta≈0.95) and bonus = the draft's
own next prediction d(K+1) (P≈beta). Because the draft's KV for d1..dK (and for
d(K+1) once it generates it) was written *by the draft itself* during generation, a
full-accept hit leaves the draft KV already correct — the "speculative" chain is
literally the same chain continuing. Hit rate ≈ beta^(K+1) ≈ 0.86/request (greedy).

## 3. Per-request resolution (batched; hits and misses coexist)

- **Hit** (accepted all K, bonus == d(K+1)): next verify's draft tokens =
  d(K+2)..d(2K+1), already computed; draft KV already advanced; nothing to do.
- **Miss** (first rejection at j, or bonus mismatch): committed prefix = accepted
  d1..dj + corrected token. Everything the draft computed past d_j is invalid →
  **rewind = the existing machinery**: `set_inputs_first_pass` already takes
  per-request `num_rejected_tokens` and rebuilds positions/seq_lens/slot mappings;
  stale draft-KV slots are overwritten before they are ever read (attention masks by
  seq_lens; slots are per-position). The miss cohort gets a fresh sub-batch
  propose — the only draft work left exposed, expected ≈ miss_rate·K·T_d ≈
  0.14·K·T_d per cycle.
- The next verify batch mixes per-request draft-token lists — the existing
  `SpecDecodeMetadata` path already supports per-request variable draft counts.

## 4. Engine changes (mapped to the code)

1. **Side stream + separate cudagraph pool** for the draft's captured graphs
   (concurrent replay of pool-sharing graphs is a data race — gate OV0b).
2. **Host loop** (`gpu_model_runner.execute_model` spec path): launch verify async →
   `drafter.run_ahead()` (K+1 chain steps on the side stream, uniform full batch,
   assume full-accept) → sync verify → rejection sampling → D2H of
   (num_accepted, bonus==b̂) per request → cohort split.
3. **Sub-batch propose for misses** (`llm_base_proposer`): index-select the miss
   rows, rebuild CommonAttentionMetadata for the subset (same shape of work as
   padded-drafter batching), run the normal step-0 + chain. Piecewise graphs absorb
   the variable sub-batch via existing size padding.
4. **DP handling**: the draft is comm-free (real=0), so per-rank cohort sizes may
   differ. The chain's DP batch coordination (`_determine_batch_execution_and_padding`
   → `coordinate_batch_across_dp`) is a collective and would deadlock on divergent
   participation → under the OV1 flag, skip DP coordination for ALL draft forwards
   (each rank pads to its own graph size). Safe because the draft issues no
   collectives; also removes the Phase-45 DP-sync barrier from the draft path.
5. **Buffers**: the ahead-chain (side stream) and the miss re-propose (main stream,
   later) both use the proposer's persistent buffers → main stream waits on the
   shadow/ahead event before propose touches them (event wait, not sync).
6. **Bonus-guess sampling**: greedy-only v0 (gate on `sampling_metadata.all_greedy`,
   like sample-in-graph). Under temperature the bonus is a random draw and the hit
   rate degrades to a sample-collision probability.

## 5. What OV1 does NOT change

Losslessness: the verify is untouched and still corrects everything; speculation is
scheduling-only (a miss wastes side-stream compute that was hidden anyway). The
verify's own cost — including the (K+1)x token inflation that bounds the
bandwidth-bound single-node regime to ~1.4-1.5x — is untouched; the accept_len-scale
prize remains the latency-bound (emulated / multi-node) regime, where the comm
window also pays for a larger K.

## 6. Expected numbers (from the Phase 24 cost model + Phase 48 curve)

| regime | lockstep | OV1 |
|---|---|---|
| native PCIe b64 | 1.03x | ~1.4x (hide min(chain, M_v)≈18 ms of 65 ms cycle) |
| native PCIe b512 | ~0.96-1.0x | ~1.4x (chain fully hidden; verify inflation binds) |
| emulated 500us (f≈0.6) | 1.43-2.12x | → accept_len·T_ns/T_v ≈ 2.4-2.6x |
| emulated 1000us + K=4 | — | toward accept(K4)=4.7 (window fits 4 forwards) |
