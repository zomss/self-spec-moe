# Phase 97 P4 — chunked-prefill probe attempt V2

Date: 2026-08-09

Status: **STOPPED after one GPU-4 engine boot; V2 is consumed, no cohort ran,
and no probe result or score exists**

## Outcome

The exact V2 command was executed once. Parent preflight passed, the child
booted the compiled target-matching K4 engine on physical GPU 4, shared all 36
draft attention layers with the target-owned KV cache, captured CUDA graphs,
and completed the resource precheck. Registration of the first marked cohort
request then failed closed before any scheduler work.

The engine retained the authorized configured bound
`max_num_batched_tokens=8192`. Draft-model speculative decoding reserves one
extra slot for each of 32 possible sequences, so vLLM correctly set the
effective scheduler budget to `max_num_scheduled_tokens=8160`. This is the
same 8,160-token budget registered by the CPU design. The new live admission
guard nevertheless compared the effective value with 8,192 and rejected it.

V2 was not retried or resumed. No fallback GPU, V10, scoring, P4a work, action
admission, or performance claim is authorized.

## Diagnostic evidence

Before the request-level refusal, the boot observed:

- 291 target/draft weight aliases;
- 36 draft layers bound to the target KV cache and no private draft KV;
- a 0.55 GiB actual CUDA graph pool versus a 0.47 GiB estimate;
- 392,432 target-KV tokens, or 24,527 blocks at the 16-token block size; and
- zero GPU memory and zero compute processes after the stopped child exited.

The 24,527-block observation exceeds the registered 21,682-block floor, but it
is diagnostic partial evidence only. The child emitted no `probe_result.json`,
and none of R4, R5, or R5cot reached the cohort barrier.

## Test gap

The live scheduler regression constructed `Scheduler` directly with an
8,192-token batch bound. It did not apply
`VllmConfig._set_max_num_scheduled_tokens`, so the test scheduler fell back to
an effective value of 8,192 rather than the real draft-model value of 8,160.
That made the incorrect guard comparison invisible in CPU review.

## Required repair and next gate

The repair must keep two quantities distinct:

1. require configured `scheduler_config.max_num_batched_tokens == 8192`;
2. independently require the registered effective scheduler budget of 8,160;
3. report configured and effective budgets under separate evidence fields; and
4. add an execution-path regression that applies real speculative-slot
   normalization before registering the first cohort request.

After focused and Phase 97 CPU validation, the repaired sources would require
a fresh source-bound V3 authorization and a new create-only output. This
document does not authorize that probe or its execution.

That repair and CPU review later passed. The separate V3 authorization is
documented in `results_p4_b0_chunked_prefill_probe_authorization_v3.md`; V3
remains unexecuted and this immutable V2 output remains untouched.

## Immutable artifacts

- failure record:
  `data/p4/run_b0_chunked_prefill_probe_v2/failure.json`
  (`3558f037e909f4ac418822e7f60e82e78f25b6bdd73f383549b86a7386732714`);
- child log:
  `data/p4/run_b0_chunked_prefill_probe_v2/child.log`
  (`86270e6be220b61fa9daf2a79ccab7577b79a4b74b03d87494de4e163b4cf1db`);
- K/OFF trace header:
  `data/p4/run_b0_chunked_prefill_probe_v2/koff_trace.jsonl`
  (`7563679bf88676f33034225235e5cc3d072038f72353fe1c420def9a6fae5743`);
  and
- preparation:
  `data/p4/run_b0_chunked_prefill_probe_v2/preparation.json`
  (`19fe2dfece118b8f4f008833f57033a3b1539b038f9c0ea60dc564894bbc807c`).
