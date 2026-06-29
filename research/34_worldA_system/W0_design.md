# W0 design: lockstep self-spec via the V1 `draft_model` path (NOT a custom driver)

**Decision (corrects the RE-SCOPE "custom driver"):** the earlier W0 code-read rejected the
**V2 `AutoRegressiveSpeculator`** (head-oriented: hidden-state-conditioned forward; empty
`draft_attn_layer_names` for a self-draft). But vLLM's **V1 `draft_model`** path is a
*different*, purpose-built "separate full model as the draft" mechanism, and it FITS a
full-model self-draft. So W0 is NOT a from-scratch driver -- it is the framework's
draft_model path + our already-merged W1 flag, wired in.

## Why it fits (the two V2 blockers are both gone)
- `DraftModelProposer` has `pass_hidden_states_to_model=False` (`v1/spec_decode/draft_model.py:27`)
  -> the draft runs plain `self.model(input_ids, positions, inputs_embeds)` with NO
  hidden_states kwarg (`v1/spec_decode/llm_base_proposer.py:651-667`). (V2 blocker 1 gone.)
- The draft is a separate full instance, so `_draft_attn_layer_names` is NON-empty
  (`llm_base_proposer.py:1250`) -> it gets its OWN attention layers + KV. (V2 blocker 2 gone.)
- draft == target checkpoint is allowed: only vocab-size equality is asserted
  (`speculative.py:1060-1074`); `_maybe_share_embeddings/lm_head` are no-ops
  (`draft_model.py:80-88`) -> full second instance (2x weights, "bf16 full replica" = our
  Stage-1 plan).
- Chain of `num_speculative_tokens` (`llm_base_proposer.py:613`) = World A's chain draft.

## What the framework gives for FREE (the custom-driver risk list evaporates)
verify forward; **lossless rejection sampling** (`v1/sample/rejection_sampler.py:394`,
called `gpu_model_runner.py:3588`); accept/commit (`parse_output`, `_bookkeeping_sync`);
**draft-KV slot management + rollback** of rejected tokens (`llm_base_proposer.py:1077-1182`);
scheduler/continuous-batching integration; cudagraphs. We do NOT manage scratch-slot KV
ourselves. `draft_model` forces the V1 runner (`config/vllm.py:2031-2032`), which fully
supports MoE+EP for the target (our forced-PCIe AgRs + W1 flag already operate there).

## The changes to make (small, surgical)
1. **Thread `additional_kwargs` through the public `set_forward_context`**
   (`forward_context.py:272-282` + pass it at the `create_forward_context` call ~`:331-341`).
   The field + reader + `create_forward_context` arg already exist (`:180,210-223,234,252`);
   only the public context manager doesn't forward it. ~3 lines.
2. **Inject the flag at the draft's 3 forward sites** in `llm_base_proposer.py`:
   step-0 prefill `:514`, decode loop `:659`, dummy/capture `:535` ->
   `additional_kwargs={"self_spec_local_route": True}`. The consumer side (router masking
   `moe_runner.py:559`, AgRs skip `all2all.py:153/199/237`) is already merged (W1). ~3 lines.
3. **Propagate EP into the draft parallel config** -- THE one real gotcha.
   `create_draft_parallel_config` (`speculative.py:985-1003`) builds a fresh ParallelConfig
   that DROPS `enable_expert_parallel` (defaults False) (+ DP fields). Since the draft's
   FusedMoE reads the draft parallel config, the draft builds `use_ep=False` ->
   `expert_map is None` -> **the W1 flag is a silent no-op on the draft.** Fix: copy
   `enable_expert_parallel` (+ `data_parallel_size`/`all2all_backend` if DP+EP) in
   `draft_model.py:_create_draft_vllm_config` (`:54-66`) or `create_draft_parallel_config`.
   The unconditional TP-match raise (`draft_model.py:36-51`) is NOT a blocker as long as
   `draft_tp == target_tp` (which we want: EP=TP). ~3-5 lines.

## Run config
target: DeepSeek-V2-Lite `--enable-expert-parallel --tensor-parallel-size N`;
`speculative_config = {method: "draft_model", model: <same checkpoint>,
num_speculative_tokens: K, draft_tensor_parallel_size: N}`.

## Build & validate ladder
- **(i) Increment 0 -- wiring sanity:** draft=target, FULL routing (flag OFF). Confirm the
  draft_model lockstep runs and is ~lossless with beta≈1 (draft==verify). Isolates framework
  wiring from the comm-free path.
- **(ii) EP on the draft:** add change 3; confirm the draft builds `use_ep=True` + non-None
  `expert_map` (log it).
- **(iii) Comm-free draft:** add changes 1-2; with the flag ON, the draft forward issues
  `real=0` AgRs collectives (existing `VLLM_SELF_SPEC_LOG_A2A_COUNTS` counter,
  `all2all.py:255`) while the VERIFY stays full-EP (real>0), output stays LOSSLESS vs the
  no-spec greedy baseline (rejection sampling corrects the lossy local-routing draft).
- **(iv) -> W7:** tokens/s on forced-PCIe vs no-spec baseline.

## Caveats
- 2x weights + 2x KV (separate draft instance). Acceptable for the demo; and it's a *feature*
  for W2 -- the draft can hold its OWN FP4 resident experts independent of the bf16 verify
  shard (the dual-residency design). But it means the draft's experts = shard (low coverage,
  low beta) until W2.
- `quant_config=None` for the draft (`draft_model.py:60`): fine for bf16; override for W2 FP4.
- Greedy-target is lossless out of the box; for temperature sampling confirm the draft-probs
  path is enabled (`llm_base_proposer.py:236-239`) so rejection stays lossless.

## Status: DONE, validated, merged (worktree ssm-w0 -> research/self-spec-moe)
5 files, ~70 lines, additive + flag-gated; default draft_model behavior unchanged on non-EP.

**Design correction made during the build:** gate the draft injection on a NEW dedicated
producer env `VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE` (NOT the W1 reader env
`VLLM_SELF_SPEC_LOCAL_ROUTE`). The reader env is a global fallback, so using it would also
make the VERIFY forward (which carries no `additional_kwargs` key) go comm-free -> broken
full-EP verify. Run config: `VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1`, reader env stays 0.
`set_forward_context` now merges caller `additional_kwargs` over the platform context
(caller wins). EP propagation is unconditional for draft_model+EP targets (changes the
draft from TP-sharded-experts to EP-sharded) -- desired here; gate it on the flag if
upstreaming.

**Validation (DeepSeek-V2-Lite, native deepseek_v2, greedy):**
- (i) Lockstep + accept: full-routing draft (producer OFF) -> **acceptance 1.0, accept len
  4/4 (beta=1)**; local-routing draft (producer ON) -> **0.758** (verify corrects the lossy
  draft). Lockstep works.
- (ii) Draft EP: both workers log `use_ep=True expert_map=set`. The W1 flag is now live on
  the draft.
- (iii) Comm-free (DP=2+EP AgRs path; pure TP+EP has no AgRs so DP is needed to exercise the
  W1 skip): producer OFF `real=7280` -> producer ON **`real=1820`** (draft's ~75% of
  collectives eliminated; verify retains full-EP). Holds under forced-PCIe.
- **Losslessness (honest):** lossless in the rejection-sampling sense -- every output token
  is the verify's greedy argmax given the accepted prefix. NOT bit-identical to no-spec
  greedy, but that gap is a **pre-existing vLLM property**: on UNMODIFIED main the
  draft_model spec path already diverges from no-spec greedy in 4/16 prompts (batched-verify
  near-tie FP flips). Producer-OFF reproduces main exactly (16/16); producer-ON adds 1 extra
  near-tie flip (15/16 vs producer-OFF). So W0 is **as lossless as vLLM's native spec
  decoding** -- consistent with the B1 greedy-numerics caveat (batched-vs-sequential MoE FP
  non-associativity). The paper's "lossless" = the standard distribution-preserving
  rejection-sampling guarantee + this documented FP caveat.

**Next:** W2 (FP4 resident cache to raise the 0.758 draft acceptance) + W7 (tokens/s vs
no-spec baseline, measured in the **DP+EP layout** where the W1 comm-skip lives).
