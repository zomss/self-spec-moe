# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib as _contextlib
import copy
import os
from importlib.util import find_spec
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    replace,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import (
    SELF_SPEC_DRAFT_SCRATCHPAD_KEY,
    SELF_SPEC_LOCAL_ROUTE_KEY,
    SELF_SPEC_NODE_LOCAL_KEY,
    BatchDescriptor,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.model_executor.models.qwen3_eagle3 import Eagle3Qwen3ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import (
    empty_exponential_noise_like,
    sample_with_exponential_noise,
)
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.self_spec_profiler import (
    get_profiler as _self_spec_profiler,
)
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
    copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata,
    extend_all_queries_by_N,
    next_power_of_2,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class _DraftDecodeForwardSample(nn.Module):
    """W7 sampling-in-graph runnable for the draft FULL cudagraph.

    Wraps the (unwrapped) draft model so the decode-step forward, the
    ``compute_logits`` GEMM and the greedy ``argmax`` are recorded as a single
    captured graph -- mirroring V2's ``AutoRegressiveSpeculator._generate_draft``
    which captures forward + sample + update as one graph. Only the greedy
    argmax path is captured (probabilistic draft sampling stays eager).

    Returns ``(last_hidden_states, hidden_states, draft_token_ids)`` so the
    proposer reads the sampled tokens straight out of the graph output with no
    separate eager ``compute_logits`` launch per step.

    Pointer stability: the inputs are the proposer's persistent buffers (same as
    the plain-forward FULL graph), so the captured graph replays correctly.
    """

    def __init__(self, model: nn.Module, use_local_argmax_reduction: bool,
                 vres_map: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.use_local_argmax_reduction = use_local_argmax_reduction
        # Phase 85: sliced-vocab draft -- remap argmax local->global INSIDE
        # the captured graph (a gather; pointer-stable buffer).
        self.vres_map = vres_map
        logger.info("[vres] _DraftDecodeForwardSample built: vres_map=%s",
                    "None" if vres_map is None else tuple(vres_map.shape))

    def forward(self, **model_kwargs: Any):
        ret = self.model(**model_kwargs)
        if isinstance(ret, tuple):
            last_hidden_states, hidden_states = ret
        else:
            last_hidden_states = ret
            hidden_states = ret
        if self.use_local_argmax_reduction:
            draft_token_ids = self.model.get_top_tokens(last_hidden_states)
        else:
            draft_token_ids = self.model.compute_logits(last_hidden_states).argmax(
                dim=-1
            )
            if self.vres_map is not None:
                draft_token_ids = self.vres_map[draft_token_ids]
        return last_hidden_states, hidden_states, draft_token_ids


class SpecDecodeBaseProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        self._share_mtp_indices = False
        # Reference to the GPUModelRunner that owns this proposer. Used by the
        # W7 draft FULL-cudagraph path to reach the runner's persistent
        # block-table / seq-len buffers when building capture-time attention
        # metadata (see _build_draft_decode_capture_metadata).
        self.runner = runner

        # Self-spec W0: signal the comm-free local-routing MoE path on the draft
        # forward only (verify runs in a separate forward context with no key, so
        # it stays full-EP). Value-driven via ForwardContext.additional_kwargs;
        # gated to the draft_model method (so EAGLE/MTP heads are unaffected) AND
        # to the opt-in VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE env so default
        # draft_model behavior is unchanged. Note we gate on the PRODUCER env,
        # not VLLM_SELF_SPEC_LOCAL_ROUTE (the reader fallback) -- that one must
        # stay 0 so the verify forward does not inherit local routing. None when
        # inactive -> no change to the forward context.
        # Phase 54: DRAFT_NODE_LOCAL selects the node-local (intra-node EP)
        # draft instead; mutually exclusive with DRAFT_LOCAL_ROUTE (which wins
        # if both are set). Note the node-local draft DOES issue intra-node
        # collectives, unlike the comm-free device-local path.
        self._draft_forward_additional_kwargs: dict[str, Any] | None = (
            {SELF_SPEC_LOCAL_ROUTE_KEY: True}
            if (
                self.method == "draft_model"
                and envs.VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE
            )
            else {SELF_SPEC_NODE_LOCAL_KEY: True}
            if (
                self.method == "draft_model"
                and envs.VLLM_SELF_SPEC_DRAFT_NODE_LOCAL
            )
            else None
        )

        # Phase 55: per-cycle memo for coordinate_batch_across_dp (see
        # VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD). Cleared at propose() entry;
        # consulted only inside a real propose (dummy runs always coordinate,
        # keeping mixed steps and capture symmetric across ranks).
        self._dp_coord_memo: dict = {}
        self._in_propose = False

        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # DeepSeek V4 MTP consumes the target's pre-hc_head residual stream,
        # shape (T, hc_mult * hidden_size). Expand the hidden_states buffer
        # so target_hidden_states fits; detect DeepseekV4 via draft hf_config.
        draft_hf_config = self.draft_model_config.hf_config
        if hasattr(draft_hf_config, "compress_ratios") and hasattr(
            draft_hf_config, "hc_mult"
        ):
            self.hidden_size = self.hidden_size * draft_hf_config.hc_mult

        # Unifying eagle, draft model, and parallel drafting support.
        # DFlash always uses parallel drafting (all tokens in one pass),
        # but has an additional slot for the next_token_id (does not shift like EAGLE)
        self.parallel_drafting: bool = self.speculative_config.parallel_drafting
        self.extra_slots_per_request = (
            1 if not self.parallel_drafting else self.num_speculative_tokens
        )
        self.net_num_new_slots_per_request = self.extra_slots_per_request - (
            1 if (self.pass_hidden_states_to_model and self.method != "dflash") else 0
        )
        self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0

        # When True, all draft steps reuse the same position as the
        # first step instead of advancing by one each iteration.
        # Used by draft models with Q-only attention that share KV
        # with the target and always predict from the same position.
        self.constant_draft_positions: bool = False

        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
        if self.parallel_drafting:
            self._init_parallel_drafting_params()
        self.use_local_argmax_reduction: bool = (
            self.speculative_config.use_local_argmax_reduction
        )
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel

        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens, dtype=np.int32)

        # Can be specialized by methods like DFlash to reduce the limit
        self.max_query_tokens = self.max_num_tokens
        self.max_positions = self.max_num_tokens

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.draft_attn_groups: list[AttentionGroup] = []
        self.kv_cache_gid: int = -1
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        self.compilation_config = self.vllm_config.compilation_config

        # W7: opt-in FULL cudagraphs for the draft decode-step forwards (see
        # VLLM_SELF_SPEC_DRAFT_FULL_CG). Default off -> PIECEWISE as before.
        self.use_full_cudagraphs = envs.VLLM_SELF_SPEC_DRAFT_FULL_CG
        # Phase 55: cache of step0 FULL-capture constants (strided
        # query_start_loc tensors referenced by captured graphs), keyed by
        # (num_reqs, qlen); doubles as the keepalive.
        self._step0_qsl_cache: dict = {}
        # Phase 55: the step-0 FULL graph reads seq_lens through this
        # proposer-owned buffer (extend_all_queries_by_N returns a fresh
        # seq_lens tensor each cycle, so the runner's buffer can't be the
        # captured pointer). Replay copies the live extended seq_lens in and
        # zeroes the padded-request rows. _step0_captured records the FULL
        # uniform shapes actually captured (guards against dispatching a
        # keyed-but-never-captured shape at runtime).
        self._step0_seq_lens: torch.Tensor | None = None
        self._step0_captured: set[int] = set()
        # Phase 55 (K>=2): FULL uniform-decode shapes actually captured for
        # the chain graphs (model + fws). The capture pass drives the drafter
        # at num_reqs of each runner uniform desc (= runner_size/(K+1)
        # padded), so the dispatcher's largest FULL keys are never captured;
        # the chain claims uniform only for captured shapes (pre-flight in
        # propose) -- a keyed-but-uncaptured shape would attempt a forbidden
        # runtime capture in the wrapper. Capture is lockstep on all ranks,
        # so the claim stays rank-symmetric.
        self._full_captured: set[int] = set()
        # Debug: per-dispatch shape/mode logging (W7_STEP0_DEBUG=1) and
        # runtime force-piecewise bisect knob (W7_STEP0_FORCE_PW=1).
        self._step0_debug = bool(os.environ.get("W7_STEP0_DEBUG"))
        self._step0_force_pw = bool(os.environ.get("W7_STEP0_FORCE_PW"))
        # Phase 55 debug: tag the draft's Python-side collectives with their
        # source (propose-step0 / chain / dummy<i>) for the [a2a-dbg] lines.
        self._a2a_dbg = bool(os.environ.get("W7_A2A_DEBUG"))
        if envs.VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG:
            self._step0_seq_lens = torch.zeros(
                self.max_batch_size, dtype=torch.int32, device=device
            )
        # W7-gqa: when True, the FULL-CG draft chain runs its attention EAGERLY
        # (per-step kernel launch) instead of replaying the captured decode
        # graph. Set in initialize_attn_backend for non-MLA (FA3 / GQA) draft
        # backends, whose decode kernel freezes its work distribution at capture
        # and can't replay the draft's intra-loop growing sequence. MLA keeps
        # the captured graph (fg2). Env override W7_GQA_FORCE_EAGER_ATTN.
        self._draft_chain_force_eager_attn = False
        # W7-piecewise: when the chain forces eager attention (above), run the
        # model BODY on captured PIECEWISE graph pieces instead of the whole
        # forward eager (NONE). Attention still runs eager (splitting op) so
        # numerics / accept are byte-identical to the NONE chain, but the GEMM/
        # MoE body replays a captured graph (fast). Opt-in; default off keeps the
        # NONE chain. See VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE.
        self._draft_chain_piecewise = envs.VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE
        self._logged_chain_pw = False
        # Phase 62: window-KV drafting (StreamingLLM-style). When > 0, draft
        # forward steps attend only over sink + trailing-window KV pages: each
        # request's block-table row is compacted into proposer-owned buffers
        # and the draft-side kv seq_len shrunk to the tokens present in the
        # kept pages (see _apply_draft_kv_window). Verify metadata untouched.
        self._kv_window = envs.VLLM_SELF_SPEC_DRAFT_KV_WINDOW
        self._kv_window_sinks = envs.VLLM_SELF_SPEC_DRAFT_KV_SINKS
        self._win_block_table: torch.Tensor | None = None
        self._win_seq_lens: torch.Tensor | None = None
        self._win_col_arange: torch.Tensor | None = None
        # Phase 65: persistent temporaries for the per-step compaction
        # (avoids reallocating the [bs, n_cols] gather-index tensor per step).
        self._win_src_cols: torch.Tensor | None = None
        self._win_col_ge_sink: torch.Tensor | None = None
        # Phase 65: lightweight chain metadata (build once per chain, update
        # data in place per step). FA backends only; see envs.py.
        self._chain_light_md = envs.VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD
        self._slot_mapping_dict_cache: dict[int, dict[str, torch.Tensor]] = {}
        # Phase 66: shared-KV self-draft. The draft's attention layers bind to
        # the TARGET layers' KV tensors (registered in load_model via the
        # runner's shared_kv_cache_layers); the duplicate 48-layer allocation
        # disappears and the pool is sized target-only. Write discipline: the
        # step-0 slot mapping is PAD-masked down to the appended sampled-token
        # slot (every other step-0 token was just verify-written -- the draft
        # must not clobber target-exact KV); chain steps keep their writes
        # (provisional, overwritten by the next verify pass over the same
        # slots). draft_model (self-spec) method only.
        self._shared_kv = (
            envs.VLLM_SELF_SPEC_SHARED_KV and self.method == "draft_model"
        )
        if envs.VLLM_SELF_SPEC_SHARED_KV and self.method != "draft_model":
            raise ValueError(
                "VLLM_SELF_SPEC_SHARED_KV requires the draft_model "
                f"(self-spec) method, got {self.method!r}: only a draft that "
                "IS the target model has position-compatible KV."
            )
        # Phase 67: with shared KV, run the step-0 draft forward as a q=1 decode
        # of only the appended sampled token per request (the K+1 re-ingested
        # verify tokens are wasted FLOPs -- their KV writes are PAD-masked and
        # their hidden states are discarded). Only the appended token's hidden
        # state feeds draft-1 sampling. The compacted decode reuses the same
        # windowed seq_lens/block_table so the appended token attends to an
        # identical (bf16 cached) key set; bit-exact at window=0, but the
        # varlen->decode kernel switch perturbs accept in the windowed regime
        # (see envs.py + research/67_propose_fixed_opt). Requires shared KV +
        # decode-shaped propose. Default off.
        self._shared_kv_step0_decode = (
            self._shared_kv and envs.VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE
        )
        self._step0_decode_logged = False
        # draft layer name -> target layer name (filled by
        # _register_shared_kv_layers at load).
        self._shared_kv_target_layer: dict[str, str] = {}
        self._kv_window_warned = False
        self._kv_window_debug = bool(os.environ.get("W7_KV_WINDOW_DEBUG"))
        self._kv_window_calls = 0
        # Phase 69: window-scratchpad FULL-CG draft chain attention. Each chain
        # step gathers the sinks+window KV into a FIXED-shape dense scratchpad
        # and runs a masked SDPA-style attention (CUDA-graph-capturable) instead
        # of the paged FA3 decode kernel. Composes with (requires) the KV window.
        self._draft_fullcg = (
            envs.VLLM_SELF_SPEC_DRAFT_FULLCG and self.method == "draft_model"
        )
        if self._draft_fullcg and self._kv_window <= 0:
            raise ValueError(
                "VLLM_SELF_SPEC_DRAFT_FULLCG requires "
                "VLLM_SELF_SPEC_DRAFT_KV_WINDOW > 0 (the scratchpad materialises "
                "the sinks+window key set)."
            )
        # Persistent [1, cap] slot arange for the scratchpad padding mask and the
        # per-cycle scratchpad context (references the window buffers, updated in
        # place each step). Built lazily in the chain.
        self._sp_col_arange: torch.Tensor | None = None
        self._sp_ctx = None
        self._sp_logged = False
        # W7 sampling-in-graph: FULL-wrapped forward+compute_logits+argmax
        # runnable for the decode-step chain (set by the runner's wrap pass when
        # use_full_cudagraphs). None -> plain-forward path (sampling stays
        # eager). Enabled by default whenever the draft FULL graph is on; the
        # capture (dummy_run) and replay (decode loop) paths both use it.
        self._decode_fwd_sample: nn.Module | None = None
        # W7 A/B knobs (default ON under FULL-CG): allow disabling each per-step
        # reduction independently on the SAME binary for clean measurement.
        # 0 -> keep the reduction (default); 1 -> revert to the old behavior.
        self._disable_skip_rebuild = bool(
            int(os.environ.get("W7_DISABLE_SKIP_REBUILD", "0"))
        )
        self._disable_sample_in_graph = bool(
            int(os.environ.get("W7_DISABLE_SAMPLE_IN_CG", "0"))
        )
        # Phase 82: WHOLE-CHAIN capture -- the K-1 chain steps recorded as
        # ONE CUDA graph (single launch per cycle; chain cost becomes
        # immune to CPU dispatch speed). Requires the scratchpad chain
        # (static metadata) + greedy. Captured lazily per batch size on
        # the first eligible propose; the capture pass records (garbage
        # values) and is immediately replayed for real values. The
        # step-0 -> chain handoff goes through a persistent seed buffer
        # so replay reads a stable address.
        self._wholechain = (
            envs.VLLM_SELF_SPEC_DRAFT_WHOLECHAIN
            and envs.VLLM_SELF_SPEC_DRAFT_FULLCG
            and self.method == "draft_model"
        )
        self._wc_graphs: dict[int, dict] = {}
        self._wc_failed: set[int] = set()
        self._wc_seed: torch.Tensor | None = None
        # Phase 69: the window-scratchpad chain uses the plain-forward FULL graph
        # + EAGER sample, not the combined forward+compute_logits+argmax (fws)
        # graph. The fws graph replayed K-1 times with the scratchpad collapses
        # accept on chain steps 2+ (measured 16k K4 4.63 -> 1.97; DP4 canary K4
        # recovers 1.97 -> 4.48 with the fws graph off) -- the sampled-token
        # output of one fws replay does not correctly drive the next step's
        # input under the scratchpad. The plain-forward FULL graph replays
        # correctly (each step reads the in-place-refreshed window buffers) and
        # is faster (no per-step fws capture; +93% tok/s at the DP4 canary).
        if self._draft_fullcg:
            self._disable_sample_in_graph = True
        # Last dispatched FULL batch descriptor (set by
        # _determine_batch_execution_and_padding) so callers can put it in the
        # forward context for the FULL CUDAGraphWrapper.
        self._last_batch_desc: BatchDescriptor | None = None
        # OV0b (phase 49): shadow-chain overlap probe. When >0, the runner calls
        # shadow_replay_chain() right after the verify forward is enqueued; it
        # replays this many draft decode-step forwards on a side stream using
        # the dispatch state cached at the end of the previous propose. Timing
        # only: the slot buffer is filled with PADDING_SLOT_ID (KV writes
        # discarded) and outputs are never read. propose() waits on
        # _shadow_done_event before touching shared buffers/graphs.
        self._shadow_chain_steps = envs.VLLM_SELF_SPEC_SHADOW_CHAIN
        self._shadow_stream: torch.cuda.Stream | None = None
        self._shadow_ctx: dict[str, Any] | None = None
        self._shadow_done_event: torch.cuda.Event | None = None
        # OV1(a) (phase 49): free-running ahead-chain, VALIDATION mode. After
        # each propose the chain CONTINUES K+1 steps on the side stream during
        # the verify (input = its own last token; first ahead token = the bonus
        # guess), with real KV writes (covered by the extended scheduler
        # lookahead). Outputs are DISCARDED in this mode -- the normal propose
        # still runs, so the served output is byte-identical -- while the
        # machinery is exercised for real and the per-request hit rate
        # P(all K accepted AND bonus == guess) is measured (the free-running
        # design's speculation prior, expected ~beta^(K+1)).
        self._ahead_chain = (
            envs.VLLM_SELF_SPEC_AHEAD_CHAIN and self.method == "draft_model"
        )
        self._ahead_state: dict[str, Any] | None = None
        self._ahead_tokens: torch.Tensor | None = None
        self._ahead_batch_size = -1
        self._ahead_hits: torch.Tensor | None = None
        self._ahead_total = 0
        self._ahead_compared = 0
        # OV1(b) (phase 49): CONSUME mode -- the free-running loop. The ahead
        # chain re-anchors every cycle on the actual committed token and its
        # drafts REPLACE the lockstep propose (skipped in steady state).
        # Pending-token state machine (all per-row GPU tensors, no syncs):
        #   _ahead_out         [bs, K]  drafts to serve at the next consume
        #   _ahead_pending_tok [bs]     trailing guess token (KV pending; the
        #                               next run's continuation anchor)
        #   _ahead_pending_pos [bs]     its position
        #   _ahead_anchor_used [bs]     the anchor the in-flight run consumed
        #                               (the served drafts' bonus assumption;
        #                               tested against the actual bonus)
        #   _reanchor          stash from consume: fresh block-table snapshot
        #                               (b1: all-hit gated, so continuation
        #                               needs no per-row rewind state)
        #   _ahead_dispatch / _ahead_cad  persistent dispatch keys + private
        #                               metadata object (bootstrapped by propose)
        self._consume_mode = (
            self._ahead_chain and envs.VLLM_SELF_SPEC_CONSUME_AHEAD
        )
        self._ahead_out: torch.Tensor | None = None
        self._ahead_outs_full: torch.Tensor | None = None
        self._ahead_anchor_committed: torch.Tensor | None = None
        self._ahead_pending_tok: torch.Tensor | None = None
        self._ahead_pending_pos: torch.Tensor | None = None
        self._ahead_valid: torch.Tensor | None = None
        self._expect_tok: torch.Tensor | None = None
        self._expect_rej: torch.Tensor | None = None
        self._reanchor: dict[str, Any] | None = None
        self._ra_keepalive: dict[str, Any] | None = None
        self._ahead_dispatch: dict[str, Any] | None = None
        self._ahead_cad: Any = None
        # Private caching-allocator pool for the shadow's TRANSIENT allocations
        # (eager attention outputs, forward-context temporaries). vLLM's
        # piecewise replay relies on allocator ADDRESS DETERMINISM for the
        # inter-piece eager outputs (captured graphs read them at baked
        # addresses; the DEBUG-mode address check exists for exactly this).
        # Shadow allocations from the shared allocator perturb the free lists,
        # move the REAL chain's inter-piece tensors, and silently corrupt the
        # draft tokens (accept 2.9 -> ~1.0, no fault). Keeping every shadow
        # alloc in a private pool leaves the main pattern untouched.
        # A/B off-knob: W7_SHADOW_NO_MEM_POOL=1.
        self._shadow_mem_pool: Any = None
        # Ordering for the shadow: the side stream must not start replaying the
        # chain graphs while the PREVIOUS propose's chain kernels are still
        # running on the main stream (same graphs -> same workspaces -> race,
        # observed as a sticky CUDA illegal-instruction). Recorded on the main
        # stream at the end of propose(); the shadow waits on it -- this orders
        # shadow-after-previous-chain on the DEVICE while still overlapping the
        # verify (enqueued after propose on the same main stream).
        self._propose_done_event: torch.cuda.Event | None = (
            torch.cuda.Event() if self._shadow_chain_steps > 0 else None
        )

        # Cudagraph dispatcher for the drafter. PIECEWISE by default; FULL for
        # the decode-step forwards when use_full_cudagraphs is set.
        # Keys are initialized later via initialize_cudagraph_keys() called from
        # gpu_model_runner._check_and_update_cudagraph_mode after
        # adjust_cudagraph_sizes_for_spec_decode is called.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        # Use draft model's M-RoPE setting, not target model's
        # Draft models may be text-only even if target is multimodal
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.uses_xdrope_dim = self.vllm_config.model_config.uses_xdrope_dim
        self.draft_uses_xdrope_dim = self.draft_model_config.uses_xdrope_dim
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros(
                (3, self.max_positions + 1), dtype=torch.int64, device=device
            )
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_positions + 1),
                dtype=torch.int64,
                device=device,
            )
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(
                self.max_positions,
                dtype=torch.int64,
                device=device,
            )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # Will be set when we initialize the attention backend
        self.block_size: int = -1

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_num_slots_for_arange = max(self.max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )
        # W7 phase-45 CPU-orch: persistent CPU arange for the chain's
        # query_start_loc_cpu, so the per-cycle chain_setup does not allocate a
        # fresh tensor via torch.from_numpy(...).clone() each decode step. The
        # values are the constant [0, 1, ..., batch_size]; a view is stored so
        # it must NOT be mutated by callers (the drafting chain never does).
        self._cpu_orch = bool(envs.VLLM_SELF_SPEC_CPU_ORCH)
        self._chain_qsl_cpu = torch.arange(
            max_num_slots_for_arange, dtype=torch.int32
        )

        if self.needs_extra_input_slots:
            self._raise_if_padded_drafter_batch_disabled()
            self._warn_if_multimodal()
            self._raise_if_mrope()

        self.is_rejected_token_mask: torch.Tensor | None = None
        self.is_masked_token_mask: torch.Tensor | None = None
        if self.needs_extra_input_slots:
            # For draft models and parallel drafting, we need to keep track of
            # which tokens are rejected to update the slot mapping with padding slots.
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )
            # For parallel drafting, we also need to keep track of which tokens
            # are parallel-padding tokens used to sample at later positions.
            # We populate this tensor even when using draft models for simplicity.
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
            device=device,
            with_numpy=True,
        )
        self._enable_probabilistic_draft_probs = (
            self.speculative_config.rejection_sample_method == "standard"
            and self.speculative_config.draft_sample_method == "probabilistic"
        )
        self._last_draft_probs: torch.Tensor | None = None

        self._slot_mapping_buffer = torch.zeros(
            self.max_positions, dtype=torch.int64, device=device
        )

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            from vllm.models.deepseek_v4.amd.rocm import (
                DeepseekV4ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterSparseSWAMetadata,
            )

            # MiniMax-M3 sparse (lightning-indexer) attention. The multi-step
            # drafting machinery is shared code at num_speculative_tokens>1.
            # this just opts the metadata into the ROCm allowlist.
            from vllm.models.minimax_m3.common.sparse_attention import (
                MiniMaxM3SparseMetadata,
            )
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerMetadata,
            )
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
                ROCMAiterMLASparseMetadata,
            )
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterSparseSWAMetadata,
                DeepseekV32IndexerMetadata,
                MiniMaxM3SparseMetadata,
            ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention backend support
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)

    def _raise_if_padded_drafter_batch_disabled(self):
        if self.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting only "
                "supports padded drafter batch. Please unset "
                "disable_padded_drafter_batch in the speculative_config."
            )

    def _warn_if_multimodal(self):
        if self.supports_mm_inputs:
            logger.warning(
                "Speculative Decoding with draft models or parallel drafting "
                "does not fully support multimodal models yet. "
                "Proceeding with text-only speculative decoding."
            )

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support M-RoPE yet"
            )

    def _init_parallel_drafting_params(self):
        # For parallel drafting, we need the token ID to use for masked slots
        # And for EAGLE + parallel drafting, we need the hidden state tensor to use
        # for those masked slots.

        model_hf_config = self.draft_model_config.hf_config
        # DFlash stores mask_token_id in dflash_config
        dflash_config = getattr(model_hf_config, "dflash_config", None)
        if dflash_config and "mask_token_id" in dflash_config:
            self.parallel_drafting_token_id = dflash_config["mask_token_id"]
        elif hasattr(model_hf_config, "pard_token"):
            self.parallel_drafting_token_id = model_hf_config.pard_token
        elif hasattr(model_hf_config, "ptd_token_id"):
            self.parallel_drafting_token_id = model_hf_config.ptd_token_id
        else:
            raise ValueError(
                "For parallel drafting, the draft model config must have "
                "`pard_token`, `ptd_token_id`, or "
                "`dflash_config.mask_token_id` specified in its config.json."
            )

        if self.pass_hidden_states_to_model:
            self.parallel_drafting_hidden_state_tensor = torch.empty(
                self.hidden_size, dtype=self.dtype, device=self.device
            )

    def _get_positions(self, num_tokens: int):
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            return self.xdrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[:, :num_tokens] = positions
        else:
            # Convert M-RoPE positions if target model uses M-RoPE
            # but draft doesn't, For text inputs, all M-RoPE
            # dimensions are identical
            if self.vllm_config.model_config.uses_mrope:
                positions = positions[0]
            self.positions[:num_tokens] = positions

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for EAGLE layers.

        If slot_mapping is provided, copies it into the buffer first.
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        # Phase 65 (CHAIN_LIGHT_MD): the dict is {layer: buffer[:num_tokens]}
        # -- deterministic per num_tokens over the persistent buffer, so cache
        # it instead of rebuilding ~num_layers entries per chain step.
        if self._chain_light_md:
            cached = self._slot_mapping_dict_cache.get(num_tokens)
            if cached is None:
                view = self._slot_mapping_buffer[:num_tokens]
                cached = {name: view for name in self._draft_attn_layer_names}
                self._slot_mapping_dict_cache[num_tokens] = cached
            return cached

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self._draft_attn_layer_names}

    @property
    def _step0_full_cg(self) -> bool:
        """Phase 55: step-0 FULL cudagraph eligibility (K=1 only)."""
        return (
            envs.VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG
            and self.use_full_cudagraphs
            and self.speculative_config.num_speculative_tokens == 1
        )

    @property
    def _step0_q(self) -> int:
        """Phase 55: the step-0 batch's per-request query len. The padded
        drafter batch keeps all K+1 verify tokens per request and
        set_inputs_first_pass appends net_num_new_slots_per_request more
        (draft_model: +1 -> q=3 at K=1), so steady-state step-0 batches are
        already per-request uniform at this q (rejected slots are PAD-masked).
        """
        return (
            self.speculative_config.num_speculative_tokens
            + 1
            + self.net_num_new_slots_per_request
        )

    def _tag_a2a(self, src: str) -> None:
        """Phase 55 debug (W7_A2A_DEBUG): tag the source of the draft's
        Python-side collectives so [a2a-dbg] lines diff as sequences."""
        if self._a2a_dbg:
            from vllm.distributed.device_communicators.all2all import (
                w7_set_a2a_src,
            )

            w7_set_a2a_src(src)

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for the drafter.

        By default only PIECEWISE cudagraphs are supported (via mixed_mode),
        which leaves the per-layer inter-piece glue (attention launches /
        boundaries) eager. When VLLM_SELF_SPEC_DRAFT_FULL_CG is set and the
        engine's decode cudagraph mode is FULL, the drafter instead captures a
        FULL cudagraph for its decode-step forwards (each step is 1 token/seq,
        a uniform decode shape FULL supports), mirroring the verify model.
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        """
        if self.speculative_config.enforce_eager:
            eagle_cudagraph_mode = CUDAGraphMode.NONE
        elif (
            self.use_full_cudagraphs
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        ):
            # Draft decode steps are uniform 1-token-per-seq batches, so a
            # decode-only FULL routine is what we want. PIECEWISE keys are
            # still added for the step-0 (variable-shape) draft forward.
            eagle_cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        elif cudagraph_mode.mixed_mode() in [
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL,
        ]:
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        # The draft dispatcher's uniform_decode_query_len defaults to
        # 1 + num_speculative_tokens (the *verify* query len); override to 1
        # because each draft decode-step forward is a single token per seq.
        # Phase 55: at K=1 there are no q=1 chain forwards -- the only draft
        # forward is step-0, per-request uniform at q=_step0_q (=K+2 for
        # draft_model: K+1 verify tokens + the appended sampled-token slot).
        # With STEP0_FULL_CG, key the draft's uniform-decode FULL graphs at
        # that q so steady-state step-0 batches replay FULL.
        _qlen = self._step0_q if self._step0_full_cg else 1
        self.cudagraph_dispatcher.uniform_decode_query_len = _qlen
        # W7[70] fullcg coverage: for the q=1 chain-decode FULL graphs the
        # drafter keys by num_reqs (== num_tokens, since each step is one token
        # per seq). But adjust_cudagraph_sizes_for_spec_decode rounded the shared
        # cudagraph_capture_sizes UP to multiples of the verify query len
        # (1 + num_speculative_tokens), so the SMALLEST draft FULL graph is
        # num_reqs = (1+K) (e.g. 5). A per-rank draft batch of 1 then pads UP to
        # a (1+K)-sequence forward -- (1+K)x the window-scratchpad gather (cap544
        # at 16k) and MoE-expert work EVERY chain step, so the FULL graph
        # replays but on 4 wasted padding rows (draft step 23 -> 52 ms at 16k).
        # Augment the draft dispatcher's capture sizes with the small raw
        # num_reqs values (< 1+K) the runner already drives during capture, so
        # b1/b2/b4 chains replay a true bN graph. q=1 chain-decode FULL only;
        # step-0 FULL-CG (q=_step0_q) keeps the shared sizes.
        # Scoped to the window-scratchpad FULL-CG chain (_draft_fullcg): only
        # there does the draft chain run FULL cudagraphs AND the warmup capture
        # actually record the small num_reqs graphs. For the plain DRAFT_FULL_CG
        # path (FA3 chain force-eager / PIECEWISE) the draft never FULL-captures,
        # so adding uncaptured small FULL keys would fault a later FULL dispatch
        # in the wrapper -- keep that path byte-identical.
        if (
            self._draft_fullcg
            and eagle_cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
            and _qlen == 1
        ):
            self._augment_draft_capture_sizes_for_num_reqs(
                eagle_cudagraph_mode, _qlen
            )
        else:
            self.cudagraph_dispatcher.initialize_cudagraph_keys(
                eagle_cudagraph_mode, uniform_decode_query_len=_qlen
            )

    def _augment_draft_capture_sizes_for_num_reqs(
        self, eagle_cudagraph_mode: CUDAGraphMode, qlen: int
    ) -> None:
        """Initialize the draft dispatcher's keys with small num_reqs capture
        sizes added (W7[70]).

        The shared ``cudagraph_capture_sizes`` were rounded to multiples of the
        verify query len ``vq = 1 + num_speculative_tokens``. Each draft
        decode-step forward is one token per seq, so it keys by num_reqs; the
        runner drives capture at num_reqs = size // vq for every shared size,
        but the sub-``vq`` values (1..K) currently pad up to ``vq``. Add those
        raw num_reqs values as draft capture sizes so a per-rank b1 chain
        replays a b1 graph. Restores the shared config afterward (the runner's
        dispatcher was already initialized).
        """
        cc = self.vllm_config.compilation_config
        sizes = cc.cudagraph_capture_sizes
        vq = 1 + self.num_speculative_tokens
        extra = (
            sorted({s // vq for s in sizes if 1 <= s // vq < vq})
            if sizes and vq > 1
            else []
        )
        if not extra:
            self.cudagraph_dispatcher.initialize_cudagraph_keys(
                eagle_cudagraph_mode, uniform_decode_query_len=qlen
            )
            return
        augmented = sorted(set(sizes) | set(extra))
        saved = cc.cudagraph_capture_sizes
        cc.cudagraph_capture_sizes = augmented
        try:
            self.cudagraph_dispatcher.initialize_cudagraph_keys(
                eagle_cudagraph_mode, uniform_decode_query_len=qlen
            )
        finally:
            cc.cudagraph_capture_sizes = saved
        logger.info(
            "Draft FULL-CG (q=1) capture sizes augmented with small num_reqs "
            "%s (verify query len=%d): per-rank b<%d chains now replay an "
            "exact bN graph instead of padding up to b%d.",
            extra,
            vq,
            vq,
            vq,
        )

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        return self._vres_remap(
            self.model.compute_logits(hidden_states).argmax(dim=-1))

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs:
            return self._vres_remap(logits.argmax(dim=-1)), None
        if sampling_metadata.all_greedy:
            return self._vres_remap(logits.argmax(dim=-1)), None
        return compute_probs_and_sample_next_token(
            logits, sampling_metadata, self.use_fp64_gumbel
        )

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._greedy_sample(hidden_states), None
        logits = self.model.compute_logits(hidden_states)
        return self._sample_from_logits(logits, sampling_metadata)

    def take_last_draft_probs(self) -> torch.Tensor | None:
        return self._last_draft_probs

    def propose(
        self,
        num_speculative_tokens,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        # OV0b: the shadow chain (side stream) reads the persistent buffers and
        # replays the same graphs this propose is about to use -- order the real
        # propose after it (event wait, not a sync; measures max() semantics).
        if self._shadow_done_event is not None:
            torch.cuda.current_stream().wait_event(self._shadow_done_event)
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        self._dp_coord_memo.clear()
        self._in_propose = True
        batch_size = common_attn_metadata.batch_size()

        # OV1(a) hit-rate check: the previous cycle's ahead chain speculated
        # "all K accepted, bonus == its guess". next_token_ids is the token the
        # committed sequence actually continues from; a hit means the ahead
        # chain's remaining K tokens ARE the next drafts (in full OV1 this
        # propose would be skipped for hit requests). Validation-mode: count
        # and continue with the normal propose.
        if (
            self._ahead_tokens is not None
            and self._ahead_batch_size == batch_size
            and num_rejected_tokens_gpu is not None
        ):
            hit = (num_rejected_tokens_gpu[:batch_size] == 0) & (
                next_token_ids[:batch_size] == self._ahead_tokens[:batch_size, 0]
            )
            hits = hit.sum()
            self._ahead_hits = (
                hits if self._ahead_hits is None else self._ahead_hits + hits
            )
            self._ahead_total += batch_size
            self._ahead_compared += 1
            if self._ahead_compared % 50 == 0:
                logger.info(
                    "Self-spec OV1(a) ahead-chain hit rate: %.3f "
                    "(%d requests over %d cycles)",
                    self._ahead_hits.item() / max(self._ahead_total, 1),
                    self._ahead_total,
                    self._ahead_compared,
                )
        self._ahead_tokens = None

        if self.method in ("eagle3", "dflash"):
            model = self.model
            if isinstance(model, BreakableCUDAGraphWrapper):
                model = model.unwrap()
            assert isinstance(
                model,
                (
                    Eagle3LlamaForCausalLM,
                    Eagle3DeepseekV2ForCausalLM,
                    DFlashQwen3ForCausalLM,
                    Eagle3Qwen3ForCausalLM,
                ),
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        with _self_spec_profiler().region("step0_set_inputs"):
            num_tokens, token_indices_to_sample, common_attn_metadata = (
                self.set_inputs_first_pass(
                    target_token_ids=target_token_ids,
                    next_token_ids=next_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    token_indices_to_sample=token_indices_to_sample,
                    cad=common_attn_metadata,
                    num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                )
            )

        # Phase 67: shared-KV step-0 decode compaction (q=1 over appended token).
        step0_decode = self._step0_decode_active(
            common_attn_metadata, num_rejected_tokens_gpu
        )
        step0_appended_slots: torch.Tensor | None = None
        with _self_spec_profiler().region("step0_build_attn_md"):
            # Phase 62: window the step-0 draft attention (decode-shaped
            # proposes only). The windowed view is a shallow copy over
            # proposer-owned buffers; common_attn_metadata (used for slot
            # mappings / per-step updates) keeps the TRUE block table.
            md_cad = common_attn_metadata
            if self._kv_window_step0_active(
                common_attn_metadata, num_rejected_tokens_gpu
            ):
                md_cad = self._apply_draft_kv_window(common_attn_metadata)
            if step0_decode:
                _orig_ntok = common_attn_metadata.num_actual_tokens
                (
                    md_cad,
                    num_tokens,
                    token_indices_to_sample,
                    step0_appended_slots,
                ) = self._compact_step0_decode(
                    common_attn_metadata,
                    md_cad,
                    token_indices_to_sample,
                    num_rejected_tokens_gpu,
                )
                if not self._step0_decode_logged:
                    self._step0_decode_logged = True
                    logger.info(
                        "[step0-decode] shared-KV step-0 compacted to q=1 "
                        "decode: %d appended tokens (was %d)",
                        num_tokens,
                        _orig_ntok,
                    )
            per_group_attn_metadata, per_layer_attn_metadata = (
                self.build_per_group_and_layer_attn_metadata(md_cad)
            )

        with _self_spec_profiler().region("step0_determine_batch"):
            if step0_decode:
                # Phase 67: the compacted step-0 is a uniform q=1 decode --
                # dispatch it exactly like a chain step (PIECEWISE body + eager
                # windowed attention under the window; FULL only when captured).
                chain_piecewise = (
                    self._draft_chain_force_eager_attn
                    and self._draft_chain_piecewise
                )
                chain_use_cudagraphs = (
                    not self._draft_chain_force_eager_attn or chain_piecewise
                )
                chain_uniform = True
                if (
                    self.use_full_cudagraphs
                    and chain_use_cudagraphs
                    and not chain_piecewise
                ):
                    _mode, _desc = self.cudagraph_dispatcher.dispatch(
                        batch_size, uniform_decode=True
                    )
                    if (
                        _mode == CUDAGraphMode.FULL
                        and _desc.num_tokens not in self._full_captured
                    ):
                        chain_uniform = False
                (
                    cudagraph_runtime_mode,
                    num_input_tokens,
                    num_tokens_across_dp,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens,
                    uniform_decode=chain_uniform,
                    use_cudagraphs=chain_use_cudagraphs,
                    piecewise_only=chain_piecewise,
                )
            else:
                # Phase 55: with STEP0_FULL_CG (K=1), the steady-state step-0
                # batch is per-request uniform at q=_step0_q (=3: 2 verify tokens
                # + the appended sampled-token slot; rejected slots PAD-masked by
                # set_inputs_first_pass) and dispatches to the FULL graphs
                # captured at that shape; mixed prefill/decode steps fall back
                # naturally. The max_query_len check excludes mixed batches that
                # only sum to batch*q.
                _q = (
                    self.num_speculative_tokens
                    + 1
                    + self.net_num_new_slots_per_request
                )
                _step0_uniform = (
                    self._step0_full_cg
                    and num_tokens == batch_size * _q
                    and common_attn_metadata.max_query_len == _q
                )
                if _step0_uniform and self._step0_force_pw:
                    # Debug bisect knob (W7_STEP0_FORCE_PW=1): keep all capture
                    # machinery but never dispatch step-0 FULL at runtime.
                    _step0_uniform = False
                if _step0_uniform and self._kv_window > 0:
                    # Phase 62: the captured step-0 FULL graph reads the full
                    # block table through capture-time pointers; window-KV
                    # drafting needs the live (windowed) metadata, so fall back
                    # to the relaxed (PIECEWISE) dispatch.
                    _step0_uniform = False
                if _step0_uniform:
                    # Pre-flight (pure, no DP collective): only claim uniform if
                    # the dispatch target is a FULL shape we actually captured;
                    # a keyed-but-uncaptured shape would fault in the wrapper.
                    _mode, _desc = self.cudagraph_dispatcher.dispatch(
                        num_tokens, uniform_decode=True
                    )
                    if (
                        _mode != CUDAGraphMode.FULL
                        or _desc.num_tokens not in self._step0_captured
                    ):
                        _step0_uniform = False
                # Phase 82 (SKIP_PREFILL_DRAFT): a BOOTSTRAP step-0 (no drafts
                # verified this step, e.g. right after a skipped draft prefill)
                # arrives q=1-shaped at arbitrary batch -- geometries the
                # capture pass never drives (it only sees verify-width
                # multiples). A keyed-but-uncaptured PIECEWISE shape would
                # fault in the wrapper; run these rare boundary steps eager.
                _step0_bootstrap = (
                    envs.VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT
                    and num_rejected_tokens_gpu is None
                    and self.method == "draft_model"
                )
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens,
                        uniform_decode=_step0_uniform,
                        use_cudagraphs=not _step0_bootstrap,
                    )
                )
                if (
                    _step0_uniform
                    and cudagraph_runtime_mode == CUDAGraphMode.FULL
                ):
                    # The captured step-0 graph reads seq_lens through the
                    # proposer-owned buffer (the live extended seq_lens is a
                    # fresh tensor each cycle); refresh it. Padded-request rows
                    # are zeroed so they attend over nothing.
                    assert self._step0_seq_lens is not None
                    self._step0_seq_lens[:batch_size].copy_(
                        common_attn_metadata.seq_lens[:batch_size]
                    )
                    self._step0_seq_lens[batch_size:].zero_()
                _k = (num_tokens, batch_size, _step0_uniform)
                if envs.VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG and _k not in getattr(
                    self, "_step0_cg_logged", set()
                ):
                    self._step0_cg_logged = getattr(
                        self, "_step0_cg_logged", set()
                    ) | {_k}
                    logger.info(
                        "[step0-cg] num_tokens=%d batch=%d K=%d uniform=%s -> "
                        "mode=%s padded=%d",
                        num_tokens, batch_size, self.num_speculative_tokens,
                        _step0_uniform, cudagraph_runtime_mode, num_input_tokens,
                    )

            model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
                num_tokens, num_input_tokens, mm_embed_inputs
            )
        # Step 0 of index_share_for_mtp_iteration: let the MTP layer
        # compute its own indices (skip_topk=False) so subsequent steps
        # can reuse them.
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(False)

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=self._last_batch_desc,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size,
                step0_appended_slots
                if step0_decode
                else common_attn_metadata.slot_mapping,
            ),
            additional_kwargs=self._draft_forward_additional_kwargs,
        ):
            # Self-spec W7 micro-bench: the first (step-0) draft forward. Timed
            # under a distinct label since its input shape can differ from the
            # subsequent decode-step loop forwards.
            self._tag_a2a("propose-step0")
            with _self_spec_profiler().region("draft_forward_first"):
                ret_hidden_states = self.model(**model_kwargs)
            self._tag_a2a("")
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        # After step 0: switch to reuse mode so steps 1+ skip the indexer
        # and read the indices that step 0 just wrote into the shared buffer.
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(True)

        if self._step0_debug:
            logger.info(
                "[draft-replay] rank=%d src=step0 mode=%s ntok=%d",
                self.dp_rank,
                cudagraph_runtime_mode,
                num_input_tokens,
            )

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # No draft tokens requested (e.g. Dynamic SD decided K=0).
        # The prefill forward pass above already ran to keep the drafter
        # KV cache in sync, so just return an empty tensor.
        if self.num_speculative_tokens == 0:
            return torch.empty(
                batch_size,
                0,
                device=sample_hidden_states.device,
                dtype=torch.int64,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if self.constant_draft_positions:
            # Write the sampling positions into the front of the
            # positions buffer so that subsequent loop iterations
            # (which read via _get_positions) use the correct values.
            self.positions[:batch_size] = positions

        with _self_spec_profiler().region("step0_sample"):
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        with _self_spec_profiler().region("chain_setup"):
            # The K-chain decode-step forwards are uniform 1-token-per-seq
            # batches, eligible for the draft's FULL cudagraph (when enabled).
            #
            # W7-gqa: the FULL-CG draft chain replays ONE captured decode graph
            # K-1 times with the sequence GROWING by 1 each replay. The FA3
            # (GQA) decode kernel freezes its host-side work distribution at
            # capture-time seq_len=1 and does NOT re-derive it from the live,
            # growing ``seqused_k`` at replay -> the 2nd+ draft token attends
            # over a truncated context -> accept_len collapses (~4.9 -> ~1.9 on
            # Qwen3-8B). This is INSENSITIVE to scheduler_metadata / num_splits /
            # capture seq_len (all measured identical); running the SAME chain
            # eagerly restores accept_len fully. (The MLA decode kernel does not
            # have this dependency, so MLA keeps the FULL graph -- see fg2.)
            # Fix: for non-MLA (FA3) draft backends, take the eager (NONE) path
            # for the chain so each step launches a fresh attention kernel
            # against live seq_lens. We pass use_cudagraphs=False so the DP batch
            # coordination matches the eager path -- overriding the mode AFTER
            # coordination would desync num_tokens_across_dp and trip the DP
            # assert in set_forward_context at DP>1.
            # W7-piecewise: when the chain forces eager attention (FA3/GQA under
            # DRAFT_FULL_CG), the default takes the fully-eager NONE path
            # (use_cudagraphs=False). With DRAFT_CHAIN_PIECEWISE, instead run the
            # body on captured PIECEWISE pieces (piecewise_only) while attention
            # stays eager -- same numerics, faster body. DP stays consistent: all
            # ranks dispatch PIECEWISE so coordinate_batch_across_dp syncs to
            # PIECEWISE and pads uniformly (mirrors normal decode).
            chain_piecewise = (
                self._draft_chain_force_eager_attn and self._draft_chain_piecewise
            )
            chain_use_cudagraphs = (
                not self._draft_chain_force_eager_attn or chain_piecewise
            )
            # Phase 55 (K>=2): pre-flight (pure, no DP collective) -- only
            # claim uniform-decode FULL for the chain when the dispatch
            # target is a FULL shape the capture pass actually drove (see
            # _full_captured). The largest FULL keys (> max_runner_size/(K+1))
            # are keyed but never captured; dispatching them would attempt a
            # forbidden runtime capture in the wrapper. Fallback is the
            # relaxed (PIECEWISE) claim; the DP sync (mode-min) then keeps
            # busy/idle collective sequences aligned as usual.
            chain_uniform = True
            if (
                self.use_full_cudagraphs
                and chain_use_cudagraphs
                and not chain_piecewise
            ):
                _mode, _desc = self.cudagraph_dispatcher.dispatch(
                    batch_size, uniform_decode=True
                )
                if (
                    _mode == CUDAGraphMode.FULL
                    and _desc.num_tokens not in self._full_captured
                ):
                    chain_uniform = False
            cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
                self._determine_batch_execution_and_padding(
                    batch_size,
                    uniform_decode=chain_uniform,
                    use_cudagraphs=chain_use_cudagraphs,
                    piecewise_only=chain_piecewise,
                )
            )
            if chain_piecewise and not self._logged_chain_pw:
                self._logged_chain_pw = True
                logger.info(
                    "Draft chain PIECEWISE: runtime_mode=%s batch_size=%d "
                    "input_batch_size=%d (attention eager, body cudagraph).",
                    cudagraph_runtime_mode,
                    batch_size,
                    input_batch_size,
                )
            # W7[70] fullcg-coverage debug: one-time dump of the chain dispatch
            # descriptor vs the drafter.model FULL wrapper's captured keys and
            # the _full_captured num_tokens set. Gated by W7_FULLCG_DBG.
            if os.environ.get("W7_FULLCG_DBG") and not getattr(
                self, "_logged_fullcg_dbg", False
            ):
                self._logged_fullcg_dbg = True
                _wrapper = getattr(self.model, "cudagraph_wrapper", None)
                _caps = (
                    sorted(
                        (d.num_tokens, d.num_reqs, d.uniform)
                        for d in _wrapper.concrete_cudagraph_entries
                    )
                    if _wrapper is not None
                    else None
                )
                logger.info(
                    "[fullcg-dbg] rank=%d chain dispatch: batch_size=%d mode=%s "
                    "last_desc=%s full_captured=%s model_wrapper_captured=%s",
                    self.dp_rank,
                    batch_size,
                    cudagraph_runtime_mode,
                    self._last_batch_desc,
                    sorted(self._full_captured),
                    _caps,
                )

            common_attn_metadata.num_actual_tokens = batch_size
            common_attn_metadata.max_query_len = 1
            common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
            if self._cpu_orch:
                # Reuse the persistent CPU arange view (constant contents); the
                # chain never mutates query_start_loc_cpu in place, so no clone.
                common_attn_metadata.query_start_loc_cpu = self._chain_qsl_cpu[
                    : batch_size + 1
                ]
            else:
                common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                    self.token_arange_np[: batch_size + 1]
                ).clone()

            # In padded drafter batch, we need to adjust the sequence lengths
            # to remove the "padding" (i.e. rejected tokens).
            # Only apply this adjustment when we have rejected tokens
            # (i.e., not the first proposal).
            if (
                self.num_speculative_tokens > 1
                and num_rejected_tokens_gpu is not None
            ):
                common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
                # Invalidate the CPU-side shadows to avoid H<>D sync.
                common_attn_metadata._seq_lens_cpu = None
                common_attn_metadata._num_computed_tokens_cpu = None

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."
        # Phase 65 (CHAIN_LIGHT_MD): per-group metadata built on the first
        # chain step, reused (data updated in place) by the later steps.
        chain_md_groups: list | None = None
        # Phase 82 whole-chain capture: arm when eligible. The seed buffer
        # replaces the step-0 output in the list so the captured first
        # step reads a persistent address; per-propose state (positions,
        # seq_lens, window buffers) already lives in persistent buffers.
        _wc_armed = (
            self._wholechain
            and self.num_speculative_tokens > 1
            and sampling_metadata.all_greedy
            and batch_size not in self._wc_failed
            and batch_size_across_dp is None
            and not self._ahead_chain
            and self._shadow_chain_steps == 0
        )
        self._wc_dbg_n = getattr(self, "_wc_dbg_n", 0) + 1
        if False:
            logger.info(
                "[wc-dbg] armed=%s K=%d greedy=%s failed=%s dp=%s ahead=%s "
                "shadow=%d rej=%s",
                _wc_armed, self.num_speculative_tokens,
                sampling_metadata.all_greedy, batch_size in self._wc_failed,
                batch_size_across_dp is None, self._ahead_chain,
                self._shadow_chain_steps, num_rejected_tokens_gpu is not None,
            )
        _wc_ncols = common_attn_metadata.block_table_tensor.shape[1]
        # K is baked into the captured loop (dynamic-SD switches it per
        # propose) -- key graphs by (batch, table width, K).
        _wc_key = (batch_size, _wc_ncols, self.num_speculative_tokens)
        _wc = self._wc_graphs.get(_wc_key) if _wc_armed else None
        _wc_capture = _wc_armed and _wc is None
        _wc_stack = _contextlib.ExitStack()
        if _wc_armed:
            if self._wc_seed is None:
                self._wc_seed = torch.zeros(
                    self.max_batch_size,
                    dtype=draft_token_ids_list[-1].dtype,
                    device=self.device,
                )
            self._wc_seed[:batch_size].copy_(
                draft_token_ids_list[-1][:batch_size]
            )
            draft_token_ids_list[-1] = self._wc_seed[:batch_size]
            # Persistent MIRRORS for per-propose tensors the captured
            # kernels read (the padded prepare path allocates seq_lens /
            # block_table fresh each propose; a captured graph would read
            # the capture-time tensor's dead storage). The chain's
            # in-place updates (seq_lens growth, slot advance) land on
            # the mirrors and are re-seeded here every propose.
            _wc_mir = getattr(self, "_wc_mirrors", None)
            if _wc_mir is None or _wc_mir["bt"].shape[1] != _wc_ncols:
                self._wc_mirrors = _wc_mir = dict(
                    bt=torch.zeros(
                        (self.max_batch_size, _wc_ncols),
                        dtype=common_attn_metadata.block_table_tensor.dtype,
                        device=self.device,
                    ),
                    sl=torch.zeros(
                        self.max_batch_size,
                        dtype=common_attn_metadata.seq_lens.dtype,
                        device=self.device,
                    ),
                )
                self._wc_graphs.clear()
            _wc_mir["bt"][:batch_size].copy_(
                common_attn_metadata.block_table_tensor[:batch_size]
            )
            _wc_mir["sl"][:batch_size].copy_(
                common_attn_metadata.seq_lens[:batch_size]
            )
            common_attn_metadata.block_table_tensor = _wc_mir["bt"][:batch_size]
            common_attn_metadata.seq_lens = _wc_mir["sl"][:batch_size]
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None
            # positions too: some step-0 branches produce a per-propose
            # tensor (b32/short-ctx and large-mnb builds) -- the captured
            # first position-update reads its baked address (was the
            # residual accept collapse at those geometries).
            if "pos" not in _wc_mir:
                _wc_mir["pos"] = torch.zeros(
                    self.max_batch_size,
                    dtype=positions.dtype,
                    device=self.device,
                )
                self._wc_graphs.clear()
            _wc_mir["pos"][:batch_size].copy_(
                positions[:batch_size]
                if positions.dim() == 1
                else positions[0, :batch_size]
            )
            if positions.dim() == 1:
                positions = _wc_mir["pos"][:batch_size]
        if _wc_capture:
            torch.cuda.synchronize()
            _wc_graph_obj = torch.cuda.CUDAGraph()
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            # Profiler regions CUDA-sync -- illegal inside stream capture.
            _wc_prof = _self_spec_profiler()
            _wc_prof_saved = _wc_prof.enabled
            _wc_prof.enabled = False
            _wc_stack.enter_context(torch.cuda.graph(_wc_graph_obj))
        for token_index in (
            range(self.num_speculative_tokens - 1)
            if _wc is None
            else range(0)
        ):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()

            # W7 micro-bench region (b): position/slot-mapping/seq-len update.
            with _self_spec_profiler().region("step_pos_slot_update"):
                if not self.constant_draft_positions:
                    positions = self._update_positions_dependent_metadata(
                        positions,
                        common_attn_metadata,
                        batch_size,
                        input_batch_size,
                        block_size,
                    )

            # Rebuild attention metadata. When draft positions are constant
            # (e.g. Gemma4 MTP), common_attn_metadata is invariant across
            # loop iterations so we build once and reuse.
            # W7 micro-bench region (a): per-step attn-metadata (re)build.
            #
            # W7-perf: under the draft FULL cudagraph, the replayed graph reads
            # the live attention tensors (seq_lens / block_table / slot_mapping)
            # through the pointers captured at graph-capture time -- it never
            # re-reads the per-step attn_metadata object from the forward
            # context (CUDAGraphWrapper.__call__ just replays the captured
            # graph). _update_positions_dependent_metadata already advances
            # those tensors IN-PLACE (the seq_lens view aliases runner.seq_lens
            # which capture pointed at, and the slot mapping lands in the
            # persistent _slot_mapping_buffer). So the fresh build_for_drafting
            # each step is dead work for FULL-CG replay: skip it and keep the
            # step-0 metadata object only as a placeholder for set_forward_context
            # (its contents are ignored on replay). Guarded by the opt-in flag;
            # the eager / PIECEWISE path is unchanged.
            with _self_spec_profiler().region("step_build_attn_md"):
                skip_rebuild_full_cg = (
                    self.use_full_cudagraphs
                    and cudagraph_runtime_mode == CUDAGraphMode.FULL
                    and not self._disable_skip_rebuild
                )
                # Phase 69: under FULL-CG the metadata rebuild is skipped (the
                # replay reads capture-time pointers), but the window-scratchpad
                # attention reads _win_block_table / _win_seq_lens -- refresh
                # those buffers IN PLACE each step (data-only, graph-safe) so the
                # captured gather sees the newest sinks+window+drafted pages.
                if (
                    self._draft_fullcg
                    and self._kv_window > 0
                    and skip_rebuild_full_cg
                ):
                    self._apply_draft_kv_window(
                        common_attn_metadata, tokens_drafted=token_index + 1
                    )
                # When the FA (GQA) chain falls back to eager attention,
                # cudagraph_runtime_mode is NONE here, so skip_rebuild_full_cg is
                # False and the per-step build_for_drafting runs -- giving each
                # eager attention launch a fresh metadata object with live
                # seq_lens / slot mapping (the correct path, matching PIECEWISE).
                if not skip_rebuild_full_cg and (
                    not self.constant_draft_positions or token_index == 0
                ):
                    # Phase 62: window-KV drafting -- give the eager per-step
                    # attention a compacted (sinks + trailing window) view.
                    # Recomputed per step so the pages holding the chain's
                    # newly appended KV are always in the kept set.
                    if chain_md_groups is not None:
                        # Phase 65 (CHAIN_LIGHT_MD): every tensor field of the
                        # step-1 FlashAttentionMetadata lives in a persistent
                        # buffer the per-step updates rewrite IN PLACE
                        # (seq_lens / block_table via the window compaction or
                        # the fused eagle-step kernel; slot_mapping via
                        # _slot_mapping_buffer; query_start_loc is constant).
                        # Only the max_seq_len scalar changes value: run the
                        # data updates and sync it, reuse the objects.
                        if self._kv_window > 0:
                            _win_cad = self._apply_draft_kv_window(
                                common_attn_metadata,
                                tokens_drafted=token_index + 1,
                            )
                            _new_max = _win_cad.max_seq_len
                        else:
                            _new_max = common_attn_metadata.max_seq_len
                        for _md in chain_md_groups:
                            _md.max_seq_len = _new_max
                    else:
                        md_cad = common_attn_metadata
                        if self._kv_window > 0:
                            md_cad = self._apply_draft_kv_window(
                                common_attn_metadata,
                                tokens_drafted=token_index + 1,
                            )
                        per_group_md, per_layer_attn_metadata = (
                            self.build_per_group_and_layer_attn_metadata(
                                md_cad, draft_index=token_index + 1
                            )
                        )
                        # Phase 65: reuse is only valid for FA-style metadata
                        # (all fields either persistent-buffer views or the
                        # max_seq_len scalar synced above). Other backends
                        # (e.g. MLA's CPU-derived decode splits) rebuild.
                        if self._chain_light_md and all(
                            type(_m).__name__ == "FlashAttentionMetadata"
                            for _m in per_group_md
                        ):
                            chain_md_groups = per_group_md
                elif self._kv_window > 0 and not self._kv_window_warned:
                    self._kv_window_warned = True
                    logger.warning(
                        "VLLM_SELF_SPEC_DRAFT_KV_WINDOW=%d is set but the "
                        "draft chain skips the per-step metadata rebuild "
                        "(FULL-CG replay reads capture-time pointers); the "
                        "KV window is NOT applied to chain steps. Use the "
                        "eager or PIECEWISE chain.",
                        self._kv_window,
                    )

            # W7 micro-bench region (d): input buffering + forward-context setup
            # (the part of the step that is neither attn-md build, pos/slot
            # update, the forward itself, nor sampling).
            with _self_spec_profiler().region("step_input_buffering"):
                # copy inputs to buffer for cudagraph
                self.input_ids[:batch_size] = input_ids
                self.hidden_states[:batch_size] = hidden_states
                if self.supports_mm_inputs:
                    self.inputs_embeds[:batch_size] = self.model.embed_input_ids(
                        input_ids
                    )

                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:input_batch_size]
                else:
                    input_ids = self.input_ids[:input_batch_size]
                    inputs_embeds = None

                # Run the model.
                model_kwargs = {
                    "input_ids": input_ids,
                    "positions": self._get_positions(input_batch_size),
                    "inputs_embeds": inputs_embeds,
                }
                if self.pass_hidden_states_to_model:
                    model_kwargs["hidden_states"] = self.hidden_states[
                        :input_batch_size
                    ]

            # W7 sampling-in-graph: when the draft FULL graph is active and the
            # request is greedy, run forward + compute_logits + argmax as one
            # captured graph (mirrors V2 _generate_draft). Otherwise fall back
            # to the plain forward + eager sample.
            sample_in_graph = (
                self._decode_fwd_sample is not None
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not self._disable_sample_in_graph
                and (
                    not self._enable_probabilistic_draft_probs
                    or sampling_metadata.all_greedy
                )
            )

            # Phase 69: carry the window-scratchpad attention context on the
            # chain forward so unified_attention_with_output runs the fixed-shape
            # dense windowed attention (the KV window buffers were just refreshed
            # by _apply_draft_kv_window above). Chain-only: step-0 / verify never
            # see the key.
            chain_add_kwargs = self._draft_forward_additional_kwargs
            if self._draft_fullcg and self._kv_window > 0:
                _sp_ctx = self._ensure_scratchpad_ctx(batch_size)
                chain_add_kwargs = dict(self._draft_forward_additional_kwargs or {})
                chain_add_kwargs[SELF_SPEC_DRAFT_SCRATCHPAD_KEY] = _sp_ctx

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=self._last_batch_desc,
                slot_mapping=self._get_slot_mapping(input_batch_size),
                additional_kwargs=chain_add_kwargs,
            ):
                # Self-spec W7 micro-bench: time a SINGLE draft model forward
                # (one decode-step forward inside the K-step chain) so it can
                # be compared against K of them and the whole chain.
                self._tag_a2a("chain")
                with _self_spec_profiler().region("draft_forward"):
                    if sample_in_graph:
                        last_hidden_states, hidden_states, draft_token_ids = (
                            self._decode_fwd_sample(**model_kwargs)
                        )
                    else:
                        ret_hidden_states = self.model(**model_kwargs)
                        if not self.model_returns_tuple():
                            last_hidden_states = ret_hidden_states
                            hidden_states = ret_hidden_states
                        else:
                            last_hidden_states, hidden_states = ret_hidden_states

            self._tag_a2a("")
            if self._step0_debug:
                logger.info(
                    "[draft-replay] rank=%d src=chain step=%d mode=%s "
                    "graph=%s ntok=%d",
                    self.dp_rank,
                    token_index,
                    cudagraph_runtime_mode,
                    "fws" if sample_in_graph else "model",
                    input_batch_size,
                )

            hidden_states = hidden_states[:batch_size]
            # W7 micro-bench region (c): draft sampling (compute_logits + argmax).
            with _self_spec_profiler().region("step_sample"):
                if sample_in_graph:
                    # Token ids were already argmax'd inside the captured graph.
                    draft_token_ids = draft_token_ids[:batch_size]
                    draft_probs = None
                else:
                    draft_token_ids, draft_probs = self._sample_draft_tokens(
                        last_hidden_states[:batch_size], sampling_metadata
                    )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        # Phase 82 whole-chain: end capture (stash graph + output refs,
        # then replay once for real values -- the capture pass only
        # records), or replay an existing graph.
        if _wc_capture:
            _wc_stack.close()
            _wc_prof.enabled = _wc_prof_saved
            entry = dict(
                graph=_wc_graph_obj,
                outs=list(draft_token_ids_list[1:]),
                hidden=hidden_states,
                last_hidden=last_hidden_states,
            )
            self._wc_graphs[_wc_key] = entry
            _wc_graph_obj.replay()
            logger.info(
                "Whole-chain graph captured (bs=%d, %d steps).",
                batch_size,
                self.num_speculative_tokens - 1,
            )
        elif _wc is not None:
            with _self_spec_profiler().region("wc_replay"):
                _wc["graph"].replay()
            draft_token_ids_list.extend(_wc["outs"])
            hidden_states = _wc["hidden"]
            last_hidden_states = _wc["last_hidden"]
            draft_token_ids = _wc["outs"][-1]

        # OV0b: cache the chain's dispatch state so the next verify's shadow
        # replay can re-launch an identical decode-step forward on the side
        # stream. The attn-metadata objects and buffer slices stay alive via
        # this reference; their (stale) values are fine for a timing replay.
        if self._shadow_chain_steps > 0 and self.num_speculative_tokens > 1:
            self._shadow_ctx = {
                "per_layer_attn_metadata": per_layer_attn_metadata,
                "mode": cudagraph_runtime_mode,
                "input_batch_size": input_batch_size,
                "batch_size_across_dp": batch_size_across_dp,
                "batch_desc": self._last_batch_desc,
            }
            assert self._propose_done_event is not None
            self._propose_done_event.record()

        # OV1(a): cache the LIVE chain-end state so run_ahead_chain() can
        # CONTINUE this chain (not replay it) on the side stream during the
        # verify. Greedy only (the bonus guess is the argmax continuation).
        if (
            self._ahead_chain
            and self.num_speculative_tokens > 1
            and sampling_metadata.all_greedy
        ):
            # PRIVATIZE the mutable metadata for the ahead loop. seq_lens and
            # block_table_tensor are views of RUNNER-owned buffers; in lockstep
            # that is safe (serial), but the free-running ahead chain advances
            # seq_lens on the side stream WHILE the runner's next-step input
            # prep and the verify read/write the same buffers on the main
            # stream -> torn values -> FA3 indexes garbage pages (observed
            # illegal memory access). Clone them here (main stream, end of
            # propose -> a consistent snapshot); the shallow copy keeps the
            # scalar mutations (max_seq_len etc.) private too. The proposer's
            # own buffers (positions/input_ids/hidden/slot) stay shared -- the
            # next propose waits on the ahead-done event before touching them.
            cad_priv = copy.copy(common_attn_metadata)
            cad_priv.seq_lens = common_attn_metadata.seq_lens.clone()
            cad_priv.block_table_tensor = (
                common_attn_metadata.block_table_tensor.clone()
            )
            cad_priv._seq_lens_cpu = None
            cad_priv._num_computed_tokens_cpu = None
            self._ahead_state = {
                "positions": positions,
                "cad": cad_priv,
                "batch_size": batch_size,
                "input_batch_size": input_batch_size,
                "block_size": block_size,
                "mode": cudagraph_runtime_mode,
                "batch_size_across_dp": batch_size_across_dp,
                "batch_desc": self._last_batch_desc,
                "hidden_states": hidden_states,
                "last_tokens": draft_token_ids,
                # For the W7_AHEAD_FROZEN_MD bisect: the chain's last built
                # per-layer metadata, reused verbatim by the ahead steps.
                "per_layer_attn_metadata": per_layer_attn_metadata,
            }
            if self._consume_mode:
                # OV1(b) bootstrap: persist the dispatch keys + private
                # metadata object for the steady-state re-anchor cycles
                # (propose stops running), and seed the state machine: the
                # chain's last draft d_K is KV-pending.
                self._ahead_dispatch = {
                    "input_batch_size": input_batch_size,
                    "block_size": block_size,
                    "mode": cudagraph_runtime_mode,
                    "batch_size_across_dp": batch_size_across_dp,
                    "batch_desc": self._last_batch_desc,
                    "per_layer_attn_metadata": per_layer_attn_metadata,
                }
                self._ahead_cad = cad_priv
            if self._propose_done_event is None:
                self._propose_done_event = torch.cuda.Event()
            self._propose_done_event.record()

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    def _side_stream_and_pool(self):
        """Lazily create the side stream + done event; return the mem-pool ctx
        (private pool preserving the main allocator's determinism) unless
        disabled via W7_SHADOW_NO_MEM_POOL."""
        from contextlib import nullcontext

        if self._shadow_stream is None:
            self._shadow_stream = torch.cuda.Stream()
        if self._shadow_done_event is None:
            self._shadow_done_event = torch.cuda.Event()
        if bool(int(os.environ.get("W7_SHADOW_NO_MEM_POOL", "0"))):
            return nullcontext()
        if self._shadow_mem_pool is None:
            self._shadow_mem_pool = torch.cuda.MemPool()
            logger.info(
                "Self-spec OV0b/OV1: side-stream transient allocations pinned "
                "to a private MemPool (preserves the main allocator's address "
                "determinism for piecewise replay)."
            )
        return torch.cuda.use_mem_pool(self._shadow_mem_pool)

    def consume_ahead(
        self,
        next_token_ids: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None,
        common_attn_metadata,
    ) -> torch.Tensor | None:
        """OV1(b2) DELTA-SLICE consumption (post-sampler, depth-1 truth).

        Always stashes the ground-truth anchor state for the next
        run_ahead_chain (the actual bonus/corrected token, committed lengths,
        fresh block-table snapshot). If the previous run's 2K+1 predictions
        are available, serves outs[Delta : Delta+K] where Delta =
        committed_now - committed_the_run_anchored_on -- exactly the
        predictions for the live positions. Rows whose intervening commits
        diverged from the chain's predictions serve garbage (losslessly
        rejected; the next run re-anchors on actuals and heals in one cycle).
        Returns None on bootstrap/fence cycles (caller runs propose).
        """
        bs = common_attn_metadata.batch_size()
        if num_rejected_tokens_gpu is None:
            self._ahead_out = None
            return None
        # OV1(b2) per-row consumption: ALWAYS serve the ahead drafts once warm
        # (no host sync). Rows whose speculation failed carry garbage drafts
        # for one cycle -- losslessly rejected by the verify -- and the next
        # run re-anchors them per row from the stash below. seq_lens -
        # rejected = the committed length through the accepted prefix
        # INCLUDING this step's input token; the new bonus/corrected token
        # (next_token_ids) sits AT that position (0-indexed).
        rejected = num_rejected_tokens_gpu[:bs]
        # DP1 deterministic repro (W7_B2_DUMP=path): one JSONL event per
        # consume with row-0's verify verdict + the drafts being served.
        # Host-syncing; debug only.
        _dump = os.environ.get("W7_B2_DUMP", "")
        if _dump:
            import json as _json

            self._dump_cyc = getattr(self, "_dump_cyc", 0) + 1
            with open(f"{_dump}.rank{self.dp_rank}.jsonl", "a") as f:
                f.write(
                    _json.dumps(
                        {
                            "ev": "consume",
                            "cyc": self._dump_cyc,
                            "b": int(next_token_ids[0].item()),
                            "rej": int(rejected[0].item()),
                            "committed": int(
                                (
                                    common_attn_metadata.seq_lens[0]
                                    - rejected[0]
                                ).item()
                            ),
                            "served": (
                                self._ahead_outs_full[0].tolist()
                                if self._ahead_outs_full is not None
                                else []
                            ),
                        }
                    )
                    + "\n"
                )
        committed_now = (common_attn_metadata.seq_lens[:bs] - rejected).long()
        self._reanchor = {
            "b": next_token_ids[:bs].long().clone(),
            "rejected": rejected.clone(),
            "committed": committed_now.clone(),
            "block_table": common_attn_metadata.block_table_tensor.clone(),
        }
        # Delta-slice serve: predictions exist for anchor_committed+1 ..
        # anchor_committed+2K+1; the live positions start at committed_now+1.
        if (
            self._ahead_outs_full is None
            or self._ahead_outs_full.shape[0] != bs
            or self._ahead_anchor_committed is None
        ):
            self._ahead_out = None
            # Order the next side run after the stash clones above.
            if self._propose_done_event is None:
                self._propose_done_event = torch.cuda.Event()
            self._propose_done_event.record()
            return None
        k = self.num_speculative_tokens
        delta = (
            (committed_now - self._ahead_anchor_committed[:bs])
            .clamp(1, k + 1)
            .unsqueeze(1)
        )
        idx = delta + torch.arange(
            k, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        served = torch.gather(self._ahead_outs_full, 1, idx)
        self._ahead_outs_full = None
        # Record AFTER the gather: the next side run waits on this event, so
        # its writes into (possibly allocator-reused) blocks cannot race the
        # gather's reads (cross-stream reuse hazard seen in the DP1 repro).
        if self._propose_done_event is None:
            self._propose_done_event = torch.cuda.Event()
        self._propose_done_event.record()
        return served

    def fence_ahead_chain(self) -> None:
        """OV1(a): order the MAIN stream after the in-flight ahead chain.

        Called by the runner before executing a step that is NOT a pure
        uniform-decode continuation (contains prefills / batch-composition
        changes). At such boundaries the ahead chain's assumptions are void
        AND its in-flight side-stream work races the main stream: the new
        step's DRAFT prefill shares the draft workspace and persistent
        buffers, freed KV pages get reallocated to new prompts while the
        ahead chain may still write its (cloned-table) slots, and large
        prefill allocations can reclaim cached blocks the in-flight replay
        has baked. Pure-decode steady state never takes this fence, so the
        overlap is untouched where it matters.
        """
        if self._shadow_done_event is not None:
            torch.cuda.current_stream().wait_event(self._shadow_done_event)
        self._ahead_tokens = None
        self._ahead_state = None
        # Consume-mode state is void across batch boundaries: drop the drafts,
        # the re-anchor stash, and the pending-token state so the next cycle
        # bootstraps via propose.
        self._ahead_out = None
        self._ahead_outs_full = None
        self._ahead_anchor_committed = None
        self._reanchor = None
        self._ahead_pending_tok = None
        self._ahead_pending_pos = None
        self._expect_tok = None
        self._expect_rej = None

    def run_ahead_chain(self) -> None:
        """OV1(a) (phase 49): continue the draft chain K+1 steps on the side
        stream, concurrently with the verify the runner just enqueued.

        This is the free-running draft's core move: the chain's most-likely
        future is its own continuation (all K drafts accepted; the bonus token
        = the chain's next argmax). Real inputs, real position/metadata
        updates, real KV writes (positions covered by the extended scheduler
        lookahead). VALIDATION mode: the produced tokens [bonus-guess,
        e1..eK] are stashed for the hit-rate check at the next propose and
        then discarded -- the normal propose still runs, so the served output
        is byte-identical while the overlap machinery runs for real.

        Requires the phase-49 isolation stack (dedicated draft cudagraph pool
        + dedicated draft MoE workspace); the next propose waits on the done
        event before touching shared buffers.
        """
        if self.uses_mrope or self.supports_mm_inputs:
            return
        # Mode select: re-anchor (OV1(b) consume, steady state -- start from
        # the ACTUAL committed token, per-row) vs continuation (post-propose
        # bootstrap / OV1(a) validation -- continue the chain's own guess).
        ra = self._reanchor if self._consume_mode else None
        self._reanchor = None
        # Cross-stream lifetime: ra's tensors are read by SIDE-stream kernels
        # enqueued below; freeing them at host-return lets the allocator
        # reuse their blocks for main-stream work while side reads are still
        # pending (observed: position-arithmetic values leaking into step-0
        # inputs). Hold them until the next run replaces the reference.
        self._ra_keepalive = ra if ra is not None else self._ra_keepalive
        state = self._ahead_state
        self._ahead_state = None
        if ra is not None and self._ahead_dispatch is not None:
            disp = self._ahead_dispatch
        elif state is not None and not self._consume_mode:
            # OV1(a) validation-mode continuation only; consume mode always
            # anchors on the stash (bootstrap = a propose cycle, no special
            # ahead mode).
            ra = None
            disp = None
        else:
            return
        mem_pool_ctx = self._side_stream_and_pool()
        n_ahead = int(
            os.environ.get(
                "W7_AHEAD_STEPS", str(self.num_speculative_tokens + 1)
            )
        )
        # Bisect knob: suppress the ahead chain's KV writes (fill the slot
        # buffer with PADDING after each position update). Isolates the
        # KV-write/block-bounds path; the ahead drafts degrade (no KV for
        # ahead tokens) but that is irrelevant while bisecting a crash.
        no_kv = bool(int(os.environ.get("W7_AHEAD_NO_KV", "0")))
        # Bisect knob: skip the per-step compute_logits+argmax (feed the same
        # token every step instead). Isolates the LM-head/greedy path.
        no_sample = bool(int(os.environ.get("W7_AHEAD_NO_SAMPLE", "0")))
        # Bisect knob: FREEZE the metadata path -- skip the position/slot
        # update kernel and the per-step metadata build, reusing the chain's
        # last built metadata (shadow-style). Implies KV-off (slots stale).
        # Every crashing config so far ran the live metadata path; the clean
        # concurrent shadow never did -- this isolates it.
        frozen_md = bool(int(os.environ.get("W7_AHEAD_FROZEN_MD", "0")))
        if frozen_md:
            no_kv = True
        # Debug knob (W7_AHEAD_SYNC=1): synchronize the SIDE STREAM ONLY after
        # each ahead sub-op (main-stream concurrency -- the repro condition --
        # is preserved) so the sticky async CUDA error surfaces at the exact
        # faulting sub-op instead of at a later unrelated launch.
        dbg_sync = bool(int(os.environ.get("W7_AHEAD_SYNC", "0")))
        self._ahead_cycles = getattr(self, "_ahead_cycles", 0) + 1

        def _ck(label: str, j: int) -> None:
            if not dbg_sync:
                return
            try:
                self._shadow_stream.synchronize()
            except Exception:
                logger.error(
                    "OV1(a) FAULT at cycle %d, ahead step %d, after %s",
                    self._ahead_cycles,
                    j,
                    label,
                )
                raise
        if disp is not None:
            # PER-ROW mode (OV1(b2)). Test what the in-flight verify was
            # EXPECTED to do (recorded by the previous run): normal rows
            # expect rej==0 and bonus == the anchor they consumed;
            # re-anchored rows expect their stale drafts fully rejected
            # (rej==K) and the correction == their out[0] guess. Valid rows
            # continue from the pending guess token; invalid rows re-anchor
            # on the actual bonus/corrected token at the committed position.
            # Uniform K+2 forwards; drafts by per-row offset gather.
            bs = ra["b"].shape[0]
            ibs = disp["input_batch_size"]
            cad = self._ahead_cad
            # DELTA-SLICE design (the DP1 repro's verdict): the run always
            # anchors on ground truth -- the actual bonus/corrected token at
            # the actual committed position from its (one-cycle-old) stash --
            # and produces 2K+1 predictions for positions committed+1 ..
            # committed+2K+1. By consume time the live commit level has
            # advanced by Delta in [1, K+1]; consume serves outs[Delta :
            # Delta+K], exactly the predictions for the live positions.
            # Anchors are never speculative, so misses self-heal in one
            # cycle; no pending/expect/bootstrap state machine exists.
            n_ahead = 2 * self.num_speculative_tokens + 1
            if self._ahead_pending_tok is not None:
                valid = (ra["rejected"] == 0) & (
                    ra["b"] == self._ahead_pending_tok[:bs]
                )  # diagnostic only (hit-rate logging)
            else:
                valid = ra["rejected"] == 0
            if bool(int(os.environ.get("W7_AHEAD_DEBUG", "0"))):
                self._dbg_n = getattr(self, "_dbg_n", 0) + 1
                if self._dbg_n % 50 == 1:
                    logger.info(
                        "OV1(b2) dbg cycle %d: hit=%.2f rej0=%.2f "
                        "tok_match=%.2f",
                        self._dbg_n,
                        valid.float().mean().item(),
                        (ra["rejected"] == 0).float().mean().item(),
                        (ra["b"] == self._ahead_pending_tok[:bs])
                        .float()
                        .mean()
                        .item(),
                    )
                    # Value-level view of the first mismatching row: is the
                    # expectation a repeat of the anchor, shifted by one, etc.
                    mism = (~(ra["b"] == self._ahead_pending_tok[:bs])).nonzero()
                    if mism.numel() > 0:
                        r = int(mism[0].item())
                        logger.info(
                            "OV1(b2) dbg row %d: b=%d expect=%d rej=%d "
                            "expect_rej=%d pending=%d committed=%d "
                            "pending_pos=%d",
                            r,
                            int(ra["b"][r].item()),
                            int(self._ahead_pending_tok[r].item()),
                            int(ra["rejected"][r].item()),
                            0,
                            int(self._ahead_pending_tok[r].item()),
                            int(ra["committed"][r].item()),
                            int(self._ahead_pending_pos[r].item()),
                        )
            anchor = ra["b"]
            p1 = ra["committed"]
            cad.block_table_tensor = ra["block_table"]
            cad._seq_lens_cpu = None
            cad._num_computed_tokens_cpu = None
            cad.max_seq_len = self.max_model_len
            # NOTE: the seq_lens/positions preset COPIES are deferred into the
            # side-stream context below. Enqueued here (main stream) they land
            # AFTER the just-launched verify and are NOT covered by the
            # consume-time propose_done event, so the side-stream forwards can
            # race them -- observed as position-correlated garbage tokens
            # (out[0] id == p1+2) on losing cycles in the DP1 repro.
            positions = self.positions[:bs]
            self._ahead_pending_pos = p1 + self.num_speculative_tokens + 1
            self._ahead_valid = valid
            hidden_states = None
            last_tokens = None
            block_size = disp["block_size"]
            if bool(int(os.environ.get("W7_AHEAD_DEBUG2", "0"))):
                self._dbg2_n = getattr(self, "_dbg2_n", 0) + 1
                if self._dbg2_n <= 8:
                    logger.info(
                        "OV1(b2) trace run %d row0: valid=%d anchor=%d p1=%d "
                        "| stash b=%d rej=%d committed=%d | expect=%d "
                        "expect_rej=%d pending=%d@%d",
                        self._dbg2_n,
                        int(valid[0].item()),
                        int(anchor[0].item()),
                        int(p1[0].item()),
                        int(ra["b"][0].item()),
                        int(ra["rejected"][0].item()),
                        int(ra["committed"][0].item()),
                        int(self._ahead_pending_tok[0].item()),
                        0,
                        int(self._ahead_pending_tok[0].item()),
                        int(self._ahead_pending_pos[0].item())
                        - self.num_speculative_tokens
                        - 1,
                    )
        else:
            bs = state["batch_size"]
            ibs = state["input_batch_size"]
            cad = state["cad"]
            positions = state["positions"]
            hidden_states = state["hidden_states"]
            last_tokens = state["last_tokens"]
            block_size = state["block_size"]
            disp = {
                "mode": state["mode"],
                "batch_size_across_dp": state["batch_size_across_dp"],
                "batch_desc": state["batch_desc"],
                "input_batch_size": ibs,
                "block_size": block_size,
                "per_layer_attn_metadata": state.get("per_layer_attn_metadata"),
            }
            if self._consume_mode:
                # Bootstrap: one extra forward so the run ends on a pending
                # GUESS token (outputs [b-guess, e1..eK, next-guess]), entering
                # the steady-state shape. Record the first consumed position
                # (d_K, one past the chain's current positions) for the
                # pending-position tracker.
                n_ahead += 1
                boot_p1 = positions.long() + 1
                self._ahead_pending_pos = boot_p1 + n_ahead
        # Allocated OUTSIDE the private pool: read by the next propose (main
        # stream, after the done-event wait) for the hit-rate check.
        tokens_out = torch.empty(
            (bs, n_ahead), dtype=torch.int64, device=self.device
        )
        assert self._shadow_stream is not None
        assert self._propose_done_event is not None
        # Debug knob (W7_SHADOW_MAIN_STREAM=1): run the ahead loop serialized
        # on the main stream -- with CUDA_LAUNCH_BLOCKING=1 this surfaces the
        # true faulting kernel if the loop has an intrinsic bug.
        if bool(int(os.environ.get("W7_SHADOW_MAIN_STREAM", "0"))):
            ahead_stream = torch.cuda.current_stream()
        else:
            ahead_stream = self._shadow_stream
        # Start only after the real chain's device work completes; the verify
        # was enqueued after it on the main stream, so this still overlaps.
        ahead_stream.wait_event(self._propose_done_event)
        with mem_pool_ctx, torch.cuda.stream(ahead_stream):
            if ra is not None:
                # Deferred presets, enqueued ON THE SIDE STREAM so they are
                # ordered before this run's forwards (see note above).
                cad.seq_lens[:bs].copy_(p1)
                self.positions[:bs].copy_(p1 - 1)
            for j in range(n_ahead):
                # Per-step input selection. Re-anchor mode: step 0 consumes
                # the KV-pending last draft (2-pending rows) or the actual
                # bonus token (1-pending rows); step 1 consumes the bonus
                # (2-pending) or the step-0 output (1-pending); later steps
                # consume their own previous output. Continuation mode: step 0
                # consumes the chain's last token, later steps their output.
                if ra is not None:
                    last_tokens = anchor if j == 0 else tokens_out[:, j - 1]
                input_ids = last_tokens.int()
                if frozen_md:
                    per_layer_attn_metadata = disp["per_layer_attn_metadata"]
                    self._slot_mapping_buffer[:ibs].fill_(PADDING_SLOT_ID)
                else:
                    positions = self._update_positions_dependent_metadata(
                        positions, cad, bs, ibs, block_size
                    )
                    _ck("pos_slot_update", j)
                    if no_kv:
                        self._slot_mapping_buffer[:ibs].fill_(PADDING_SLOT_ID)
                    _, per_layer_attn_metadata = (
                        self.build_per_group_and_layer_attn_metadata(
                            cad,
                            draft_index=self.num_speculative_tokens + j + 1,
                        )
                    )
                    _ck("metadata_build", j)
                self.input_ids[:bs] = input_ids
                if hidden_states is not None:
                    self.hidden_states[:bs] = hidden_states
                model_kwargs: dict[str, Any] = {
                    "input_ids": self.input_ids[:ibs],
                    "positions": self._get_positions(ibs),
                    "inputs_embeds": None,
                }
                if self.pass_hidden_states_to_model:
                    model_kwargs["hidden_states"] = self.hidden_states[:ibs]
                with set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=ibs,
                    num_tokens_across_dp=disp["batch_size_across_dp"],
                    cudagraph_runtime_mode=disp["mode"],
                    batch_descriptor=disp["batch_desc"],
                    slot_mapping=self._get_slot_mapping(ibs),
                    additional_kwargs=self._draft_forward_additional_kwargs,
                ):
                    ret_hidden_states = self.model(**model_kwargs)
                _ck("forward", j)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
                hidden_states = hidden_states[:bs]
                if no_sample:
                    tokens_out[:, j] = last_tokens[:bs].long()
                else:
                    last_tokens = self._greedy_sample(last_hidden_states[:bs])
                    tokens_out[:, j] = last_tokens
                _ck("sample", j)
                if bool(int(os.environ.get("W7_AHEAD_STEPLOG", "0"))):
                    self._steplog_n = getattr(self, "_steplog_n", 0)
                    if j == 0:
                        self._steplog_n += 1
                    if self._steplog_n <= 4:
                        torch.cuda.current_stream().synchronize()
                        logger.info(
                            "OV1(b2) step run%d j%d row0: in=%d pos=%d "
                            "slot=%d seq=%d -> out=%d",
                            self._steplog_n,
                            j,
                            int(self.input_ids[0].item()),
                            int(self.positions[0].item()),
                            int(self._slot_mapping_buffer[0].item()),
                            int(cad.seq_lens[0].item())
                            if cad is not None
                            else -1,
                            int(tokens_out[0, j].item()),
                        )
            assert self._shadow_done_event is not None
            self._shadow_done_event.record()
        self._ahead_tokens = tokens_out
        self._ahead_batch_size = bs
        # OV1(b) consume mode. Steady (re-anchor) runs produce [d1..dK, guess]
        # (K+1 outputs): drafts = [:K], the trailing guess is the next pending
        # anchor (pending_pos already advanced above). Bootstrap runs produce
        # [b-guess, e1..eK, next-guess] (K+2 outputs): drafts = [1:K+1].
        if self._consume_mode and n_ahead >= self.num_speculative_tokens + 1:
            k = self.num_speculative_tokens
            if ra is not None:
                # Delta-slice design: keep ALL 2K+1 predictions (positions
                # anchor_committed+1 .. +2K+1); consume slices per row by the
                # live commit delta. tokens_out[:, K] doubles as the
                # diagnostic bonus guess.
                self._ahead_outs_full = tokens_out
                self._ahead_anchor_committed = ra["committed"]
                self._ahead_out = tokens_out[:, :k]
                self._ahead_pending_tok = tokens_out[:, k]
            else:
                # Bootstrap: outputs [b-guess, e1..eK, next-guess]; propose's
                # real drafts face the verify normally (expect rej==0).
                self._ahead_out = tokens_out[:, 1 : k + 1]
                self._ahead_pending_tok = tokens_out[:, k + 1]
                # Bootstrap trailing guess doubles as the diagnostic
                # pending token; positioning no longer depends on it.
            _dump = os.environ.get("W7_B2_DUMP", "")
            if _dump:
                import json as _json

                self._dump_run = getattr(self, "_dump_run", 0) + 1
                if ra is not None:
                    rec = {
                        "ev": "run",
                        "n": self._dump_run,
                        "mode": "steady",
                        "anchor": int(anchor[0].item()),
                        "p1": int(p1[0].item()),
                        "outs": tokens_out[0].tolist(),
                    }
                else:
                    rec = {
                        "ev": "run",
                        "n": self._dump_run,
                        "mode": "boot",
                        "anchor": int(last_tokens[0].item())
                        if last_tokens is not None
                        else -1,
                        "p1": int(boot_p1[0].item()),
                        "outs": tokens_out[0].tolist(),
                    }
                with open(f"{_dump}.rank{self.dp_rank}.jsonl", "a") as f:
                    f.write(_json.dumps(rec) + "\n")
            if bool(int(os.environ.get("W7_AHEAD_DEBUG2", "0"))):
                if getattr(self, "_dbg2_n", 0) <= 8:
                    logger.info(
                        "OV1(b2) trace out row0 (%s): outputs=%s -> drafts=%s "
                        "pending=%d expect=%d",
                        "steady" if ra is not None else "boot",
                        tokens_out[0].tolist(),
                        self._ahead_out[0].tolist(),
                        int(self._ahead_pending_tok[0].item()),
                        int(self._ahead_pending_tok[0].item()),
                    )

    def shadow_replay_chain(self) -> None:
        """OV0b (phase 49): replay draft-chain decode-step forwards on a side
        stream, concurrently with the verify forward the runner just enqueued.

        Timing-only overlap probe. Uses the dispatch state cached at the end of
        the previous propose (stale token/position values are irrelevant to
        timing); fills the persistent slot-mapping buffer with PADDING_SLOT_ID
        first (enqueued on the side stream) so every KV write of the shadow
        forwards is discarded by the cache kernels -- the draft KV, the verify,
        and the served output are untouched. The real propose of this step
        waits on _shadow_done_event before reusing the buffers/graphs, so the
        measured step time is max(verify, shadow) + the serial remainder.

        What it probes: (a) whether the comm-free draft's compute can hide
        inside the verify's A2A window on this engine (tok/s unchanged =>
        hidden), and (b) cudagraph memory-pool aliasing between concurrently
        replaying draft and verify graphs (accept_len collapse => aliased).
        """
        ctx = self._shadow_ctx
        if (
            ctx is None
            or self._shadow_chain_steps <= 0
            or self.supports_mm_inputs
        ):
            return
        if self._shadow_stream is None:
            self._shadow_stream = torch.cuda.Stream()
            self._shadow_done_event = torch.cuda.Event()
            logger.info(
                "Self-spec OV0b: shadow chain ACTIVE (%d step(s)/cycle on a "
                "side stream, KV writes discarded).",
                self._shadow_chain_steps,
            )
            # Diagnostic: the KV-write suppression relies on the cached
            # attention metadata's slot_mapping ALIASING _slot_mapping_buffer
            # (so the -1 fill propagates into the eager attention's cache
            # write). Log whether that holds; if False, shadow KV writes hit
            # real slots and corrupt the draft KV (accept collapse).
            for _md in ctx["per_layer_attn_metadata"].values():
                _sm = getattr(_md, "slot_mapping", None)
                if _sm is not None:
                    logger.info(
                        "Self-spec OV0b: metadata slot_mapping aliases the "
                        "proposer slot buffer: %s",
                        _sm.data_ptr() == self._slot_mapping_buffer.data_ptr(),
                    )
                break
        ibs = ctx["input_batch_size"]
        # A/B discriminator (W7_SHADOW_MAIN_STREAM=1): run the shadow on the
        # MAIN stream (fully serialized after the verify) instead of the side
        # stream. If the accept collapse persists serialized, the corruption is
        # in the replay itself (KV/metadata); if it disappears, it needs
        # concurrency (streams/allocator).
        if bool(int(os.environ.get("W7_SHADOW_MAIN_STREAM", "0"))):
            shadow_stream = torch.cuda.current_stream()
        else:
            shadow_stream = self._shadow_stream
        # Do not race the PREVIOUS propose's chain kernels (same graphs/
        # workspaces + this fill races their slot-buffer reads): start only
        # after its device work completes. The verify was enqueued after it on
        # the main stream, so this wait still overlaps the verify.
        assert self._propose_done_event is not None
        shadow_stream.wait_event(self._propose_done_event)
        from contextlib import nullcontext

        if bool(int(os.environ.get("W7_SHADOW_NO_MEM_POOL", "0"))):
            mem_pool_ctx: Any = nullcontext()
        else:
            if self._shadow_mem_pool is None:
                self._shadow_mem_pool = torch.cuda.MemPool()
                logger.info(
                    "Self-spec OV0b: shadow transient allocations pinned to a "
                    "private MemPool (preserves the main allocator's address "
                    "determinism for piecewise replay)."
                )
            mem_pool_ctx = torch.cuda.use_mem_pool(self._shadow_mem_pool)
        # Bisection knobs: NO_FORWARD keeps the fill/context/event machinery but
        # skips the model call; GEMM_ONLY replaces the model call with private
        # matmuls (no vLLM state touched at all).
        no_forward = bool(int(os.environ.get("W7_SHADOW_NO_FORWARD", "0")))
        gemm_only = bool(int(os.environ.get("W7_SHADOW_GEMM_ONLY", "0")))
        with mem_pool_ctx, torch.cuda.stream(shadow_stream):
            # Discard all shadow KV writes (the cache kernels skip slot -1).
            # The real propose refills this buffer every step before use.
            self._slot_mapping_buffer.fill_(PADDING_SLOT_ID)
            if gemm_only:
                if not hasattr(self, "_shadow_gemm_ab"):
                    self._shadow_gemm_ab = (
                        torch.randn(4096, 4096, dtype=self.dtype,
                                    device=self.device),
                        torch.randn(4096, 4096, dtype=self.dtype,
                                    device=self.device),
                    )
                a, b = self._shadow_gemm_ab
                # ~one K=2 chain's worth of side-stream compute (~30 ms).
                for _ in range(90 * self._shadow_chain_steps):
                    torch.mm(a, b)
                assert self._shadow_done_event is not None
                self._shadow_done_event.record()
                return
            model_kwargs: dict[str, Any] = {
                "input_ids": self.input_ids[:ibs],
                "positions": self._get_positions(ibs),
                "inputs_embeds": None,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:ibs]
            for _ in range(self._shadow_chain_steps):
                with set_forward_context(
                    ctx["per_layer_attn_metadata"],
                    self.vllm_config,
                    num_tokens=ibs,
                    num_tokens_across_dp=ctx["batch_size_across_dp"],
                    cudagraph_runtime_mode=ctx["mode"],
                    batch_descriptor=ctx["batch_desc"],
                    slot_mapping=self._get_slot_mapping(ibs),
                    additional_kwargs=self._draft_forward_additional_kwargs,
                ):
                    if not no_forward:
                        self.model(**model_kwargs)
            assert self._shadow_done_event is not None
            self._shadow_done_event.record()

    def _update_positions_dependent_metadata(
        self,
        positions: torch.Tensor,
        common_attn_metadata,
        batch_size: int,
        input_batch_size: int,
        block_size: int,
    ) -> torch.Tensor:
        """Update positions, slot mappings, and sequence metadata for the
        next draft step. Returns the updated positions tensor."""
        positions_1d = positions[0] if self.uses_mrope else positions
        if self.uses_mrope:
            out_pos = self.mrope_positions[0, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            out_pos = self.xdrope_positions[0, :batch_size]
        else:
            out_pos = self.positions[:batch_size]
        eagle_step_update_slot_mapping_and_metadata(
            positions_1d=positions_1d,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=block_size,
            max_model_len=self.max_model_len,
            out_clamped_positions=out_pos,
            out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
            input_batch_size=input_batch_size,
        )
        common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
        if self.uses_mrope:
            self.mrope_positions[1:, :batch_size] = self.mrope_positions[0, :batch_size]
            positions = self.mrope_positions[:, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                0, :batch_size
            ]
            positions = self.xdrope_positions[0, :batch_size]
        else:
            positions = self.positions[:batch_size]
        common_attn_metadata.max_seq_len = min(
            common_attn_metadata.max_seq_len + 1,
            self.max_model_len,
        )

        if common_attn_metadata._seq_lens_cpu is not None:
            common_attn_metadata._seq_lens_cpu += 1
        if common_attn_metadata._num_computed_tokens_cpu is not None:
            common_attn_metadata._num_computed_tokens_cpu += 1
        if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
            common_attn_metadata.seq_lens_cpu_upper_bound += 1

        return positions

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        if not self.needs_extra_input_slots:
            # Default EAGLE pathway: no reshaping of input tensors needed.
            # Simply rotate the input ids and leave the positions unchanged,
            # Inserting the next token ids at the last slot in each request.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]
            # Shift the input ids by one token.
            # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            # Replace the last token with the next token.
            # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
            self.input_ids[token_indices_to_sample] = next_token_ids

            # copy inputs to buffer for cudagraph
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]
            self._set_positions(num_tokens, target_positions)

            self.hidden_states[:num_tokens] = target_hidden_states

            return num_tokens, token_indices_to_sample, cad
        else:
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # 1.
            # Call a custom triton kernel to copy input_ids and positions
            # into the correct slots in the preallocated buffers self.input_ids,
            # self.positions.
            batch_size = cad.batch_size()
            # Since we might have to copy a lot of data for prefills, we select the
            # block size based on the max query length and limit to max 256 slots/block.
            max_num_tokens_per_request = (
                cad.max_query_len + self.net_num_new_slots_per_request
            )
            BLOCK_SIZE_TOKENS = min(256, next_power_of_2(max_num_tokens_per_request))
            num_blocks = (
                max_num_tokens_per_request + BLOCK_SIZE_TOKENS - 1
            ) // BLOCK_SIZE_TOKENS
            total_num_input_tokens = target_token_ids.shape[0]
            total_num_output_tokens = total_num_input_tokens + (
                self.net_num_new_slots_per_request * batch_size
            )

            token_indices_to_sample = torch.empty(
                batch_size * self.extra_slots_per_request,
                dtype=torch.int32,
                device=self.device,
            )

            # Destination indices to write target_hidden_states into drafting buffer.
            out_hidden_state_mapping = torch.empty(
                total_num_input_tokens, dtype=torch.int32, device=self.device
            )

            # Kernel grid: one program per request (row)
            grid = (batch_size, num_blocks)
            query_start_loc = cad.query_start_loc
            query_end_loc = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                query_end_loc = query_end_loc - num_rejected_tokens_gpu

            copy_and_expand_eagle_inputs_kernel[grid](
                # (Padded) Inputs from the target model
                target_token_ids_ptr=target_token_ids,
                target_positions_ptr=target_positions,
                next_token_ids_ptr=next_token_ids,  # sampled tokens, one per request
                # Outputs to the drafting buffers
                out_input_ids_ptr=self.input_ids,
                out_positions_ptr=self.positions,  # Doesn't support mrope for now
                out_is_rejected_token_mask_ptr=self.is_rejected_token_mask,
                out_is_masked_token_mask_ptr=self.is_masked_token_mask,
                out_new_token_indices_ptr=token_indices_to_sample,
                out_hidden_state_mapping_ptr=out_hidden_state_mapping,
                # Input metadata
                query_start_loc_ptr=query_start_loc,
                query_end_loc_ptr=query_end_loc,
                padding_token_id=0,
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                # Sizing info
                # Note that we can deduce batch_size for free from the grid size
                total_input_tokens=total_num_input_tokens,
                num_padding_slots_per_request=self.extra_slots_per_request,
                shift_input_ids=self.pass_hidden_states_to_model,
                BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
            )
            if self.pass_hidden_states_to_model:
                assert self.parallel_drafting_hidden_state_tensor is not None
                self.hidden_states[out_hidden_state_mapping] = target_hidden_states
                # Use torch.where to avoid DtoH sync from boolean indexing
                mask = self.is_masked_token_mask[:total_num_output_tokens]
                torch.where(
                    mask.unsqueeze(1),
                    self.parallel_drafting_hidden_state_tensor,
                    self.hidden_states[:total_num_output_tokens],
                    out=self.hidden_states[:total_num_output_tokens],
                )

            # 2.
            # Recompute the slot mapping based on the new positions and
            # rejection mask.
            assert self.block_size > 0, "block_size has not been initialized."
            new_slot_mapping = compute_new_slot_mapping(
                cad=cad,
                new_positions=self.positions[:total_num_output_tokens],
                is_rejected_token_mask=self.is_rejected_token_mask[
                    :total_num_output_tokens
                ],
                block_size=self.block_size,
                num_new_tokens=self.net_num_new_slots_per_request,
                max_model_len=self.max_model_len,
            )

            if self._shared_kv:
                # Shared-KV self-draft: every step-0 input token except the
                # appended sampled-token slot was processed by the verify
                # forward THIS step (prompt chunks included), so its slot
                # already holds target-exact KV -- a draft rewrite would
                # clobber it (fatal for the fp8-replica draft, ULP-noise for
                # the bf16 self-draft). PAD-mask everything but the appended
                # slots; those are not yet verified and the next verify pass
                # overwrites them (the existing provisional-KV discipline).
                keep = new_slot_mapping[token_indices_to_sample]
                new_slot_mapping = torch.full_like(
                    new_slot_mapping, PADDING_SLOT_ID
                )
                new_slot_mapping[token_indices_to_sample] = keep

            # 3. Update the common attention metadata with the new (meta)data
            new_cad = extend_all_queries_by_N(
                cad,
                N=self.net_num_new_slots_per_request,
                arange=self.arange,
                new_slot_mapping=new_slot_mapping,
            )

            return total_num_output_tokens, token_indices_to_sample, new_cad

    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

        return model_kwargs, num_input_tokens

    def _kv_window_step0_active(
        self,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> bool:
        """Whether the step-0 draft forward should be windowed.

        Only decode-shaped proposes are windowed: prompt-ingest (prefill)
        passes stay full-KV so the draft's prompt KV is written exactly
        (the window then only affects reads and tokens generated after it
        engaged).
        """
        return (
            self._kv_window > 0
            and num_rejected_tokens_gpu is not None
            and cad.max_query_len
            <= self.num_speculative_tokens + 1 + self.net_num_new_slots_per_request
        )

    def _step0_decode_active(
        self,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> bool:
        """Whether to compact the step-0 draft forward to a q=1 decode.

        Only for the shared-KV text draft_model path on decode-shaped proposes
        (num_rejected_tokens_gpu present => not prompt-ingest). Restricted to
        the plain text case (no M-RoPE / xdrope / mm / hidden-state passthrough)
        so the compaction only has to gather input_ids/positions.
        """
        active = (
            self._shared_kv_step0_decode
            and self.num_speculative_tokens > 1
            and num_rejected_tokens_gpu is not None
            and not self.uses_mrope
            and self.uses_xdrope_dim == 0
            and not self.supports_mm_inputs
            and not self.pass_hidden_states_to_model
            and cad.max_query_len
            <= self.num_speculative_tokens + 1 + self.net_num_new_slots_per_request
        )
        # Phase 81 E2b: engagement debug (first calls only) -- the compaction
        # engaged intermittently on the composed stack; log WHICH condition
        # gates each decode-shaped propose.
        if self._shared_kv_step0_decode and os.environ.get("W7_STEP0_DEBUG"):
            n = getattr(self, "_step0_dbg_n", 0)
            if n < 12:
                self._step0_dbg_n = n + 1
                logger.info(
                    "[step0-gate] active=%s max_query_len=%s limit=%s(K=%s+1+net=%s) "
                    "rej_gpu=%s",
                    active, cad.max_query_len,
                    self.num_speculative_tokens + 1 + self.net_num_new_slots_per_request,
                    self.num_speculative_tokens, self.net_num_new_slots_per_request,
                    num_rejected_tokens_gpu is not None,
                )
        return active

    def _compact_step0_decode(
        self,
        cad: CommonAttentionMetadata,
        md_cad: CommonAttentionMetadata,
        token_indices_to_sample: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, int, torch.Tensor, torch.Tensor]:
        """Compact the step-0 draft forward to a q=1 decode of the appended
        sampled token(s) under shared KV.

        Gathers the appended input_ids/positions to the FRONT of the proposer
        buffers and returns a q=1 CommonAttentionMetadata that reuses md_cad's
        (windowed) per-request seq_lens/block_table -- so the appended token
        attends to the identical key set the q=(K+2) step-0 would have used,
        skipping the wasted forward over the K+1 re-ingested verify tokens
        whose draft KV writes are PAD-masked anyway.

        seq_lens from the padded path span the FULL padded query (rejected
        slots included): correct for the causal ragged forward, but a q=1
        decode has no causal mask and would attend the stale KV of this
        step's rejected draft tokens -- fatal under a KV window, where those
        slots sit among the ~window most-recent keys. Trim per-request
        seq_lens by num_rejected so the appended token attends exactly the
        keys its causal counterpart would have. Returns
        (compact_cad, num_tokens=batch, token_indices=arange, appended_slots).
        """
        bs = cad.num_reqs
        idx = token_indices_to_sample
        # RHS advanced indexing returns a copy, so front-gather is aliasing-safe.
        self.input_ids[:bs] = self.input_ids[idx]
        self.positions[:bs] = self.positions[idx]
        appended_slots = cad.slot_mapping[idx]
        if md_cad is not cad:
            # Windowed view: seq_lens is a proposer-owned buffer rewritten by
            # _apply_draft_kv_window each step -- trim in place (captured
            # graphs read it through capture-time pointers).
            md_cad.seq_lens[:bs].sub_(num_rejected_tokens_gpu[:bs])
            compact_seq_lens = md_cad.seq_lens
        else:
            # Full-KV: seq_lens is the runner's shared buffer -- copy.
            compact_seq_lens = md_cad.seq_lens.clone()
            compact_seq_lens[:bs].sub_(num_rejected_tokens_gpu[:bs])
        compact = md_cad.replace(
            seq_lens=compact_seq_lens,
            query_start_loc=self.arange[: bs + 1],
            query_start_loc_cpu=torch.from_numpy(
                self.token_arange_np[: bs + 1]
            ).clone(),
            num_actual_tokens=bs,
            max_query_len=1,
            slot_mapping=appended_slots,
        )
        return compact, bs, self.arange[:bs], appended_slots

    def _scratchpad_n_kept_blocks(self) -> int:
        """Fixed number of kept (sink + trailing-window) pages per chain step.

        Uses the MAX drafted count (num_speculative_tokens) so the gather shape
        is cycle-constant even as the drafted suffix grows -- the padding mask
        (col >= live seq_len) excludes the not-yet-filled slots. For a
        block-multiple window this equals the paged per-step ``n_last``.
        """
        block_size = self.block_size
        n_sink = -(-self._kv_window_sinks // block_size)
        n_last = -(-(self._kv_window + self.num_speculative_tokens) // block_size)
        return n_sink + n_last

    def _ensure_scratchpad_ctx(self, bs: int):
        """Build/refresh the per-cycle draft scratchpad context.

        References the proposer's PERSISTENT window buffers (block table /
        seq_lens), which ``_apply_draft_kv_window`` rewrites in place each step,
        so a captured graph reads the live data through capture-time pointers.
        """
        from vllm.v1.spec_decode.scratchpad_attn import DraftScratchpadCtx

        n_kept = self._scratchpad_n_kept_blocks()
        cap = n_kept * self.block_size
        if self._sp_col_arange is None or self._sp_col_arange.shape[1] != cap:
            self._sp_col_arange = torch.arange(
                cap, device=self.device
            ).unsqueeze(0)
        self._sp_ctx = DraftScratchpadCtx(
            block_table=self._win_block_table,
            seq_lens=self._win_seq_lens,
            col_arange=self._sp_col_arange,
            block_size=self.block_size,
            n_kept_blocks=n_kept,
            num_reqs=bs,
        )
        if self._draft_fullcg and not self._sp_logged:
            self._sp_logged = True
            logger.info(
                "Draft window-scratchpad attention active: n_kept_blocks=%d "
                "cap=%d (sinks=%d window=%d K=%d block=%d).",
                n_kept,
                cap,
                self._kv_window_sinks,
                self._kv_window,
                self.num_speculative_tokens,
                self.block_size,
            )
        return self._sp_ctx

    def _populate_dummy_scratchpad(self, n: int) -> None:
        """Allocate + fill the window buffers with safe in-range dummy data for
        the FULL-cudagraph CAPTURE of the scratchpad forward.

        Matches ``_apply_draft_kv_window``'s allocation exactly (same n_cols =
        the runner draft block-table width, same dtypes) so the real propose
        reuses the buffers WITHOUT reallocation -> the captured gather pointers
        stay live. Block 0 (always in-range) + all-valid seq_len avoids OOB /
        all-masked-NaN during capture.
        """
        runner = self.runner
        assert runner is not None
        blk = runner.input_batch.block_table[self.kv_cache_gid].get_device_tensor(n)
        block_size = self.block_size
        n_sink = -(-self._kv_window_sinks // block_size)
        rows = max(self.max_batch_size, n)
        # Phase 93 IMA fix: fixed max-width allocation, identical to
        # _apply_draft_kv_window (captured graphs hold these pointers;
        # never reallocate).
        max_cols = -(-self.max_model_len // block_size)
        if self._win_block_table is None:
            self._win_block_table = torch.zeros(
                (rows, max_cols), dtype=blk.dtype, device=self.device
            )
            self._win_seq_lens = torch.zeros(
                rows, dtype=runner.seq_lens.dtype, device=self.device
            )
            self._win_col_arange = torch.arange(
                max_cols, device=self.device
            ).unsqueeze(0)
            self._win_src_cols = torch.zeros(
                (rows, max_cols), dtype=torch.long, device=self.device
            )
            self._win_col_ge_sink = (self._win_col_arange >= n_sink).to(torch.long)
        self._win_block_table[:n].zero_()
        cap = self._scratchpad_n_kept_blocks() * block_size
        self._win_seq_lens[:n] = cap

    def _apply_draft_kv_window(
        self, cad: CommonAttentionMetadata, tokens_drafted: int = 0
    ) -> CommonAttentionMetadata:
        """Compact each request's page list to sinks + trailing window.

        Keeps the first ceil(SINKS/block_size) blocks plus the last blocks
        covering (window + tokens_drafted) tokens, and shrinks the draft-side
        kv seq_len to the token count actually present in the kept pages
        (partial last block included). Paged KV entries carry their true
        RoPE'd positions and FA applies the causal mask aligned at the END of
        the provided KV sequence, so the compacted view is exactly
        sinks+window attention -- no position surgery.

        Returns a shallow ``replace`` of ``cad`` pointing at proposer-owned
        persistent buffers; ``cad`` itself (the TRUE state used for
        slot-mapping/position updates) is not modified, so draft KV writes
        still land in the correct global slots. Recomputed per draft step so
        the newest pages (the KV the chain appends) are always in the kept
        set. Same-stream ordering makes the in-place buffer rewrite safe
        (data change, not shape change).
        """
        bs = cad.num_reqs
        block_size = self.block_size
        n_sink = -(-self._kv_window_sinks // block_size)
        n_last = -(-(self._kv_window + tokens_drafted) // block_size)
        src_bt = cad.block_table_tensor
        n_cols = src_bt.shape[1]
        # Phase 93 IMA fix: the buffers are read by CAPTURED graphs through
        # capture-time pointers, so they must NEVER be reallocated after the
        # first capture ("data change, not shape change"). The runner's block
        # table WIDTH grows dynamically as sequences lengthen; the old
        # width-tracking realloc freed the captured pointers mid-serving
        # (racy IMA, batch/mml-dependent). Allocate ONCE at the maximum
        # width and operate on left-slices.
        max_cols = -(-self.max_model_len // block_size)
        assert n_cols <= max_cols, (n_cols, max_cols)
        if self._win_block_table is None:
            rows = max(self.max_batch_size, bs)
            self._win_block_table = torch.zeros(
                (rows, max_cols), dtype=src_bt.dtype, device=self.device
            )
            self._win_seq_lens = torch.zeros(
                rows, dtype=cad.seq_lens.dtype, device=self.device
            )
            self._win_col_arange = torch.arange(
                max_cols, device=self.device
            ).unsqueeze(0)
            # Phase 65: persistent gather-index buffer + (col >= n_sink) mask
            # (n_sink is constant for the proposer's lifetime).
            self._win_src_cols = torch.zeros(
                (rows, max_cols), dtype=torch.long, device=self.device
            )
            self._win_col_ge_sink = (
                self._win_col_arange >= n_sink
            ).to(torch.long)
        assert self._win_seq_lens is not None
        assert self._win_col_arange is not None
        assert self._win_src_cols is not None
        assert self._win_col_ge_sink is not None
        seq_lens = cad.seq_lens[:bs].long()
        n_total = (seq_lens + block_size - 1) // block_size
        # First kept trailing block; rows with dropped == 0 stay uncompacted
        # (short rows where sinks + window already cover every block).
        start_last = torch.minimum(
            torch.clamp(n_total - n_last, min=n_sink), n_total
        )
        dropped = torch.clamp(start_last - n_sink, min=0)
        # left-slices of the fixed max-width buffers (see alloc note above)
        src_cols = self._win_src_cols[:bs, :n_cols]
        torch.mul(
            self._win_col_ge_sink[:, :n_cols],
            dropped.unsqueeze(1),
            out=src_cols,
        )
        src_cols += self._win_col_arange[:, :n_cols]
        src_cols.clamp_(max=n_cols - 1)
        torch.gather(
            src_bt[:bs], 1, src_cols,
            out=self._win_block_table[:bs, :n_cols],
        )
        self._win_seq_lens[:bs] = (seq_lens - dropped * block_size).to(
            self._win_seq_lens.dtype
        )
        if self._kv_window_debug:
            self._kv_window_calls += 1
            if self._kv_window_calls % 500 == 1:
                logger.info(
                    "[kv-window] call=%d bs=%d drafted=%d true_max_seq=%d "
                    "win_seq(min/max)=%d/%d kept_blocks<=%d",
                    self._kv_window_calls,
                    bs,
                    tokens_drafted,
                    cad.max_seq_len,
                    int(self._win_seq_lens[:bs].min()),
                    int(self._win_seq_lens[:bs].max()),
                    n_sink + n_last,
                )
        return cad.replace(
            block_table_tensor=self._win_block_table[:bs],
            seq_lens=self._win_seq_lens[:bs],
            max_seq_len=min(cad.max_seq_len, (n_sink + n_last) * block_size),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
        )

    def build_per_group_and_layer_attn_metadata(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _build_draft_decode_capture_metadata(
        self, batch_size: int, qlen: int = 1
    ) -> dict[str, object]:
        """Build per-layer attention metadata for the draft's FULL-cudagraph
        capture pass (W7, VLLM_SELF_SPEC_DRAFT_FULL_CG).

        The draft's per-decode-step forwards are uniform 1-token-per-seq decode
        batches. To capture the *real* attention kernel (and not the
        attn_metadata-is-None zero-fill stub), capture must run with a real
        CommonAttentionMetadata built from the SAME persistent buffers the
        per-step replay reads -- the runner's block table + seq_lens, the
        proposer's arange/slot_mapping buffer, and the builder's persistent
        scheduler_metadata. Pointer stability across capture/replay is what
        makes the captured FULL graph attend against live metadata.

        Backend-agnostic: the metadata is built through the draft builder's
        own ``build_for_cudagraph_capture`` (no backend branching), so it
        covers both FLASH_ATTN_MLA (DeepSeek-V2-Lite) and FLASH_ATTN (GQA
        models such as Qwen3-30B). Both builders share the same FA3
        scheduler_metadata + split machinery; the matching split cap is
        applied in ``initialize_attn_backend``. Other backends without that
        machinery still build real metadata here (no-op split cap).
        """
        runner = self.runner
        assert runner is not None, (
            "Draft FULL cudagraph capture requires a runner reference."
        )
        # Runner's persistent block table for the draft's kv-cache group; the
        # runner already populated seq_lens (= max_query_len) before calling
        # the draft dummy_run during capture.
        num_reqs = batch_size // qlen
        blk_table = runner.input_batch.block_table[self.kv_cache_gid]
        block_table_tensor = blk_table.get_device_tensor(num_reqs)
        seq_lens = runner.seq_lens[:num_reqs]
        seq_lens_cpu = runner.optimistic_seq_lens_cpu[:num_reqs]
        if qlen > 1:
            # Phase 55 (step0 FULL capture): capture reads seq_lens through
            # the proposer-owned buffer -- the live step-0 metadata's
            # seq_lens (extend_all_queries_by_N) is a fresh tensor each
            # cycle, so replay refreshes this buffer instead (propose()).
            assert self._step0_seq_lens is not None
            self._step0_seq_lens[:num_reqs].copy_(seq_lens)
            self._step0_seq_lens[num_reqs:].zero_()
            seq_lens = self._step0_seq_lens[:num_reqs]
        # Bake a LARGE step-invariant max_seq_len upper bound into the captured
        # graph (mirrors V2's draft_max_seq_len). The runner's capture-time
        # seq_lens are tiny (= the verify query len) while replay seq_lens grow
        # into the hundreds; using a large upper bound keeps the captured FA3
        # schedule valid across all steps. The live per-step seq_lens tensor
        # (with scheduler_metadata recomputed each step) drives the actual key
        # count at replay.
        max_seq_len = self.max_model_len

        if qlen == 1:
            query_start_loc = self.arange[: batch_size + 1]
            query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone()
        else:
            # Phase 55 (step0 FULL capture): strided starts for uniform q>1.
            # Constant per shape; cached so capture and any later rebuild of
            # the same shape share ONE tensor (kept alive for the captured
            # graph's lifetime; runtime dummy rebuilds must not grow it).
            _qsl_key = (num_reqs, qlen)
            if _qsl_key not in self._step0_qsl_cache:
                self._step0_qsl_cache[_qsl_key] = (
                    self.arange[: num_reqs + 1] * qlen
                ).contiguous()
            query_start_loc = self._step0_qsl_cache[_qsl_key]
            query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: num_reqs + 1] * qlen
            ).clone()

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            _seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=batch_size,
            max_query_len=qlen,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_tensor,
            slot_mapping=self._slot_mapping_buffer[:batch_size],
            causal=True,
        )

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            builder = attn_group.get_metadata_builder()
            attn_metadata = builder.build_for_cudagraph_capture(common_attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_layer_attn_metadata

    def model_returns_tuple(self) -> bool:
        if self.method == "mtp":
            # DeepSeek-family MTP (deepseek_mtp.py) recycles the post-final-
            # norm hidden, so its forward returns (logit_hidden,
            # recycle_hidden). Other MTP families return a single tensor.
            return "DeepSeekMTPModel" in (
                self.draft_model_config.hf_config.architectures or []
            )
        return self.method not in ("mtp", "draft_model", "dflash")

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute backup token IDs for discarded requests.
        num_reqs = gpu_input_batch.num_reqs
        for i in range(num_reqs):
            self.backup_next_token_ids.np[i] = requests[
                gpu_input_batch.req_ids[i]
            ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        # upper_bound - rejected = actual post-rejection seq_lens (no D2H sync).
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        new_seq_lens_cpu = (
            common_attn_metadata.seq_lens_cpu_upper_bound - num_rejected_tokens
        )

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = async_tensor_h2d(token_indices_np, device=device)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=async_tensor_h2d(new_query_start_loc_cpu, device=device),
            seq_lens=async_tensor_h2d(new_seq_lens_cpu, device=device),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        if hasattr(model, "module"):  # multi-GPU
            model = model.module
        return model.__class__.__name__

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Return a VllmConfig with kernel-level overrides for the proposer.
        Subclasses may override to apply additional config changes.
        """
        spec_cfg = self.speculative_config
        base = self.vllm_config

        if spec_cfg.moe_backend is not None:
            base = replace(
                base,
                kernel_config=replace(
                    base.kernel_config,
                    moe_backend=spec_cfg.moe_backend,
                ),
            )

        # Note (matt): Never inherit the attention backend from base, because there are
        # many opportunities for incompatibility, so we always independently autoselect
        # unless explicitly specified in the speculative config.
        base = replace(
            base,
            attention_config=replace(
                base.attention_config,
                backend=spec_cfg.attention_backend,
            ),
        )

        # Self-spec W2a FULL REPLICA: create_draft_parallel_config builds the
        # draft ParallelConfig with enable_expert_parallel overridden to False
        # (full bf16 replica) when VLLM_SELF_SPEC_DRAFT_FULL_REPLICA is set. The
        # draft model is built from `base` (the TARGET vllm_config), whose
        # parallel_config keeps the target's enable_expert_parallel=True -- so the
        # override was silently discarded and the draft built use_ep=True, an
        # EP-shard (e.g. 7-8/60 experts/rank at DP=8) instead of a full replica.
        # Propagate ONLY the enable_expert_parallel override onto the base
        # parallel_config (preserving all runtime DP fields like data_parallel_rank
        # and the master IP/port) so the draft's FusedMoE builds use_ep=False ->
        # expert_map=None -> every rank holds ALL experts. Gated to the flag so
        # default draft_model behavior is unchanged.
        if (
            envs.VLLM_SELF_SPEC_DRAFT_FULL_REPLICA
            and base.parallel_config.enable_expert_parallel
        ):
            base = replace(
                base,
                parallel_config=replace(
                    base.parallel_config,
                    enable_expert_parallel=False,
                ),
            )

        return base

    @staticmethod
    @_contextlib.contextmanager
    def draft_compile_ranges(vllm_config, spec_cfg):
        """Phase 82 (search-vs-trace gap G-B): the draft chain runs
        decode-sized shapes, but the inherited single compile range
        [1, max_num_batched_tokens] makes inductor tune those small
        shapes over the whole prefill-sized dynamic range -- measured 2x
        chain slowdown at endpoint 16384 vs 8192 (b8/14k: 513 vs 1077
        tok/s). Insert an endpoint at the draft's decode bound FOR THE
        DRAFT COMPILE ONLY (in place, restored after: the compilation
        config carries shared registries that must not be forked)."""
        cc = vllm_config.compilation_config
        sched = vllm_config.scheduler_config
        decode_bound = sched.max_num_seqs * (
            spec_cfg.num_speculative_tokens + 2
        )
        old = cc.compile_ranges_endpoints
        existing = old or [sched.max_num_batched_tokens]
        if decode_bound >= min(existing):
            yield
            return
        cc.compile_ranges_endpoints = sorted({decode_bound, *existing})
        logger.info(
            "Draft compile ranges split at decode bound %d (endpoints %s).",
            decode_bound, cc.compile_ranges_endpoints,
        )
        try:
            yield
        finally:
            cc.compile_ranges_endpoints = old

    def _get_model(self) -> nn.Module:
        """
        Default method to call get_model(). Can be overridden by subclasses which
        need to customize model loading.
        """
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        # The FusedMoE layer reads the parallel config from the GLOBAL
        # get_current_vllm_config() at construction time, not from the
        # vllm_config passed to get_model(). Without this context the draft
        # MoE would inherit the engine's (target) parallel_config -- discarding
        # the W2a full-replica enable_expert_parallel=False override (the draft
        # would build use_ep=True, an EP shard, not a full replica). Set the
        # draft config as current so the draft's MoE honors draft_vllm_config.
        # Phase 83: draft PARTIAL replica -- activate the build context so
        # FusedMoE construction installs the frequency-profiled per-layer
        # expert maps (weights for non-resident experts are never loaded).
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            draft_partial_replica_build,
        )

        with (
            set_current_vllm_config(draft_vllm_config),
            set_model_tag("eagle_head"),
            self.draft_compile_ranges(draft_vllm_config, self.speculative_config),
            draft_partial_replica_build(),
        ):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        return model

    def _register_shared_kv_layers(
        self,
        all_attn_layers: dict[str, Any],
        target_attn_layer_names: set[str],
    ) -> None:
        """Phase 66 shared-KV self-draft: map each draft attention layer to
        its target twin and register the pairs in the runner's
        shared_kv_cache_layers. The runner then skips the draft layers in
        get_kv_cache_spec() (no duplicate allocation, pool sized
        target-only), appends them to the target's KV cache group
        (maybe_add_kv_sharing_layers_to_kv_cache_groups), and aliases the
        TARGET tensors to the draft modules in initialize_kv_cache_tensors().
        The draft modules' own kv_sharing_target_layer_name stays None so
        their KV writes remain ENABLED: chain-drafted slots must be written
        (the next verify pass overwrites them with target-exact KV).
        """
        assert self.runner is not None, "shared-KV drafting needs the runner"
        mapping: dict[str, str] = {}
        for draft_name in sorted(self._draft_attn_layer_names):
            prefix, _, target_name = draft_name.partition(".")
            if prefix != "draft_model" or target_name not in target_attn_layer_names:
                raise ValueError(
                    "VLLM_SELF_SPEC_SHARED_KV: no target twin for draft "
                    f"attention layer {draft_name!r} (expected "
                    f"{target_name!r} among the target layers)"
                )
            draft_spec = all_attn_layers[draft_name].get_kv_cache_spec(
                self.vllm_config
            )
            target_spec = all_attn_layers[target_name].get_kv_cache_spec(
                self.vllm_config
            )
            if draft_spec != target_spec:
                raise ValueError(
                    "VLLM_SELF_SPEC_SHARED_KV: KV cache spec mismatch: "
                    f"{draft_name!r} {draft_spec} vs {target_name!r} "
                    f"{target_spec}"
                )
            mapping[draft_name] = target_name
        self._shared_kv_target_layer = mapping
        self.runner.shared_kv_cache_layers.update(mapping)
        logger.info(
            "Self-spec shared-KV drafter: %d draft attention layers bind to "
            "the target layers' KV cache (no draft-side KV allocation).",
            len(mapping),
        )

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        # Filter to only layers that have KV cache specs.
        self._draft_attn_layer_names = {
            name
            for name in (set(all_attn_layers.keys()) - target_attn_layer_names)
            if all_attn_layers[name].get_kv_cache_spec(self.vllm_config) is not None
        }

        if self._shared_kv:
            self._register_shared_kv_layers(all_attn_layers, target_attn_layer_names)

        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # handle multimodality
            assert hasattr(target_model, "config")
            if self.get_model_name(target_model) in [
                "Cohere2VisionForConditionalGeneration",
                "Exaone4_5_ForConditionalGeneration",
                "GlmOcrForConditionalGeneration",
                "HunYuanVLForConditionalGeneration",
                "InternS2PreviewForConditionalGeneration",
                "MiMoV2OmniForCausalLM",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Gemma4ForConditionalGeneration",
                "Gemma4UnifiedForConditionalGeneration",
                "Step3p7ForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            elif self.get_model_name(target_model) == "KimiK25ForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.media_placeholder_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()
        else:
            target_language_model = target_model

        self._maybe_share_embeddings(target_language_model)
        self._maybe_share_lm_head(target_language_model)
        self._maybe_vres_slice()

        if (
            self.parallel_drafting
            and self.pass_hidden_states_to_model
            and self.parallel_drafting_hidden_state_tensor is not None
        ):
            flat_mask = self.model.mask_hidden.view(-1)
            if self.eagle3_use_aux_hidden_state:
                # EAGLE3: mask_hidden stores all aux hidden states,
                # project through combine_hidden_states
                self.parallel_drafting_hidden_state_tensor.copy_(
                    self.model.combine_hidden_states(flat_mask)
                )
            else:
                self.parallel_drafting_hidden_state_tensor.copy_(flat_mask)

    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own embedding layers, and some may
        have a duplicate copy of the target model's embedding layers. In these cases,
        we share the target model's embedding layers with the draft model to save
        memory.
        """
        if get_pp_group().world_size == 1:
            inner_model = getattr(target_language_model, "model", None)
            if inner_model is None:
                raise AttributeError("Target model does not have 'model' attribute")
            if hasattr(inner_model, "embed_tokens"):
                target_embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "embedding"):
                target_embed_tokens = inner_model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

    def _maybe_vres_slice(self) -> None:
        """Phase 85 (map-v6 queue): draft vocab restriction with a REAL
        lm_head slice. VLLM_SELF_SPEC_DRAFT_VOCAB_KEEP points at a saved
        LongTensor of kept token ids; the draft's lm_head weight is REBUILT
        (fresh Parameter -- the head may be shared with the target) to only
        those rows, shrinking the logits GEMM ~vocab/keep-fold; every draft
        argmax is remapped local->global via _vres_map (a gather: capture-
        safe inside FULL graphs). Greedy drafts only."""
        self._vres_map: torch.Tensor | None = None
        path = os.environ.get("VLLM_SELF_SPEC_DRAFT_VOCAB_KEEP", "").strip()
        if not path:
            return
        lm_head = getattr(self.model, "lm_head", None)
        assert lm_head is not None and hasattr(lm_head, "weight"), (
            "VLLM_SELF_SPEC_DRAFT_VOCAB_KEEP: draft has no lm_head")
        assert not self._enable_probabilistic_draft_probs, (
            "vres slice supports greedy drafts only")
        keep = torch.load(path, weights_only=False).to(
            lm_head.weight.device).long()
        new_w = nn.Parameter(
            lm_head.weight.data.index_select(0, keep).clone(),
            requires_grad=False)
        # The draft's lm_head may BE the target's module
        # (_maybe_share_lm_head fires when the checkpoints' heads are
        # byte-identical -- true for the W4 recipe, which skips lm_head).
        # NEVER mutate it: build a detached shallow copy with its own
        # parameter dict and hang the sliced weight there.
        import copy as _copy

        new_head = _copy.copy(lm_head)
        new_head._parameters = dict(lm_head._parameters)
        new_head._buffers = dict(lm_head._buffers)
        new_head.weight = new_w
        for attr in ("num_embeddings", "num_embeddings_padded",
                     "org_vocab_size"):
            if hasattr(new_head, attr):
                setattr(new_head, attr, keep.numel())
        self.model.lm_head = new_head
        self._vres_map = keep.to(torch.int64)
        logger.info(
            "[vres] draft lm_head sliced to %d rows (detached copy; "
            "target head untouched)", keep.numel(),
        )

    def _vres_remap(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map sliced-vocab argmax indices back to global token ids."""
        if self._vres_map is None:
            return token_ids
        if getattr(self, "_vres_dbg_n", 0) < 8 and os.environ.get(
                "W7_VRES_DEBUG"):
            self._vres_dbg_n = getattr(self, "_vres_dbg_n", 0) + 1
            loc = token_ids.flatten()[:8].tolist()
            glob = self._vres_map[token_ids].flatten()[:8].tolist()
            logger.info("[vres-dbg] shape=%s local=%s -> global=%s",
                        tuple(token_ids.shape), loc, glob)
        return self._vres_map[token_ids]

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own LM head, and some may have a
        duplicate copy of the target model's LM head. In these cases, we share
        the target model's LM head with the draft model to save memory.
        """
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and hasattr(target_language_model.lm_head, "weight")
                and hasattr(self.model.lm_head, "weight")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

            # MTP models call compute_logits via shared_head.head (a
            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
            # If the checkpoint omits a copy of the lm_head weights at the
            # MTP layer path, shared_head.head stays uninitialised and
            # produces NaN logits. Always share it explicitly.
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None) if inner else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )

        if hasattr(target_language_model.model, "topk_indices_buffer"):
            target_buffer = target_language_model.model.topk_indices_buffer
            if hasattr(self.model.model, "topk_indices_buffer"):
                del self.model.model.topk_indices_buffer
            self.model.model.topk_indices_buffer = target_buffer
            # Also share at per-module level so that the indexer and
            # sparse-attention backends in each MTP layer read from
            # the target model's buffer.
            for _, module in self.model.model.named_modules():
                if hasattr(module, "topk_indices_buffer"):
                    module.topk_indices_buffer = target_buffer
            logger.info(
                "Detected MTP model with topk_indices_buffer. "
                "Sharing target model topk_indices_buffer with the draft model."
            )

        # Detect index_share_for_mtp_iteration: when True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        self._share_mtp_indices = getattr(
            draft_hf_config, "index_share_for_mtp_iteration", False
        )

        if self.use_local_argmax_reduction:
            if not hasattr(self.model, "get_top_tokens"):
                raise ValueError(
                    "use_local_argmax_reduction is enabled but draft model "
                    f"{self.model.__class__.__name__} does not implement "
                    "get_top_tokens()."
                )
            logger.info(
                "Using local argmax reduction for draft token generation "
                "(communication: O(2*tp_size) vs O(vocab_size))."
            )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
        uniform_decode: bool = False,
    ) -> None:
        # Phase 55: a dummy run means this rank is NOT in a real propose --
        # disable the coordination memo so mixed steps stay rank-symmetric.
        self._in_propose = False
        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        only_one_forward_pass = is_graph_capturing or self.parallel_drafting
        for fwd_idx in range(
            1 if only_one_forward_pass else self.num_speculative_tokens
        ):
            if fwd_idx <= 1:
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens,
                        use_cudagraphs=use_cudagraphs,
                        uniform_decode=uniform_decode,
                    )
                )

            # Make sure to use EAGLE's own buffer during cudagraph capture.
            if (
                self._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(self._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            # W7: when capturing the draft's FULL (uniform-decode) cudagraph,
            # build REAL attention metadata from persistent buffers so the
            # captured graph records the real attention kernel instead of the
            # attn_metadata-is-None zero-fill stub (which would make every
            # captured decode step attend against garbage -> accept-length
            # collapse). Replay reads the same persistent buffers, so the
            # captured pointers stay live. Default (flag off) keeps None.
            capture_full_attn = (
                self.use_full_cudagraphs
                and uniform_decode
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and self.runner is not None
                and self._draft_attn_layer_names
                # W7-gqa: FA3 draft chains run attention eagerly and never replay
                # the captured decode graph -- skip building the real capture
                # metadata (and the sample-in-graph capture below) for them.
                and not self._draft_chain_force_eager_attn
            )
            _capture_qlen = self._step0_q if self._step0_full_cg else 1
            draft_attn_metadata = (
                self._build_draft_decode_capture_metadata(
                    num_input_tokens, qlen=_capture_qlen
                )
                if capture_full_attn
                else None
            )
            # Phase 55: keep idle-rank dummy REPLAYS of the step-0 FULL graph
            # inert -- PAD slots (no KV writes) and zero seq_lens (no reads).
            # Real propose() refreshes both buffers before its replay.
            if (
                self._step0_full_cg
                and capture_full_attn
                and not is_graph_capturing
            ):
                self._slot_mapping_buffer[:num_input_tokens].fill_(
                    PADDING_SLOT_ID
                )
                assert self._step0_seq_lens is not None
                self._step0_seq_lens.zero_()

            # Phase 69: carry the window-scratchpad ctx on the FULL-CG CAPTURE
            # (and idle-rank replays) so the captured draft forward records the
            # dense scratchpad attention kernels. Dummy window buffers are
            # in-range + all-valid (no OOB / NaN); the real propose refreshes
            # them before its replay.
            dummy_add_kwargs = self._draft_forward_additional_kwargs
            if (
                self._draft_fullcg
                and self._kv_window > 0
                and capture_full_attn
            ):
                self._populate_dummy_scratchpad(num_input_tokens)
                dummy_add_kwargs = dict(self._draft_forward_additional_kwargs or {})
                dummy_add_kwargs[SELF_SPEC_DRAFT_SCRATCHPAD_KEY] = (
                    self._ensure_scratchpad_ctx(num_input_tokens)
                )

            with set_forward_context(
                draft_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=self._last_batch_desc,
                slot_mapping=slot_mapping_dict,
                additional_kwargs=dummy_add_kwargs,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                kwargs = dict(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=inputs_embeds,
                )
                if self.pass_hidden_states_to_model:
                    kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
                self._tag_a2a(f"dummy{fwd_idx}")
                self.model(**kwargs)
                # W7 sampling-in-graph: on the draft FULL (uniform-decode)
                # capture pass, also capture the combined forward+sample graph
                # so its per-step replay in the decode loop finds the key.
                # Phase 55 (STEP0_FULL_CG, K=1): skip it entirely -- there is
                # no chain, so the fws graph is never replayed by propose();
                # and replaying TWO graphs in runtime idle-rank dummies while
                # busy ranks replay ONE desyncs the in-graph collectives of
                # the node-local draft across DP ranks (device deadlock at
                # batch drain).
                # Phase 55 (K>=2 dummy/chain replay symmetry): capture pass
                # ONLY. A runtime idle-rank dummy must replay exactly ONE
                # graph per draft forward, mirroring the busy propose (one
                # graph per chain step -- fws or model, same in-graph
                # node-collective sequence). Replaying model+fws here issued
                # a second collective sequence with no partner on busy ranks
                # -> device deadlock at batch drain (same class as the K=1
                # incident above, resurfacing the first time the chain ran
                # with real node-local collectives).
                _run_fws = (
                    capture_full_attn
                    and self._decode_fwd_sample is not None
                    and not self._step0_full_cg
                    and is_graph_capturing
                )
                if _run_fws:
                    self._decode_fwd_sample(**kwargs)
            self._tag_a2a("")
            if self._step0_debug:
                logger.info(
                    "[draft-replay] rank=%d src=dummy fwd=%d mode=%s graph=%s "
                    "ntok=%d capturing=%s",
                    self.dp_rank,
                    fwd_idx,
                    cudagraph_runtime_mode,
                    "model+fws" if _run_fws else "model",
                    num_input_tokens,
                    is_graph_capturing,
                )
            # Phase 55: record the step-0 FULL shapes actually captured;
            # propose() only claims uniform for these (see pre-flight check).
            if self._step0_full_cg and capture_full_attn and is_graph_capturing:
                self._step0_captured.add(num_input_tokens)
            # Phase 55 (K>=2): record the chain FULL shapes actually captured
            # (model + fws graphs); the chain pre-flight in propose() only
            # claims uniform-decode FULL for these.
            if capture_full_attn and is_graph_capturing:
                self._full_captured.add(num_input_tokens)

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        Some eagle3 heads (e.g., nvidia/gpt-oss-120b-Eagle3-v2) do not use auxiliary
        hidden states and directly uses the last layer output just like eagle1.
        They might indicate this by setting "use_aux_hidden_state" to False
        inside the "eagle_config" dict of their hf_config.
        """
        if self.method != "eagle3":
            return False
        # Assume that eagle3 heads use aux hidden states by default
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all drafting layers belong to the same KVCacheGroup.
        Need this assumption to ensure all drafting layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self._draft_attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """
        Initialize AttentionGroups for draft layers using kv_cache_config.
        Called from the model runner's initialize_metadata_builders.
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # Find which kv_cache_group the draft layers belong to
        self.validate_same_kv_cache_group(kv_cache_config)
        kv_cache_spec = None
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                self.kv_cache_gid = gid
                kv_cache_spec = group.kv_cache_spec
                break

        attention_groups: dict[tuple[str, str], AttentionGroup] = {}
        if kv_cache_spec is not None:
            for layer_name in self._draft_attn_layer_names:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                backend_key = attn_backend.full_cls_name()
                if backend_key not in attention_groups:
                    layer_kv_cache_spec = kv_cache_spec
                    if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                        # Shared-KV draft layers have no own spec entry; use
                        # the target twin's.
                        layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                            self._shared_kv_target_layer.get(layer_name, layer_name)
                        ]

                    kernel_block_size = (
                        kernel_block_sizes[self.kv_cache_gid]
                        if kernel_block_sizes is not None
                        and self.kv_cache_gid < len(kernel_block_sizes)
                        else None
                    )
                    attn_group = AttentionGroup(
                        backend=attn_backend,
                        layer_names=[layer_name],
                        kv_cache_spec=layer_kv_cache_spec,
                        kv_cache_group_id=self.kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    attention_groups[backend_key] = attn_group
                else:
                    attention_groups[backend_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self.block_size = (
            self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        )
        logger.debug("Using block size %d for drafting layers", self.block_size)

        # W7: with the draft FULL cudagraph (VLLM_SELF_SPEC_DRAFT_FULL_CG), cap
        # the MLA draft attention builder's FA3 split count to 1. The draft
        # decode steps attend over short, growing sequences, where the default
        # 32-way split-reduction produces INCORRECT results when captured into
        # the draft's FULL graph: the split-combine reads per-split partial
        # buffers whose layout is tied to the capture-time (padded, uniform)
        # schedule and does not replay correctly for the per-step growing
        # sequences (measured on V2-Lite: accept_len 2.85 -> 1.9 with 32 splits,
        # recovers to 2.78 with 1 split). One split = no combine = exact
        # attention. An env override is kept for experimentation.
        #
        # W7-gqa: this MLA cap does NOT fix FA3 (GQA). On FA3 the captured decode
        # kernel freezes its host-side work distribution at capture-time
        # seq_len=1 regardless of num_splits / scheduler_metadata (measured
        # identical accept ~1.99 for splits 0/1/2/32 and with/without a refreshed
        # AOT schedule). So instead of a split cap, FA3 draft chains run their
        # attention EAGERLY (see _draft_chain_force_eager_attn) -- the proven
        # correct path (accept restored to the PIECEWISE reference). Detect MLA
        # vs FA3 by builder type and apply the right remedy per backend.
        if self.use_full_cudagraphs:
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadataBuilder,
            )

            draft_splits = int(os.environ.get("W7_FG2_DRAFT_SPLITS", "1"))
            force_eager_attn = False
            for attn_group in self.draft_attn_groups:
                builder = attn_group.get_metadata_builder()
                is_mla = isinstance(builder, MLACommonMetadataBuilder)
                # Phase 79 (MLA challenger): FLASH_ATTN_MLA is FA3-family --
                # its captured schedule freezes like FA3-GQA's (constant-
                # geometry law), NOT like FLASHMLA's. Measured: captured
                # chain accept 2.00 vs eager 5.89 (K=5 self-draft,
                # DeepSeek-V2-Lite b32/16k). Treat it as non-replay-safe.
                is_fa3_mla = type(builder).__name__ == "FlashAttnMLAMetadataBuilder"
                if is_mla and not is_fa3_mla and getattr(builder, "max_num_splits", 0):
                    builder.max_num_splits = draft_splits
                elif (not is_mla or is_fa3_mla) and not self._draft_fullcg:
                    # FA3 / GQA (or any non-MLA backend, or FA3-family MLA):
                    # the captured decode graph cannot replay the draft's
                    # intra-loop growing sequence -> run the chain attention
                    # eagerly.
                    # Phase 69: with the window-scratchpad attention
                    # (VLLM_SELF_SPEC_DRAFT_FULLCG) the chain attention is a
                    # fixed-shape dense op that IS replay-safe, so keep the FULL
                    # graph (force_eager_attn stays False).
                    force_eager_attn = True
            # Env override for A/B (1 -> force eager, 0 -> force captured graph).
            _env = os.environ.get("W7_GQA_FORCE_EAGER_ATTN")
            if _env is not None:
                force_eager_attn = bool(int(_env))
            self._draft_chain_force_eager_attn = force_eager_attn
            if force_eager_attn:
                logger.info(
                    "Draft FULL-CG: running the draft chain attention eagerly "
                    "(non-MLA backend); the captured decode graph is not "
                    "replay-safe for the draft's growing sequence."
                )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        uniform_decode: bool = False,
        piecewise_only: bool = False,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        # FULL draft graphs only match keys for uniform-decode batches (the
        # K-chain decode-step forwards). The step-0 forward stays PIECEWISE.
        uniform_decode = uniform_decode and self.use_full_cudagraphs
        # W7-piecewise: for the FA3 (GQA) draft chain we want the model BODY on
        # captured PIECEWISE graph pieces while attention stays eager (a
        # splitting op, so PIECEWISE runs it live -> identical numerics to the
        # NONE chain). Constrain the dispatch to PIECEWISE (+NONE fallback) so it
        # never keys the FULL decode graph (not replay-safe for the growing
        # sequence). uniform_decode is forced off so the plain batch descriptor
        # matches the relaxed PIECEWISE key.
        if piecewise_only:
            uniform_decode = False
        # FULL cudagraphs require the batch_descriptor in the forward context so
        # the FULL CUDAGraphWrapper can key its capture/replay. Stash the
        # dispatched descriptor for the caller to forward (only used for FULL;
        # harmless otherwise).
        self._last_batch_desc = None
        if not use_cudagraphs:
            first_valid_modes = {CUDAGraphMode.NONE}
        elif piecewise_only:
            first_valid_modes = {CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE}
        else:
            first_valid_modes = None
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            uniform_decode=uniform_decode,
            valid_modes=first_valid_modes,
        )
        num_tokens_padded = batch_desc.num_tokens

        # Extra coordination when running data-parallel since we need to
        # coordinate across ranks
        # TODO(Flechman): support DBO ubatching
        should_ubatch, num_tokens_across_dp = False, None
        if (
            (self._consume_mode or envs.VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD)
            and self._draft_forward_additional_kwargs is not None
            and self.vllm_config.parallel_config.data_parallel_size > 1
        ):
            # OV1(b): the comm-free draft issues NO collectives, so per-rank
            # dispatch divergence is safe -- and MANDATORY: in consume mode,
            # ranks independently decide consume-vs-propose (bootstrap/fence
            # timing differs per DP engine), so a DP-coordination collective
            # here deadlocks the ranks that propose against those that don't.
            # W7 (phase 52): with DRAFT_SKIP_DP_COORD the same skip applies to
            # the LOCKSTEP comm-free draft: coordinate_batch_across_dp's
            # all_reduce + .item() is a per-chain-step DP-wide rendezvous
            # (measured 66% of the 2-node spec cycle) that buys nothing the
            # draft needs -- it has no collectives to keep consistent.
            # Build a locally-uniform num_tokens_across_dp for the forward
            # context (only comm paths read it; the draft has none).
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            num_tokens_across_dp = torch.full(
                (dp_size,), num_tokens_padded, dtype=torch.int32, device="cpu"
            )
        elif self.vllm_config.parallel_config.data_parallel_size > 1:
            # Phase 55: amortize the per-forward DP rendezvous to ONE
            # coordination per distinct shape per spec cycle. Lockstep ranks
            # produce identical key sequences, so the collective still runs
            # symmetrically on first occurrence; dummy runs and capture
            # (not inside propose) always coordinate.
            _memo_key = None
            if envs.VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD and self._in_propose:
                _memo_key = (
                    num_tokens,
                    num_tokens_padded,
                    cudagraph_mode.value,
                    uniform_decode,
                )
            if _memo_key is not None and _memo_key in self._dp_coord_memo:
                should_ubatch, _ntad, synced_cudagraph_mode = (
                    self._dp_coord_memo[_memo_key]
                )
                num_tokens_across_dp = (
                    _ntad.clone() if _ntad is not None else None
                )
            else:
                should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                    coordinate_batch_across_dp(
                        num_tokens_unpadded=num_tokens,
                        parallel_config=self.vllm_config.parallel_config,
                        allow_microbatching=False,
                        num_tokens_padded=num_tokens_padded,
                        cudagraph_mode=cudagraph_mode.value,
                        # Phase 65: on the CPU group the .item() readbacks
                        # below never sync the GPU stream (the NCCL path's
                        # chain coordination drains the whole step-0 draft
                        # forward). Same result, no stream drain.
                        force_cpu_group=(
                            envs.VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU
                        ),
                    )
                )
                if _memo_key is not None:
                    self._dp_coord_memo[_memo_key] = (
                        should_ubatch,
                        num_tokens_across_dp.clone()
                        if num_tokens_across_dp is not None
                        else None,
                        synced_cudagraph_mode,
                    )
            assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct
                # batch_descriptor.
                # Phase 55: the DP-agreed padded size is final; keep the
                # uniform claim only when the synced mode is FULL (all ranks
                # uniform, sizes multiples of q) so the dispatcher's uniform
                # bump cannot alter the agreed size and trip the assert.
                _sync_uniform = (
                    uniform_decode
                    and synced_cudagraph_mode == CUDAGraphMode.FULL.value
                )
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    uniform_decode=_sync_uniform,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct
                # otherwise num_tokens_across_dp will no-longer be valid
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        # FULL needs the exact (uniform, num_reqs) descriptor for keying. For the
        # PIECEWISE chain forward (piecewise_only) stash the resolved (relaxed)
        # descriptor too, so the caller's set_forward_context keys the captured
        # body graph directly. Other PIECEWISE callers leave it None (their
        # forward context derives BatchDescriptor(num_tokens), which is the same
        # relaxed key) so the default path is untouched.
        if cudagraph_mode == CUDAGraphMode.FULL or (
            piecewise_only and cudagraph_mode == CUDAGraphMode.PIECEWISE
        ):
            self._last_batch_desc = batch_desc
        if self._step0_debug:
            logger.info(
                "[draft-dbg] rank=%d src=%s ntok=%d padded=%d mode=%s "
                "uniform=%s desc=%s",
                self.dp_rank,
                "propose" if self._in_propose else "dummy",
                num_tokens, num_tokens_padded, cudagraph_mode,
                uniform_decode, batch_desc,
            )
        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    use_fp64_gumbel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = sample_with_exponential_noise(probs.clone(), q)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs
