# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.forward_context import (
    get_forward_context,
    self_spec_local_route_enabled,
    self_spec_node_local_enabled,
)
from vllm.logger import init_logger
from vllm.utils.flashinfer import (
    has_flashinfer_nvlink_one_sided,
    has_flashinfer_nvlink_two_sided,
)
from vllm.utils.import_utils import has_deep_ep, has_deep_ep_v2, has_mori

from .base_device_communicator import All2AllManagerBase, Cache

if has_flashinfer_nvlink_two_sided():
    from flashinfer.comm import Mapping  # type: ignore[import-not-found]
    from flashinfer.comm.mnnvl import MnnvlConfig  # type: ignore[import-not-found]
    from flashinfer.comm.trtllm_alltoall import (
        MnnvlMoe,  # type: ignore[import-not-found]
    )

if has_flashinfer_nvlink_one_sided():
    from flashinfer.comm import Mapping  # type: ignore[import-not-found]
    from flashinfer.comm.mnnvl import MnnvlConfig  # type: ignore[import-not-found]
    from flashinfer.comm.trtllm_moe_alltoall import (
        MoeAlltoAll,  # type: ignore[import-not-found]
        moe_a2a_get_workspace_size_per_rank,
    )


logger = init_logger(__name__)


class AgRsAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)
        self._emulated_a2a_count = 0
        self._active_emulated_a2a_count = 0
        # Number of REAL cross-rank collectives issued (all_gatherv /
        # reduce_scatterv). Stays 0 under the comm-free local-routing path,
        # which is how we verify the draft step issues no all-to-all.
        self._real_collective_count = 0
        # Number of hook calls SHIELDED from the emulated A2A delay because the
        # comm-free local-routing path was active (no real collective issued ->
        # no delay charged). Stays 0 on default paths and on the verify.
        self._shielded_a2a_count = 0
        # A/B revert knob (Phase 48): charge the emulated delay on the
        # comm-free local-route path anyway, reproducing the pre-shielding
        # (conservative) Phase 42/47 curves on the same binary.
        self._charge_draft_a2a = bool(
            int(os.environ.get("W7_CHARGE_DRAFT_A2A", "0"))
        )
        # Self-spec Phase 54 (node-local draft): cached positions of the
        # intra-node subgroup's members within the DP group, for slicing the
        # DP-shaped sizes vector down to the node subgroup.
        self._node_dp_indices: list[int] | None = None
        self._node_sizes_last: list[int] | None = None
        # GPU-clock cycles per microsecond for torch.cuda._sleep injection.
        # Calibrated once here (eager, before any CUDA-graph capture) so the
        # delay can be enqueued on the stream and captured into the graph.
        self._cycles_per_us = 0.0
        if envs.VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US > 0 and torch.cuda.is_available():
            try:
                self._cycles_per_us = self._calibrate_cycles_per_us()
                logger.info(
                    "Self-spec A2A delay: %.1f us/collective on GPU stream "
                    "(%.1f cycles/us)",
                    envs.VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US,
                    self._cycles_per_us,
                )
            except Exception as e:  # pragma: no cover - research hook
                logger.warning("Self-spec A2A delay calibration failed: %s", e)

    @staticmethod
    def _calibrate_cycles_per_us() -> float:
        torch.cuda._sleep(1_000_000)  # warmup
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        n = 200_000_000
        start.record()
        torch.cuda._sleep(n)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        return n / (ms * 1000.0) if ms > 0 else 0.0

    def _emulate_exposed_a2a_delay(self) -> None:
        self._emulated_a2a_count += 1
        active_file = envs.VLLM_SELF_SPEC_A2A_COUNT_ACTIVE_FILE
        if not active_file or Path(active_file).exists():
            self._active_emulated_a2a_count += 1
        if (
            self_spec_local_route_enabled() or self_spec_node_local_enabled()
        ) and not self._charge_draft_a2a:
            # (Node-local draft: shielded too -- the emulated delay models the
            # INTER-node hop, which node-local genuinely avoids; its real
            # intra-node collective cost is paid natively above.)
            # Comm-free local-routing forward (the self-spec draft): the
            # dispatch/combine above issued NO real collective, so charge no
            # emulated A2A delay either. Without this gate the draft pays the
            # sleep for communication it never performs, understating the win
            # (the Phase 42/47 curves were this conservative lower bound). The
            # branch resolves at CUDA-graph capture time per model -- the draft
            # and verify capture separate graphs, so the draft's graphs contain
            # no sleep while the verify's keep it.
            self._shielded_a2a_count += 1
            return
        delay_us = envs.VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US
        if delay_us > 0 and self._cycles_per_us > 0:
            # Enqueue on the current stream; captured into the CUDA graph so it
            # replays every decode step (host time.sleep would not).
            torch.cuda._sleep(int(delay_us * self._cycles_per_us))
        delay_ms = envs.VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    @staticmethod
    def _skip_gatherv(tensors, sizes, rank):
        """Shape-preserving local stand-in for all_gatherv (no cross-rank comm).

        Research-only, TIMING ONLY (dummy weights): tiles this rank's local chunk
        to fill the gathered shape, so values are valid (no out-of-range expert
        ids -> no OOB in the MoE kernel) and the routing spread is realistic, but
        the data is not the true cross-rank gather -- outputs are wrong by design.
        """
        total = sum(sizes)
        out = []
        for t in tensors:
            n = t.shape[0]
            if n == 0:
                out.append(t.new_zeros((total, *t.shape[1:])))
                continue
            reps = (total + n - 1) // n
            g = t.repeat((reps,) + (1,) * (t.dim() - 1))[:total]
            out.append(g.contiguous())
        return out

    @staticmethod
    def _skip_reduce_scatterv(t, sizes, rank):
        """Shape-preserving local stand-in for reduce_scatterv (no comm)."""
        off = sum(sizes[:rank])
        return t[off : off + sizes[rank]].contiguous()

    def _node_group_and_sizes(self, sizes, own_len):
        """Intra-node DP subgroup + node-sliced sizes (self-spec Phase 54).

        The draft's forward context can carry stale dp_metadata in
        prefill-adjacent steps (the comm-free path never consumed it, so this
        was invisible before). When the sliced sizes disagree with the actual
        input length, fall back to locally-uniform sizes -- correct for the
        uniform lockstep batches this research path targets. The result is
        stashed so combine() reuses the exact gather sizes.
        """
        from vllm.distributed.parallel_state import get_dp_node_group

        node_group = get_dp_node_group()
        if self._node_dp_indices is None:
            dp_ranks = get_dp_group().ranks
            self._node_dp_indices = [
                dp_ranks.index(r) for r in node_group.ranks
            ]
        node_sizes = [sizes[i] for i in self._node_dp_indices]
        if node_sizes[node_group.rank_in_group] != own_len:
            node_sizes = [own_len] * len(node_group.ranks)
        self._node_sizes_last = node_sizes
        return node_group, node_sizes

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
        extra_tensors_per_rank: list[bool] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        gathered_tensors = self._dispatch_gather(
            [hidden_states, router_logits],
            is_sequence_parallel,
            extra_tensors,
            extra_tensors_per_rank,
        )
        if extra_tensors is not None:
            return (gathered_tensors[0], gathered_tensors[1], gathered_tensors[2:])
        return gathered_tensors[0], gathered_tensors[1]

    def _dispatch_gather(
        self,
        head_tensors: list[torch.Tensor],
        is_sequence_parallel: bool,
        extra_tensors: list[torch.Tensor] | None,
        extra_tensors_per_rank: list[bool] | None,
    ) -> list[torch.Tensor]:
        """Shared gather for dispatch/dispatch_router_logits.

        head_tensors[0] is hidden_states; all head tensors and unmarked
        extras are token-major and gather with the per-rank token sizes.
        Phase 55: extras marked in extra_tensors_per_rank are per-RANK
        tensors (per-tensor quant scales, shape (1,)); they gather with one
        row per rank instead -- with uniform token sizes this produces the
        exact same (group_world,) tensor that used to fall out of the plain
        all_gather, and with NON-uniform sizes (mixed prefill/decode steps)
        it no longer trips the `1 != num_tokens` size assert. The marking is
        quant-config-driven on every rank, so the collective sequence stays
        rank-symmetric (the DEADLOCK class to avoid here).
        """
        hidden_states = head_tensors[0]
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        num_head = len(head_tensors)
        tensors_to_gather = list(head_tensors)
        per_rank_tensors: list[torch.Tensor] = []
        per_rank_positions: list[int] = []
        gathered_per_rank: list[torch.Tensor] = []
        if extra_tensors is not None:
            for i, t in enumerate(extra_tensors):
                if extra_tensors_per_rank is not None and extra_tensors_per_rank[i]:
                    per_rank_positions.append(num_head + i)
                    per_rank_tensors.append(t)
                else:
                    tensors_to_gather.append(t)

        if os.environ.get("W7_A2A_DEBUG") and (
            os.environ.get("W7_A2A_DEBUG_ALL") or any(s != sizes[0] for s in sizes)
        ):
            fc = get_forward_context()
            logger.info(
                "[a2a-dbg] dispatch node_local=%s local_route=%s sp=%s "
                "sizes=%s shapes=%s per_rank=%s kwargs=%s",
                self_spec_node_local_enabled(),
                self_spec_local_route_enabled(),
                is_sequence_parallel,
                sizes,
                [tuple(t.shape) for t in tensors_to_gather],
                [tuple(t.shape) for t in per_rank_tensors],
                sorted(fc.additional_kwargs.keys()),
            )

        if self_spec_node_local_enabled() and not is_sequence_parallel:
            # Self-spec Phase 54: node-local draft -- gather over the
            # intra-node subgroup only (NVLink); router already masked to
            # node-resident experts upstream.
            node_group, node_sizes = self._node_group_and_sizes(
                sizes, hidden_states.shape[0]
            )
            gathered_tensors = node_group.all_gatherv(
                tensors_to_gather, dim=0, sizes=node_sizes
            )
            if per_rank_tensors:
                gathered_per_rank = node_group.all_gatherv(
                    per_rank_tensors,
                    dim=0,
                    sizes=[1] * len(node_group.ranks),
                )
            self._real_collective_count += 1
        elif self_spec_local_route_enabled():
            # CORRECT comm-free local routing: no gather, use local tokens
            # as-is (router already masked to resident experts upstream).
            gathered_tensors = tensors_to_gather
            gathered_per_rank = per_rank_tensors
        elif envs.VLLM_SELF_SPEC_SKIP_A2A:
            gathered_tensors = self._skip_gatherv(
                tensors_to_gather, sizes, dist_group.rank_in_group
            )
            gathered_per_rank = self._skip_gatherv(
                per_rank_tensors,
                [1] * dist_group.world_size,
                dist_group.rank_in_group,
            )
        else:
            gathered_tensors = dist_group.all_gatherv(
                tensors_to_gather,
                dim=0,
                sizes=sizes,
            )
            if per_rank_tensors:
                gathered_per_rank = dist_group.all_gatherv(
                    per_rank_tensors,
                    dim=0,
                    sizes=[1] * dist_group.world_size,
                )
            self._real_collective_count += 1
        self._emulate_exposed_a2a_delay()

        # Re-interleave the per-rank extras at their original positions.
        for pos, g in zip(per_rank_positions, gathered_per_rank):
            gathered_tensors.insert(pos, g)
        return gathered_tensors

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
        extra_tensors_per_rank: list[bool] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        gathered_tensors = self._dispatch_gather(
            [hidden_states, topk_weights, topk_ids],
            is_sequence_parallel,
            extra_tensors,
            extra_tensors_per_rank,
        )

        hidden_states = gathered_tensors[0]
        topk_weights = gathered_tensors[1]
        topk_ids = gathered_tensors[2]

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids

        return hidden_states, topk_weights, topk_ids, gathered_tensors[3:]

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Reduce-scatter hidden_states across all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None

        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        if os.environ.get("W7_A2A_DEBUG") and (
            os.environ.get("W7_A2A_DEBUG_ALL") or any(s != sizes[0] for s in sizes)
        ):
            logger.info(
                "[a2a-dbg] combine node_local=%s local_route=%s sp=%s "
                "sizes=%s shape=%s",
                self_spec_node_local_enabled(),
                self_spec_local_route_enabled(),
                is_sequence_parallel,
                sizes,
                tuple(hidden_states.shape),
            )
        if self_spec_node_local_enabled() and not is_sequence_parallel:
            # Self-spec Phase 54: node-local draft -- reduce-scatter over the
            # intra-node subgroup only (NVLink).
            from vllm.distributed.parallel_state import get_dp_node_group

            node_group = get_dp_node_group()
            assert self._node_sizes_last is not None
            node_sizes = self._node_sizes_last
            hidden_states = node_group.reduce_scatterv(
                hidden_states, dim=0, sizes=node_sizes
            )
            self._real_collective_count += 1
        elif self_spec_local_route_enabled():
            # CORRECT comm-free local routing: no combine, output is already
            # local (each rank computed only its resident experts on its own
            # tokens, so there is nothing to reduce-scatter across ranks).
            pass
        elif envs.VLLM_SELF_SPEC_SKIP_A2A:
            hidden_states = self._skip_reduce_scatterv(
                hidden_states, sizes, dist_group.rank_in_group
            )
        else:
            hidden_states = dist_group.reduce_scatterv(
                hidden_states, dim=0, sizes=sizes
            )
            self._real_collective_count += 1
        self._emulate_exposed_a2a_delay()
        return hidden_states

    def destroy(self):
        if envs.VLLM_SELF_SPEC_LOG_A2A_COUNTS:
            logger.info(
                "Self-spec AgRs all2all count: total=%d active=%d real=%d "
                "shielded=%d",
                self._emulated_a2a_count,
                self._active_emulated_a2a_count,
                self._real_collective_count,
                self._shielded_a2a_count,
            )


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_deep_ep(), (
            "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md"
            " to install DeepEP kernels."
        )  # noqa
        super().__init__(cpu_group, tcp_store_group)
        self.handle_cache = Cache()

        # This is the DeepEP default. Stick to it till we can establish
        # reasonable defaults based on profiling.
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
            num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None
        # TODO: remove platform-specific logic
        # once ROCm DeepEP is updated with the latest APIs.
        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank,
            explicitly_destroy=True,
        )
        return kwargs

    def get_handle(self, kwargs):
        assert len(kwargs) == 0, (
            "DeepEPHTAll2AllManager expects no arguments. All the required "
            "args are computed in the Manager itself."
        )

        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs()
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    def set_num_sms(self, num_sms: int):
        import deep_ep  # type: ignore[import-not-found]

        # Right now the buffers are sized for only what the kernels were
        # created with. So we can only reduce the number of SMS used
        # but not increase it.
        if num_sms > self.num_sms:
            num_sms = self.num_sms
        deep_ep.Buffer.set_num_sms(num_sms)


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP Low-Latency kernels.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        """
        max_num_tokens_per_dp_rank : the maximum number of tokens a DP rank
          can dispatch all the ranks must hold the same value.
        token_hidden_size: the hidden dimension of each token.
        num_ep_ranks: the number of EP group ranks.
        num_global_experts: Number of experts in the model.
        num_local_experts: Number of experts in an EP rank.
        """
        import deep_ep  # type: ignore[import-not-found]

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts,
        )

        assert num_rdma_bytes is not None
        # TODO: remove platform-specific logic
        # once ROCm DeepEP is updated with the latest APIs.
        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
            explicitly_destroy=True,
        )
        return kwargs

    def get_handle(self, kwargs):
        """
        The kwargs for DeepEPLLAll2AllManager is dictated by
        _make_all2all_kwargs.
        """
        import deep_ep  # type: ignore[import-not-found]

        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    # DeepEP LL uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0


@dataclass
class _NixlEPBufferState:
    buffer: Any
    connected_ep_size: int
    active_ep_size: int


class NixlEPAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on NIXL EP kernels.
    This backend supports elastic EP with dynamic rank connection/disconnection.
    """

    _buffer: _NixlEPBufferState | None = None
    _lock = threading.RLock()

    def __init__(self, cpu_group, tcp_store_group=None):
        if tcp_store_group is None:
            tcp_store_group = StatelessProcessGroup(
                rank=cpu_group.rank(),
                world_size=cpu_group.size(),
                store=dist.PrefixStore("nixl_ep", cpu_group.get_group_store()),
            )
        super().__init__(cpu_group, tcp_store_group)

        self.max_num_ep_ranks = envs.VLLM_NIXL_EP_MAX_NUM_RANKS

    def _init_buffer(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_experts_per_rank: int,
    ) -> None:
        from nixl_ep import Buffer  # type: ignore[import-not-found]

        max_num_global_experts = self.max_num_ep_ranks * num_experts_per_rank
        num_rdma_bytes = Buffer.get_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=self.max_num_ep_ranks,
            num_experts=max_num_global_experts,
        )
        assert NixlEPAll2AllManager._buffer is None, (
            "NIXL EP buffer already initialized"
        )
        buffer = Buffer(
            rank=self.rank,
            tcp_store_group=self.tcp_store_group.store,
        )
        buffer.update_memory_buffers(
            num_ranks=self.max_num_ep_ranks,
            num_experts_per_rank=num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )
        ranks_to_connect = list(range(self.world_size))
        buffer.connect_ranks(ranks_to_connect)
        NixlEPAll2AllManager._buffer = _NixlEPBufferState(
            buffer=buffer,
            connected_ep_size=self.world_size,
            active_ep_size=self.world_size,
        )

    def _connect_to_ep_size(self, ep_size: int, *, make_active: bool) -> None:
        assert NixlEPAll2AllManager._buffer is not None
        state = NixlEPAll2AllManager._buffer
        if ep_size <= state.connected_ep_size:
            return

        state.buffer.set_tcp_store_group(self.tcp_store_group.store)
        ranks_to_connect = list(range(state.connected_ep_size, ep_size))
        state.buffer.connect_ranks(ranks_to_connect, activate=make_active)
        state.connected_ep_size = ep_size
        if make_active:
            state.active_ep_size = ep_size

    def _disconnect_to_ep_size(self, ep_size: int) -> None:
        assert NixlEPAll2AllManager._buffer is not None
        state = NixlEPAll2AllManager._buffer
        if ep_size >= state.connected_ep_size:
            return

        state.buffer.set_tcp_store_group(self.tcp_store_group.store)
        ranks_to_disconnect = list(range(ep_size, state.connected_ep_size))
        state.buffer.disconnect_ranks(ranks_to_disconnect)
        state.connected_ep_size = ep_size
        state.active_ep_size = min(state.active_ep_size, ep_size)

    def _unmask_connected_ranks(self, target_ep_size: int) -> None:
        assert NixlEPAll2AllManager._buffer is not None
        state = NixlEPAll2AllManager._buffer
        state.buffer.set_tcp_store_group(self.tcp_store_group.store)
        if target_ep_size <= state.active_ep_size:
            return
        assert state.connected_ep_size >= target_ep_size

        for rank in range(state.active_ep_size, target_ep_size):
            state.buffer.update_mask_buffer(rank, mask=False)
        state.active_ep_size = target_ep_size

    def _stage_ep_size(self) -> None:
        assert NixlEPAll2AllManager._buffer is not None
        state = NixlEPAll2AllManager._buffer
        target_ep_size = self.world_size

        # Scale-up can safely connect standby ranks while leaving them masked.
        # Scale-down must not disconnect active ranks until commit.
        if target_ep_size > state.connected_ep_size:
            self._connect_to_ep_size(target_ep_size, make_active=False)

    def commit_staged_state(self) -> None:
        """Commit staged NIXL EP state to the active communication set."""
        with NixlEPAll2AllManager._lock:
            assert NixlEPAll2AllManager._buffer is not None
            state = NixlEPAll2AllManager._buffer
            target_ep_size = self.world_size

            if target_ep_size < state.connected_ep_size:
                self._disconnect_to_ep_size(target_ep_size)
            elif target_ep_size > state.connected_ep_size:
                self._connect_to_ep_size(target_ep_size, make_active=True)

            self._unmask_connected_ranks(target_ep_size)

    def _ensure_ep_size(self, *, stage: bool) -> None:
        if stage:
            self._stage_ep_size()
        else:
            self.commit_staged_state()

    def get_handle(self, kwargs):
        with NixlEPAll2AllManager._lock:
            stage = bool(kwargs.get("stage", False))
            state = NixlEPAll2AllManager._buffer
            if state is None:
                assert not stage, (
                    "NIXL EP staged initialization requires an existing buffer"
                )
                max_num_tokens_per_dp_rank = kwargs["max_num_tokens_per_dp_rank"]
                num_experts_per_rank = (
                    kwargs["num_global_experts"] // kwargs["num_ep_ranks"]
                )
                self._init_buffer(
                    max_num_tokens_per_dp_rank=max_num_tokens_per_dp_rank,
                    token_hidden_size=kwargs["token_hidden_size"],
                    num_experts_per_rank=num_experts_per_rank,
                )
            else:
                self._ensure_ep_size(stage=stage)

            assert NixlEPAll2AllManager._buffer is not None
            handle = NixlEPAll2AllManager._buffer.buffer
            return handle

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        # NOTE(yongji): NIXLEPAll2AllManager instance is recreated during
        # scale-up/down, so we cannot destroy the persistent buffer here.
        assert NixlEPAll2AllManager._buffer is not None
        buffer = NixlEPAll2AllManager._buffer.buffer
        buffer.set_tcp_store_group(None)

    # NIXL EP uses RDMA so no SMs are used for communication
    def max_sms_used(self) -> int | None:
        return 0


class FlashInferNVLinkTwoSidedManager(All2AllManagerBase):
    """
    All2All communication based on flashinfer all2allv/two-sided NVLink kernels.
    """

    # This type lint could be removed after all of the work in
    # https://github.com/vllm-project/vllm/issues/26533 done.
    rank: int
    world_size: int

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_flashinfer_nvlink_two_sided(), (
            "flashinfer all2all module not found. Please install/check flashinfer"
        )  # noqa
        super().__init__(cpu_group, tcp_store_group)
        logger.debug(
            "Initialize for flashinfer All2All rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.initialized = False
        self.alltoall_info = None

    def initialize(
        self,
        world_size: int,
        rank: int,
        gpus_per_node: int,
    ):
        """Initialize workspace"""
        if self.initialized:
            return

        self.cleanup()
        logger.debug("making map: rank=%d, world size=%d", rank, world_size)
        self.mapping = Mapping(
            world_size,
            rank,
            gpus_per_node,
            tp_size=world_size,
        )

        from vllm.distributed.device_communicators.mnnvl_compat import (
            CustomCommunicator,
        )

        # MNNVL workspace is allocated per rank in the comm_backend's group; the
        # flashinfer kernel asserts workspace.size(0) == moe_ep_size, so the backend
        # must span the EP group (= DP*PCP*TP), not the DP group.
        ep_config = MnnvlConfig(
            comm_backend=CustomCommunicator(self.cpu_group),
            fabric_page_size=1 << 29,  # 512MB
            allocation_granularity=0,  # Auto-detect
        )

        self.workspace_tensor = MnnvlMoe.get_moe_workspaces(self.mapping, ep_config)
        self.prepare_workspace_tensor = MnnvlMoe.get_moe_prepare_workspace(
            self.mapping, ep_config
        )

        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.initialized = True

        logger.info(
            "FlashInfer All2All initialized for rank %s, size %s", rank, world_size
        )

    def ensure_alltoall_workspace_initialized(self):
        """Ensure workspace is initialized"""
        if not has_flashinfer_nvlink_two_sided():
            return False

        if self.world_size <= 1:
            return False

        if not self.initialized:
            self.initialize(
                world_size=self.world_size,
                rank=self.rank,
                gpus_per_node=torch.accelerator.device_count,
            )
        return self.initialized

    def get_handle(self, kwargs):
        return self

    def cleanup(self):
        """Clean up workspace"""
        if (
            self.initialized
            and self.workspace_tensor is not None
            and self.prepare_workspace_tensor is not None
        ):
            try:
                del self.workspace_tensor
                del self.prepare_workspace_tensor
            except Exception as e:
                logger.warning("Failed to cleanup FlashInfer workspace: %s", e)
            finally:
                self.workspace_tensor = None
                self.prepare_workspace_tensor = None
                self.mapping = None
                self.initialized = False


class FlashInferNVLinkOneSidedManager(All2AllManagerBase):
    """
    All2All communication based on FlashInfer's MoeAlltoAll/One-sided NVLink kernel.
    This is a newer kernel from trtllm that should perform better than the kernel
    used by flashinfer_nvlink_two_sided.
    """

    rank: int
    world_size: int

    def __init__(self, cpu_group):
        assert has_flashinfer_nvlink_one_sided(), (
            "flashinfer trtllm_moe_alltoall module not found. "
            "Please install/check flashinfer"
        )
        super().__init__(cpu_group)
        logger.debug(
            "Initialize FlashInfer One-sided NVLink rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.initialized = False
        self.moe_alltoall: MoeAlltoAll | None = None
        self.mapping = None
        self.workspace_size = 0
        self.max_num_tokens = 0
        self.top_k = 0
        self.num_experts = 0

    def initialize(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dispatch_dtype_bytes_per_elem: int = 0,
        dispatch_scale_bytes_per_token: int = 0,
    ):
        """Initialize (or grow) the MoeAlltoAll workspace."""
        if dispatch_dtype_bytes_per_elem == 0:
            hidden_bytes = hidden_size // 2
        else:
            hidden_bytes = hidden_size * dispatch_dtype_bytes_per_elem
        total_dispatch_payload_size_per_token = (
            hidden_bytes
            + dispatch_scale_bytes_per_token
            + top_k * 4  # int32 topks ids
            + top_k * 4  # float32 topk weights
        )
        combine_payload_size_per_token = hidden_size * 2  # bf16 hidden states
        needed_workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.world_size,
            max_num_tokens=max_num_tokens,
            total_dispatch_payload_size_per_token=total_dispatch_payload_size_per_token,
            combine_payload_size_per_token=combine_payload_size_per_token,
        )
        # workspace_size and max_num_tokens are kernel-side max-bounds, so
        # heterogeneous MoE layers (e.g. NVFP4 base + bf16 MTP head) only
        # need the shared workspace grown to the union. top_k and num_experts
        # must match across layers: top_k is a strict-equality assert at
        # dispatch (FlashInfer csrc/trtllm_moe_alltoall.cu), and num_experts
        # feeds the expert-to-rank routing math, so any mismatch would crash
        # or silently corrupt routing. All ranks see the same MoE layers in
        # the same order with identical shapes, so the skip / rebuild
        # branches are taken consistently across ranks.
        if self.initialized:
            assert top_k == self.top_k, (
                "FlashInfer one-sided MoeAlltoAll does not support "
                f"heterogeneous top_k across MoE layers (got {top_k}, "
                f"was built with {self.top_k})"
            )
            assert num_experts == self.num_experts, (
                "FlashInfer one-sided MoeAlltoAll does not support "
                f"heterogeneous num_experts across MoE layers (got "
                f"{num_experts}, was built with {self.num_experts})"
            )
            if (
                needed_workspace_size <= self.workspace_size
                and max_num_tokens <= self.max_num_tokens
            ):
                return

        self.workspace_size = max(self.workspace_size, needed_workspace_size)
        self.max_num_tokens = max(self.max_num_tokens, max_num_tokens)
        self.top_k = top_k
        self.num_experts = num_experts

        self.cleanup()
        from vllm.platforms.interface import get_assigned_physical_gpu_ids

        assigned_physical_gpu_ids = get_assigned_physical_gpu_ids()
        gpus_per_node = (
            len(assigned_physical_gpu_ids)
            if assigned_physical_gpu_ids is not None
            else torch.accelerator.device_count()
        )
        logger.debug(
            "Making One-sided NVLink mapping: rank=%d, world size=%d",
            self.rank,
            self.world_size,
        )
        self.mapping = Mapping(
            self.world_size,
            self.rank,
            gpus_per_node,
            tp_size=self.world_size,
            moe_ep_size=self.world_size,
        )

        from vllm.distributed.device_communicators.mnnvl_compat import (
            CustomCommunicator,
        )

        # MNNVL workspace is allocated per rank in the comm_backend's group; the
        # flashinfer kernel asserts workspace.size(0) == moe_ep_size, so the backend
        # must span the EP group (= DP*PCP*TP), not the DP group.
        ep_config = MnnvlConfig(
            comm_backend=CustomCommunicator(self.cpu_group),
        )

        self.moe_alltoall = MoeAlltoAll(
            mapping=self.mapping,
            max_num_tokens=self.max_num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
            workspace_size_per_rank=self.workspace_size,
            mnnvl_config=ep_config,
        )

        self.gpus_per_node = gpus_per_node
        self.initialized = True

        logger.info(
            "FlashInfer One-sided NVLink initialized for rank %s, size %s",
            self.rank,
            self.world_size,
        )
        # Scope barrier to the EP group: with PP, different EP groups can
        # rebuild a different number of times if their MoE layers have
        # different shape sequences, so a world-level barrier would deadlock.
        dist.barrier(group=self.cpu_group)

    def get_handle(self, kwargs):
        return self

    def cleanup(self):
        """Clean up resources."""
        if self.initialized and self.moe_alltoall is not None:
            try:
                del self.moe_alltoall
            except Exception as e:
                logger.warning(
                    "Failed to cleanup FlashInfer One-sided NVLink workspace: %s", e
                )
            finally:
                self.moe_alltoall = None
                self.mapping = None
                self.initialized = False


class MoriAll2AllManager(All2AllManagerBase):
    def __init__(self, cpu_group, all2all_backend: str):
        assert has_mori(), (
            "MoRI kernels not found. Please follow https://github.com/ROCm/mori/blob/main/README.md"
            " to install MoRI kernels."
        )  # noqa
        assert all2all_backend in (
            "mori_high_throughput",
            "mori_low_latency",
        ), f"unsupported MoRI all2all backend: {all2all_backend!r}"
        import mori

        super().__init__(cpu_group)
        self._all2all_backend = all2all_backend
        self.handle_cache = Cache()

        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        mori.shmem.shmem_torch_process_group_init("mori")

    def _make_all2all_kwargs(
        self,
        rank: int,
        num_ep_ranks: int,
        input_dtype: torch.dtype,
        quant_dtype: torch.dtype,
        token_hidden_size: int,
        scale_dim: int,
        scale_type_size: int,
        max_num_tokens_per_dp_rank: int,
        num_local_experts: int,
        num_experts_per_token: int,
    ):
        import mori  # type: ignore[import-not-found]

        from vllm.platforms.rocm import on_gfx942, on_gfx950

        assert on_gfx942() or on_gfx950(), (
            "mori currently only support arch gfx942 and gfx950"
        )

        if not self.internode:
            # single node
            kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
            rdma_block_num = 0
            warp_num_per_block = 16
            block_num = 80
        else:
            # Multi-node: kernel follows --all2all-backend (mirrors deepep_* split).
            # mori_low_latency → InterNodeV1LL; mori_high_throughput → V1.
            if self._all2all_backend == "mori_low_latency":
                kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
            else:
                kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
            if on_gfx942():
                warp_num_per_block = 16
                block_num = 32
                rdma_block_num = 16
            elif on_gfx950():
                warp_num_per_block = 8
                block_num = 64
                rdma_block_num = 32
            else:
                raise NotImplementedError(
                    "mori currently only support arch gfx942 and gfx950"
                )

        return dict(
            rank=rank,
            world_size=num_ep_ranks,
            data_type=quant_dtype,
            hidden_dim=token_hidden_size,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=input_dtype.itemsize,
            max_num_inp_token_per_rank=max_num_tokens_per_dp_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            kernel_type=kernel_type,
            rdma_block_num=rdma_block_num,
            gpu_per_node=min(8, num_ep_ranks),
        )

    def _make_handle(self, **kwargs):
        import mori  # type: ignore[import-not-found]

        mori_config = mori.ops.EpDispatchCombineConfig(**kwargs)
        handle = mori.ops.EpDispatchCombineOp(mori_config)
        return handle

    def get_handle(self, kwargs):
        import mori  # type: ignore[import-not-found]

        mori_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("MoRI all2all args %s", mori_kwargs)
        handle: mori.ops.EpDispatchCombineOp = self.handle_cache.get_or_create(
            mori_kwargs, self._make_handle
        )
        return handle


class DeepEPV2All2AllManager(All2AllManagerBase):
    """
    All2All communication based on DeepEP v2 ElasticBuffer (unified API).
    Uses NCCL Gin backend with analytical SM calculation.
    """

    def __init__(self, cpu_group, tcp_store_group=None, device_group=None):
        assert has_deep_ep_v2(), (
            "DeepEP v2 (ElasticBuffer) not available. Requires DeepEP >= 2.0 "
            "(https://github.com/deepseek-ai/DeepEP) and NCCL >= 2.30.4."
        )
        super().__init__(cpu_group, tcp_store_group)
        self._device_group = device_group
        self.handle_cache = Cache()
        self._num_sms: int | None = None

    def _make_all2all_kwargs(
        self,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        use_fp8_dispatch: bool,
    ) -> dict:
        return dict(
            group=self._device_group
            if self._device_group is not None
            else self.cpu_group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            allow_hybrid_mode=envs.VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE,
            prefer_overlap_with_compute=envs.VLLM_DEEPEP_V2_PREFER_OVERLAP,
            allow_multiple_reduction=(envs.VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION),
            explicitly_destroy=True,
        )

    def get_handle(self, kwargs):
        import deep_ep  # type: ignore[import-not-found]

        num_experts = kwargs.pop("num_experts", 256)
        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP v2 all2all args %s", buffer_kwargs)
        handle: deep_ep.ElasticBuffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.ElasticBuffer
        )
        if self._num_sms is None:
            self._num_sms = handle.get_theoretical_num_sms(
                num_experts=num_experts,
                num_topk=kwargs["num_topk"],
            )
        return handle

    def max_sms_used(self) -> int | None:
        return self._num_sms

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()
