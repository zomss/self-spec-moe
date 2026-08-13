# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict live contract for the Phase 97 minimal-B0 K4/OFF runtime."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OFF_ACTION_ID = "off"
K4_ACTION_ID = "target-matching-k4"
W512_ACTION_ID = "target-matching-w512-masked-k4"
ALLOWED_K_VALUES = frozenset({0, 4})
P4_MIN_SHARED_KV_BLOCKS = 21682
P4_CAPTURE_PLAN_CONTRACT_ID = "p4-b0-same-boot-capture-plan-v1"
P4_CAPTURE_COHORT_CONTRACT_ID = "p4-capture-cohort-barrier-v1"
P4_CAPTURE_COHORT_EXTRA_ARGS_KEY = "p4_capture_cohort_barrier"
P4_ACTION_REALIZATIONS = {
    OFF_ACTION_ID: "live-b0-forced-off",
    K4_ACTION_ID: "live-b0-target-matching-k4",
    W512_ACTION_ID: "boot-static-mask-equivalent-surrogate",
}
P4_ACTION_ORDERS = {
    1: (OFF_ACTION_ID, K4_ACTION_ID, W512_ACTION_ID),
    2: (K4_ACTION_ID, W512_ACTION_ID, OFF_ACTION_ID),
    3: (W512_ACTION_ID, OFF_ACTION_ID, K4_ACTION_ID),
}
P4_REGIME_ORDER = ("R4", "R5", "R5cot", "R8", "R1", "R6")
P4_CONTENT_SEEDS = (0, 1)
P4_ROUNDS = (1, 2, 3, 4)
_P4_LOGICAL_WEIGHT_VERSION = re.compile(r"^target-matching-config-sha256-[0-9a-f]{64}$")
_P4_INTERNAL_REQUEST_ID_SUFFIX = re.compile(r"^[0-9a-f]{8}$")


class KOffRuntimeError(RuntimeError):
    """Raised when the live K4/OFF contract cannot be proven."""


# Registered boot scopes. minimal-b0 is Phase 97's B0 screen and stays the
# default, so an unset environment behaves exactly as before. w98-lattice is
# Phase 98's selector demonstration, whose quant x window x skip lattice
# minimal-b0 forbids by construction. Adding a scope is deliberate: the
# contract stays fail-closed and an unknown scope is refused.
BOOT_SCOPE_MINIMAL_B0 = "minimal-b0"
BOOT_SCOPE_W98_LATTICE = "w98-lattice"
BOOT_SCOPES = frozenset({BOOT_SCOPE_MINIMAL_B0, BOOT_SCOPE_W98_LATTICE})
# The w98 scope relaxes exactly three axes, each bounded to the frozen lattice
# in research/98_selector_demo/data/prereg/w98_prereg_matrix.json.
W98_WINDOWS = frozenset({0, 128, 256, 512, 1024})
W98_SKIP_COUNTS = frozenset({0, 4, 8})
W98_WINDOW_SINKS = 16


_RUNTIME_OBSERVATION: dict[str, str] = {}


def observe_chain_runtime_mode(mode: str) -> None:
    """Record the cudagraph mode the draft chain actually dispatched.

    The proposer decides the chain's mode per forward. Captures previously
    carried only the runner's *declared* ``graph_grade``, so a silent fallback
    (PIECEWISE -> NONE on a dispatch miss) left no trace in the evidence.

    Args:
        mode: The dispatched ``CUDAGraphMode``, stringified.
    """
    _RUNTIME_OBSERVATION["chain_runtime_mode"] = str(mode)


def observed_hardware_id() -> str:
    """Return the UUID of the CUDA device this process is actually using.

    Returns:
        The live device UUID, or ``"unavailable"`` when CUDA cannot report it.
    """
    cached = _RUNTIME_OBSERVATION.get("observed_hardware_id")
    if cached is not None:
        return cached
    observed = "unavailable"
    try:
        from vllm.platforms import current_platform

        raw = current_platform.get_device_uuid(0)
        text = str(raw)
        observed = text if text.startswith("GPU-") else f"GPU-{text}"
    except Exception:  # noqa: BLE001 - observation must never break a capture
        observed = "unavailable"
    _RUNTIME_OBSERVATION["observed_hardware_id"] = observed
    return observed


def canonicalize_p4_request_ids(
    request_ids: Sequence[object],
    frozen_request_ids: Sequence[str],
) -> list[str]:
    """Map vLLM internal request IDs to the frozen P4 manifest IDs."""
    frozen_ids = tuple(frozen_request_ids)
    if (
        not frozen_ids
        or any(not isinstance(value, str) or not value for value in frozen_ids)
        or len(set(frozen_ids)) != len(frozen_ids)
    ):
        raise KOffRuntimeError("P4 frozen request IDs are empty or duplicated")

    frozen_set = set(frozen_ids)
    canonical_ids: list[str] = []
    internal_by_frozen: dict[str, str] = {}
    for request_id in request_ids:
        if not isinstance(request_id, str) or not request_id:
            raise KOffRuntimeError("P4 request ID is missing or is not a string")
        if request_id in frozen_set:
            frozen_id = request_id
        else:
            matches = [
                candidate
                for candidate in frozen_ids
                if request_id.startswith(f"{candidate}-")
                and _P4_INTERNAL_REQUEST_ID_SUFFIX.fullmatch(
                    request_id[len(candidate) + 1 :]
                )
            ]
            if len(matches) != 1:
                raise KOffRuntimeError(
                    "P4 request ID is neither frozen nor a frozen ID followed by "
                    f"exactly eight lowercase hex characters: {request_id!r}"
                )
            frozen_id = matches[0]

        prior_internal_id = internal_by_frozen.get(frozen_id)
        if prior_internal_id is not None:
            raise KOffRuntimeError(
                "P4 request IDs collide after canonicalization: "
                f"{prior_internal_id!r} and {request_id!r} map to {frozen_id!r}"
            )
        internal_by_frozen[frozen_id] = request_id
        canonical_ids.append(frozen_id)

    return canonical_ids


@dataclass(frozen=True)
class P4CohortRequestState:
    """Scheduler-visible state for one P4 capture-cohort request."""

    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    num_preemptions: int = 0


@dataclass(frozen=True)
class P4CaptureCohortSpec:
    """Validated measurement-only cohort metadata carried by a request."""

    cohort_id: str
    frozen_request_ids: tuple[str, ...]
    action_id: str
    measured_decode_tokens: int
    unmeasured_prefill_tokens: int


def parse_p4_capture_cohort_spec(
    extra_args: Mapping[str, Any] | None,
) -> P4CaptureCohortSpec | None:
    """Parse the explicit measurement-only capture-cohort marker.

    Args:
        extra_args: Request ``SamplingParams.extra_args`` payload.

    Returns:
        A validated cohort specification, or ``None`` when the marker is absent.

    Raises:
        KOffRuntimeError: If a present marker differs from the closed contract.
    """
    if extra_args is None or P4_CAPTURE_COHORT_EXTRA_ARGS_KEY not in extra_args:
        return None
    value = extra_args[P4_CAPTURE_COHORT_EXTRA_ARGS_KEY]
    if not isinstance(value, Mapping):
        raise KOffRuntimeError("P4 capture-cohort marker must be an object")
    expected_fields = {
        "contract_id",
        "measurement_only",
        "cohort_id",
        "frozen_request_ids",
        "action_id",
        "measured_decode_tokens",
        "unmeasured_prefill_tokens",
    }
    if set(value) != expected_fields:
        raise KOffRuntimeError(
            "P4 capture-cohort marker fields differ from the closed contract"
        )
    if (
        value["contract_id"] != P4_CAPTURE_COHORT_CONTRACT_ID
        or value["measurement_only"] is not True
    ):
        raise KOffRuntimeError(
            "P4 capture-cohort marker is not the measurement-only contract"
        )

    cohort_id = value["cohort_id"]
    frozen_request_ids = value["frozen_request_ids"]
    action_id = value["action_id"]
    measured_decode_tokens = value["measured_decode_tokens"]
    unmeasured_prefill_tokens = value["unmeasured_prefill_tokens"]
    if not isinstance(cohort_id, str) or not cohort_id:
        raise KOffRuntimeError("P4 capture-cohort id is empty")
    if not isinstance(frozen_request_ids, (list, tuple)):
        raise KOffRuntimeError("P4 capture-cohort members must be an ordered list")

    # Reuse the state machine constructor as the single source of validation
    # for identities, action, and token accounting.
    P4CaptureCohortBarrier(
        cohort_id,
        frozen_request_ids,
        action_id=action_id,
        measured_decode_tokens=measured_decode_tokens,
        unmeasured_prefill_tokens=unmeasured_prefill_tokens,
    )
    return P4CaptureCohortSpec(
        cohort_id=cohort_id,
        frozen_request_ids=tuple(frozen_request_ids),
        action_id=action_id,
        measured_decode_tokens=measured_decode_tokens,
        unmeasured_prefill_tokens=unmeasured_prefill_tokens,
    )


class P4CaptureCohortBarrier:
    """Hold completed prefills until one capture cohort can decode together."""

    _QUERY_WIDTH_BY_ACTION = {
        OFF_ACTION_ID: 1,
        K4_ACTION_ID: 5,
        W512_ACTION_ID: 5,
    }
    _DEADLOCK_STEP_LIMIT = 2

    def __init__(
        self,
        cohort_id: str,
        frozen_request_ids: Sequence[str],
        *,
        action_id: str,
        measured_decode_tokens: int,
        unmeasured_prefill_tokens: int = 1,
    ) -> None:
        """Create an unsealed measurement-only capture cohort.

        Args:
            cohort_id: Stable identifier for one microbatch.
            frozen_request_ids: Exact manifest IDs in queue order.
            action_id: Boot-static OFF, K4, or w512 action.
            measured_decode_tokens: Required commits per request.
            unmeasured_prefill_tokens: Sampled prefill tokens excluded from value.

        Raises:
            KOffRuntimeError: If the cohort contract is malformed.
        """
        frozen_ids = tuple(frozen_request_ids)
        if not isinstance(cohort_id, str) or not cohort_id:
            raise KOffRuntimeError("P4 capture cohort id is empty")
        if (
            not frozen_ids
            or any(not isinstance(value, str) or not value for value in frozen_ids)
            or len(set(frozen_ids)) != len(frozen_ids)
        ):
            raise KOffRuntimeError(
                "P4 capture cohort frozen request IDs are empty or duplicated"
            )
        if (
            not isinstance(action_id, str)
            or action_id not in self._QUERY_WIDTH_BY_ACTION
        ):
            raise KOffRuntimeError(
                f"P4 capture cohort has unknown action {action_id!r}"
            )
        if type(measured_decode_tokens) is not int or measured_decode_tokens <= 0:
            raise KOffRuntimeError(
                "P4 capture cohort measured decode work must be positive"
            )
        if unmeasured_prefill_tokens != 1:
            raise KOffRuntimeError(
                "P4 capture cohort requires exactly one unmeasured prefill token"
            )

        self.cohort_id = cohort_id
        self.frozen_request_ids = frozen_ids
        self.action_id = action_id
        self.measured_decode_tokens = measured_decode_tokens
        self.unmeasured_prefill_tokens = unmeasured_prefill_tokens
        self._state = "registering"
        self._abort_reason: str | None = None
        self._internal_by_frozen: dict[str, str] = {}
        self._frozen_by_internal: dict[str, str] = {}
        self._prompt_tokens_by_internal: dict[str, int] = {}
        self._committed_by_internal: dict[str, int] = {}
        self._finished_internal_ids: set[str] = set()
        self._held_internal_ids: set[str] = set()
        self._step_open = False
        self._step_mode = ""
        self._first_decode_observed = False
        self._prefill_step_count = 0
        self._decode_step_count = 0
        self._idle_step_count = 0

    @property
    def state(self) -> str:
        """Return the current fail-closed barrier state."""
        return self._state

    @property
    def abort_reason(self) -> str | None:
        """Return the first terminal failure reason, if any."""
        return self._abort_reason

    @property
    def abort_request_ids(self) -> tuple[str, ...]:
        """Return registered unfinished internal IDs in frozen queue order."""
        return tuple(
            self._internal_by_frozen[frozen_id]
            for frozen_id in self.frozen_request_ids
            if frozen_id in self._internal_by_frozen
            and self._internal_by_frozen[frozen_id] not in self._finished_internal_ids
        )

    def _fail(self, message: str) -> None:
        if self._abort_reason is None:
            self._abort_reason = message
        self._state = "aborted"
        self._step_open = False
        raise KOffRuntimeError(f"P4 capture cohort {self.cohort_id!r}: {message}")

    def register(self, request_id: str) -> str:
        """Register one runtime request in exact frozen queue order.

        Args:
            request_id: Internal vLLM request ID, with optional random suffix.

        Returns:
            The canonical frozen manifest ID.

        Raises:
            KOffRuntimeError: If identity, order, or lifecycle differs.
        """
        if self._state != "registering":
            self._fail("request registration occurred after cohort sealing")
        if len(self._internal_by_frozen) >= len(self.frozen_request_ids):
            self._fail("request registration exceeded the frozen cohort")
        try:
            frozen_id = canonicalize_p4_request_ids(
                [request_id], self.frozen_request_ids
            )[0]
        except KOffRuntimeError as exc:
            self._fail(str(exc))
        expected_id = self.frozen_request_ids[len(self._internal_by_frozen)]
        if frozen_id != expected_id:
            self._fail(
                "request registration order drifted: "
                f"expected {expected_id!r}, got {frozen_id!r}"
            )
        if (
            request_id in self._frozen_by_internal
            or frozen_id in self._internal_by_frozen
        ):
            self._fail("request registration repeated a cohort identity")
        self._internal_by_frozen[frozen_id] = request_id
        self._frozen_by_internal[request_id] = frozen_id
        self._committed_by_internal[request_id] = 0
        return frozen_id

    def seal(self) -> None:
        """Seal only after every exact member was queued before execution."""
        if self._state != "registering":
            self._fail("cohort sealing occurred outside registration")
        if tuple(self._internal_by_frozen) != self.frozen_request_ids:
            self._fail("cohort sealed before every frozen request was queued")
        self._state = "prefilling"

    def _validate_snapshots(
        self, snapshots: Sequence[P4CohortRequestState]
    ) -> dict[str, P4CohortRequestState]:
        by_id = {snapshot.request_id: snapshot for snapshot in snapshots}
        if len(by_id) != len(snapshots):
            self._fail("scheduler snapshot repeated a request")
        unknown = set(by_id) - set(self._frozen_by_internal)
        if unknown:
            self._fail(
                f"scheduler snapshot contains foreign requests: {sorted(unknown)}"
            )

        required = {
            request_id
            for request_id, committed in self._committed_by_internal.items()
            if committed < self.measured_decode_tokens
        }
        if set(by_id) != required:
            self._fail(
                "scheduler snapshot does not contain every unfinished cohort member"
            )

        for request_id, snapshot in by_id.items():
            values = (
                snapshot.num_prompt_tokens,
                snapshot.num_computed_tokens,
                snapshot.num_output_tokens,
                snapshot.num_preemptions,
            )
            if any(type(value) is not int or value < 0 for value in values):
                self._fail(f"request {request_id!r} has invalid scheduler counters")
            if snapshot.num_prompt_tokens <= 0:
                self._fail(f"request {request_id!r} has an empty prompt")
            prior_prompt_tokens = self._prompt_tokens_by_internal.setdefault(
                request_id, snapshot.num_prompt_tokens
            )
            if snapshot.num_prompt_tokens != prior_prompt_tokens:
                self._fail(f"request {request_id!r} prompt length changed")
            if snapshot.num_preemptions:
                self._fail(f"request {request_id!r} was preempted")
        return by_id

    def begin_step(self, snapshots: Sequence[P4CohortRequestState]) -> dict[str, Any]:
        """Open one scheduler boundary and return the exact hold set.

        Args:
            snapshots: States for every unfinished cohort member.

        Returns:
            Gate state containing held IDs and whether release starts now.

        Raises:
            KOffRuntimeError: If membership, state, or progress is invalid.
        """
        if self._state not in {"prefilling", "released"}:
            self._fail(f"scheduler step began in terminal state {self._state!r}")
        if self._step_open:
            self._fail("scheduler began a second step before closing the first")
        by_id = self._validate_snapshots(snapshots)

        release = False
        if self._state == "prefilling":
            ready: set[str] = set()
            for request_id, snapshot in by_id.items():
                prompt_tokens = snapshot.num_prompt_tokens
                computed_tokens = snapshot.num_computed_tokens
                output_tokens = snapshot.num_output_tokens
                if computed_tokens > prompt_tokens:
                    self._fail(f"request {request_id!r} decoded before cohort release")
                if output_tokens not in {0, self.unmeasured_prefill_tokens}:
                    self._fail(
                        f"request {request_id!r} has {output_tokens} pre-release "
                        "output tokens"
                    )
                if (computed_tokens == prompt_tokens) != (
                    output_tokens == self.unmeasured_prefill_tokens
                ):
                    self._fail(
                        f"request {request_id!r} prefill sample boundary is invalid"
                    )
                if output_tokens == self.unmeasured_prefill_tokens:
                    ready.add(request_id)
            self._held_internal_ids = ready
            if len(ready) == len(self.frozen_request_ids):
                self._held_internal_ids.clear()
                self._state = "released"
                self._step_mode = "first_decode"
                release = True
            else:
                self._step_mode = "prefill"
        else:
            for request_id, snapshot in by_id.items():
                if (
                    snapshot.num_computed_tokens < snapshot.num_prompt_tokens
                    or snapshot.num_output_tokens < self.unmeasured_prefill_tokens
                ):
                    self._fail(
                        f"released request {request_id!r} regressed into prefill"
                    )
            self._held_internal_ids.clear()
            self._step_mode = "decode"

        self._step_open = True
        return {
            "state": self._state,
            "release": release,
            "held_request_ids": tuple(
                request_id
                for request_id in self.abort_request_ids
                if request_id in self._held_internal_ids
            ),
        }

    @staticmethod
    def _validate_quality_counter(value: int, name: str) -> None:
        if type(value) is not int or value < 0:
            raise KOffRuntimeError(f"P4 cohort {name} counter is invalid")

    def end_step(
        self,
        *,
        scheduled_request_ids: Sequence[str],
        pure_decode: bool,
        action_id: str | None,
        target_query_widths: Mapping[str, int],
        committed_tokens: Mapping[str, int],
        prefill_progress_tokens: int = 0,
        preemptions: int = 0,
        recomputed_tokens: int = 0,
        invalid_spec_tokens: int = 0,
    ) -> dict[str, Any]:
        """Close one scheduler/model event and enforce release accounting.

        Args:
            scheduled_request_ids: Runtime IDs selected for this event.
            pure_decode: Whether every selected request was a decode row.
            action_id: Verified steady-state action, or ``None`` for prefill.
            target_query_widths: Target query width by selected runtime ID.
            committed_tokens: Post-stop measured commits by runtime ID.
            prefill_progress_tokens: Prompt tokens computed in a prefill event.
            preemptions: Event-level preemption count.
            recomputed_tokens: Event-level recomputed token count.
            invalid_spec_tokens: Event-level invalid speculative token count.

        Returns:
            Stable state and accounting summary after the event.

        Raises:
            KOffRuntimeError: If release, quality, or accounting fails closed.
        """
        if not self._step_open:
            self._fail("scheduler ended a step that was not opened")
        scheduled_ids = tuple(scheduled_request_ids)
        if len(set(scheduled_ids)) != len(scheduled_ids):
            self._fail("scheduled cohort request IDs are duplicated")
        if set(scheduled_ids) - set(self._frozen_by_internal):
            self._fail("scheduler selected a request outside the capture cohort")
        try:
            for name, value in (
                ("preemptions", preemptions),
                ("recomputed_tokens", recomputed_tokens),
                ("invalid_spec_tokens", invalid_spec_tokens),
            ):
                self._validate_quality_counter(value, name)
        except KOffRuntimeError as exc:
            self._fail(str(exc))
        if preemptions or recomputed_tokens or invalid_spec_tokens:
            self._fail(
                "cohort event contains preemption, recomputation, or invalid "
                "speculative tokens"
            )

        if self._step_mode == "prefill":
            if pure_decode or action_id is not None:
                self._fail("prefill event was labeled as measured decode")
            if target_query_widths or committed_tokens:
                self._fail("prefill event contributed measured decode accounting")
            if set(scheduled_ids) & self._held_internal_ids:
                self._fail("a completed prefill was scheduled before cohort release")
            if type(prefill_progress_tokens) is not int or prefill_progress_tokens < 0:
                self._fail("prefill progress counter is invalid")
            if scheduled_ids and prefill_progress_tokens > 0:
                self._idle_step_count = 0
            else:
                self._idle_step_count += 1
            if self._idle_step_count >= self._DEADLOCK_STEP_LIMIT:
                self._fail("cohort prefill made no progress and would deadlock")
            self._prefill_step_count += 1
        else:
            if not pure_decode:
                self._fail("released cohort entered a mixed prefill/decode event")
            if action_id != self.action_id:
                self._fail("released cohort verified another action")
            if prefill_progress_tokens != 0:
                self._fail("released cohort reported prefill progress")
            active_ids = {
                request_id
                for request_id, committed in self._committed_by_internal.items()
                if committed < self.measured_decode_tokens
            }
            if set(scheduled_ids) != active_ids:
                self._fail(
                    "released cohort did not schedule every unfinished member together"
                )
            if set(target_query_widths) != active_ids:
                self._fail("released cohort target query-width rows are incomplete")
            expected_width = self._QUERY_WIDTH_BY_ACTION[self.action_id]
            if any(width != expected_width for width in target_query_widths.values()):
                self._fail("released cohort target query width differs from its action")
            if set(committed_tokens) != active_ids:
                self._fail("released cohort commit rows are incomplete")
            for request_id, committed in committed_tokens.items():
                if type(committed) is not int or not 0 <= committed <= expected_width:
                    self._fail(
                        f"request {request_id!r} has an invalid same-event commit"
                    )
                new_total = self._committed_by_internal[request_id] + committed
                if new_total > self.measured_decode_tokens:
                    self._fail(f"request {request_id!r} exceeded measured decode work")
                self._committed_by_internal[request_id] = new_total
            if sum(committed_tokens.values()) > 0:
                self._idle_step_count = 0
            else:
                self._idle_step_count += 1
            if self._idle_step_count >= self._DEADLOCK_STEP_LIMIT:
                self._fail("released cohort made no commit progress and would deadlock")
            self._first_decode_observed = True
            self._decode_step_count += 1

        self._step_open = False
        self._step_mode = ""
        return self.summary()

    def finish_request(self, request_id: str, output_tokens: int) -> None:
        """Prove frontend work is one prefill sample plus measured commits."""
        if self._state != "released" or self._step_open:
            self._fail("request finished outside a closed released boundary")
        if request_id not in self._frozen_by_internal:
            self._fail("an unknown request finished the capture cohort")
        if request_id in self._finished_internal_ids:
            self._fail("a capture-cohort request finished twice")
        if self._committed_by_internal[request_id] != self.measured_decode_tokens:
            self._fail("request finished before exact measured decode work")
        expected_output_tokens = (
            self.unmeasured_prefill_tokens + self.measured_decode_tokens
        )
        if output_tokens != expected_output_tokens:
            self._fail("frontend output work does not preserve the prefill offset")
        self._finished_internal_ids.add(request_id)
        if len(self._finished_internal_ids) == len(self.frozen_request_ids):
            self._state = "complete"

    def abort_member(self, request_id: str) -> tuple[str, ...]:
        """Abort the entire cohort if one member disappears or is cancelled."""
        if self._state in {"complete", "aborted"}:
            self._fail("member abort occurred after cohort termination")
        if request_id not in self._frozen_by_internal:
            self._fail("an unknown request attempted to abort the cohort")
        abort_ids = self.abort_request_ids
        self._abort_reason = f"member {request_id!r} aborted"
        self._state = "aborted"
        self._step_open = False
        return abort_ids

    def abort(self, reason: str) -> tuple[str, ...]:
        """Abort every unfinished member for an integration-level failure."""
        if self._state in {"complete", "aborted"}:
            self._fail("cohort abort occurred after cohort termination")
        if not isinstance(reason, str) or not reason:
            self._fail("cohort abort reason is empty")
        abort_ids = self.abort_request_ids
        self._abort_reason = reason
        self._state = "aborted"
        self._step_open = False
        return abort_ids

    def close(self) -> None:
        """Accept shutdown only after exact frontend and measured work closes."""
        if self._state != "complete":
            self._fail("cohort closed before exact work completion")

    def summary(self) -> dict[str, Any]:
        """Return stable lifecycle and accounting evidence."""
        committed_by_frozen = {
            frozen_id: self._committed_by_internal.get(
                self._internal_by_frozen.get(frozen_id, ""), 0
            )
            for frozen_id in self.frozen_request_ids
        }
        return {
            "cohort_id": self.cohort_id,
            "state": self._state,
            "action_id": self.action_id,
            "registered_request_count": len(self._internal_by_frozen),
            "expected_request_count": len(self.frozen_request_ids),
            "unmeasured_prefill_tokens_per_request": (self.unmeasured_prefill_tokens),
            "measured_decode_tokens_per_request": self.measured_decode_tokens,
            "committed_by_request": committed_by_frozen,
            "finished_request_count": len(self._finished_internal_ids),
            "prefill_step_count": self._prefill_step_count,
            "decode_step_count": self._decode_step_count,
            "first_decode_observed": self._first_decode_observed,
            "abort_reason": self._abort_reason,
        }


@dataclass(frozen=True)
class KOffAction:
    """One executable action in the minimal-B0 registry."""

    action_id: str
    k: int
    target_graph_id: str
    target_query_width: int
    draft_graph_id: str | None
    draft_query_width: int | None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


OFF_ACTION = KOffAction(
    action_id=OFF_ACTION_ID,
    k=0,
    target_graph_id="target-k1",
    target_query_width=1,
    draft_graph_id=None,
    draft_query_width=None,
)
K4_ACTION = KOffAction(
    action_id=K4_ACTION_ID,
    k=4,
    target_graph_id="target-k5",
    target_query_width=5,
    draft_graph_id="draft-target-matching-k1",
    draft_query_width=1,
)
ACTIONS_BY_ID = {
    OFF_ACTION.action_id: OFF_ACTION,
    K4_ACTION.action_id: K4_ACTION,
}
ACTIONS_BY_K = {action.k: action for action in ACTIONS_BY_ID.values()}


@dataclass(frozen=True)
class KOffBootOptions:
    """Environment-controlled mechanism requirements for minimal B0."""

    enabled: bool
    trace_path: str
    p4_capture_config_path: str
    p4_capture_output_path: str
    p4_boot_action_id: str
    p4_logical_weight_version: str
    p4_min_kv_blocks: int
    shared_kv: bool
    shared_kv_step0_decode: bool
    share_weights: bool
    draft_kv_dtype: str
    draft_kv_window: int
    draft_kv_sinks: int
    draft_skip_layers: str
    draft_partial_replica: str
    ahead_chain: bool
    consume_ahead: bool
    # Defaulted so every existing caller keeps Phase 97 semantics untouched.
    boot_scope: str = BOOT_SCOPE_MINIMAL_B0


@dataclass(frozen=True)
class KOffSchedulerMetadata:
    """Action and accounting metadata transported with one engine step."""

    engine_step_index: int
    verified_action_id: str | None
    next_action_id: str
    selection_intent: str
    capture_cohort_arm: bool
    decode_req_ids: tuple[str, ...]
    pure_decode: bool
    total_scheduled_kv_tokens: int
    generated_suffix_min: int | None
    generated_suffix_max: int | None
    shared_target_kv_blocks_in_use: int
    shared_target_kv_block_capacity: int
    preemptions: int
    recomputed_tokens: int
    scheduled_at_s: float
    aborted_action_id: str | None
    discarded_draft_width: int
    aborted_draft_request_count: int


@dataclass(frozen=True)
class KOffDraftAbort:
    """Draft rows discarded before a mixed target pass."""

    action_id: str | None
    draft_width: int
    request_count: int

    @property
    def token_count(self) -> int:
        """Return the aggregate number of discarded draft tokens."""
        return self.draft_width * self.request_count


@dataclass(frozen=True)
class SharedKVIdentity:
    """Proof summary for target-owned draft/target KV aliases."""

    binding_id: str
    pool_id: str
    layer_count: int
    storage_alias_count: int


@dataclass(frozen=True)
class SharedWeightIdentity:
    """Proof summary for target-matching draft parameter aliases."""

    target_version_id: str
    draft_version_id: str
    parameter_alias_count: int
    alias_binding_id: str = ""


@dataclass(frozen=True)
class KOffRunnerEvidence:
    """Worker-side evidence for target execution and the next draft dispatch."""

    verified_action_id: str | None
    next_action_id: str
    target_graph_id: str | None
    verified_draft_graph_id: str | None
    next_draft_graph_id: str | None
    target_query_width: int | None
    target_runtime_mode: str
    draft_step0_query_width: int | None
    draft_step0_num_tokens: int | None
    draft_step0_batch_size: int | None
    draft_step0_runtime_mode: str | None
    draft_chain_runtime_mode: str | None
    produced_draft_width: int
    draft_dispatched: bool
    binding_id: str
    pool_id: str
    true_slot_mapping_id: str
    shared_kv_layer_count: int
    shared_kv_storage_alias_count: int
    target_weight_version_id: str
    draft_weight_version_id: str
    shared_weight_binding_id: str
    shared_weight_parameter_count: int
    aborted_action_id: str | None
    discarded_draft_width: int
    aborted_draft_request_count: int
    diagnostic: Mapping[str, Any] | None = None


def options_from_env() -> KOffBootOptions:
    """Read the opt-in live contract options from vLLM's environment."""
    from vllm import envs

    return KOffBootOptions(
        enabled=envs.VLLM_SELF_SPEC_KOFF_RUNTIME,
        boot_scope=envs.VLLM_SELF_SPEC_BOOT_SCOPE,
        trace_path=envs.VLLM_SELF_SPEC_KOFF_TRACE,
        p4_capture_config_path=envs.VLLM_SELF_SPEC_P4_CAPTURE_CONFIG,
        p4_capture_output_path=envs.VLLM_SELF_SPEC_P4_CAPTURE_OUTPUT,
        p4_boot_action_id=envs.VLLM_SELF_SPEC_P4_BOOT_ACTION,
        p4_logical_weight_version=(envs.VLLM_SELF_SPEC_P4_LOGICAL_WEIGHT_VERSION),
        p4_min_kv_blocks=envs.VLLM_SELF_SPEC_P4_MIN_KV_BLOCKS,
        shared_kv=envs.VLLM_SELF_SPEC_SHARED_KV,
        shared_kv_step0_decode=envs.VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE,
        share_weights=envs.VLLM_SELF_SPEC_SHARE_WEIGHTS,
        draft_kv_dtype=envs.VLLM_SELF_SPEC_DRAFT_KV_DTYPE,
        draft_kv_window=envs.VLLM_SELF_SPEC_DRAFT_KV_WINDOW,
        draft_kv_sinks=envs.VLLM_SELF_SPEC_DRAFT_KV_SINKS,
        draft_skip_layers=envs.VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS,
        draft_partial_replica=envs.VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA,
        ahead_chain=envs.VLLM_SELF_SPEC_AHEAD_CHAIN,
        consume_ahead=envs.VLLM_SELF_SPEC_CONSUME_AHEAD,
    )


def action_for_k(k: int) -> KOffAction:
    """Resolve K to the closed two-action registry."""
    try:
        return ACTIONS_BY_K[k]
    except KeyError as exc:
        raise KOffRuntimeError(
            f"minimal-B0 permits only K=0 or K=4, got K={k}"
        ) from exc


def action_for_id(action_id: str) -> KOffAction:
    """Resolve an action ID to the closed two-action registry."""
    try:
        return ACTIONS_BY_ID[action_id]
    except KeyError as exc:
        raise KOffRuntimeError(f"unknown minimal-B0 action {action_id!r}") from exc


def validate_k_values(values: Sequence[int], source: str) -> None:
    """Reject any boot-time schedule capable of selecting another K."""
    invalid = sorted(set(values) - ALLOWED_K_VALUES)
    if invalid:
        raise KOffRuntimeError(
            f"{source} contains K values outside minimal-B0: {invalid}"
        )


def validate_policy_k_values(policy: Sequence[Mapping[str, Any]] | None) -> None:
    """Validate K values in the existing compiled-policy cell format."""
    if policy is None:
        return
    values = [int(option["K"]) for cell in policy for option in cell.get("options", ())]
    validate_k_values(values, "compiled policy")


def validate_boot_config(
    vllm_config: Any,
    options: KOffBootOptions,
    *,
    dynamic_k_values: Sequence[int] = (),
    policy: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Fail closed unless the engine is the minimal target-matching B0."""
    capture_paths = (
        options.p4_capture_config_path,
        options.p4_capture_output_path,
    )
    capture_enabled = all(capture_paths)
    conformance_fields = (
        bool(options.p4_boot_action_id),
        bool(options.p4_logical_weight_version),
        options.p4_min_kv_blocks != 0,
    )
    if any(capture_paths) and not all(capture_paths):
        raise KOffRuntimeError(
            "P4 capture requires both VLLM_SELF_SPEC_P4_CAPTURE_CONFIG and "
            "VLLM_SELF_SPEC_P4_CAPTURE_OUTPUT"
        )
    if any(conformance_fields) and not all(conformance_fields):
        raise KOffRuntimeError(
            "P4 capture conformance requires boot action, logical weight "
            "version, and minimum KV blocks together"
        )
    if any(conformance_fields) and not capture_enabled:
        raise KOffRuntimeError(
            "P4 capture conformance fields require an explicit capture"
        )
    if capture_enabled and not all(conformance_fields):
        raise KOffRuntimeError(
            "P4 capture requires boot action, logical weight version, and "
            "minimum KV blocks"
        )
    if not options.enabled:
        if options.trace_path or any(capture_paths) or any(conformance_fields):
            raise KOffRuntimeError(
                "K/OFF tracing and P4 capture require VLLM_SELF_SPEC_KOFF_RUNTIME=1"
            )
        return
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if capture_enabled and bool(getattr(scheduler_config, "async_scheduling", False)):
        raise KOffRuntimeError("P4 same-event capture requires synchronous scheduling")

    if capture_enabled:
        if options.p4_boot_action_id not in P4_ACTION_REALIZATIONS:
            raise KOffRuntimeError(
                f"unknown P4 boot action {options.p4_boot_action_id!r}"
            )
        if not _P4_LOGICAL_WEIGHT_VERSION.fullmatch(options.p4_logical_weight_version):
            raise KOffRuntimeError(
                "P4 logical weight version must be "
                "target-matching-config-sha256-<64 lowercase hex>"
            )
        if options.p4_min_kv_blocks != P4_MIN_SHARED_KV_BLOCKS:
            raise KOffRuntimeError(
                "P4 minimum KV blocks must equal the registered "
                f"{P4_MIN_SHARED_KV_BLOCKS}-block floor"
            )

    spec_config = getattr(vllm_config, "speculative_config", None)
    if spec_config is None or getattr(spec_config, "method", None) != "draft_model":
        method = None if spec_config is None else spec_config.method
        raise KOffRuntimeError(
            f"minimal-B0 requires speculative method 'draft_model', got {method!r}"
        )
    if getattr(spec_config, "num_speculative_tokens", None) != 4:
        raise KOffRuntimeError("minimal-B0 requires num_speculative_tokens=4")
    if getattr(spec_config, "disable_padded_drafter_batch", False):
        raise KOffRuntimeError("minimal-B0 requires the padded draft-model batch")

    if options.boot_scope not in BOOT_SCOPES:
        raise KOffRuntimeError(
            f"unknown boot scope {options.boot_scope!r}; "
            f"expected one of {sorted(BOOT_SCOPES)}"
        )

    # Invariants both scopes share. Shared target KV is the Phase 97 invariant
    # that Phase 98 explicitly inherits, so it is never relaxed.
    shared = (
        (options.shared_kv, "VLLM_SELF_SPEC_SHARED_KV=1"),
        (
            options.shared_kv_step0_decode,
            "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1",
        ),
        (not options.draft_kv_dtype, "no draft-only KV dtype"),
        (not options.draft_partial_replica, "no partial draft replica"),
        (not options.ahead_chain, "no ahead draft chain"),
        (not options.consume_ahead, "no consumed ahead draft chain"),
    )

    if options.boot_scope == BOOT_SCOPE_W98_LATTICE:
        skip_layers = [
            token for token in options.draft_skip_layers.split(",") if token.strip()
        ]
        skip_parses = all(token.strip().isdigit() for token in skip_layers)
        window = options.draft_kv_window
        expected_sinks = W98_WINDOW_SINKS if window else 0
        requirements = shared + (
            (
                window in W98_WINDOWS,
                f"w98 draft KV window must be one of {sorted(W98_WINDOWS)}",
            ),
            (
                options.draft_kv_sinks == expected_sinks,
                f"w98 window={window} requires sinks={expected_sinks}",
            ),
            (skip_parses, "w98 skip layers must be a comma list of integers"),
            (
                len(skip_layers) in W98_SKIP_COUNTS,
                f"w98 skip count must be one of {sorted(W98_SKIP_COUNTS)}",
            ),
        )
    else:
        window_valid = options.draft_kv_window == 0
        window_description = "no draft KV window"
        if capture_enabled and options.p4_boot_action_id == W512_ACTION_ID:
            window_valid = (
                options.draft_kv_window == 512 and options.draft_kv_sinks == 16
            )
            window_description = "w512 capture requires window=512 and sinks=16"
        requirements = shared + (
            (options.share_weights, "VLLM_SELF_SPEC_SHARE_WEIGHTS=1"),
            (window_valid, window_description),
            (not options.draft_skip_layers.strip(), "no skipped draft layers"),
        )
    missing = [description for valid, description in requirements if not valid]
    if missing:
        raise KOffRuntimeError(
            f"{options.boot_scope} boot contract violation: " + "; ".join(missing)
        )

    target = getattr(vllm_config, "model_config", None)
    draft = getattr(spec_config, "draft_model_config", None)
    if target is None or draft is None:
        raise KOffRuntimeError(
            f"{options.boot_scope} requires target and draft model configs"
        )
    same_checkpoint = getattr(draft, "model", None) == getattr(target, "model", None)
    same_quantization = getattr(draft, "quantization", None) == getattr(
        target, "quantization", None
    )
    if options.boot_scope == BOOT_SCOPE_W98_LATTICE:
        # The quant axis is the point of this scope, so a differing draft is
        # admitted -- but only as a declared quantized realization. A bare
        # unrelated checkpoint is still refused, and a self-draft must still
        # match on both fields.
        if same_checkpoint and not same_quantization:
            raise KOffRuntimeError("w98 self-draft must match the target quantization")
        if not same_checkpoint and not getattr(draft, "quantization", None):
            raise KOffRuntimeError(
                "w98 draft checkpoint differs from the target but declares no "
                "quantization; only a quantized realization may differ"
            )
    else:
        if not same_checkpoint:
            raise KOffRuntimeError("minimal-B0 draft checkpoint must match the target")
        if not same_quantization:
            raise KOffRuntimeError(
                "minimal-B0 draft quantization must match the target"
            )

    validate_k_values(dynamic_k_values, "dynamic speculative schedule")
    validate_policy_k_values(policy)


def validate_p4_shared_kv_capacity(capacity: int, minimum: int) -> None:
    """Stop P4 capture when the live target-owned pool is below its floor."""
    if type(capacity) is not int or capacity < 0:
        raise KOffRuntimeError(f"invalid shared-KV block capacity {capacity!r}")
    if minimum != P4_MIN_SHARED_KV_BLOCKS:
        raise KOffRuntimeError(
            "P4 capture must retain the registered "
            f"{P4_MIN_SHARED_KV_BLOCKS}-block floor"
        )
    if capacity < minimum:
        raise KOffRuntimeError(
            f"P4 shared-KV capacity {capacity} is below the {minimum}-block floor"
        )


def abort_mixed_decode_drafts(
    decode_req_ids: Sequence[str],
    num_scheduled_tokens: MutableMapping[str, int],
    scheduled_spec_decode_tokens: MutableMapping[str, list[int]],
    draft_action_by_req: MutableMapping[str, str],
) -> KOffDraftAbort:
    """Discard one uniform K4 dispatch before a mixed target pass.

    The scheduler has already admitted the batch when this runs. A mixed
    prefill/decode batch must execute existing decodes as q=1/OFF, so pending
    K4 rows are removed and their provenance is replaced with OFF provenance.
    """
    decode_ids = tuple(decode_req_ids)
    if not decode_ids or len(decode_ids) == len(num_scheduled_tokens):
        return KOffDraftAbort(None, 0, 0)

    unexpected = set(scheduled_spec_decode_tokens) - set(decode_ids)
    if unexpected:
        raise KOffRuntimeError(
            "mixed boundary has draft rows for non-decode requests: "
            f"{sorted(unexpected)}"
        )

    drafted_ids = [
        req_id for req_id in decode_ids if scheduled_spec_decode_tokens.get(req_id)
    ]
    if not drafted_ids:
        return KOffDraftAbort(None, 0, 0)
    if len(drafted_ids) != len(decode_ids):
        raise KOffRuntimeError(
            "mixed boundary cannot partially abort a decode dispatch"
        )

    widths = {len(scheduled_spec_decode_tokens[req_id]) for req_id in drafted_ids}
    provenance = {draft_action_by_req.get(req_id) for req_id in drafted_ids}
    if widths != {K4_ACTION.k} or provenance != {K4_ACTION_ID}:
        raise KOffRuntimeError(
            "mixed boundary may abort only one complete K4 dispatch; "
            f"widths={widths}, provenance={provenance}"
        )
    for req_id in drafted_ids:
        query_width = num_scheduled_tokens.get(req_id)
        if query_width != K4_ACTION.target_query_width:
            raise KOffRuntimeError(
                f"mixed-boundary request {req_id!r} has target query width "
                f"{query_width}, expected {K4_ACTION.target_query_width}"
            )

    for req_id in drafted_ids:
        scheduled_spec_decode_tokens.pop(req_id)
        num_scheduled_tokens[req_id] = OFF_ACTION.target_query_width
        draft_action_by_req[req_id] = OFF_ACTION_ID

    return KOffDraftAbort(K4_ACTION_ID, K4_ACTION.k, len(drafted_ids))


def infer_verified_action(
    decode_req_ids: Sequence[str],
    num_scheduled_tokens: Mapping[str, int],
    scheduled_spec_decode_tokens: Mapping[str, Sequence[int]],
    draft_action_by_req: Mapping[str, str],
) -> KOffAction | None:
    """Infer and provenance-check the action verified by a target pass."""
    if not decode_req_ids:
        return None

    widths = {
        len(scheduled_spec_decode_tokens.get(req_id, ())) for req_id in decode_req_ids
    }
    if len(widths) != 1:
        raise KOffRuntimeError(
            f"one action must cover the whole decode dispatch, got widths {widths}"
        )
    action = action_for_k(widths.pop())
    for req_id in decode_req_ids:
        query_width = num_scheduled_tokens[req_id]
        if query_width != action.target_query_width:
            raise KOffRuntimeError(
                f"request {req_id!r} scheduled target query width "
                f"{query_width}, expected {action.target_query_width} for "
                f"{action.action_id}"
            )
        provenance = draft_action_by_req.get(req_id)
        if provenance != action.action_id:
            raise KOffRuntimeError(
                f"request {req_id!r} draft provenance is {provenance!r}, "
                f"expected {action.action_id!r}"
            )
    return action


def make_scheduler_metadata(
    *,
    engine_step_index: int,
    next_k: int,
    selection_intent: str,
    decode_req_ids: Sequence[str],
    num_scheduled_tokens: Mapping[str, int],
    scheduled_spec_decode_tokens: Mapping[str, Sequence[int]],
    draft_action_by_req: Mapping[str, str],
    context_tokens_by_req: Mapping[str, int],
    generated_suffix_by_req: Mapping[str, int],
    shared_target_kv_blocks_in_use: int,
    shared_target_kv_block_capacity: int,
    preemptions: int,
    recomputed_tokens: int,
    scheduled_at_s: float,
    aborted_action_id: str | None = None,
    discarded_draft_width: int = 0,
    aborted_draft_request_count: int = 0,
    capture_cohort_arm: bool = False,
) -> KOffSchedulerMetadata:
    """Build scheduler-to-runner metadata after all K gates have fired."""
    if selection_intent not in {"exploit", "probe", "force_off"}:
        raise KOffRuntimeError(
            f"invalid minimal-B0 selection intent {selection_intent!r}"
        )
    next_action = action_for_k(next_k)
    verified = infer_verified_action(
        decode_req_ids,
        num_scheduled_tokens,
        scheduled_spec_decode_tokens,
        draft_action_by_req,
    )
    pure_decode = bool(decode_req_ids) and len(decode_req_ids) == len(
        num_scheduled_tokens
    )
    contains_prefill = len(decode_req_ids) < len(num_scheduled_tokens)
    if type(capture_cohort_arm) is not bool:
        raise KOffRuntimeError("capture-cohort arm marker must be boolean")
    if capture_cohort_arm and (
        not contains_prefill
        or decode_req_ids
        or selection_intent != "exploit"
        or aborted_action_id is not None
    ):
        raise KOffRuntimeError(
            "capture-cohort arming requires one pure-prefill exploit event"
        )
    if (
        contains_prefill
        and not capture_cohort_arm
        and (next_action is not OFF_ACTION or selection_intent != "force_off")
    ):
        raise KOffRuntimeError(
            "a prefill or mixed target pass must dispatch OFF with force_off intent"
        )
    if contains_prefill and verified is K4_ACTION and aborted_action_id is None:
        raise KOffRuntimeError("a mixed target pass cannot verify K4 without aborting")
    if aborted_action_id is None:
        if discarded_draft_width or aborted_draft_request_count:
            raise KOffRuntimeError(
                "non-abort metadata must use discarded width/count 0"
            )
    else:
        aborted_action = action_for_id(aborted_action_id)
        if aborted_action is not K4_ACTION:
            raise KOffRuntimeError("minimal-B0 may abort only K4")
        if discarded_draft_width != aborted_action.k:
            raise KOffRuntimeError(
                "discarded draft width does not match the aborted action"
            )
        if aborted_draft_request_count != len(decode_req_ids) or not decode_req_ids:
            raise KOffRuntimeError(
                "aborted draft row count must equal the mixed decode count"
            )
        if pure_decode:
            raise KOffRuntimeError("a pure-decode step cannot carry a draft abort")
        if verified is not OFF_ACTION:
            raise KOffRuntimeError("an aborted target pass must verify OFF at q=1")
        if next_action is not OFF_ACTION or selection_intent != "force_off":
            raise KOffRuntimeError(
                "a mixed-boundary abort must dispatch OFF with force_off intent"
            )
    suffixes = [generated_suffix_by_req[req_id] for req_id in decode_req_ids]
    return KOffSchedulerMetadata(
        engine_step_index=engine_step_index,
        verified_action_id=None if verified is None else verified.action_id,
        next_action_id=next_action.action_id,
        selection_intent=selection_intent,
        capture_cohort_arm=capture_cohort_arm,
        decode_req_ids=tuple(decode_req_ids),
        pure_decode=pure_decode,
        total_scheduled_kv_tokens=sum(
            context_tokens_by_req[req_id] for req_id in decode_req_ids
        ),
        generated_suffix_min=min(suffixes) if suffixes else None,
        generated_suffix_max=max(suffixes) if suffixes else None,
        shared_target_kv_blocks_in_use=shared_target_kv_blocks_in_use,
        shared_target_kv_block_capacity=shared_target_kv_block_capacity,
        preemptions=preemptions,
        recomputed_tokens=recomputed_tokens,
        scheduled_at_s=scheduled_at_s,
        aborted_action_id=aborted_action_id,
        discarded_draft_width=discarded_draft_width,
        aborted_draft_request_count=aborted_draft_request_count,
    )


def validate_draft_token_batch(
    action_id: str | None,
    draft_token_ids: Sequence[Sequence[int]],
) -> KOffAction:
    """Validate the action tag and every row returned to the scheduler."""
    if action_id is None:
        raise KOffRuntimeError("live draft tokens are missing their action tag")
    action = action_for_id(action_id)
    bad = [len(row) for row in draft_token_ids if len(row) != action.k]
    if bad:
        raise KOffRuntimeError(
            f"{action.action_id} produced draft row widths {bad}, expected {action.k}"
        )
    return action


def validate_action_transport(
    metadata: KOffSchedulerMetadata,
    *,
    next_k: int,
    num_scheduled_tokens: Mapping[str, int],
    scheduled_spec_decode_tokens: Mapping[str, Sequence[int]],
) -> None:
    """Revalidate scheduler action IDs and query widths in the worker."""
    next_action = action_for_k(next_k)
    if next_action.action_id != metadata.next_action_id:
        raise KOffRuntimeError(
            "transported next action does not match num_spec_tokens_to_schedule"
        )
    contains_prefill = len(metadata.decode_req_ids) < len(num_scheduled_tokens)
    if metadata.capture_cohort_arm and (
        not contains_prefill
        or metadata.decode_req_ids
        or metadata.selection_intent != "exploit"
        or metadata.aborted_action_id is not None
    ):
        raise KOffRuntimeError(
            "transported capture-cohort arm is not a pure-prefill exploit"
        )
    if (
        contains_prefill
        and not metadata.capture_cohort_arm
        and (
            metadata.next_action_id != OFF_ACTION_ID
            or metadata.selection_intent != "force_off"
        )
    ):
        raise KOffRuntimeError(
            "transported prefill or mixed pass did not force next OFF"
        )
    if (
        contains_prefill
        and metadata.verified_action_id == K4_ACTION_ID
        and metadata.aborted_action_id is None
    ):
        raise KOffRuntimeError("transported mixed pass attempted to verify K4")
    if metadata.aborted_action_id is None:
        if metadata.discarded_draft_width or metadata.aborted_draft_request_count:
            raise KOffRuntimeError(
                "transported non-abort metadata has discarded draft rows"
            )
    else:
        aborted_action = action_for_id(metadata.aborted_action_id)
        if (
            aborted_action is not K4_ACTION
            or metadata.discarded_draft_width != K4_ACTION.k
            or metadata.aborted_draft_request_count != len(metadata.decode_req_ids)
            or not metadata.decode_req_ids
        ):
            raise KOffRuntimeError("transported mixed-boundary abort is malformed")
        if (
            metadata.pure_decode
            or metadata.verified_action_id != OFF_ACTION_ID
            or metadata.next_action_id != OFF_ACTION_ID
            or metadata.selection_intent != "force_off"
        ):
            raise KOffRuntimeError(
                "transported mixed-boundary abort did not force q=1/OFF"
            )
    if metadata.verified_action_id is None:
        if metadata.decode_req_ids:
            raise KOffRuntimeError("decode requests are missing a verified action")
        return
    verified = action_for_id(metadata.verified_action_id)
    for req_id in metadata.decode_req_ids:
        width = len(scheduled_spec_decode_tokens.get(req_id, ()))
        if width != verified.k:
            raise KOffRuntimeError(
                f"worker received draft width {width} for "
                f"{verified.action_id}, expected {verified.k}"
            )
        query_width = num_scheduled_tokens.get(req_id)
        if query_width != verified.target_query_width:
            raise KOffRuntimeError(
                f"worker received target query width {query_width} for "
                f"{verified.action_id}, expected {verified.target_query_width}"
            )


def draft_output_width(output: Any) -> int:
    """Return one uniform draft width from a tensor or nested sequence."""
    if hasattr(output, "ndim") and hasattr(output, "shape"):
        if output.ndim != 2:
            raise KOffRuntimeError(
                f"draft output must be rank 2, got rank {output.ndim}"
            )
        return int(output.shape[1])
    if not isinstance(output, Sequence):
        raise KOffRuntimeError(f"unsupported draft output type {type(output)!r}")
    widths = {len(row) for row in output}
    if len(widths) > 1:
        raise KOffRuntimeError(f"draft output has mixed row widths {widths}")
    return widths.pop() if widths else 0


def draft_output_batch_size(output: Any) -> int:
    """Return the number of draft rows from a tensor or nested sequence."""
    if hasattr(output, "ndim") and hasattr(output, "shape"):
        if output.ndim != 2:
            raise KOffRuntimeError(
                f"draft output must be rank 2, got rank {output.ndim}"
            )
        return int(output.shape[0])
    if not isinstance(output, Sequence):
        raise KOffRuntimeError(f"unsupported draft output type {type(output)!r}")
    return len(output)


def validate_draft_dispatch(
    metadata: KOffSchedulerMetadata,
    output: Any,
    *,
    proposal_called: bool,
) -> int:
    """Check that the selected next action executed with the promised width."""
    action = action_for_id(metadata.next_action_id)
    width = draft_output_width(output)
    if width != action.k:
        raise KOffRuntimeError(
            f"{action.action_id} produced width {width}, expected {action.k}"
        )
    if action.k and not proposal_called:
        raise KOffRuntimeError(
            f"{action.action_id} was selected but the draft proposal did not run"
        )
    return width


def _short_digest(parts: Sequence[str]) -> str:
    payload = "\n".join(parts).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _storage_description(tensor: Any) -> str:
    try:
        storage = tensor.untyped_storage()
        return ":".join(
            (
                str(storage.data_ptr()),
                str(tensor.storage_offset()),
                str(tuple(tensor.shape)),
                str(tuple(tensor.stride())),
                str(tensor.dtype),
            )
        )
    except (AttributeError, RuntimeError) as exc:
        raise KOffRuntimeError("expected a storage-backed tensor") from exc


def _storage_root_description(tensor: Any) -> str:
    try:
        storage = tensor.untyped_storage()
        return f"{storage.data_ptr()}:{tensor.dtype}"
    except (AttributeError, RuntimeError) as exc:
        raise KOffRuntimeError("expected a storage-backed tensor") from exc


def validate_shared_kv_aliases(
    layer_mapping: Mapping[str, str],
    kv_caches: Mapping[str, Any],
    pool_object: Any,
) -> SharedKVIdentity:
    """Prove every draft layer is the target twin's exact KV tensor object."""
    if not layer_mapping:
        raise KOffRuntimeError("minimal-B0 shared-KV layer mapping is empty")
    binding_parts: list[str] = []
    storage_parts: list[str] = []
    storage_roots: set[str] = set()
    for draft_layer, target_layer in sorted(layer_mapping.items()):
        if draft_layer not in kv_caches or target_layer not in kv_caches:
            raise KOffRuntimeError(
                f"missing live KV tensor for {draft_layer!r} or {target_layer!r}"
            )
        draft_cache = kv_caches[draft_layer]
        target_cache = kv_caches[target_layer]
        if draft_cache is not target_cache:
            raise KOffRuntimeError(
                f"draft KV {draft_layer!r} is not target tensor {target_layer!r}"
            )
        description = _storage_description(target_cache)
        binding_parts.append(f"{draft_layer}->{target_layer}:{description}")
        storage_parts.append(f"{target_layer}:{description}")
        storage_roots.add(_storage_root_description(target_cache))
    pool_parts = [f"allocator:{id(pool_object):x}", *sorted(storage_roots)]
    return SharedKVIdentity(
        binding_id=f"live-kv-binding-{_short_digest(binding_parts)}",
        pool_id=f"target-kv-pool-{_short_digest(pool_parts)}",
        layer_count=len(layer_mapping),
        storage_alias_count=len(storage_parts),
    )


def _skipped_layer_prefixes(skip_layers: str) -> tuple[str, ...]:
    """Return the parameter-name prefixes a skip set legitimately removes."""
    return tuple(
        f"model.layers.{token.strip()}."
        for token in skip_layers.split(",")
        if token.strip()
    )


def validate_independent_draft_weights(
    target_model: Any,
    draft_model: Any,
) -> SharedWeightIdentity:
    """Prove a distinct draft realization shares NO weight storage.

    A quantized draft cannot alias the target, so the target-matching proof is
    false by design. The property that remains true and worth proving is the
    opposite one: the two parameter sets are genuinely independent, so a
    half-aliased state cannot pass unnoticed. Shared target KV is proven
    separately by ``validate_shared_kv_aliases`` and is unaffected.

    Args:
        target_model: The target module.
        draft_model: The distinct draft realization.

    Returns:
        An identity recording zero aliases.

    Raises:
        KOffRuntimeError: If any draft parameter shares target storage.
    """
    target_params = dict(target_model.named_parameters())
    draft_params = dict(draft_model.named_parameters())
    if not draft_params:
        raise KOffRuntimeError("w98 draft model has no parameters")
    target_storage = {_storage_description(param) for param in target_params.values()}
    shared = sorted(
        name
        for name, param in draft_params.items()
        if _storage_description(param) in target_storage
    )
    if shared:
        raise KOffRuntimeError(
            "w98 distinct draft realization unexpectedly shares target weight "
            f"storage: {shared[:8]}"
        )
    parts = [
        f"{name}:{_storage_description(param)}"
        for name, param in sorted(draft_params.items())
    ]
    binding_id = f"w98-independent-draft-{_short_digest(parts)}"
    return SharedWeightIdentity(
        target_version_id=f"target-{_short_digest(sorted(target_storage))}",
        draft_version_id=binding_id,
        parameter_alias_count=0,
        alias_binding_id=binding_id,
    )


def validate_shared_weight_aliases(
    target_model: Any,
    draft_model: Any,
    logical_version_id: str | None = None,
    *,
    boot_scope: str = BOOT_SCOPE_MINIMAL_B0,
    skip_layers: str = "",
) -> SharedWeightIdentity:
    """Prove every target-matching draft parameter aliases its target twin.

    Under ``w98-lattice`` a skip set legitimately removes whole layers, so the
    proof weakens from set EQUALITY to a SUBSET: every parameter the draft
    still has must alias its target twin, the draft may add nothing, and the
    only permitted absences are the declared skipped layers. That is strictly
    weaker than minimal-b0 and still a real proof.

    Args:
        target_model: The target module.
        draft_model: The draft module.
        logical_version_id: The registered logical weight version, if any.
        boot_scope: The registered boot scope being enforced.
        skip_layers: The declared skip set, for the subset allowance.

    Returns:
        The alias identity.

    Raises:
        KOffRuntimeError: If the alias property fails for the active scope.
    """
    if logical_version_id is not None and not _P4_LOGICAL_WEIGHT_VERSION.fullmatch(
        logical_version_id
    ):
        raise KOffRuntimeError(
            "P4 logical weight version must be "
            "target-matching-config-sha256-<64 lowercase hex>"
        )
    target_params = dict(target_model.named_parameters())
    draft_params = dict(draft_model.named_parameters())
    if not draft_params:
        raise KOffRuntimeError("minimal-B0 draft model has no parameters")
    missing = sorted(target_params.keys() - draft_params.keys())
    extra = sorted(draft_params.keys() - target_params.keys())
    if extra:
        raise KOffRuntimeError(
            f"draft adds parameters the target lacks: extra_in_draft={extra}"
        )
    if missing and boot_scope == BOOT_SCOPE_W98_LATTICE:
        allowed = _skipped_layer_prefixes(skip_layers)
        unexplained = sorted(
            name for name in missing if not (allowed and name.startswith(allowed))
        )
        if unexplained:
            raise KOffRuntimeError(
                "w98 draft is missing parameters outside its declared skip "
                f"set: {unexplained[:8]}"
            )
    elif missing:
        raise KOffRuntimeError(
            "target/draft parameter sets differ: "
            f"missing_from_draft={missing}, extra_in_draft={extra}"
        )
    version_parts: list[str] = []
    for name, draft_param in sorted(draft_params.items()):
        target_param = target_params.get(name)
        if target_param is None or draft_param is not target_param:
            raise KOffRuntimeError(
                f"draft parameter {name!r} does not alias its target twin"
            )
        version_parts.append(f"{name}:{_storage_description(target_param)}")
    binding_id = f"target-alias-{_short_digest(version_parts)}"
    version_id = logical_version_id or binding_id
    return SharedWeightIdentity(
        target_version_id=version_id,
        draft_version_id=version_id,
        parameter_alias_count=len(draft_params),
        alias_binding_id=binding_id,
    )


def slot_mapping_identity(slot_mappings: Any) -> str:
    """Fingerprint canonical backing storage passed into the draft."""
    storage_roots: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                visit(child)
        else:
            storage_roots.add(_storage_root_description(value))

    if slot_mappings is None:
        raise KOffRuntimeError("shared-KV draft requires target slot mappings")
    visit(slot_mappings)
    if not storage_roots:
        raise KOffRuntimeError("shared-KV target slot mapping is empty")
    return f"target-true-slots-{_short_digest(sorted(storage_roots))}"


def make_runner_evidence(
    *,
    metadata: KOffSchedulerMetadata,
    target_runtime_mode: str,
    output: Any,
    proposal_called: bool,
    draft_step0_query_width: int | None,
    draft_step0_num_tokens: int | None,
    draft_step0_batch_size: int | None,
    draft_step0_runtime_mode: str | None,
    draft_chain_runtime_mode: str | None,
    shared_kv: SharedKVIdentity,
    shared_weights: SharedWeightIdentity,
    true_slot_mapping_id: str,
    diagnostic: Mapping[str, Any] | None = None,
) -> KOffRunnerEvidence:
    """Validate worker execution and build serializable evidence."""
    next_action = action_for_id(metadata.next_action_id)
    verified = (
        None
        if metadata.verified_action_id is None
        else action_for_id(metadata.verified_action_id)
    )
    width = validate_draft_dispatch(metadata, output, proposal_called=proposal_called)
    if next_action.k:
        work_values = (draft_step0_num_tokens, draft_step0_batch_size)
        if any(type(value) is not int or value < 1 for value in work_values):
            raise KOffRuntimeError(
                f"{next_action.action_id} is missing positive draft step-0 work "
                "evidence"
            )
        assert draft_step0_num_tokens is not None
        assert draft_step0_batch_size is not None
        if draft_step0_num_tokens < draft_step0_batch_size:
            raise KOffRuntimeError(
                f"{next_action.action_id} draft step-0 token count "
                f"{draft_step0_num_tokens} is smaller than batch size "
                f"{draft_step0_batch_size}"
            )
        output_batch_size = draft_output_batch_size(output)
        if output_batch_size != draft_step0_batch_size:
            raise KOffRuntimeError(
                f"{next_action.action_id} produced {output_batch_size} draft rows, "
                f"expected step-0 batch size {draft_step0_batch_size}"
            )
        if draft_step0_query_width is not None and (
            type(draft_step0_query_width) is not int
            or draft_step0_query_width < 1
            or draft_step0_num_tokens
            != draft_step0_batch_size * draft_step0_query_width
        ):
            raise KOffRuntimeError(
                f"{next_action.action_id} draft step-0 uniform query width "
                "does not match its work evidence"
            )
        if (
            metadata.pure_decode
            and draft_step0_query_width != next_action.draft_query_width
        ):
            raise KOffRuntimeError(
                f"{next_action.action_id} used draft step-0 query width "
                f"{draft_step0_query_width}, expected "
                f"{next_action.draft_query_width}"
            )
        if not metadata.pure_decode and (
            not metadata.capture_cohort_arm or metadata.decode_req_ids
        ):
            raise KOffRuntimeError(
                f"{next_action.action_id} non-decode dispatch requires "
                "pure-prefill cohort arming"
            )
        if draft_step0_runtime_mode is None or draft_chain_runtime_mode is None:
            raise KOffRuntimeError(
                f"{next_action.action_id} is missing draft execution modes"
            )
    return KOffRunnerEvidence(
        verified_action_id=metadata.verified_action_id,
        next_action_id=metadata.next_action_id,
        target_graph_id=None if verified is None else verified.target_graph_id,
        verified_draft_graph_id=(None if verified is None else verified.draft_graph_id),
        next_draft_graph_id=next_action.draft_graph_id,
        target_query_width=(None if verified is None else verified.target_query_width),
        target_runtime_mode=target_runtime_mode,
        draft_step0_query_width=draft_step0_query_width,
        draft_step0_num_tokens=draft_step0_num_tokens,
        draft_step0_batch_size=draft_step0_batch_size,
        draft_step0_runtime_mode=draft_step0_runtime_mode,
        draft_chain_runtime_mode=draft_chain_runtime_mode,
        produced_draft_width=width,
        draft_dispatched=bool(next_action.k),
        binding_id=shared_kv.binding_id,
        pool_id=shared_kv.pool_id,
        true_slot_mapping_id=true_slot_mapping_id,
        shared_kv_layer_count=shared_kv.layer_count,
        shared_kv_storage_alias_count=shared_kv.storage_alias_count,
        target_weight_version_id=shared_weights.target_version_id,
        draft_weight_version_id=shared_weights.draft_version_id,
        shared_weight_binding_id=shared_weights.alias_binding_id,
        shared_weight_parameter_count=shared_weights.parameter_alias_count,
        aborted_action_id=metadata.aborted_action_id,
        discarded_draft_width=metadata.discarded_draft_width,
        aborted_draft_request_count=metadata.aborted_draft_request_count,
        diagnostic=diagnostic,
    )


def build_live_step_record(
    *,
    metadata: KOffSchedulerMetadata,
    evidence: KOffRunnerEvidence,
    raw_generated_lengths: Mapping[str, int],
    committed_lengths: Mapping[str, int],
    invalid_spec_tokens: int,
    elapsed_s: float,
    boot_scope: str = BOOT_SCOPE_MINIMAL_B0,
) -> dict[str, Any]:
    """Close one passive engine-step record or mark it replay-ineligible.

    Args:
        boot_scope: The registered scope. Under ``w98-lattice`` a distinct
            draft realization legitimately carries its own weight version, so
            the step records both instead of asserting they are equal.
    """
    if evidence.verified_action_id != metadata.verified_action_id:
        raise KOffRuntimeError("scheduler/runner verified-action mismatch")
    if evidence.next_action_id != metadata.next_action_id:
        raise KOffRuntimeError("scheduler/runner next-action mismatch")
    abort_metadata = (
        metadata.aborted_action_id,
        metadata.discarded_draft_width,
        metadata.aborted_draft_request_count,
    )
    abort_evidence = (
        evidence.aborted_action_id,
        evidence.discarded_draft_width,
        evidence.aborted_draft_request_count,
    )
    if abort_evidence != abort_metadata:
        raise KOffRuntimeError("scheduler/runner draft-abort mismatch")
    weight_versions_match = (
        evidence.target_weight_version_id == evidence.draft_weight_version_id
    )
    if not weight_versions_match and boot_scope != BOOT_SCOPE_W98_LATTICE:
        raise KOffRuntimeError("target-matching draft weight version diverged")
    if elapsed_s <= 0:
        raise KOffRuntimeError(f"non-positive engine-step time {elapsed_s}")

    decode_ids = metadata.decode_req_ids
    raw_lengths = [raw_generated_lengths.get(req_id, 0) for req_id in decode_ids]
    committed = [committed_lengths.get(req_id, 0) for req_id in decode_ids]
    if any(kept < 0 or kept > raw for kept, raw in zip(committed, raw_lengths)):
        raise KOffRuntimeError("committed token count exceeds generated tokens")

    h_steps = len(decode_ids)
    action = (
        None
        if metadata.verified_action_id is None
        else action_for_id(metadata.verified_action_id)
    )
    d_armed = h_steps if action is not None and action.k else 0
    accepted = sum(max(length - 1, 0) for length in raw_lengths)
    emitted = sum(committed)
    clipped = sum(raw - kept for raw, kept in zip(raw_lengths, committed, strict=True))
    closure = emitted + clipped == accepted + h_steps
    if metadata.aborted_action_id is not None and (d_armed or accepted):
        raise KOffRuntimeError(
            "aborted drafts must contribute neither armed nor accepted counts"
        )

    exclusions: list[str] = []
    if action is None:
        exclusions.append("no_decode_action")
    if not metadata.pure_decode:
        exclusions.append("prefill_or_mixed_batch")
    if metadata.aborted_action_id is not None:
        exclusions.append("aborted_mixed_boundary_draft")
    if any(length == 0 for length in raw_lengths):
        exclusions.append("missing_decode_output")
    if metadata.preemptions:
        exclusions.append("preemption")
    if metadata.recomputed_tokens:
        exclusions.append("recomputation")
    if invalid_spec_tokens:
        exclusions.append("invalid_spec_tokens")
    if not closure:
        exclusions.append("counter_closure")
    if action is not None and accepted > action.k * d_armed:
        exclusions.append("accepted_token_bound")
    if action is OFF_ACTION and (d_armed or accepted):
        exclusions.append("off_accounting")

    target_graph_id = None if action is None else action.target_graph_id
    draft_graph_id = None if action is None else action.draft_graph_id
    return {
        "schema_version": 1,
        "record_type": "koff_engine_step",
        "scored": False,
        "boot_scope": boot_scope,
        # Recorded, not assumed. Under w98-lattice a distinct draft carries its
        # own weight version, so the step states both rather than asserting
        # they are equal.
        "target_matching_weights": weight_versions_match,
        "target_weight_version_id": evidence.target_weight_version_id,
        "draft_weight_version_id": evidence.draft_weight_version_id,
        "engine_step_index": metadata.engine_step_index,
        "target_step_boundary": True,
        "verified_action_id": metadata.verified_action_id,
        "next_action_id": metadata.next_action_id,
        "selection_intent": metadata.selection_intent,
        "aborted_action_id": metadata.aborted_action_id,
        "discarded_draft_width": metadata.discarded_draft_width,
        "aborted_draft_request_count": metadata.aborted_draft_request_count,
        "eligible_for_p3_replay": not exclusions,
        "exclusion_reasons": exclusions,
        "counters": {
            "H_target_steps": h_steps,
            "D_armed": d_armed,
            "A_accepted": accepted,
            "C_clipped": clipped,
            "E_committed": emitted,
            "decode_time_s": elapsed_s,
            "shared_target_kv_blocks_in_use": (metadata.shared_target_kv_blocks_in_use),
            "preemptions": metadata.preemptions,
            "recomputed_tokens": metadata.recomputed_tokens,
            "closure_holds": closure,
        },
        "engine": {
            "total_scheduled_kv_tokens": metadata.total_scheduled_kv_tokens,
            "active_request_count": h_steps,
            "generated_suffix_min": metadata.generated_suffix_min,
            "generated_suffix_max": metadata.generated_suffix_max,
            "target_graph_id": target_graph_id,
            "draft_graph_id": draft_graph_id,
        },
        "resources": {
            "shared_target_kv_block_capacity": (
                metadata.shared_target_kv_block_capacity
            ),
            "binding_id": evidence.binding_id,
            "pool_id": evidence.pool_id,
            "true_slot_mapping_id": evidence.true_slot_mapping_id,
        },
        "execution": asdict(evidence),
    }


def build_same_event_record(
    *,
    capture_id: str,
    metadata: KOffSchedulerMetadata,
    evidence: KOffRunnerEvidence,
    accepted_draft_tokens: Mapping[str, int],
    raw_generated_tokens: Mapping[str, int],
    committed_tokens: Mapping[str, int],
    draft_armed: Mapping[str, bool],
    invalid_spec_tokens: Mapping[str, int],
    elapsed_s: float,
    boot_scope: str = BOOT_SCOPE_MINIMAL_B0,
) -> dict[str, Any]:
    """Build one explicit P4 record from a single synchronous engine event.

    Args:
        boot_scope: The registered scope. Under ``w98-lattice`` the draft is a
            distinct realization by design -- quantized, or with layers replaced
            by passthroughs -- so it legitimately carries its own weight
            version. The record states both versions instead of asserting they
            are equal, exactly as the passive step record does.
    """
    if not capture_id:
        raise KOffRuntimeError("P4 capture id is empty")
    if metadata.verified_action_id is None or not metadata.decode_req_ids:
        raise KOffRuntimeError("P4 event requires an observed target decode row")
    if evidence.verified_action_id != metadata.verified_action_id:
        raise KOffRuntimeError("scheduler/runner verified-action mismatch")
    if evidence.next_action_id != metadata.next_action_id:
        raise KOffRuntimeError("scheduler/runner next-action mismatch")
    # Gate only. The B0 capture schema is frozen with additionalProperties:
    # false, so the observation is NOT written into the record here -- the
    # passive step record already carries target_matching_weights and both
    # version ids. Extending the capture schema is a separate, deliberate
    # phase-97 decision, not a side effect of admitting a new scope.
    if (
        evidence.target_weight_version_id != evidence.draft_weight_version_id
        and boot_scope != BOOT_SCOPE_W98_LATTICE
    ):
        raise KOffRuntimeError("target-matching draft weight version diverged")
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise KOffRuntimeError(f"non-positive P4 engine-event time {elapsed_s}")

    request_ids = metadata.decode_req_ids
    expected_ids = set(request_ids)
    inputs: tuple[tuple[str, Mapping[str, Any]], ...] = (
        ("accepted draft tokens", accepted_draft_tokens),
        ("raw generated tokens", raw_generated_tokens),
        ("committed tokens", committed_tokens),
        ("draft-armed flags", draft_armed),
        ("invalid speculative tokens", invalid_spec_tokens),
    )
    for name, values in inputs:
        actual_ids = set(values)
        if actual_ids != expected_ids:
            raise KOffRuntimeError(
                f"P4 {name} do not exactly cover the scheduler event: "
                f"missing={sorted(expected_ids - actual_ids)}, "
                f"extra={sorted(actual_ids - expected_ids)}"
            )

    action = action_for_id(metadata.verified_action_id)
    rows: list[dict[str, Any]] = []
    for request_id in request_ids:
        accepted = accepted_draft_tokens[request_id]
        raw = raw_generated_tokens[request_id]
        committed = committed_tokens[request_id]
        armed = draft_armed[request_id]
        invalid = invalid_spec_tokens[request_id]
        integer_values = (accepted, raw, committed, invalid)
        if any(type(value) is not int for value in integer_values):
            raise KOffRuntimeError(
                f"P4 request {request_id!r} has a non-integer token counter"
            )
        if type(armed) is not bool:
            raise KOffRuntimeError(
                f"P4 request {request_id!r} has a non-boolean draft flag"
            )
        if accepted < 0 or raw < 1 or committed < 0 or invalid < 0:
            raise KOffRuntimeError(
                f"P4 request {request_id!r} has a negative or empty observation"
            )
        if armed is not bool(action.k):
            raise KOffRuntimeError(
                f"P4 request {request_id!r} draft flag differs from {action.action_id}"
            )
        if accepted > (action.k if armed else 0):
            raise KOffRuntimeError(
                f"P4 request {request_id!r} exceeds its acceptance bound"
            )
        if raw != accepted + 1:
            raise KOffRuntimeError(
                f"P4 request {request_id!r} raw output is not accepted+target"
            )
        if committed > raw:
            raise KOffRuntimeError(
                f"P4 request {request_id!r} commits more tokens than generated"
            )
        rows.append(
            {
                "request_id": request_id,
                "target_processed": True,
                "draft_armed": armed,
                "accepted_draft_tokens": accepted,
                "raw_generated_tokens": raw,
                "committed_tokens": committed,
                "clipped_tokens": raw - committed,
            }
        )

    h_steps = len(rows)
    d_armed = sum(row["draft_armed"] for row in rows)
    accepted = sum(row["accepted_draft_tokens"] for row in rows)
    clipped = sum(row["clipped_tokens"] for row in rows)
    committed = sum(row["committed_tokens"] for row in rows)
    if committed + clipped != accepted + h_steps:
        raise KOffRuntimeError("P4 same-event token closure E+C=A+H failed")

    invalid_total = sum(invalid_spec_tokens.values())
    exclusions: list[str] = []
    if not metadata.pure_decode:
        exclusions.append("prefill_or_mixed_batch")
    if metadata.preemptions:
        exclusions.append("preemption")
    if metadata.recomputed_tokens:
        exclusions.append("recomputation")
    if invalid_total:
        exclusions.append("invalid_spec_tokens")

    event_id = f"{capture_id}:step:{metadata.engine_step_index}"
    return {
        "schema_version": 1,
        "contract_id": "p4-same-event-target-step-v1",
        "event_id": event_id,
        "action_id": action.action_id,
        "k": action.k,
        "engine_step_index": metadata.engine_step_index,
        "complete": True,
        "pure_decode": metadata.pure_decode,
        "source": {
            "scheduler_event_id": event_id,
            "verified_action_id": metadata.verified_action_id,
            "next_action_id": metadata.next_action_id,
            "scheduler_output_observed": True,
            "model_runner_output_observed": True,
            "post_stop_commit_observed": True,
            "prometheus_interval_delta_used": False,
        },
        "quality": {
            "preemptions": metadata.preemptions,
            "recomputed_tokens": metadata.recomputed_tokens,
            "invalid_spec_tokens": invalid_total,
        },
        "timing": {
            "engine_event_elapsed_s": elapsed_s,
            "request_decode_time_s": elapsed_s * h_steps,
            "source": "same_scheduler_event_monotonic",
            "queue_time_included": False,
            "prefill_time_included": False,
        },
        "request_steps": rows,
        "counters": {
            "H_target_steps": h_steps,
            "D_armed": d_armed,
            "A_accepted": accepted,
            "C_clipped": clipped,
            "E_committed": committed,
            "U_unarmed": h_steps - d_armed,
            "closure_holds": True,
            "draft_subset_holds": True,
        },
        "score_eligible": not exclusions,
        "exclusion_reasons": exclusions,
    }


class P4SameEventRecorder:
    """Collect one or 48 same-boot cells under the frozen raw contract."""

    _TOP_LEVEL_FIELDS = {
        "schema_version",
        "capture_contract_id",
        "capture_id",
        "scored",
        "complete",
        "warmup_complete",
        "runner",
        "matrix",
        "generation",
        "events",
    }
    _RUNNER_FIELDS = {
        "preregistration_id",
        "prompt_manifest_id",
        "prompt_manifest_sha256",
        "prompt_bundle_sha256",
        "target_checkpoint_revision",
        "target_quantization",
        "target_kv_dtype",
        "draft_weight_version",
        "shared_kv_binding_id",
        "shared_kv_alias_proven",
        "true_slot_mapping_id",
        "true_slot_identity_proven",
        "hardware_id",
        "parallel_layout",
        "kernel_backend",
        "graph_grade",
        "warmup_policy",
        "measurement_currency",
    }
    _MATRIX_FIELDS = {
        "boot_block_id",
        "boot_id",
        "action_id",
        "action_position",
        "action_realization",
        "regime_id",
        "content_seed",
        "round_index",
    }
    _GENERATION_FIELDS = {
        "prompt_record_ids",
        "generation_seed",
        "batch",
        "max_output_tokens",
        "temperature",
        "ignore_eos",
        "requested_output_tokens",
    }
    _PLAN_FIELDS = {
        "schema_version",
        "capture_plan_contract_id",
        "boot_id",
        "boot_action_id",
        "logical_draft_weight_version",
        "minimum_shared_kv_blocks",
        "cells",
    }
    _SAFE_CAPTURE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

    def __init__(
        self,
        config_path: str,
        output_path: str,
        *,
        boot_action_id: str = "",
        logical_weight_version: str = "",
        minimum_shared_kv_blocks: int = 0,
        boot_scope: str = BOOT_SCOPE_MINIMAL_B0,
    ) -> None:
        self._boot_scope = boot_scope
        self.config_path = Path(config_path)
        self.output_path = Path(output_path)
        if self.config_path.resolve() == self.output_path.resolve():
            raise KOffRuntimeError("P4 capture config and output paths must differ")
        if not self.config_path.is_file():
            raise KOffRuntimeError(
                f"P4 capture config does not exist: {self.config_path}"
            )
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KOffRuntimeError(f"cannot load P4 capture config: {exc}") from exc
        if not isinstance(config, dict):
            raise KOffRuntimeError("P4 capture config must be a JSON object")

        conformance = (
            bool(boot_action_id),
            bool(logical_weight_version),
            minimum_shared_kv_blocks != 0,
        )
        if any(conformance) and not all(conformance):
            raise KOffRuntimeError(
                "P4 recorder requires boot action, logical weight version, and "
                "minimum KV blocks together"
            )
        if all(conformance):
            if boot_action_id not in P4_ACTION_REALIZATIONS:
                raise KOffRuntimeError(f"unknown P4 boot action {boot_action_id!r}")
            if not _P4_LOGICAL_WEIGHT_VERSION.fullmatch(logical_weight_version):
                raise KOffRuntimeError("invalid P4 logical weight version")
            if minimum_shared_kv_blocks != P4_MIN_SHARED_KV_BLOCKS:
                raise KOffRuntimeError("invalid P4 minimum shared-KV block floor")

        is_plan = "capture_plan_contract_id" in config
        if is_plan:
            if not all(conformance):
                raise KOffRuntimeError(
                    "a same-boot capture plan requires validated boot conformance"
                )
            templates = self._validate_plan(
                config,
                boot_action_id=boot_action_id,
                logical_weight_version=logical_weight_version,
                minimum_shared_kv_blocks=minimum_shared_kv_blocks,
            )
            if not self.output_path.is_dir():
                raise KOffRuntimeError(
                    "P4 same-boot plan output must be an existing directory"
                )
            if any(self.output_path.iterdir()):
                raise KOffRuntimeError(
                    f"P4 same-boot output directory is not empty: {self.output_path}"
                )
            output_paths = [
                self.output_path / f"{template['capture_id']}.json"
                for template in templates
            ]
        else:
            expected_action = boot_action_id or None
            templates = [
                self._validate_capture_template(
                    config,
                    expected_action=expected_action,
                    logical_weight_version=logical_weight_version or None,
                )
            ]
            if templates[0]["matrix"]["action_id"] == W512_ACTION_ID and not all(
                conformance
            ):
                raise KOffRuntimeError(
                    "w512 capture requires a validated boot-static contract"
                )
            if not self.output_path.parent.is_dir():
                raise KOffRuntimeError(
                    "P4 capture output parent does not exist: "
                    f"{self.output_path.parent}"
                )
            output_paths = [self.output_path]

        self._templates = templates
        self._output_paths = output_paths
        self._is_plan = is_plan
        self._boot_action_id = boot_action_id or templates[0]["matrix"]["action_id"]
        self._logical_weight_version = logical_weight_version
        self._minimum_shared_kv_blocks = minimum_shared_kv_blocks
        self._cell_index = -1
        self._capture: dict[str, Any] = {}
        self._current_output_path = self.output_path
        self._current_temp_path = self.output_path
        self._frozen_request_ids: tuple[str, ...] = ()
        self._expected_request_ids: set[str] = set()
        self._max_output_tokens = 0
        self._committed_by_request: dict[str, int] = {}
        self._event_ids: set[str] = set()
        self._last_step_index: int | None = None
        self._runtime_identity: tuple[str, str, str, str, str] | None = None
        self._completed_capture_count = 0
        self._finalized = False
        self._activate_cell(0)

    @classmethod
    def _validate_capture_template(
        cls,
        value: Mapping[str, Any],
        *,
        expected_action: str | None,
        logical_weight_version: str | None,
    ) -> dict[str, Any]:
        template = copy.deepcopy(dict(value))
        if set(template) != cls._TOP_LEVEL_FIELDS:
            raise KOffRuntimeError(
                "P4 capture config fields differ from the frozen capture contract"
            )
        if (
            template["schema_version"] != 1
            or template["capture_contract_id"] != "p4-b0-same-event-capture-v1"
            or template["scored"] is not False
            or template["complete"] is not False
            or template["warmup_complete"] is not True
            or template["events"] != []
        ):
            raise KOffRuntimeError(
                "P4 capture config must be a warm, unscored, incomplete empty cell"
            )
        capture_id = template["capture_id"]
        if not isinstance(capture_id, str) or not cls._SAFE_CAPTURE_ID.fullmatch(
            capture_id
        ):
            raise KOffRuntimeError("P4 capture id is empty or is not filename-safe")

        runner = template.get("runner")
        matrix = template.get("matrix")
        generation = template.get("generation")
        if not all(isinstance(value, dict) for value in (runner, matrix, generation)):
            raise KOffRuntimeError(
                "P4 capture runner, matrix, and generation must exist"
            )
        assert isinstance(runner, dict)
        assert isinstance(matrix, dict)
        assert isinstance(generation, dict)
        if (
            set(runner) != cls._RUNNER_FIELDS
            or set(matrix) != cls._MATRIX_FIELDS
            or set(generation) != cls._GENERATION_FIELDS
        ):
            raise KOffRuntimeError(
                "P4 capture nested fields differ from the frozen capture contract"
            )
        if (
            runner["preregistration_id"] != "p4-b0-off-k4-w512-value-screen-v1"
            or runner["prompt_manifest_id"] != "p4-b0-six-regime-prompts-v1"
            or runner["measurement_currency"] != "S_dec"
        ):
            raise KOffRuntimeError("P4 capture runner constants drifted")
        action_id = matrix.get("action_id")
        if action_id not in P4_ACTION_REALIZATIONS:
            raise KOffRuntimeError(f"unknown P4 capture action {action_id!r}")
        if expected_action is not None and action_id != expected_action:
            raise KOffRuntimeError("P4 capture action differs from the boot action")
        if matrix.get("action_realization") != P4_ACTION_REALIZATIONS[action_id]:
            raise KOffRuntimeError("P4 capture action realization is not live B0")
        block_id = matrix.get("boot_block_id")
        position = matrix.get("action_position")
        if block_id not in P4_ACTION_ORDERS:
            raise KOffRuntimeError("P4 capture has an unknown boot block")
        expected_position = P4_ACTION_ORDERS[block_id].index(action_id) + 1
        if position != expected_position:
            raise KOffRuntimeError("P4 capture action position is not canonical")
        if (
            not isinstance(matrix.get("boot_id"), str)
            or not matrix["boot_id"]
            or matrix.get("regime_id") not in P4_REGIME_ORDER
            or matrix.get("content_seed") not in P4_CONTENT_SEEDS
            or matrix.get("round_index") not in P4_ROUNDS
        ):
            raise KOffRuntimeError("P4 capture matrix coordinates are invalid")
        if (
            logical_weight_version is not None
            and runner.get("draft_weight_version") != logical_weight_version
        ):
            raise KOffRuntimeError(
                "P4 capture logical weight version differs from its boot"
            )
        prompt_ids = generation.get("prompt_record_ids")
        if (
            not isinstance(prompt_ids, list)
            or len(prompt_ids) != 32
            or any(not isinstance(value, str) or not value for value in prompt_ids)
            or len(set(prompt_ids)) != len(prompt_ids)
        ):
            raise KOffRuntimeError("P4 capture requires 32 unique prompt record ids")
        max_output_tokens = generation.get("max_output_tokens")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise KOffRuntimeError("P4 capture max_output_tokens must be positive")
        if generation.get("requested_output_tokens") != (
            len(prompt_ids) * max_output_tokens
        ):
            raise KOffRuntimeError("P4 capture requested output work does not close")
        if generation.get("ignore_eos") is not True:
            raise KOffRuntimeError("P4 capture requires ignore_eos=true")
        batch = generation.get("batch")
        if type(batch) is not int or batch < 1 or batch > len(prompt_ids):
            raise KOffRuntimeError("P4 capture batch is outside [1, 32]")
        if (
            runner.get("shared_kv_alias_proven") is not True
            or runner.get("true_slot_identity_proven") is not True
        ):
            raise KOffRuntimeError("P4 capture must require live shared-KV proofs")
        return template

    @classmethod
    def _validate_plan(
        cls,
        plan: Mapping[str, Any],
        *,
        boot_action_id: str,
        logical_weight_version: str,
        minimum_shared_kv_blocks: int,
    ) -> list[dict[str, Any]]:
        if set(plan) != cls._PLAN_FIELDS:
            raise KOffRuntimeError("P4 capture-plan fields differ from its contract")
        if (
            plan["schema_version"] != 1
            or plan["capture_plan_contract_id"] != P4_CAPTURE_PLAN_CONTRACT_ID
            or plan["boot_action_id"] != boot_action_id
            or plan["logical_draft_weight_version"] != logical_weight_version
            or plan["minimum_shared_kv_blocks"] != minimum_shared_kv_blocks
        ):
            raise KOffRuntimeError("P4 capture plan differs from its validated boot")
        boot_id = plan["boot_id"]
        if not isinstance(boot_id, str) or not boot_id:
            raise KOffRuntimeError("P4 capture plan has no boot id")
        cells = plan["cells"]
        if not isinstance(cells, list) or len(cells) != 48:
            raise KOffRuntimeError("P4 same-boot plan requires exactly 48 cells")
        templates = [
            cls._validate_capture_template(
                cell,
                expected_action=boot_action_id,
                logical_weight_version=logical_weight_version,
            )
            for cell in cells
        ]
        coordinates = [
            (
                template["matrix"]["regime_id"],
                template["matrix"]["content_seed"],
                template["matrix"]["round_index"],
            )
            for template in templates
        ]
        expected_coordinates = [
            (regime, seed, round_index)
            for regime in P4_REGIME_ORDER
            for seed in P4_CONTENT_SEEDS
            for round_index in P4_ROUNDS
        ]
        if coordinates != expected_coordinates:
            raise KOffRuntimeError("P4 same-boot cells are not in canonical order")
        if any(template["matrix"]["boot_id"] != boot_id for template in templates):
            raise KOffRuntimeError("P4 capture plan spans more than one boot id")
        capture_ids = [template["capture_id"] for template in templates]
        if len(set(capture_ids)) != len(capture_ids):
            raise KOffRuntimeError("P4 capture plan repeats a capture id")
        block_positions = {
            (
                template["matrix"]["boot_block_id"],
                template["matrix"]["action_position"],
            )
            for template in templates
        }
        if len(block_positions) != 1:
            raise KOffRuntimeError("P4 capture plan spans boot blocks or positions")
        return templates

    def _activate_cell(self, index: int) -> None:
        self._cell_index = index
        self._capture = copy.deepcopy(self._templates[index])
        self._current_output_path = self._output_paths[index]
        self._current_temp_path = self._current_output_path.with_name(
            self._current_output_path.name + ".tmp"
        )
        if self._current_output_path.exists() or self._current_temp_path.exists():
            raise KOffRuntimeError(
                f"refusing to overwrite P4 capture output: {self._current_output_path}"
            )
        prompt_ids = self._capture["generation"]["prompt_record_ids"]
        self._frozen_request_ids = tuple(prompt_ids)
        self._expected_request_ids = set(prompt_ids)
        self._max_output_tokens = self._capture["generation"]["max_output_tokens"]
        self._committed_by_request = dict.fromkeys(prompt_ids, 0)
        try:
            self._current_output_path.open("x", encoding="utf-8").close()
        except FileExistsError as exc:
            raise KOffRuntimeError(
                f"refusing to overwrite P4 capture output: {self._current_output_path}"
            ) from exc

    @property
    def capture_id(self) -> str:
        """Return the configured capture-cell identifier."""
        if self._finalized:
            raise KOffRuntimeError("P4 capture plan is already complete")
        return str(self._capture["capture_id"])

    @property
    def completed_capture_count(self) -> int:
        """Return the number of complete create-new cells written so far."""
        return self._completed_capture_count

    def _normalize_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(dict(event))
        expected_action = self._capture["matrix"]["action_id"]
        source = normalized.get("source")
        if not isinstance(source, dict):
            raise KOffRuntimeError("P4 event has no explicit action source")
        observed = (
            normalized.get("action_id"),
            source.get("verified_action_id"),
            source.get("next_action_id"),
        )
        if expected_action == W512_ACTION_ID:
            if self._boot_action_id != W512_ACTION_ID:
                raise KOffRuntimeError("w512 event is not bound to a w512 boot")
            if observed != (K4_ACTION_ID, K4_ACTION_ID, K4_ACTION_ID):
                raise KOffRuntimeError(
                    "w512 may relabel only an exact live K4-to-K4 event"
                )
            normalized["action_id"] = W512_ACTION_ID
            source["verified_action_id"] = W512_ACTION_ID
            source["next_action_id"] = W512_ACTION_ID
        elif observed != (expected_action, expected_action, expected_action):
            raise KOffRuntimeError(
                "P4 OFF/K4 event action does not exactly match its boot"
            )
        expected_k = 0 if expected_action == OFF_ACTION_ID else 4
        if normalized.get("k") != expected_k:
            raise KOffRuntimeError("P4 event K differs from its capture action")
        rows = normalized.get("request_steps")
        if not isinstance(rows, list) or not rows:
            raise KOffRuntimeError("P4 event has no explicit request rows")
        if any(not isinstance(row, dict) for row in rows) or any(
            row.get("draft_armed") is not bool(expected_k) for row in rows
        ):
            raise KOffRuntimeError("P4 event draft rows differ from its action")
        if normalized.get("score_eligible") is not True:
            raise KOffRuntimeError("P4 capture refuses an ineligible live event")
        return normalized

    def record(
        self,
        event: Mapping[str, Any],
        evidence: KOffRunnerEvidence,
    ) -> None:
        """Append one explicit event and finalize after exact equal work."""
        if self._finalized:
            raise KOffRuntimeError("P4 capture received an event after finalization")
        identity = (
            evidence.binding_id,
            evidence.pool_id,
            evidence.true_slot_mapping_id,
            evidence.shared_weight_binding_id,
            evidence.draft_weight_version_id,
        )
        if (
            evidence.target_weight_version_id != evidence.draft_weight_version_id
            and self._boot_scope != BOOT_SCOPE_W98_LATTICE
        ):
            raise KOffRuntimeError("P4 capture observed divergent target/draft weights")
        if (
            self._logical_weight_version
            and evidence.draft_weight_version_id != self._logical_weight_version
        ):
            raise KOffRuntimeError("P4 capture observed another logical weight version")
        if self._is_plan and not evidence.shared_weight_binding_id:
            raise KOffRuntimeError("P4 capture is missing its live weight binding")
        if self._runtime_identity is None:
            self._runtime_identity = identity
        elif identity != self._runtime_identity:
            raise KOffRuntimeError(
                "P4 same-boot KV, slot, or weight binding identity changed"
            )

        runner = self._capture["runner"]
        runner["draft_weight_version"] = evidence.draft_weight_version_id
        runner["shared_kv_binding_id"] = evidence.binding_id
        runner["shared_kv_alias_proven"] = True
        runner["true_slot_mapping_id"] = evidence.true_slot_mapping_id
        runner["true_slot_identity_proven"] = True
        normalized = self._normalize_event(event)

        event_id = normalized.get("event_id")
        step_index = normalized.get("engine_step_index")
        if not isinstance(event_id, str) or not event_id:
            raise KOffRuntimeError("P4 event has no explicit event id")
        if type(step_index) is not int or step_index < 0:
            raise KOffRuntimeError("P4 event has no explicit engine-step index")
        if event_id in self._event_ids:
            raise KOffRuntimeError(f"duplicate P4 scheduler event {event_id}")
        if self._last_step_index is not None and step_index <= self._last_step_index:
            raise KOffRuntimeError("P4 engine-step indices are not increasing")
        if not event_id.startswith(f"{self.capture_id}:step:"):
            raise KOffRuntimeError("P4 event id differs from the active capture cell")

        rows = normalized["request_steps"]
        row_ids = [row.get("request_id") for row in rows if isinstance(row, dict)]
        if len(row_ids) != len(rows):
            raise KOffRuntimeError("P4 event request rows are missing")
        canonical_ids = canonicalize_p4_request_ids(
            row_ids,
            self._frozen_request_ids,
        )
        for row, request_id in zip(rows, canonical_ids, strict=True):
            row["request_id"] = request_id
        row_ids = canonical_ids
        unexpected = set(row_ids) - self._expected_request_ids
        if unexpected:
            raise KOffRuntimeError(
                f"P4 event contains requests outside the frozen prompt cell: "
                f"{sorted(unexpected)}"
            )
        for row in rows:
            request_id = row["request_id"]
            committed = row.get("committed_tokens")
            if type(committed) is not int or committed < 0:
                raise KOffRuntimeError(
                    f"P4 request {request_id!r} has no explicit commit count"
                )
            new_total = self._committed_by_request[request_id] + committed
            if new_total > self._max_output_tokens:
                raise KOffRuntimeError(
                    f"P4 request {request_id!r} exceeded fixed output work"
                )
            self._committed_by_request[request_id] = new_total

        self._capture["events"].append(normalized)
        self._event_ids.add(event_id)
        self._last_step_index = step_index
        if all(
            value == self._max_output_tokens
            for value in self._committed_by_request.values()
        ):
            self._write_capture(complete=True)

    def close(self) -> None:
        """Emit an incomplete, adapter-rejected capture when work did not close."""
        if not self._finalized and self._capture["events"]:
            self._write_capture(complete=False)

    def _write_capture(self, *, complete: bool) -> None:
        self._capture["complete"] = complete
        payload = json.dumps(self._capture, indent=2, sort_keys=True) + "\n"
        try:
            with self._current_temp_path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(self._current_temp_path, self._current_output_path)
        except FileExistsError as exc:
            raise KOffRuntimeError(
                f"refusing to overwrite P4 capture output: {self._current_output_path}"
            ) from exc
        if complete:
            self._completed_capture_count += 1
        next_index = self._cell_index + 1
        if complete and next_index < len(self._templates):
            self._activate_cell(next_index)
        else:
            self._finalized = True


def make_trace_header(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Create the first record in a non-scored live JSONL trace."""
    return {
        "schema_version": 1,
        "record_type": "koff_runtime_header",
        "created_unix_s": time.time(),
        "scored": False,
        "contract": "phase97-minimal-b0-k4-off",
        "actions": [OFF_ACTION.to_record(), K4_ACTION.to_record()],
        "environment": dict(environment),
    }


class KOffTraceWriter:
    """Crash-tolerant append-only writer for passive live records."""

    def __init__(self, path: str, header: Mapping[str, Any]) -> None:
        self.path = Path(path)
        if not self.path.parent.is_dir():
            raise KOffRuntimeError(
                f"K/OFF trace parent does not exist: {self.path.parent}"
            )
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(header), sort_keys=True) + "\n")
        except FileExistsError as exc:
            raise KOffRuntimeError(
                f"refusing to overwrite existing K/OFF trace: {self.path}"
            ) from exc

    def write(self, record: Mapping[str, Any]) -> None:
        """Append and flush one complete JSON object."""
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(record), sort_keys=True) + "\n")
