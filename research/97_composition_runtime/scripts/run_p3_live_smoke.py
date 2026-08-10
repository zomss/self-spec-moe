#!/usr/bin/env python3
"""Run one Phase 97 minimal-B0 live correctness smoke or its AR oracle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-8B"
MODEL = os.environ.get("P3_MODEL", MODEL_ID)
LONG_REQUEST_ID = "long"
SHORT_REQUEST_ID = "short"
PROMPTS = {
    LONG_REQUEST_ID: (
        "Continue the sequence and explain its rule in one concise paragraph: "
        "2, 4, 6, 8."
    ),
    SHORT_REQUEST_ID: "Write a brief description of a clear daytime sky.",
}
MAX_TOKENS = {LONG_REQUEST_ID: 40, SHORT_REQUEST_ID: 3}
K4_ACTION = "target-matching-k4"
OFF_ACTION = "off"
SPEC_SCHEDULES = {
    "dynamic": [[1, 1, 4], [2, 2, 0]],
    "off": [[1, 2, 0]],
    # K4 is unreachable at max_num_seqs=2 but keeps the resident graph pool
    # identical to the dynamic K4/OFF boot for a fixed-OFF control.
    "off_pool": [[1, 2, 0], [3, 3, 4]],
    "k4": [[1, 2, 4]],
}


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _stable_k4_record(path: Path) -> dict[str, Any] | None:
    for record in reversed(_read_trace(path)):
        if (
            record.get("record_type") == "koff_engine_step"
            and record.get("eligible_for_p3_replay")
            and record.get("verified_action_id") == K4_ACTION
            and record.get("next_action_id") == K4_ACTION
        ):
            return record
    return None


def _sampling_params(request_id: str, logprobs: int | None):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS[request_id],
        ignore_eos=True,
        seed=0,
        logprobs=logprobs,
    )


def _engine_args(mode: str, spec_schedule: str):
    from vllm import EngineArgs

    speculative_config = None
    if mode == "spec":
        speculative_config = {
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": 4,
            "num_speculative_tokens_per_batch_size": SPEC_SCHEDULES[spec_schedule],
            "draft_tensor_parallel_size": 1,
        }
    return EngineArgs(
        model=MODEL,
        speculative_config=speculative_config,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.80,
        enable_prefix_caching=False,
        disable_log_stats=True,
        async_scheduling=False,
        enforce_eager=False,
        seed=0,
        generation_config="vllm",
    )


def _serialize_logprobs(logprobs: Any) -> list[list[dict[str, Any]]] | None:
    if logprobs is None:
        return None
    return [
        [
            {
                "token_id": token_id,
                "logprob": entry.logprob,
                "rank": entry.rank,
                "decoded_token": entry.decoded_token,
            }
            for token_id, entry in sorted(
                position.items(), key=lambda item: item[1].rank or 0
            )
        ]
        for position in logprobs
    ]


def _update_outputs(
    request_outputs: list[Any], outputs: dict[str, dict[str, Any]]
) -> None:
    for request_output in request_outputs:
        if not request_output.outputs:
            continue
        completion = request_output.outputs[0]
        outputs[request_output.request_id] = {
            "token_ids": list(completion.token_ids),
            "text": completion.text,
            "finished": bool(request_output.finished),
            "finish_reason": completion.finish_reason,
            "logprobs": _serialize_logprobs(completion.logprobs),
        }


def _step_until_done(engine: Any, outputs: dict[str, dict[str, Any]]) -> int:
    steps = 0
    while engine.has_unfinished_requests():
        _update_outputs(engine.step(), outputs)
        steps += 1
        if steps > 256:
            raise RuntimeError("live smoke exceeded 256 engine steps")
    return steps


def _run_spec(
    engine: Any,
    trace_path: Path,
    logprobs: int | None,
    inject_after_long_tokens: int | None,
    skip_short: bool,
) -> tuple[dict[str, Any], int]:
    outputs: dict[str, dict[str, Any]] = {}
    engine.add_request(
        LONG_REQUEST_ID,
        PROMPTS[LONG_REQUEST_ID],
        _sampling_params(LONG_REQUEST_ID, logprobs),
    )

    warm_steps = 0
    stable_record = None
    if inject_after_long_tokens is None:
        while stable_record is None:
            _update_outputs(engine.step(), outputs)
            warm_steps += 1
            stable_record = _stable_k4_record(trace_path)
            if warm_steps > 32:
                raise RuntimeError("did not reach a stable eligible K4 step")
            if outputs.get(LONG_REQUEST_ID, {}).get("finished"):
                raise RuntimeError("long request finished before K4 became stable")
    else:
        while len(outputs.get(LONG_REQUEST_ID, {}).get("token_ids", ())) < (
            inject_after_long_tokens
        ):
            _update_outputs(engine.step(), outputs)
            warm_steps += 1
            if warm_steps > 256:
                raise RuntimeError("spec control did not reach its injection point")

    injected_after_tokens = len(outputs[LONG_REQUEST_ID]["token_ids"])
    if not skip_short:
        engine.add_request(
            SHORT_REQUEST_ID,
            PROMPTS[SHORT_REQUEST_ID],
            _sampling_params(SHORT_REQUEST_ID, logprobs),
        )
    tail_steps = _step_until_done(engine, outputs)
    injection = {
        "after_engine_step_index": (
            stable_record["engine_step_index"] if stable_record else None
        ),
        "after_long_output_tokens": injected_after_tokens,
        "trigger": (
            "first eligible stable K4-to-K4 record"
            if inject_after_long_tokens is None
            else "explicit long-output-token count"
        ),
        "short_request_added": not skip_short,
    }
    return {"requests": outputs, "injection": injection}, warm_steps + tail_steps


def _run_ar(
    engine: Any,
    inject_after_long_tokens: int | None,
    logprobs: int | None,
    skip_short: bool,
) -> tuple[dict[str, Any], int]:
    outputs: dict[str, dict[str, Any]] = {}
    engine.add_request(
        LONG_REQUEST_ID,
        PROMPTS[LONG_REQUEST_ID],
        _sampling_params(LONG_REQUEST_ID, logprobs),
    )
    steps = 0
    if inject_after_long_tokens is not None:
        if inject_after_long_tokens < 1:
            raise ValueError("AR injection token count must be positive")
        while len(outputs.get(LONG_REQUEST_ID, {}).get("token_ids", ())) < (
            inject_after_long_tokens
        ):
            _update_outputs(engine.step(), outputs)
            steps += 1
            if steps > 256:
                raise RuntimeError("AR oracle did not reach its injection point")
    if not skip_short:
        engine.add_request(
            SHORT_REQUEST_ID,
            PROMPTS[SHORT_REQUEST_ID],
            _sampling_params(SHORT_REQUEST_ID, logprobs),
        )
    steps += _step_until_done(engine, outputs)
    injection = {
        "after_long_output_tokens": inject_after_long_tokens,
        "trigger": "matched switched-engine arrival point",
        "short_request_added": not skip_short,
    }
    return {"requests": outputs, "injection": injection}, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("spec", "ar"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-after-long-tokens", type=int)
    parser.add_argument("--logprobs", type=int)
    parser.add_argument(
        "--spec-schedule", choices=tuple(SPEC_SCHEDULES), default="dynamic"
    )
    parser.add_argument("--skip-short", action="store_true")
    args = parser.parse_args()

    trace_path_value = os.environ.get("VLLM_SELF_SPEC_KOFF_TRACE", "")
    trace_path = Path(trace_path_value) if trace_path_value else None
    if args.mode == "spec" and trace_path is None:
        raise RuntimeError("spec mode requires VLLM_SELF_SPEC_KOFF_TRACE")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    from vllm import LLMEngine

    engine = LLMEngine.from_engine_args(_engine_args(args.mode, args.spec_schedule))
    try:
        if args.mode == "spec":
            assert trace_path is not None
            run_result, engine_steps = _run_spec(
                engine,
                trace_path,
                args.logprobs,
                args.inject_after_long_tokens,
                args.skip_short,
            )
        else:
            run_result, engine_steps = _run_ar(
                engine,
                args.inject_after_long_tokens,
                args.logprobs,
                args.skip_short,
            )
    finally:
        engine.engine_core.shutdown()

    expected_ids = {LONG_REQUEST_ID}
    if not args.skip_short:
        expected_ids.add(SHORT_REQUEST_ID)
    actual_ids = set(run_result["requests"])
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"missing request outputs: expected {expected_ids}, got {actual_ids}"
        )
    for request_id in expected_ids:
        expected_length = MAX_TOKENS[request_id]
        result = run_result["requests"][request_id]
        if not result["finished"]:
            raise RuntimeError(f"request {request_id} did not finish")
        if len(result["token_ids"]) != expected_length:
            raise RuntimeError(
                f"request {request_id} returned {len(result['token_ids'])} "
                f"tokens, expected {expected_length}"
            )

    result = {
        "schema_version": 1,
        "record_type": "p3_live_smoke_output",
        "scored": False,
        "mode": args.mode,
        "model": MODEL,
        "model_id": MODEL_ID,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "engine": {
            "tensor_parallel_size": 1,
            "max_model_len": 2048,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.80,
            "enforce_eager": False,
            "async_scheduling": False,
            "use_v2_model_runner_env": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
            "engine_steps_observed": engine_steps,
        },
        "speculative_config": (
            {
                "method": "draft_model",
                "model": MODEL,
                "num_speculative_tokens": 4,
                "num_speculative_tokens_per_batch_size": SPEC_SCHEDULES[
                    args.spec_schedule
                ],
            }
            if args.mode == "spec"
            else None
        ),
        "prompts": {request_id: PROMPTS[request_id] for request_id in expected_ids},
        "sampling": {
            request_id: {
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "seed": 0,
                "logprobs": args.logprobs,
            }
            for request_id, max_tokens in MAX_TOKENS.items()
            if request_id in expected_ids
        },
        **run_result,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "mode": args.mode, "output": str(args.output)}))


if __name__ == "__main__":
    main()
