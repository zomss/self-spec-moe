"""CPU-only conformance tests for the held Phase 97 B0 matrix runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from run_p4_b0_value_screen import (  # noqa: E402
    ACTION_ORDERS,
    EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
    FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
    GPU0_UUID,
    GPU4_UUID,
    INPROCESS_ENGINE_CORE_CLASS,
    MEASUREMENT_GPU_MEMORY_UTILIZATION,
    MINIMUM_SHARED_KV_BLOCKS,
    NATIVE_SAMPLER_ENV,
    NATIVE_SAMPLER_VALUE,
    PREFILL_SAMPLED_TOKENS_PER_REQUEST,
    V1_MULTIPROCESSING_ENV,
    V1_MULTIPROCESSING_VALUE,
    V6_AUTHORIZATION_PATH,
    V6_OUTPUT_PATH,
    P4RunnerError,
    _boot_child_environment,
    _capture_cohort_metadata,
    _load_prompt_rows,
    _preflight_gpu4_identity_and_idle,
    _preflight_gpu_identity_and_idle,
    _preflight_inprocess_engine_core,
    _preflight_native_sampler,
    _run_prompt_chunk,
    _total_output_tokens,
    _validate_plan,
    apply_full_prefill_engine_contract,
    build_boot_specs,
    execute_run,
    full_prefill_budget_evidence,
    logical_weight_version,
    prepare_run,
    validate_execution_authority,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization.json"


def _authorization() -> dict:
    package = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    package = apply_full_prefill_engine_contract(package)
    package["run_contract"]["engine"]["gpu_memory_utilization"] = (
        MEASUREMENT_GPU_MEMORY_UTILIZATION
    )
    return package


class P4B0CaptureRunnerTests(unittest.TestCase):
    """Prove exact planning and fail-closed GPU authority on CPU."""

    def test_builds_nine_canonical_same_boot_plans(self) -> None:
        output_dir = Path("/tmp/not-created-by-in-memory-plan")
        authorization = _authorization()
        specs = build_boot_specs(authorization, output_dir)

        self.assertEqual(len(specs), 9)
        self.assertEqual(
            [spec["action_id"] for spec in specs],
            [action for block in ACTION_ORDERS.values() for action in block],
        )
        self.assertEqual(len({spec["boot_id"] for spec in specs}), 9)
        self.assertEqual(
            {spec["logical_draft_weight_version"] for spec in specs},
            {logical_weight_version(authorization)},
        )
        for spec in specs:
            plan = spec["plan"]
            _validate_plan(plan)
            self.assertEqual(len(plan["cells"]), 48)
            self.assertEqual(plan["boot_action_id"], spec["action_id"])
            self.assertEqual(
                spec["environment"][NATIVE_SAMPLER_ENV], NATIVE_SAMPLER_VALUE
            )
            self.assertEqual(
                spec["environment"][V1_MULTIPROCESSING_ENV],
                V1_MULTIPROCESSING_VALUE,
            )
            self.assertEqual(
                spec["engine"]["max_num_batched_tokens"],
                FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
            )
            self.assertTrue(spec["engine"]["enable_chunked_prefill"])
            self.assertEqual(
                spec["engine"]["gpu_memory_utilization"],
                MEASUREMENT_GPU_MEMORY_UTILIZATION,
            )
            self.assertEqual(
                spec["full_prefill_budget"]["max_microbatch_prompt_tokens"],
                EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
            )
            self.assertGreater(
                spec["full_prefill_budget"]["headroom_tokens"],
                0,
            )
            self.assertEqual(
                {cell["generation"]["generation_seed"] for cell in plan["cells"]},
                {0},
            )
            self.assertEqual(plan["minimum_shared_kv_blocks"], MINIMUM_SHARED_KV_BLOCKS)
            self.assertEqual(
                {cell["matrix"]["boot_id"] for cell in plan["cells"]},
                {spec["boot_id"]},
            )

    def test_prepare_only_writes_no_capture_or_gpu_claim(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "prepared"
            specs = prepare_run(_authorization(), output_dir)
            preparation = json.loads(
                (output_dir / "preparation.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(specs), 9)
            self.assertFalse(preparation["gpu_executed"])
            self.assertFalse(preparation["gpu_authority_granted"])
            self.assertEqual(preparation["sampler_backend"], "pytorch_native")
            self.assertFalse(preparation["flashinfer_sampler_enabled"])
            self.assertEqual(
                preparation["engine_core_class"], INPROCESS_ENGINE_CORE_CLASS
            )
            self.assertFalse(preparation["v1_multiprocessing"])
            self.assertTrue(preparation["queue_all_before_first_step"])
            self.assertTrue(
                preparation["full_prefill_budget"]["chunked_prefill_enabled"]
            )
            self.assertEqual(
                preparation["generation_work_contract"],
                {
                    "measurement_currency": "S_dec",
                    "measured_decode_tokens_field": "generation.max_output_tokens",
                    "prefill_sampled_tokens_per_request": 1,
                    "sampling_max_tokens_formula": (
                        "generation.max_output_tokens + "
                        "prefill_sampled_tokens_per_request"
                    ),
                },
            )
            self.assertEqual(
                preparation["full_prefill_budget"]["max_microbatch_prompt_tokens"],
                EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
            )
            self.assertEqual(preparation["physical_boot_count"], 9)
            self.assertEqual(preparation["raw_capture_count"], 432)
            self.assertEqual(len(list((output_dir / "plans").glob("*.json"))), 9)
            capture_dirs = list((output_dir / "captures").iterdir())
            self.assertEqual(len(capture_dirs), 9)
            self.assertTrue(all(not any(path.iterdir()) for path in capture_dirs))

    def test_rejects_the_consumed_chunked_prefill_budget(self) -> None:
        authorization = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(P4RunnerError, "full-prefill budget"):
            build_boot_specs(authorization, Path("/tmp/not-created"))

    def test_budget_is_derived_from_exact_bundle_lengths(self) -> None:
        authorization = _authorization()
        manifest, prompt_rows = _load_prompt_rows()

        evidence = full_prefill_budget_evidence(
            manifest,
            prompt_rows,
            authorization["run_contract"]["engine"],
        )

        self.assertEqual(
            evidence["max_microbatch_prompt_tokens"],
            EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
        )
        self.assertEqual(evidence["effective_scheduler_token_budget"], 114656)
        self.assertEqual(evidence["headroom_tokens"], 1748)
        self.assertEqual(
            evidence["regime_max_microbatch_prompt_tokens"]["R5cot"],
            EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
        )

    def test_budget_rejects_token_bundle_length_drift(self) -> None:
        authorization = _authorization()
        manifest, prompt_rows = _load_prompt_rows()
        prompt_rows = {key: dict(value) for key, value in prompt_rows.items()}
        first_id = manifest["prompts"][0]["record_id"]
        prompt_rows[first_id]["token_ids"] = prompt_rows[first_id]["token_ids"][:-1]

        with self.assertRaisesRegex(P4RunnerError, "token count drifted"):
            full_prefill_budget_evidence(
                manifest,
                prompt_rows,
                authorization["run_contract"]["engine"],
            )

    def test_child_environment_prepends_active_environment_bin(self) -> None:
        executable_dir = Path(sys.prefix) / "bin"
        with mock.patch.dict(
            os.environ,
            {"PATH": "/usr/bin", NATIVE_SAMPLER_ENV: "1"},
        ):
            environment = _boot_child_environment(
                {"P4_TEST_VALUE": 7}, executable_dir=executable_dir
            )

        self.assertEqual(environment["PATH"].split(os.pathsep)[0], str(executable_dir))
        self.assertEqual(environment["P4_TEST_VALUE"], "7")
        self.assertEqual(environment[NATIVE_SAMPLER_ENV], NATIVE_SAMPLER_VALUE)
        self.assertEqual(environment[V1_MULTIPROCESSING_ENV], V1_MULTIPROCESSING_VALUE)

    def test_child_environment_rejects_missing_ninja(self) -> None:
        with (
            TemporaryDirectory() as directory,
            self.assertRaisesRegex(P4RunnerError, "does not provide ninja"),
        ):
            _boot_child_environment({}, executable_dir=Path(directory))

    def test_child_environment_rejects_path_override(self) -> None:
        with self.assertRaisesRegex(P4RunnerError, "must not override PATH"):
            _boot_child_environment({"PATH": "/tmp"})

    def test_child_environment_rejects_flashinfer_sampler_enable(self) -> None:
        with self.assertRaisesRegex(P4RunnerError, "must not enable"):
            _boot_child_environment({NATIVE_SAMPLER_ENV: "1"})

    def test_child_environment_rejects_multiprocess_engine_core(self) -> None:
        with self.assertRaisesRegex(P4RunnerError, "multiprocess EngineCore"):
            _boot_child_environment({V1_MULTIPROCESSING_ENV: "1"})

    def test_native_sampler_preflight_rejects_missing_policy(self) -> None:
        with self.assertRaisesRegex(P4RunnerError, "requires.*=0"):
            _preflight_native_sampler({})

    def test_native_sampler_preflight_proves_binding(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"backend": "native", "flashinfer_enabled": false}\n',
            stderr="",
        )
        with mock.patch(
            "run_p4_b0_value_screen.subprocess.run", return_value=completed
        ) as run:
            evidence = _preflight_native_sampler(
                {NATIVE_SAMPLER_ENV: NATIVE_SAMPLER_VALUE},
                python_executable=Path("/test/python"),
            )

        self.assertEqual(evidence, {"backend": "native", "flashinfer_enabled": False})
        self.assertEqual(run.call_args.args[0][0], "/test/python")
        self.assertEqual(
            run.call_args.kwargs["env"][NATIVE_SAMPLER_ENV],
            NATIVE_SAMPLER_VALUE,
        )

    def test_inprocess_preflight_rejects_missing_policy(self) -> None:
        with self.assertRaisesRegex(P4RunnerError, "requires.*=0"):
            _preflight_inprocess_engine_core({})

    def test_inprocess_preflight_proves_binding(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"engine_core_class": "InprocClient", "v1_multiprocessing": false}\n'
            ),
            stderr="",
        )
        with mock.patch(
            "run_p4_b0_value_screen.subprocess.run", return_value=completed
        ) as run:
            evidence = _preflight_inprocess_engine_core(
                {V1_MULTIPROCESSING_ENV: V1_MULTIPROCESSING_VALUE},
                python_executable=Path("/test/python"),
            )

        self.assertEqual(
            evidence,
            {
                "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
                "v1_multiprocessing": False,
            },
        )
        self.assertEqual(run.call_args.args[0][0], "/test/python")
        self.assertEqual(
            run.call_args.kwargs["env"][V1_MULTIPROCESSING_ENV],
            V1_MULTIPROCESSING_VALUE,
        )

    def test_gpu4_preflight_rejects_active_compute_process(self) -> None:
        identity = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"{GPU4_UUID}\n", stderr=""
        )
        applications = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"{GPU4_UUID}, 1234\n", stderr=""
        )
        with (
            mock.patch(
                "run_p4_b0_value_screen.subprocess.run",
                side_effect=[identity, applications],
            ),
            self.assertRaisesRegex(P4RunnerError, "active compute processes"),
        ):
            _preflight_gpu4_identity_and_idle()

    def test_source_bound_gpu_preflight_selects_exact_physical_index(self) -> None:
        identity = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"{GPU0_UUID}\n", stderr=""
        )
        applications = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with mock.patch(
            "run_p4_b0_value_screen.subprocess.run",
            side_effect=[identity, applications],
        ) as run:
            evidence = _preflight_gpu_identity_and_idle(0, GPU0_UUID)

        self.assertEqual(
            evidence,
            {
                "physical_gpu_index": 0,
                "gpu_uuid": GPU0_UUID,
                "active_compute_processes": 0,
            },
        )
        self.assertIn("--id=0", run.call_args_list[0].args[0])

    def test_source_bound_gpu_preflight_rejects_uuid_drift(self) -> None:
        identity = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"{GPU4_UUID}\n", stderr=""
        )
        applications = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        with (
            mock.patch(
                "run_p4_b0_value_screen.subprocess.run",
                side_effect=[identity, applications],
            ),
            self.assertRaisesRegex(P4RunnerError, "GPU 0 identity drifted"),
        ):
            _preflight_gpu_identity_and_idle(0, GPU0_UUID)

    def test_prompt_chunk_queues_every_request_before_step(self) -> None:
        class Output:
            def __init__(self, request_id: str) -> None:
                self.request_id = request_id
                self.finished = True
                self.outputs = [type("Sample", (), {"token_ids": [1, 2]})()]

        class Engine:
            def __init__(self) -> None:
                self.request_ids: list[str] = []
                self.stepped = False

            def add_request(self, request_id, prompt, params) -> None:
                self.request_ids.append(request_id)

            def get_num_unfinished_requests(self) -> int:
                return 0 if self.stepped else len(self.request_ids)

            def has_unfinished_requests(self) -> bool:
                return not self.stepped

            def step(self):
                self.assert_all_queued()
                self.stepped = True
                return [Output(request_id) for request_id in self.request_ids]

            def assert_all_queued(self) -> None:
                self_test.assertEqual(self.request_ids, ["p0", "p1"])

        self_test = self
        engine = Engine()
        cell = {
            "generation": {
                "temperature": 0.0,
                "max_output_tokens": 1,
                "generation_seed": 0,
            }
        }
        prompt_rows = {
            "p0": {"token_ids": [1, 2]},
            "p1": {"token_ids": [3, 4]},
        }
        with mock.patch(
            "run_p4_b0_value_screen._sampling_params", return_value=object()
        ):
            _run_prompt_chunk(engine, cell, ["p0", "p1"], prompt_rows)

        self.assertTrue(engine.stepped)

    def test_frontend_work_adds_one_unmeasured_prefill_sample(self) -> None:
        measured_by_action = {
            "off": 512,
            "target-matching-k4": 3072,
            "target-matching-w512-masked-k4": 256,
        }
        for action_id, measured_tokens in measured_by_action.items():
            with self.subTest(action_id=action_id):
                cell = {
                    "matrix": {"action_id": action_id},
                    "generation": {"max_output_tokens": measured_tokens},
                }
                self.assertEqual(
                    _total_output_tokens(cell),
                    measured_tokens + PREFILL_SAMPLED_TOKENS_PER_REQUEST,
                )

    def test_prompt_slice_carries_exact_measurement_only_cohort_marker(self) -> None:
        cell = {
            "capture_id": "capture-r4",
            "matrix": {"action_id": "target-matching-k4"},
            "generation": {"max_output_tokens": 512},
        }

        marker = _capture_cohort_metadata(cell, ["p0", "p1"])

        self.assertEqual(marker["contract_id"], "p4-capture-cohort-barrier-v1")
        self.assertIs(marker["measurement_only"], True)
        self.assertEqual(marker["frozen_request_ids"], ["p0", "p1"])
        self.assertEqual(marker["action_id"], "target-matching-k4")
        self.assertEqual(marker["measured_decode_tokens"], 512)
        self.assertEqual(marker["unmeasured_prefill_tokens"], 1)

    def test_prompt_chunk_rejects_measured_only_frontend_output(self) -> None:
        class Output:
            request_id = "p0"
            finished = True
            outputs = [type("Sample", (), {"token_ids": [1]})()]

        class Engine:
            stepped = False

            def add_request(self, request_id, prompt, params) -> None:
                pass

            def get_num_unfinished_requests(self) -> int:
                return 0 if self.stepped else 1

            def has_unfinished_requests(self) -> bool:
                return not self.stepped

            def step(self):
                self.stepped = True
                return [Output()]

        cell = {
            "generation": {
                "temperature": 0.0,
                "max_output_tokens": 1,
                "generation_seed": 0,
            }
        }
        with (
            mock.patch(
                "run_p4_b0_value_screen._sampling_params", return_value=object()
            ),
            self.assertRaisesRegex(P4RunnerError, "exact frontend work"),
        ):
            _run_prompt_chunk(
                Engine(),
                cell,
                ["p0"],
                {"p0": {"token_ids": [1, 2]}},
            )

    def test_execution_preflights_sampler_before_output_creation(self) -> None:
        events: list[str] = []
        child_environment = {
            NATIVE_SAMPLER_ENV: NATIVE_SAMPLER_VALUE,
            V1_MULTIPROCESSING_ENV: V1_MULTIPROCESSING_VALUE,
        }
        with (
            mock.patch("run_p4_b0_value_screen.validate_execution_authority"),
            mock.patch(
                "run_p4_b0_value_screen._boot_child_environment",
                return_value=child_environment,
            ),
            mock.patch(
                "run_p4_b0_value_screen._preflight_native_sampler",
                side_effect=lambda _: events.append("sampler"),
            ),
            mock.patch(
                "run_p4_b0_value_screen._preflight_inprocess_engine_core",
                side_effect=lambda _: events.append("engine_core"),
            ),
            mock.patch(
                "run_p4_b0_value_screen._preflight_gpu4_identity_and_idle",
                side_effect=lambda: events.append("gpu"),
            ),
            mock.patch(
                "run_p4_b0_value_screen.prepare_run",
                side_effect=lambda *_: events.append("prepare") or [],
            ),
            mock.patch(
                "run_p4_b0_value_screen._adapt_and_score",
                side_effect=lambda _: events.append("score"),
            ),
        ):
            execute_run(
                Path(V6_AUTHORIZATION_PATH),
                {},
                Path(V6_OUTPUT_PATH),
            )

        self.assertEqual(
            events,
            ["sampler", "engine_core", "gpu", "prepare", "score"],
        )

    def test_current_hold_package_cannot_execute(self) -> None:
        with self.assertRaisesRegex(P4RunnerError, "HOLD package"):
            validate_execution_authority(_authorization())

    def test_flipping_gpu_flag_cannot_bypass_source_bound_reapproval(self) -> None:
        authorization = _authorization()
        authorization["authorizations"]["gpu_measurement"] = True
        authorization["decision"]["state"] = "authorized"
        authorization["claims"]["executable_run_ready"] = True
        invocation = authorization["run_contract"]["invocation"]
        invocation["runner_exists"] = True
        invocation["launchable_now"] = True
        with self.assertRaisesRegex(P4RunnerError, "HOLD package"):
            validate_execution_authority(authorization)


if __name__ == "__main__":
    unittest.main()
