"""CPU tests for the Phase 97 full-prefill resource-repair probe."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from probe_p4_b0_full_prefill_resource import (  # noqa: E402
    P4FullPrefillResourceProbeError,
)
from probe_p4_b0_full_prefill_resource import (  # noqa: E402
    _probe_spec as _base_probe_spec,
)
from probe_p4_b0_full_prefill_resource_repair import (  # noqa: E402
    AUTHORIZATION_PATH,
    GPU_UUID,
    OUTPUT_PATH,
    PROBE_GPU_MEMORY_UTILIZATION,
    SCHEMA_PATH,
    SERVING_CHUNKED_PREFILL_BOUNDARY,
    SERVING_MAX_NUM_BATCHED_TOKENS,
    SOURCE_PATHS,
    _probe_spec,
    _reference,
    _resource_result,
    validate_repair_probe_authorization,
)


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


def _authorization_with_current_sources() -> dict:
    authorization = _authorization()
    authorization["source_artifacts"] = {
        role: _reference(path) for role, path in SOURCE_PATHS.items()
    }
    return authorization


class P4B0FullPrefillResourceRepairProbeTests(unittest.TestCase):
    """Prove the narrow memory repair and serving-boundary separation."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_authorization_is_consumed_and_historical(self) -> None:
        result = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(result["decision"]["state"], "pass")
        self.assertEqual(result["shared_kv_capacity"]["num_gpu_blocks"], 22113)
        self.assertEqual(result["shared_kv_capacity"]["headroom_blocks"], 431)
        self.assertEqual(result["engine"]["requests_executed"], 0)
        with self.assertRaisesRegex(
            P4FullPrefillResourceProbeError,
            "resource-repair source closure drifted",
        ):
            validate_repair_probe_authorization(
                _authorization(), require_output_absent=False
            )

    def test_probe_changes_only_memory_utilization(self) -> None:
        base, _ = _base_probe_spec()
        repaired, evidence = _probe_spec()
        changed = {
            key
            for key in repaired["engine"]
            if repaired["engine"][key] != base["engine"][key]
        }

        self.assertEqual(changed, {"gpu_memory_utilization"})
        self.assertEqual(
            repaired["engine"]["gpu_memory_utilization"],
            PROBE_GPU_MEMORY_UTILIZATION,
        )
        self.assertEqual(repaired["engine"]["max_num_batched_tokens"], 114688)
        self.assertTrue(repaired["engine"]["enable_chunked_prefill"])
        self.assertEqual(evidence["max_microbatch_prompt_tokens"], 112908)

    def test_real_serving_chunked_prefill_boundary_is_preserved(self) -> None:
        authorization = _authorization()

        self.assertEqual(
            authorization["serving_chunked_prefill_boundary"],
            SERVING_CHUNKED_PREFILL_BOUNDARY,
        )
        self.assertEqual(
            SERVING_CHUNKED_PREFILL_BOUNDARY["max_num_batched_tokens"],
            SERVING_MAX_NUM_BATCHED_TOKENS,
        )
        self.assertTrue(SERVING_CHUNKED_PREFILL_BOUNDARY["chunked_prefill_enabled"])
        self.assertFalse(
            SERVING_CHUNKED_PREFILL_BOUNDARY["measurement_override_is_serving_default"]
        )

    def test_rejects_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            P4FullPrefillResourceProbeError,
            "resource-repair source closure drifted",
        ):
            validate_repair_probe_authorization(
                authorization, require_output_absent=False
            )

    def test_current_source_closure_is_still_consumed_by_output(self) -> None:
        authorization = _authorization_with_current_sources()

        validate_repair_probe_authorization(authorization, require_output_absent=False)
        with self.assertRaisesRegex(
            P4FullPrefillResourceProbeError,
            "resource-repair output already exists",
        ):
            validate_repair_probe_authorization(authorization)

    def test_rejects_memory_utilization_drift(self) -> None:
        authorization = _authorization()
        authorization["probe_contract"]["gpu_memory_utilization"] = 0.95
        with self.assertRaises(P4FullPrefillResourceProbeError):
            validate_repair_probe_authorization(
                authorization, require_output_absent=False
            )

    def test_rejects_serving_budget_promotion(self) -> None:
        authorization = _authorization()
        boundary = authorization["serving_chunked_prefill_boundary"]
        boundary["max_num_batched_tokens"] = 114688
        boundary["measurement_override_is_serving_default"] = True
        with self.assertRaises(P4FullPrefillResourceProbeError):
            validate_repair_probe_authorization(
                authorization, require_output_absent=False
            )

    def test_rejects_value_screen_authority(self) -> None:
        authorization = _authorization()
        authorization["authorizations"]["gpu_value_screen"] = True
        with self.assertRaises(P4FullPrefillResourceProbeError):
            validate_repair_probe_authorization(
                authorization, require_output_absent=False
            )

    def test_rejects_fallback_gpu(self) -> None:
        authorization = _authorization()
        authorization["execution_policy"]["fallback_gpu_authorized"] = True
        with self.assertRaises(P4FullPrefillResourceProbeError):
            validate_repair_probe_authorization(
                authorization, require_output_absent=False
            )

    def test_result_closes_capacity_and_preserves_serving_boundary(self) -> None:
        cache = SimpleNamespace(
            num_gpu_blocks=22000,
            block_size=16,
            kv_cache_size_tokens=352000,
            gpu_memory_utilization=PROBE_GPU_MEMORY_UTILIZATION,
        )
        engine = SimpleNamespace(
            vllm_config=SimpleNamespace(cache_config=cache),
        )
        _, budget = _probe_spec()

        result = _resource_result(
            engine,
            _authorization(),
            {
                "physical_index": 4,
                "uuid": GPU_UUID,
                "name": "NVIDIA H100 80GB HBM3",
                "memory_mib": 81559,
            },
            budget,
        )

        self.assertEqual(result["decision"]["state"], "pass")
        self.assertEqual(result["shared_kv_capacity"]["headroom_blocks"], 318)
        self.assertEqual(
            result["engine"]["gpu_memory_utilization"],
            PROBE_GPU_MEMORY_UTILIZATION,
        )
        self.assertEqual(
            result["serving_chunked_prefill_boundary"],
            SERVING_CHUNKED_PREFILL_BOUNDARY,
        )
        self.assertFalse(result["claims"]["serving_configuration_validated"])
        self.assertFalse(result["decision"]["authority_granted"])

    def test_schema_rejects_extra_authority(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["authorizations"]["production"] = True
        with self.assertRaises(P4FullPrefillResourceProbeError):
            validate_repair_probe_authorization(
                authorization, require_output_absent=False
            )


if __name__ == "__main__":
    unittest.main()
