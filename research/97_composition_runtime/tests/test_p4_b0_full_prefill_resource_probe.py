"""CPU tests for the Phase 97 full-prefill resource probe."""

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
    AUTHORIZATION_PATH,
    GPU_UUID,
    OUTPUT_PATH,
    SCHEMA_PATH,
    SOURCE_PATHS,
    P4FullPrefillResourceProbeError,
    _probe_spec,
    _reference,
    _resource_result,
    validate_probe_authorization,
)


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


def _authorization_with_current_sources() -> dict:
    authorization = _authorization()
    authorization["source_artifacts"] = {
        role: _reference(path) for role, path in SOURCE_PATHS.items()
    }
    return authorization


class P4B0FullPrefillResourceProbeTests(unittest.TestCase):
    """Prove the narrow one-initialization capacity-probe boundary."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_authorization_is_consumed_and_historical(self) -> None:
        result = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(result["decision"]["state"], "fail")
        self.assertEqual(result["shared_kv_capacity"]["num_gpu_blocks"], 19928)
        self.assertEqual(result["shared_kv_capacity"]["headroom_blocks"], -1754)
        self.assertEqual(result["engine"]["requests_executed"], 0)
        self.assertFalse(result["claims"]["generation_performed"])
        self.assertFalse(result["decision"]["authority_granted"])

        with self.assertRaisesRegex(
            P4FullPrefillResourceProbeError,
            "resource-probe source closure drifted",
        ):
            validate_probe_authorization(_authorization(), require_output_absent=False)

    def test_probe_spec_uses_full_prefill_budget(self) -> None:
        spec, evidence = _probe_spec()

        self.assertEqual(spec["engine"]["max_num_batched_tokens"], 114688)
        self.assertTrue(spec["engine"]["enable_chunked_prefill"])
        self.assertEqual(evidence["max_microbatch_prompt_tokens"], 112908)
        self.assertEqual(evidence["headroom_tokens"], 1748)

    def test_rejects_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            P4FullPrefillResourceProbeError, "source closure drifted"
        ):
            validate_probe_authorization(authorization)

    def test_rejects_value_screen_authority(self) -> None:
        authorization = _authorization()
        authorization["authorizations"]["gpu_value_screen"] = True
        with self.assertRaises(P4FullPrefillResourceProbeError):
            validate_probe_authorization(authorization)

    def test_rejects_fallback_gpu(self) -> None:
        authorization = _authorization()
        authorization["execution_policy"]["fallback_gpu_authorized"] = True
        with self.assertRaises(P4FullPrefillResourceProbeError):
            validate_probe_authorization(authorization)

    def test_rejects_budget_drift(self) -> None:
        authorization = _authorization()
        authorization["probe_contract"]["max_num_batched_tokens"] = 8192
        with self.assertRaises(P4FullPrefillResourceProbeError):
            validate_probe_authorization(authorization)

    def test_result_closes_capacity_arithmetic(self) -> None:
        cache = SimpleNamespace(
            num_gpu_blocks=22000,
            block_size=16,
            kv_cache_size_tokens=352000,
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
        self.assertFalse(result["decision"]["authority_granted"])

    def test_result_records_failure_below_floor(self) -> None:
        cache = SimpleNamespace(
            num_gpu_blocks=21000,
            block_size=16,
            kv_cache_size_tokens=336000,
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

        self.assertEqual(result["decision"]["state"], "fail")
        self.assertFalse(
            result["decision"]["value_screen_may_be_separately_authorized"]
        )

    def test_output_existence_consumes_authorization(self) -> None:
        authorization = _authorization_with_current_sources()
        validate_probe_authorization(authorization, require_output_absent=False)
        with self.assertRaisesRegex(
            P4FullPrefillResourceProbeError, "output already exists"
        ):
            validate_probe_authorization(authorization)

    def test_schema_rejects_extra_authority(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["authorizations"]["production"] = True
        with self.assertRaises(P4FullPrefillResourceProbeError):
            validate_probe_authorization(authorization)


if __name__ == "__main__":
    unittest.main()
