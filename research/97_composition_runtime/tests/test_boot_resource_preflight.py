"""CPU-only tests for the Phase 97 boot resource preflight."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from plan_boot_class import (  # noqa: E402
    ResourcePreflightError,
    load_schema,
    plan_boot_classes,
    validate_candidate,
    validate_environment,
    validate_workload,
)

DATA_DIR = PHASE_DIR / "data" / "preflight"


def load_artifact(name: str) -> dict:
    """Load one checked-in preflight artifact."""
    return json.loads((DATA_DIR / name).read_text())


def valid_inputs() -> tuple[dict, dict, dict, dict]:
    """Return independent environment, workload, B0, and B1 inputs."""
    return (
        load_artifact("environment_qwen3_8b_h100_tp1.json"),
        load_artifact("workload_rl_capacity_v1.json"),
        load_artifact("candidate_b0_measured.json"),
        load_artifact("candidate_b1_projected.json"),
    )


def make_exact_b1(candidate: dict) -> dict:
    """Convert the projected fixture into synthetic exact B1 evidence."""
    gib = 1024**3
    candidate = copy.deepcopy(candidate)
    candidate["evidence"]["grade"] = "measured_exact"
    candidate["evidence"]["candidate_realization_match"] = True
    capacity = candidate["evidence"]["capacity"]
    capacity["measurement_relation"] = "exact_candidate"
    capacity["available_shared_target_kv_blocks"] = 23000

    terms = candidate["resource_terms"]
    terms["co_resident_weight_hbm_bytes"] = {
        "value_bytes": 4 * gib,
        "grade": "measured",
        "source": "synthetic exact test evidence",
        "included_in_capacity_evidence": True,
    }
    terms["co_resident_graphs_and_workspaces_hbm_bytes"] = {
        "value_bytes": gib,
        "grade": "measured",
        "source": "synthetic exact test evidence",
        "included_in_capacity_evidence": True,
    }
    terms["weight_refresh_peak_hbm_bytes"] = {
        "value_bytes": gib,
        "grade": "measured",
        "source": "synthetic exact test evidence",
        "included_in_capacity_evidence": True,
    }
    terms["weight_refresh_pinned_host_bytes"] = {
        "value_bytes": gib,
        "grade": "measured",
        "source": "synthetic exact test evidence",
    }
    return candidate


class ResourcePreflightSchemaTests(unittest.TestCase):
    """Validate schema documents and checked-in artifacts."""

    def test_schema_documents_are_valid(self) -> None:
        for name in (
            "environment.schema.json",
            "workload.schema.json",
            "boot_candidate.schema.json",
        ):
            with self.subTest(schema=name):
                self.assertEqual(
                    load_schema(name)["$schema"].split("/")[-1],
                    "schema",
                )

    def test_checked_in_artifacts_pass(self) -> None:
        environment, workload, b0, b1 = valid_inputs()
        validate_environment(environment)
        validate_workload(environment, workload)
        validate_candidate(b0)
        validate_candidate(b1)

    def test_assumed_workload_cannot_be_scored(self) -> None:
        environment, workload, _, _ = valid_inputs()
        workload["scored"] = True
        with self.assertRaises(ResourcePreflightError):
            validate_workload(environment, workload)

    def test_trace_cannot_escape_hard_bound(self) -> None:
        environment, workload, _, _ = valid_inputs()
        workload["live_kv"]["frozen_trace_tokens"].append(320001)
        with self.assertRaises(ResourcePreflightError):
            validate_workload(environment, workload)


class ResourcePreflightDecisionTests(unittest.TestCase):
    """Exercise admission, unknown-resource, and capacity gates."""

    def test_measured_b0_is_admitted(self) -> None:
        environment, workload, b0, _ = valid_inputs()
        result = plan_boot_classes(environment, workload, [b0])
        decision = result["decisions"][0]
        self.assertEqual(decision["decision"], "admit")
        self.assertEqual(decision["required_live_kv_blocks"], 21000)
        self.assertEqual(decision["shared_target_kv_headroom_blocks"], 3529)

    def test_projected_b1_fails_closed_on_unknowns(self) -> None:
        environment, workload, _, b1 = valid_inputs()
        result = plan_boot_classes(environment, workload, [b1])
        decision = result["decisions"][0]
        codes = {reason["code"] for reason in decision["reasons"]}
        self.assertEqual(decision["decision"], "reject")
        self.assertTrue(
            {
                "capacity_evidence_not_exact",
                "unknown_co_resident_weight_hbm",
                "unknown_graphs_and_workspaces_hbm",
                "unknown_weight_refresh_peak_hbm",
                "unknown_weight_refresh_pinned_host",
            }.issubset(codes)
        )
        self.assertEqual(decision["shared_target_kv_headroom_blocks"], 928)

    def test_capacity_shortfall_rejects_b0(self) -> None:
        environment, workload, b0, _ = valid_inputs()
        live_kv = workload["live_kv"]
        live_kv["high_percentile_tokens"] = 392464
        live_kv["hard_admission_tokens"] = 392464
        live_kv["frozen_trace_tokens"][-1] = 392464
        result = plan_boot_classes(environment, workload, [b0])
        decision = result["decisions"][0]
        codes = {reason["code"] for reason in decision["reasons"]}
        self.assertEqual(decision["decision"], "reject")
        self.assertIn("insufficient_shared_kv_capacity", codes)
        self.assertEqual(decision["shared_target_kv_headroom_blocks"], -1000)

    def test_synthetic_exact_b1_can_be_admitted(self) -> None:
        environment, workload, _, b1 = valid_inputs()
        result = plan_boot_classes(
            environment,
            workload,
            [make_exact_b1(b1)],
        )
        self.assertEqual(result["decisions"][0]["decision"], "admit")

    def test_unreserved_b1_graph_memory_rejects(self) -> None:
        environment, workload, _, b1 = valid_inputs()
        b1 = make_exact_b1(b1)
        term = b1["resource_terms"]["co_resident_graphs_and_workspaces_hbm_bytes"]
        term["included_in_capacity_evidence"] = False
        result = plan_boot_classes(environment, workload, [b1])
        codes = {reason["code"] for reason in result["decisions"][0]["reasons"]}
        self.assertIn("graphs_and_workspaces_not_reserved", codes)

    def test_pinned_host_budget_rejects(self) -> None:
        environment, workload, _, b1 = valid_inputs()
        b1 = make_exact_b1(b1)
        host_term = b1["resource_terms"]["weight_refresh_pinned_host_bytes"]
        host_term["value_bytes"] = (
            environment["hardware"]["pinned_host_budget_bytes"] + 1
        )
        result = plan_boot_classes(environment, workload, [b1])
        codes = {reason["code"] for reason in result["decisions"][0]["reasons"]}
        self.assertIn("pinned_host_budget_exceeded", codes)

    def test_ordinary_inference_does_not_require_refresh_workspace(self) -> None:
        environment, workload, _, b1 = valid_inputs()
        b1 = make_exact_b1(b1)
        b1["resource_terms"]["weight_refresh_peak_hbm_bytes"] = {
            "value_bytes": None,
            "grade": "unknown",
            "source": "not used after ordinary-inference initialization",
            "included_in_capacity_evidence": False,
        }
        b1["resource_terms"]["weight_refresh_pinned_host_bytes"] = {
            "value_bytes": None,
            "grade": "unknown",
            "source": "not used after ordinary-inference initialization",
        }
        workload["mode"] = "ordinary_inference"
        workload["rl_actor_sync"] = None
        result = plan_boot_classes(environment, workload, [b1])
        self.assertEqual(result["decisions"][0]["decision"], "admit")

    def test_kv_block_contract_mismatch_rejects(self) -> None:
        environment, workload, b0, _ = valid_inputs()
        b0["evidence"]["capacity"]["kv_bytes_per_block"] += 1
        result = plan_boot_classes(environment, workload, [b0])
        decision = result["decisions"][0]
        codes = {reason["code"] for reason in decision["reasons"]}
        self.assertIn("kv_block_bytes_mismatch", codes)
        self.assertIsNone(decision["shared_target_kv_headroom_blocks"])

    def test_environment_mismatch_rejects(self) -> None:
        environment, workload, b0, _ = valid_inputs()
        b0["fixed_environment_id"] = "another-environment"
        result = plan_boot_classes(environment, workload, [b0])
        codes = {reason["code"] for reason in result["decisions"][0]["reasons"]}
        self.assertIn("fixed_environment_mismatch", codes)


if __name__ == "__main__":
    unittest.main()
