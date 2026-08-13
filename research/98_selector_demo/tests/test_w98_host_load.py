# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proofs for the W98-R2 per-boot host-load gate.

The regression tests at the bottom are the ones that matter: they replay the
REAL boot logs the thresholds were derived from, so a later edit that widens the
band until a contaminated boot slips through fails here rather than silently in
the campaign.
"""

from pathlib import Path

import pytest
from w98_host_load import (
    CHEAP_REGIMES_BY_BATCH,
    DEFAULT_MAX_ATTEMPTS,
    MIN_RELATIVE_SPREAD,
    QUIET_MAX_CAPTURE_S,
    QUIET_MAX_COMPILE_S,
    BootLoadSignature,
    GpuTelemetry,
    HostLoadError,
    _lane_cpu_ids,
    guarded_boot,
    host_snapshot,
    measurement_verdict,
    require_quiet,
    signature_from_log,
    signature_from_log_path,
)

DATA = Path(__file__).resolve().parent.parent / "data"

QUIET_LOG = """
INFO backends.py:393] Compiling a graph for compile range (1, 8192) takes 5.34 s
INFO backends.py:393] Compiling a graph for compile range (1, 8192) takes 4.49 s
Capturing (PIECEWISE): 100%|##| 37/37 [00:04<00:00,  7.56it/s]
Capturing CUDA graphs (decode, FULL): 100%|##| 25/25 [00:02<00:00, 11.58it/s]
INFO llm_base_proposer.py:1451] Draft chain PIECEWISE: runtime_mode=PIECEWISE
INFO gpu_model_runner.py:7122] Graph capturing finished in 8 secs, took 0.51 GiB
"""

LOUD_LOG = """
INFO backends.py:393] Compiling a graph for compile range (1, 8192) takes 8.69 s
INFO backends.py:393] Compiling a graph for compile range (1, 8192) takes 6.94 s
Capturing (PIECEWISE): 100%|##| 37/37 [00:07<00:00,  5.01it/s]
Capturing CUDA graphs (decode, FULL): 100%|##| 25/25 [00:03<00:00,  6.63it/s]
INFO llm_base_proposer.py:1451] Draft chain PIECEWISE: runtime_mode=PIECEWISE
INFO gpu_model_runner.py:7122] Graph capturing finished in 13 secs, took 0.47 GiB
"""


def test_parses_every_field():
    sig = signature_from_log(QUIET_LOG)
    assert sig.compile_s == pytest.approx(9.83)
    assert sig.capture_s == 8.0
    assert sig.piecewise_it_s == 7.56
    assert sig.full_it_s == 11.58


def test_quiet_and_loud_logs_separate():
    assert signature_from_log(QUIET_LOG).verdict == "quiet"
    assert signature_from_log(LOUD_LOG).verdict == "loud"


def test_reason_names_the_failing_threshold():
    reason = signature_from_log(LOUD_LOG).reason
    assert "compile" in reason and "capture" in reason
    assert signature_from_log(QUIET_LOG).reason is None


def test_require_quiet_raises_only_on_loud():
    require_quiet(signature_from_log(QUIET_LOG), "cell")
    with pytest.raises(HostLoadError, match="loud"):
        require_quiet(signature_from_log(LOUD_LOG), "cell")


def test_missing_markers_fail_closed():
    """A log without the markers must not pass as quiet."""
    sig = signature_from_log("nothing useful here")
    assert sig.verdict == "unknown"
    with pytest.raises(HostLoadError, match="unknown"):
        require_quiet(sig, "cell")


def test_missing_log_file_is_unknown(tmp_path):
    assert signature_from_log_path(tmp_path / "absent.log").verdict == "unknown"


def test_tqdm_rewrites_use_the_final_rate():
    """tqdm re-prints the same line; the completed rate is the last one."""
    partial = (
        "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): "
        "100%|##| 37/37 [00:04<00:00,  9.99it/s]"
        "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): "
        "100%|##| 37/37 [00:04<00:00,  7.56it/s]"
    )
    assert signature_from_log(partial).piecewise_it_s == 7.56


def test_thresholds_sit_between_the_populations():
    """Quiet tops out at 11.29s/9s; loud starts at 15.63s/13s."""
    assert 11.29 < QUIET_MAX_COMPILE_S < 15.63
    assert 9 < QUIET_MAX_CAPTURE_S < 13
    assert DEFAULT_MAX_ATTEMPTS >= 1


def test_boundary_is_inclusive():
    at_limit = BootLoadSignature(
        QUIET_MAX_COMPILE_S, QUIET_MAX_CAPTURE_S, None, None, "piecewise"
    )
    assert at_limit.verdict == "quiet"
    over = BootLoadSignature(QUIET_MAX_COMPILE_S + 0.01, 1.0, None, None, "piecewise")
    assert over.verdict == "loud"


def test_non_piecewise_runtime_is_not_judged():
    """Whole-chain capture time reflects its graph budget, not contention."""
    wholechain = LOUD_LOG.replace(
        "Draft chain PIECEWISE: runtime_mode=PIECEWISE",
        "Draft FULL-CG (q=1) capture sizes augmented with small num_reqs",
    )
    signature = signature_from_log(wholechain)
    assert signature.runtime == "wholechain"
    assert signature.verdict == "not_applicable"
    # Still fails closed: the campaign pins piecewise, so a boot that ran
    # something else is a contract violation rather than a pass.
    with pytest.raises(HostLoadError, match="not_applicable"):
        require_quiet(signature, "cell")


def test_runtime_is_read_from_behaviour_not_configuration():
    """A boot with neither marker is unknown, never assumed piecewise."""
    assert signature_from_log("Graph capturing finished in 8 secs").runtime == "unknown"


def test_lane_cpu_ids_expands_ranges_and_singletons():
    assert _lane_cpu_ids("0-3") == {0, 1, 2, 3}
    assert _lane_cpu_ids("0-1,5") == {0, 1, 5}
    assert _lane_cpu_ids("") == set()


def test_host_snapshot_reports_loadavg_and_optional_lane():
    plain = host_snapshot()
    assert len(plain["loadavg"]) == 3
    assert "foreign_lane_processes" not in plain
    with_lane = host_snapshot("0-15")
    assert isinstance(with_lane["foreign_lane_processes"], list)


def test_record_round_trips_verdict():
    record = signature_from_log(LOUD_LOG).as_record()
    assert record["verdict"] == "loud"
    assert record["compile_s"] == pytest.approx(15.63)


# --- a warm compile cache is normal, not suspicious ---

# Assembled rather than written literally: the marker must sit on ONE log line
# for the parser to see it, and that line is longer than the source limit.
_CACHE_LINE = (
    "INFO backends.py:292] Directly load the compiled graph(s) for "
    "compile range (1, 8192) from the cache, took {} s\n"
)
CACHED_LOG = (
    _CACHE_LINE.format("1.254")
    + _CACHE_LINE.format("1.178")
    + "Capturing (PIECEWISE): 100%|##| 37/37 [00:04<00:00,  7.56it/s]\n"
    + "INFO llm_base_proposer.py:1451] Draft chain PIECEWISE: runtime_mode=PIECEWISE\n"
    + "INFO gpu_model_runner.py:7122] Graph capturing finished in 9 secs\n"
)


def test_warm_cache_is_quiet_not_unknown():
    """The bug that would have aborted the campaign at its own anchors.

    ANCHOR_REPEATS boots each anchor three times per stage, so all but the
    first hit a warm cache by construction. Failing closed on the absent
    compile time rejected them on a perfectly quiet box.
    """
    sig = signature_from_log(CACHED_LOG)
    assert sig.compile_s is None
    assert sig.compile_cached is True
    assert sig.cache_load_s == pytest.approx(2.432)
    assert sig.verdict == "quiet"
    require_quiet(sig, "cell")


def test_warm_cache_still_judged_on_capture():
    """Losing the compile limb must not lose the gate entirely."""
    slow = CACHED_LOG.replace("finished in 9 secs", "finished in 14 secs")
    sig = signature_from_log(slow)
    assert sig.compile_cached is True
    assert sig.verdict == "loud"
    assert "capture" in sig.reason


def test_no_compile_and_no_cache_evidence_stays_unknown():
    """A log this gate does not understand still does not get a pass."""
    mystery = CACHED_LOG.replace("Directly load the compiled", "Something else")
    sig = signature_from_log(mystery)
    assert sig.compile_cached is False
    assert sig.verdict == "unknown"


def test_a_cold_compile_is_not_marked_cached():
    sig = signature_from_log(QUIET_LOG)
    assert sig.compile_cached is False
    assert sig.cache_load_s is None


def test_missing_capture_is_unknown_even_with_compile():
    """A boot killed before capture cannot be judged."""
    truncated = QUIET_LOG.replace(
        "INFO gpu_model_runner.py:7122] Graph capturing finished in 8 secs, "
        "took 0.51 GiB",
        "",
    )
    assert signature_from_log(truncated).verdict == "unknown"


# --- guarded_boot: the campaign runner's whole integration surface ---


def test_guarded_boot_accepts_a_quiet_first_attempt():
    sig, attempts = guarded_boot("cell", lambda attempt: QUIET_LOG)
    assert sig.verdict == "quiet"
    assert len(attempts) == 1


def test_guarded_boot_retries_past_a_loud_box():
    logs = [LOUD_LOG, LOUD_LOG, QUIET_LOG]
    sig, attempts = guarded_boot("cell", lambda attempt: logs[attempt - 1])
    assert sig.verdict == "quiet"
    # Rejected attempts stay in the record; a runner that reported only the
    # survivor would be selecting rather than measuring.
    assert [a["verdict"] for a in attempts] == ["loud", "loud", "quiet"]


def test_guarded_boot_gives_up_and_names_every_attempt():
    with pytest.raises(HostLoadError) as excinfo:
        guarded_boot("mycell", lambda attempt: LOUD_LOG, max_attempts=2)
    message = str(excinfo.value)
    assert "mycell" in message and "#1" in message and "#2" in message


def test_guarded_boot_reports_rejections_for_cleanup():
    seen = []
    guarded_boot(
        "cell",
        lambda attempt: LOUD_LOG if attempt == 1 else QUIET_LOG,
        on_reject=lambda label, attempt, sig: seen.append((label, attempt)),
    )
    assert seen == [("cell", 1)]


def test_guarded_boot_passes_the_attempt_number():
    seen = []
    guarded_boot("cell", lambda attempt: seen.append(attempt) or QUIET_LOG)
    assert seen == [1]


# --- the measurement-shape gate ---


def obs(r1_ms, r8_ms, r6_ms):
    return {
        "R1": {"draft_chain_s": r1_ms / 1000.0},
        "R8": {"draft_chain_s": r8_ms / 1000.0},
        "R6": {"draft_chain_s": r6_ms / 1000.0},
    }


def test_clean_measurement_passes():
    """v6's anchor: rises with batch, healthy spread."""
    verdict = measurement_verdict(obs(30.36, 31.75, 33.92))
    assert verdict.verdict == "clean"
    assert verdict.monotonic is True
    assert verdict.reason is None


def test_inverted_ordering_is_clamped():
    """The aborted G98-C boot: R1 and R8 lifted above R6."""
    verdict = measurement_verdict(obs(38.18, 38.26, 34.54))
    assert verdict.verdict == "clamped"
    assert verdict.monotonic is False
    # Spread alone would have passed this, which is why ordering is checked.
    assert verdict.relative_spread > MIN_RELATIVE_SPREAD
    assert "not increasing in batch" in verdict.reason


def test_collapsed_spread_is_clamped():
    """X17's contaminated skip16 boot: everything on one floor."""
    verdict = measurement_verdict(obs(29.56, 29.69, 29.70))
    assert verdict.verdict == "clamped"
    assert verdict.monotonic is True
    assert "relative spread" in verdict.reason


def test_missing_cheap_regime_fails_closed():
    assert measurement_verdict({"R1": {"draft_chain_s": 0.03}}) is None
    assert measurement_verdict(obs(30.0, 31.0, 32.0) | {"R6": {}}) is None


def test_threshold_sits_between_the_populations():
    """Clean boots bottom out at 0.046; clamped ones top out at 0.022."""
    assert 0.022 < MIN_RELATIVE_SPREAD < 0.046


def test_guarded_boot_rejects_a_clamped_measurement():
    """A boot can pass the startup gate and still be thrown out."""
    seen = []
    logs = [QUIET_LOG, QUIET_LOG]
    checks = ["cheap regimes are not increasing in batch", None]
    signature, attempts = guarded_boot(
        "cell",
        lambda attempt: logs[attempt - 1],
        on_reject=lambda label, attempt, sig: seen.append(attempt),
        inspect=lambda attempt: checks[attempt - 1],
    )
    assert signature.verdict == "quiet"
    assert seen == [1]
    # The rejection is recorded as clamped, not misfiled as a loud box.
    assert attempts[0]["verdict"] == "clamped"
    assert attempts[1]["verdict"] == "quiet"


def test_guarded_boot_does_not_inspect_a_loud_boot():
    """No point judging the shape of a measurement already known bad."""
    calls = []
    with pytest.raises(HostLoadError):
        guarded_boot(
            "cell",
            lambda attempt: LOUD_LOG,
            max_attempts=2,
            inspect=lambda attempt: calls.append(attempt) or None,
        )
    assert calls == []


# --- GPU telemetry: forensic, never gating ---


def make_telemetry(samples):
    telem = GpuTelemetry.__new__(GpuTelemetry)
    telem.samples = samples
    return telem


def sample(sm=1980.0, power=400.0, temp=40.0, util=100.0, throttle=0):
    return {
        "sm_mhz": sm,
        "power_w": power,
        "temp_c": temp,
        "util_pct": util,
        "throttle": throttle,
    }


def test_telemetry_names_only_interesting_throttle_bits():
    """GpuIdle fires constantly and means nothing here; power caps do."""
    telem = make_telemetry(
        [sample(throttle=0x1), sample(throttle=0x4), sample(throttle=0x81)]
    )
    reasons = telem.summary()["throttle_reasons"]
    assert "SW_POWER_CAP" in reasons
    assert "HW_POWER_BRAKE" in reasons
    assert not any("IDLE" in r.upper() for r in reasons)


def test_telemetry_reports_the_clock_floor():
    """A cap shows up as a depressed minimum, not a depressed median."""
    telem = make_telemetry([sample(sm=1980.0)] * 9 + [sample(sm=1200.0)])
    summary = telem.summary()
    assert summary["sm_mhz_median"] == 1980.0
    assert summary["sm_mhz_min"] == 1200.0
    assert summary["n"] == 10


def test_telemetry_is_empty_when_nothing_sampled():
    assert make_telemetry([]).summary() == {}


def test_telemetry_stays_off_the_measurement_lane():
    """It spawns nvidia-smi, so it must never share the lane's CPUs."""
    from w98_host_load import TELEMETRY_CPUS, _lane_cpu_ids

    assert not (TELEMETRY_CPUS & _lane_cpu_ids("0-15"))


# --- regressions against the recorded boots the thresholds came from ---
#
# These read the CHECKED-IN signature artifact, not the logs. Boot logs are
# gitignored (`*.log`), so a test that globbed them would pass here and fail on
# a fresh checkout, where the campaign directories exist (their JSON is tracked)
# but every log is absent.

SIGNATURES = DATA / "host_load_signatures.json"


def _recorded():
    import json

    return json.loads(SIGNATURES.read_text(encoding="utf-8"))["boots"]


@pytest.mark.skipif(not SIGNATURES.is_file(), reason="signature artifact absent")
def test_every_scored_v6_boot_is_quiet():
    """Every scored v6 boot ran on a quiet box; none may read as loud."""
    v6 = {k: v for k, v in _recorded().items() if k.startswith("g98_b_v6/")}
    assert v6, "no v6 boots recorded"
    assert {v["verdict"] for v in v6.values()} == {"quiet"}, v6


@pytest.mark.skipif(not SIGNATURES.is_file(), reason="signature artifact absent")
def test_x17_contamination_is_detected():
    """X17: the w256 boot was clean, the other two were host-starved."""
    boots = _recorded()
    assert boots["probe_skip16/target-matching_w256_skip16"]["verdict"] == "quiet"
    assert boots["probe_skip16/target-matching_woff_skip16"]["verdict"] == "loud"
    assert boots["probe_skip16/w4a16-quantized_w256_skip16"]["verdict"] == "loud"


@pytest.mark.skipif(not SIGNATURES.is_file(), reason="signature artifact absent")
def test_wholechain_boots_are_not_judged_by_the_piecewise_band():
    """The 86 s whole-chain capture is a graph budget, not contention."""
    boots = _recorded()
    wc = boots["probe_wholechain_v2/r0_skip4_wcon"]
    assert wc["capture_s"] > 60 and wc["runtime"] == "wholechain"
    assert wc["verdict"] == "not_applicable"


CALIBRATION = DATA / "measurement_gate_calibration.json"


@pytest.mark.skipif(not CALIBRATION.is_file(), reason="calibration artifact absent")
def test_measurement_gate_separates_the_labelled_boots():
    """The gate's whole justification, kept honest.

    25 clean boots and 4 known-clamped ones. If a later tweak trades a catch
    for a false positive, or the other way, this fails and the trade has to be
    argued rather than absorbed.
    """
    import json

    boots = json.loads(CALIBRATION.read_text())["boots"]
    clean = {k: v for k, v in boots.items() if v["label"] == "clean"}
    clamped = {k: v for k, v in boots.items() if v["label"] == "clamped"}
    assert len(clean) == 25 and len(clamped) == 4
    false_positives = {k for k, v in clean.items() if v["verdict"] != "clean"}
    missed = {k for k, v in clamped.items() if v["verdict"] != "clamped"}
    assert not false_positives, false_positives
    assert not missed, missed


@pytest.mark.skipif(not CALIBRATION.is_file(), reason="calibration artifact absent")
def test_neither_limb_of_the_gate_is_redundant():
    """Ordering and spread each catch cases the other misses."""
    import json

    boots = json.loads(CALIBRATION.read_text())["boots"]
    clamped = [v for v in boots.values() if v["label"] == "clamped"]
    caught_by_order = [v for v in clamped if not v["monotonic"]]
    caught_by_spread = [
        v for v in clamped if v["monotonic"] and v["relative_spread"] < 0.03
    ]
    assert caught_by_order, "ordering limb catches nothing; it would be dead code"
    assert caught_by_spread, "spread limb catches nothing; it would be dead code"
    assert len(caught_by_order) + len(caught_by_spread) == len(clamped)


@pytest.mark.skipif(not CALIBRATION.is_file(), reason="calibration artifact absent")
def test_calibration_thresholds_match_the_module():
    import json

    thresholds = json.loads(CALIBRATION.read_text())["thresholds"]
    assert thresholds["min_relative_spread"] == MIN_RELATIVE_SPREAD
    assert thresholds["cheap_regimes_by_batch"] == list(CHEAP_REGIMES_BY_BATCH)


@pytest.mark.skipif(not SIGNATURES.is_file(), reason="signature artifact absent")
def test_recorded_thresholds_match_the_module():
    import json

    thresholds = json.loads(SIGNATURES.read_text(encoding="utf-8"))["thresholds"]
    assert thresholds["quiet_max_compile_s"] == QUIET_MAX_COMPILE_S
    assert thresholds["quiet_max_capture_s"] == QUIET_MAX_CAPTURE_S


@pytest.mark.skipif(not SIGNATURES.is_file(), reason="signature artifact absent")
def test_signature_keys_are_unique_paths():
    """Keys are data-root-relative; every campaign has its own `singles/`."""
    boots = _recorded()
    assert "g98_b_v6/singles/target-matching_woff_skip0" in boots
    suffix = "/singles/target-matching_woff_skip0"
    assert sum(1 for k in boots if k.endswith(suffix)) > 1


def test_live_logs_still_parse_to_the_recorded_verdict():
    """Guard the parser against drift where the logs are still present.

    Only logs older than the snapshot are compared. A probe running while the
    tests run keeps appending to its log, so a newer log legitimately parses to
    a different verdict than the snapshot recorded mid-write -- that is the
    artifact being stale, not the parser drifting.
    """
    if not SIGNATURES.is_file():
        pytest.skip("signature artifact absent")
    snapshot_mtime = SIGNATURES.stat().st_mtime
    checked = 0
    for key, recorded in _recorded().items():
        log = DATA / f"{key}.log"
        if not log.is_file() or log.stat().st_mtime > snapshot_mtime:
            continue
        assert signature_from_log_path(log).verdict == recorded["verdict"], key
        checked += 1
    if not checked:
        pytest.skip("no boot logs present in this checkout")
