# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identity, naming and integrity for measurement artifacts.

Written after a filename collision silently dropped a cell from a scored
grid. The full-grid run reported 31 cells while measuring 30: boots were
named by their LEVER values, and two grid entries share every lever -- the
OFF cell and the unlevered armed cell (`target-matching/woff/skip0`) --
differing only in whether the draft is armed. Both mapped to one filename,
and because a completed boot is skipped rather than overwritten, whichever
ran second vanished without an error.

Three properties of that failure are what this module exists to prevent, and
each is worse than the plain collision:

1. **It was silent.** The reported cell count came from the plan, not from
   the records on disk, so the summary asserted a coverage the data did not
   have.
2. **It resolved differently per round.** Round 1 reverses the boot order, so
   the OFF cell won the name in one round and the armed cell won it in the
   other -- the same filename holding different configurations.
3. **Repair by filename made it worse.** Renaming "the OFF file" assumed the
   first round's semantics held in both, which mislabelled an armed
   measurement as OFF. Only a consistency check caught it.

So the rule enforced here is: **a record's identity lives in its content, a
name is only a hint, and a plan must prove its names are unique before any
measurement runs.**
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# Every axis that can distinguish two configurations. A name built from a
# subset of these is exactly the bug this module exists to prevent, so the
# key is built from all of them and unknown extras are rejected rather than
# ignored.
AXES = ("action", "quant", "window", "skip_count")
OFF_ACTION = "off"
IDENTITY_FIELD = "cell"
CONFIG_FIELD = "config"


class ArtifactError(RuntimeError):
    """Raised when an artifact's identity cannot be established or trusted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def canonical(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The identity-bearing subset of a configuration.

    Keys starting with an underscore are treated as run bookkeeping (repeat
    index and the like) and excluded, so they cannot change identity.
    """
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    unknown = set(out) - set(AXES)
    _require(
        not unknown,
        f"configuration carries axes this module does not know how to name: "
        f"{sorted(unknown)}. Add them to AXES rather than letting them go "
        f"unnamed, or the same collision returns.",
    )
    return out


def cell_key(cfg: Mapping[str, Any]) -> str:
    """Canonical identity of a configuration.

    The arming action is part of the identity, not decoration: OFF and the
    unlevered armed cell carry identical levers and are different
    measurements.
    """
    clean = canonical(cfg)
    if clean.get("action") == OFF_ACTION:
        return OFF_ACTION
    window = clean.get("window", "off")
    window = "woff" if window == "off" else f"w{window}"
    return f"{clean.get('quant')}/{window}/skip{clean.get('skip_count')}"


def slug(cfg: Mapping[str, Any], repeat: int | None = None) -> str:
    """Filesystem-safe stem derived from the identity, never from a subset."""
    stem = cell_key(cfg).replace("/", "_")
    return stem if repeat is None else f"{stem}__r{repeat}"


def plan(cells: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map slug to configuration, proving the naming is injective.

    Raises:
        ArtifactError: If two configurations claim one slug, naming both.
            This is the check that would have failed the original grid at
            plan time, before a single boot ran.
    """
    out: dict[str, dict[str, Any]] = {}
    for cfg in cells:
        name = slug(cfg)
        if name in out and canonical(out[name]) != canonical(cfg):
            raise ArtifactError(
                f"slug collision on {name!r}: {canonical(out[name])} and "
                f"{canonical(cfg)} would share one file"
            )
        out[name] = dict(cfg)
    return out


def envelope(
    cfg: Mapping[str, Any], record_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Wrap a measurement so its identity travels inside the file."""
    return {
        "schema_version": 1,
        "record_type": record_type,
        CONFIG_FIELD: dict(canonical(cfg)),
        IDENTITY_FIELD: cell_key(cfg),
        **dict(payload),
    }


def read(path: Path) -> dict[str, Any]:
    """Load a record and check it is self-consistent."""
    with Path(path).open(encoding="utf-8") as handle:
        record = json.load(handle)
    _require(isinstance(record, dict), f"{path}: not a JSON object")
    _require(CONFIG_FIELD in record, f"{path}: record carries no config")
    declared = record.get(IDENTITY_FIELD)
    derived = cell_key(record[CONFIG_FIELD])
    _require(
        declared == derived,
        f"{path}: identity {declared!r} does not match its own config "
        f"({derived!r}). The file's content is authoritative; do not repair "
        f"such a record by renaming it.",
    )
    return record


def claim(path: Path, cfg: Mapping[str, Any]) -> bool:
    """Decide whether an intended measurement is already on disk.

    Returns True when `path` already holds THIS configuration's measurement,
    so the boot may be skipped. A file whose content is a different
    configuration raises instead of being silently accepted -- the original
    bug skipped exactly such a file.
    """
    path = Path(path)
    if not path.is_file():
        return False
    record = read(path)
    want = cell_key(cfg)
    _require(
        record[IDENTITY_FIELD] == want,
        f"{path} holds {record[IDENTITY_FIELD]!r} but was claimed for "
        f"{want!r}: refusing to skip a boot on a mismatched file",
    )
    return True


def audit(
    roots: Iterable[Path],
    expected: Iterable[Mapping[str, Any]],
    replicates: int = 1,
) -> dict[str, Any]:
    """Check a measured tree against the set it was supposed to cover.

    Counts come from the RECORDS, never from the plan, because the defect
    that motivated this module was a summary that trusted the plan.

    Args:
        roots: Directories holding records, one per replicate round.
        expected: The configurations the run intended to measure.
        replicates: How many records each cell should have in total.

    Returns:
        A report with per-cell counts and any missing, extra or duplicated
        identities. `ok` is True only if the coverage is exactly right.
    """
    want = {cell_key(cfg) for cfg in expected}
    seen: dict[str, int] = {}
    mismatched: list[str] = []
    for root in roots:
        for path in sorted(Path(root).glob("*.json")):
            if path.name.endswith(".telemetry.json") or path.name == "summary.json":
                continue
            try:
                record = read(path)
            except ArtifactError as exc:
                mismatched.append(str(exc))
                continue
            if slug(record[CONFIG_FIELD]) != path.stem.split("__")[0]:
                mismatched.append(
                    f"{path}: name says {path.stem!r}, content says "
                    f"{slug(record[CONFIG_FIELD])!r}"
                )
            seen[record[IDENTITY_FIELD]] = seen.get(record[IDENTITY_FIELD], 0) + 1
    missing = sorted(want - set(seen))
    extra = sorted(set(seen) - want)
    wrong_count = {c: n for c, n in sorted(seen.items()) if n != replicates}
    return {
        "record_type": "w98_artifact_audit",
        "expected_cells": len(want),
        "observed_cells": len(seen),
        "records": sum(seen.values()),
        "replicates_expected": replicates,
        "missing": missing,
        "unexpected": extra,
        "wrong_replicate_count": wrong_count,
        "identity_mismatches": mismatched,
        "ok": not (missing or extra or wrong_count or mismatched),
    }
