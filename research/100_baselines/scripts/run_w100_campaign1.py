#!/usr/bin/env python3
"""W100 campaign 1: the seven engine-ready arms on the registered grid.

Barrier: `data/registration/w100_barrier.json` (c90a0b8200eddd98,
commit 4157ee777). Protocol: `w100_prereg.md`, enforced via
`w100_protocol` gates. Prompts: the frozen file only.

## Arms

| arm | engine |
| --- | --- |
| stock       | no speculative_config, no self-spec env (denominator) |
| off         | self-spec loaded, K schedule pinned to 0 |
| base        | target-matching draft, K=4, no levers |
| w4a16       | W4A16-INT4 draft, K=4 |
| w1024       | target-matching + window 1024 + 16 sinks |
| knap8       | target-matching + skip layers 2,4,6,7,8,9,10,16 (G98-F) |
| magicdec512 | target-matching + window 512 + 16 sinks (their budget) |

Lever env semantics copied from the phase-98 round-1 runner
(`boot_environment`), geometry re-pinned for W100: max_model_len 40960,
max_num_seqs = the group's batch, K schedule [[1, 64, K]]. Profiler and
koff trace OFF -- wall clock is the score and the instrument costs
+2.4 ms/step (X25).

## Structure

One subprocess boot per (arm, batch group); each boot runs every cell
whose sweep contains that batch. Groups run in order 8, 16, 32, 64, 1
(the b1 LO drain is the whale; everything else lands first), stock first
within each group so the identity reference exists before any arm needs
it. Completed boots are skipped on resume (delete a JSON to re-run it).

Per (cell, batch): the registered n = min(cell_max, max(16, 2b)) frozen
prompts are submitted at once; the score is wall-clock to drain; EOS is
respected (caps are safety nets). Gates applied inline: boot_gate before
every spawn; identity_gate and cap_rule per cell as results land;
measurement_gate per arm once its four SS points exist, with one
automatic re-run on failure. SS step-ms for the gate is defined as
wall_s / (total_out_tokens / batch) * 1000 -- time per full-batch decode
round, monotone-increasing in batch on a clean box.

Violations are recorded in the result files and the campaign summary;
a violated cell is never silently retried into a pass.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE = SCRIPT_DIR.parent
REG = PHASE / "data" / "registration"
OUT = PHASE / "data" / "campaign1"
sys.path.insert(0, str(SCRIPT_DIR))

TARGET_SNAPSHOT = (
    "/h/v-sukmincho/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
W4A16_CKPT = "/h/v-sukmincho/ckpts/Qwen3-8B-W4A16-INT4"
G98F_PREDICTIONS = (PHASE.parent / "98_selector_demo" / "data" / "g98_f"
                    / "f_predictions.json")
KNAP8 = "2,4,6,7,8,9,10,16"

MAX_MODEL_LEN = 40_960
MAX_NUM_BATCHED_TOKENS = 8192
GPU_MEM_UTIL = 0.90
K_ARMED = 4
WINDOW_SINKS = 16

LANES = {
    "h103": {"physical_gpu_index": 7,
             "physical_gpu_uuid": "GPU-6da30cbb-5974-e66f-a406-389946903db7",
             "cpu_affinity": "96-111"},
    "h104": {"physical_gpu_index": 7,
             "physical_gpu_uuid": "GPU-7a8308d6-2a78-2929-99f2-11d2f11a45f3",
             "cpu_affinity": "96-111"},
}

ARMS = ("stock", "off", "base", "w4a16", "w1024", "knap8", "magicdec512")
# arm -> (quant, window, skip_layers, k)
ARM_CFG = {
    "off": ("target-matching", 0, "", 0),
    "base": ("target-matching", 0, "", K_ARMED),
    "w4a16": ("w4a16-quantized", 0, "", K_ARMED),
    "w1024": ("target-matching", 1024, "", K_ARMED),
    "knap8": ("target-matching", 0, KNAP8, K_ARMED),
    "magicdec512": ("target-matching", 512, "", K_ARMED),
}
QUANT_CKPT = {"target-matching": TARGET_SNAPSHOT,
              "w4a16-quantized": W4A16_CKPT}

BATCH_GROUPS = (8, 16, 32, 64, 1)
# cell -> batches (mirror of the registered clamps)
CELL_BATCHES = {"LI": (1, 8, 16), "LO": (1, 8, 16, 32),
                "LIO": (1, 8, 16), "SS": (1, 8, 32, 64)}
CAPS = {"LI": 2048, "LO": 32_768, "LIO": 4096, "SS": 1024}
LO_T1 = {"batch": 8, "n": 16, "temperature": 1.0, "seed": 1234}

BOOT_RETRY_S = 300
BOOT_RETRIES = 12


def _lane():
    host = socket.gethostname().split(".")[0]
    if host not in LANES:
        raise RuntimeError(f"host {host!r} has no registered lane")
    return host, LANES[host]


def _load_frozen_prompts() -> dict[str, list[list[int]]]:
    man = json.loads((REG / "w100_prompt_manifest.json").read_text())
    gz = REG / "w100_prompts.jsonl.gz"
    digest = hashlib.sha256(gz.read_bytes()).hexdigest()
    if digest != man["prompt_file"]["sha256"]:
        raise RuntimeError("frozen prompt file hash mismatch vs manifest")
    cells: dict[str, list[list[int]]] = {}
    with gzip.open(gz, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            cells.setdefault(row["cell"], []).append(
                (row["prompt_index"], row["token_ids"]))
    return {c: [ids for _, ids in sorted(rows)] for c, rows in cells.items()}


def _boot_plan(batch: int) -> list[dict]:
    import w100_protocol as P
    plan = [{"cell": c, "batch": batch, "n": P.n_for(c, batch),
             "temperature": 0.0, "seed": None, "cap": CAPS[c]}
            for c in ("SS", "LI", "LIO", "LO") if batch in CELL_BATCHES[c]]
    if batch == LO_T1["batch"]:
        plan.append({"cell": "LO", "batch": batch, "n": LO_T1["n"],
                     "temperature": LO_T1["temperature"],
                     "seed": LO_T1["seed"], "cap": CAPS["LO"],
                     "label": "LO_T1"})
    return plan


def _worker_env(arm: str, lane: dict) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("VLLM", "CUDA_VISIBLE"))}
    env.update({
        "CUDA_VISIBLE_DEVICES": str(lane["physical_gpu_index"]),
        "VLLM_CACHE_ROOT": "/tmp/v-sukmincho-w100/vllm-cache/campaign1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_DISABLED_KERNELS": "MacheteLinearKernel",
    })
    if arm == "stock":
        return env
    quant, window, skip, _ = ARM_CFG[arm]
    env.update({
        "VLLM_SELF_SPEC_BOOT_SCOPE": "w100-grid",
        "VLLM_SELF_SPEC_SHARED_KV": "1",
        "VLLM_SELF_SPEC_SHARE_WEIGHTS": (
            "1" if quant == "target-matching" else "0"),
        "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": str(window),
        "VLLM_SELF_SPEC_DRAFT_KV_SINKS": (
            str(WINDOW_SINKS) if window else "0"),
        "VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS": skip,
        "VLLM_SELF_SPEC_DRAFT_WHOLECHAIN": "0",
        "VLLM_SELF_SPEC_DRAFT_FULLCG": "0",
    })
    return env


# ---------------------------- worker ----------------------------------

def worker(arm: str, batch: int, out_path: Path) -> None:
    host, lane = _lane()
    lo, hi = (int(x) for x in lane["cpu_affinity"].split("-"))
    os.sched_setaffinity(0, set(range(lo, hi + 1)))

    prompts_by_cell = _load_frozen_prompts()
    plan = _boot_plan(batch)

    from vllm import LLM, SamplingParams
    kwargs = dict(model=TARGET_SNAPSHOT, max_model_len=MAX_MODEL_LEN,
                  max_num_seqs=batch,
                  max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
                  enable_chunked_prefill=True,
                  gpu_memory_utilization=GPU_MEM_UTIL,
                  enable_prefix_caching=False, enforce_eager=False,
                  seed=0, disable_log_stats=True)
    if arm != "stock":
        quant, _, _, k = ARM_CFG[arm]
        kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": QUANT_CKPT[quant],
            "num_speculative_tokens": K_ARMED,
            "num_speculative_tokens_per_batch_size": [[1, 64, k]],
            "draft_tensor_parallel_size": 1,
        }
    llm = LLM(**kwargs)

    record = {"arm": arm, "batch": batch, "host": host,
              "boot_unix": time.time(), "cells": {}}
    # One warmup per boot: shapes and allocator, not scored.
    llm.generate([{"prompt_token_ids": prompts_by_cell["SS"][0]}],
                 SamplingParams(temperature=0.0, max_tokens=8))
    for item in plan:
        cell, n = item["cell"], item["n"]
        label = item.get("label", cell)
        toks = prompts_by_cell[cell][:n]
        sp = SamplingParams(
            temperature=item["temperature"], max_tokens=item["cap"],
            **({"seed": item["seed"]} if item["seed"] is not None else {}))
        reqs = [{"prompt_token_ids": t} for t in toks]
        t0 = time.perf_counter()
        outs = llm.generate(reqs, sp)
        wall = time.perf_counter() - t0
        rows = []
        for i, o in enumerate(outs):
            c = o.outputs[0]
            rows.append({"prompt_index": i,
                         "prompt_toks": len(o.prompt_token_ids),
                         "out_toks": len(c.token_ids),
                         "finish": c.finish_reason,
                         "sha256": hashlib.sha256(
                             c.text.encode()).hexdigest()})
        total_out = sum(r["out_toks"] for r in rows)
        record["cells"][label] = {
            "cell": cell, "n": n, "wall_s": round(wall, 3),
            "total_out_toks": total_out,
            "temperature": item["temperature"], "seed": item["seed"],
            "cap": item["cap"], "requests": rows,
        }
        print(f"[worker {arm} b{batch}] {label}: wall={wall:.1f}s "
              f"out={total_out}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1))
    print(f"[worker {arm} b{batch}] -> {out_path.name}", flush=True)


# --------------------------- orchestrator -----------------------------

def _gpu_idle(lane: dict) -> tuple[bool, str]:
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    row = {int(r.split(",")[0]): r for r in rows}[lane["physical_gpu_index"]]
    _, uuid, mem = (f.strip() for f in row.split(","))
    if uuid != lane["physical_gpu_uuid"]:
        return False, f"uuid mismatch {uuid}"
    if int(mem) >= 1024:
        return False, f"{mem} MiB in use"
    return True, "idle"


def _boot_path(arm: str, batch: int) -> Path:
    return OUT / f"{arm}__b{batch}.json"


def _apply_cell_gates(arm: str, batch: int, record: dict,
                      stock_record: dict | None) -> list[str]:
    import w100_protocol as P
    violations = []
    for label, e in record["cells"].items():
        finishes = [r["finish"] for r in e["requests"]]
        try:
            e["cap_report"] = P.cap_rule(e["cell"], finishes)
        except P.ProtocolViolation as exc:
            violations.append(str(exc))
            e["cap_report"] = {"violation": str(exc)}
        if (arm != "stock" and stock_record is not None
                and e["temperature"] == 0.0):
            ref = {r["prompt_index"]: r["sha256"]
                   for r in stock_record["cells"][label]["requests"]}
            got = {r["prompt_index"]: r["sha256"] for r in e["requests"]}
            try:
                P.identity_gate(label, ref, got)
                e["identity"] = "match"
            except P.ProtocolViolation as exc:
                violations.append(str(exc))
                e["identity"] = str(exc)
    return violations


def _ss_step_ms(record: dict) -> float:
    e = record["cells"]["SS"]
    return e["wall_s"] / (e["total_out_toks"] / record["batch"]) * 1000.0


def orchestrate() -> None:
    host, lane = _lane()
    import w100_protocol as P
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[campaign1] {m}", flush=True)
    ss_retried: set[str] = set()

    boots = [(b, a) for b in BATCH_GROUPS for a in ARMS]
    passes = 0
    while True:
        pending = [(b, a) for b, a in boots
                   if not _boot_path(a, b).exists()
                   and not (OUT / f"{a}__gate_failure.json").exists()]
        if not pending:
            break
        passes += 1
        if passes > 3:
            raise RuntimeError(f"boots still pending after 3 passes: "
                               f"{pending}")
        _run_boots(pending, lane, P, log, ss_retried)
    _summarize()


def _run_boots(pending, lane, P, log, ss_retried) -> None:
    for batch, arm in pending:
        target = _boot_path(arm, batch)
        if target.exists():
            continue
        for attempt in range(BOOT_RETRIES):
            idle, why = _gpu_idle(lane)
            if idle:
                try:
                    snapshot = P.boot_gate(lane["cpu_affinity"])
                    break
                except P.ProtocolViolation as exc:
                    why = str(exc)
            log(f"lane not clean for {arm} b{batch}: {why}; "
                f"retry in {BOOT_RETRY_S}s")
            time.sleep(BOOT_RETRY_S)
        else:
            raise RuntimeError(f"lane never came clean for {arm} b{batch}")

        log(f"boot {arm} b{batch}")
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--worker", "--arm", arm, "--batch", str(batch),
             "--out", str(target)],
            env=_worker_env(arm, lane), cwd=str(PHASE.parents[1]))
        if proc.returncode != 0 or not target.exists():
            raise RuntimeError(
                f"worker {arm} b{batch} failed rc={proc.returncode}")
        record = json.loads(target.read_text())
        record["host_snapshot"] = snapshot
        record["boot_wall_s"] = round(time.monotonic() - t0, 1)

        stock_rec = None
        if arm != "stock":
            stock_rec = json.loads(_boot_path("stock", batch).read_text())
        violations = _apply_cell_gates(arm, batch, record, stock_rec)
        record["violations"] = violations
        target.write_text(json.dumps(record, indent=1))
        for v in violations:
            log(f"VIOLATION {arm} b{batch}: {v}")

        # Measurement gate once this arm's four SS points exist.
        ss_batches = [b for b in CELL_BATCHES["SS"]
                      if _boot_path(arm, b).exists()]
        if len(ss_batches) == len(CELL_BATCHES["SS"]):
            steps = {b: _ss_step_ms(json.loads(
                _boot_path(arm, b).read_text())) for b in ss_batches}
            try:
                gate = P.measurement_gate(steps)
                log(f"measurement gate {arm}: clean {gate['step_ms']}")
            except P.ProtocolViolation as exc:
                if arm not in ss_retried:
                    ss_retried.add(arm)
                    log(f"measurement gate {arm} FAILED ({exc}); "
                        "deleting this arm's boots for one re-run")
                    for b in CELL_BATCHES["SS"]:
                        _boot_path(arm, b).unlink(missing_ok=True)
                else:
                    log(f"measurement gate {arm} FAILED twice: {exc}; "
                        "arm marked contaminated")
                    (OUT / f"{arm}__gate_failure.json").write_text(
                        json.dumps({"arm": arm, "error": str(exc),
                                    "step_ms": steps}, indent=1))


def _summarize() -> None:
    rows = []
    recs = {}
    for p in sorted(OUT.glob("*__b*.json")):
        r = json.loads(p.read_text())
        recs[(r["arm"], r["batch"])] = r
    for (arm, batch), r in sorted(recs.items()):
        if arm == "stock":
            continue
        stock = recs.get(("stock", batch))
        if not stock:
            continue
        for label, e in r["cells"].items():
            se = stock["cells"][label]
            rows.append({
                "cell": label, "batch": batch, "arm": arm,
                "speedup_vs_stock": round(se["wall_s"] / e["wall_s"], 4),
                "wall_s": e["wall_s"], "stock_wall_s": se["wall_s"],
                "identity": e.get("identity"),
                "cap_hits": e.get("cap_report", {}).get("cap_hits"),
            })
    (OUT / "campaign1_summary.json").write_text(
        json.dumps({"rows": rows}, indent=1))
    print(f"[campaign1] summary: {len(rows)} scored cells -> "
          f"campaign1_summary.json", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.worker:
        worker(args.arm, args.batch, args.out)
    else:
        # Provenance check: the knapsack set must match the G98-F barrier.
        pred = json.loads(G98F_PREDICTIONS.read_text())
        registered = json.dumps(pred).count(KNAP8)
        if not registered:
            raise RuntimeError("knapsack_k8 set drifted from G98-F barrier")
        orchestrate()


if __name__ == "__main__":
    main()
