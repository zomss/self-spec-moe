#!/usr/bin/env python3
"""W100 pilot: measured natural-length distributions for the final grid.

Runs the four cells of `final_eval_design.md` on stock decode (no
speculative_config, no VLLM_SELF_SPEC env) and records, per request:
prompt tokens, output tokens, finish reason (stop = natural EOS,
length = cap hit), and the sha256 of the output text (the future
cross-arm identity gate needs a reference). Lengths at T=0 are
scheduler-independent, so this pilot needs no host-load verdict; rates
it prints are informational only.

Outputs `data/pilot/pilot_lengths.json` and a per-cell summary with a
KV-feasibility recomputation from measured p95 lengths.

Lane: the registered phase-98 lane (GPU 7, NUMA-1 cores 96-111) on
h103/h104, fail-closed on host, device UUID, and device idleness. Never
touches other processes: a busy lane aborts the run.
"""
from __future__ import annotations

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
OUT_DIR = PHASE / "data" / "pilot"

TARGET_SNAPSHOT = (
    "/h/v-sukmincho/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
MAX_MODEL_LEN = 40_960          # Qwen3-8B native max_position_embeddings
MAX_NUM_SEQS = 8                # pilot concurrency; lengths are unaffected
KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2   # Qwen3-8B GQA, BF16
WEIGHT_BYTES = 16.4e9
HBM_BYTES = 80e9
GPU_MEM_UTIL = 0.90

# Same lane the phase-98 campaigns registered (run_w98_g98e_d3.LANES),
# copied rather than imported so this script does not drag campaign
# authorization machinery into a pilot.
LANES = {
    "h103": {"physical_gpu_index": 7,
             "physical_gpu_uuid": "GPU-6da30cbb-5974-e66f-a406-389946903db7",
             "cpu_affinity": (96, 111)},
    "h104": {"physical_gpu_index": 7,
             "physical_gpu_uuid": "GPU-7a8308d6-2a78-2929-99f2-11d2f11a45f3",
             "cpu_affinity": (96, 111)},
}

PILOT_N = {"LI": 16, "LO": 16, "LIO": 16, "SS": 32}
LO_T1_N = 8                     # secondary T=1 arm, LO only
LO_T1_SEED = 1234


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _lane():
    host = socket.gethostname().split(".")[0]
    _require(host in LANES, f"host {host!r} has no registered lane")
    return host, LANES[host]


def _preflight(lane) -> None:
    idx = lane["physical_gpu_index"]
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    row = {int(r.split(",")[0]): r for r in rows}[idx]
    _, uuid, mem = (f.strip() for f in row.split(","))
    _require(uuid == lane["physical_gpu_uuid"],
             f"GPU {idx} uuid {uuid} != registered {lane['physical_gpu_uuid']}")
    _require(int(mem) < 1024,
             f"GPU {idx} busy ({mem} MiB in use) -- refusing to boot; "
             "will not touch existing processes")
    stray = [k for k in os.environ if k.startswith("VLLM_SELF_SPEC")]
    _require(not stray, f"self-spec env leaked into pilot: {stray}")


def main() -> None:
    host, lane = _lane()
    _preflight(lane)
    lo, hi = lane["cpu_affinity"]
    os.sched_setaffinity(0, set(range(lo, hi + 1)))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(lane["physical_gpu_index"])
    os.environ.setdefault(
        "VLLM_CACHE_ROOT", "/tmp/v-sukmincho-w100/vllm-cache/pilot")

    sys.path.insert(0, str(SCRIPT_DIR))
    import w100_eval_datasets as wed

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET_SNAPSHOT)

    print(f"[pilot] loading prompts on {host}", flush=True)
    cells = {}
    for cell in wed.CELLS:
        prompts, spec = wed.load_cell(cell, tok, n=PILOT_N[cell], seed=0)
        cells[cell] = (prompts, spec)
        print(f"[pilot] {cell}: {len(prompts)} prompts ready", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=TARGET_SNAPSHOT, max_model_len=MAX_MODEL_LEN,
              max_num_seqs=MAX_NUM_SEQS,
              gpu_memory_utilization=GPU_MEM_UTIL)

    record = {"host": host, "model": TARGET_SNAPSHOT,
              "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS,
              "pins": wed.PINS, "cells": {}}

    def run(name, prompts, sp):
        t0 = time.monotonic()
        outs = llm.generate(prompts, sp)
        wall = time.monotonic() - t0
        reqs = []
        for o in outs:
            c = o.outputs[0]
            reqs.append({
                "prompt_toks": len(o.prompt_token_ids),
                "out_toks": len(c.token_ids),
                "finish": c.finish_reason,
                "sha256": hashlib.sha256(c.text.encode()).hexdigest(),
            })
        total_out = sum(r["out_toks"] for r in reqs)
        entry = {"wall_s": round(wall, 1), "n": len(reqs),
                 "total_out_toks": total_out,
                 "out_toks_per_s": round(total_out / wall, 1),
                 "temperature": sp.temperature,
                 "max_tokens": sp.max_tokens, "requests": reqs}
        outs_ = sorted(r["out_toks"] for r in reqs)
        q = lambda p: outs_[min(len(outs_) - 1, int(p * len(outs_)))]
        caps = sum(r["finish"] == "length" for r in reqs)
        print(f"[pilot] {name}: n={len(reqs)} wall={wall:.0f}s "
              f"out median={q(0.5)} p95={q(0.95)} max={outs_[-1]} "
              f"cap_hits={caps}/{len(reqs)}", flush=True)
        return entry

    for cell, (prompts, spec) in cells.items():
        sp = SamplingParams(temperature=spec["temperature"],
                            max_tokens=spec["max_tokens"])
        record["cells"][cell] = run(cell, prompts, sp)

    lo_prompts, lo_spec = cells["LO"]
    sp1 = SamplingParams(temperature=1.0, max_tokens=lo_spec["max_tokens"],
                         seed=LO_T1_SEED)
    record["cells"]["LO_T1"] = run("LO_T1", lo_prompts[:LO_T1_N], sp1)

    # KV feasibility from measured p95 lengths.
    budget = HBM_BYTES * GPU_MEM_UTIL - WEIGHT_BYTES
    feas = {}
    for cell, e in record["cells"].items():
        rs = e["requests"]
        tot = sorted(r["prompt_toks"] + r["out_toks"] for r in rs)
        p95 = tot[min(len(tot) - 1, int(0.95 * len(tot)))]
        feas[cell] = {"p95_total_toks": p95, "batch_verdicts": {
            b: ("ok" if p95 * KV_BYTES_PER_TOKEN * b <= budget
                else "infeasible")
            for b in (1, 8, 16, 32, 64)}}
        print(f"[pilot] feasibility {cell}: p95_total={p95} "
              f"{feas[cell]['batch_verdicts']}", flush=True)
    record["kv_feasibility"] = {"budget_bytes": budget,
                                "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
                                "cells": feas}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "pilot_lengths.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"[pilot] DONE -> {path}", flush=True)


if __name__ == "__main__":
    main()
