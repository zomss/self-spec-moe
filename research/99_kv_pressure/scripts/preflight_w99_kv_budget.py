#!/usr/bin/env python3
"""Preflight for the KV-pressure campaign: where does the draft stop fitting?

The campaign's premise is that at a large enough KV working set the separate
~6.1 GB w4a16 draft is no longer affordable, so no single static
configuration can serve a mix that spans both sides of that line. Before any
of it is preregistered, two things have to be true and neither is obvious:

1. the crossover must be REACHABLE inside the model's context limit and the
   box's memory, and
2. the largest cell the campaign wants to run must actually boot and generate.

Closed-form arithmetic says the crossover sits near 355k total KV tokens on
an 80 GiB H100 -- which at batch 8 is ~44k tokens per sequence, BEYOND
Qwen3-8B's 40960 max_position_embeddings. If that holds, a batch-8 sweep
measures nothing and the campaign has to move to higher batch. This script
replaces the arithmetic with vLLM's own accounting.

Each probe runs in a SUBPROCESS so an OOM kills the probe and not the sweep,
and every probe reports the engine's reported KV block count rather than a
derived estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PHASE = Path(__file__).resolve().parent.parent
REPO = PHASE.parent.parent

# Lane: h104 GPU 7 / NUMA node 1. NUMA 0 carries other tenants' CPU work, and
# CPUs 96-111 stay clear of the hash-bound telemetry set (64-95).
LANE = {
    "physical_gpu_index": 7,
    "physical_gpu_uuid": "GPU-7a8308d6-2a78-2929-99f2-11d2f11a45f3",
    "cpu_affinity": "96-111",
    "cache_root": "/tmp/v-sukmincho-w99/vllm-cache/lane-a",
}

TARGET_CKPT = "/h/v-sukmincho/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots"
QUANT_CKPT = "/h/v-sukmincho/ckpts/Qwen3-8B-W4A16-INT4"

GPU_MEMORY_UTILIZATION = 0.90
MAX_NUM_BATCHED_TOKENS = 8192


def resolve_target() -> str:
    root = Path(TARGET_CKPT)
    if root.is_dir():
        for snapshot in sorted(root.iterdir()):
            if (snapshot / "config.json").is_file():
                return str(snapshot)
    raise SystemExit(f"target checkpoint not found under {TARGET_CKPT}")


CHILD = r'''
import json, os, sys
from vllm import LLM

spec = json.loads(sys.argv[1])
kwargs = dict(
    model=spec["target"],
    tensor_parallel_size=1,
    max_model_len=spec["max_model_len"],
    max_num_batched_tokens=spec["max_num_batched_tokens"],
    max_num_seqs=spec["max_num_seqs"],
    enable_chunked_prefill=True,
    gpu_memory_utilization=spec["gpu_memory_utilization"],
    enable_prefix_caching=False,
    enforce_eager=False,
    seed=0,
    disable_log_stats=True,
)
if spec["draft"]:
    kwargs["speculative_config"] = {
        "method": "draft_model",
        "model": spec["draft"],
        "num_speculative_tokens": 4,
        "draft_tensor_parallel_size": 1,
    }

llm = LLM(**kwargs)
cache = llm.llm_engine.vllm_config.cache_config
blocks = cache.num_gpu_blocks
block_size = cache.block_size
out = {
    "ok": True,
    "num_gpu_blocks": blocks,
    "block_size": block_size,
    "kv_tokens": (blocks or 0) * block_size,
}

if spec.get("generate_tokens"):
    from vllm import SamplingParams
    prompt_ids = [[1000 + (i % 5000) for i in range(spec["prompt_tokens"])]
                  for _ in range(spec["max_num_seqs"])]
    result = llm.generate(
        prompt_token_ids=prompt_ids,
        sampling_params=SamplingParams(
            temperature=0.0, max_tokens=spec["generate_tokens"], ignore_eos=True
        ),
    )
    out["generated_sequences"] = len(result)
    out["generated_tokens"] = sum(len(o.outputs[0].token_ids) for o in result)

print("PREFLIGHT_RESULT " + json.dumps(out))
'''


def probe(spec: dict[str, Any], timeout: int) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(LANE["physical_gpu_index"]),
            "VLLM_CACHE_ROOT": LANE["cache_root"],
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_DISABLED_KERNELS": "MacheteLinearKernel",
            "VLLM_SELF_SPEC_SHARED_KV": "1",
            "VLLM_SELF_SPEC_SHARE_WEIGHTS": "0" if spec["draft"] else "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    cmd = [
        "numactl",
        f"--physcpubind={LANE['cpu_affinity']}",
        "--membind=1",
        str(REPO / ".venv/bin/python"),
        "-c",
        CHILD,
        json.dumps(spec),
    ]
    try:
        done = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "timeout_s": timeout}

    for line in done.stdout.splitlines():
        if line.startswith("PREFLIGHT_RESULT "):
            return json.loads(line[len("PREFLIGHT_RESULT ") :])
    tail = "\n".join(done.stdout.strip().splitlines()[-25:])
    return {"ok": False, "error": "no result line", "returncode": done.returncode,
            "tail": tail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--out", type=Path, default=PHASE / "data/preflight_kv_budget.json"
    )
    parser.add_argument(
        "--largest-cell",
        action="store_true",
        help="also generate on the largest candidate cell",
    )
    args = parser.parse_args()

    target = resolve_target()
    base = {
        "target": target,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    }

    probes: list[dict[str, Any]] = []
    for label, draft in (("no_draft", None), ("w4a16_draft", QUANT_CKPT)):
        spec = {**base, "draft": draft}
        print(f"[probe] {label} max_model_len={args.max_model_len} ...", flush=True)
        result = probe(spec, args.timeout)
        result["label"] = label
        result["spec"] = {k: v for k, v in spec.items() if k != "target"}
        probes.append(result)
        if result.get("ok"):
            print(
                f"  KV tokens available: {result['kv_tokens']:,} "
                f"({result['num_gpu_blocks']} blocks x {result['block_size']})"
            )
        else:
            print(f"  FAILED: {result.get('error')}")
            if result.get("tail"):
                print(result["tail"])

    record: dict[str, Any] = {
        "record_type": "w99_preflight_kv_budget",
        "schema_version": 1,
        "host": os.uname().nodename,
        "lane": LANE,
        "probes": probes,
    }

    ok = {p["label"]: p for p in probes if p.get("ok")}
    if "no_draft" in ok and "w4a16_draft" in ok:
        free = ok["no_draft"]["kv_tokens"]
        with_draft = ok["w4a16_draft"]["kv_tokens"]
        record["draft_cost_kv_tokens"] = free - with_draft
        record["crossover"] = {
            "kv_tokens_without_draft": free,
            "kv_tokens_with_draft": with_draft,
            "per_batch": {
                str(b): {
                    "seq_len_without_draft": free // b,
                    "seq_len_with_draft": with_draft // b,
                }
                for b in (8, 16, 32, 64)
            },
        }
        print("\n=== crossover ===")
        print(f"  KV tokens  no draft: {free:,}   with draft: {with_draft:,}")
        print(f"  draft costs {free - with_draft:,} KV tokens")
        print("  batch   max seq len (no draft / with draft)")
        for b in (8, 16, 32, 64):
            print(f"  {b:5d}   {free // b:8,} / {with_draft // b:8,}")

    if args.largest_cell and "w4a16_draft" in ok:
        cell = {
            **base,
            "draft": QUANT_CKPT,
            "max_num_seqs": 16,
            "prompt_tokens": 14336,
            "generate_tokens": 8192,
        }
        print("\n[probe] largest candidate cell: batch 16, 14336 in, 8192 out ...",
              flush=True)
        result = probe(cell, args.timeout * 2)
        result["label"] = "largest_cell"
        result["spec"] = {k: v for k, v in cell.items() if k != "target"}
        record["largest_cell"] = result
        print(
            f"  {'OK' if result.get('ok') else 'FAILED'}: "
            f"{json.dumps({k: v for k, v in result.items() if k != 'tail'})}"
        )
        if result.get("tail"):
            print(result["tail"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
