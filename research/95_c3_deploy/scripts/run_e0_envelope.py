#!/usr/bin/env python3
"""E0: the window-switching ENVELOPE on real data. No engine change.

Today's runner already boots one window per engine, so the `oracle` arm --
per-cell max over windows, switch cost zeroed -- is reachable now. That
envelope is the UPPER BOUND on what any window switcher could deliver,
measured on the deployment path rather than on compile cells.

Gate (pre-registered, phase README): if the envelope over `konly` is < 1% on
real traces, the switching thesis is refuted on the deployment path and the
multi-capture engine work (E1) is NOT built.

One boot = one (arch, window) arm, or the AR anchor. Each boot runs the same
regimes with the window's own policy table, so K/OFF is chosen per step
exactly as `konly` does today.

env: E95_ARCH    dense | llama
     E95_WINDOW  512 | 2048 | none | off   ("off" = AR anchor, no spec)
     E95_REGIMES comma list (default below)

Regime choice is part of the measurement. A window can only matter where it
BINDS, i.e. where the end context (prompt + generation) exceeds it; measured
end contexts (seed 0, Qwen tokenizer):

    R1    b1   1087    crosses 512 only
    R4    b8   8691    crosses both
    R5    b8  14549    crosses both
    R5cot b8  17193    crosses both
    R8    b16  2166    crosses both (just)
    R6    b32   318    crosses NEITHER  <- null control

So the default set is R4,R5,R5cot,R8 (discriminating) + R1 (512 only) + R6.
**R6 is a deliberate negative control**: the map's window choice cannot
matter there, so its envelope estimates this metric's noise floor. If R6's
envelope is comparable to the discriminating regimes', the measurement is
noise and the gate must not be read as passed.

     E95_ITERS   timed rounds per regime (default 3)
     E95_SEED    seed tag, recorded in the output (default 0)
     E95_OUT     output json path
"""
import json
import os
import sys
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE.parent / "88_regime_eval/scripts"))
from regime_datasets import load_regime  # noqa: E402

ARCH = os.environ.get("E95_ARCH", "dense")
WINDOW = os.environ.get("E95_WINDOW", "512")
ITERS = int(os.environ.get("E95_ITERS", "3"))
SEED = int(os.environ.get("E95_SEED", "0"))
REGIMES = os.environ.get(
    "E95_REGIMES", "R4,R5,R5cot,R8,R1,R6").split(",")

# quant lever is CONSTANT per arch (the finding that removes the DRAM swap);
# skip is boot-fixed at the aggregate-best value (b2 for both).
ARCHS = {
    "dense": {
        "model": "Qwen/Qwen3-8B",
        "draft": os.path.expanduser("~/ckpts/Qwen3-8B-W4A8-gptq"),
        "config": "q-hum_s-b2",
        "kv_limit": 330000,
        # skip sets are SEARCH-DERIVED PER ARCHITECTURE -- not a shared
        # constant. 93/data/skipsets_*.json; dense's b2 is 2,8 (as used by
        # 94/run_map_validate.sh), llama's is 3,8.
        "skip": "2,8",
    },
    "llama": {
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "draft": os.path.expanduser(
            "~/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym"),
        "config": "q-w4a16_s-b2",
        "kv_limit": 260000,
        "skip": "3,8",
    },
}
KMAX = 4                  # policy chooses K in {0, 2, 4}


def spec_counters(llm):
    acc = drafts = 0
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            acc = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            drafts = m.value
    return acc, drafts


def main():
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    a = ARCHS[ARCH]
    spec = None
    if WINDOW != "off":
        # 96/W4: E95_POLICY overrides the table (e.g. the park-everywhere
        # fixture that measures parked-engine cost).
        tbl = Path(os.environ["E95_POLICY"]) if os.environ.get(
            "E95_POLICY") else (
            PHASE / "data" / f"policy_{ARCH}_{a['config']}_w{WINDOW}.json")
        if not tbl.exists():
            raise SystemExit(f"missing policy table {tbl}")
        os.environ["VLLM_SELF_SPEC_POLICY_FILE"] = str(tbl)
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] = (
            "0" if WINDOW == "none" else WINDOW)
        os.environ["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = a["skip"]
        spec = {"method": "draft_model", "model": a["draft"],
                "num_speculative_tokens": KMAX,
                "draft_tensor_parallel_size": 1}

    extra = {}
    if os.environ.get("E95_TUNE", "1") != "1":
        # 96/W1: flashinfer autotune is a boot-scoped kernel lottery
        # (40% swing); scored runs set E95_TUNE=0. Default preserves the
        # historical boot path.
        extra["kernel_config"] = {"enable_flashinfer_autotune": False}
    llm = LLM(model=a["model"], speculative_config=spec,
              tensor_parallel_size=1, max_model_len=20480,
              gpu_memory_utilization=0.90, max_num_seqs=32,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192, **extra)
    tok = AutoTokenizer.from_pretrained(a["model"])

    out = {"arch": ARCH, "window": WINDOW, "seed": SEED,
           "config": a["config"], "skip_set": a["skip"] if spec else None,
           "kmax": KMAX if spec else 0,
           "model": a["model"], "draft": a["draft"] if spec else None,
           "regimes": {}}
    for rid in REGIMES:
        n_load = 32 if rid == "R6" else 16
        prompts, gs = load_regime(rid, tok, n=n_load, seed=SEED)
        b = gs["batch"]
        sp = SamplingParams(max_tokens=gs["max_tokens"],
                            temperature=gs["temperature"],
                            seed=SEED if gs["temperature"] else None,
                            ignore_eos=(rid in ("R5cot", "R8")))
        llm.generate(prompts[:b], sp, use_tqdm=False)      # warmup
        rates, accepts = [], []
        for _ in range(ITERS):
            a0, d0 = spec_counters(llm)
            t0 = time.perf_counter()
            ntok = 0
            if b == 1:
                for p in prompts[:4]:
                    o = llm.generate([p], sp, use_tqdm=False)
                    ntok += sum(len(x.outputs[0].token_ids) for x in o)
            else:
                o = llm.generate(prompts[:b], sp, use_tqdm=False)
                ntok = sum(len(x.outputs[0].token_ids) for x in o)
            dt = time.perf_counter() - t0
            a1, d1 = spec_counters(llm)
            rates.append(ntok / dt)
            if d1 > d0:
                accepts.append(1 + (a1 - a0) / (d1 - d0))
        rates.sort()
        med = rates[len(rates) // 2]
        out["regimes"][rid] = {
            "toks": round(med, 1),
            "all": [round(r, 1) for r in rates],
            "accept": round(sum(accepts) / len(accepts), 3) if accepts else None,
            "batch": b, "max_tokens": gs["max_tokens"],
            "temp": gs["temperature"],
        }
        print(f"[E0] {ARCH} w={WINDOW} seed={SEED} {rid} b={b} "
              f"toks={med:.1f} accept={out['regimes'][rid]['accept']}",
              flush=True)

    path = os.environ.get(
        "E95_OUT", str(PHASE / "data" / f"e0_{ARCH}_w{WINDOW}_s{SEED}.json"))
    Path(path).write_text(json.dumps(out, indent=1))
    print("[E0] saved ->", path, flush=True)


if __name__ == "__main__":
    main()
