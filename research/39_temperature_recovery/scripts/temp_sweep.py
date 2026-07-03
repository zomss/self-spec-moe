"""Phase 39: TEMPERATURE-recovery MEASUREMENT (no source changes).

Hypothesis (the recovery). World A's comm-free FULL-REPLICA draft accept
"collapses" at large EP under GREEDY (phase 38: A=2.18 on the proxy; phase 34:
~1.0 on Qwen3-30B). But every prior measurement was greedy -- the WORST case for
a tiny FP-reduction-order perturbation (comm-free local sum vs EP all-to-all) that
flips near-tie ARGMAXES. Under TEMPERATURE sampling, the lossless distributional
rejection sampler accepts at rate min(1, p_target/p_draft); the full-replica draft
computes the SAME experts (summed locally), so p_draft ~= p_target (small TV) and
the FP divergence should be ABSORBED -> accept RISES toward K+1, while staying
LOSSLESS (standard rejection-sampling guarantee).

To make the rejection sampler use the DRAFT's probabilities (required for the
lossless ratio test at temperature>0), the speculative config sets
  draft_sample_method   = "probabilistic"   (default is "greedy" -> no draft_probs)
  rejection_sample_method = "standard"       (default; lossless ratio test)
Verified: llm_base_proposer._enable_probabilistic_draft_probs gates on exactly
these two -> the draft returns its softmax probs to the rejection sampler. With
greedy temperature=0 the draft falls back to argmax (draft_probs=None) -> the
existing greedy collapse baseline (cross-checks phase 38 A=2.18).

Proxy: Qwen/Qwen1.5-MoE-A2.7B (qwen2_moe, 60 experts top-4, 24 layers, non-MLA),
DP=8 -> EP=8, forced-PCIe, bf16 (NO FP8 confound). Full-replica comm-free draft:
  VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1 VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1
  VLLM_SELF_SPEC_DRAFT_FULL_CG=1      VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
  VLLM_SELF_SPEC_LOCAL_ROUTE=0  (verify stays full-EP all-to-all)

Env knobs:
  E_TEMP     sampling temperature        (required, e.g. 0.0 0.5 0.7 1.0)
  E_DP       data-parallel size (=EP)    default 8
  E_TP       tensor-parallel size        default 1
  E_K        num_speculative_tokens      default 4
  E_BATCH    batch size                  default 16
  E_OUTLEN   output length               default 128
  E_MAXLEN   max_model_len               default 2048
  E_GPU_MEM  gpu_memory_utilization      default 0.85
  E_SEED     sampling seed               default 0
  E_PROBABILISTIC  1 -> draft_sample_method=probabilistic (default 1)
  E_TAG      filename tag                default temp
  E_OUT      output dir
"""
import json
import os
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "E_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
    "/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
)
TEMP = float(os.environ["E_TEMP"])
DP = int(os.environ.get("E_DP", "8"))
TP = int(os.environ.get("E_TP", "1"))
K = int(os.environ.get("E_K", "4"))
BATCH = int(os.environ.get("E_BATCH", "16"))
OUTLEN = int(os.environ.get("E_OUTLEN", "128"))
MAXLEN = int(os.environ.get("E_MAXLEN", "2048"))
GPU_MEM = float(os.environ.get("E_GPU_MEM", "0.85"))
SEED = int(os.environ.get("E_SEED", "0"))
PROBABILISTIC = os.environ.get("E_PROBABILISTIC", "1") == "1"
# CFG (phase-38 routings): A = full-replica comm-free (the collapse, default);
# C = EP-full draft (real all-to-all = the verify; the upper-bound control).
CFG = os.environ.get("E_CFG", "A").strip().upper()
# CFG -> (DRAFT_FULL_REPLICA, DRAFT_LOCAL_ROUTE)
CFG_MAP = {"A": ("1", "1"), "C": ("0", "0")}
assert CFG in CFG_MAP, f"E_CFG must be A or C, got {CFG}"
FULL_REPLICA, LOCAL_ROUTE = CFG_MAP[CFG]
TAG = os.environ.get("E_TAG", "temp")
OUT = os.environ.get(
    "E_OUT", "/data/smcho/self-spec-moe/research/39_temperature_recovery/data"
)


def _mv(metrics, name):
    total = 0.0
    found = False
    for m in metrics:
        if m.name == name:
            found = True
            v = getattr(m, "value", None)
            if v is not None:
                total += v
            else:
                vals = getattr(m, "values", None)
                if vals is not None:
                    total += sum(vals)
    return total if found else None


def _vec(metrics, name):
    for m in metrics:
        if m.name == name:
            vals = getattr(m, "values", None)
            if vals is not None:
                return list(vals)
    return None


def worker(rank, master_ip, master_port, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    # CFG A: full-replica comm-free draft. CFG C: EP-full draft (real all-to-all
    # = the verify; control). Verify stays full-EP (LOCAL_ROUTE=0). Compile fixed.
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = LOCAL_ROUTE
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = FULL_REPLICA
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
    os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"

    from vllm import LLM, SamplingParams

    spec_cfg = {
        "method": "draft_model",
        "model": MODEL,
        "num_speculative_tokens": K,
        "draft_tensor_parallel_size": TP,
        # Lossless standard rejection sampler + probabilistic draft so the
        # draft returns its probs (required for the ratio test at temp>0).
        "rejection_sample_method": "standard",
    }
    if PROBABILISTIC:
        spec_cfg["draft_sample_method"] = "probabilistic"

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=TP,
        enable_expert_parallel=True,
        trust_remote_code=True,
        max_model_len=MAXLEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,
        disable_log_stats=False,
        speculative_config=spec_cfg,
    )
    try:
        llm = LLM(**kwargs)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"init: {repr(e)[:600]}"})
        return

    # Echo the resolved spec config so the log proves the probabilistic path.
    if rank == 0:
        try:
            sc = llm.llm_engine.vllm_config.speculative_config
            print(
                "[CFG] draft_sample_method="
                f"{getattr(sc, 'draft_sample_method', '?')} "
                "rejection_sample_method="
                f"{getattr(sc, 'rejection_sample_method', '?')}",
                flush=True,
            )
        except Exception as e:
            print(f"[CFG] could not read spec config: {e!r}", flush=True)

    base = ("The history of artificial intelligence began in antiquity, with "
            "myths and stories of artificial beings. In modern times, the field "
            "was founded in")
    prompts = [f"{base} the year {1900 + i}." for i in range(BATCH)]
    sp = SamplingParams(temperature=TEMP, max_tokens=OUTLEN, ignore_eos=True,
                        seed=SEED)

    POS = "vllm:spec_decode_num_accepted_tokens_per_pos"

    def snap():
        m = llm.get_metrics()
        return {
            "accepted": _mv(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
            "drafted": _mv(m, "vllm:spec_decode_num_draft_tokens") or 0.0,
            "ndrafts": _mv(m, "vllm:spec_decode_num_drafts") or 0.0,
            "per_pos": _vec(m, POS) or [],
        }

    try:
        # warmup (captures cudagraphs + clears transient); greedy warmup is fine.
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8, seed=0),
                     use_tqdm=False)
        s0 = snap()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        s1 = snap()
        if rank == 0:
            tokens = [list(o.outputs[0].token_ids) for o in outs]
            ntok = sum(len(t) for t in tokens)
            accepted = s1["accepted"] - s0["accepted"]
            drafted = s1["drafted"] - s0["drafted"]
            ndrafts = s1["ndrafts"] - s0["ndrafts"]
            pp0, pp1 = s0["per_pos"], s1["per_pos"]
            n = max(len(pp0), len(pp1))
            pp0 += [0] * (n - len(pp0))
            pp1 += [0] * (n - len(pp1))
            per_pos = [pp1[i] - pp0[i] for i in range(n)]
            al = (1 + accepted / ndrafts) if ndrafts else None
            per_tok = (accepted / drafted) if drafted else None
            pos_rate = [p / ndrafts if ndrafts else None for p in per_pos]
            q.put({
                "temp": TEMP, "probabilistic": PROBABILISTIC, "cfg": CFG,
                "dp": DP, "ep": DP, "K": K, "batch": BATCH, "outlen": OUTLEN,
                "seed": SEED,
                "accept_len": al, "per_tok": per_tok,
                "accepted": accepted, "drafted": drafted, "ndrafts": ndrafts,
                "per_pos_accepted": per_pos, "per_pos_rate": pos_rate,
                "gen_s": dt, "ntok": ntok,
                "tokens": tokens,
            })
    except Exception as e:
        if rank == 0:
            q.put({"error": f"gen: {repr(e)[:600]}", "temp": TEMP})


def main():
    from vllm.utils.network_utils import get_open_port
    master_ip, master_port = "127.0.0.1", get_open_port()
    q = Queue()
    procs = [Process(target=worker, args=(r, master_ip, master_port, q))
             for r in range(DP)]
    for p in procs:
        p.start()
    res = None
    try:
        res = q.get(timeout=2400)
    except Exception:
        res = {"error": "no result (timeout)"}
    for p in procs:
        p.join(timeout=90)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=30)
    os.makedirs(OUT, exist_ok=True)
    tstr = f"t{TEMP:g}".replace(".", "p")
    pstr = "prob" if PROBABILISTIC else "greedydraft"
    path = os.path.join(OUT, f"temp_{TAG}_cfg{CFG}_{tstr}_{pstr}_dp{DP}_K{K}.json")
    with open(path, "w") as f:
        json.dump(res, f)
    short = {k: v for k, v in (res or {}).items() if k != "tokens"}
    print(f"RESULT temp={TEMP} prob={PROBABILISTIC} {json.dumps(short)[:500]}",
          flush=True)
    print(f"[TEMP] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
