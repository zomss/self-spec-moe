"""W7-2node: wall-clock decode tok/s on a REAL 2-node fabric (Phase 52).

Phase-local copy of research/34_worldA_system/scripts/w7_fp8_timing.py (which
stays UNCHANGED); same two-length-slope methodology and W7_* knobs. Deltas:
  - node-rank-aware spawning: global DP rank = W7_NODE_RANK*W7_LOCAL_WORLD +
    local_rank; VLLM_DP_RANK_LOCAL = local_rank (was = global rank);
  - fixed pre-agreed DP master ip/port (W7_MASTER_IP/W7_MASTER_PORT, +k per K)
    instead of get_open_port();
  - only global rank 0 (node 0) collects and writes results.

Run on both nodes with W7_NODE_RANK=0 (h107, writes results) and 1 (h106).
"""
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get("W7_MODEL", "Qwen/Qwen3-30B-A3B")
NODE_RANK = int(os.environ.get("W7_NODE_RANK", "0"))
NODES = int(os.environ.get("W7_NODES", "2"))
LOCAL_WORLD = int(os.environ.get("W7_LOCAL_WORLD", "8"))
DP = NODES * LOCAL_WORLD
TP = int(os.environ.get("W7_TP", "1"))
TRC = os.environ.get("W7_TRC", "0") == "1"
DRAFT_QUANT = os.environ.get("W7_DRAFT_QUANT", "").strip()
MASTER_IP = os.environ.get("W7_MASTER_IP", "10.31.199.17")
MASTER_PORT = int(os.environ.get("W7_MASTER_PORT", "13355"))
OUT_DIR = os.environ.get(
    "W7_OUT", "/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e/data"
)
TAG = os.environ.get("W7_TAG", "qwen30b_2node")
OUTLEN = int(os.environ.get("W7_OUTLEN", "160"))
SHORTLEN = int(os.environ.get("W7_SHORTLEN", "32"))
ITERS = int(os.environ.get("W7_ITERS", "3"))
WARMUP = int(os.environ.get("W7_WARMUP", "2"))
GPU_MEM = float(os.environ.get("W7_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("W7_MAX_MODEL_LEN", "2048"))
BATCHES = [int(x) for x in os.environ.get("W7_BATCHES", "32,64,128").split(",")]
KS = [int(x) for x in os.environ.get("W7_KS", "2").split(",")]
EAGER = os.environ.get("W7_EAGER", "0") == "1"
A2A_US = float(os.environ.get("W7_A2A_US", "0"))

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
)


def _mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def _metric_value(metrics, name):
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


def worker(rank, local_rank, dp, tp, master_ip, master_port, mode, k, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    spec = mode == "spec"
    # Phase 57 (World B): W7_SPEC_METHOD selects the drafter. The self-spec
    # env stack (VLLM_SELF_SPEC_DRAFT_*) is ONLY valid for the draft_model
    # (self-spec) path; for a real EAGLE head (method != draft_model) those
    # flags are wrong and must not be set (cf. Phase 50 STACK=0).
    spec_method = os.environ.get("W7_SPEC_METHOD", "draft_model")
    if A2A_US > 0:
        os.environ["VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US"] = str(A2A_US)
    if spec and spec_method == "draft_model":
        # Phase 54: W7_DRAFT_LOCAL_ROUTE=0 + W7_DRAFT_NODE_LOCAL=1 -> the
        # node-local (intra-node EP over NVLink) draft instead of the
        # device-local comm-free draft.
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = os.environ.get(
            "W7_DRAFT_LOCAL_ROUTE", "1"
        )
        os.environ["VLLM_SELF_SPEC_DRAFT_NODE_LOCAL"] = os.environ.get(
            "W7_DRAFT_NODE_LOCAL", "0"
        )
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        # Phase 53: W7_DRAFT_FULL_REPLICA=0 -> EP-shard comm-free draft (the
        # at-scale config; a full replica does not fit >=~40B models).
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = os.environ.get(
            "W7_DRAFT_FULL_REPLICA", "1"
        )

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        trust_remote_code=TRC,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=EAGER,
        disable_log_stats=False,
    )
    # Phase 53/54: cap CUDA-graph capture shapes (graph private pools are not
    # covered by gpu_memory_utilization; at 236B the default 51 shapes OOM).
    cg_sizes = os.environ.get("W7_CG_SIZES", "").strip()
    if cg_sizes:
        kwargs["compilation_config"] = {
            "cudagraph_capture_sizes": [int(x) for x in cg_sizes.split(",")]
        }
    # Phase 53: cap the profile forward (the fused-MoE workspace scales with
    # DP-gathered max_num_batched_tokens; 8192 default -> ~15 GiB at 236B).
    mnb = os.environ.get("W7_MAX_NUM_BATCHED", "").strip()
    if mnb:
        kwargs["max_num_batched_tokens"] = int(mnb)
    # Phase 57: EAGLE spec methods auto-enable async scheduling, whose
    # batch-queue path deadlocks the `sample_tokens` RPC under DP16 multi-step
    # (K>1) on the 2-node fabric (self-spec draft_model runs it OFF already).
    # W7_ASYNC_SCHED=0/1 forces it; unset -> vLLM default (self-spec unchanged).
    async_sched = os.environ.get("W7_ASYNC_SCHED", "").strip()
    if async_sched:
        kwargs["async_scheduling"] = async_sched == "1"
    if spec:
        # Phase 57 (World B): W7_SPEC_METHOD/W7_SPEC_MODEL select the drafter
        # (e.g. a real EAGLE3 head) on the 2-node testbed. Default keeps the
        # self-spec draft_model path byte-identical.
        spec_cfg = {
            "method": spec_method,
            "model": os.environ.get("W7_SPEC_MODEL", MODEL),
            "num_speculative_tokens": k,
        }
        if spec_method == "draft_model":
            spec_cfg["draft_tensor_parallel_size"] = tp
            if DRAFT_QUANT:
                spec_cfg["quantization"] = DRAFT_QUANT
        kwargs["speculative_config"] = spec_cfg
    llm = LLM(**kwargs)

    # Phase 57-A2: W7_PROMPT_FILE (one prompt/line) supplies on-distribution
    # natural-language prompts instead of the synthetic default (which depresses
    # a trained EAGLE head's acceptance). Cycled to fill the batch. Default
    # unset -> the synthetic BASE_PROMPT behavior is byte-identical.
    _prompt_bank = None
    _pf = os.environ.get("W7_PROMPT_FILE", "").strip()
    if _pf:
        with open(_pf) as _f:
            _prompt_bank = [ln.strip() for ln in _f if ln.strip()]
    # Phase 57-A2b: W7_CHAT=1 wraps each prompt in the model's chat template
    # (add_generation_prompt) so the target produces an ASSISTANT response --
    # the actual distribution a trained EAGLE3 instruct head predicts. Raw
    # continuation of arbitrary text (W7_CHAT unset) is off-distribution.
    _use_chat = os.environ.get("W7_CHAT", "0") == "1"
    if _use_chat and _prompt_bank:
        _tok = llm.get_tokenizer()
        _prompt_bank = [
            _tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False,
            )
            for p in _prompt_bank
        ]

    def run_batch(batch, out_len):
        if _prompt_bank:
            prompts = [_prompt_bank[i % len(_prompt_bank)] for i in range(batch)]
        else:
            prompts = [f"{BASE_PROMPT} the year {1900 + i}." for i in range(batch)]
        sp = SamplingParams(
            temperature=0.0, max_tokens=out_len, ignore_eos=True, seed=0,
        )
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        return dt, ntok

    def _snap():
        if not (spec and rank == 0):
            return None
        m = llm.get_metrics()
        return (
            _metric_value(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_draft_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_drafts") or 0.0,
        )

    results = []
    for batch in BATCHES:
        try:
            for _ in range(WARMUP):
                run_batch(batch, OUTLEN)
                run_batch(batch, SHORTLEN)

            long_times, short_times, full_ntoks = [], [], []
            accepted = drafted = ndrafts = None
            snap0 = _snap()
            for _ in range(ITERS):
                lt, ln = run_batch(batch, OUTLEN)
                st, _ = run_batch(batch, SHORTLEN)
                long_times.append(lt)
                short_times.append(st)
                full_ntoks.append(ln)
            snap1 = _snap()
            if snap0 is not None:
                accepted = snap1[0] - snap0[0]
                drafted = snap1[1] - snap0[1]
                ndrafts = snap1[2] - snap0[2]

            decode_times = [lt - st for lt, st in zip(long_times, short_times)]
            out_decode = batch * (OUTLEN - SHORTLEN)
            toks = [out_decode / dt for dt in decode_times]
            dt_m, dt_s = _mean_std(decode_times)
            tps_m, tps_s = _mean_std(toks)
            lt_m, _ = _mean_std(long_times)
            st_m, _ = _mean_std(short_times)
            suspect = bool(
                (st_m > 0 and dt_m < 0.15 * st_m)
                or (tps_m > 0 and tps_s / tps_m > 0.15)
            )
            row = {
                "batch": batch, "out_len": OUTLEN, "short_len": SHORTLEN,
                "iters": ITERS, "warmup": WARMUP,
                "long_s_mean": lt_m, "short_s_mean": st_m,
                "long_s_all": long_times, "short_s_all": short_times,
                "decode_s_mean": dt_m, "decode_s_std": dt_s,
                "tok_s_mean": tps_m, "tok_s_std": tps_s,
                "out_tokens_decode": out_decode,
                "full_ntoks": full_ntoks[-1],
                "suspect": suspect,
            }
            if spec and rank == 0:
                al = (1 + accepted / ndrafts) if (accepted and ndrafts) else None
                row["accept_len"] = al
                row["num_accepted"] = accepted
                row["num_drafted"] = drafted
                row["num_drafts"] = ndrafts
            results.append(row)
            if rank == 0:
                tag = f"K={k}" if spec else "nospec"
                al_s = (
                    f" accept_len={row.get('accept_len'):.3f}"
                    if spec and row.get("accept_len") else ""
                )
                print(f"[W7-2N {tag}] batch={batch} decode={dt_m:.3f}s "
                      f"tok/s={tps_m:.1f}+-{tps_s:.1f}{al_s}", flush=True)
        except Exception as e:
            if rank == 0:
                results.append({"batch": batch, "error": repr(e)[:300]})
                print(f"[W7-2N] batch={batch} FAILED: {repr(e)[:200]}", flush=True)
            break

    if rank == 0:
        q.put(results)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("W7_MODE", "nospec")
    assert mode in ("nospec", "spec")
    ks = KS if mode == "spec" else [0]

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = ("eager" if EAGER else "cg")
    if A2A_US > 0:
        suffix += f"_a2a{int(A2A_US)}us"
    all_out = {"mode": mode, "model": MODEL, "tag": TAG, "dp": DP, "tp": TP,
               "nodes": NODES, "node_rank": NODE_RANK,
               "draft_quant": DRAFT_QUANT or "bf16",
               "out_len": OUTLEN, "short_len": SHORTLEN, "iters": ITERS,
               "warmup": WARMUP, "gpu_mem": GPU_MEM, "eager": EAGER,
               "a2a_emulated_us": A2A_US, "suffix": suffix, "by_k": {}}

    for k in ks:
        master_port = MASTER_PORT + k
        q = Queue()
        procs = []
        for local_rank in range(LOCAL_WORLD):
            rank = NODE_RANK * LOCAL_WORLD + local_rank
            p = Process(target=worker,
                        args=(rank, local_rank, DP, TP, MASTER_IP, master_port,
                              mode, k, q))
            p.start()
            procs.append(p)
        res = None
        if NODE_RANK == 0:
            try:
                res = q.get(timeout=5400)
            except Exception:
                res = [{"error": "no result from rank 0 (timeout)"}]
        for p in procs:
            p.join()
        if NODE_RANK != 0:
            continue
        all_out["by_k"][str(k)] = res
        ktag = f"K{k}" if mode == "spec" else "nospec"
        path = os.path.join(OUT_DIR, f"w72n_{TAG}_{mode}_{suffix}_{ktag}.json")
        with open(path, "w") as f:
            json.dump({**all_out, "this_k": k, "results": res}, f, indent=2)
        print(f"[W7-2N] wrote {path}", flush=True)

    if NODE_RANK == 0:
        path = os.path.join(OUT_DIR, f"w72n_{TAG}_{mode}_{suffix}.json")
        with open(path, "w") as f:
            json.dump(all_out, f, indent=2)
        print(f"[W7-2N] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
