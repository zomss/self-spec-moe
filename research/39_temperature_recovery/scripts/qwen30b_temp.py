"""Phase 39: Qwen3-30B-A3B temperature-recovery + wall-clock (real forced-PCIe EP).

The headline. If temperature recovers accept on the proxy, repeat on the real
model: full-replica comm-free FP8 draft at DP=8 EP=8, measure accept_len at
temps {0, 0.7, 1.0} AND the decode tok/s speedup vs no-spec at temp 0.7.

Stack (spec): VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1 + DRAFT_FULL_REPLICA=1 +
DRAFT_FULL_CG (W7_FULL_CG, default 1) + draft FP8. Lossless standard rejection
sampler + probabilistic draft (draft returns probs at temp>0). Verify stays
full-EP all-to-all. Forced-PCIe set by the driver.

NOTE: prior W7 found the draft FULL_CG attention capture is lossy on Qwen3's FA3
backend (accept 2.0->1.6 even at DP=2, isolated to FULL_CG). So run BOTH FULL_CG
(named headline) AND PIECEWISE (W7_FULL_CG=0, correct attention) to separate the
temperature-recovery signal from the FA3 capture bug.

Steady-state DECODE tok/s via the two-length slope (cancels prefill).

Env knobs:
  W7_MODE        nospec | spec
  W7_TEMP        sampling temperature      default 0.0
  W7_DP          data-parallel size (=EP)  default 8
  W7_TP          tensor-parallel size      default 1
  W7_DRAFT_QUANT draft quantization        default fp8 ("" -> bf16)
  W7_BATCHES     comma list                default 32
  W7_KS          comma list (spec only)    default 4
  W7_OUTLEN      long output length        default 160
  W7_SHORTLEN    short output length        default 32
  W7_ITERS       timed iterations          default 3
  W7_WARMUP      warmup iterations         default 2
  W7_GPU_MEM     gpu_memory_utilization    default 0.90
  W7_MAX_MODEL_LEN  default 2048
  W7_FULL_CG     1 -> set DRAFT_FULL_CG    default 1
  W7_PROBABILISTIC 1 -> draft_sample_method=probabilistic default 1
  W7_SEED        sampling seed             default 0
  W7_LOG_A2A     1 -> set LOG_A2A_COUNTS   default 0
  W7_OUT / W7_TAG
"""
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "W7_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B"
    "/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
)
DP = int(os.environ.get("W7_DP", "8"))
TP = int(os.environ.get("W7_TP", "1"))
TEMP = float(os.environ.get("W7_TEMP", "0.0"))
SEED = int(os.environ.get("W7_SEED", "0"))
DRAFT_QUANT = os.environ.get("W7_DRAFT_QUANT", "fp8").strip()
OUT_DIR = os.environ.get(
    "W7_OUT", "/data/smcho/self-spec-moe/research/39_temperature_recovery/data"
)
TAG = os.environ.get("W7_TAG", "qwen30b")
OUTLEN = int(os.environ.get("W7_OUTLEN", "160"))
SHORTLEN = int(os.environ.get("W7_SHORTLEN", "32"))
ITERS = int(os.environ.get("W7_ITERS", "3"))
WARMUP = int(os.environ.get("W7_WARMUP", "2"))
GPU_MEM = float(os.environ.get("W7_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("W7_MAX_MODEL_LEN", "2048"))
BATCHES = [int(x) for x in os.environ.get("W7_BATCHES", "32").split(",")]
KS = [int(x) for x in os.environ.get("W7_KS", "4").split(",")]
FULL_CG = os.environ.get("W7_FULL_CG", "1") == "1"
FULL_REPLICA = os.environ.get("W7_FULL_REPLICA", "1") == "1"
PROBABILISTIC = os.environ.get("W7_PROBABILISTIC", "1") == "1"
LOG_A2A = os.environ.get("W7_LOG_A2A", "0") == "1"

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
    total, found = 0.0, False
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


def _metric_vec(metrics, name):
    for m in metrics:
        if m.name == name:
            vals = getattr(m, "values", None)
            if vals is not None:
                return list(vals)
    return None


def worker(rank, dp, tp, master_ip, master_port, mode, k, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    if LOG_A2A:
        os.environ["VLLM_SELF_SPEC_LOG_A2A_COUNTS"] = "1"

    spec = mode == "spec"
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        if FULL_REPLICA:
            os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
        if FULL_CG:
            os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
            os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        trust_remote_code=False,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,
        disable_log_stats=False,
    )
    if spec:
        spec_cfg = {
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": k,
            "draft_tensor_parallel_size": tp,
            "rejection_sample_method": "standard",
        }
        if PROBABILISTIC:
            spec_cfg["draft_sample_method"] = "probabilistic"
        if DRAFT_QUANT:
            spec_cfg["quantization"] = DRAFT_QUANT
        kwargs["speculative_config"] = spec_cfg
    llm = LLM(**kwargs)

    if rank == 0 and spec:
        try:
            sc = llm.llm_engine.vllm_config.speculative_config
            print("[CFG] draft_sample_method="
                  f"{getattr(sc, 'draft_sample_method', '?')} "
                  "rejection_sample_method="
                  f"{getattr(sc, 'rejection_sample_method', '?')} "
                  f"full_cg={FULL_CG} temp={TEMP}", flush=True)
        except Exception as e:
            print(f"[CFG] could not read spec config: {e!r}", flush=True)

    def run_batch(batch, out_len):
        prompts = [f"{BASE_PROMPT} the year {1900 + i}." for i in range(batch)]
        sp = SamplingParams(
            temperature=TEMP, max_tokens=out_len, ignore_eos=True, seed=SEED,
        )
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        return dt, ntok

    POS = "vllm:spec_decode_num_accepted_tokens_per_pos"

    def _snap():
        if not (spec and rank == 0):
            return None
        m = llm.get_metrics()
        return (
            _metric_value(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_draft_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_drafts") or 0.0,
            _metric_vec(m, POS) or [],
        )

    results = []
    for batch in BATCHES:
        try:
            for _ in range(WARMUP):
                run_batch(batch, OUTLEN)
                run_batch(batch, SHORTLEN)

            long_times, short_times, full_ntoks = [], [], []
            accepted = drafted = ndrafts = None
            per_pos = None
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
                pp0, pp1 = snap0[3], snap1[3]
                n = max(len(pp0), len(pp1))
                pp0 = pp0 + [0] * (n - len(pp0))
                pp1 = pp1 + [0] * (n - len(pp1))
                per_pos = [pp1[i] - pp0[i] for i in range(n)]

            decode_times = [lt - st for lt, st in zip(long_times, short_times)]
            out_decode = batch * (OUTLEN - SHORTLEN)
            toks = [out_decode / dt for dt in decode_times]
            dt_m, dt_s = _mean_std(decode_times)
            tps_m, tps_s = _mean_std(toks)
            lt_m, _ = _mean_std(long_times)
            st_m, _ = _mean_std(short_times)
            row = {
                "batch": batch, "out_len": OUTLEN, "short_len": SHORTLEN,
                "temp": TEMP, "iters": ITERS, "warmup": WARMUP,
                "long_s_mean": lt_m, "short_s_mean": st_m,
                "decode_s_mean": dt_m, "decode_s_std": dt_s,
                "tok_s_mean": tps_m, "tok_s_std": tps_s,
                "out_tokens_decode": out_decode,
                "full_ntoks": full_ntoks[-1],
            }
            if spec and rank == 0:
                al = (1 + accepted / ndrafts) if (accepted and ndrafts) else None
                per_tok = (accepted / drafted) if (accepted and drafted) else None
                pos_rate = [p / ndrafts if ndrafts else None for p in (per_pos or [])]
                row["accept_len"] = al
                row["accept_rate_per_tok"] = per_tok
                row["num_accepted"] = accepted
                row["num_drafted"] = drafted
                row["num_drafts"] = ndrafts
                row["per_pos_accepted"] = per_pos
                row["per_pos_rate"] = pos_rate
            results.append(row)
            if rank == 0:
                tag = f"K={k}" if spec else "nospec"
                al_s = (
                    f" accept_len={row.get('accept_len'):.3f}"
                    if spec and row.get("accept_len") else ""
                )
                print(f"[Q30T dp={dp} {tag} T={TEMP}] batch={batch} "
                      f"decode={dt_m:.3f}s tok/s={tps_m:.1f}+-{tps_s:.1f}{al_s}",
                      flush=True)
        except Exception as e:
            if rank == 0:
                results.append({"batch": batch, "error": repr(e)[:300]})
                print(f"[Q30T dp={dp}] batch={batch} FAILED: {repr(e)[:200]}",
                      flush=True)
            break

    if rank == 0:
        q.put(results)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("W7_MODE", "nospec")
    assert mode in ("nospec", "spec")
    ks = KS if mode == "spec" else [0]

    from vllm.utils.network_utils import get_open_port

    os.makedirs(OUT_DIR, exist_ok=True)
    tstr = f"t{TEMP:g}".replace(".", "p")
    suffix = ("fullcg" if (FULL_CG and mode == "spec") else "pw"
              ) if mode == "spec" else "nospec"

    for k in ks:
        master_ip = "127.0.0.1"
        master_port = None
        for _ in range(10):
            p = get_open_port()
            if p != master_port:
                master_port = p
                break
        q = Queue()
        procs = []
        for rank in range(DP):
            p = Process(target=worker,
                        args=(rank, DP, TP, master_ip, master_port, mode, k, q))
            p.start()
            procs.append(p)
        res = None
        try:
            res = q.get(timeout=7200)
        except Exception:
            res = [{"error": "no result from rank 0 (timeout)"}]
        for p in procs:
            p.join(timeout=120)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=30)
        ktag = f"K{k}" if mode == "spec" else "nospec"
        path = os.path.join(
            OUT_DIR, f"q30t_{TAG}_dp{DP}_{mode}_{tstr}_{suffix}_{ktag}.json")
        with open(path, "w") as f:
            json.dump({"mode": mode, "model": MODEL, "tag": TAG, "dp": DP, "ep": DP,
                       "tp": TP, "temp": TEMP, "draft_quant": DRAFT_QUANT or "bf16",
                       "full_cg": FULL_CG, "probabilistic": PROBABILISTIC,
                       "this_k": k, "out_len": OUTLEN, "short_len": SHORTLEN,
                       "iters": ITERS, "warmup": WARMUP, "results": res}, f, indent=2)
        print(f"[Q30T] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
