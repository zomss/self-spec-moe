#!/usr/bin/env python3
"""E0 assertion table: parse per-arm engine logs, verify infra parity vs the L0 row.

Every lever arm must match its denominator on attention backend + cudagraph mode;
the ONLY allowed differences are the intended lever delta (quant method, MoE kernel,
window/layer config, routing markers). One arm (d_kvq_e5m2) is a deliberate TRAP
CONTROL: it PASSES iff its backend FLIPS off the reference -- proving this checker
detects the P74 trap class rather than merely recording it.

MLA arms with unknown support status (ds_fp8block, ds_kvq, ds_win) are INFO arms:
their outcome (works / broken-cell) is an E0 *finding*, not a gate failure.
"""

import argparse
import re
import sys
from pathlib import Path

ATTN = re.compile(r"Using (\w+) attention backend out of potential")
CG = re.compile(r"cudagraph_mode': <CUDAGraphMode\.(\w+)")
MOE = re.compile(r"Using (\S+) (?:Fp8|Unquantized|MxFp8) MoE backend")
PREP = re.compile(r"Using (\S*Prepare\S*)")
QUANT = re.compile(r"quantization=([\w.-]+)")
PROBE = re.compile(r"^E0PROBE (.*)$", re.M)
KERNEL = re.compile(r"^E0KERNEL=(\S+)", re.M)
EXIT = re.compile(r"^E0EXIT=(\d+)", re.M)
ERR = re.compile(r"(ValueError|RuntimeError|AssertionError|CUDA error|NotImplementedError)[:\s].{0,110}")

# arm -> (ref_arm, [named extra checks]); ref None = is a reference row.
# extra checks: quant=X | moe~SUBSTR | kernel=X | probe:k=v | marker:SUBSTR
#               attn_flip (trap control) | info (never gates)
ARMS = {
    "d_bf16":      (None, []),
    "d_w4auto":    ("d_bf16", ["quant=compressed-tensors", "kernel=MacheteLinearKernel"]),
    "d_w4marlin":  ("d_bf16", ["quant=compressed-tensors", "kernel=MarlinLinearKernel"]),
    "d_fp8w8a8":   ("d_bf16", ["quant=fp8"]),
    "d_kvq":       ("d_bf16", []),
    "d_kvq_e5m2":  ("d_bf16", ["attn_flip"]),
    "d_win":       ("d_bf16", ["probe:sliding_window=512", "probe:max_model_len=1024"]),
    "d_bf16dummy": (None, []),
    "d_skip50":    ("d_bf16dummy", ["probe:num_hidden_layers=14"]),
    "m_bf16":      (None, []),
    "m_fp8marlin": ("m_bf16", ["quant=fp8", "moe~MARLIN"]),
    "m_fp8block":  ("m_bf16", ["quant=fp8_per_block", "moe~FLASHINFER_CUTLASS"]),
    "m_kvq":       ("m_bf16", []),
    "m_win":       ("m_bf16", ["probe:sliding_window=512", "probe:max_model_len=1024"]),
    "m_bf16dummy": (None, []),
    "m_skip50":    ("m_bf16dummy", ["probe:num_hidden_layers=24"]),
    "m_localroute": ("m_bf16", ["marker:LOCAL_ROUTE engaged"]),
    "m_skipa2a":   ("m_bf16", ["marker:SKIP_A2A engaged"]),
    "ds_bf16":     (None, []),
    "ds_fp8marlin": ("ds_bf16", ["quant=fp8", "info"]),
    "ds_fp8block": ("ds_bf16", ["quant=fp8_per_block", "info"]),
    "ds_kvq":      ("ds_bf16", ["info"]),
    "ds_win":      ("ds_bf16", ["probe:sliding_window=512", "info"]),
    "ds_bf16dummy": (None, []),
    "ds_skip50":   ("ds_bf16dummy", ["probe:num_hidden_layers=14"]),
}
GROUPS = {"dense": "d_", "moe": "m_", "mla": "ds_"}


def parse(log: Path) -> dict:
    t = log.read_text(errors="replace")
    status = log.with_suffix(log.suffix + ".status")
    if status.exists():  # markers live apart from the server's own fd (see runner)
        t += "\n" + status.read_text(errors="replace")
    d = {
        "attn": (ATTN.search(t) or [None, "?"])[1],
        "cg": (CG.search(t) or [None, "?"])[1],
        "moe": (MOE.search(t) or [None, "-"])[1],
        "prep": (PREP.search(t) or [None, "-"])[1],
        "quant": (QUANT.search(t) or [None, "-"])[1],
        "kernel": (KERNEL.search(t) or [None, "-"])[1],
        "probe": {},
        "markers": [],
        "done": bool(EXIT.search(t) and EXIT.search(t)[1] == "0"
                     and ("Avg latency" in t or "E0SERVE_OK" in t)),
        "err": "",
    }
    m = PROBE.findall(t)
    if m:
        d["probe"] = dict(kv.split("=", 1) for kv in m[-1].split() if "=" in kv)
    for mk in ("LOCAL_ROUTE engaged", "SKIP_A2A engaged"):
        if mk in t:
            d["markers"].append(mk)
    if not d["done"]:
        e = ERR.findall(t)
        d["err"] = e[-1][:110] if e else "no-completion"
    return d


def check(name: str, d: dict, ref: dict | None, extras: list[str]) -> tuple[str, list[str]]:
    info = "info" in extras
    fails = []
    if not d["done"]:
        fails.append(f"did-not-complete ({d['err']})")
        return ("INFO:BROKEN-CELL" if info else "FAIL"), fails
    if ref is not None:
        if "attn_flip" in extras:
            if d["attn"] == ref["attn"]:
                fails.append(f"TRAP NOT DETECTED: attn stayed {d['attn']}")
        elif d["attn"] != ref["attn"]:
            fails.append(f"attn {d['attn']} != ref {ref['attn']}")
        if d["cg"] != ref["cg"]:
            fails.append(f"cudagraph {d['cg']} != ref {ref['cg']}")
    for x in extras:
        if x in ("info", "attn_flip"):
            continue
        kind, _, want = x.partition(":") if ":" in x else ("", "", "")
        if x.startswith("quant="):
            if d["quant"] != x[6:]:
                fails.append(f"quant={d['quant']} want {x[6:]}")
        elif x.startswith("moe~"):
            if x[4:] not in d["moe"]:
                fails.append(f"moe={d['moe']} want ~{x[4:]}")
        elif x.startswith("kernel="):
            if d["kernel"] != x[7:]:
                fails.append(f"kernel={d['kernel']} want {x[7:]}")
        elif kind == "probe":
            k, _, v = want.partition("=")
            if d["probe"].get(k) != v:
                fails.append(f"probe {k}={d['probe'].get(k)} want {v}")
        elif kind == "marker":
            if want not in d["markers"]:
                fails.append(f"marker '{want}' absent")
    if fails:
        return ("INFO:BROKEN-CELL" if info else "FAIL"), fails
    return ("INFO:OK" if info else "PASS"), []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="all")
    a = ap.parse_args()
    logs = Path(a.logs)
    pref = GROUPS.get(a.group)

    rows, parsed, gate_ok = [], {}, True
    for name, (refname, extras) in ARMS.items():
        if pref and not name.startswith(pref):
            continue
        if a.group == "moe" and name.startswith("ds_"):  # 'm_' prefix collision guard
            continue
        f = logs / f"e0_{name}.log"
        if not f.exists():
            rows.append((name, "-", "-", "-", "-", "-", "-", "SKIP", "no log"))
            continue
        parsed[name] = parse(f)

    for name, (refname, extras) in ARMS.items():
        if name not in parsed:
            continue
        d = parsed[name]
        ref = parsed.get(refname) if refname else None
        if refname and ref is None:
            verdict, notes = "FAIL", [f"reference arm {refname} missing"]
        else:
            verdict, notes = check(name, d, ref, extras)
        if verdict == "FAIL":
            gate_ok = False
        extra = d["kernel"] if d["kernel"] != "-" else (
            ";".join(d["markers"]) or " ".join(f"{k}={v}" for k, v in d["probe"].items()) or "-")
        rows.append((name, d["attn"], d["cg"], d["moe"], d["prep"][:34], d["quant"],
                     extra[:38], verdict, "; ".join(notes)))

    hdr = ("arm", "attn", "cudagraph", "moe_backend", "prepare_finalize", "quant",
           "extra", "verdict", "notes")
    lines = ["# E0 infrastructure-parity table", "",
             "| " + " | ".join(hdr) + " |",
             "|" + "|".join("---" for _ in hdr) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    lines += ["", f"GATE: {'OPEN' if gate_ok else 'CLOSED'} (group={a.group})"]
    out = "\n".join(lines)
    Path(a.out).write_text(out + "\n")
    print(out)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
