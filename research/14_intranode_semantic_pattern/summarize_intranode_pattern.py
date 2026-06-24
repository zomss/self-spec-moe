#!/usr/bin/env python3
"""Summarize the intra-node Semantic-Parallelism-like pattern."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "research" / "14_intranode_semantic_pattern"
INPUTS = {
    "Qwen3-30B-A3B": (
        ROOT
        / "research"
        / "13_affinity_placement_gated_draft"
        / "data"
        / "qwen3_affinity.json"
    ),
    "GPT-OSS-20B": (
        ROOT
        / "research"
        / "13_affinity_placement_gated_draft"
        / "data"
        / "gptoss_affinity.json"
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, path in INPUTS.items():
        data = load_json(path)
        configs = data["by_config"]
        groups = sorted(
            int(key.rsplit("G", 1)[1])
            for key in configs
            if key.startswith("affinity_G")
        )
        for group in groups:
            affinity = configs[f"affinity_G{group}"]
            contiguous = configs[f"contiguous_G{group}"]
            affinity_lar = float(affinity["LAR_mean"])
            contiguous_lar = float(contiguous["LAR_mean"])
            affinity_remote = 1.0 - affinity_lar
            contiguous_remote = 1.0 - contiguous_lar
            remote_ratio = affinity_remote / contiguous_remote
            acceptance = affinity.get("acceptance", {})
            rows.append(
                {
                    "model": model_name,
                    "num_groups": group,
                    "contiguous_lar": contiguous_lar,
                    "affinity_lar": affinity_lar,
                    "lar_abs_gain": affinity_lar - contiguous_lar,
                    "lar_rel_gain": affinity_lar / contiguous_lar - 1.0,
                    "contiguous_remote_volume": contiguous_remote,
                    "affinity_remote_volume": affinity_remote,
                    "remote_volume_vs_contiguous": remote_ratio,
                    "remote_volume_reduction": 1.0 - remote_ratio,
                    "draftable_frac_ge_0_9": safe_float(
                        affinity.get("draftable_frac_ge_thr")
                    ),
                    "acceptance_all_steps": safe_float(
                        acceptance.get("all_steps_mean_sampled")
                    ),
                    "acceptance_draftable": safe_float(
                        acceptance.get("draftable_mean_sampled")
                    ),
                    "draftable_n": acceptance.get("draftable_n"),
                }
            )
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.{digits}f}%"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def write_svg(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 920
    height = 360
    left = 70
    right = 20
    top = 35
    bottom = 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    group_w = chart_w / len(rows)
    bar_w = min(34, group_w / 4)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="460" y="22" text-anchor="middle" '
            'font-family="sans-serif" font-size="16" font-weight="bold">'
            "Local activation rate: contiguous vs affinity"
            "</text>"
        ),
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = top + chart_h * (1 - value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
            'font-family="sans-serif" font-size="11">'
            f"{value:.1f}</text>"
        )

    for idx, row in enumerate(rows):
        center = left + group_w * (idx + 0.5)
        labels = [
            ("contiguous", row["contiguous_lar"], "#9ca3af", -bar_w / 1.7),
            ("affinity", row["affinity_lar"], "#2563eb", bar_w / 1.7),
        ]
        for name, value, color, offset in labels:
            h = chart_h * float(value)
            x = center + offset - bar_w / 2
            y = top + chart_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" fill="{color}">'
                f"<title>{html.escape(name)} LAR {value:.3f}</title></rect>"
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" '
                'text-anchor="middle" font-family="sans-serif" font-size="10">'
                f"{value:.2f}</text>"
            )
        label = f'{row["model"].replace("-A3B", "")} G{row["num_groups"]}'
        parts.append(
            f'<text x="{center:.1f}" y="{height-bottom+24}" '
            'text-anchor="middle" font-family="sans-serif" font-size="11">'
            f"{html.escape(label)}</text>"
        )
        parts.append(
            f'<text x="{center:.1f}" y="{height-bottom+40}" '
            'text-anchor="middle" font-family="sans-serif" font-size="10">'
            f'remote -{pct(row["remote_volume_reduction"], 0)}</text>'
        )

    parts.extend(
        [
            f'<line x1="{left}" y1="{top + chart_h}" x2="{width-right}" '
            f'y2="{top + chart_h}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + chart_h}" stroke="#111827"/>',
            '<rect x="680" y="40" width="14" height="14" fill="#9ca3af"/>',
            '<text x="700" y="52" font-family="sans-serif" font-size="12">'
            "contiguous</text>",
            '<rect x="680" y="60" width="14" height="14" fill="#2563eb"/>',
            '<text x="700" y="72" font-family="sans-serif" font-size="12">'
            "affinity</text>",
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Groups | Contig LAR | Affinity LAR | LAR gain | "
        "Remote-volume reduction | All-step acc | Draftable frac | "
        "Draftable acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {group} | {contig} | {aff} | {gain} | {remote} | "
            "{acc_all} | {draft_frac} | {acc_draft} |".format(
                model=row["model"],
                group=row["num_groups"],
                contig=fmt(row["contiguous_lar"]),
                aff=fmt(row["affinity_lar"]),
                gain=pct(row["lar_rel_gain"]),
                remote=pct(row["remote_volume_reduction"]),
                acc_all=fmt(row["acceptance_all_steps"]),
                draft_frac=pct(row["draftable_frac_ge_0_9"]),
                acc_draft=fmt(row["acceptance_draftable"]),
            )
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, Any]], path: Path) -> None:
    by_model_group = {
        (row["model"], row["num_groups"]): row
        for row in rows
    }
    qwen_g2 = by_model_group[("Qwen3-30B-A3B", 2)]
    qwen_g4 = by_model_group[("Qwen3-30B-A3B", 4)]
    gpt_g2 = by_model_group[("GPT-OSS-20B", 2)]
    gpt_g4 = by_model_group[("GPT-OSS-20B", 4)]
    report = f"""# Results: Intra-node Semantic-Parallelism Pattern

Date: 2026-06-24

## Question

Can our current MoE routing traces show the same intra-node pattern as
Speculative MoE / Semantic Parallelism: higher local activation rate after
affinity-aware expert placement and request scheduling?

## Result

Yes at the routing/communication-volume level. Affinity placement plus oracle
request-to-group scheduling consistently increases LAR over contiguous placement
and cuts the remote-volume proxy `(1 - LAR)`.

![LAR comparison](figures/lar_comparison.svg)

{markdown_table(rows)}

## Reading

**G=2 reproduces the paper-like intra-node pattern strongly.** Qwen3 raises LAR
from {fmt(qwen_g2["contiguous_lar"])} to {fmt(qwen_g2["affinity_lar"])}, cutting
remote volume by {pct(qwen_g2["remote_volume_reduction"])}. GPT-OSS raises LAR
from {fmt(gpt_g2["contiguous_lar"])} to {fmt(gpt_g2["affinity_lar"])}, cutting
remote volume by {pct(gpt_g2["remote_volume_reduction"])}. This is the same
qualitative shape as the paper's intra-node result: better semantic grouping
turns remote expert routing into local expert routing.

**G=4 still improves locality but is no longer enough for a useful draft.** LAR
improves for both models, but the absolute LAR remains below 0.5 and no step
crosses the Phase 13 draftable threshold (`coverage >= 0.9`). The local-draft
acceptance proxy is also much lower: Qwen3 {fmt(qwen_g4["acceptance_all_steps"])}
and GPT-OSS {fmt(gpt_g4["acceptance_all_steps"])}.

**This phase is not an end-to-end speedup claim.** The output is a locality and
remote-volume proxy derived from Phase 13 traces. Phase 12 already showed that
single-node NVLink EP-width changes move vLLM decode latency only slightly, so
large end-to-end gains are unlikely without an explicit communication kernel
microbenchmark or a slower/heterogeneous intra-node fabric.

## Conclusion

It is not hard to show the same *pattern* as the paper: our traces already show
that affinity/request-aware scheduling increases LAR and reduces remote routing,
especially at `G=2`. It is hard to show a strong end-to-end speedup on this
single NVLink node, because the intra-node all-to-all is too cheap relative to
full decode compute.

## Next step

Run a controlled intra-node all-to-all/all-to-allv microbenchmark where LAR is
swept from contiguous-like to affinity-like values. That would convert this
volume proxy into the latency-vs-LAR curve needed to mirror the paper's Figure 5
more directly.
"""
    path.write_text(report)


def main() -> int:
    rows = build_rows()
    write_csv(rows, PHASE / "data" / "intranode_semantic_pattern.csv")
    write_json(rows, PHASE / "data" / "intranode_semantic_pattern.json")
    write_svg(rows, PHASE / "figures" / "lar_comparison.svg")
    write_report(rows, PHASE / "results_intranode_semantic_pattern.md")
    print(markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
