#!/usr/bin/env python3
"""Generate paper-facing reporting tables for the convergence-gap draft.

The script intentionally reads only compact paper-synthesis JSON/CSV artifacts.
It does not rerun GPU experiments; it makes the manuscript tables reproducible
from the same summaries that the claim checker audits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("results/paper_synthesis")

ENDPOINT_TABLE = "results/paper_synthesis/exp22_endpoint_deconfounded_table.csv"
FIXED_HISTORY_IT_JSON = "results/paper_synthesis/exp22_fixed_history_template_audit.json"
FIXED_HISTORY_IT_EFFECTS = "results/paper_synthesis/exp22_fixed_history_template_audit_effects.csv"
FIXED_HISTORY_PT_JSON = "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit.json"
FIXED_HISTORY_PT_EFFECTS = "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit_effects.csv"
EXP9_SUMMARY = "results/exp09_cross_model_observational_replication/data/exp9_summary.json"
EXP11_METRICS = (
    "results/exp11_matched_prefix_mlp_graft/plots/"
    "exp11_exp3_600rand_v11_depthablation_full/depth_ablation_metrics.json"
)
EXP14_SUMMARY = (
    "results/exp14_symmetric_matched_prefix_causality/"
    "exp13exp14_full_20260416/exp13_full_summary.json"
)
EXP55_EFFECTS = "results/paper_synthesis/exp55_late_window_robustness_effects.csv"

SOURCE_ARTIFACTS = [
    ENDPOINT_TABLE,
    FIXED_HISTORY_IT_JSON,
    FIXED_HISTORY_IT_EFFECTS,
    FIXED_HISTORY_PT_JSON,
    FIXED_HISTORY_PT_EFFECTS,
    EXP9_SUMMARY,
    EXP11_METRICS,
    EXP14_SUMMARY,
    EXP55_EFFECTS,
]

MODEL_ORDER = [
    "gemma3_4b",
    "llama31_8b",
    "qwen3_4b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]
DENSE_ORDER = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]


@dataclass(frozen=True)
class Table:
    key: str
    title: str
    headers: list[str]
    rows: list[list[str]]
    align: list[str]

    @property
    def markdown(self) -> str:
        return markdown_table(self.headers, self.rows, self.align)


def load_json(repo: Path, relpath: str) -> Any:
    return json.loads((repo / relpath).read_text())


def load_csv(repo: Path, relpath: str) -> list[dict[str, str]]:
    with (repo / relpath).open(newline="") as f:
        return list(csv.DictReader(f))


def by_key(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    for row in rows:
        if row.get("key") == key:
            return row
    raise KeyError(key)


def fmt_num(value: float | str, digits: int = 3, *, signed: bool = True) -> str:
    number = float(value)
    sign = "+" if signed and number >= 0 else ""
    return f"{sign}{number:.{digits}f}"


def fmt_ci(row: dict[str, str], value_col: str, *, digits: int = 3, unit: str = "") -> str:
    value = fmt_num(row[value_col], digits)
    lo = row.get("ci95_low", "")
    hi = row.get("ci95_high", "")
    unit_part = f" {unit}" if unit else ""
    if lo != "" and hi != "":
        return f"`{value}`{unit_part} `[{fmt_num(lo, digits)}, {fmt_num(hi, digits)}]`"
    return f"`{value}`{unit_part}"


def fmt_effect(row: dict[str, str], *, digits: int = 3) -> str:
    return fmt_ci(row, "estimate", digits=digits)


def fmt_signed(value: float, digits: int = 3) -> str:
    return f"`{fmt_num(value, digits)}`"


def fmt_int(value: float | int | str) -> str:
    return f"`{int(float(value)):,}`"


def markdown_table(headers: list[str], rows: list[list[str]], align: list[str]) -> str:
    if len(headers) != len(align):
        raise ValueError("header and alignment lengths differ")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row has {len(row)} cells for {len(headers)} headers")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_endpoint_controls(repo: Path) -> Table:
    rows = load_csv(repo, ENDPOINT_TABLE)
    specs = [
        (
            "endpoint_matched_raw_late_kl",
            "Endpoint-matched raw late KL",
            "nats",
            "raw-lens gap remains",
        ),
        (
            "endpoint_matched_tuned_late_kl",
            "Endpoint-matched tuned late KL",
            "nats",
            "tuned-lens gap remains",
        ),
        (
            "endpoint_free_remaining_adj_js",
            "Endpoint-free adjacent JS",
            "JS",
            "IT has more remaining layer-to-layer movement",
        ),
        (
            "endpoint_free_future_top1_flips",
            "Endpoint-free future top-1 flips",
            "flips",
            "IT changes top-1 later",
        ),
    ]
    table_rows = [
        [label, fmt_ci(by_key(rows, key), "estimate_it_minus_pt", unit=unit), interpretation]
        for key, label, unit, interpretation in specs
    ]
    return Table(
        key="endpoint_controls",
        title="Endpoint and history controls",
        headers=["Check", "Estimate", "Interpretation"],
        rows=table_rows,
        align=["---", "---:", "---"],
    )


def build_intervention_windows(repo: Path) -> Table:
    exp11 = load_json(repo, EXP11_METRICS)
    exp14 = load_json(repo, EXP14_SUMMARY)
    graft = exp11["dense_family_means"]
    swap = exp14["dense_family_means"]["it_side_final20_kl_delta"]
    table_rows = [
        [
            "IT MLP graft into PT host",
            fmt_signed(graft["B_early_raw"]["final_20pct_delta_kl_to_own_final"], 2),
            fmt_signed(graft["B_mid_raw"]["final_20pct_delta_kl_to_own_final"], 2),
            fmt_signed(graft["B_late_raw"]["final_20pct_delta_kl_to_own_final"], 2),
        ],
        [
            "PT MLP swap into IT host",
            fmt_signed(swap["D_early_ptswap"], 2),
            fmt_signed(swap["D_mid_ptswap"], 2),
            fmt_signed(swap["D_late_ptswap"], 2),
        ],
    ]
    return Table(
        key="intervention_windows",
        title="Dense-family matched-prefix intervention summary",
        headers=["Intervention", "Early", "Middle", "Late"],
        rows=table_rows,
        align=["---", "---:", "---:", "---:"],
    )


def build_fixed_history_replay(repo: Path) -> Table:
    it_effects = load_csv(repo, FIXED_HISTORY_IT_EFFECTS)
    pt_effects = load_csv(repo, FIXED_HISTORY_PT_EFFECTS)
    it_quality = load_json(repo, FIXED_HISTORY_IT_JSON)["quality"]
    pt_quality = load_json(repo, FIXED_HISTORY_PT_JSON)["quality"]

    def effects(
        rows: list[dict[str, str]],
        prefix: str,
        method: str,
    ) -> list[str]:
        method_token = "" if method == "paired_same_prompt_step" else "cem_"
        return [
            fmt_effect(by_key(rows, f"{prefix}_native_fixed_{method_token}raw_late_kl")),
            fmt_effect(by_key(rows, f"{prefix}_raw_fixed_{method_token}raw_late_kl")),
            fmt_effect(by_key(rows, f"{prefix}_template_delta_{method_token}raw_late_kl")),
        ]

    table_rows = [
        [
            "IT-native continuation",
            "paired same prompt/step",
            *effects(it_effects, "it_native", "paired_same_prompt_step"),
            "primary fixed-history replay",
        ],
        [
            "IT-native continuation",
            "endpoint-matched CEM",
            *effects(it_effects, "it_native", "cem"),
            f"retention `{float(it_quality['min_retained_fraction']):.3f}`; max SMD `{float(it_quality['max_smd_after']):.3f}`",
        ],
        [
            "PT-raw continuation",
            "paired same prompt/step",
            *effects(pt_effects, "pt_raw", "paired_same_prompt_step"),
            "same-sign reverse stress test",
        ],
        [
            "PT-raw continuation",
            "endpoint-matched CEM",
            *effects(pt_effects, "pt_raw", "cem"),
            f"balance caveat: retention `{float(pt_quality['min_retained_fraction']):.3f}`; max SMD `{float(pt_quality['max_smd_after']):.3f}`",
        ],
    ]
    return Table(
        key="fixed_history_replay",
        title="Fixed-history replay audit",
        headers=[
            "Teacher history",
            "Estimator",
            "IT native - PT raw",
            "IT raw - PT raw",
            "IT native - IT raw",
            "Use/quality",
        ],
        rows=table_rows,
        align=["---", "---", "---:", "---:", "---:", "---"],
    )


def build_discovery_counts(repo: Path) -> Table:
    summary = load_json(repo, EXP9_SUMMARY)
    rows = []
    for model in MODEL_ORDER:
        row = summary[model]
        rows.append(
            [
                row["label"],
                fmt_int(row["tuned_lens_raw_kl_final_n_steps_pt"]),
                fmt_int(row["tuned_lens_raw_kl_final_n_steps_it"]),
                fmt_int(row["n_layers"]),
            ]
        )
    return Table(
        key="discovery_counts",
        title="Discovery-curve token-step counts",
        headers=["Family", "PT token steps", "IT token steps", "Layers"],
        rows=rows,
        align=["---", "---:", "---:", "---:"],
    )


def build_late_window_family(repo: Path) -> Table:
    exp9 = load_json(repo, EXP9_SUMMARY)
    exp11 = load_json(repo, EXP11_METRICS)
    exp14 = load_json(repo, EXP14_SUMMARY)
    exp11_models = {row["model"]: row for row in exp11["models"]}
    exp14_models = exp14["models"]

    rows = []
    for model in DENSE_ORDER:
        graft = exp11_models[model]["pipelines"]["B_late_raw"]
        swap = exp14_models[model]["it_side"]["D_late_ptswap"]
        rows.append(
            [
                exp9[model]["label"],
                fmt_signed(graft["regions"]["final_20pct"]["kl_to_own_final"]["delta"]),
                fmt_signed(swap["regions"]["final_20pct"]["kl_to_own_final"]["delta"]),
                f"`{graft['graft_window']['display_range']}`",
            ]
        )
    rows.append(
        [
            "Dense mean",
            fmt_signed(exp11["dense_family_means"]["B_late_raw"]["final_20pct_delta_kl_to_own_final"]),
            fmt_signed(exp14["dense_family_means"]["it_side_final20_kl_delta"]["D_late_ptswap"]),
            "-",
        ]
    )
    return Table(
        key="late_window_family",
        title="Per-family late-window effects",
        headers=["Family", "Late IT graft into PT host", "Late PT swap into IT host", "Late window"],
        rows=rows,
        align=["---", "---:", "---:", "---"],
    )


def build_late_window_width_center(repo: Path) -> Table:
    effects = load_csv(repo, EXP55_EFFECTS)

    def effect(window: str, side: str, region: str) -> str:
        for row in effects:
            if (
                row["level"] == "dense5_model_mean"
                and row["window_name"] == window
                and row["side"] == side
                and row["region"] == region
                and row["metric"] == "kl_to_own_final"
            ):
                return fmt_signed(row["estimate"])
        raise KeyError((window, side, region))

    specs = [
        ("prelate_half", "Pre-late half"),
        ("late_full", "Late full"),
        ("late_front_half", "Late front half"),
        ("late_center_half", "Late center half"),
        ("late_terminal_half", "Late terminal half"),
        ("terminal_quarter", "Terminal quarter"),
    ]
    rows = [
        [
            label,
            effect(window, "it_graft_into_pt", "final_20pct"),
            effect(window, "pt_swap_into_it", "final_20pct"),
            effect(window, "it_graft_into_pt", "graft_window"),
            effect(window, "pt_swap_into_it", "graft_window"),
        ]
        for window, label in specs
    ]
    return Table(
        key="late_window_width_center",
        title="Late-window width/center audit",
        headers=[
            "Window",
            "Final-20 IT graft into PT",
            "Final-20 PT swap into IT",
            "Edited-window IT graft into PT",
            "Edited-window PT swap into IT",
        ],
        rows=rows,
        align=["---", "---:", "---:", "---:", "---:"],
    )


def build_tables(repo: Path) -> list[Table]:
    return [
        build_endpoint_controls(repo),
        build_intervention_windows(repo),
        build_fixed_history_replay(repo),
        build_discovery_counts(repo),
        build_late_window_family(repo),
        build_late_window_width_center(repo),
    ]


def payload(tables: list[Table]) -> dict[str, Any]:
    return {
        "description": "Generated convergence-gap reporting tables.",
        "source_artifacts": SOURCE_ARTIFACTS,
        "tables": {
            table.key: {
                "title": table.title,
                "headers": table.headers,
                "rows": table.rows,
                "align": table.align,
                "markdown": table.markdown,
            }
            for table in tables
        },
    }


def render_md(tables: list[Table]) -> str:
    chunks = [
        "<!-- Generated by scripts/analysis/build_convergence_gap_reporting_tables.py. -->",
        "",
    ]
    for table in tables:
        chunks.extend(
            [
                f"## {table.title}",
                "",
                f"<!-- REPORT_TABLE: {table.key} -->",
                table.markdown,
                "<!-- /REPORT_TABLE -->",
                "",
            ]
        )
    return "\n".join(chunks).rstrip() + "\n"


def render_csv(tables: list[Table]) -> str:
    lines: list[list[str]] = [["table", "row_index", "row_label", "column", "value"]]
    for table in tables:
        for row_index, row in enumerate(table.rows):
            row_label = row[0]
            for header, value in zip(table.headers, row):
                lines.append([table.key, str(row_index), row_label, header, value])
    out: list[str] = []
    for row in lines:
        escaped = []
        for cell in row:
            if any(ch in cell for ch in [",", '"', "\n"]):
                escaped.append('"' + cell.replace('"', '""') + '"')
            else:
                escaped.append(cell)
        out.append(",".join(escaped))
    return "\n".join(out) + "\n"


def expected_outputs(repo: Path, out_dir: Path) -> dict[Path, str]:
    tables = build_tables(repo)
    data = payload(tables)
    out_root = repo / out_dir
    return {
        out_root / "convergence_gap_reporting_tables.json": json.dumps(data, indent=2, sort_keys=True) + "\n",
        out_root / "convergence_gap_reporting_tables.md": render_md(tables),
        out_root / "convergence_gap_reporting_tables.csv": render_csv(tables),
    }


def write_outputs(outputs: dict[Path, str]) -> None:
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def check_outputs(outputs: dict[Path, str]) -> list[str]:
    failures = []
    for path, expected in outputs.items():
        if not path.exists():
            failures.append(f"missing generated table artifact: {path}")
            continue
        observed = path.read_text()
        if observed != expected:
            failures.append(f"stale generated table artifact: {path}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true", help="fail if generated outputs are stale")
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_dir = args.out_dir
    outputs = expected_outputs(repo, out_dir)
    if args.check:
        failures = check_outputs(outputs)
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        print(f"ok   generated reporting tables current: {len(outputs)} artifacts")
        return 0

    write_outputs(outputs)
    for path in outputs:
        print(path.relative_to(repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
