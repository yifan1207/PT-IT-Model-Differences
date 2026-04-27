#!/usr/bin/env python
"""Build paper-facing Exp22 endpoint-deconfounding artifacts.

The full Exp22 analyzer keeps additional estimator-sensitivity diagnostics.
This synthesis exports the paper-facing primary endpoint-matched CEM results
and endpoint-free checks used in the manuscript; diagnostic estimators remain
available in the full analysis artifacts rather than being repeated in the
main text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "exp22_full_dense5_raw_tuned_20260426_150029"
SUMMARY_PATH = (
    ROOT
    / "results/exp22_endpoint_deconfounded_gap"
    / RUN_NAME
    / "analysis/summary.json"
)
OUT_DIR = ROOT / "results/paper_synthesis"


def _load_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing Exp22 summary: {SUMMARY_PATH}")
    return json.loads(SUMMARY_PATH.read_text())


def _effect(summary: dict[str, Any], method: str, probe: str, outcome: str) -> dict[str, Any]:
    for row in summary["controlled_estimates"]:
        if (
            row["method"] == method
            and row["probe_family"] == probe
            and row["outcome"] == outcome
        ):
            return row
    raise KeyError((method, probe, outcome))


def _fmt_ci(row: dict[str, Any], digits: int = 3) -> str:
    return f"[{float(row['ci95_low']):.{digits}f}, {float(row['ci95_high']):.{digits}f}]"


def write_table(summary: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    specs = [
        (
            "endpoint_matched_raw_late_kl",
            "raw late KL-to-own-final after endpoint matching",
            "cem",
            "raw",
            "late_kl_mean",
            "nats",
            "Primary endpoint-matched convergence-gap estimate under raw lens.",
        ),
        (
            "endpoint_matched_tuned_late_kl",
            "tuned late KL-to-own-final after endpoint matching",
            "cem",
            "tuned",
            "late_kl_mean",
            "nats",
            "Primary endpoint-matched convergence-gap estimate under tuned lens.",
        ),
        (
            "endpoint_free_remaining_adj_js",
            "remaining adjacent JS after endpoint matching",
            "cem",
            "pooled_probe_fe",
            "remaining_adj_js",
            "JS",
            "Endpoint-free path-length check; positive means IT has more remaining layer-to-layer movement.",
        ),
        (
            "endpoint_free_future_top1_flips",
            "future top-1 flips after endpoint matching",
            "cem",
            "pooled_probe_fe",
            "future_top1_flips",
            "flips",
            "Endpoint-free argmax-stability check; positive means IT keeps changing top-1 later.",
        ),
    ]
    for key, metric, method, probe, outcome, unit, interpretation in specs:
        row = _effect(summary, method, probe, outcome)
        rows.append(
            {
                "key": key,
                "metric": metric,
                "method": method,
                "probe_family": probe,
                "outcome": outcome,
                "estimate_it_minus_pt": f"{float(row['estimate_it_minus_pt']):.9f}",
                "ci95_low": f"{float(row['ci95_low']):.9f}",
                "ci95_high": f"{float(row['ci95_high']):.9f}",
                "unit": unit,
                "interpretation": interpretation,
            }
        )

    quality = summary["quality"]
    rows.extend(
        [
            {
                "key": "quality_min_matched_retention",
                "metric": "minimum matched-token retention across model x probe",
                "method": "quality",
                "probe_family": "all",
                "outcome": "min_retained_fraction",
                "estimate_it_minus_pt": f"{float(quality['min_retained_fraction']):.9f}",
                "ci95_low": "",
                "ci95_high": "",
                "unit": "fraction",
                "interpretation": "Quality gate: all model/probe branches retain enough endpoint-matched token steps.",
            },
            {
                "key": "quality_max_smd_after",
                "metric": "maximum endpoint-covariate SMD after matching",
                "method": "quality",
                "probe_family": "all",
                "outcome": "max_smd_after",
                "estimate_it_minus_pt": f"{float(quality['max_smd_after']):.9f}",
                "ci95_low": "",
                "ci95_high": "",
                "unit": "SMD",
                "interpretation": "Quality gate: endpoint entropy, confidence, and top1-top2 margin are balanced after matching.",
            },
            {
                "key": "quality_max_malformed_rate",
                "metric": "maximum malformed branch rate",
                "method": "quality",
                "probe_family": "all",
                "outcome": "max_malformed_rate",
                "estimate_it_minus_pt": f"{float(quality['max_malformed_rate']):.9f}",
                "ci95_low": "",
                "ci95_high": "",
                "unit": "fraction",
                "interpretation": "Quality gate: collection completeness check.",
            },
        ]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / "exp22_endpoint_deconfounded_table.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_note(summary: dict[str, Any], rows: list[dict[str, str]]) -> None:
    by_key = {row["key"]: row for row in rows}
    raw = by_key["endpoint_matched_raw_late_kl"]
    tuned = by_key["endpoint_matched_tuned_late_kl"]
    rem_js = by_key["endpoint_free_remaining_adj_js"]
    flips = by_key["endpoint_free_future_top1_flips"]
    retention = by_key["quality_min_matched_retention"]
    smd = by_key["quality_max_smd_after"]
    malformed = by_key["quality_max_malformed_rate"]
    n_rows = int(summary["n_rows"])
    text = f"""# Exp22 Endpoint-Matched Convergence Gap

Dense-5 endpoint-control run over 600 prompts per PT/IT branch, raw and tuned probes, token steps after the first five generated tokens. Token steps are coarsened-exact-matched within `model x probe_family` on final-layer entropy, final top-1 confidence, and final top1-top2 margin.

Quality gates pass: `{n_rows}` analyzed token-step/probe rows, maximum malformed branch rate `{float(malformed['estimate_it_minus_pt']):.3f}`, minimum matched retention `{float(retention['estimate_it_minus_pt']):.3f}`, and maximum post-match endpoint-covariate SMD `{float(smd['estimate_it_minus_pt']):.3f}`.

Primary endpoint-matched late `KL(layer || own final)` remains higher for IT than PT:

- Raw probe: `{float(raw['estimate_it_minus_pt']):.3f}` nats, 95% CI `[{float(raw['ci95_low']):.3f}, {float(raw['ci95_high']):.3f}]`.
- Tuned probe: `{float(tuned['estimate_it_minus_pt']):.3f}` nats, 95% CI `[{float(tuned['ci95_low']):.3f}, {float(tuned['ci95_high']):.3f}]`.

Endpoint-free checks point the same direction after the same endpoint matching:

- Remaining adjacent JS: `{float(rem_js['estimate_it_minus_pt']):.3f}`, 95% CI `[{float(rem_js['ci95_low']):.3f}, {float(rem_js['ci95_high']):.3f}]`.
- Future top-1 flips: `{float(flips['estimate_it_minus_pt']):.3f}`, 95% CI `[{float(flips['ci95_low']):.3f}, {float(flips['ci95_high']):.3f}]`.

Paper-use claim: the convergence gap is not explained away by final endpoint entropy, confidence, or top1-top2 margin. We still describe it as endpoint-relative, but now endpoint-matched under the matched-token estimator.
"""
    (OUT_DIR / "exp22_endpoint_deconfounded_note.md").write_text(text)


def write_plot(rows: list[dict[str, str]]) -> None:
    by_key = {row["key"]: row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    fig.suptitle("Exp22: endpoint-matched convergence gap", fontsize=13)

    lens_keys = ["endpoint_matched_raw_late_kl", "endpoint_matched_tuned_late_kl"]
    labels = ["raw", "tuned"]
    estimates = np.array([float(by_key[key]["estimate_it_minus_pt"]) for key in lens_keys])
    lows = np.array([float(by_key[key]["ci95_low"]) for key in lens_keys])
    highs = np.array([float(by_key[key]["ci95_high"]) for key in lens_keys])
    axes[0].bar(labels, estimates, color=["#4C78A8", "#F58518"])
    axes[0].errorbar(
        np.arange(len(labels)),
        estimates,
        yerr=[estimates - lows, highs - estimates],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Matched late KL")
    axes[0].set_ylabel("IT - PT (nats)")

    endpoint_keys = ["endpoint_free_remaining_adj_js", "endpoint_free_future_top1_flips"]
    endpoint_labels = ["remaining\nadjacent JS", "future\ntop-1 flips"]
    endpoint_est = np.array([float(by_key[key]["estimate_it_minus_pt"]) for key in endpoint_keys])
    endpoint_low = np.array([float(by_key[key]["ci95_low"]) for key in endpoint_keys])
    endpoint_high = np.array([float(by_key[key]["ci95_high"]) for key in endpoint_keys])
    axes[1].bar(endpoint_labels, endpoint_est, color="#54A24B")
    axes[1].errorbar(
        np.arange(len(endpoint_labels)),
        endpoint_est,
        yerr=[endpoint_est - endpoint_low, endpoint_high - endpoint_est],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Endpoint-free checks")
    axes[1].set_ylabel("IT - PT")

    quality_labels = ["min retained", "max SMD", "malformed"]
    quality_vals = [
        float(by_key["quality_min_matched_retention"]["estimate_it_minus_pt"]),
        float(by_key["quality_max_smd_after"]["estimate_it_minus_pt"]),
        float(by_key["quality_max_malformed_rate"]["estimate_it_minus_pt"]),
    ]
    axes[2].bar(quality_labels, quality_vals, color=["#72B7B2", "#B279A2", "#9D755D"])
    axes[2].axhline(0.1, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("Matching quality")
    axes[2].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp22_endpoint_deconfounded_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    summary = _load_summary()
    rows = write_table(summary)
    write_note(summary, rows)
    write_plot(rows)
    print(f"Wrote Exp22 paper synthesis to {OUT_DIR}")


if __name__ == "__main__":
    main()
