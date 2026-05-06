"""Analyze Exp53 controlled-domain fine-tune foil outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .common import DOMAIN_TO_ALIAS, paths_for, write_json


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value


def _factorial_rows(run_root: Path, support_name: str) -> list[dict[str, Any]]:
    path = run_root / "factorial" / support_name / "analysis" / "portable_coadapted_table.csv"
    rows = []
    for row in _read_csv(path):
        if row.get("scope") != "alias" or row.get("readout") != "common_it":
            continue
        if row.get("model") not in DOMAIN_TO_ALIAS.values():
            continue
        if row.get("slice") not in {"full_1400", "full_600"} and not str(row.get("slice", "")).startswith("full"):
            # Domain support categories are intentionally custom and often only
            # appear in the generic full slice.
            continue
        rows.append({"support": support_name, **row})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sanity: dict[str, Any] = {}
    for domain, alias in DOMAIN_TO_ALIAS.items():
        paths = paths_for(args.run_root, domain)
        eval_loss = _read_json(paths.eval_loss) if paths.eval_loss.exists() else {"missing": str(paths.eval_loss)}
        merge = _read_json(paths.merge_check) if paths.merge_check.exists() else {"missing": str(paths.merge_check)}
        health = _read_json(paths.generation_health) if paths.generation_health.exists() else {"missing": str(paths.generation_health)}
        data_manifest = _read_json(paths.data_manifest) if paths.data_manifest.exists() else {"missing": str(paths.data_manifest)}
        sanity[domain] = {
            "alias": alias,
            "eval_loss": eval_loss,
            "merge_equivalence": merge,
            "generation_health": health,
            "data_manifest": data_manifest,
        }
    for support_name in ("main_eval", "code_support", "biomed_support"):
        rows.extend(_factorial_rows(args.run_root, support_name))

    for row in rows:
        model = str(row.get("model"))
        domain = next((d for d, a in DOMAIN_TO_ALIAS.items() if a == model), "unknown")
        row["domain"] = domain
        row["nll_relative_improvement"] = sanity.get(domain, {}).get("eval_loss", {}).get("relative_nll_improvement")
        row["domain_gate"] = sanity.get(domain, {}).get("eval_loss", {}).get("passes_domain_gate")
        m = _float(row, "M")
        p = _float(row, "P")
        c = _float(row, "C")
        row["portable_share_P_over_M_recomputed"] = None if m in (None, 0.0) or p is None else p / m
        row["coadapted_share_C_over_M_recomputed"] = None if m in (None, 0.0) or c is None else c / m

    _write_csv(out_dir / "domain_foil_table.csv", rows)
    summary = {
        "experiment": "exp53_controlled_domain_finetunes",
        "run_root": str(args.run_root),
        "sanity": sanity,
        "n_factorial_summary_rows": len(rows),
        "supports_present": sorted({row["support"] for row in rows}),
        "models_present": sorted({row["model"] for row in rows}),
    }
    write_json(out_dir / "summary.json", summary)
    _plot(rows, out_dir)
    _write_report(rows, summary, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    main = [row for row in rows if row.get("support") == "main_eval"]
    if not main:
        return
    labels = [str(row["domain"]) for row in main]
    c_vals = [_float(row, "C") or 0.0 for row in main]
    p_share = [_float(row, "portable_share_P_over_M_recomputed") or 0.0 for row in main]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(labels, c_vals, color="#4c78a8")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Co-adapted Extra C")
    axes[0].set_ylabel("logit margin")
    axes[1].bar(labels, p_share, color="#f58518")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Portable Share P/M")
    axes[1].set_ylabel("ratio")
    fig.tight_layout()
    fig.savefig(out_dir / "controlled_domain_foils.png", dpi=180)
    plt.close(fig)


def _write_report(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    lines = ["# Exp53 Controlled Domain Foils", ""]
    lines.append(f"Factorial summary rows: {summary['n_factorial_summary_rows']}")
    lines.append("")
    for row in rows:
        if row.get("support") != "main_eval":
            continue
        lines.append(
            "- {domain}: C={C}, P/M={portable}, NLL improvement={nll}".format(
                domain=row.get("domain"),
                C=row.get("C"),
                portable=row.get("portable_share_P_over_M_recomputed"),
                nll=row.get("nll_relative_improvement"),
            )
        )
    (out_dir / "paper_claims_exp53.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

