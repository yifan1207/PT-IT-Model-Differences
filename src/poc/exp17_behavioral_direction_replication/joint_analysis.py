"""Joint analysis for exp17 against existing convergence-gap summaries."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.poc.exp17_behavioral_direction_replication.shared import VALID_MODELS, VALID_VARIANTS


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


DEFAULT_EXP9_SUMMARY = Path("results/exp09_cross_model_observational_replication/data/convergence_gap_values.json")
DEFAULT_EXP17_ROOT = Path("results/exp17_behavioral_direction_replication")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exp17 layers against convergence-gap peaks.")
    parser.add_argument("--exp9-summary", default=str(DEFAULT_EXP9_SUMMARY))
    parser.add_argument("--exp17-root", default=str(DEFAULT_EXP17_ROOT))
    parser.add_argument("--lens", default="tuned", choices=["tuned", "raw"])
    parser.add_argument("--out", default=str(DEFAULT_EXP17_ROOT / "joint_analysis" / "summary.json"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _pick_layer(summary: dict[str, Any]) -> tuple[int | None, str | None]:
    if summary.get("selected_layer") is not None:
        return int(summary["selected_layer"]), "selected_layer"
    if summary.get("peak_direction_norm_layer") is not None:
        return int(summary["peak_direction_norm_layer"]), "peak_direction_norm_layer"
    return None, None


def main() -> None:
    args = parse_args()
    exp9 = _load_json(Path(args.exp9_summary))[args.lens]
    exp17_root = Path(args.exp17_root)
    component_roots = {
        "du_truthfulness": exp17_root / "du_truthfulness",
        "du_refusal": exp17_root / "du_refusal",
        "lu_replication": exp17_root / "lu_replication",
    }

    rows: list[dict[str, Any]] = []
    for component, root in component_roots.items():
        for model in VALID_MODELS:
            peak_layer = exp9.get(model, {}).get("peak_layer")
            n_layers = exp9.get(model, {}).get("n_layers")
            if peak_layer is None or n_layers is None:
                continue
            for variant in VALID_VARIANTS:
                summary_path = root / model / variant / "summary.json"
                if not summary_path.exists():
                    summary_path = root / model / variant / "launch_config.json"
                if not summary_path.exists():
                    continue
                summary = _load_json(summary_path)
                exp17_layer, layer_field = _pick_layer(summary)
                if exp17_layer is None or layer_field is None:
                    continue
                rows.append(
                    {
                        "component": component,
                        "model": model,
                        "variant": variant,
                        "n_layers": n_layers,
                        "convergence_peak_layer": peak_layer,
                        "exp17_layer": exp17_layer,
                        "layer_field": layer_field,
                        "abs_layer_diff": abs(exp17_layer - peak_layer),
                        "abs_depth_diff_normalized": abs(exp17_layer - peak_layer) / max(1, n_layers - 1),
                    }
                )

    aggregates: dict[str, dict[str, float]] = {}
    for component in component_roots:
        component_rows = [row for row in rows if row["component"] == component]
        if not component_rows:
            continue
        aggregates[component] = {
            "n_rows": float(len(component_rows)),
            "mean_abs_layer_diff": sum(row["abs_layer_diff"] for row in component_rows) / len(component_rows),
            "mean_abs_depth_diff_normalized": (
                sum(row["abs_depth_diff_normalized"] for row in component_rows) / len(component_rows)
            ),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "lens": args.lens,
                "rows": rows,
                "aggregates": aggregates,
                "note": (
                    "Rows use `selected_layer` when present, otherwise fall back to "
                    "`peak_direction_norm_layer`. Interpret fallback rows as provisional."
                ),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    log.info("Saved exp17 joint analysis -> %s", out_path)


if __name__ == "__main__":
    main()
