#!/usr/bin/env python3
"""Bootstrap derived Exp35 stage-share ratios from scored cell records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RUN = Path(
    "results/exp35_olmo_base_anchored_stage_decomposition/"
    "exp35_full_olmo_stage_8a100_20260502_2300"
)


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _unit_interactions(scored_path: Path, readout: str) -> list[dict[str, Any]]:
    cells_by_prompt: dict[str, dict[str, float]] = {}
    for row in _json_rows(scored_path):
        if row.get("readout") != readout:
            continue
        prompt_id = str(row["prompt_id"])
        cells_by_prompt.setdefault(prompt_id, {})[str(row["cell"])] = float(row["margin_rlvr_minus_base"])

    units: list[dict[str, Any]] = []
    for prompt_id, cells in cells_by_prompt.items():
        try:
            y_bb = cells["U_B__L_B"]
            payload = {"prompt_id": prompt_id}
            for stage in ("S", "D", "R"):
                payload[stage] = (
                    cells[f"U_{stage}__L_{stage}"]
                    - cells[f"U_{stage}__L_B"]
                    - cells[f"U_B__L_{stage}"]
                    + y_bb
                )
            units.append(payload)
        except KeyError:
            continue
    if not units:
        raise RuntimeError(f"No complete Exp35 units found for readout={readout} at {scored_path}")
    return units


def _summarize(units: list[dict[str, Any]], n_bootstrap: int, seed: int, readout: str) -> list[dict[str, Any]]:
    arrays = {stage: np.asarray([float(unit[stage]) for unit in units], dtype=float) for stage in ("S", "D", "R")}
    estimates = {stage: float(values.mean()) for stage, values in arrays.items()}

    def metrics(means: dict[str, float]) -> dict[str, float]:
        return {
            "S": means["S"],
            "D": means["D"],
            "R": means["R"],
            "S_share": means["S"] / means["R"],
            "D_share": means["D"] / means["R"],
            "D_minus_S_share": (means["D"] - means["S"]) / means["R"],
            "R_minus_D_share": (means["R"] - means["D"]) / means["R"],
        }

    point = metrics(estimates)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in point}
    n = len(units)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means = {stage: float(values[idx].mean()) for stage, values in arrays.items()}
        for key, value in metrics(means).items():
            draws[key].append(value)

    rows = []
    for key, estimate in point.items():
        arr = np.asarray(draws[key], dtype=float)
        rows.append(
            {
                "readout": readout,
                "metric": key,
                "estimate": estimate,
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5)),
                "n_units": n,
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--readout", default="common_r")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scored_path = args.run_root / "analysis" / "scored_cells.jsonl.gz"
    out_path = args.out or (args.run_root / "analysis" / "stage_ratio_bootstrap.csv")
    rows = _summarize(
        _unit_interactions(scored_path, args.readout),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        readout=args.readout,
    )
    _write_csv(out_path, rows)
    print(json.dumps({"out": str(out_path), "rows": len(rows), "readout": args.readout}, indent=2))


if __name__ == "__main__":
    main()
