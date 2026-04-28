#!/usr/bin/env python3
"""Compatibility-amplification and label-swap control for Exp23.

This is a CPU-only analysis of the residual-state x late-stack factorial.
It re-expresses the standard interaction as a late-stack own-token
compatibility amplification:

    IT boost = Y(U_IT, L_IT) - Y(U_PT, L_IT)
    PT boost = Y(U_IT, L_PT) - Y(U_PT, L_PT)
    amplification = IT boost - PT boost

where Y is the IT-token-vs-PT-token margin. The amplification is algebraically
the Exp23 interaction term. The label-swap control preserves each prompt's four
cell values and randomly swaps PT/IT orientation within each model/prompt
cluster, after averaging any repeated event kinds inside prompt. This flips the
sign of that cluster's amplification under the null that the PT/IT label
orientation is arbitrary.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]
CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _extract_unit(payload: dict[str, Any], readout: str) -> dict[str, float] | None:
    cells = payload.get("cells") or {}
    margins: dict[str, float] = {}
    for cell in CELLS:
        value = _finite(((cells.get(cell) or {}).get(readout) or {}).get("it_vs_pt_margin"))
        if value is None:
            return None
        margins[cell] = value
    return margins


def _unit_stats(margins: dict[str, float]) -> dict[str, float]:
    y00 = margins["U_PT__L_PT"]
    y01 = margins["U_PT__L_IT"]
    y10 = margins["U_IT__L_PT"]
    y11 = margins["U_IT__L_IT"]
    it_boost = y11 - y01
    pt_boost = y10 - y00
    return {
        "U_PT__L_PT": y00,
        "U_PT__L_IT": y01,
        "U_IT__L_PT": y10,
        "U_IT__L_IT": y11,
        "it_compatibility_boost": it_boost,
        "pt_compatibility_boost": pt_boost,
        "compatibility_amplification": it_boost - pt_boost,
    }


def _load_units(run_root: Path, models: list[str], prompt_mode: str, readout: str) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = defaultdict(list)
    for model in models:
        path = run_root / "residual_factorial" / prompt_mode / model / "records.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in _json_rows(path):
            prompt_id = str(row.get("prompt_id", ""))
            for _event_kind, payload in (row.get("events") or {}).items():
                if not payload or payload.get("duplicate_of") or not payload.get("valid"):
                    continue
                margins = _extract_unit(payload, readout)
                if margins is not None:
                    out[model].append({"prompt_id": prompt_id, **_unit_stats(margins)})
    return dict(out)


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values.astype(float), [2.5, 97.5])
    return float(lo), float(hi)


def _summarize(units_by_model: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    keys = [
        "U_PT__L_PT",
        "U_PT__L_IT",
        "U_IT__L_PT",
        "U_IT__L_IT",
        "it_compatibility_boost",
        "pt_compatibility_boost",
        "compatibility_amplification",
    ]
    by_model: dict[str, Any] = {}
    for model, units in units_by_model.items():
        prompt_units: dict[str, list[dict[str, float]]] = defaultdict(list)
        for unit in units:
            prompt_units[str(unit.get("prompt_id", ""))].append(unit)
        prompt_means = {
            prompt_id: {key: _mean([unit[key] for unit in prompt_group]) for key in keys}
            for prompt_id, prompt_group in prompt_units.items()
        }
        by_model[model] = {
            "n_units": len(units),
            "n_prompt_clusters": len(prompt_means),
            **{key: _mean([row[key] for row in prompt_means.values()]) for key in keys},
        }
    pooled = {
        "n_units": sum(row["n_units"] for row in by_model.values()),
        "n_prompt_clusters": sum(row["n_prompt_clusters"] for row in by_model.values()),
        **{key: _mean([row[key] for row in by_model.values()]) for key in keys},
    }
    return {"by_model": by_model, "pooled_model_mean": pooled}


def _label_swap_null(
    units_by_model: dict[str, list[dict[str, float]]],
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    model_arrays = {}
    for model, units in units_by_model.items():
        by_prompt: dict[str, list[float]] = defaultdict(list)
        for unit in units:
            by_prompt[str(unit.get("prompt_id", ""))].append(float(unit["compatibility_amplification"]))
        model_arrays[model] = np.asarray([np.mean(vals) for vals in by_prompt.values() if vals], dtype=np.float64)
    observed_by_model = {model: float(np.mean(values)) for model, values in model_arrays.items()}
    observed = float(np.mean(list(observed_by_model.values())))

    null = np.zeros(n_permutations, dtype=np.float64)
    for values in model_arrays.values():
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_permutations, values.size))
        null += np.mean(signs * values[None, :], axis=1)
    null /= max(1, len(model_arrays))

    p_upper = float((1 + np.sum(null >= observed)) / (n_permutations + 1))
    p_two_sided = float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1))
    q = np.percentile(null, [0.1, 1, 2.5, 50, 97.5, 99, 99.9])
    return {
        "observed": observed,
        "observed_by_model": observed_by_model,
        "n_permutations": int(n_permutations),
        "p_upper": p_upper,
        "p_two_sided": p_two_sided,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null, ddof=1)),
        "null_quantiles": {
            "q0.1": float(q[0]),
            "q1": float(q[1]),
            "q2.5": float(q[2]),
            "q50": float(q[3]),
            "q97.5": float(q[4]),
            "q99": float(q[5]),
            "q99.9": float(q[6]),
        },
        "null_max": float(np.max(null)),
        "null_min": float(np.min(null)),
        "z_against_null": float((observed - np.mean(null)) / np.std(null, ddof=1)),
        "null_samples_preview": [float(x) for x in null[:1000]],
    }


def _write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Exp23 Compatibility-Amplification Label Control",
        "",
        f"Run root: `{payload['run_root']}`",
        f"Prompt mode: `{payload['prompt_mode']}`",
        f"Readout: `{payload['readout']}`",
        "",
        "The compatibility amplification is algebraically the Exp23 interaction, but is read as an own-token compatibility test: IT late stack gains more from IT upstream than PT late stack gains from PT upstream.",
        "",
        "| Model | Records | Prompt clusters | IT compatibility boost | PT compatibility boost | IT-over-PT amplification |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in payload["summary"]["by_model"].items():
        lines.append(
            f"| `{model}` | `{row['n_units']}` | `{row['n_prompt_clusters']}` | `{row['it_compatibility_boost']:+.3f}` | "
            f"`{row['pt_compatibility_boost']:+.3f}` | `{row['compatibility_amplification']:+.3f}` |"
        )
    pooled = payload["summary"]["pooled_model_mean"]
    lines.extend(
        [
            f"| **Dense mean** | `{pooled['n_units']}` | `{pooled['n_prompt_clusters']}` | `{pooled['it_compatibility_boost']:+.3f}` | "
            f"`{pooled['pt_compatibility_boost']:+.3f}` | `{pooled['compatibility_amplification']:+.3f}` |",
            "",
            "## Label-Swap Null",
            "",
            "The null randomly swaps PT/IT label orientation within each model/prompt cluster. For prompts with multiple divergence-event records, the event-level amplification is averaged within prompt before the sign flip.",
            "",
            f"- Observed amplification: `{payload['permutation']['observed']:+.3f}` logits",
            f"- Null mean: `{payload['permutation']['null_mean']:+.4f}` logits",
            f"- Null std: `{payload['permutation']['null_std']:.4f}` logits",
            f"- Null 99.9th percentile: `{payload['permutation']['null_quantiles']['q99.9']:+.3f}` logits",
            f"- One-sided permutation p-value: `{payload['permutation']['p_upper']:.6g}`",
            f"- Two-sided permutation p-value: `{payload['permutation']['p_two_sided']:.6g}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _plot(payload: dict[str, Any], path: Path) -> None:
    perm = payload["permutation"]
    samples = np.asarray(perm["null_samples_preview"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(samples, bins=50, color="#9ecae1", edgecolor="white", alpha=0.9)
    ax.axvline(perm["observed"], color="#d62728", linewidth=2.0, label="observed")
    ax.axvline(perm["null_quantiles"]["q99.9"], color="#525252", linestyle="--", linewidth=1.2, label="null 99.9%")
    ax.set_xlabel("Compatibility amplification under label-swap null")
    ax.set_ylabel("Count in preview sample")
    ax.set_title("Exp23 compatibility-amplification label control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def analyze(
    *,
    run_root: Path,
    out_dir: Path,
    models: list[str],
    prompt_mode: str,
    readout: str,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    units_by_model = _load_units(run_root, models, prompt_mode, readout)
    summary = _summarize(units_by_model)
    permutation = _label_swap_null(units_by_model, n_permutations=n_permutations, seed=seed)
    payload = {
        "run_root": str(run_root),
        "prompt_mode": prompt_mode,
        "readout": readout,
        "models": models,
        "summary": summary,
        "permutation": permutation,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exp23_compatibility_permutation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    _write_report(payload, out_dir / "exp23_compatibility_permutation_report.md")
    _plot(payload, out_dir / "exp23_compatibility_permutation_null.png")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--readout", default="common_it")
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis" / "compatibility_permutation")
    payload = analyze(
        run_root=args.run_root,
        out_dir=out_dir,
        models=list(args.models),
        prompt_mode=args.prompt_mode,
        readout=args.readout,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "observed": payload["permutation"]["observed"],
                "p_upper": payload["permutation"]["p_upper"],
                "null_q99.9": payload["permutation"]["null_quantiles"]["q99.9"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
