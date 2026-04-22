#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]
DENSE5_MODELS = [model for model in MODEL_ORDER if model != "deepseek_v2_lite"]
PT_TARGET_PAIRS = ["JS_Bearly_C", "JS_Bmid_C", "JS_Blate_C"]
IT_TARGET_PAIRS = ["JS_Dearly_A", "JS_Dmid_A", "JS_Dlate_A"]
PT_HOST_PAIRS = ["JS_Bearly_A", "JS_Bmid_A", "JS_Blate_A"]
IT_HOST_PAIRS = ["JS_Dearly_C", "JS_Dmid_C", "JS_Dlate_C"]
WINDOW_LABEL = {
    "JS_Bearly_C": "early",
    "JS_Bmid_C": "mid",
    "JS_Blate_C": "late",
    "JS_Dearly_A": "early",
    "JS_Dmid_A": "mid",
    "JS_Dlate_A": "late",
    "JS_Bearly_A": "early",
    "JS_Bmid_A": "mid",
    "JS_Blate_A": "late",
    "JS_Dearly_C": "early",
    "JS_Dmid_C": "mid",
    "JS_Dlate_C": "late",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _collect_model_dirs(run_root: Path) -> list[tuple[str, Path]]:
    base = run_root / "merged" if (run_root / "merged").exists() else run_root
    rows: list[tuple[str, Path]] = []
    subdirs = {path.name: path for path in base.iterdir() if path.is_dir()}
    for model in MODEL_ORDER:
        path = subdirs.get(model)
        if path is not None:
            rows.append((model, path))
    return rows


def _safe_mean(total: float, count: float) -> float | None:
    if count <= 0:
        return None
    return float(total / count)


def _pair_layer_curve(pair_stats: dict[str, Any]) -> list[float | None]:
    sums = pair_stats.get("sum", [])
    counts = pair_stats.get("count", [])
    return [_safe_mean(float(total), float(count)) for total, count in zip(sums, counts, strict=True)]


def _weighted_region(pair_stats: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    sums = [float(v) for v in pair_stats.get("sum", [])[start:end]]
    counts = [float(v) for v in pair_stats.get("count", [])[start:end]]
    total_sum = float(sum(sums))
    total_count = float(sum(counts))
    return {
        "mean": _safe_mean(total_sum, total_count),
        "count": int(total_count),
    }


def _pair_payload(
    *,
    pair_stats: dict[str, Any],
    n_layers: int,
    corrective_onset: int,
    final_region_start: int,
    prompt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    region_names = ["full_stack", "pre_corrective", "final_20pct", "graft_window"]
    prompt_summary: dict[str, Any] = {}
    for region_name in region_names:
        mean_key = f"{region_name}_mean"
        count_key = f"{region_name}_count"
        values = [float(row[mean_key]) for row in prompt_rows if row.get(mean_key) is not None]
        counts = [int(row.get(count_key, 0)) for row in prompt_rows if row.get(mean_key) is not None]
        prompt_summary[region_name] = {
            "mean": (sum(values) / len(values)) if values else None,
            "n_prompts": len(values),
            "support_count_sum": int(sum(counts)),
        }
    return {
        "metadata": {
            "pair_name": pair_stats["pair_name"],
            "current_pipeline": pair_stats["current_pipeline"],
            "baseline_pipeline": pair_stats["baseline_pipeline"],
            "group": pair_stats["group"],
            "current_prompt_mode": pair_stats["current_prompt_mode"],
            "baseline_prompt_mode": pair_stats["baseline_prompt_mode"],
            "graft_window": pair_stats.get("graft_window"),
        },
        "layer_curve": _pair_layer_curve(pair_stats),
        "regions_weighted": {
            "full_stack": _weighted_region(pair_stats, 0, n_layers),
            "pre_corrective": _weighted_region(pair_stats, 0, corrective_onset),
            "final_20pct": _weighted_region(pair_stats, final_region_start, n_layers),
            "graft_window": (
                _weighted_region(
                    pair_stats,
                    int(pair_stats["graft_window"]["start_layer"]),
                    int(pair_stats["graft_window"]["end_layer_exclusive"]),
                )
                if pair_stats.get("graft_window") is not None
                else {"mean": None, "count": 0}
            ),
        },
        "regions_prompt_mean": prompt_summary,
    }


def _delta_payload(pair_payload: dict[str, Any], baseline_payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for region_name in ("pre_corrective", "final_20pct", "graft_window", "full_stack"):
        pair_weighted = pair_payload["regions_weighted"][region_name]["mean"]
        baseline_weighted = baseline_payload["regions_weighted"][region_name]["mean"]
        pair_prompt = pair_payload["regions_prompt_mean"][region_name]["mean"]
        baseline_prompt = baseline_payload["regions_prompt_mean"][region_name]["mean"]
        out[region_name] = {
            "weighted_delta": (
                float(pair_weighted - baseline_weighted)
                if pair_weighted is not None and baseline_weighted is not None
                else None
            ),
            "prompt_mean_delta": (
                float(pair_prompt - baseline_prompt)
                if pair_prompt is not None and baseline_prompt is not None
                else None
            ),
        }
    return out


def _aggregate_prompt_rows(rows_by_model: list[list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for rows in rows_by_model:
        for row in rows:
            out.setdefault(row["pair_name"], []).append(row)
    return out


def _aggregate_pair_stats(pair_stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    first = pair_stats_list[0]
    merged = {
        "pair_name": first["pair_name"],
        "current_pipeline": first["current_pipeline"],
        "baseline_pipeline": first["baseline_pipeline"],
        "group": first["group"],
        "current_prompt_mode": first["current_prompt_mode"],
        "baseline_prompt_mode": first["baseline_prompt_mode"],
        "graft_window": first.get("graft_window"),
        "sum": [0.0] * len(first.get("sum", [])),
        "count": [0] * len(first.get("count", [])),
    }
    for pair_stats in pair_stats_list:
        for idx, value in enumerate(pair_stats.get("sum", [])):
            merged["sum"][idx] += float(value)
        for idx, value in enumerate(pair_stats.get("count", [])):
            merged["count"][idx] += int(value)
    return merged


def _bundle_payload(
    *,
    name: str,
    models: list[str],
    model_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not models:
        return {"name": name, "models": []}
    n_layers = int(model_payloads[models[0]]["n_layers"])
    corrective_onset = int(model_payloads[models[0]]["corrective_onset"])
    final_region_start = int(model_payloads[models[0]]["final_region_start"])
    prompt_rows_by_model = [model_payloads[model]["prompt_rows"] for model in models]
    aggregated_prompt_rows = _aggregate_prompt_rows(prompt_rows_by_model)

    pair_names = sorted(model_payloads[models[0]]["pair_stats"].keys())
    pairs: dict[str, Any] = {}
    for pair_name in pair_names:
        pair_stats = _aggregate_pair_stats([model_payloads[model]["pair_stats"][pair_name] for model in models])
        pairs[pair_name] = _pair_payload(
            pair_stats=pair_stats,
            n_layers=n_layers,
            corrective_onset=corrective_onset,
            final_region_start=final_region_start,
            prompt_rows=aggregated_prompt_rows.get(pair_name, []),
        )

    baseline = pairs["JS_AC"]
    pt_side = {
        WINDOW_LABEL[pair_name]: _delta_payload(pairs[pair_name], baseline)
        for pair_name in PT_TARGET_PAIRS
        if pair_name in pairs
    }
    it_side = {
        WINDOW_LABEL[pair_name]: _delta_payload(pairs[pair_name], baseline)
        for pair_name in IT_TARGET_PAIRS
        if pair_name in pairs
    }
    pt_host = {
        WINDOW_LABEL[pair_name]: pairs[pair_name]["regions_weighted"]
        for pair_name in PT_HOST_PAIRS
        if pair_name in pairs
    }
    it_host = {
        WINDOW_LABEL[pair_name]: pairs[pair_name]["regions_weighted"]
        for pair_name in IT_HOST_PAIRS
        if pair_name in pairs
    }
    return {
        "name": name,
        "models": models,
        "n_layers": n_layers,
        "corrective_onset": corrective_onset,
        "final_region_start": final_region_start,
        "pairs": pairs,
        "pt_side_gap_closure": pt_side,
        "it_side_gap_collapse": it_side,
        "host_local_pt": pt_host,
        "host_local_it": it_host,
        "ordering_final_20pct": {
            "pt_side": {
                window: payload["final_20pct"]["weighted_delta"]
                for window, payload in pt_side.items()
            },
            "it_side": {
                window: payload["final_20pct"]["weighted_delta"]
                for window, payload in it_side.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exp16 matched-prefix JS replay outputs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to <run-root>/js_summary.json")
    args = parser.parse_args()

    model_dirs = _collect_model_dirs(args.run_root)
    if not model_dirs:
        raise FileNotFoundError(f"No model dirs found under {args.run_root}")

    model_payloads: dict[str, dict[str, Any]] = {}
    summary_models: dict[str, Any] = {}
    for model, model_dir in model_dirs:
        config = _read_json(model_dir / "config.json")
        js_layer_stats = _read_json(model_dir / "js_layer_stats.json")
        prompt_rows = _load_jsonl(model_dir / "js_prompt_region_metrics.jsonl")
        teacher_cap = _read_json(model_dir / "teacher_cap_diagnostics.json")
        if not js_layer_stats:
            raise FileNotFoundError(f"Missing js_layer_stats.json for {model_dir}")
        n_layers = int(js_layer_stats["n_layers"])
        corrective_onset = int(js_layer_stats["corrective_onset"])
        final_region_start = int(js_layer_stats["final_region_start"])
        prompt_rows_by_pair: dict[str, list[dict[str, Any]]] = {}
        for row in prompt_rows:
            prompt_rows_by_pair.setdefault(row["pair_name"], []).append(row)

        pairs: dict[str, Any] = {}
        for pair_name, pair_stats in js_layer_stats.get("pairs", {}).items():
            pairs[pair_name] = _pair_payload(
                pair_stats=pair_stats,
                n_layers=n_layers,
                corrective_onset=corrective_onset,
                final_region_start=final_region_start,
                prompt_rows=prompt_rows_by_pair.get(pair_name, []),
            )

        baseline = pairs["JS_AC"]
        pt_side = {
            WINDOW_LABEL[pair_name]: _delta_payload(pairs[pair_name], baseline)
            for pair_name in PT_TARGET_PAIRS
            if pair_name in pairs
        }
        it_side = {
            WINDOW_LABEL[pair_name]: _delta_payload(pairs[pair_name], baseline)
            for pair_name in IT_TARGET_PAIRS
            if pair_name in pairs
        }
        summary_models[model] = {
            "config_path": str(model_dir / "config.json"),
            "teacher_cap_diagnostics": teacher_cap,
            "n_layers": n_layers,
            "corrective_onset": corrective_onset,
            "final_region_start": final_region_start,
            "pairs": pairs,
            "pt_side_gap_closure": pt_side,
            "it_side_gap_collapse": it_side,
            "host_local_pt": {
                WINDOW_LABEL[pair_name]: pairs[pair_name]["regions_weighted"]
                for pair_name in PT_HOST_PAIRS
                if pair_name in pairs
            },
            "host_local_it": {
                WINDOW_LABEL[pair_name]: pairs[pair_name]["regions_weighted"]
                for pair_name in IT_HOST_PAIRS
                if pair_name in pairs
            },
        }
        model_payloads[model] = {
            "n_layers": n_layers,
            "corrective_onset": corrective_onset,
            "final_region_start": final_region_start,
            "pair_stats": js_layer_stats["pairs"],
            "prompt_rows": prompt_rows,
            "config": config,
        }

    dense_models = [model for model in DENSE5_MODELS if model in model_payloads]
    deepseek_models = [model for model in ["deepseek_v2_lite"] if model in model_payloads]

    summary = {
        "run_root": str(args.run_root),
        "models": summary_models,
        "dense5": _bundle_payload(name="dense5", models=dense_models, model_payloads=model_payloads),
        "deepseek": _bundle_payload(name="deepseek", models=deepseek_models, model_payloads=model_payloads),
        "model_order_present": [model for model in MODEL_ORDER if model in summary_models],
    }

    out_path = args.out or (args.run_root / "js_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[exp16] wrote summary to {out_path}")


if __name__ == "__main__":
    main()
