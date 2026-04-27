#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]
DENSE5_MODELS = [model for model in MODEL_ORDER if model != "deepseek_v2_lite"]
DEFAULT_N_BOOT = 2000
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


def _stable_seed(*parts: str) -> int:
    text = "::".join(parts)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % (2**32)


def _prompt_region_mean_ci(
    prompt_rows: list[dict[str, Any]],
    region_name: str,
    *,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict[str, Any]:
    mean_key = f"{region_name}_mean"
    means: list[float] = []
    for row in prompt_rows:
        mean = row.get(mean_key)
        if mean is None:
            continue
        means.append(float(mean))
    if not means:
        return {"ci95_low": None, "ci95_high": None, "n_boot": 0, "n_prompts": 0}
    mean_arr = np.array(means, dtype=float)
    rng = np.random.default_rng(_stable_seed(region_name, str(len(prompt_rows)), str(round(float(mean_arr.sum()), 6))))
    boot: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, mean_arr.size, size=mean_arr.size)
        boot.append(float(mean_arr[idx].mean()))
    if not boot:
        return {"ci95_low": None, "ci95_high": None, "n_boot": 0, "n_prompts": int(mean_arr.size)}
    lo, hi = np.percentile(np.array(boot, dtype=float), [2.5, 97.5])
    return {"ci95_low": float(lo), "ci95_high": float(hi), "n_boot": len(boot), "n_prompts": int(mean_arr.size)}


def _add_prompt_region_cis(regions: dict[str, dict[str, Any]], prompt_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {name: dict(payload) for name, payload in regions.items()}
    for region_name in out:
        out[region_name].update(_prompt_region_mean_ci(prompt_rows, region_name))
    return out


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
    weighted_regions = {
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
        "regions_weighted": weighted_regions,
        "regions_prompt_mean": _add_prompt_region_cis(prompt_summary, prompt_rows),
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
    max_len = max(len(pair_stats.get("sum", [])) for pair_stats in pair_stats_list)
    merged = {
        "pair_name": first["pair_name"],
        "current_pipeline": first["current_pipeline"],
        "baseline_pipeline": first["baseline_pipeline"],
        "group": first["group"],
        "current_prompt_mode": first["current_prompt_mode"],
        "baseline_prompt_mode": first["baseline_prompt_mode"],
        "graft_window": first.get("graft_window"),
        "sum": [0.0] * max_len,
        "count": [0] * max_len,
    }
    for pair_stats in pair_stats_list:
        for idx, value in enumerate(pair_stats.get("sum", [])):
            merged["sum"][idx] += float(value)
        for idx, value in enumerate(pair_stats.get("count", [])):
            merged["count"][idx] += int(value)
    return merged


def _resample_curve(values: list[float | None], out_bins: int) -> list[float]:
    if out_bins <= 0:
        return []
    filtered = [float("nan") if value is None else float(value) for value in values]
    if not filtered:
        return [float("nan")] * out_bins
    if len(filtered) == 1:
        return [filtered[0]] * out_bins

    out: list[float] = []
    last_idx = len(filtered) - 1
    for out_idx in range(out_bins):
        pos = (out_idx / (out_bins - 1)) * last_idx
        left = int(math.floor(pos))
        right = int(math.ceil(pos))
        if left == right:
            out.append(filtered[left])
            continue
        left_val = filtered[left]
        right_val = filtered[right]
        frac = pos - left
        if math.isnan(left_val) and math.isnan(right_val):
            out.append(float("nan"))
        elif math.isnan(left_val):
            out.append(right_val)
        elif math.isnan(right_val):
            out.append(left_val)
        else:
            out.append(left_val + frac * (right_val - left_val))
    return out


def _combine_region_weighted(pair_payloads: list[dict[str, Any]], region_name: str) -> dict[str, Any]:
    total_sum = 0.0
    total_count = 0
    for payload in pair_payloads:
        region = payload["regions_weighted"][region_name]
        mean = region.get("mean")
        count = int(region.get("count", 0))
        if mean is None or count <= 0:
            continue
        total_sum += float(mean) * count
        total_count += count
    return {
        "mean": _safe_mean(total_sum, float(total_count)),
        "count": total_count,
    }


def _combine_pair_payloads(
    *,
    pair_payloads: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    curve_bins: int = 101,
) -> dict[str, Any]:
    metadata = dict(pair_payloads[0]["metadata"])
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

    resampled = [_resample_curve(payload["layer_curve"], curve_bins) for payload in pair_payloads]
    pooled_curve: list[float | None] = []
    for idx in range(curve_bins):
        values = [curve[idx] for curve in resampled if not math.isnan(curve[idx])]
        pooled_curve.append((sum(values) / len(values)) if values else None)

    return {
        "metadata": metadata,
        "layer_curve": pooled_curve,
        "regions_weighted": {
            region_name: _combine_region_weighted(pair_payloads, region_name)
            for region_name in region_names
        },
        "regions_prompt_mean": _add_prompt_region_cis(prompt_summary, prompt_rows),
    }


def _bundle_payload(
    *,
    name: str,
    models: list[str],
    model_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not models:
        return {"name": name, "models": []}
    n_layers_list = [int(model_payloads[model]["n_layers"]) for model in models]
    corrective_onset_list = [int(model_payloads[model]["corrective_onset"]) for model in models]
    final_region_start_list = [int(model_payloads[model]["final_region_start"]) for model in models]
    same_geometry = (
        len(set(n_layers_list)) == 1
        and len(set(corrective_onset_list)) == 1
        and len(set(final_region_start_list)) == 1
    )

    n_layers = n_layers_list[0]
    corrective_onset = corrective_onset_list[0]
    final_region_start = final_region_start_list[0]
    prompt_rows_by_model = [model_payloads[model]["prompt_rows"] for model in models]
    aggregated_prompt_rows = _aggregate_prompt_rows(prompt_rows_by_model)

    pair_names = sorted(model_payloads[models[0]]["pair_stats"].keys())
    pairs: dict[str, Any] = {}
    for pair_name in pair_names:
        pair_payloads = []
        for model in models:
            payload = model_payloads[model]
            pair_payloads.append(
                _pair_payload(
                    pair_stats=payload["pair_stats"][pair_name],
                    n_layers=int(payload["n_layers"]),
                    corrective_onset=int(payload["corrective_onset"]),
                    final_region_start=int(payload["final_region_start"]),
                    prompt_rows=[row for row in payload["prompt_rows"] if row["pair_name"] == pair_name],
                )
            )
        if same_geometry:
            pair_stats = _aggregate_pair_stats([model_payloads[model]["pair_stats"][pair_name] for model in models])
            pairs[pair_name] = _pair_payload(
                pair_stats=pair_stats,
                n_layers=n_layers,
                corrective_onset=corrective_onset,
                final_region_start=final_region_start,
                prompt_rows=aggregated_prompt_rows.get(pair_name, []),
            )
        else:
            pairs[pair_name] = _combine_pair_payloads(
                pair_payloads=pair_payloads,
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
        "n_layers": n_layers if same_geometry else None,
        "corrective_onset": corrective_onset if same_geometry else None,
        "final_region_start": final_region_start if same_geometry else None,
        "layer_axis_mode": "layer_index" if same_geometry else "normalized_depth",
        "layer_axis_values": (
            list(range(n_layers))
            if same_geometry
            else [idx / 100.0 for idx in range(101)]
        ),
        "corrective_onset_frac": (
            corrective_onset / max(n_layers - 1, 1)
            if same_geometry
            else sum(onset / max(nl - 1, 1) for onset, nl in zip(corrective_onset_list, n_layers_list, strict=True))
            / len(models)
        ),
        "final_region_start_frac": (
            final_region_start / max(n_layers - 1, 1)
            if same_geometry
            else sum(start / max(nl - 1, 1) for start, nl in zip(final_region_start_list, n_layers_list, strict=True))
            / len(models)
        ),
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
