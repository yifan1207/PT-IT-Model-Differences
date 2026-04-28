"""CPU-side Exp18 analysis over existing matched-prefix traces."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, revision_for_model_id
from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import (
    classify_generated_tokens_by_word,
)
from src.poc.exp18_midlate_token_handoff.metrics import (
    COLLAPSED_CATEGORIES,
    RAW_CATEGORIES,
    WindowSpec,
    add_category_value,
    collapse_category,
    disjoint_windows,
    finalize_category_values,
    first_layer_in_topk,
    make_category_template,
    overlapping_windows,
    rank_gain,
    top1_top20_delta,
)

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - tests do not require tokenizers.
    AutoTokenizer = None


DEPTH_WINDOWS = {
    "gemma3_4b": {
        "B_early_raw": (0, 14),
        "B_mid_raw": (10, 24),
        "B_late_raw": (20, 34),
    },
    "llama31_8b": {
        "B_early_raw": (0, 13),
        "B_mid_raw": (9, 22),
        "B_late_raw": (19, 32),
    },
    "qwen3_4b": {
        "B_early_raw": (0, 14),
        "B_mid_raw": (11, 25),
        "B_late_raw": (22, 36),
    },
    "mistral_7b": {
        "B_early_raw": (0, 13),
        "B_mid_raw": (9, 22),
        "B_late_raw": (19, 32),
    },
    "olmo2_7b": {
        "B_early_raw": (0, 13),
        "B_mid_raw": (9, 22),
        "B_late_raw": (19, 32),
    },
    "deepseek_v2_lite": {
        "B_early_raw": (0, 11),
        "B_mid_raw": (8, 19),
        "B_late_raw": (16, 27),
    },
}

DEFAULT_DEPTH_ROOTS = [
    Path("results/exp11_matched_prefix_mlp_graft/data/exp11_exp3_600rand_v11_depthablation"),
    Path("results/exp11_matched_prefix_mlp_graft/data/exp11_exp3_600rand_v11_depthablation_flat"),
]


@dataclass(frozen=True)
class TeacherStep:
    token_id: int
    token_str: str
    raw_category: str
    collapsed_category: str


class DecodeCache:
    def __init__(self) -> None:
        self._tokenizers: dict[str, Any] = {}
        self._failed_model_ids: set[str] = set()
        self._strings: dict[tuple[str, int], str] = {}
        self._categories: dict[tuple[str, int], str] = {}

    def decode(self, model_key: str, model_id: str, token_id: int) -> str:
        key = (model_key, int(token_id))
        if key in self._strings:
            return self._strings[key]
        token_str = f"<id:{token_id}>"
        if AutoTokenizer is not None and model_id not in self._failed_model_ids:
            try:
                if model_id not in self._tokenizers:
                    revision = revision_for_model_id(model_id)
                    kwargs = {"revision": revision} if revision else {}
                    self._tokenizers[model_id] = AutoTokenizer.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        local_files_only=True,
                        **kwargs,
                    )
                token_str = self._tokenizers[model_id].decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                self._failed_model_ids.add(model_id)
        self._strings[key] = token_str
        return token_str

    def category(self, model_key: str, model_id: str, token_id: int) -> str:
        key = (model_key, int(token_id))
        if key not in self._categories:
            token_str = self.decode(model_key, model_id, token_id)
            cat = classify_generated_tokens_by_word([{"token_str": token_str}])[0]
            self._categories[key] = cat if cat in RAW_CATEGORIES else "OTHER"
        return self._categories[key]


class WindowAccumulator:
    def __init__(self) -> None:
        self.metrics = {
            "support_target_delta": make_category_template(),
            "repulsion_top10_delta": make_category_template(),
            "margin_delta": make_category_template(),
            "top1_displacement": make_category_template(),
            "top20_entries": make_category_template(),
            "top20_exits": make_category_template(),
            "teacher_rank_gain": make_category_template(),
            "first_top1_layer": make_category_template(),
            "first_top5_layer": make_category_template(),
            "first_top20_layer": make_category_template(),
            "handoff": make_category_template(),
        }
        self.suppressed_token_categories = Counter({cat: 0 for cat in RAW_CATEGORIES})
        self.supported_token_categories = Counter({cat: 0 for cat in RAW_CATEGORIES})
        self.n_steps = 0

    def add(self, raw_category: str, metric_name: str, value: float | int | None) -> None:
        add_category_value(self.metrics[metric_name], raw_category, value)

    def add_displacement(self, suppressed: str, supported: str) -> None:
        self.suppressed_token_categories[suppressed] += 1
        self.supported_token_categories[supported] += 1

    def finalize(self) -> dict[str, Any]:
        by_category: dict[str, dict[str, Any]] = {
            cat: {
                "support_target_delta": None,
                "repulsion_top10_delta": None,
                "margin_delta": None,
            }
            for cat in RAW_CATEGORIES + COLLAPSED_CATEGORIES
        }
        metric_key_map = {
            "top1_displacement": "fraction_top1_displaced",
            "top20_entries": "mean_top20_entries",
            "top20_exits": "mean_top20_exits",
            "teacher_rank_gain": "mean_teacher_rank_gain",
            "first_top1_layer": "mean_first_top1_layer",
            "first_top5_layer": "mean_first_top5_layer",
            "first_top20_layer": "mean_first_top20_layer",
            "handoff": "handoff_rate",
        }
        for metric_name, mean_key in metric_key_map.items():
            finalized = finalize_category_values(
                self.metrics[metric_name],
                mean_key=mean_key,
                positive_key=f"{metric_name}_fraction_positive",
            )
            for cat, payload in finalized.items():
                by_category.setdefault(cat, {})
                by_category[cat].update(payload)
        for unavailable in ("support_target_delta", "repulsion_top10_delta", "margin_delta"):
            for cat in by_category:
                by_category[cat].setdefault(unavailable, None)
        return {
            "n_steps": self.n_steps,
            "by_token_category": by_category,
            "suppressed_token_categories": _normalize_counter(self.suppressed_token_categories),
            "supported_token_categories": _normalize_counter(self.supported_token_categories),
        }


def _normalize_counter(counter: Counter) -> dict[str, dict[str, float | int | None]]:
    total = sum(counter.values())
    return {
        cat: {
            "count": int(counter.get(cat, 0)),
            "fraction": (int(counter.get(cat, 0)) / total if total else None),
        }
        for cat in RAW_CATEGORIES
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                yield json.loads(raw.decode("utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue


def resolve_depth_dir(model: str, roots: list[Path]) -> Path:
    candidates = []
    for root in roots:
        candidates.extend(
            [
                root / model,
                root / model / f"exp11_exp3_600rand_v11_depthablation_{model}",
                root / model / f"exp11_exp3_600rand_v11_depthablation_flat_{model}",
            ]
        )
    for candidate in candidates:
        if (candidate / "step_metrics.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"No depth-ablation step_metrics.jsonl found for {model}")


def stream_prompt_rows(
    path: Path,
    pipelines: set[str],
    max_prompt_groups: int | None = None,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, int]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    current_prompt_id: str | None = None
    current_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicates = Counter()
    seen: set[tuple[str, str, int]] = set()
    n_groups = 0

    def flush_current() -> None:
        nonlocal current_prompt_id, current_rows, seen, n_groups
        if current_prompt_id is None:
            return
        for rows in current_rows.values():
            rows.sort(key=lambda item: int(item["step"]))
        grouped[current_prompt_id] = dict(current_rows)
        n_groups += 1
        current_prompt_id = None
        current_rows = defaultdict(list)
        seen = set()

    for row in _iter_jsonl(path):
        pipeline = row.get("pipeline")
        if pipeline not in pipelines:
            continue
        prompt_id = str(row.get("prompt_id"))
        if current_prompt_id is None:
            current_prompt_id = prompt_id
        if prompt_id != current_prompt_id:
            flush_current()
            if max_prompt_groups is not None and n_groups >= max_prompt_groups:
                break
            current_prompt_id = prompt_id
        step = int(row.get("step", -1))
        key = (prompt_id, pipeline, step)
        if key in seen:
            duplicates[pipeline] += 1
            continue
        seen.add(key)
        current_rows[pipeline].append(row)
    else:
        flush_current()
    return grouped, dict(duplicates)


def visit_prompt_rows(
    path: Path,
    pipelines: set[str],
    on_prompt: Any,
    max_prompt_groups: int | None = None,
) -> dict[str, int]:
    """Stream a sorted step_metrics.jsonl file one prompt at a time."""

    duplicates = Counter()
    current_prompt_id: str | None = None
    current_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    n_groups = 0

    def flush_current() -> None:
        nonlocal current_prompt_id, current_rows, seen, n_groups
        if current_prompt_id is None:
            return
        for rows in current_rows.values():
            rows.sort(key=lambda item: int(item["step"]))
        on_prompt(current_prompt_id, dict(current_rows))
        n_groups += 1
        current_prompt_id = None
        current_rows = defaultdict(list)
        seen = set()

    for row in _iter_jsonl(path):
        pipeline = row.get("pipeline")
        if pipeline not in pipelines:
            continue
        prompt_id = str(row.get("prompt_id"))
        if current_prompt_id is None:
            current_prompt_id = prompt_id
        if prompt_id != current_prompt_id:
            flush_current()
            if max_prompt_groups is not None and n_groups >= max_prompt_groups:
                break
            current_prompt_id = prompt_id
        step = int(row.get("step", -1))
        key = (prompt_id, pipeline, step)
        if key in seen:
            duplicates[pipeline] += 1
            continue
        seen.add(key)
        current_rows[pipeline].append(row)
    else:
        flush_current()
    return dict(duplicates)


def teacher_steps_from_rows(rows: list[dict[str, Any]]) -> list[TeacherStep]:
    rows = sorted(rows, key=lambda row: int(row["step"]))
    generated_tokens = [
        {"token_id": int(row["token_id"]), "token_str": str(row.get("token_str", ""))}
        for row in rows
    ]
    categories = classify_generated_tokens_by_word(generated_tokens)
    return [
        TeacherStep(
            token_id=int(row["token_id"]),
            token_str=str(row.get("token_str", "")),
            raw_category=categories[idx],
            collapsed_category=collapse_category(categories[idx]),
        )
        for idx, row in enumerate(rows)
    ]


def _avg_first_layer(row: dict[str, Any], token_id: int, *, k: int, window: WindowSpec) -> int | None:
    top20 = row.get("metrics", {}).get("top20_ids", [])
    first = first_layer_in_topk(top20, token_id, k=k)
    return first if window.contains(first) else None


def _top1_first_layer(row: dict[str, Any], token_id: int, window: WindowSpec) -> int | None:
    top1 = row.get("metrics", {}).get("top1_token", [])
    for layer, top_id in enumerate(top1):
        if int(top_id) == int(token_id):
            return layer if window.contains(layer) else None
    return None


def _add_pair_displacements(
    *,
    acc: WindowAccumulator,
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    window: WindowSpec,
    model: str,
    model_id: str,
    decode_cache: DecodeCache,
) -> None:
    top1_a = row_a.get("metrics", {}).get("top1_token", [])
    top1_b = row_b.get("metrics", {}).get("top1_token", [])
    for layer in window.layers:
        if layer >= len(top1_a) or layer >= len(top1_b):
            continue
        if int(top1_a[layer]) == int(top1_b[layer]):
            continue
        suppressed = decode_cache.category(model, model_id, int(top1_a[layer]))
        supported = decode_cache.category(model, model_id, int(top1_b[layer]))
        acc.add_displacement(suppressed, supported)


def analyze_model_depth(
    *,
    model: str,
    depth_dir: Path,
    decode_cache: DecodeCache,
    max_prompts: int | None = None,
) -> dict[str, Any]:
    spec = get_spec(model)
    model_id = spec.pt_id
    primary_windows = disjoint_windows(
        n_layers=spec.n_layers,
        phase_boundary=spec.phase_boundary,
        corrective_onset=spec.corrective_onset,
    )
    continuity_windows = overlapping_windows(DEPTH_WINDOWS[model])
    all_windows = {**primary_windows, **continuity_windows}
    accumulators = {name: WindowAccumulator() for name in all_windows}
    handoff_acc = WindowAccumulator()
    needed = {"A_prime_raw", "B_early_raw", "B_mid_raw", "B_late_raw", "C_it_chat"}
    missing = []
    processed_prompts = 0
    processed_steps = 0

    def handle_prompt(prompt_id: str, by_pipeline: dict[str, list[dict[str, Any]]]) -> None:
        nonlocal processed_prompts, processed_steps
        if not needed.issubset(by_pipeline):
            missing.append({"prompt_id": prompt_id, "missing": sorted(needed - set(by_pipeline))})
            return
        processed_prompts += 1
        teacher_steps = teacher_steps_from_rows(by_pipeline["C_it_chat"])
        by_step = {
            name: {int(row["step"]): row for row in rows}
            for name, rows in by_pipeline.items()
        }
        common_steps = sorted(
            set.intersection(*(set(rows) for rows in by_step.values()))
            & set(range(len(teacher_steps)))
        )
        for step in common_steps:
            processed_steps += 1
            teacher = teacher_steps[step]
            row_a = by_step["A_prime_raw"][step]
            row_mid = by_step["B_mid_raw"][step]
            row_late = by_step["B_late_raw"][step]
            for window_name, window in all_windows.items():
                compare_pipeline = (
                    window_name
                    if window_name in continuity_windows
                    else {
                        "early": "B_early_raw",
                        "mid_policy": "B_mid_raw",
                        "late_reconciliation": "B_late_raw",
                    }[window_name]
                )
                row_b = by_step[compare_pipeline][step]
                acc = accumulators[window_name]
                acc.n_steps += 1
                deltas = top1_top20_delta(row_a, row_b, window.layers)
                acc.add(teacher.raw_category, "top1_displacement", deltas["top1_change_fraction"])
                acc.add(teacher.raw_category, "top20_entries", deltas["mean_top20_entries"])
                acc.add(teacher.raw_category, "top20_exits", deltas["mean_top20_exits"])
                acc.add(teacher.raw_category, "teacher_rank_gain", rank_gain(row_a, row_b, window.layers))
                acc.add(teacher.raw_category, "first_top1_layer", _top1_first_layer(row_b, teacher.token_id, window))
                acc.add(teacher.raw_category, "first_top5_layer", _avg_first_layer(row_b, teacher.token_id, k=5, window=window))
                acc.add(teacher.raw_category, "first_top20_layer", _avg_first_layer(row_b, teacher.token_id, k=20, window=window))
                _add_pair_displacements(
                    acc=acc,
                    row_a=row_a,
                    row_b=row_b,
                    window=window,
                    model=model,
                    model_id=model_id,
                    decode_cache=decode_cache,
                )

            mid_window = primary_windows["mid_policy"]
            late_window = primary_windows["late_reconciliation"]
            mid_first20 = first_layer_in_topk(
                row_mid.get("metrics", {}).get("top20_ids", []),
                teacher.token_id,
                k=20,
            )
            mid_first5 = first_layer_in_topk(
                row_mid.get("metrics", {}).get("top20_ids", []),
                teacher.token_id,
                k=5,
            )
            late_gain_vs_a = rank_gain(row_a, row_late, late_window.layers)
            late_gain_vs_mid = rank_gain(row_mid, row_late, late_window.layers)
            mid_selected = mid_window.contains(mid_first20) or mid_window.contains(mid_first5)
            late_reconciled = (
                (late_gain_vs_a is not None and late_gain_vs_a > 0)
                or (late_gain_vs_mid is not None and late_gain_vs_mid > 0)
            )
            handoff_acc.n_steps += 1
            handoff_acc.add(teacher.raw_category, "handoff", 1.0 if mid_selected and late_reconciled else 0.0)
            handoff_acc.add(teacher.raw_category, "teacher_rank_gain", late_gain_vs_a)
            handoff_acc.add(teacher.raw_category, "first_top5_layer", mid_first5)
            handoff_acc.add(teacher.raw_category, "first_top20_layer", mid_first20)

    duplicates = visit_prompt_rows(
        depth_dir / "step_metrics.jsonl",
        needed,
        handle_prompt,
        max_prompt_groups=max_prompts,
    )

    return {
        "model": model,
        "family": "deepseek_separate" if model == "deepseek_v2_lite" else "dense5",
        "variant": "matched_prefix_graft",
        "prompt_mode": "raw_format_b_teacher_forced",
        "source_dir": str(depth_dir),
        "n_prompts": processed_prompts,
        "n_steps": processed_steps,
        "windows": {name: acc.finalize() for name, acc in accumulators.items()},
        "handoff": handoff_acc.finalize(),
        "window_definitions": {name: window.to_json() for name, window in all_windows.items()},
        "integrity": {
            "duplicates": duplicates,
            "missing_pipeline_examples": missing[:25],
            "n_missing_pipeline_prompts": len(missing),
        },
    }


def _merge_category_metric(
    dest: dict[str, dict[str, float | int | None]],
    src: dict[str, dict[str, float | int | None]],
    keys: list[str],
) -> None:
    for cat, payload in src.items():
        dest.setdefault(cat, {"count": 0})
        src_count = int(payload.get("count") or 0)
        old_count = int(dest[cat].get("count") or 0)
        new_count = old_count + src_count
        for key in keys:
            src_mean = payload.get(key)
            old_mean = dest[cat].get(key)
            if src_mean is None:
                continue
            if old_mean is None or old_count == 0:
                dest[cat][key] = src_mean
            else:
                dest[cat][key] = (float(old_mean) * old_count + float(src_mean) * src_count) / max(new_count, 1)
        dest[cat]["count"] = new_count


def pool_models(models: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    metric_keys = [
        "fraction_top1_displaced",
        "mean_top20_entries",
        "mean_top20_exits",
        "mean_teacher_rank_gain",
        "mean_first_top1_layer",
        "mean_first_top5_layer",
        "mean_first_top20_layer",
        "handoff_rate",
    ]
    pooled: dict[str, Any] = {"models": selected, "windows": {}, "handoff": {"by_token_category": {}}}
    for model in selected:
        if model not in models:
            continue
        for window, payload in models[model]["windows"].items():
            pooled["windows"].setdefault(window, {"by_token_category": {}})
            _merge_category_metric(
                pooled["windows"][window]["by_token_category"],
                payload["by_token_category"],
                metric_keys,
            )
        _merge_category_metric(
            pooled["handoff"]["by_token_category"],
            models[model]["handoff"]["by_token_category"],
            metric_keys,
        )
    return pooled


def load_supporting_appendix() -> dict[str, Any]:
    out: dict[str, Any] = {
        "role": "supporting_appendix_only",
        "caution": "Older Exp3/Exp7 evidence is not treated as cross-family proof.",
    }
    category_path = Path("results/exp07_methodology_validation_tier0/0E/category_enrichment.json")
    if not category_path.exists():
        category_path = Path("results/exp07_methodology_validation_tier0/data/category_enrichment.json")
    if category_path.exists():
        data = json.loads(category_path.read_text())
        out["exp7_category_enrichment"] = {
            key: data.get(key)
            for key in ["A1_baseline", "A1_alpha_3", "A1_alpha_5", "A1_alpha_-3"]
            if key in data
        }
        out["exp7_category_enrichment_source"] = str(category_path)
    layer_path = Path("results/exp07_methodology_validation_tier0/data/0F_single_layer_importance.json")
    if layer_path.exists():
        data = json.loads(layer_path.read_text())
        out["exp7_single_layer_importance"] = {
            "baseline_str": data.get("baseline_str"),
            "most_important_layers": data.get("most_important_layers"),
            "source": str(layer_path),
        }
    exp3_plot_root = Path("results/exp03_corrective_stage_characterization/plots")
    out["exp3_supporting_plots"] = [
        str(exp3_plot_root / name)
        for name in [
            "plot3_attraction_repulsion.png",
            "plot4_token_stratification.png",
            "plot_e3_10_mind_change.png",
            "plot_e3_13_candidate_reshuffling.png",
        ]
        if (exp3_plot_root / name).exists()
    ]
    return out


def run_analysis(
    *,
    models: list[str],
    depth_roots: list[Path],
    out_dir: Path,
    max_prompts: int | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    started = time.time()
    decode_cache = DecodeCache()
    model_payloads: dict[str, Any] = {}
    missing_models: dict[str, str] = {}
    for model in models:
        try:
            depth_dir = resolve_depth_dir(model, depth_roots)
        except FileNotFoundError as exc:
            if allow_missing:
                missing_models[model] = str(exc)
                continue
            raise
        model_payloads[model] = analyze_model_depth(
            model=model,
            depth_dir=depth_dir,
            decode_cache=decode_cache,
            max_prompts=max_prompts,
        )
    dense_models = [m for m in ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"] if m in model_payloads]
    summary = {
        "analysis": "exp18_midlate_token_handoff",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "hypothesis": (
            "Mid layers select/expose IT candidates; late layers reconcile them with "
            "next-token prediction by supporting target tokens and suppressing alternatives."
        ),
        "models": model_payloads,
        "pooled": {
            "dense5": pool_models(model_payloads, dense_models),
            "deepseek_separate": (
                pool_models(model_payloads, ["deepseek_v2_lite"])
                if "deepseek_v2_lite" in model_payloads
                else None
            ),
        },
        "missing_models": missing_models,
        "supporting_appendix": load_supporting_appendix(),
        "limitations": [
            "Matched-prefix traces do not store per-layer logits, so support_target_delta, repulsion_top10_delta, and margin_delta are null here.",
            "Direct promote/suppress logit deltas come from the pure PT/IT collector.",
            "Findings are correlational/mediational unless followed by a targeted causal ablation.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp18 mid-to-late handoff from existing traces.")
    parser.add_argument("--models", default=",".join(MODEL_REGISTRY), help="Comma-separated model keys or 'all'.")
    parser.add_argument(
        "--depth-root",
        action="append",
        default=[],
        help="Depth-ablation root. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp18_midlate_token_handoff/matched_prefix_latest"),
    )
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = list(MODEL_REGISTRY) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    roots = [Path(p) for p in args.depth_root] or DEFAULT_DEPTH_ROOTS
    summary = run_analysis(
        models=models,
        depth_roots=roots,
        out_dir=args.out_dir,
        max_prompts=args.max_prompts,
        allow_missing=args.allow_missing,
    )
    print(f"[exp18] wrote {args.out_dir / 'summary.json'} with {len(summary['models'])} models")


if __name__ == "__main__":
    main()
