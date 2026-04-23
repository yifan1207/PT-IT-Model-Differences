#!/usr/bin/env python3
"""Aggregate Exp20 divergence-token counterfactual records."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CONDITIONS = [
    "A_pt_raw",
    "B_early_raw",
    "B_mid_raw",
    "B_late_raw",
    "C_it_chat",
    "D_early_ptswap",
    "D_mid_ptswap",
    "D_late_ptswap",
]

DIVERGENCE_KINDS = ["first_diff", "first_nonformat_diff", "first_assistant_marker_diff"]
PRIMARY_WINDOWS = ["early", "mid_policy", "late_reconciliation"]
CONTINUITY_WINDOWS = ["exp11_early", "exp11_mid", "exp11_late", "condition_graft_window"]
DENSE5 = {"gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"}


class NumericStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.min_value: float | None = None
        self.max_value: float | None = None

    def add(self, value: Any) -> None:
        if value is None:
            return
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value_f):
            return
        self.count += 1
        self.total += value_f
        self.min_value = value_f if self.min_value is None else min(self.min_value, value_f)
        self.max_value = value_f if self.max_value is None else max(self.max_value, value_f)

    def finalize(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else None,
            "min": self.min_value,
            "max": self.max_value,
        }


def _stats_dict() -> defaultdict[str, NumericStats]:
    return defaultdict(NumericStats)


def _counter_fraction(counter: Counter) -> dict[str, dict[str, float | int | None]]:
    total = sum(counter.values())
    return {
        str(key): {
            "count": int(value),
            "fraction": (int(value) / total if total else None),
        }
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            yield json.loads(raw.decode("utf-8", errors="ignore"))


def find_record_files(run_dir: Path) -> list[Path]:
    merged = sorted(run_dir.rglob("exp20_records.jsonl"))
    if merged:
        return merged
    return sorted(run_dir.rglob("exp20_records_w*.jsonl"))


def _compact_event_token(event: dict[str, Any], side: str) -> dict[str, Any] | None:
    token = event.get(f"{side}_token")
    if not isinstance(token, dict):
        return None
    return {
        "token_id": token.get("token_id"),
        "token_str": token.get("token_str", ""),
        "token_category": token.get("token_category"),
        "token_category_collapsed": token.get("token_category_collapsed"),
        "assistant_marker": bool(token.get("assistant_marker", False)),
    }


class Bucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.n_records = 0
        self.prompt_ids: set[str] = set()
        self.models = Counter()
        self.prompt_modes = Counter()
        self.max_new_tokens = Counter()
        self.free_condition_steps = defaultdict(NumericStats)
        self.free_condition_short = Counter()
        self.pairwise: dict[str, dict[str, NumericStats | int]] = defaultdict(lambda: {
            "agreement_fraction": NumericStats(),
            "first_divergence_step": NumericStats(),
            "no_divergence_count": 0,
        })
        self.cluster_stats = _stats_dict()
        self.cluster_condition_steps: dict[str, dict[str, NumericStats]] = defaultdict(_stats_dict)
        self.divergence = defaultdict(self._new_divergence_bucket)
        self.readouts = defaultdict(self._new_readout_kind_bucket)

    @staticmethod
    def _new_divergence_bucket() -> dict[str, Any]:
        return {
            "present": 0,
            "missing": 0,
            "shared_prefix_clean": 0,
            "step": NumericStats(),
            "pt_categories": Counter(),
            "it_categories": Counter(),
            "pt_collapsed_categories": Counter(),
            "it_collapsed_categories": Counter(),
            "pt_assistant_marker": Counter(),
            "it_assistant_marker": Counter(),
            "examples": [],
        }

    @staticmethod
    def _new_condition_bucket() -> dict[str, Any]:
        return {
            "count": 0,
            "winner": Counter(),
            "token_class_at_step": Counter(),
            "top_layers": defaultdict(NumericStats),
            "windows": defaultdict(lambda: defaultdict(NumericStats)),
        }

    @classmethod
    def _new_readout_kind_bucket(cls) -> dict[str, Any]:
        return {
            "present": 0,
            "missing": 0,
            "by_condition": defaultdict(cls._new_condition_bucket),
        }

    def add(self, record: dict[str, Any]) -> None:
        self.n_records += 1
        prompt_id = str(record.get("prompt_id", "unknown"))
        self.prompt_ids.add(prompt_id)
        self.models[str(record.get("model", "unknown"))] += 1
        self.prompt_modes[str(record.get("prompt_mode", "unknown"))] += 1
        self.max_new_tokens[str(record.get("max_new_tokens", "unknown"))] += 1

        max_new = record.get("max_new_tokens")
        free_runs = record.get("free_runs", {})
        if isinstance(free_runs, dict):
            for condition in CONDITIONS:
                payload = free_runs.get(condition, {})
                if not isinstance(payload, dict):
                    continue
                n_steps = payload.get("n_steps")
                self.free_condition_steps[condition].add(n_steps)
                if isinstance(max_new, int) and isinstance(n_steps, int) and n_steps < max_new:
                    self.free_condition_short[condition] += 1

        for pair, payload in (record.get("pairwise_agreement") or {}).items():
            if not isinstance(payload, dict):
                continue
            bucket = self.pairwise[str(pair)]
            assert isinstance(bucket["agreement_fraction"], NumericStats)
            assert isinstance(bucket["first_divergence_step"], NumericStats)
            bucket["agreement_fraction"].add(payload.get("agreement_fraction"))
            first = payload.get("first_divergence_step")
            if first is None:
                bucket["no_divergence_count"] = int(bucket["no_divergence_count"]) + 1
            else:
                bucket["first_divergence_step"].add(first)

        cluster = record.get("cluster_summary") or {}
        for key in ["mean_cluster_entropy", "mean_unique_token_count", "mean_majority_size"]:
            self.cluster_stats[key].add(cluster.get(key))
        for field in ["leaves_majority_step", "unique_token_step"]:
            for condition, value in (cluster.get(field) or {}).items():
                self.cluster_condition_steps[field][condition].add(value)

        self._add_divergences(record)
        self._add_readouts(record)

    def _add_divergences(self, record: dict[str, Any]) -> None:
        for kind in DIVERGENCE_KINDS:
            event = (record.get("divergence_events") or {}).get(kind)
            bucket = self.divergence[kind]
            if not isinstance(event, dict):
                bucket["missing"] += 1
                continue
            bucket["present"] += 1
            if event.get("shared_prefix_clean"):
                bucket["shared_prefix_clean"] += 1
            bucket["step"].add(event.get("step"))
            for side in ["pt", "it"]:
                token = _compact_event_token(event, side)
                if token is None:
                    continue
                bucket[f"{side}_categories"][str(token.get("token_category", "unknown"))] += 1
                bucket[f"{side}_collapsed_categories"][str(token.get("token_category_collapsed", "unknown"))] += 1
                bucket[f"{side}_assistant_marker"][str(bool(token.get("assistant_marker")))] += 1
            if len(bucket["examples"]) < 5:
                bucket["examples"].append(
                    {
                        "prompt_id": record.get("prompt_id"),
                        "step": event.get("step"),
                        "pt_token": _compact_event_token(event, "pt"),
                        "it_token": _compact_event_token(event, "it"),
                    }
                )

    def _add_readouts(self, record: dict[str, Any]) -> None:
        readouts = record.get("readouts") or {}
        for kind in DIVERGENCE_KINDS:
            payload = readouts.get(kind)
            bucket = self.readouts[kind]
            if not isinstance(payload, dict):
                bucket["missing"] += 1
                continue
            bucket["present"] += 1
            token_class = payload.get("condition_token_at_step") or {}
            conditions = payload.get("conditions") or {}
            for condition in CONDITIONS:
                cond_bucket = bucket["by_condition"][condition]
                cond_bucket["count"] += 1
                cls = (token_class.get(condition) or {}).get("class", "missing")
                cond_bucket["token_class_at_step"][str(cls)] += 1
                cond_payload = conditions.get(condition)
                if not isinstance(cond_payload, dict):
                    continue
                cond_bucket["winner"][str(cond_payload.get("winner", "unknown"))] += 1
                layerwise = cond_payload.get("layerwise") or {}
                for key in [
                    "y_it_first_top1_layer",
                    "y_it_first_top5_layer",
                    "y_it_first_top20_layer",
                    "y_pt_first_top1_layer",
                    "y_pt_first_top5_layer",
                    "y_pt_first_top20_layer",
                ]:
                    cond_bucket["top_layers"][key].add(layerwise.get(key))
                for window, window_payload in (cond_payload.get("windows") or {}).items():
                    if not isinstance(window_payload, dict):
                        continue
                    for metric_name in ["y_it_logit", "y_pt_logit", "it_minus_pt_margin"]:
                        metric_payload = window_payload.get(metric_name) or {}
                        for stat_name in ["mean_step_delta", "total_delta", "start_value", "end_value"]:
                            cond_bucket["windows"][window][f"{metric_name}.{stat_name}"].add(
                                metric_payload.get(stat_name)
                            )

    def finalize(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_unique_prompts": len(self.prompt_ids),
            "models": dict(self.models),
            "prompt_modes": dict(self.prompt_modes),
            "max_new_tokens": dict(self.max_new_tokens),
            "free_runs": self._finalize_free_runs(),
            "pairwise_agreement": self._finalize_pairwise(),
            "cluster_summary": self._finalize_cluster(),
            "divergence": self._finalize_divergence(),
            "readouts": self._finalize_readouts(),
        }

    def _finalize_free_runs(self) -> dict[str, Any]:
        return {
            condition: {
                "n_steps": stats.finalize(),
                "stopped_before_limit": int(self.free_condition_short.get(condition, 0)),
                "stopped_before_limit_fraction": (
                    int(self.free_condition_short.get(condition, 0)) / stats.count if stats.count else None
                ),
            }
            for condition, stats in self.free_condition_steps.items()
        }

    def _finalize_pairwise(self) -> dict[str, Any]:
        out = {}
        for pair, payload in sorted(self.pairwise.items()):
            agreement = payload["agreement_fraction"]
            first = payload["first_divergence_step"]
            assert isinstance(agreement, NumericStats)
            assert isinstance(first, NumericStats)
            out[pair] = {
                "agreement_fraction": agreement.finalize(),
                "first_divergence_step": first.finalize(),
                "no_divergence_count": int(payload["no_divergence_count"]),
            }
        return out

    def _finalize_cluster(self) -> dict[str, Any]:
        return {
            "means": {key: stats.finalize() for key, stats in self.cluster_stats.items()},
            "condition_steps": {
                field: {
                    condition: stats.finalize()
                    for condition, stats in sorted(condition_stats.items())
                }
                for field, condition_stats in sorted(self.cluster_condition_steps.items())
            },
        }

    def _finalize_divergence(self) -> dict[str, Any]:
        out = {}
        for kind, payload in self.divergence.items():
            present = int(payload["present"])
            total = present + int(payload["missing"])
            out[kind] = {
                "present": present,
                "missing": int(payload["missing"]),
                "present_fraction": present / total if total else None,
                "shared_prefix_clean": int(payload["shared_prefix_clean"]),
                "shared_prefix_clean_fraction": int(payload["shared_prefix_clean"]) / present if present else None,
                "step": payload["step"].finalize(),
                "pt_categories": _counter_fraction(payload["pt_categories"]),
                "it_categories": _counter_fraction(payload["it_categories"]),
                "pt_collapsed_categories": _counter_fraction(payload["pt_collapsed_categories"]),
                "it_collapsed_categories": _counter_fraction(payload["it_collapsed_categories"]),
                "pt_assistant_marker": _counter_fraction(payload["pt_assistant_marker"]),
                "it_assistant_marker": _counter_fraction(payload["it_assistant_marker"]),
                "examples": payload["examples"],
            }
        return out

    def _finalize_readouts(self) -> dict[str, Any]:
        out = {}
        for kind, payload in self.readouts.items():
            out[kind] = {
                "present": int(payload["present"]),
                "missing": int(payload["missing"]),
                "by_condition": {},
            }
            for condition, condition_payload in sorted(payload["by_condition"].items()):
                windows = {
                    window: {
                        metric: stats.finalize()
                        for metric, stats in sorted(metric_payload.items())
                    }
                    for window, metric_payload in sorted(condition_payload["windows"].items())
                }
                out[kind]["by_condition"][condition] = {
                    "count": int(condition_payload["count"]),
                    "winner": _counter_fraction(condition_payload["winner"]),
                    "token_class_at_step": _counter_fraction(condition_payload["token_class_at_step"]),
                    "top_layers": {
                        key: stats.finalize()
                        for key, stats in sorted(condition_payload["top_layers"].items())
                    },
                    "windows": windows,
                }
        return out


def _bucket_names_for_model(model: str) -> list[str]:
    names = ["all6"]
    if model in DENSE5:
        names.append("dense5")
    if model == "deepseek_v2_lite":
        names.append("deepseek_only")
    return names


def analyze_run(run_dir: Path) -> dict[str, Any]:
    files = find_record_files(run_dir)
    model_buckets: dict[str, Bucket] = {}
    pooled_buckets = {
        "all6": Bucket("all6"),
        "dense5": Bucket("dense5"),
        "deepseek_only": Bucket("deepseek_only"),
    }
    malformed_rows = 0
    duplicate_prompt_ids = Counter()
    seen_ids: set[tuple[str, str]] = set()

    for path in files:
        try:
            rows = _iter_jsonl(path)
            for record in rows:
                model = str(record.get("model", "unknown"))
                prompt_id = str(record.get("prompt_id", "unknown"))
                key = (model, prompt_id)
                if key in seen_ids:
                    duplicate_prompt_ids[model] += 1
                seen_ids.add(key)
                model_buckets.setdefault(model, Bucket(model)).add(record)
                for bucket_name in _bucket_names_for_model(model):
                    pooled_buckets[bucket_name].add(record)
        except json.JSONDecodeError:
            malformed_rows += 1

    by_model = {model: bucket.finalize() for model, bucket in sorted(model_buckets.items())}
    pooled = {name: bucket.finalize() for name, bucket in pooled_buckets.items()}
    warnings = []
    if not files:
        warnings.append("No exp20_records*.jsonl files found.")
    if malformed_rows:
        warnings.append(f"Malformed JSONL rows/files encountered: {malformed_rows}")
    for model, count in sorted(duplicate_prompt_ids.items()):
        if count:
            warnings.append(f"{model}: duplicate prompt rows found: {count}")
    for model, summary in by_model.items():
        if summary["n_records"] <= 0:
            warnings.append(f"{model}: no records")
        first = summary.get("divergence", {}).get("first_diff", {})
        if first.get("present", 0) <= 0:
            warnings.append(f"{model}: no first_diff events")
    return {
        "run_dir": str(run_dir),
        "record_files": [str(path) for path in files],
        "by_model": by_model,
        "pooled": pooled,
        "quality": {
            "ok": not warnings,
            "warning_count": len(warnings),
            "warnings": warnings,
            "malformed_rows": malformed_rows,
            "duplicate_prompt_ids": dict(duplicate_prompt_ids),
            "models_seen": sorted(by_model),
            "n_record_files": len(files),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp20 divergence-token counterfactual results.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = analyze_run(args.run_dir)
    out_path = args.out or (args.run_dir / "summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[exp20] wrote {out_path}")


if __name__ == "__main__":
    main()
