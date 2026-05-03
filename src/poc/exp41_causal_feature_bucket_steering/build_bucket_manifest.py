"""Build frozen Exp41 feature-bucket manifests from Exp39 taxonomy outputs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.poc.exp41_causal_feature_bucket_steering.config import (
    BUCKET_TO_CATEGORIES,
    EXP39_RUN,
    EXPANDED_SENSITIVITY,
    PRIMARY_MODELS,
    STRICT_PRIMARY,
    TERMINAL_LAYERS,
    ManifestMode,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


BASE_COLUMNS = [
    "manifest_mode",
    "model",
    "bucket",
    "layer",
    "latent_id",
    "feature_id",
    "causal_rank",
    "score_sum",
    "score_mean",
    "score_abs_mean",
    "active_union_rate",
    "density_bin",
    "n_events",
    "result_root",
    "paper_category",
    "paper_niche_label",
    "paper_use",
    "paper_behavior_score",
    "paper_safety_alignment_score",
    "paper_human_review_confidence",
    "paper_human_override_note",
    "include_primary",
    "include_reason",
]

CONTROL_COLUMNS = BASE_COLUMNS + [
    "control_kind",
    "matched_to_layer",
    "matched_to_latent_id",
    "matched_to_feature_id",
    "source_bucket",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _feature_id(row: dict[str, Any], role: str = "causal") -> str:
    return f"{row.get('model')}:L{int(float(row.get('layer')))}:F{int(float(row.get('latent_id')))}:{role}"


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bucket_for_category(category: str) -> str | None:
    for bucket, categories in BUCKET_TO_CATEGORIES.items():
        if category in categories:
            return bucket
    return None


def _include_reason(row: dict[str, Any], *, mode: ManifestMode) -> tuple[bool, str]:
    model = str(row.get("model"))
    layer = _int(row, "layer", -1)
    bucket = str(row.get("bucket"))
    paper_use = str(row.get("paper_use", ""))
    confidence = _float(row, "paper_human_review_confidence")
    behavior_score = _float(row, "paper_behavior_score")
    safety_score = _float(row, "paper_safety_alignment_score")

    if model not in PRIMARY_MODELS:
        return False, "not_primary_model"
    if layer not in TERMINAL_LAYERS.get(model, ()):
        return False, "outside_terminal_crosscoder_layers"
    if confidence < mode.min_confidence:
        return False, f"confidence_below_{mode.min_confidence:.2f}"

    if bucket == "artifact_repetition":
        if paper_use != "diagnostic_artifact":
            return False, "artifact_bucket_requires_diagnostic_artifact_use"
        return True, "included_artifact_diagnostic"

    if bucket == "surface_punctuation":
        if paper_use not in {"showcase", "support", "weak_support"}:
            return False, "surface_requires_showcase_support_or_weak_support"
        return True, "included_surface_positive_control"

    if bucket == "safety_advice_boundary":
        if paper_use not in {"showcase", "support", "weak_support"}:
            return False, "safety_requires_showcase_support_or_weak_support"
        if safety_score < 1:
            return False, "safety_alignment_score_below_1"
        return True, "included_human_reviewed_safety_advice_exploratory"

    if paper_use not in {"showcase", "support"}:
        if not (mode.allow_weak_support and paper_use == "weak_support"):
            return False, "behavior_bucket_requires_support_or_showcase"
    if behavior_score < 2:
        return False, "behavior_score_below_2"
    return True, "included_behavior_bucket"


def _merge_rows(
    *,
    taxonomy_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, str]],
    mode: ManifestMode,
) -> list[dict[str, Any]]:
    selected_by_id = {_feature_id(row, "causal"): row for row in selected_rows}
    candidates: list[dict[str, Any]] = []
    for tax in taxonomy_rows:
        if tax.get("role") != "causal":
            continue
        category = str(tax.get("paper_category", ""))
        bucket = _bucket_for_category(category)
        if bucket is None:
            continue
        fid = str(tax.get("feature_id") or _feature_id(tax, "causal"))
        selected = selected_by_id.get(fid)
        if selected is None:
            log.warning("Taxonomy feature missing from selected_features.csv: %s", fid)
            continue
        row: dict[str, Any] = {
            **selected,
            **tax,
            "manifest_mode": mode.name,
            "bucket": bucket,
            "feature_id": fid,
            "include_primary": False,
            "include_reason": "",
        }
        include, reason = _include_reason(row, mode=mode)
        row["include_primary"] = include
        row["include_reason"] = reason
        candidates.append(row)

    included = [row for row in candidates if row["include_primary"]]
    included.sort(
        key=lambda row: (
            str(row.get("model")),
            str(row.get("bucket")),
            -_float(row, "score_mean"),
            _int(row, "layer"),
            _int(row, "latent_id"),
        )
    )
    capped: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for row in included:
        key = (str(row["model"]), str(row["bucket"]))
        if counts[key] >= mode.max_features_per_model_bucket:
            continue
        counts[key] += 1
        capped.append(row)
    return capped


def _build_controls(
    *,
    feature_rows: list[dict[str, Any]],
    control_rows: list[dict[str, str]],
    mode: ManifestMode,
) -> list[dict[str, Any]]:
    matched_by_target: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in control_rows:
        if row.get("control_kind") != "matched_noncausal":
            continue
        key = (
            str(row.get("model")),
            _int(row, "matched_to_layer", -1),
            _int(row, "matched_to_latent_id", -1),
        )
        matched_by_target[key].append(row)

    out: list[dict[str, Any]] = []
    for source in feature_rows:
        key = (str(source.get("model")), _int(source, "layer"), _int(source, "latent_id"))
        choices = matched_by_target.get(key, [])
        if not choices:
            log.warning("No matched_noncausal control for %s", source.get("feature_id"))
            continue
        control = sorted(choices, key=lambda row: abs(_float(row, "score_mean")))[0]
        control_feature_id = _feature_id(control, "control")
        out.append(
            {
                **control,
                "manifest_mode": mode.name,
                "bucket": f"{source.get('bucket')}__matched_random",
                "feature_id": control_feature_id,
                "paper_category": "matched_noncausal_control",
                "paper_niche_label": "matched noncausal crosscoder control",
                "paper_use": "control",
                "paper_behavior_score": "",
                "paper_safety_alignment_score": "",
                "paper_human_review_confidence": "",
                "paper_human_override_note": "",
                "include_primary": True,
                "include_reason": "matched_noncausal_control_for_frozen_bucket_feature",
                "control_kind": "matched_noncausal",
                "matched_to_feature_id": source.get("feature_id"),
                "source_bucket": source.get("bucket"),
            }
        )
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("source_bucket")), _int(row, "layer"), _int(row, "latent_id")))
    return out


def _summarize(features: list[dict[str, Any]], controls: list[dict[str, Any]], *, mode: ManifestMode) -> dict[str, Any]:
    by_bucket_model: dict[str, dict[str, int]] = defaultdict(dict)
    by_bucket = Counter()
    by_model = Counter()
    for row in features:
        model = str(row.get("model"))
        bucket = str(row.get("bucket"))
        by_bucket[bucket] += 1
        by_model[model] += 1
        by_bucket_model[bucket][model] = by_bucket_model[bucket].get(model, 0) + 1
    return {
        "manifest_mode": mode.name,
        "n_features": len(features),
        "n_controls": len(controls),
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_model": dict(sorted(by_model.items())),
        "by_bucket_model": {bucket: dict(sorted(counts.items())) for bucket, counts in sorted(by_bucket_model.items())},
        "terminal_layers": {key: list(value) for key, value in TERMINAL_LAYERS.items()},
    }


def build_manifests(*, exp39_run: Path, out_dir: Path) -> None:
    taxonomy_rows = _read_jsonl(exp39_run / "autointerp" / "causal_paper_taxonomy_v3.jsonl")
    selected_rows = _read_csv(exp39_run / "feature_selection" / "selected_features.csv")
    control_rows = _read_csv(exp39_run / "feature_selection" / "control_features.csv")

    combined_summary: dict[str, Any] = {
        "exp39_run": str(exp39_run),
        "manifest_root": str(out_dir),
        "modes": {},
    }
    note_lines = [
        "# Exp41 Frozen Bucket Manifest",
        "",
        f"Source Exp39 run: `{exp39_run}`",
        "",
        "The strict-primary manifest is the paper-facing feature set. The expanded manifest is sensitivity only.",
        "",
    ]

    for mode in (STRICT_PRIMARY, EXPANDED_SENSITIVITY):
        mode_dir = out_dir / mode.name
        features = _merge_rows(taxonomy_rows=taxonomy_rows, selected_rows=selected_rows, mode=mode)
        controls = _build_controls(feature_rows=features, control_rows=control_rows, mode=mode)
        _write_csv(mode_dir / "bucket_features.csv", features, BASE_COLUMNS)
        _write_csv(mode_dir / "bucket_controls.csv", controls, CONTROL_COLUMNS)
        summary = _summarize(features, controls, mode=mode)
        _write_json(mode_dir / "bucket_manifest_summary.json", summary)
        combined_summary["modes"][mode.name] = summary

        note_lines.append(f"## {mode.name}")
        note_lines.append("")
        for bucket, count in summary["by_bucket"].items():
            models = summary["by_bucket_model"].get(bucket, {})
            model_str = ", ".join(f"{model}={n}" for model, n in models.items())
            note_lines.append(f"- `{bucket}`: {count} features ({model_str})")
        note_lines.append(f"- controls: {summary['n_controls']}")
        note_lines.append("")

    _write_json(out_dir / "bucket_manifest_summary.json", combined_summary)
    (out_dir / "bucket_manifest_note.md").write_text("\n".join(note_lines).rstrip() + "\n")
    log.info("Wrote Exp41 bucket manifests under %s", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp39-run", type=Path, default=EXP39_RUN)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifests(exp39_run=args.exp39_run, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

