"""Exp39 feature-level interpretation for causal terminal crosscoder features.

The pipeline is intentionally conservative:

* it selects features only from held-out causal rankings;
* it carries same-layer, same-active-rate noncausal controls beside them;
* dashboards keep interpretation and validation prompt splits separate;
* LLM labels are optional and marked non-paper-grade when unavailable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_raw_prompt, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp28_late_mlp_crosscoder_mediation.cache_activations import (
    _append_pt_greedy_tokens,
    _capture_variant,
    _make_batch,
    _special_ids,
)
from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import BatchTopKCrossCoder
from src.poc.exp39_causal_feature_interpretation.config import (
    ALL_FAMILIES,
    DEFAULT_DASHBOARD_DATASETS,
    DEFAULT_EXCLUDE_DATASETS,
    DEFAULT_FAMILIES,
    DEFAULT_OUT_ROOT,
    Exp39Family,
)


ONTOLOGY = [
    "formatting / structure",
    "assistant register / conversational style",
    "refusal / safety",
    "instruction compliance",
    "factuality / correction",
    "reasoning / plan structure",
    "content domain",
    "punctuation / tokenization",
    "generic frequency / unclear",
]


@dataclass(frozen=True)
class FeatureKey:
    model: str
    layer: int
    latent_id: int
    role: str
    control_kind: str

    @property
    def id(self) -> str:
        if self.role == "causal":
            return f"{self.model}:L{self.layer}:F{self.latent_id}:causal"
        return f"{self.model}:L{self.layer}:F{self.latent_id}:control:{self.control_kind}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _write_optional_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _density_bin(active_rate: float) -> str:
    if active_rate < 0.1:
        return "sparse"
    if active_rate < 0.5:
        return "medium"
    if active_rate < 0.8:
        return "dense"
    return "very_dense"


def _stable_split(prompt_id: str, validation_fraction: float) -> str:
    digest = hashlib.sha1(prompt_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if value < validation_fraction else "interpretation"


def _families_from_args(names: list[str] | None) -> dict[str, Exp39Family]:
    if not names:
        return dict(DEFAULT_FAMILIES)
    out: dict[str, Exp39Family] = {}
    for name in names:
        if name not in ALL_FAMILIES:
            raise ValueError(f"Unknown Exp39 family {name!r}; choose from {sorted(ALL_FAMILIES)}")
        out[name] = ALL_FAMILIES[name]
    return out


def _run_dir(out_root: Path, run_name: str) -> Path:
    return out_root / run_name


def _artifact_status(family: Exp39Family) -> dict[str, Any]:
    root = family.result_root
    layers = list(family.layers)
    required = {
        "result_root": root,
        "causal_feature_scores": root / "feature_stats" / "causal_feature_scores.csv",
        "causal_top_features": root / "feature_stats" / "causal_top_features.csv",
        "analysis_summary": root / "analysis" / "summary.json",
    }
    layer_rows = []
    missing: list[str] = []
    for name, path in required.items():
        if not Path(path).exists():
            missing.append(name)
    for layer in layers:
        layer_dir = root / "dictionaries" / f"layer_{layer}"
        layer_payload = {
            "layer": layer,
            "config": str(layer_dir / "config.json"),
            "crosscoder": str(layer_dir / "crosscoder.pt"),
            "cache": str(root / "cache" / f"layer_{layer}.pt"),
            "config_exists": (layer_dir / "config.json").exists(),
            "crosscoder_exists": (layer_dir / "crosscoder.pt").exists(),
            "cache_exists": (root / "cache" / f"layer_{layer}.pt").exists(),
        }
        for key in ("config_exists", "crosscoder_exists"):
            if not layer_payload[key]:
                missing.append(f"layer_{layer}_{key.removesuffix('_exists')}")
        layer_rows.append(layer_payload)
    all_latents = root / "feature_stats" / "all_latents.csv"
    return {
        "model": family.model,
        "status": family.status,
        "result_root": str(root),
        "gcs_uri": family.gcs_uri,
        "layers": layers,
        "required": {key: str(path) for key, path in required.items()},
        "required_exists": {key: Path(path).exists() for key, path in required.items()},
        "layers_detail": layer_rows,
        "all_latents": str(all_latents),
        "all_latents_exists": all_latents.exists(),
        "missing_required": sorted(set(missing)),
        "selected_diagnostics_materializable": True,
        "notes": [
            "Exp39 can compute selected/control diagnostics from fresh dashboard activations "
            "when all_latents or training caches are absent.",
        ],
    }


def preflight(
    *,
    out_root: Path,
    run_name: str,
    families: dict[str, Exp39Family],
    strict: bool,
) -> dict[str, Any]:
    run_dir = _run_dir(out_root, run_name)
    manifest = {
        "run_name": run_name,
        "created_unix": time.time(),
        "families": {name: _artifact_status(family) for name, family in families.items()},
    }
    missing = {
        name: payload["missing_required"]
        for name, payload in manifest["families"].items()
        if payload["missing_required"]
    }
    _write_json(run_dir / "preflight" / "artifact_manifest.json", manifest)
    _write_json(run_dir / "preflight" / "missing_artifacts.json", missing)
    if strict and missing:
        raise RuntimeError(f"Missing required Exp39 artifacts: {missing}")
    return manifest


def _load_layer_config(root: Path, layer: int) -> dict[str, Any]:
    return _read_json(root / "dictionaries" / f"layer_{layer}" / "config.json", {}) or {}


def _summary_effects(root: Path) -> dict[str, Any]:
    summary = _read_json(root / "analysis" / "summary.json", {}) or {}
    out: dict[str, Any] = {}
    gates = summary.get("success_gates") or {}
    for key, value in gates.items():
        out[f"gate_{key}"] = value
    for effect in summary.get("effects") or []:
        if effect.get("feature_set") == "causal_top" and int(effect.get("k", -1)) in {25, 50, 100, 200}:
            k = int(effect["k"])
            out[f"causal_top{k}_interaction_drop_mean"] = effect.get("interaction_drop_mean")
            out[f"causal_top{k}_mediation_fraction_mean"] = effect.get("mediation_fraction_mean")
        if effect.get("feature_set") == "causal_matched_random" and int(effect.get("k", -1)) in {25, 50, 100, 200}:
            k = int(effect["k"])
            out.setdefault(f"causal_matched_random{k}_interaction_drop_values", []).append(
                effect.get("interaction_drop_mean")
            )
    for key, values in list(out.items()):
        if key.endswith("_values"):
            numeric = [float(v) for v in values if v not in (None, "")]
            mean_key = key.removesuffix("_values") + "_mean"
            out[mean_key] = sum(numeric) / len(numeric) if numeric else None
    return out


def _matched_controls(
    *,
    rows: list[dict[str, str]],
    selected: list[dict[str, Any]],
    per_selected: int,
    seed: int,
    exclude_keys: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected_keys = {(int(row["layer"]), int(row["latent_id"])) for row in selected}
    excluded = set(exclude_keys or set()) | selected_keys
    by_layer: dict[int, list[dict[str, str]]] = defaultdict(list)
    score_abs_values = [abs(_float(row, "score_mean")) for row in rows]
    score_cutoff = sorted(score_abs_values)[max(0, int(0.35 * max(len(score_abs_values), 1)) - 1)] if rows else 0.0
    for row in rows:
        key = (_int(row, "layer"), _int(row, "latent_id"))
        if key in excluded:
            continue
        if _float(row, "score_mean") > 0 and abs(_float(row, "score_mean")) > score_cutoff:
            continue
        by_layer[key[0]].append(row)
    controls: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    for target in selected:
        target_layer = int(target["layer"])
        target_rate = float(target.get("active_union_rate", 0.0))
        target_abs = abs(float(target.get("score_abs_mean", target.get("score_mean", 0.0))))
        pool = [row for row in by_layer.get(target_layer, []) if (_int(row, "layer"), _int(row, "latent_id")) not in used]
        if not pool:
            pool = [row for row in rows if (_int(row, "layer"), _int(row, "latent_id")) not in excluded | used]
        ranked = sorted(
            pool,
            key=lambda row: (
                abs(math.log1p(_float(row, "active_union_rate")) - math.log1p(target_rate)),
                abs(math.log1p(abs(_float(row, "score_abs_mean", abs(_float(row, "score_mean"))))) - math.log1p(target_abs)),
                rng.random(),
            ),
        )
        for row in ranked[:per_selected]:
            key = (_int(row, "layer"), _int(row, "latent_id"))
            used.add(key)
            control = {**row}
            control["matched_to_layer"] = target_layer
            control["matched_to_latent_id"] = int(target["latent_id"])
            controls.append(_normalize_feature_row(control, role="control", control_kind="matched_noncausal"))
    return controls


def _random_active_controls(
    *,
    rows: list[dict[str, str]],
    selected: list[dict[str, Any]],
    per_selected: int,
    seed: int,
    exclude_keys: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Sample active noncausal controls from the trained dictionary background.

    This is intentionally not pure random. Pure random latents are often dead or
    tiny-norm, which makes them an unfair control for dashboard interpretability.
    The random-active control asks whether a typical active, low-causal-score
    crosscoder feature from the same layer/density regime gets equally good labels.
    """

    rng = random.Random(seed)
    selected_keys = {(int(row["layer"]), int(row["latent_id"])) for row in selected}
    excluded = set(exclude_keys or set()) | selected_keys
    abs_scores = sorted(abs(_float(row, "score_mean")) for row in rows)
    score_cutoff = abs_scores[max(0, int(0.50 * max(len(abs_scores), 1)) - 1)] if abs_scores else 0.0
    pools: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    layer_pools: dict[int, list[dict[str, str]]] = defaultdict(list)
    global_pool: list[dict[str, str]] = []
    for row in rows:
        key = (_int(row, "layer"), _int(row, "latent_id"))
        if key in excluded:
            continue
        active_rate = _float(row, "active_union_rate")
        if active_rate <= 0.0:
            continue
        if _float(row, "score_mean") > 0 and abs(_float(row, "score_mean")) > score_cutoff:
            continue
        density = _density_bin(active_rate)
        layer = key[0]
        pools[(layer, density)].append(row)
        layer_pools[layer].append(row)
        global_pool.append(row)

    controls: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    for target in selected:
        target_layer = int(target["layer"])
        target_density = str(target.get("density_bin") or _density_bin(float(target.get("active_union_rate", 0.0))))
        candidates = [
            row for row in pools.get((target_layer, target_density), [])
            if (_int(row, "layer"), _int(row, "latent_id")) not in used
        ]
        if not candidates:
            candidates = [
                row for row in layer_pools.get(target_layer, [])
                if (_int(row, "layer"), _int(row, "latent_id")) not in used
            ]
        if not candidates:
            candidates = [row for row in global_pool if (_int(row, "layer"), _int(row, "latent_id")) not in used]
        rng.shuffle(candidates)
        for row in candidates[:per_selected]:
            key = (_int(row, "layer"), _int(row, "latent_id"))
            used.add(key)
            control = {**row}
            control["matched_to_layer"] = target_layer
            control["matched_to_latent_id"] = int(target["latent_id"])
            controls.append(_normalize_feature_row(control, role="control", control_kind="random_active_noncausal"))
    return controls


def _normalize_feature_row(
    row: dict[str, Any],
    *,
    role: str,
    control_kind: str = "",
    rank: int | None = None,
    model: str | None = None,
    result_root: Path | None = None,
) -> dict[str, Any]:
    active_rate = _float(row, "active_union_rate")
    score_mean = _float(row, "score_mean")
    out: dict[str, Any] = {
        "model": model or row.get("model", ""),
        "result_root": str(result_root or row.get("result_root", "")),
        "layer": _int(row, "layer"),
        "latent_id": _int(row, "latent_id"),
        "role": role,
        "control_kind": control_kind,
        "causal_rank": rank if rank is not None else row.get("causal_rank", ""),
        "score_sum": _float(row, "score_sum"),
        "score_mean": score_mean,
        "score_pos_mean": _float(row, "score_pos_mean"),
        "score_abs_mean": _float(row, "score_abs_mean", abs(score_mean)),
        "native_attr_mean": _float(row, "native_attr_mean"),
        "ptup_attr_mean": _float(row, "ptup_attr_mean"),
        "active_native_count": _int(row, "active_native_count"),
        "active_ptup_count": _int(row, "active_ptup_count"),
        "active_union_count": _int(row, "active_union_count"),
        "active_union_rate": active_rate,
        "density_bin": _density_bin(active_rate),
        "n_events": _int(row, "n_events"),
        "matched_to_layer": row.get("matched_to_layer", ""),
        "matched_to_latent_id": row.get("matched_to_latent_id", ""),
    }
    return out


def select_features(
    *,
    out_root: Path,
    run_name: str,
    families: dict[str, Exp39Family],
    top_n: int,
    control_per_feature: int,
    random_control_per_feature: int,
    seed: int,
) -> dict[str, Any]:
    run_dir = _run_dir(out_root, run_name)
    selected_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"run_name": run_name, "top_n": top_n, "families": {}}
    for name, family in families.items():
        root = family.result_root
        score_rows = _read_csv(root / "feature_stats" / "causal_feature_scores.csv")
        if not score_rows:
            raise FileNotFoundError(f"Missing causal feature score CSV for {name}: {root}")
        positive = [row for row in score_rows if _float(row, "score_mean") > 0]
        positive.sort(key=lambda row: _float(row, "score_mean"), reverse=True)
        chosen = []
        for rank, row in enumerate(positive[:top_n], start=1):
            item = _normalize_feature_row(row, role="causal", rank=rank, model=family.model, result_root=root)
            chosen.append(item)
        controls = _matched_controls(
            rows=score_rows,
            selected=chosen,
            per_selected=control_per_feature,
            seed=seed + len(selected_rows),
        )
        used_control_keys = {(_int(row, "layer"), _int(row, "latent_id")) for row in controls}
        random_controls = _random_active_controls(
            rows=score_rows,
            selected=chosen,
            per_selected=random_control_per_feature,
            seed=seed + 1009 + len(selected_rows),
            exclude_keys=used_control_keys,
        )
        controls.extend(random_controls)
        for control in controls:
            control["model"] = family.model
            control["result_root"] = str(root)
        effects = _summary_effects(root)
        layer_metrics = {}
        for layer in family.layers:
            config = _load_layer_config(root, layer)
            metrics = config.get("metrics", {})
            layer_metrics[str(layer)] = {
                "heldout_variance_explained_pt": metrics.get("heldout_variance_explained_pt"),
                "heldout_variance_explained_it": metrics.get("heldout_variance_explained_it"),
                "effective_l0": metrics.get("effective_l0"),
                "alive_fraction": metrics.get("alive_fraction"),
                "inference_threshold": metrics.get("inference_threshold"),
                "dict_size": (config.get("crosscoder") or {}).get("dict_size"),
                "k": (config.get("crosscoder") or {}).get("k"),
            }
        for row in chosen + controls:
            row.update({key: value for key, value in effects.items() if not isinstance(value, list)})
            metrics = layer_metrics.get(str(row["layer"]), {})
            for metric_key, value in metrics.items():
                row[f"dict_{metric_key}"] = value
        selected_rows.extend(chosen)
        control_rows.extend(controls)
        diag_rows.extend(chosen + controls)
        summary["families"][name] = {
            "result_root": str(root),
            "layers": list(family.layers),
            "n_positive_features": len(positive),
            "n_selected": len(chosen),
            "n_controls": len(controls),
            "n_matched_controls": sum(1 for row in controls if row.get("control_kind") == "matched_noncausal"),
            "n_random_active_controls": sum(
                1 for row in controls if row.get("control_kind") == "random_active_noncausal"
            ),
            "density_counts_selected": _count_by(chosen, "density_bin"),
            "density_counts_controls": _count_by(controls, "density_bin"),
            "control_kind_counts": _count_by(controls, "control_kind"),
            "effect_summary": effects,
            "layer_metrics": layer_metrics,
        }
    _write_csv(run_dir / "feature_selection" / "selected_features.csv", selected_rows)
    _write_csv(run_dir / "feature_selection" / "control_features.csv", control_rows)
    _write_csv(run_dir / "feature_selection" / "selected_feature_diagnostics.csv", diag_rows)
    _write_json(run_dir / "feature_selection" / "selected_features_summary.json", summary)
    return summary


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        out[value] = out.get(value, 0) + 1
    return out


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_dashboard_records(
    *,
    dataset_paths: list[Path],
    exclude_dataset_paths: list[Path],
    n_prompts: int,
) -> list[dict[str, Any]]:
    excluded_ids: set[str] = set()
    excluded_prompts: set[str] = set()
    for path in exclude_dataset_paths:
        if not path.exists():
            continue
        for row in _load_jsonl_records(path):
            excluded_ids.add(str(row.get("id", row.get("record_id", ""))))
            excluded_prompts.add(get_raw_prompt(row))
    out: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for path in dataset_paths:
        if not path.exists():
            continue
        for row in _load_jsonl_records(path):
            prompt_id = str(row.get("id", row.get("record_id", len(out))))
            raw = get_raw_prompt(row)
            if prompt_id in excluded_ids or raw in excluded_prompts:
                continue
            if raw in seen_prompts:
                continue
            seen_prompts.add(raw)
            item = dict(row)
            item["_exp39_source_path"] = str(path)
            item["_exp39_raw_prompt"] = raw
            out.append(item)
            if len(out) >= n_prompts:
                return out
    return out


def _tokenize_dashboard_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        prompt_id = str(record.get("id", record.get("record_id", idx)))
        raw = str(record.get("_exp39_raw_prompt") or get_raw_prompt(record))
        input_ids = tokenizer.encode(raw, add_special_tokens=True)
        if max_seq_len > 0:
            input_ids = input_ids[:max_seq_len]
        if len(input_ids) < 2:
            continue
        rows.append(
            {
                "prompt_id": prompt_id,
                "input_ids": [int(x) for x in input_ids],
                "prompt_token_len": len(input_ids),
                "raw_prompt": raw,
                "source_path": str(record.get("_exp39_source_path", "")),
                "category": str(record.get("category", record.get("task", record.get("dataset", "")))),
                "record": record,
            }
        )
    return rows


def _encode_selected(
    crosscoder: BatchTopKCrossCoder,
    x: torch.Tensor,
    *,
    branch: int,
    latent_ids: torch.Tensor,
    use_threshold: bool,
) -> torch.Tensor:
    if latent_ids.numel() == 0:
        return torch.zeros((x.shape[0], 0), device=x.device, dtype=torch.float32)
    dtype = crosscoder.encoder.dtype
    latent_ids = latent_ids.to(device=x.device, dtype=torch.long)
    mean = crosscoder.input_mean[branch].to(device=x.device, dtype=dtype)
    encoder = crosscoder.encoder[branch, :, latent_ids].to(device=x.device, dtype=dtype)
    bias = crosscoder.encoder_bias[latent_ids].to(device=x.device, dtype=dtype)
    acts = F.relu((x.to(dtype=dtype) - mean) @ encoder + bias)
    if use_threshold and float(crosscoder.inference_threshold.item()) > 0:
        threshold = crosscoder.inference_threshold.to(device=x.device, dtype=acts.dtype)
        acts = acts * (acts >= threshold).to(acts.dtype)
    return acts.float()


def _context_for_token(
    *,
    tokenizer: Any,
    row: dict[str, Any],
    token_pos: int,
    token_id: int,
    context_before: int,
    context_after: int,
) -> dict[str, Any]:
    ids = [int(x) for x in row["input_ids"]]
    start = max(0, token_pos - context_before)
    end = min(len(ids), token_pos + context_after + 1)
    before = tokenizer.decode(ids[start:token_pos], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    token = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    after = tokenizer.decode(ids[token_pos + 1 : end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    full_prompt = str(row.get("raw_prompt", ""))
    return {
        "context_before": before,
        "token_text": token,
        "context_after": after,
        "context_window": before + token + after,
        "raw_prompt": full_prompt if len(full_prompt) <= 2000 else full_prompt[:2000] + "...",
    }


def _candidate_example(
    *,
    tokenizer: Any,
    row: dict[str, Any],
    token_pos: int,
    token_id: int,
    feature: FeatureKey,
    kind: str,
    score: float,
    activation_it: float,
    activation_pt: float,
    context_before: int,
    context_after: int,
) -> dict[str, Any]:
    ctx = _context_for_token(
        tokenizer=tokenizer,
        row=row,
        token_pos=token_pos,
        token_id=token_id,
        context_before=context_before,
        context_after=context_after,
    )
    token_source = "prompt" if token_pos < int(row.get("prompt_token_len", 0)) else "pt_greedy_continuation"
    return {
        "feature_id": feature.id,
        "model": feature.model,
        "layer": feature.layer,
        "latent_id": feature.latent_id,
        "role": feature.role,
        "control_kind": feature.control_kind,
        "kind": kind,
        "score": float(score),
        "activation_it": float(activation_it),
        "activation_pt": float(activation_pt),
        "prompt_id": str(row["prompt_id"]),
        "prompt_category": str(row.get("category", "")),
        "source_path": str(row.get("source_path", "")),
        "token_index": int(token_pos),
        "token_id": int(token_id),
        "token_source": token_source,
        "is_generated_continuation": token_source != "prompt",
        "split": _stable_split(str(row["prompt_id"]), 0.25),
        **ctx,
    }


class TopHeap:
    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.items: list[tuple[float, int, dict[str, Any]]] = []
        self.counter = 0

    def add(self, score: float, item: dict[str, Any]) -> None:
        if not math.isfinite(score):
            return
        key = (float(score), self.counter, item)
        self.counter += 1
        if len(self.items) < self.limit:
            heapq.heappush(self.items, key)
        elif score > self.items[0][0]:
            heapq.heapreplace(self.items, key)

    def rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = [item for _, _, item in sorted(self.items, key=lambda x: x[0], reverse=True)]
        if limit is not None:
            rows = rows[:limit]
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for row in rows:
            key = (str(row.get("prompt_id")), int(row.get("token_index", -1)), str(row.get("kind", "")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if limit is not None and len(deduped) >= limit:
                break
        return deduped


def _load_feature_rows(run_dir: Path, families: dict[str, Exp39Family], *, include_controls: bool) -> list[dict[str, Any]]:
    path = run_dir / "feature_selection" / "selected_feature_diagnostics.csv"
    rows = _read_csv(path)
    if not rows:
        raise FileNotFoundError(f"Run feature selection first; missing {path}")
    allowed = set(families)
    out = []
    for row in rows:
        if row.get("model") not in allowed:
            continue
        if row.get("role") == "control" and not include_controls:
            continue
        out.append(row)
    return out


def _load_crosscoders(
    *,
    root: Path,
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[int, BatchTopKCrossCoder]:
    out: dict[int, BatchTopKCrossCoder] = {}
    for layer in layers:
        ckpt = root / "dictionaries" / f"layer_{layer}" / "crosscoder.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing crosscoder checkpoint: {ckpt}")
        model = BatchTopKCrossCoder.load(ckpt, device=device)
        if dtype is not None:
            model = model.to(dtype=dtype)
        model.eval()
        out[layer] = model
    return out


def _projection_for_features(
    *,
    crosscoder: BatchTopKCrossCoder,
    latent_ids: list[int],
    tokenizer: Any,
    unembed_weight: torch.Tensor,
    top_k: int,
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    weight = unembed_weight.detach().float()
    device = weight.device
    for latent_id in latent_ids:
        decoder_it = crosscoder.decoder[int(latent_id), 1, :].detach().float().to(device)
        scores = weight @ decoder_it
        k = min(int(top_k), scores.numel())
        promoted = torch.topk(scores, k=k, largest=True)
        suppressed = torch.topk(scores, k=k, largest=False)
        out[int(latent_id)] = {
            "projection_kind": "decoder_it_to_unembedding_local_linear_proxy",
            "top_promoted": [
                {
                    "token_id": int(idx),
                    "token_text": tokenizer.decode([int(idx)], skip_special_tokens=False),
                    "score": float(val),
                }
                for val, idx in zip(promoted.values.tolist(), promoted.indices.tolist(), strict=False)
            ],
            "top_suppressed": [
                {
                    "token_id": int(idx),
                    "token_text": tokenizer.decode([int(idx)], skip_special_tokens=False),
                    "score": float(val),
                }
                for val, idx in zip(suppressed.values.tolist(), suppressed.indices.tolist(), strict=False)
            ],
        }
    return out


def _decoder_diagnostics(crosscoder: BatchTopKCrossCoder, latent_id: int) -> dict[str, float]:
    dec = crosscoder.decoder.detach().float()
    pt = dec[int(latent_id), 0, :]
    it = dec[int(latent_id), 1, :]
    norm_pt = float(pt.norm().item())
    norm_it = float(it.norm().item())
    cosine = float(F.cosine_similarity(pt[None, :], it[None, :], dim=1).item())
    return {
        "decoder_norm_pt": norm_pt,
        "decoder_norm_it": norm_it,
        "decoder_norm_ratio_it_pt": (norm_it + 1e-8) / (norm_pt + 1e-8),
        "decoder_cosine_pt_it": cosine,
    }


def collect_dashboards(
    *,
    out_root: Path,
    run_name: str,
    families: dict[str, Exp39Family],
    datasets: list[Path],
    exclude_datasets: list[Path],
    n_prompts: int,
    max_seq_len: int,
    append_pt_greedy_tokens: int,
    batch_size: int,
    examples_per_kind: int,
    validation_examples_per_kind: int,
    context_before: int,
    context_after: int,
    include_controls: bool,
    use_threshold: bool,
    device_name: str,
    crosscoder_dtype_name: str,
    projection_top_k: int,
    prompt_shard_index: int = 0,
    prompt_shard_count: int = 1,
    output_suffix: str = "",
) -> dict[str, Any]:
    if prompt_shard_count < 1:
        raise ValueError(f"prompt_shard_count must be >= 1, got {prompt_shard_count}")
    if prompt_shard_index < 0 or prompt_shard_index >= prompt_shard_count:
        raise ValueError(
            f"prompt_shard_index must be in [0, {prompt_shard_count}), got {prompt_shard_index}"
        )
    run_dir = _run_dir(out_root, run_name)
    feature_rows = _load_feature_rows(run_dir, families, include_controls=include_controls)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_model[str(row["model"])].append(row)
    all_dashboard_rows: list[dict[str, Any]] = []
    all_observed_rows: list[dict[str, Any]] = []
    dtype = None
    if crosscoder_dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif crosscoder_dtype_name == "float16":
        dtype = torch.float16
    elif crosscoder_dtype_name == "float32":
        dtype = torch.float32
    summary: dict[str, Any] = {"run_name": run_name, "families": {}}
    for model_name, rows in by_model.items():
        family = families[model_name]
        device = torch.device(device_name)
        spec = get_spec(model_name)
        layers_needed = sorted({int(row["layer"]) for row in rows})
        features_by_layer: dict[int, list[FeatureKey]] = defaultdict(list)
        diag_by_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = FeatureKey(
                model=model_name,
                layer=int(row["layer"]),
                latent_id=int(row["latent_id"]),
                role=str(row["role"]),
                control_kind=str(row.get("control_kind", "")),
            )
            features_by_layer[key.layer].append(key)
            diag_by_key[key.id] = dict(row)

        pt_model, pt_tokenizer = load_model_and_tokenizer(
            model_id_for_variant(spec, "pt"),
            device,
            multi_gpu=spec.multi_gpu,
        )
        it_model, _it_tokenizer = load_model_and_tokenizer(
            model_id_for_variant(spec, "it"),
            device,
            multi_gpu=spec.multi_gpu,
        )
        pt_model.requires_grad_(False)
        it_model.requires_grad_(False)
        adapter = get_steering_adapter(model_name)
        pt_layers = adapter.get_layers(pt_model)
        it_layers = adapter.get_layers(it_model)
        crosscoders = _load_crosscoders(root=family.result_root, layers=layers_needed, device=device, dtype=dtype)
        all_records = _load_dashboard_records(
            dataset_paths=datasets,
            exclude_dataset_paths=exclude_datasets,
            n_prompts=n_prompts,
        )
        if prompt_shard_count > 1:
            records = [
                record
                for idx, record in enumerate(all_records)
                if idx % prompt_shard_count == prompt_shard_index
            ]
        else:
            records = all_records
        token_rows = _tokenize_dashboard_records(records, pt_tokenizer, max_seq_len=max_seq_len)
        if append_pt_greedy_tokens > 0:
            _append_pt_greedy_tokens(
                model=pt_model,
                tokenizer=pt_tokenizer,
                rows=token_rows,
                device=device,
                max_new_tokens=append_pt_greedy_tokens,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
            )
        special_ids = _special_ids(pt_tokenizer)
        heaps: dict[str, dict[str, dict[str, TopHeap]]] = defaultdict(lambda: defaultdict(dict))
        stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        projection_by_feature: dict[str, dict[str, Any]] = {}
        decoder_diag_by_feature: dict[str, dict[str, Any]] = {}
        unembed = it_model.get_output_embeddings().weight.detach()
        for layer, keys in features_by_layer.items():
            latent_ids = [key.latent_id for key in keys]
            projections = _projection_for_features(
                crosscoder=crosscoders[layer],
                latent_ids=latent_ids,
                tokenizer=pt_tokenizer,
                unembed_weight=unembed,
                top_k=projection_top_k,
            )
            for key in keys:
                projection_by_feature[key.id] = projections[key.latent_id]
                decoder_diag_by_feature[key.id] = _decoder_diagnostics(crosscoders[layer], key.latent_id)
                for split in ("interpretation", "validation"):
                    multiplier = 4 if split == "interpretation" else 3
                    limit = (examples_per_kind if split == "interpretation" else validation_examples_per_kind) * multiplier
                    for kind in (
                        "top_it",
                        "top_pt",
                        "contrast_it_over_pt",
                        "contrast_pt_over_it",
                        "inactive_random",
                    ):
                        heaps[key.id][split][kind] = TopHeap(max(limit, 1))
        n_tokens_seen = 0
        for start in range(0, len(token_rows), batch_size):
            batch_rows = token_rows[start : start + batch_size]
            input_ids, attention_mask, keep, prompt_ids, token_pos, token_ids = _make_batch(
                batch_rows,
                tokenizer=pt_tokenizer,
                device=device,
                special_ids=special_ids,
            )
            if not prompt_ids:
                continue
            pt_caps = _capture_variant(
                model=pt_model,
                layers=pt_layers,
                layer_ids=layers_needed,
                input_ids=input_ids,
                attention_mask=attention_mask,
                keep_mask=keep,
            )
            it_caps = _capture_variant(
                model=it_model,
                layers=it_layers,
                layer_ids=layers_needed,
                input_ids=input_ids,
                attention_mask=attention_mask,
                keep_mask=keep,
            )
            row_by_prompt = {str(row["prompt_id"]): row for row in batch_rows}
            meta_rows = []
            for idx, prompt_id in enumerate(prompt_ids):
                row = row_by_prompt[str(prompt_id)]
                meta_rows.append(
                    {
                        "row": row,
                        "prompt_id": str(prompt_id),
                        "token_pos": int(token_pos[idx].item()),
                        "token_id": int(token_ids[idx].item()),
                        "split": _stable_split(str(prompt_id), 0.25),
                    }
                )
            n_tokens_seen += len(meta_rows)
            for layer, keys in features_by_layer.items():
                latent_tensor = torch.tensor([key.latent_id for key in keys], dtype=torch.long, device=device)
                z_pt = _encode_selected(
                    crosscoders[layer],
                    pt_caps[layer].to(device=device, dtype=torch.float32),
                    branch=0,
                    latent_ids=latent_tensor,
                    use_threshold=use_threshold,
                )
                z_it = _encode_selected(
                    crosscoders[layer],
                    it_caps[layer].to(device=device, dtype=torch.float32),
                    branch=1,
                    latent_ids=latent_tensor,
                    use_threshold=use_threshold,
                )
                for col, key in enumerate(keys):
                    pt_vals = z_pt[:, col].detach().cpu()
                    it_vals = z_it[:, col].detach().cpu()
                    stats[key.id]["n_tokens"] += float(len(meta_rows))
                    stats[key.id]["sum_activation_pt"] += float(pt_vals.sum().item())
                    stats[key.id]["sum_activation_it"] += float(it_vals.sum().item())
                    stats[key.id]["fire_pt"] += float((pt_vals > 0).sum().item())
                    stats[key.id]["fire_it"] += float((it_vals > 0).sum().item())
                    union = (pt_vals > 0) | (it_vals > 0)
                    stats[key.id]["fire_union"] += float(union.sum().item())
                    if len(meta_rows) == 0:
                        continue
                    _add_top_examples_for_feature(
                        tokenizer=pt_tokenizer,
                        key=key,
                        meta_rows=meta_rows,
                        it_vals=it_vals,
                        pt_vals=pt_vals,
                        heaps=heaps,
                        examples_per_kind=max(examples_per_kind, validation_examples_per_kind),
                        context_before=context_before,
                        context_after=context_after,
                    )
            if (start // max(batch_size, 1)) % 10 == 0:
                print(f"[exp39-dashboard] {model_name} batches={start // max(batch_size, 1)} tokens={n_tokens_seen}", flush=True)

        for key_id, split_heaps in heaps.items():
            row_diag = diag_by_key[key_id]
            examples = {
                kind: heap.rows(examples_per_kind)
                for kind, heap in split_heaps["interpretation"].items()
            }
            validation_examples = {
                kind: heap.rows(validation_examples_per_kind)
                for kind, heap in split_heaps["validation"].items()
            }
            examples["near_miss"] = _near_misses(examples, limit=examples_per_kind)
            validation_examples["near_miss"] = _near_misses(validation_examples, limit=validation_examples_per_kind)
            observed = _observed_stats(stats[key_id])
            feature_payload = {
                **row_diag,
                **decoder_diag_by_feature.get(key_id, {}),
                **observed,
                "feature_id": key_id,
                "examples": examples,
                "validation_examples": validation_examples,
                "output_projection": projection_by_feature.get(key_id, {}),
                "causal_effect_summary": _feature_causal_summary(row_diag),
                "dashboard_split_note": "examples use prompt-id interpretation split; validation_examples are held out from autointerp prompts",
            }
            all_dashboard_rows.append(feature_payload)
            flat_observed = {
                "feature_id": key_id,
                "model": row_diag.get("model"),
                "layer": row_diag.get("layer"),
                "latent_id": row_diag.get("latent_id"),
                "role": row_diag.get("role"),
                "control_kind": row_diag.get("control_kind"),
                **decoder_diag_by_feature.get(key_id, {}),
                **observed,
            }
            all_observed_rows.append(flat_observed)
        summary["families"][model_name] = {
            "n_prompts_requested": n_prompts,
            "n_prompt_records_total_before_shard": len(all_records),
            "prompt_shard_index": prompt_shard_index,
            "prompt_shard_count": prompt_shard_count,
            "n_prompts_used": len(token_rows),
            "n_tokens_seen": n_tokens_seen,
            "layers": layers_needed,
            "n_features": len(rows),
            "append_pt_greedy_tokens": append_pt_greedy_tokens,
            "max_seq_len": max_seq_len,
            "use_threshold": use_threshold,
        }
        del pt_model, it_model, crosscoders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dash_dir = run_dir / "dashboards"
    suffix = f"_{output_suffix}" if output_suffix else ""
    _write_jsonl(dash_dir / f"feature_dashboards{suffix}.jsonl", all_dashboard_rows)
    _write_optional_parquet(dash_dir / f"feature_dashboards{suffix}.parquet", all_dashboard_rows)
    _write_csv(dash_dir / f"feature_observed_stats{suffix}.csv", all_observed_rows)
    _write_json(dash_dir / f"dashboard_summary{suffix}.json", summary)
    _write_dashboard_report(dash_dir / f"feature_dashboard_report{suffix}.md", all_dashboard_rows, summary)
    return summary


def merge_dashboards(
    *,
    out_root: Path,
    run_name: str,
    suffixes: list[str],
) -> dict[str, Any]:
    run_dir = _run_dir(out_root, run_name)
    dash_dir = run_dir / "dashboards"
    rows_by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    merged_summary: dict[str, Any] = {"run_name": run_name, "families": {}, "merged_suffixes": suffixes}
    for suffix in suffixes:
        suffix_part = f"_{suffix}" if suffix else ""
        for row in _dashboard_rows(dash_dir / f"feature_dashboards{suffix_part}.jsonl"):
            rows_by_feature[str(row.get("feature_id"))].append(row)
        for row in _read_csv(dash_dir / f"feature_observed_stats{suffix_part}.csv"):
            observed_by_feature[str(row.get("feature_id"))].append(row)
        summary = _read_json(dash_dir / f"dashboard_summary{suffix_part}.json", {}) or {}
        for model, payload in summary.get("families", {}).items():
            merged_summary["families"][model] = _merge_family_summary(
                merged_summary["families"].get(model, {}),
                payload,
            )
    rows = [_merge_dashboard_feature_rows(parts) for parts in rows_by_feature.values()]
    observed_rows = [_merge_observed_feature_rows(parts) for parts in observed_by_feature.values()]
    rows.sort(key=lambda row: (str(row.get("model")), str(row.get("role")), -_float(row, "score_mean")))
    observed_rows.sort(key=lambda row: (str(row.get("model")), str(row.get("role")), str(row.get("layer")), str(row.get("latent_id"))))
    feature_counts = _count_by(rows, "model")
    for model, count in feature_counts.items():
        merged_summary["families"].setdefault(model, {})["n_features"] = count
    _write_jsonl(dash_dir / "feature_dashboards.jsonl", rows)
    _write_optional_parquet(dash_dir / "feature_dashboards.parquet", rows)
    _write_csv(dash_dir / "feature_observed_stats.csv", observed_rows)
    _write_json(dash_dir / "dashboard_summary.json", merged_summary)
    _write_dashboard_report(dash_dir / "feature_dashboard_report.md", rows, merged_summary)
    return merged_summary


def _add_top_examples_for_feature(
    *,
    tokenizer: Any,
    key: FeatureKey,
    meta_rows: list[dict[str, Any]],
    it_vals: torch.Tensor,
    pt_vals: torch.Tensor,
    heaps: dict[str, dict[str, dict[str, TopHeap]]],
    examples_per_kind: int,
    context_before: int,
    context_after: int,
) -> None:
    n = len(meta_rows)
    k = min(max(examples_per_kind, 1), n)
    if k <= 0:
        return
    candidates = {
        "top_it": it_vals,
        "top_pt": pt_vals,
        "contrast_it_over_pt": it_vals - pt_vals,
        "contrast_pt_over_it": pt_vals - it_vals,
    }
    for kind, values in candidates.items():
        top = torch.topk(values, k=k, largest=True)
        for score, rel_idx in zip(top.values.tolist(), top.indices.tolist(), strict=False):
            idx = int(rel_idx)
            meta = meta_rows[idx]
            split = str(meta["split"])
            example = _candidate_example(
                tokenizer=tokenizer,
                row=meta["row"],
                token_pos=int(meta["token_pos"]),
                token_id=int(meta["token_id"]),
                feature=key,
                kind=kind,
                score=float(score),
                activation_it=float(it_vals[idx].item()),
                activation_pt=float(pt_vals[idx].item()),
                context_before=context_before,
                context_after=context_after,
            )
            heaps[key.id][split][kind].add(float(score), example)
    inactive_score = -(it_vals.abs() + pt_vals.abs()) + torch.rand_like(it_vals) * 1e-6
    top_inactive = torch.topk(inactive_score, k=k, largest=True)
    for score, rel_idx in zip(top_inactive.values.tolist(), top_inactive.indices.tolist(), strict=False):
        idx = int(rel_idx)
        meta = meta_rows[idx]
        split = str(meta["split"])
        example = _candidate_example(
            tokenizer=tokenizer,
            row=meta["row"],
            token_pos=int(meta["token_pos"]),
            token_id=int(meta["token_id"]),
            feature=key,
            kind="inactive_random",
            score=float(score),
            activation_it=float(it_vals[idx].item()),
            activation_pt=float(pt_vals[idx].item()),
            context_before=context_before,
            context_after=context_after,
        )
        heaps[key.id][split]["inactive_random"].add(float(score), example)


def _near_misses(examples: dict[str, list[dict[str, Any]]], *, limit: int) -> list[dict[str, Any]]:
    active_tokens = {
        int(row["token_id"])
        for kind in ("top_it", "contrast_it_over_pt", "top_pt")
        for row in examples.get(kind, [])
    }
    misses = [
        {**row, "kind": "near_miss", "near_miss_basis": "same_token_low_activation"}
        for row in examples.get("inactive_random", [])
        if int(row.get("token_id", -1)) in active_tokens
    ]
    if len(misses) < limit:
        misses.extend(
            {**row, "kind": "near_miss", "near_miss_basis": "low_activation_control"}
            for row in examples.get("inactive_random", [])
        )
    return misses[:limit]


def _observed_stats(raw: dict[str, float]) -> dict[str, float]:
    n = max(float(raw.get("n_tokens", 0.0)), 1.0)
    mean_pt = float(raw.get("sum_activation_pt", 0.0)) / n
    mean_it = float(raw.get("sum_activation_it", 0.0)) / n
    freq_pt = float(raw.get("fire_pt", 0.0)) / n
    freq_it = float(raw.get("fire_it", 0.0)) / n
    freq_union = float(raw.get("fire_union", 0.0)) / n
    return {
        "dashboard_n_tokens": n,
        "dashboard_mean_activation_pt": mean_pt,
        "dashboard_mean_activation_it": mean_it,
        "dashboard_freq_pt": freq_pt,
        "dashboard_freq_it": freq_it,
        "dashboard_freq_union": freq_union,
        "dashboard_activation_ratio_it_pt": (mean_it + 1e-8) / (mean_pt + 1e-8),
        "dashboard_density_bin": _density_bin(freq_union),
    }


def _feature_causal_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "causal_score_mean": _float(row, "score_mean"),
        "causal_score_sum": _float(row, "score_sum"),
        "native_attr_mean": _float(row, "native_attr_mean"),
        "ptup_attr_mean": _float(row, "ptup_attr_mean"),
        "active_union_rate_calibration": _float(row, "active_union_rate"),
        "top25_interaction_drop_mean": row.get("causal_top25_interaction_drop_mean"),
        "top50_interaction_drop_mean": row.get("causal_top50_interaction_drop_mean"),
        "top200_interaction_drop_mean": row.get("causal_top200_interaction_drop_mean"),
        "matched_random200_interaction_drop_mean": row.get("causal_matched_random200_interaction_drop_mean"),
    }


def _write_dashboard_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = ["# Exp39 Feature Dashboard Report", ""]
    lines.append("## Summary")
    for model, payload in summary.get("families", {}).items():
        lines.append(
            f"- {model}: {payload.get('n_features')} features, "
            f"{payload.get('n_prompts_used')} prompts, {payload.get('n_tokens_seen')} tokens"
        )
    lines.append("")
    lines.append("## Feature Counts")
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('model')}:{row.get('role')}:{row.get('dashboard_density_bin')}"
        counts[key] = counts.get(key, 0) + 1
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Output projections are local decoder-to-unembedding proxies; exact final-norm effects are not claimed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dashboard_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _merge_scored_examples(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: _float(row, "score", float("-inf")), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        key = (str(row.get("prompt_id")), int(row.get("token_index", -1)), str(row.get("kind", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _merge_example_splits(rows: list[dict[str, Any]], split_key: str) -> dict[str, list[dict[str, Any]]]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    limits: dict[str, int] = defaultdict(int)
    for row in rows:
        examples = row.get(split_key) or {}
        if not isinstance(examples, dict):
            continue
        for kind, kind_rows in examples.items():
            if kind == "near_miss" or not isinstance(kind_rows, list):
                continue
            by_kind[str(kind)].extend(kind_rows)
            limits[str(kind)] = max(limits[str(kind)], len(kind_rows))
    out: dict[str, list[dict[str, Any]]] = {}
    for kind, kind_rows in by_kind.items():
        out[kind] = _merge_scored_examples(kind_rows, limit=max(limits[kind], 1))
    if limits:
        near_limit = max(limits.values())
        out["near_miss"] = _near_misses(out, limit=near_limit)
    return out


def _merge_dashboard_feature_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = dict(rows[0])
    total_tokens = sum(_float(row, "dashboard_n_tokens") for row in rows)
    if total_tokens > 0:
        weighted = {
            key: sum(_float(row, "dashboard_n_tokens") * _float(row, key) for row in rows) / total_tokens
            for key in (
                "dashboard_mean_activation_pt",
                "dashboard_mean_activation_it",
                "dashboard_freq_pt",
                "dashboard_freq_it",
                "dashboard_freq_union",
            )
        }
        base.update(weighted)
        base["dashboard_n_tokens"] = total_tokens
        base["dashboard_activation_ratio_it_pt"] = (
            weighted["dashboard_mean_activation_it"] + 1e-8
        ) / (weighted["dashboard_mean_activation_pt"] + 1e-8)
        base["dashboard_density_bin"] = _density_bin(weighted["dashboard_freq_union"])
    base["examples"] = _merge_example_splits(rows, "examples")
    base["validation_examples"] = _merge_example_splits(rows, "validation_examples")
    return base


def _merge_observed_feature_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = dict(rows[0])
    total_tokens = sum(_float(row, "dashboard_n_tokens") for row in rows)
    if total_tokens > 0:
        weighted = {
            key: sum(_float(row, "dashboard_n_tokens") * _float(row, key) for row in rows) / total_tokens
            for key in (
                "dashboard_mean_activation_pt",
                "dashboard_mean_activation_it",
                "dashboard_freq_pt",
                "dashboard_freq_it",
                "dashboard_freq_union",
            )
        }
        base.update(weighted)
        base["dashboard_n_tokens"] = total_tokens
        base["dashboard_activation_ratio_it_pt"] = (
            weighted["dashboard_mean_activation_it"] + 1e-8
        ) / (weighted["dashboard_mean_activation_pt"] + 1e-8)
        base["dashboard_density_bin"] = _density_bin(weighted["dashboard_freq_union"])
    return base


def _merge_family_summary(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    if not old:
        return dict(new)
    out = dict(old)
    out["n_prompts_used"] = int(old.get("n_prompts_used", 0) or 0) + int(new.get("n_prompts_used", 0) or 0)
    out["n_tokens_seen"] = int(old.get("n_tokens_seen", 0) or 0) + int(new.get("n_tokens_seen", 0) or 0)
    out["n_prompt_records_total_before_shard"] = max(
        int(old.get("n_prompt_records_total_before_shard", 0) or 0),
        int(new.get("n_prompt_records_total_before_shard", 0) or 0),
    )
    shard_indices = set(old.get("prompt_shard_indices", []))
    if "prompt_shard_index" in old:
        shard_indices.add(int(old.get("prompt_shard_index", 0) or 0))
    if "prompt_shard_index" in new:
        shard_indices.add(int(new.get("prompt_shard_index", 0) or 0))
    shard_indices.update(int(x) for x in new.get("prompt_shard_indices", []))
    out["prompt_shard_indices"] = sorted(shard_indices)
    out["prompt_shard_count"] = max(
        int(old.get("prompt_shard_count", 1) or 1),
        int(new.get("prompt_shard_count", 1) or 1),
    )
    return out


def _build_autointerp_packet(row: dict[str, Any]) -> str:
    examples = row.get("examples") or {}
    projection = row.get("output_projection") or {}
    payload = {
        "feature": {
            "feature_id": row.get("feature_id"),
            "model": row.get("model"),
            "layer": row.get("layer"),
            "latent_id": row.get("latent_id"),
        },
        "diagnostics": {
            "density_bin_calibration": row.get("density_bin"),
            "dashboard_density_bin": row.get("dashboard_density_bin"),
            "decoder_norm_ratio_split_a_split_b": row.get("decoder_norm_ratio_it_pt"),
            "decoder_cosine_split_a_split_b": row.get("decoder_cosine_pt_it"),
            "dashboard_activation_ratio_split_a_split_b": row.get("dashboard_activation_ratio_it_pt"),
        },
        "top_promoted_tokens": projection.get("top_promoted", [])[:12],
        "top_suppressed_tokens": projection.get("top_suppressed", [])[:12],
        "examples": {
            "split_a_high_activation": _compact_examples(examples.get("top_it", [])[:8]),
            "split_a_high_split_b_low_contrast": _compact_examples(examples.get("contrast_it_over_pt", [])[:8]),
            "split_b_high_activation": _compact_examples(examples.get("top_pt", [])[:5]),
            "near_miss": _compact_examples(examples.get("near_miss", [])[:5]),
            "low_activation_background": _compact_examples(examples.get("inactive_random", [])[:5]),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _compact_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "kind": row.get("kind"),
                "activation_it": row.get("activation_it"),
                "activation_pt": row.get("activation_pt"),
                "token_text": row.get("token_text"),
                "token_source": row.get("token_source"),
                "context_window": row.get("context_window"),
                "prompt_category": row.get("prompt_category"),
            }
        )
    return out


def _heuristic_label(row: dict[str, Any]) -> dict[str, Any]:
    projection = row.get("output_projection") or {}
    tokens = " ".join(str(tok.get("token_text", "")) for tok in projection.get("top_promoted", [])[:10])
    density = str(row.get("dashboard_density_bin") or row.get("density_bin"))
    if any(tok in tokens for tok in ["\\n", ":", "-", "*", "#"]):
        category = "formatting / structure"
        label = "formatting and response-structure token feature"
    elif density in {"dense", "very_dense"}:
        category = "generic frequency / unclear"
        label = "dense broad activation feature with unclear semantics"
    else:
        category = "generic frequency / unclear"
        label = "candidate terminal readout feature with unclear label"
    return {
        "feature_id": row.get("feature_id"),
        "label": label,
        "specific_label": label,
        "category": category,
        "coarse_category": category,
        "fires_on": "Unavailable; heuristic fallback only.",
        "mechanism_hypothesis": "Unavailable; heuristic fallback only.",
        "artifact_risk": "high",
        "evidence": ["Heuristic fallback only; no LLM API key was available."],
        "counterexamples_or_ambiguity": "Not paper-grade without LLM autointerp and held-out validation.",
        "confidence": 0.1,
        "pt_it_specificity": "unclear",
        "label_source": "heuristic_not_paper_grade",
        "role": row.get("role"),
        "control_kind": row.get("control_kind"),
        "model": row.get("model"),
        "layer": row.get("layer"),
        "latent_id": row.get("latent_id"),
    }


def _chat_json_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int = 5,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if not model.startswith("gpt-5.5"):
        kwargs["temperature"] = 0
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc)
            if "temperature" in kwargs and "Unsupported value: 'temperature'" in message:
                kwargs.pop("temperature", None)
                continue
            transient = any(
                marker in message.lower()
                for marker in (
                    "rate limit",
                    "server error",
                    "temporarily unavailable",
                    "timeout",
                    "connection",
                    "try again",
                )
            )
            if attempt + 1 >= max_retries or not transient:
                raise
            time.sleep(min(30.0, 2.0 * (2**attempt)) + random.random())
    raise RuntimeError("unreachable OpenAI retry state")


def autointerp(
    *,
    out_root: Path,
    run_name: str,
    model: str,
    max_features: int | None,
    include_controls: bool,
    parallelism: int,
) -> dict[str, Any]:
    run_dir = _run_dir(out_root, run_name)
    rows = _dashboard_rows(run_dir / "dashboards" / "feature_dashboards.jsonl")
    if not rows:
        raise FileNotFoundError("Dashboard rows are required before autointerp")
    if not include_controls:
        rows = [row for row in rows if row.get("role") != "control"]
    rows.sort(key=lambda row: (str(row.get("model")), str(row.get("role")), -_float(row, "score_mean")))
    if max_features is not None:
        rows = rows[:max_features]
    out_dir = run_dir / "autointerp"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("OPENAI_API_KEY"):
        labels = [_heuristic_label(row) for row in rows]
        _write_jsonl(out_dir / "llm_feature_labels.jsonl", labels)
        _write_csv(out_dir / "llm_feature_label_summary.csv", labels)
        marker = {
            "status": "skipped_llm_no_openai_api_key",
            "n_heuristic_labels": len(labels),
            "paper_grade": False,
        }
        _write_json(out_dir / "label_validation.json", marker)
        _write_json(out_dir / "skipped_no_openai_key.json", marker)
        return marker
    stale_skip_marker = out_dir / "skipped_no_openai_key.json"
    if stale_skip_marker.exists():
        stale_skip_marker.unlink()

    system = (
        "You label one latent from a paired-model crosscoder for mechanistic interpretability. "
        "You are blind to how this latent was selected. Give a concrete, falsifiable feature "
        "hypothesis when the evidence supports one, and explicitly assign probability mass to "
        "unknown/artifact explanations when it does not. Use the ontology only as a separate "
        "coarse_category field. Return only JSON."
    )

    def label_one(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        from openai import OpenAI

        client = OpenAI()
        packet = _build_autointerp_packet(row)
        user = (
            "Important blinding context: the packet intentionally does not tell you why this latent was selected. "
            "Do not infer that it is important from its presence here. Some latents have real, specific patterns; "
            "others are generic, polysemantic, or artifacts. If the evidence is weak, do not invent a meaning. "
            "Use a label like 'no coherent pattern: ...' or 'weak mixture: ...' with low confidence, and put "
            "substantial probability mass on unknown/artifact explanations.\n\n"
            "Calibration examples:\n"
            "1. If top examples repeatedly show a newline followed by '-' or '*' and promoted tokens include "
            "newline/list markers, use a label like 'bullet-list newline/item boundary', not 'formatting'.\n"
            "2. If examples center on ':' after fields such as Answer, Reason, or Name, use "
            "'colon after field label in response template', not 'punctuation'.\n"
            "3. If examples mix URLs, numbers, rare Unicode, and common continuation tokens with no stable "
            "semantic or structural pattern, use 'no coherent pattern: dense tokenization/artifact mixture' "
            "and paper_example_quality='not_showcase'.\n"
            "4. If examples are all multiple-choice labels like '(A)', 'B.', or option openings, use "
            "'multiple-choice option label boundary'.\n"
            "5. If the feature mainly fires on generic high-frequency words across unrelated contexts, use "
            "'no coherent pattern: common-token frequency feature'.\n\n"
            "Given this feature packet, produce JSON with these keys:\n"
            "- feature_id\n"
            "- specific_label: 5-14 words, concrete and testable. Bad: 'formatting feature'. "
            "Good: 'newline before bullet-list item', 'multiple-choice option label token', "
            "'colon after field name in Q/A template'.\n"
            "- label: same as specific_label unless you need a shorter alias.\n"
            f"- coarse_category: exactly one of {ONTOLOGY}\n"
            "- category: same as coarse_category for compatibility\n"
            "- fires_on: concise description of contexts/tokens that activate it\n"
            "- output_effect: what promoted/suppressed tokens suggest the feature pushes toward\n"
            "- mechanism_hypothesis: one sentence linking activation evidence and output evidence\n"
            "- label_distribution: 2-4 hypotheses with probability percentages summing to 100. Include "
            "'unknown/no coherent pattern' when plausible.\n"
            "- uncertainty_allocation: object with keys pattern_hypothesis, polysemantic_mixture, "
            "tokenization_or_frequency_artifact, insufficient_evidence; values are percentages summing to 100.\n"
            "- evidence: 3 short bullets from examples/projections\n"
            "- counterexamples_or_ambiguity: concrete failure modes, not generic hedging\n"
            "- artifact_risk: low, medium, or high\n"
            "- confidence: 0.0 to 1.0\n"
            "- split_specificity: one of split-A-weighted, split-B-weighted, shared-amplified, "
            "shared-suppressed, unclear\n"
            "- paper_example_quality: showcase, usable_with_caution, or not_showcase\n\n"
            "Be conservative, but do not collapse to only ontology names. If the feature is dense or polysemantic, "
            "name the most likely specific mixture and mark artifact_risk/high ambiguity. The first task is the "
            "best real label for this individual feature; grouping will happen later in a separate step.\n\n"
            f"{packet}"
        )
        try:
            response = _chat_json_completion(
                client,
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            parsed["label_source"] = f"openai:{model}"
        except Exception as exc:
            parsed = _heuristic_label(row)
            parsed["label_source"] = f"llm_failed_fallback:{type(exc).__name__}"
            parsed["error"] = str(exc)[:500]
        # Keep the canonical ID from the dashboard row. Models sometimes drop
        # the control suffix even when instructed to echo feature_id verbatim.
        parsed["feature_id"] = row.get("feature_id")
        parsed.setdefault("specific_label", parsed.get("label"))
        parsed.setdefault("label", parsed.get("specific_label"))
        if parsed.get("split_specificity") and not parsed.get("pt_it_specificity"):
            parsed["pt_it_specificity"] = parsed.get("split_specificity")
        parsed.setdefault(
            "uncertainty_allocation",
            {
                "pattern_hypothesis": int(round(100 * float(parsed.get("confidence", 0.0) or 0.0))),
                "polysemantic_mixture": 0,
                "tokenization_or_frequency_artifact": 0,
                "insufficient_evidence": int(round(100 * (1.0 - float(parsed.get("confidence", 0.0) or 0.0)))),
            },
        )
        parsed.setdefault(
            "label_distribution",
            [
                {
                    "label": parsed.get("specific_label") or parsed.get("label") or "unclear feature",
                    "probability": int(round(100 * float(parsed.get("confidence", 0.0) or 0.0))),
                    "reason": "Fallback distribution from confidence.",
                },
                {
                    "label": "unknown/no coherent pattern",
                    "probability": int(round(100 * (1.0 - float(parsed.get("confidence", 0.0) or 0.0)))),
                    "reason": "Fallback uncertainty mass.",
                },
            ],
        )
        if parsed.get("coarse_category") and not parsed.get("category"):
            parsed["category"] = parsed.get("coarse_category")
        if parsed.get("category") and not parsed.get("coarse_category"):
            parsed["coarse_category"] = parsed.get("category")
        parsed["labeler_blinded_to_selection_role"] = True
        parsed["role"] = row.get("role")
        parsed["control_kind"] = row.get("control_kind")
        parsed["model"] = row.get("model")
        parsed["layer"] = row.get("layer")
        parsed["latent_id"] = row.get("latent_id")
        return index, parsed

    labels_by_index: list[dict[str, Any] | None] = [None] * len(rows)
    max_workers = max(1, int(parallelism))
    if max_workers == 1:
        for index, row in enumerate(rows):
            done_index, parsed = label_one(index, row)
            labels_by_index[done_index] = parsed
            _write_jsonl(out_dir / "llm_feature_labels.jsonl", [label for label in labels_by_index if label is not None])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(label_one, index, row): index for index, row in enumerate(rows)}
            for future in as_completed(futures):
                done_index, parsed = future.result()
                labels_by_index[done_index] = parsed
                if sum(label is not None for label in labels_by_index) % 10 == 0:
                    _write_jsonl(
                        out_dir / "llm_feature_labels.jsonl",
                        [label for label in labels_by_index if label is not None],
                    )
    labels = [label for label in labels_by_index if label is not None]
    _write_jsonl(out_dir / "llm_feature_labels.jsonl", labels)
    _write_csv(out_dir / "llm_feature_label_summary.csv", labels)
    return {"status": "complete", "n_labels": len(labels), "model": model, "parallelism": max_workers}


def validate_labels(
    *,
    out_root: Path,
    run_name: str,
    model: str,
    max_features: int | None,
    parallelism: int,
) -> dict[str, Any]:
    run_dir = _run_dir(out_root, run_name)
    dash_rows = {row.get("feature_id"): row for row in _dashboard_rows(run_dir / "dashboards" / "feature_dashboards.jsonl")}
    labels = _dashboard_rows(run_dir / "autointerp" / "llm_feature_labels.jsonl")
    if max_features is not None:
        labels = labels[:max_features]
    if not labels:
        payload = {"status": "skipped_no_labels", "paper_grade": False}
        _write_json(run_dir / "autointerp" / "label_validation.json", payload)
        return payload
    if not os.environ.get("OPENAI_API_KEY"):
        payload = {
            "status": "skipped_no_openai_api_key",
            "paper_grade": False,
            "n_labels": len(labels),
        }
        _write_json(run_dir / "autointerp" / "label_validation.json", payload)
        return payload

    def validate_one(label: dict[str, Any]) -> dict[str, Any] | None:
        from openai import OpenAI

        client = OpenAI()
        feature_id = label.get("feature_id")
        row = dash_rows.get(feature_id)
        if not row:
            return {
                "feature_id": feature_id,
                "role": label.get("role"),
                "control_kind": label.get("control_kind"),
                "category": label.get("category"),
                "coarse_category": label.get("coarse_category") or label.get("category"),
                "specific_label": label.get("specific_label") or label.get("label"),
                "status": "skipped_missing_dashboard",
            }
        val = row.get("validation_examples") or {}
        examples = []
        for high_kind in ("top_it", "contrast_it_over_pt", "top_pt"):
            for ex in val.get(high_kind, [])[:4]:
                examples.append((1, ex))
        for low_kind in ("near_miss", "inactive_random"):
            for ex in val.get(low_kind, [])[:6]:
                examples.append((0, ex))
        if len(examples) < 4:
            return {
                "feature_id": feature_id,
                "role": label.get("role"),
                "control_kind": label.get("control_kind"),
                "category": label.get("category"),
                "coarse_category": label.get("coarse_category") or label.get("category"),
                "specific_label": label.get("specific_label") or label.get("label"),
                "n_examples": len(examples),
                "status": "skipped_insufficient_validation_examples",
            }
        prompt_examples = [
            {
                "index": idx,
                "context_window": ex.get("context_window"),
                "token_text": ex.get("token_text"),
                "token_source": ex.get("token_source"),
            }
            for idx, (_truth, ex) in enumerate(examples)
        ]
        user = json.dumps(
            {
                "task": "Predict whether each context is high activation for the supplied feature label.",
                "feature_label": label.get("specific_label") or label.get("label"),
                "coarse_category": label.get("coarse_category") or label.get("category"),
                "fires_on": label.get("fires_on"),
                "output_effect": label.get("output_effect"),
                "mechanism_hypothesis": label.get("mechanism_hypothesis"),
                "counterexamples_or_ambiguity": label.get("counterexamples_or_ambiguity"),
                "examples": prompt_examples,
                "return_schema": {"predictions": [{"index": 0, "high_activation_probability": 0.0}]},
            },
            ensure_ascii=False,
        )
        try:
            response = _chat_json_completion(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": "Return only JSON. Calibrate probabilities."},
                    {"role": "user", "content": user},
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            preds = {int(p["index"]): float(p["high_activation_probability"]) for p in parsed.get("predictions", [])}
        except Exception as exc:
            message = str(exc)
            if "cybersecurity risk" not in message.lower() and "content" not in message.lower():
                return {"feature_id": feature_id, "status": "failed", "error": message[:300]}
            sanitized_examples = [
                {
                    "index": idx,
                    "context_window": "[withheld after automated safety filter]",
                    "token_text": ex.get("token_text"),
                    "token_source": ex.get("token_source"),
                    "prompt_category": ex.get("prompt_category"),
                }
                for idx, (_truth, ex) in enumerate(examples)
            ]
            sanitized_user = json.dumps(
                {
                    "task": (
                        "Predict whether each context is high activation for the supplied feature label. "
                        "The full context was withheld after an automated safety filter; use token text, "
                        "token source, prompt category, and the feature label only. Calibrate uncertainty."
                    ),
                    "feature_label": label.get("specific_label") or label.get("label"),
                    "coarse_category": label.get("coarse_category") or label.get("category"),
                    "fires_on": label.get("fires_on"),
                    "output_effect": label.get("output_effect"),
                    "mechanism_hypothesis": label.get("mechanism_hypothesis"),
                    "counterexamples_or_ambiguity": label.get("counterexamples_or_ambiguity"),
                    "examples": sanitized_examples,
                    "return_schema": {"predictions": [{"index": 0, "high_activation_probability": 0.0}]},
                },
                ensure_ascii=False,
            )
            try:
                response = _chat_json_completion(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": "Return only JSON. Calibrate probabilities."},
                        {"role": "user", "content": sanitized_user},
                    ],
                )
                parsed = json.loads(response.choices[0].message.content or "{}")
                preds = {
                    int(p["index"]): float(p["high_activation_probability"])
                    for p in parsed.get("predictions", [])
                }
                sanitized_retry = True
            except Exception as retry_exc:
                return {
                    "feature_id": feature_id,
                    "status": "failed",
                    "error": message[:240],
                    "sanitized_retry_error": str(retry_exc)[:240],
                }
        else:
            sanitized_retry = False
        y_true = [truth for truth, _ex in examples]
        y_score = [preds.get(idx, 0.5) for idx in range(len(examples))]
        acc = sum((score >= 0.5) == bool(truth) for truth, score in zip(y_true, y_score, strict=False)) / len(y_true)
        auroc = _auroc(y_true, y_score)
        return {
            "feature_id": feature_id,
            "role": label.get("role"),
            "control_kind": label.get("control_kind"),
            "category": label.get("category"),
            "coarse_category": label.get("coarse_category") or label.get("category"),
            "specific_label": label.get("specific_label") or label.get("label"),
            "accuracy": acc,
            "auroc": auroc,
            "n_examples": len(y_true),
            "status": "complete",
            "sanitized_retry": sanitized_retry,
        }

    results_by_index: list[dict[str, Any] | None] = [None] * len(labels)
    max_workers = max(1, int(parallelism))
    if max_workers == 1:
        for index, label in enumerate(labels):
            results_by_index[index] = validate_one(label)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(validate_one, label): index for index, label in enumerate(labels)}
            for future in as_completed(futures):
                results_by_index[futures[future]] = future.result()
    results = [row for row in results_by_index if row is not None]
    baseline = _validation_baseline(results)
    payload = {
        "status": "complete",
        "model": model,
        "parallelism": max_workers,
        "n_features": len(results),
        "mean_accuracy": _mean([row.get("accuracy") for row in results]),
        "mean_auroc": _mean([row.get("auroc") for row in results]),
        "by_role": _aggregate_validation_by_role(results),
        "by_control_kind": _aggregate_validation_by_control_kind(results),
        "baseline": baseline,
        "feature_results": results,
    }
    _write_json(run_dir / "autointerp" / "label_validation.json", payload)
    return payload


def group_labels(
    *,
    out_root: Path,
    run_name: str,
    model: str,
) -> dict[str, Any]:
    run_dir = _run_dir(out_root, run_name)
    labels_path = run_dir / "autointerp" / "llm_feature_labels.jsonl"
    labels = _dashboard_rows(labels_path)
    dashboards = {row.get("feature_id"): row for row in _dashboard_rows(run_dir / "dashboards" / "feature_dashboards.jsonl")}
    out_dir = run_dir / "autointerp"
    if not labels:
        payload = {"status": "skipped_no_labels"}
        _write_json(out_dir / "label_groups.json", payload)
        return payload
    if not os.environ.get("OPENAI_API_KEY"):
        payload = {"status": "skipped_no_openai_api_key", "n_labels": len(labels)}
        _write_json(out_dir / "label_groups.json", payload)
        return payload

    compact = []
    for label in labels:
        dash = dashboards.get(label.get("feature_id"), {})
        compact.append(
            {
                "feature_id": label.get("feature_id"),
                "role": label.get("role"),
                "control_kind": label.get("control_kind", ""),
                "model": label.get("model"),
                "score_mean": dash.get("score_mean"),
                "density": dash.get("dashboard_density_bin") or dash.get("density_bin"),
                "specific_label": label.get("specific_label") or label.get("label"),
                "coarse_category": label.get("coarse_category") or label.get("category"),
                "fires_on": label.get("fires_on"),
                "output_effect": label.get("output_effect"),
                "artifact_risk": label.get("artifact_risk"),
                "confidence": label.get("confidence"),
                "paper_example_quality": label.get("paper_example_quality"),
                "ambiguity": label.get("counterexamples_or_ambiguity"),
            }
        )

    from openai import OpenAI

    client = OpenAI()
    system = (
        "You cluster fine-grained mechanistic feature labels. Build niche subgroups; do not collapse "
        "everything into broad categories like formatting or punctuation. Return only JSON."
    )
    user = json.dumps(
        {
            "task": (
                "Create fine-grained subgroups for these crosscoder feature labels. "
                "Use the specific labels and evidence fields, not only coarse_category. "
                "Controls may be incoherent; put weak random/control features into explicit "
                "'no coherent pattern' or 'generic frequency artifact' groups when warranted. "
                "Prefer 10-25 subgroups. Each subgroup should be narrow enough to be paper-useful."
            ),
            "group_schema": {
                "group_id": "short stable id like newline_list_boundary",
                "group_name": "human-readable niche subgroup",
                "parent_category": "coarse category",
                "description": "one sentence",
                "paper_use": "showcase | support | diagnostic_only | reject",
            },
            "assignment_schema": {
                "feature_id": "feature id",
                "group_id": "one group_id",
                "assignment_confidence": 0.0,
                "reason": "short reason",
            },
            "records": compact,
            "return_schema": {"groups": [], "assignments": []},
        },
        ensure_ascii=False,
    )
    try:
        response = _chat_json_completion(
            client,
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        payload["status"] = "complete"
        payload["model"] = model
    except Exception as exc:
        payload = {"status": "failed", "model": model, "error": str(exc)[:1000], "n_labels": len(labels)}
        _write_json(out_dir / "label_groups.json", payload)
        return payload

    assignments = {
        str(row.get("feature_id")): row
        for row in payload.get("assignments", [])
        if row.get("feature_id") is not None
    }
    groups = {str(row.get("group_id")): row for row in payload.get("groups", []) if row.get("group_id") is not None}
    grouped_labels = []
    for label in labels:
        assignment = assignments.get(str(label.get("feature_id")), {})
        group = groups.get(str(assignment.get("group_id")), {})
        updated = dict(label)
        updated["induced_group_id"] = assignment.get("group_id", "")
        updated["induced_group_name"] = group.get("group_name", "")
        updated["induced_group_parent_category"] = group.get("parent_category", updated.get("category", ""))
        updated["induced_group_paper_use"] = group.get("paper_use", "")
        updated["induced_group_assignment_confidence"] = assignment.get("assignment_confidence", "")
        updated["induced_group_assignment_reason"] = assignment.get("reason", "")
        grouped_labels.append(updated)
    _write_json(out_dir / "label_groups.json", payload)
    _write_jsonl(out_dir / "llm_feature_labels_grouped.jsonl", grouped_labels)
    _write_csv(out_dir / "llm_feature_label_grouped_summary.csv", grouped_labels)
    return {
        "status": payload.get("status"),
        "model": model,
        "n_labels": len(labels),
        "n_groups": len(groups),
        "n_assignments": len(assignments),
    }


def _auroc(y_true: list[int], y_score: list[float]) -> float | None:
    pos = [(s, y) for s, y in zip(y_score, y_true, strict=False) if y == 1]
    neg = [(s, y) for s, y in zip(y_score, y_true, strict=False) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for ps, _ in pos:
        for ns, _ in neg:
            total += 1.0
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total if total else None


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(v) for v in values if v not in (None, "")]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _aggregate_validation_by_role(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_role[str(row.get("role", ""))].append(row)
    return {
        role: {
            "n": len(rows),
            "n_scored": sum(row.get("accuracy") not in (None, "") for row in rows),
            "mean_accuracy": _mean(row.get("accuracy") for row in rows),
            "mean_auroc": _mean(row.get("auroc") for row in rows),
        }
        for role, rows in by_role.items()
    }


def _aggregate_validation_by_control_kind(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row.get("role") != "control":
            continue
        by_kind[str(row.get("control_kind", ""))].append(row)
    return {
        kind: {
            "n": len(rows),
            "n_scored": sum(row.get("accuracy") not in (None, "") for row in rows),
            "mean_accuracy": _mean(row.get("accuracy") for row in rows),
            "mean_auroc": _mean(row.get("auroc") for row in rows),
        }
        for kind, rows in by_kind.items()
    }


def _validation_baseline(results: list[dict[str, Any]]) -> dict[str, Any]:
    # The LLM validation prompt compares labels to context. With no second LLM pass,
    # the defensible baseline is chance plus control-feature performance.
    controls = [row for row in results if row.get("role") == "control"]
    causal = [row for row in results if row.get("role") == "causal"]
    return {
        "chance_accuracy": 0.5,
        "control_mean_accuracy": _mean(row.get("accuracy") for row in controls),
        "control_mean_auroc": _mean(row.get("auroc") for row in controls),
        "causal_minus_control_accuracy": (
            (_mean(row.get("accuracy") for row in causal) or 0.0)
            - (_mean(row.get("accuracy") for row in controls) or 0.0)
            if controls and causal
            else None
        ),
    }


def analyze(
    *,
    out_root: Path,
    run_name: str,
) -> dict[str, Any]:
    run_dir = _run_dir(out_root, run_name)
    grouped_label_path = run_dir / "autointerp" / "llm_feature_labels_grouped.jsonl"
    labels = _dashboard_rows(grouped_label_path if grouped_label_path.exists() else run_dir / "autointerp" / "llm_feature_labels.jsonl")
    dashboards = _dashboard_rows(run_dir / "dashboards" / "feature_dashboards.jsonl")
    validation = _read_json(run_dir / "autointerp" / "label_validation.json", {}) or {}
    dash_by_id = {row.get("feature_id"): row for row in dashboards}
    rows = []
    for label in labels:
        dash = dash_by_id.get(label.get("feature_id"), {})
        rows.append(
            {
                "feature_id": label.get("feature_id"),
                "model": label.get("model"),
                "layer": label.get("layer"),
                "latent_id": label.get("latent_id"),
                "role": label.get("role"),
                "control_kind": label.get("control_kind", ""),
                "category": label.get("category"),
                "coarse_category": label.get("coarse_category") or label.get("category"),
                "label": label.get("label"),
                "specific_label": label.get("specific_label") or label.get("label"),
                "confidence": label.get("confidence"),
                "label_source": label.get("label_source"),
                "artifact_risk": label.get("artifact_risk"),
                "paper_example_quality": label.get("paper_example_quality"),
                "induced_group_id": label.get("induced_group_id", ""),
                "induced_group_name": label.get("induced_group_name", ""),
                "induced_group_parent_category": label.get("induced_group_parent_category", ""),
                "induced_group_paper_use": label.get("induced_group_paper_use", ""),
                "score_mean": dash.get("score_mean"),
                "density_bin": dash.get("density_bin"),
                "dashboard_density_bin": dash.get("dashboard_density_bin"),
                "decoder_norm_ratio_it_pt": dash.get("decoder_norm_ratio_it_pt"),
                "dashboard_activation_ratio_it_pt": dash.get("dashboard_activation_ratio_it_pt"),
            }
        )
    category_rows = _category_table(rows)
    group_rows = _group_table(rows)
    recommendation = _paper_recommendation(rows, validation)
    n_llm_labeled = sum(1 for row in rows if "heuristic_not_paper_grade" not in str(row.get("label_source")))
    n_paper_grade = n_llm_labeled if recommendation.startswith("candidate_main_text") else 0
    summary = {
        "run_name": run_name,
        "n_dashboard_features": len(dashboards),
        "n_labels": len(labels),
        "n_llm_labeled_features": n_llm_labeled,
        "n_paper_grade_labels": n_paper_grade,
        "paper_grade_rule": (
            "LLM-labeled features are counted as paper-grade only when held-out label validation "
            "beats control-feature validation by the predeclared margin."
        ),
        "validation": validation,
        "category_counts": _count_by(rows, "category"),
        "category_score_mass": {
            row["category"]: row["causal_score_mean_sum"]
            for row in category_rows
        },
        "group_counts": _count_by(rows, "induced_group_name"),
        "group_score_mass": {
            row["group_name"]: row["causal_score_mean_sum"]
            for row in group_rows
            if row["group_name"]
        },
        "control_kind_counts": _count_by(rows, "control_kind"),
        "paper_integration_recommendation": recommendation,
    }
    analysis_dir = run_dir / "analysis"
    _write_csv(analysis_dir / "feature_category_table.csv", category_rows)
    _write_csv(analysis_dir / "feature_group_table.csv", group_rows)
    _write_json(analysis_dir / "model_diff_feature_summary.json", summary)
    _write_paper_note(analysis_dir / "exp39_paper_note.md", summary, category_rows, group_rows)
    return summary


def _category_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category", "unlabeled"))].append(row)
    out = []
    for category, group in sorted(grouped.items()):
        causal = [row for row in group if row.get("role") == "causal"]
        controls = [row for row in group if row.get("role") == "control"]
        out.append(
            {
                "category": category,
                "n_total": len(group),
                "n_causal": len(causal),
                "n_control": len(controls),
                "n_control_matched": sum(1 for row in controls if row.get("control_kind") == "matched_noncausal"),
                "n_control_random_active": sum(
                    1 for row in controls if row.get("control_kind") == "random_active_noncausal"
                ),
                "causal_score_mean_sum": sum(_float(row, "score_mean") for row in causal),
                "mean_confidence": _mean(row.get("confidence") for row in group),
                "models": ",".join(sorted({str(row.get("model")) for row in group})),
                "density_bins": json.dumps(_count_by(group, "dashboard_density_bin")),
            }
        )
    return out


def _group_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_name = str(row.get("induced_group_name") or "")
        if not group_name:
            continue
        grouped[group_name].append(row)
    out = []
    for group_name, group in sorted(grouped.items()):
        causal = [row for row in group if row.get("role") == "causal"]
        controls = [row for row in group if row.get("role") == "control"]
        out.append(
            {
                "group_name": group_name,
                "group_id": next((str(row.get("induced_group_id")) for row in group if row.get("induced_group_id")), ""),
                "parent_category": next(
                    (str(row.get("induced_group_parent_category")) for row in group if row.get("induced_group_parent_category")),
                    "",
                ),
                "paper_use": next(
                    (str(row.get("induced_group_paper_use")) for row in group if row.get("induced_group_paper_use")),
                    "",
                ),
                "n_total": len(group),
                "n_causal": len(causal),
                "n_control": len(controls),
                "n_control_matched": sum(1 for row in controls if row.get("control_kind") == "matched_noncausal"),
                "n_control_random_active": sum(
                    1 for row in controls if row.get("control_kind") == "random_active_noncausal"
                ),
                "causal_score_mean_sum": sum(_float(row, "score_mean") for row in causal),
                "mean_confidence": _mean(row.get("confidence") for row in group),
                "models": ",".join(sorted({str(row.get("model")) for row in group})),
                "example_labels": " | ".join(
                    str(row.get("specific_label") or row.get("label"))
                    for row in sorted(group, key=lambda r: -_float(r, "score_mean"))[:4]
                ),
            }
        )
    out.sort(key=lambda row: (-int(row["n_causal"]), -float(row["causal_score_mean_sum"])))
    return out


def _paper_recommendation(rows: list[dict[str, Any]], validation: dict[str, Any]) -> str:
    if not rows:
        return "not_ready_no_labels"
    if any("heuristic_not_paper_grade" in str(row.get("label_source")) for row in rows):
        return "not_ready_llm_autointerp_missing"
    mean_auroc = validation.get("mean_auroc")
    control = (validation.get("baseline") or {}).get("control_mean_auroc")
    if mean_auroc is not None and control is not None and float(mean_auroc) > max(0.6, float(control) + 0.05):
        return "candidate_main_text_if_categories_are_manually_confirmed"
    return "appendix_or_future_work_until_validation_strengthens"


def _write_paper_note(
    path: Path,
    summary: dict[str, Any],
    category_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]] | None = None,
) -> None:
    lines = ["# Exp39 Paper Note", ""]
    lines.append(f"Recommendation: `{summary.get('paper_integration_recommendation')}`")
    lines.append("")
    lines.append(f"Dashboard features: {summary.get('n_dashboard_features')}")
    lines.append(
        f"Labels: {summary.get('n_labels')} "
        f"(LLM-labeled: {summary.get('n_llm_labeled_features')}, "
        f"paper-grade under control-beating rule: {summary.get('n_paper_grade_labels')})"
    )
    lines.append("")
    lines.append("## Categories")
    for row in category_rows:
        lines.append(
            f"- {row['category']}: n_causal={row['n_causal']}, "
            f"score_mass={float(row['causal_score_mean_sum']):.4f}, models={row['models']}"
        )
    if group_rows:
        lines.append("")
        lines.append("## Fine-Grained Groups")
        for row in group_rows[:15]:
            lines.append(
                f"- {row['group_name']}: n_causal={row['n_causal']}, "
                f"matched_controls={row['n_control_matched']}, random_active_controls={row['n_control_random_active']}, "
                f"paper_use={row['paper_use']}"
            )
    lines.append("")
    lines.append("Use cautious wording: these are causally selected IT-weighted terminal features, not canonical atoms.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_preflight() -> None:
    parser = argparse.ArgumentParser(description="Audit Exp39 artifacts")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    payload = preflight(
        out_root=args.out_root,
        run_name=args.run_name,
        families=_families_from_args(args.families),
        strict=args.strict,
    )
    print(json.dumps(payload["families"], indent=2))


def main_select() -> None:
    parser = argparse.ArgumentParser(description="Select causal Exp39 features and controls")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--control-per-feature", type=int, default=1)
    parser.add_argument("--random-control-per-feature", type=int, default=1)
    parser.add_argument("--seed", type=int, default=39)
    args = parser.parse_args()
    payload = select_features(
        out_root=args.out_root,
        run_name=args.run_name,
        families=_families_from_args(args.families),
        top_n=args.top_n,
        control_per_feature=args.control_per_feature,
        random_control_per_feature=args.random_control_per_feature,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, default=_json_default))


def main_dashboard() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp39 feature dashboards")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--dataset", nargs="+", type=Path, default=list(DEFAULT_DASHBOARD_DATASETS))
    parser.add_argument("--exclude-dataset", nargs="*", type=Path, default=list(DEFAULT_EXCLUDE_DATASETS))
    parser.add_argument("--n-prompts", type=int, default=3000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--append-pt-greedy-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--examples-per-kind", type=int, default=20)
    parser.add_argument("--validation-examples-per-kind", type=int, default=12)
    parser.add_argument("--context-before", type=int, default=64)
    parser.add_argument("--context-after", type=int, default=16)
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--use-threshold", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crosscoder-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--projection-top-k", type=int, default=30)
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument("--prompt-shard-count", type=int, default=1)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    payload = collect_dashboards(
        out_root=args.out_root,
        run_name=args.run_name,
        families=_families_from_args(args.families),
        datasets=args.dataset,
        exclude_datasets=args.exclude_dataset,
        n_prompts=args.n_prompts,
        max_seq_len=args.max_seq_len,
        append_pt_greedy_tokens=args.append_pt_greedy_tokens,
        batch_size=args.batch_size,
        examples_per_kind=args.examples_per_kind,
        validation_examples_per_kind=args.validation_examples_per_kind,
        context_before=args.context_before,
        context_after=args.context_after,
        include_controls=not args.no_controls,
        use_threshold=args.use_threshold,
        device_name=args.device,
        crosscoder_dtype_name=args.crosscoder_dtype,
        projection_top_k=args.projection_top_k,
        prompt_shard_index=args.prompt_shard_index,
        prompt_shard_count=args.prompt_shard_count,
        output_suffix=args.output_suffix,
    )
    print(json.dumps(payload, indent=2, default=_json_default))


def main_merge_dashboards() -> None:
    parser = argparse.ArgumentParser(description="Merge per-family Exp39 dashboards")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--suffixes", nargs="+", required=True)
    args = parser.parse_args()
    payload = merge_dashboards(out_root=args.out_root, run_name=args.run_name, suffixes=args.suffixes)
    print(json.dumps(payload, indent=2, default=_json_default))


def main_autointerp() -> None:
    parser = argparse.ArgumentParser(description="Run LLM autointerp on Exp39 dashboards")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--parallelism", type=int, default=int(os.environ.get("OPENAI_PARALLELISM", "1")))
    args = parser.parse_args()
    payload = autointerp(
        out_root=args.out_root,
        run_name=args.run_name,
        model=args.model,
        max_features=args.max_features,
        include_controls=args.include_controls,
        parallelism=args.parallelism,
    )
    print(json.dumps(payload, indent=2, default=_json_default))


def main_group_labels() -> None:
    parser = argparse.ArgumentParser(description="Cluster Exp39 fine-grained labels into subgroups")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    args = parser.parse_args()
    payload = group_labels(out_root=args.out_root, run_name=args.run_name, model=args.model)
    print(json.dumps(payload, indent=2, default=_json_default))


def main_validate() -> None:
    parser = argparse.ArgumentParser(description="Validate LLM labels on held-out dashboard contexts")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--parallelism", type=int, default=int(os.environ.get("OPENAI_PARALLELISM", "1")))
    args = parser.parse_args()
    payload = validate_labels(
        out_root=args.out_root,
        run_name=args.run_name,
        model=args.model,
        max_features=args.max_features,
        parallelism=args.parallelism,
    )
    print(json.dumps(payload, indent=2, default=_json_default))


def main_analyze() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp39 labels and dashboards")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    payload = analyze(out_root=args.out_root, run_name=args.run_name)
    print(json.dumps(payload, indent=2, default=_json_default))


def main_run_all() -> None:
    parser = argparse.ArgumentParser(description="Run non-parallel Exp39 pipeline")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--control-per-feature", type=int, default=1)
    parser.add_argument("--random-control-per-feature", type=int, default=1)
    parser.add_argument("--n-prompts", type=int, default=3000)
    args, rest = parser.parse_known_args()
    families = _families_from_args(args.families)
    preflight(out_root=args.out_root, run_name=args.run_name, families=families, strict=True)
    select_features(
        out_root=args.out_root,
        run_name=args.run_name,
        families=families,
        top_n=args.top_n,
        control_per_feature=args.control_per_feature,
        random_control_per_feature=args.random_control_per_feature,
        seed=39,
    )
    dashboard_args = argparse.Namespace(
        out_root=args.out_root,
        run_name=args.run_name,
        families=args.families,
        dataset=list(DEFAULT_DASHBOARD_DATASETS),
        exclude_dataset=list(DEFAULT_EXCLUDE_DATASETS),
        n_prompts=args.n_prompts,
        max_seq_len=512,
        append_pt_greedy_tokens=384,
        batch_size=4,
        examples_per_kind=20,
        validation_examples_per_kind=12,
        context_before=64,
        context_after=16,
        no_controls=False,
        use_threshold=False,
        device="cuda:0",
        crosscoder_dtype="bfloat16",
        projection_top_k=30,
    )
    collect_dashboards(
        out_root=dashboard_args.out_root,
        run_name=dashboard_args.run_name,
        families=families,
        datasets=dashboard_args.dataset,
        exclude_datasets=dashboard_args.exclude_dataset,
        n_prompts=dashboard_args.n_prompts,
        max_seq_len=dashboard_args.max_seq_len,
        append_pt_greedy_tokens=dashboard_args.append_pt_greedy_tokens,
        batch_size=dashboard_args.batch_size,
        examples_per_kind=dashboard_args.examples_per_kind,
        validation_examples_per_kind=dashboard_args.validation_examples_per_kind,
        context_before=dashboard_args.context_before,
        context_after=dashboard_args.context_after,
        include_controls=not dashboard_args.no_controls,
        use_threshold=dashboard_args.use_threshold,
        device_name=dashboard_args.device,
        crosscoder_dtype_name=dashboard_args.crosscoder_dtype,
        projection_top_k=dashboard_args.projection_top_k,
    )
    parallelism = int(os.environ.get("OPENAI_PARALLELISM", "1"))
    autointerp(
        out_root=args.out_root,
        run_name=args.run_name,
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        max_features=None,
        include_controls=True,
        parallelism=parallelism,
    )
    group_labels(
        out_root=args.out_root,
        run_name=args.run_name,
        model=os.environ.get("OPENAI_MODEL", "gpt-5.5"),
    )
    validate_labels(
        out_root=args.out_root,
        run_name=args.run_name,
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        max_features=None,
        parallelism=parallelism,
    )
    analyze(out_root=args.out_root, run_name=args.run_name)
