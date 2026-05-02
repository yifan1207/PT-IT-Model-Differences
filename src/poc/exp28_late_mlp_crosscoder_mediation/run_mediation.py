"""Run Exp28 crosscoder feature-ablation mediation on Exp23 cells."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.boundary import (
    BoundaryStateCapture,
    BoundaryStatePatch,
    LayerResidualCapture,
)
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    DEFAULT_EVENT_KINDS,
    DEFAULT_EXP20_FALLBACK_ROOT,
    DEFAULT_EXP20_ROOT,
    _dataset_lookup,
    _late_boundary,
    _load_manifest_records,
    _logits_from_hidden,
    _make_readouts,
    _prefix_ids_for_event,
    _unique_events,
    _validate_tokenizers_and_tokens,
)
from src.poc.exp28_late_mlp_crosscoder_mediation.mediation_hooks import (
    CrosscoderMlpModifier,
    FeatureSelection,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class LatentRow:
    layer: int
    latent_id: int
    feature_type: str
    interaction_score: float
    mean_activation_it: float
    mean_activation_pt: float
    decoder_norm_it: float
    decoder_norm_pt: float
    latent_scaling_ratio: float
    local_margin_attr: float

    @property
    def match_value(self) -> float:
        return self.mean_activation_it * self.decoder_norm_it


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_latent_rows(path: Path) -> list[LatentRow]:
    rows: list[LatentRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                LatentRow(
                    layer=int(row["layer"]),
                    latent_id=int(row["latent_id"]),
                    feature_type=str(row["feature_type"]),
                    interaction_score=float(row["interaction_score"]),
                    mean_activation_it=float(row["mean_activation_it"]),
                    mean_activation_pt=float(row["mean_activation_pt"]),
                    decoder_norm_it=float(row["decoder_norm_it"]),
                    decoder_norm_pt=float(row["decoder_norm_pt"]),
                    latent_scaling_ratio=float(row["latent_scaling_ratio"]),
                    local_margin_attr=float(row["local_margin_attr"]),
                )
            )
    if not rows:
        raise RuntimeError(f"No latent rows loaded from {path}")
    return rows


def _load_manifest_records_window(
    *,
    exp20_root: Path,
    exp20_fallback_root: Path | None,
    prompt_mode: str,
    model: str,
    n_examples: int | None,
    skip_examples: int,
    worker_index: int,
    n_workers: int,
) -> list[dict[str, Any]]:
    """Load a contiguous manifest window, then shard it across workers.

    Exp28 causal feature ranking uses a calibration/heldout split over the
    same ordered holdout prompt file. The upstream Exp23 helper shards before
    slicing, so this wrapper first loads the requested global window and only
    then applies worker sharding. That makes commands like
    ``--skip-prompts 128 --n-prompts 472`` mean exactly prompts 128:600.
    """

    total = None if n_examples is None else int(skip_examples) + int(n_examples)
    rows = _load_manifest_records(
        exp20_root=exp20_root,
        exp20_fallback_root=exp20_fallback_root,
        prompt_mode=prompt_mode,
        model=model,
        n_examples=total,
        worker_index=0,
        n_workers=1,
    )
    if skip_examples:
        rows = rows[int(skip_examples):]
    if n_examples is not None:
        rows = rows[: int(n_examples)]
    sliced = rows[worker_index::n_workers]
    log.info(
        "[exp28-mediate] manifest window skip=%d n=%s -> %d rows for worker %d/%d",
        skip_examples,
        n_examples,
        len(sliced),
        worker_index,
        n_workers,
    )
    return sliced


def _load_causal_rank_rows(path: Path) -> tuple[list[LatentRow], list[LatentRow]]:
    """Load causal-rank CSV as ranked rows plus a calibration-matched pool.

    The causal-rank phase writes every feature active on the calibration slice,
    not just the positive hits. We encode active-rate and absolute attribution
    into LatentRow's existing matching fields, so causal matched-random controls
    are same-layer and similar in calibration activity/attribution mass even
    when the original all_latents.csv cache has been cleaned from a remote.
    """

    if not path.exists():
        raise FileNotFoundError(f"Causal feature CSV not found: {path}")
    rows: list[LatentRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            score_mean = float(row["score_mean"])
            active_rate = float(row.get("active_union_rate", 0.0))
            score_abs = float(row.get("score_abs_mean", abs(score_mean)))
            rows.append(
                LatentRow(
                    layer=int(row["layer"]),
                    latent_id=int(row["latent_id"]),
                    feature_type="causal_ranked",
                    interaction_score=score_mean,
                    mean_activation_it=active_rate,
                    mean_activation_pt=0.0,
                    decoder_norm_it=max(score_abs, 1e-12),
                    decoder_norm_pt=0.0,
                    latent_scaling_ratio=1.0,
                    local_margin_attr=score_mean,
                )
            )
    if not rows:
        raise RuntimeError(f"No causal-ranked features loaded from {path}")
    rows.sort(key=lambda r: r.interaction_score, reverse=True)
    positive = [row for row in rows if row.interaction_score > 0.0]
    return (positive or rows), rows


def _top_interaction(rows: list[LatentRow], k: int) -> list[LatentRow]:
    candidates = [r for r in rows if r.feature_type in {"interaction_candidate", "it_biased"}]
    if len(candidates) < k:
        candidates = rows
    return sorted(candidates, key=lambda r: r.interaction_score, reverse=True)[:k]


def _shared(rows: list[LatentRow], k: int) -> list[LatentRow]:
    candidates = [r for r in rows if r.feature_type == "shared"]
    return sorted(candidates, key=lambda r: r.mean_activation_it + r.mean_activation_pt, reverse=True)[:k]


def _pt_biased(rows: list[LatentRow], k: int) -> list[LatentRow]:
    candidates = [r for r in rows if r.feature_type == "pt_biased"]
    return sorted(candidates, key=lambda r: r.mean_activation_pt * r.decoder_norm_pt, reverse=True)[:k]


def _matched_random(rows: list[LatentRow], selected: list[LatentRow], *, seed: int) -> list[LatentRow]:
    rng = random.Random(seed)
    by_layer: dict[int, list[LatentRow]] = {}
    selected_keys = {(r.layer, r.latent_id) for r in selected}
    for row in rows:
        if (row.layer, row.latent_id) not in selected_keys:
            by_layer.setdefault(row.layer, []).append(row)
    out: list[LatentRow] = []
    for target in selected:
        pool = by_layer.get(target.layer, [])
        if not pool:
            continue
        pool_sorted = sorted(pool, key=lambda r: abs(r.match_value - target.match_value))
        window = pool_sorted[: min(len(pool_sorted), 50)]
        choice = rng.choice(window)
        out.append(choice)
        by_layer[target.layer] = [r for r in pool if r.latent_id != choice.latent_id]
    return out


def _to_selection(name: str, rows: list[LatentRow], *, mode: str = "ablate") -> FeatureSelection:
    by_layer: dict[int, list[int]] = {}
    for row in rows:
        by_layer.setdefault(row.layer, []).append(row.latent_id)
    return FeatureSelection(name=name, by_layer=by_layer, mode=mode)


def build_feature_selections(
    *,
    latent_rows: list[LatentRow],
    k_list: list[int],
    random_seeds: list[int],
    coverage_fracs: list[float] | None = None,
    coverage_metrics: list[str] | None = None,
    selection_suite: str = "default",
    causal_rows: list[LatentRow] | None = None,
    causal_pool_rows: list[LatentRow] | None = None,
) -> list[tuple[str, int, int | None, FeatureSelection]]:
    selections: list[tuple[str, int, int | None, FeatureSelection]] = []
    include_default = selection_suite in {"default", "default_plus_causal"}
    include_causal = selection_suite in {"causal", "default_plus_causal"}
    if include_default:
        selections.append(
            ("full_reconstruction", 0, None, FeatureSelection("full_reconstruction", {}, mode="full_reconstruct"))
        )
    if include_default:
        for metric in coverage_metrics or []:
            for frac in coverage_fracs or []:
                pct = int(round(float(frac) * 100))
                selections.append(
                    (
                        f"coverage_{metric}",
                        pct,
                        None,
                        FeatureSelection(
                            name=f"coverage_{metric}_{pct}",
                            by_layer={},
                            mode="coverage_ablate",
                            coverage_fraction=float(frac),
                            coverage_metric=metric,
                        ),
                    )
                )
    available_layers = sorted({r.layer for r in latent_rows})
    next_layer = {
        layer: available_layers[(idx + 1) % len(available_layers)]
        for idx, layer in enumerate(available_layers)
    }
    for k in k_list:
        if include_default:
            top = _top_interaction(latent_rows, k)
            selections.append(("top_interaction", k, None, _to_selection(f"top_interaction_k{k}", top)))
            selections.append(("shared", k, None, _to_selection(f"shared_k{k}", _shared(latent_rows, k))))
            selections.append(("pt_biased", k, None, _to_selection(f"pt_biased_k{k}", _pt_biased(latent_rows, k))))
            shifted = [
                LatentRow(
                    layer=next_layer[r.layer],
                    latent_id=r.latent_id,
                    feature_type=r.feature_type,
                    interaction_score=r.interaction_score,
                    mean_activation_it=r.mean_activation_it,
                    mean_activation_pt=r.mean_activation_pt,
                    decoder_norm_it=r.decoder_norm_it,
                    decoder_norm_pt=r.decoder_norm_pt,
                    latent_scaling_ratio=r.latent_scaling_ratio,
                    local_margin_attr=r.local_margin_attr,
                )
                for r in top
            ]
            selections.append(("shuffle_layers", k, None, _to_selection(f"shuffle_layers_k{k}", shifted)))
            for seed in random_seeds:
                rand_rows = _matched_random(latent_rows, top, seed=seed)
                selections.append(("matched_random", k, seed, _to_selection(f"matched_random_k{k}_seed{seed}", rand_rows)))
        if include_causal:
            if causal_rows is None:
                raise ValueError("selection_suite includes causal selections but causal_rows is None")
            causal_pool = causal_pool_rows or latent_rows
            causal_top = causal_rows[:k]
            selections.append(("causal_top", k, None, _to_selection(f"causal_top_k{k}", causal_top)))
            for seed in random_seeds:
                rand_rows = _matched_random(causal_pool, causal_top, seed=seed)
                selections.append(
                    (
                        "causal_matched_random",
                        k,
                        seed,
                        _to_selection(f"causal_matched_random_k{k}_seed{seed}", rand_rows),
                    )
                )
    return selections


@torch.no_grad()
def _forward_with_optional_edit(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    run_root: Path,
    steering_adapter: Any,
    selection: FeatureSelection | None = None,
    donor_boundary_state: torch.Tensor | None = None,
    crosscoder_cache: dict[int, Any] | None = None,
    crosscoder_dtype: torch.dtype | None = None,
    coverage_margin_vector: torch.Tensor | None = None,
) -> dict[str, Any]:
    patcher = None
    modifier = None
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        if donor_boundary_state is not None:
            patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state)
        if selection is not None:
            modifier = CrosscoderMlpModifier(
                model=model,
                steering_adapter=steering_adapter,
                run_root=run_root,
                selection=selection,
                device=input_ids.device,
                crosscoder_cache=crosscoder_cache,
                crosscoder_dtype=crosscoder_dtype,
                coverage_margin_vector=coverage_margin_vector,
            )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        residuals = residual_capture.snapshot()
    finally:
        if modifier is not None:
            modifier.close()
        if patcher is not None:
            patcher.close()
        residual_capture.close()
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": residuals[-1],
        "hook_summary": modifier.summary() if modifier is not None else {},
        "patch_n": patcher.n_patches if patcher is not None else 0,
    }


@torch.no_grad()
def _baseline_with_boundary(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
) -> dict[str, Any]:
    capture = BoundaryStateCapture(layers[boundary_layer])
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        boundary = capture.snapshot()
        residuals = residual_capture.snapshot()
    finally:
        capture.close()
        residual_capture.close()
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": residuals[-1],
        "boundary_state": boundary,
    }


def _margin(cell: dict[str, Any], readout_bundle: Any, y_pt: int, y_it: int) -> float:
    logits = _logits_from_hidden(cell["final_hidden"], readout_bundle)
    return float((logits[y_it] - logits[y_pt]).item())


def _interaction(cells: dict[str, float]) -> float:
    return (cells["U_IT__L_IT"] - cells["U_IT__L_PT"]) - (cells["U_PT__L_IT"] - cells["U_PT__L_PT"])


def _dtype_from_name(name: str) -> torch.dtype | None:
    if name == "float32":
        return None
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported crosscoder dtype: {name}")


def _readout_margin_vector(readout_bundle: Any, *, y_pt: int, y_it: int, device: torch.device) -> torch.Tensor:
    """Proxy local margin vector used only to rank active feature coverage.

    The Exp23 margin is scored after downstream blocks and final norm, so this
    vector is not a linear causal readout of an intermediate MLP output. It is a
    practical ranking proxy for dynamic coverage ablations; the causal quantity
    remains the full four-cell forward-pass interaction measured after editing.
    """

    weight = readout_bundle.lm_head.weight.detach().float()
    return (weight[int(y_it)] - weight[int(y_pt)]).to(device=device)


def _done_keys(path: Path) -> set[str]:
    out = set()
    candidates = [path, path.parent / "records.jsonl.gz"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for row in _json_rows(candidate):
            out.add(
                "|".join(
                    [
                        str(row.get("prompt_id")),
                        str(row.get("event_kind")),
                        str(row.get("feature_set")),
                        str(row.get("k")),
                        str(row.get("control_seed")),
                    ]
                )
            )
    return out


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = get_spec(args.model)
    steering_adapter = get_steering_adapter(args.model)
    pt_model, pt_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "pt"),
        args.device,
        multi_gpu=spec.multi_gpu,
    )
    it_model, it_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "it"),
        args.device,
        multi_gpu=spec.multi_gpu,
    )
    pt_model.requires_grad_(False)
    it_model.requires_grad_(False)
    models = {"pt": pt_model, "it": it_model}
    tokenizers = {"pt": pt_tokenizer, "it": it_tokenizer}
    real_token_masks = {
        "pt": steering_adapter.real_token_mask(pt_tokenizer, device, pt_model),
        "it": steering_adapter.real_token_mask(it_tokenizer, device, it_model),
    }
    readouts = _make_readouts(
        models=models,
        tokenizers=tokenizers,
        steering_adapter=steering_adapter,
        real_token_masks=real_token_masks,
    )
    readout_bundle = readouts[args.readout_name]
    pt_layers = steering_adapter.get_layers(pt_model)
    it_layers = steering_adapter.get_layers(it_model)
    adapter = steering_adapter.adapter
    boundary_layer = _late_boundary(args.model)
    dataset_by_id = _dataset_lookup(args.dataset)
    manifest_rows = _load_manifest_records_window(
        exp20_root=args.exp20_root,
        exp20_fallback_root=args.exp20_fallback_root,
        prompt_mode=args.prompt_mode,
        model=args.model,
        n_examples=args.n_prompts,
        skip_examples=args.skip_prompts,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
    )
    latent_path = args.run_root / "feature_stats" / "all_latents.csv"
    include_default = args.selection_suite in {"default", "default_plus_causal"}
    if include_default:
        latent_rows = _load_latent_rows(latent_path)
    else:
        latent_rows = []
    causal_rows = None
    causal_pool_rows = None
    if args.causal_feature_csv is not None:
        causal_rows, causal_pool_rows = _load_causal_rank_rows(args.causal_feature_csv)
        if not latent_rows:
            latent_rows = causal_pool_rows
    selections = build_feature_selections(
        latent_rows=latent_rows,
        k_list=[int(x) for x in args.k_list],
        random_seeds=[int(x) for x in args.random_seeds],
        coverage_fracs=[float(x) for x in args.coverage_fracs],
        coverage_metrics=list(args.coverage_metrics),
        selection_suite=args.selection_suite,
        causal_rows=causal_rows,
        causal_pool_rows=causal_pool_rows,
    )
    crosscoder_cache: dict[int, Any] = {}
    crosscoder_dtype = _dtype_from_name(args.crosscoder_dtype)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"records_w{args.worker_index}.jsonl.gz"
    done = _done_keys(out_path)
    log.info(
        "[exp28-mediate] worker=%d/%d manifest_rows=%d selections=%d suite=%s",
        args.worker_index,
        args.n_workers,
        len(manifest_rows),
        len(selections),
        args.selection_suite,
    )
    valid_events = 0
    records_written = 0
    failures = 0
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for row_idx, manifest_record in enumerate(manifest_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                log.warning("[exp28-mediate] missing dataset record prompt_id=%s", prompt_id)
                continue
            raw_prompt = get_prompt_for_variant(
                dataset_record,
                variant="pt",
                tokenizer=pt_tokenizer,
                apply_chat_template=False,
            )
            prompt_ids = pt_tokenizer.encode(raw_prompt, add_special_tokens=True)
            for event_kind, event in _unique_events(manifest_record, list(args.event_kinds)):
                if "duplicate_of" in event:
                    continue
                y_pt = int(event["pt_token"]["token_id"])
                y_it = int(event["it_token"]["token_id"])
                prefix_ids = _prefix_ids_for_event(manifest_record, event)
                validation = _validate_tokenizers_and_tokens(
                    model_name=args.model,
                    prompt_mode=args.prompt_mode,
                    dataset_record=dataset_record,
                    prefix_ids=prefix_ids,
                    y_pt=y_pt,
                    y_it=y_it,
                    tokenizers=tokenizers,
                    readouts=readouts,
                )
                if not validation.get("ok"):
                    continue
                valid_events += 1
                full_ids = prompt_ids + prefix_ids
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                try:
                    margin_vector = _readout_margin_vector(readout_bundle, y_pt=y_pt, y_it=y_it, device=device)
                    base_pt = _baseline_with_boundary(
                        model=pt_model,
                        adapter=adapter,
                        layers=pt_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                    )
                    base_it = _baseline_with_boundary(
                        model=it_model,
                        adapter=adapter,
                        layers=it_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                    )
                    full_upt_lit = _forward_with_optional_edit(
                        model=it_model,
                        adapter=adapter,
                        layers=it_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        donor_boundary_state=base_pt["boundary_state"],
                        run_root=args.run_root,
                        steering_adapter=steering_adapter,
                    )
                    full_uit_lpt = _forward_with_optional_edit(
                        model=pt_model,
                        adapter=adapter,
                        layers=pt_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        donor_boundary_state=base_it["boundary_state"],
                        run_root=args.run_root,
                        steering_adapter=steering_adapter,
                    )
                    full_cells = {
                        "U_PT__L_PT": _margin(base_pt, readout_bundle, y_pt, y_it),
                        "U_IT__L_IT": _margin(base_it, readout_bundle, y_pt, y_it),
                        "U_PT__L_IT": _margin(full_upt_lit, readout_bundle, y_pt, y_it),
                        "U_IT__L_PT": _margin(full_uit_lpt, readout_bundle, y_pt, y_it),
                    }
                    interaction_full = _interaction(full_cells)
                    for feature_set, k, control_seed, selection in selections:
                        key = "|".join([prompt_id, event_kind, feature_set, str(k), str(control_seed)])
                        if key in done:
                            continue
                        edit_it = _forward_with_optional_edit(
                            model=it_model,
                            adapter=adapter,
                            layers=it_layers,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            boundary_layer=boundary_layer,
                            run_root=args.run_root,
                            steering_adapter=steering_adapter,
                            selection=selection,
                            crosscoder_cache=crosscoder_cache,
                            crosscoder_dtype=crosscoder_dtype,
                            coverage_margin_vector=margin_vector,
                        )
                        edit_upt_lit = _forward_with_optional_edit(
                            model=it_model,
                            adapter=adapter,
                            layers=it_layers,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            boundary_layer=boundary_layer,
                            donor_boundary_state=base_pt["boundary_state"],
                            run_root=args.run_root,
                            steering_adapter=steering_adapter,
                            selection=selection,
                            crosscoder_cache=crosscoder_cache,
                            crosscoder_dtype=crosscoder_dtype,
                            coverage_margin_vector=margin_vector,
                        )
                        ablate_cells = {
                            "U_PT__L_PT": full_cells["U_PT__L_PT"],
                            "U_IT__L_PT": full_cells["U_IT__L_PT"],
                            "U_IT__L_IT": _margin(edit_it, readout_bundle, y_pt, y_it),
                            "U_PT__L_IT": _margin(edit_upt_lit, readout_bundle, y_pt, y_it),
                        }
                        interaction_ablate = _interaction(ablate_cells)
                        drop = interaction_full - interaction_ablate
                        payload = {
                            "experiment": "exp28_late_mlp_crosscoder_mediation",
                            "model": args.model,
                            "prompt_id": prompt_id,
                            "prompt_mode": args.prompt_mode,
                            "event_kind": event_kind,
                            "event_step": int(event["step"]),
                            "position_ge_3": bool(int(event["step"]) >= 3),
                            "pt_token_id": y_pt,
                            "it_token_id": y_it,
                            "pt_token_text": validation.get("pt_token_text"),
                            "it_token_text": validation.get("it_token_text"),
                            "feature_set": feature_set,
                            "k": int(k),
                            "control_seed": control_seed,
                            "readout": args.readout_name,
                            "full_cells": full_cells,
                            "ablate_cells": ablate_cells,
                            "interaction_full": interaction_full,
                            "interaction_ablate": interaction_ablate,
                            "interaction_drop": drop,
                            "mediation_fraction": (drop / interaction_full) if interaction_full else None,
                            "hook_summary_it_late": edit_it.get("hook_summary", {}),
                            "hook_summary_pt_upstream_it_late": edit_upt_lit.get("hook_summary", {}),
                        }
                        fout.write(json.dumps(payload, separators=(",", ":")) + "\n")
                        fout.flush()
                        records_written += 1
                except Exception as exc:
                    failures += 1
                    log.exception("[exp28-mediate] prompt=%s event=%s failed: %s", prompt_id, event_kind, exc)
            if (row_idx + 1) % 5 == 0:
                log.info("[exp28-mediate] worker=%d processed %d/%d prompts", args.worker_index, row_idx + 1, len(manifest_rows))
    log.info(
        "[exp28-mediate] worker=%d done valid_events=%d records_written=%d failures=%d",
        args.worker_index,
        valid_events,
        records_written,
        failures,
    )
    if manifest_rows and valid_events == 0:
        raise RuntimeError(f"Exp28 mediation worker {args.worker_index} found no valid events")
    if valid_events and records_written == 0:
        raise RuntimeError(
            f"Exp28 mediation worker {args.worker_index} wrote no records "
            f"for {valid_events} valid events; failures={failures}"
        )


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    seen: set[tuple[Any, ...]] = set()
    n_written = 0
    existing_rows = list(_json_rows(merged)) if merged.exists() else []
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for row in existing_rows:
            key = (
                row.get("prompt_id"),
                row.get("event_kind"),
                row.get("feature_set"),
                row.get("k"),
                row.get("control_seed"),
            )
            if key in seen:
                continue
            seen.add(key)
            fout.write(json.dumps(row, separators=(",", ":")) + "\n")
            n_written += 1
        for worker in range(n_workers):
            path = out_dir / f"records_w{worker}.jsonl.gz"
            if not path.exists():
                log.warning("[exp28-mediate] missing worker file %s", path)
                continue
            for row in _json_rows(path):
                key = (
                    row.get("prompt_id"),
                    row.get("event_kind"),
                    row.get("feature_set"),
                    row.get("k"),
                    row.get("control_seed"),
                )
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
                n_written += 1
    if n_written == 0:
        raise RuntimeError(f"Exp28 merge wrote zero mediation records to {merged}")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="llama31_8b")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--prompt-mode", choices=["raw_shared", "native"], default="raw_shared")
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=["first_diff"])
    parser.add_argument("--readout-name", default="common_it")
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--skip-prompts", type=int, default=0)
    parser.add_argument("--k-list", nargs="+", type=int, default=[50, 200, 1000])
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--coverage-fracs", nargs="*", type=float, default=[])
    parser.add_argument(
        "--coverage-metrics",
        nargs="*",
        choices=["norm", "activation", "margin_pos", "margin_abs"],
        default=["norm", "margin_pos"],
    )
    parser.add_argument(
        "--selection-suite",
        choices=["default", "causal", "default_plus_causal"],
        default="default",
        help="Feature-selection family to evaluate. 'causal' uses causal-rank outputs plus matched random controls.",
    )
    parser.add_argument(
        "--causal-feature-csv",
        type=Path,
        default=None,
        help="CSV from the causal-rank phase, sorted by factorial-aware feature score.",
    )
    parser.add_argument("--crosscoder-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.dataset.exists() and args.dataset.name == "eval_dataset_v2_holdout_0600_1199.jsonl":
        args.dataset = Path("data/eval_dataset_v2.jsonl")
    return args


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_workers(args.out_dir, args.n_workers)
        return
    run_worker(args)


if __name__ == "__main__":
    main()
