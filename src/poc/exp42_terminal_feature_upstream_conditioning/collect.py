"""Collect Exp42 terminal feature upstream-conditioning records.

Exp42 reuses the Exp23 residual-state x late-stack factorial and the
Exp28/34/38 terminal crosscoder mediation hooks.  The new measurement is the
activation/casual gate for the same causally ranked terminal features:

    A(U_IT, L_IT) - A(U_PT, L_IT)
    drop(U_IT, L_IT) - drop(U_PT, L_IT)

where ``A`` is selected sparse-feature activation and ``drop`` is the
finite-difference readout-margin loss after ablating the selected IT-branch
decoder contribution.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import random
from collections import defaultdict
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
    _make_readouts,
    _prefix_ids_for_event,
    _unique_events,
    _validate_tokenizers_and_tokens,
)
from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import BatchTopKCrossCoder
from src.poc.exp28_late_mlp_crosscoder_mediation.mediation_hooks import FeatureSelection
from src.poc.exp28_late_mlp_crosscoder_mediation.run_mediation import (
    LatentRow,
    _dtype_from_name,
    _forward_with_optional_edit,
    _interaction,
    _load_causal_rank_rows,
    _load_manifest_records_window,
    _margin,
    _matched_random,
    _readout_margin_vector,
    _to_selection,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


CLEAN_MODELS = ("gemma3_4b", "llama31_8b", "mistral_7b", "qwen3_4b")
DEFAULT_K_LIST = (1, 2, 5, 10, 20, 50, 100, 200)
PRIMARY_CELLS = ("U_IT__L_IT", "U_PT__L_IT")
ALL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")


@dataclass(frozen=True)
class SelectionSpec:
    feature_set: str
    k: int
    control_seed: int | None
    rows: tuple[LatentRow, ...]

    @property
    def selection(self) -> FeatureSelection:
        seed_suffix = "" if self.control_seed is None else f"_seed{self.control_seed}"
        return _to_selection(f"{self.feature_set}_k{self.k}{seed_suffix}", list(self.rows))


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _done_event_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for row in _json_rows(path):
        if row.get("record_type") == "event_done":
            out.add("|".join([str(row.get("prompt_id")), str(row.get("event_kind"))]))
    return out


def _load_latent_rows_for_model(run_root: Path) -> tuple[list[LatentRow], list[LatentRow]]:
    causal_csv = run_root / "feature_stats" / "causal_feature_scores.csv"
    return _load_causal_rank_rows(causal_csv)


def _top_active_noncausal(pool_rows: list[LatentRow], causal_rows: list[LatentRow], *, k: int) -> list[LatentRow]:
    causal_keys = {(row.layer, row.latent_id) for row in causal_rows[: max(k, max(DEFAULT_K_LIST))]}
    abs_scores = sorted(abs(row.interaction_score) for row in pool_rows)
    low_abs_cut = abs_scores[int(0.50 * (len(abs_scores) - 1))] if abs_scores else 0.0
    candidates = [
        row
        for row in pool_rows
        if (row.layer, row.latent_id) not in causal_keys
        and (row.interaction_score <= 0.0 or abs(row.interaction_score) <= low_abs_cut)
    ]
    if len(candidates) < k:
        candidates = [row for row in pool_rows if (row.layer, row.latent_id) not in causal_keys]
    candidates.sort(key=lambda row: row.mean_activation_it * row.decoder_norm_it, reverse=True)
    return candidates[:k]


def build_selection_specs(
    *,
    causal_rows: list[LatentRow],
    pool_rows: list[LatentRow],
    k_list: list[int],
    random_seeds: list[int],
    include_top_active: bool,
) -> list[SelectionSpec]:
    max_k = max(k_list)
    specs: list[SelectionSpec] = []
    causal_top_max = causal_rows[:max_k]
    fixed_random_orders = {
        int(seed): _matched_random(pool_rows, causal_top_max, seed=int(seed))
        for seed in random_seeds
    }
    for k in k_list:
        specs.append(SelectionSpec("causal_top", int(k), None, tuple(causal_rows[:k])))
        for seed in random_seeds:
            specs.append(
                SelectionSpec(
                    "causal_matched_random",
                    int(k),
                    int(seed),
                    tuple(fixed_random_orders[int(seed)][:k]),
                )
            )
        if include_top_active:
            specs.append(
                SelectionSpec(
                    "top_active_noncausal",
                    int(k),
                    None,
                    tuple(_top_active_noncausal(pool_rows, causal_rows, k=int(k))),
                )
            )
    return specs


def _layer_feature_sets(specs: list[SelectionSpec]) -> dict[int, dict[str, list[int]]]:
    by_layer: dict[int, dict[str, list[int]]] = defaultdict(dict)
    for spec in specs:
        key = _selection_key(spec)
        layer_ids: dict[int, list[int]] = defaultdict(list)
        for row in spec.rows:
            layer_ids[int(row.layer)].append(int(row.latent_id))
        for layer, ids in layer_ids.items():
            by_layer[layer][key] = ids
    return by_layer


def _selection_key(spec: SelectionSpec) -> str:
    seed = "none" if spec.control_seed is None else str(spec.control_seed)
    return f"{spec.feature_set}|{spec.k}|{seed}"


def _split_key(key: str) -> tuple[str, int, str]:
    feature_set, k, seed = key.split("|")
    return feature_set, int(k), seed


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class FeatureActivationRecorder:
    """Capture selected sparse-feature activation during one terminal-stack pass."""

    def __init__(
        self,
        *,
        model: Any,
        steering_adapter: Any,
        run_root: Path,
        layer_feature_sets: dict[int, dict[str, list[int]]],
        branch: int,
        device: torch.device,
        margin_vector: torch.Tensor,
        crosscoder_cache: dict[tuple[int, torch.dtype | None], BatchTopKCrossCoder],
        crosscoder_dtype: torch.dtype | None,
        use_threshold: bool = True,
        compute_diagnostics: bool = True,
    ) -> None:
        self.model = model
        self.steering_adapter = steering_adapter
        self.run_root = run_root
        self.layer_feature_sets = layer_feature_sets
        self.branch = int(branch)
        self.device = device
        self.margin_vector = margin_vector.detach().float().to(device)
        self.crosscoder_cache = crosscoder_cache
        self.crosscoder_dtype = crosscoder_dtype
        self.use_threshold = bool(use_threshold)
        self.compute_diagnostics = bool(compute_diagnostics)
        self.layers = steering_adapter.get_layers(model)
        self.handles: list[Any] = []
        self.records: list[dict[str, Any]] = []

    def __enter__(self) -> "FeatureActivationRecorder":
        for layer_idx in sorted(self.layer_feature_sets):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                continue
            cc = self._load_layer(layer_idx)
            feature_sets = self.layer_feature_sets[layer_idx]
            union_ids = sorted({int(latent) for ids in feature_sets.values() for latent in ids})
            if not union_ids:
                continue
            union_tensor = torch.tensor(union_ids, dtype=torch.long, device=self.device)
            pos = {latent: idx for idx, latent in enumerate(union_ids)}
            set_positions = {
                key: torch.tensor([pos[int(latent)] for latent in ids], dtype=torch.long, device=self.device)
                for key, ids in feature_sets.items()
            }

            def hook(_module, _args, output, li=layer_idx, crosscoder=cc, latents=union_tensor, positions=set_positions):
                if not torch.is_tensor(output):
                    raise RuntimeError(f"Expected tensor MLP output at layer {li}, got {type(output)}")
                update = output[:, -1, :]
                _contrib, selected_features = crosscoder.selected_branch_contribution(
                    update,
                    branch=self.branch,
                    latent_ids=latents,
                    use_threshold=self.use_threshold,
                )
                selected_features = selected_features.float()
                decoder = crosscoder.decoder[latents, self.branch, :].detach().float().to(update.device)
                decoder_norm = decoder.norm(dim=-1)
                decoder_margin = decoder @ self.margin_vector.to(update.device)

                diagnostics = self._diagnostics(crosscoder, update, selected_features)
                selected_total = selected_features.sum(dim=-1).mean().clamp_min(1e-12)
                for key, idxs in positions.items():
                    feats = selected_features[:, idxs]
                    norms = decoder_norm[idxs]
                    margins = decoder_margin[idxs]
                    feature_set, k, seed = _split_key(key)
                    sum_activation = float(feats.sum(dim=-1).mean().item())
                    record = {
                        "record_type": "activation",
                        "layer": int(li),
                        "branch": int(self.branch),
                        "feature_set": feature_set,
                        "k": int(k),
                        "control_seed": None if seed == "none" else int(seed),
                        "n_features": int(idxs.numel()),
                        "activation_rate": float((feats > 0).float().mean().item()) if feats.numel() else 0.0,
                        "mean_activation": float(feats.mean().item()) if feats.numel() else 0.0,
                        "sum_activation": sum_activation,
                        "decoder_weighted_sum": float((feats * norms[None, :]).sum(dim=-1).mean().item()) if feats.numel() else 0.0,
                        "decoder_margin_weighted_sum": float((feats * margins[None, :]).sum(dim=-1).mean().item()) if feats.numel() else 0.0,
                        "selected_feature_l0": float((feats > 0).float().sum(dim=-1).mean().item()) if feats.numel() else 0.0,
                        "selected_activation_mass_fraction": float((feats.sum(dim=-1).mean() / selected_total).item())
                        if feats.numel()
                        else 0.0,
                    }
                    record.update(diagnostics)
                    self.records.append(record)
                return output

            self.handles.append(self.layers[layer_idx].mlp.register_forward_hook(hook))
        return self

    def _load_layer(self, layer_idx: int) -> BatchTopKCrossCoder:
        key = (int(layer_idx), self.crosscoder_dtype)
        if key not in self.crosscoder_cache:
            path = self.run_root / "dictionaries" / f"layer_{layer_idx}" / "crosscoder.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing crosscoder dictionary: {path}")
            cc = BatchTopKCrossCoder.load(path, device=self.device)
            if self.crosscoder_dtype is not None:
                cc = cc.to(dtype=self.crosscoder_dtype)
            cc.eval()
            self.crosscoder_cache[key] = cc
        return self.crosscoder_cache[key]

    def _diagnostics(
        self,
        crosscoder: BatchTopKCrossCoder,
        update: torch.Tensor,
        selected_features: torch.Tensor,
    ) -> dict[str, Any]:
        if not self.compute_diagnostics:
            return {
                "reconstruction_error_rel": None,
                "feature_l0": None,
                "total_activation_mass": None,
            }
        with torch.no_grad():
            features = crosscoder.encode_branch(
                update,
                branch=self.branch,
                use_threshold=self.use_threshold,
            )
            recon = crosscoder.decode_branch(features, branch=self.branch).to(update.device).float()
            base = update.float()
            rel = (recon - base).norm(dim=-1).mean() / base.norm(dim=-1).mean().clamp_min(1e-8)
            total_mass = features.float().sum(dim=-1).mean().clamp_min(1e-12)
            return {
                "reconstruction_error_rel": float(rel.item()),
                "feature_l0": float((features > 0).float().sum(dim=-1).mean().item()),
                "total_activation_mass": float(total_mass.item()),
                "selected_union_mass_fraction": float((selected_features.sum(dim=-1).mean() / total_mass).item()),
            }

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


@torch.no_grad()
def _baseline_with_boundary_and_features(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    recorder: FeatureActivationRecorder | None,
) -> dict[str, Any]:
    capture = BoundaryStateCapture(layers[boundary_layer])
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        if recorder is None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            with recorder:
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
        "activation_records": list(recorder.records) if recorder is not None else [],
    }


@torch.no_grad()
def _patched_with_features(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    donor_boundary_state: torch.Tensor,
    recorder: FeatureActivationRecorder | None,
) -> dict[str, Any]:
    patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state)
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        if recorder is None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            with recorder:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        residuals = residual_capture.snapshot()
    finally:
        patcher.close()
        residual_capture.close()
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": residuals[-1],
        "patch_n": patcher.n_patches,
        "activation_records": list(recorder.records) if recorder is not None else [],
    }


def _attach_event_metadata(
    rows: list[dict[str, Any]],
    *,
    model: str,
    prompt_id: str,
    prompt_mode: str,
    event_kind: str,
    event: dict[str, Any],
    cell: str,
    pt_token_id: int,
    it_token_id: int,
    pt_token_text: str | None,
    it_token_text: str | None,
    boundary_layer: int,
    terminal_scope: str,
    full_cells: dict[str, float],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                **row,
                "experiment": "exp42_terminal_feature_upstream_conditioning",
                "model": model,
                "prompt_id": prompt_id,
                "prompt_mode": prompt_mode,
                "event_kind": event_kind,
                "event_step": int(event["step"]),
                "position_ge_3": bool(int(event["step"]) >= 3),
                "cell": cell,
                "pt_token_id": int(pt_token_id),
                "it_token_id": int(it_token_id),
                "pt_token_text": pt_token_text,
                "it_token_text": it_token_text,
                "boundary_layer": int(boundary_layer),
                "terminal_scope": terminal_scope,
                "full_cells": full_cells,
                "interaction_full": _interaction(full_cells),
            }
        )
    return out


def _terminal_scope(boundary_layer: int, n_layers: int) -> str:
    n = int(n_layers) - int(boundary_layer)
    return f"final{n}"


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
    boundary_layer = int(args.boundary_layer_override) if args.boundary_layer_override is not None else _late_boundary(args.model)
    if boundary_layer < 0 or boundary_layer >= min(len(pt_layers), len(it_layers)):
        raise ValueError(f"Boundary layer {boundary_layer} outside model layer range")
    terminal_scope = args.terminal_scope or _terminal_scope(boundary_layer, min(len(pt_layers), len(it_layers)))

    causal_rows, pool_rows = _load_latent_rows_for_model(args.crosscoder_root)
    k_list = [int(k) for k in args.k_list]
    specs = build_selection_specs(
        causal_rows=causal_rows,
        pool_rows=pool_rows,
        k_list=k_list,
        random_seeds=[int(seed) for seed in args.random_seeds],
        include_top_active=not args.no_top_active_control,
    )
    layer_sets = _layer_feature_sets(specs)
    log.info(
        "[exp42] model=%s boundary=%d scope=%s selections=%d layers=%s",
        args.model,
        boundary_layer,
        terminal_scope,
        len(specs),
        sorted(layer_sets),
    )

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
    crosscoder_dtype = _dtype_from_name(args.crosscoder_dtype)
    crosscoder_cache: dict[tuple[int, torch.dtype | None], BatchTopKCrossCoder] = {}
    ablation_crosscoder_cache: dict[int, Any] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"records_w{args.worker_index}.jsonl.gz"
    done = _done_event_keys(out_path)
    log.info("[exp42] worker=%d/%d rows=%d resume_done_events=%d", args.worker_index, args.n_workers, len(manifest_rows), len(done))
    n_events = 0
    n_written = 0
    n_fail = 0
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for row_idx, manifest_record in enumerate(manifest_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                log.warning("[exp42] missing dataset record prompt_id=%s", prompt_id)
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
                event_key = "|".join([prompt_id, event_kind])
                if event_key in done:
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
                n_events += 1
                full_ids = prompt_ids + prefix_ids
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                try:
                    margin_vector = _readout_margin_vector(readout_bundle, y_pt=y_pt, y_it=y_it, device=device)
                    rec_pt = FeatureActivationRecorder(
                        model=pt_model,
                        steering_adapter=steering_adapter,
                        run_root=args.crosscoder_root,
                        layer_feature_sets=layer_sets,
                        branch=0,
                        device=device,
                        margin_vector=margin_vector,
                        crosscoder_cache=crosscoder_cache,
                        crosscoder_dtype=crosscoder_dtype,
                        use_threshold=not args.no_threshold,
                        compute_diagnostics=not args.no_dictionary_diagnostics,
                    )
                    rec_it = FeatureActivationRecorder(
                        model=it_model,
                        steering_adapter=steering_adapter,
                        run_root=args.crosscoder_root,
                        layer_feature_sets=layer_sets,
                        branch=1,
                        device=device,
                        margin_vector=margin_vector,
                        crosscoder_cache=crosscoder_cache,
                        crosscoder_dtype=crosscoder_dtype,
                        use_threshold=not args.no_threshold,
                        compute_diagnostics=not args.no_dictionary_diagnostics,
                    )
                    base_pt = _baseline_with_boundary_and_features(
                        model=pt_model,
                        adapter=adapter,
                        layers=pt_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        recorder=rec_pt,
                    )
                    base_it = _baseline_with_boundary_and_features(
                        model=it_model,
                        adapter=adapter,
                        layers=it_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        recorder=rec_it,
                    )
                    rec_upt_lit = FeatureActivationRecorder(
                        model=it_model,
                        steering_adapter=steering_adapter,
                        run_root=args.crosscoder_root,
                        layer_feature_sets=layer_sets,
                        branch=1,
                        device=device,
                        margin_vector=margin_vector,
                        crosscoder_cache=crosscoder_cache,
                        crosscoder_dtype=crosscoder_dtype,
                        use_threshold=not args.no_threshold,
                        compute_diagnostics=not args.no_dictionary_diagnostics,
                    )
                    rec_uit_lpt = FeatureActivationRecorder(
                        model=pt_model,
                        steering_adapter=steering_adapter,
                        run_root=args.crosscoder_root,
                        layer_feature_sets=layer_sets,
                        branch=0,
                        device=device,
                        margin_vector=margin_vector,
                        crosscoder_cache=crosscoder_cache,
                        crosscoder_dtype=crosscoder_dtype,
                        use_threshold=not args.no_threshold,
                        compute_diagnostics=not args.no_dictionary_diagnostics,
                    )
                    full_upt_lit = _patched_with_features(
                        model=it_model,
                        adapter=adapter,
                        layers=it_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        donor_boundary_state=base_pt["boundary_state"],
                        recorder=rec_upt_lit,
                    )
                    full_uit_lpt = _patched_with_features(
                        model=pt_model,
                        adapter=adapter,
                        layers=pt_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        donor_boundary_state=base_it["boundary_state"],
                        recorder=rec_uit_lpt,
                    )
                    full_cells = {
                        "U_PT__L_PT": _margin(base_pt, readout_bundle, y_pt, y_it),
                        "U_IT__L_IT": _margin(base_it, readout_bundle, y_pt, y_it),
                        "U_PT__L_IT": _margin(full_upt_lit, readout_bundle, y_pt, y_it),
                        "U_IT__L_PT": _margin(full_uit_lpt, readout_bundle, y_pt, y_it),
                    }
                    meta = {
                        "model": args.model,
                        "prompt_id": prompt_id,
                        "prompt_mode": args.prompt_mode,
                        "event_kind": event_kind,
                        "event": event,
                        "pt_token_id": y_pt,
                        "it_token_id": y_it,
                        "pt_token_text": validation.get("pt_token_text"),
                        "it_token_text": validation.get("it_token_text"),
                        "boundary_layer": boundary_layer,
                        "terminal_scope": terminal_scope,
                        "full_cells": full_cells,
                    }
                    cell_rows = {
                        "U_PT__L_PT": base_pt["activation_records"],
                        "U_IT__L_IT": base_it["activation_records"],
                        "U_PT__L_IT": full_upt_lit["activation_records"],
                        "U_IT__L_PT": full_uit_lpt["activation_records"],
                    }
                    for cell, rows in cell_rows.items():
                        for payload in _attach_event_metadata(cell=cell, **meta, rows=rows):
                            fout.write(json.dumps(payload, separators=(",", ":")) + "\n")
                            n_written += 1

                    for spec_sel in specs:
                        selection = spec_sel.selection
                        edit_it = _forward_with_optional_edit(
                            model=it_model,
                            adapter=adapter,
                            layers=it_layers,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            boundary_layer=boundary_layer,
                            run_root=args.crosscoder_root,
                            steering_adapter=steering_adapter,
                            selection=selection,
                            crosscoder_cache=ablation_crosscoder_cache,
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
                            run_root=args.crosscoder_root,
                            steering_adapter=steering_adapter,
                            selection=selection,
                            crosscoder_cache=ablation_crosscoder_cache,
                            crosscoder_dtype=crosscoder_dtype,
                            coverage_margin_vector=margin_vector,
                        )
                        ablate_cells = {
                            "U_PT__L_PT": full_cells["U_PT__L_PT"],
                            "U_IT__L_PT": full_cells["U_IT__L_PT"],
                            "U_IT__L_IT": _margin(edit_it, readout_bundle, y_pt, y_it),
                            "U_PT__L_IT": _margin(edit_upt_lit, readout_bundle, y_pt, y_it),
                        }
                        interaction_full = _interaction(full_cells)
                        interaction_ablate = _interaction(ablate_cells)
                        drop_native = full_cells["U_IT__L_IT"] - ablate_cells["U_IT__L_IT"]
                        drop_pt_upstream = full_cells["U_PT__L_IT"] - ablate_cells["U_PT__L_IT"]
                        ablation_payload = {
                            "experiment": "exp42_terminal_feature_upstream_conditioning",
                            "record_type": "ablation",
                            "model": args.model,
                            "prompt_id": prompt_id,
                            "prompt_mode": args.prompt_mode,
                            "event_kind": event_kind,
                            "event_step": int(event["step"]),
                            "position_ge_3": bool(int(event["step"]) >= 3),
                            "feature_set": spec_sel.feature_set,
                            "k": int(spec_sel.k),
                            "control_seed": spec_sel.control_seed,
                            "n_features": len(spec_sel.rows),
                            "pt_token_id": y_pt,
                            "it_token_id": y_it,
                            "pt_token_text": validation.get("pt_token_text"),
                            "it_token_text": validation.get("it_token_text"),
                            "boundary_layer": int(boundary_layer),
                            "terminal_scope": terminal_scope,
                            "full_cells": full_cells,
                            "ablate_cells": ablate_cells,
                            "interaction_full": interaction_full,
                            "interaction_ablate": interaction_ablate,
                            "interaction_drop": interaction_full - interaction_ablate,
                            "mediation_fraction": (interaction_full - interaction_ablate) / interaction_full
                            if interaction_full
                            else None,
                            "drop_native_upstream": drop_native,
                            "drop_pt_upstream": drop_pt_upstream,
                            "feature_causal_gate": drop_native - drop_pt_upstream,
                            "hook_summary_it_late": edit_it.get("hook_summary", {}),
                            "hook_summary_pt_upstream_it_late": edit_upt_lit.get("hook_summary", {}),
                        }
                        fout.write(json.dumps(ablation_payload, separators=(",", ":")) + "\n")
                        n_written += 1
                    fout.write(
                        json.dumps(
                            {
                                "experiment": "exp42_terminal_feature_upstream_conditioning",
                                "record_type": "event_done",
                                "model": args.model,
                                "prompt_id": prompt_id,
                                "event_kind": event_kind,
                                "event_step": int(event["step"]),
                                "n_activation_rows": sum(len(rows) for rows in cell_rows.values()),
                                "n_ablation_rows": len(specs),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    fout.flush()
                except Exception as exc:
                    n_fail += 1
                    log.exception("[exp42] prompt=%s event=%s failed: %s", prompt_id, event_kind, exc)
            if (row_idx + 1) % 5 == 0:
                log.info(
                    "[exp42] model=%s worker=%d/%d prompts=%d/%d events=%d rows=%d fail=%d",
                    args.model,
                    args.worker_index,
                    args.n_workers,
                    row_idx + 1,
                    len(manifest_rows),
                    n_events,
                    n_written,
                    n_fail,
                )
    log.info("[exp42] done model=%s events=%d rows=%d fail=%d", args.model, n_events, n_written, n_fail)


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    done_keys: set[str] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker in range(n_workers):
            path = out_dir / f"records_w{worker}.jsonl.gz"
            if not path.exists():
                log.warning("[exp42] missing worker file %s", path)
                continue
            current_event: tuple[str, str] | None = None
            buffer: list[dict[str, Any]] = []
            for row in _json_rows(path):
                row_type = row.get("record_type")
                if row_type == "event_done":
                    key = (str(row.get("prompt_id")), str(row.get("event_kind")))
                    if "|".join(key) not in done_keys:
                        for buffered in buffer:
                            fout.write(json.dumps(buffered, separators=(",", ":")) + "\n")
                        fout.write(json.dumps(row, separators=(",", ":")) + "\n")
                        done_keys.add("|".join(key))
                    buffer = []
                    current_event = None
                    continue
                key = (str(row.get("prompt_id")), str(row.get("event_kind")))
                if current_event is None:
                    current_event = key
                if key != current_event:
                    # Partial previous event without event_done; discard it.
                    buffer = []
                    current_event = key
                buffer.append(row)
    if not done_keys:
        raise RuntimeError(f"Exp42 merge wrote no complete events to {merged}")
    return merged


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--prompt-mode", choices=["raw_shared"], default="raw_shared")
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=["first_diff"])
    parser.add_argument("--readout-name", default="common_it")
    parser.add_argument("--crosscoder-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-prompts", type=int, default=600)
    parser.add_argument("--skip-prompts", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--k-list", nargs="+", type=int, default=list(DEFAULT_K_LIST))
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--boundary-layer-override", type=int, default=None)
    parser.add_argument("--terminal-scope", default=None)
    parser.add_argument("--crosscoder-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-threshold", action="store_true")
    parser.add_argument("--no-top-active-control", action="store_true")
    parser.add_argument("--no-dictionary-diagnostics", action="store_true")
    parser.add_argument("--merge-only", action="store_true")


def main(args: argparse.Namespace) -> None:
    if args.merge_only:
        merge_workers(args.out_dir, args.n_workers)
        return
    run_worker(args)
