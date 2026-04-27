"""Collect Exp23 Part B residual-state x late-stack/readout factorial records."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp20_divergence_token_counterfactual.collect import DEPTH_ABLATION_WINDOWS
from src.poc.exp22_endpoint_deconfounded_gap.metrics import (
    distribution_arrays_from_logits,
    future_top1_flips,
    late_kl_mean,
    remaining_adjacent_path,
    top5_churn,
)
from src.poc.exp23_midlate_interaction_suite.boundary import (
    BoundaryStateCapture,
    BoundaryStatePatch,
    LayerResidualCapture,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


DEFAULT_EXP20_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early"
)
DEFAULT_EXP20_FALLBACK_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final"
)
DEFAULT_EVENT_KINDS = ("first_diff", "first_nonformat_diff", "first_assistant_marker_diff")
RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
NOOP_CELLS = ("U_PT__L_PT_noop_patch", "U_IT__L_IT_noop_patch")


@dataclass(frozen=True)
class CellSpec:
    name: str
    host_variant: str
    upstream_variant: str
    late_variant: str
    patch_from_variant: str | None = None
    noop_patch: bool = False


CELL_SPECS: dict[str, CellSpec] = {
    "U_PT__L_PT": CellSpec("U_PT__L_PT", "pt", "pt", "pt"),
    "U_IT__L_IT": CellSpec("U_IT__L_IT", "it", "it", "it"),
    "U_PT__L_IT": CellSpec("U_PT__L_IT", "it", "pt", "it", patch_from_variant="pt"),
    "U_IT__L_PT": CellSpec("U_IT__L_PT", "pt", "it", "pt", patch_from_variant="it"),
    "U_PT__L_PT_noop_patch": CellSpec(
        "U_PT__L_PT_noop_patch",
        "pt",
        "pt",
        "pt",
        patch_from_variant="pt",
        noop_patch=True,
    ),
    "U_IT__L_IT_noop_patch": CellSpec(
        "U_IT__L_IT_noop_patch",
        "it",
        "it",
        "it",
        patch_from_variant="it",
        noop_patch=True,
    ),
}


@dataclass(frozen=True)
class ReadoutBundle:
    name: str
    variant: str
    final_norm: torch.nn.Module
    lm_head: torch.nn.Module
    real_token_mask: torch.Tensor


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _find_manifest(root: Path, fallback_root: Path | None, prompt_mode: str, model: str) -> Path:
    candidates = [
        root / prompt_mode / model / "exp20_validation_records.jsonl",
        root / prompt_mode / model / "exp20_records.jsonl",
    ]
    if fallback_root is not None:
        candidates.extend(
            [
                fallback_root / prompt_mode / model / "exp20_validation_records.jsonl",
                fallback_root / prompt_mode / model / "exp20_records.jsonl",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No Exp20 manifest for {prompt_mode}/{model}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_manifest_records(
    *,
    exp20_root: Path,
    exp20_fallback_root: Path | None,
    prompt_mode: str,
    model: str,
    n_examples: int | None,
    worker_index: int,
    n_workers: int,
) -> list[dict[str, Any]]:
    path = _find_manifest(exp20_root, exp20_fallback_root, prompt_mode, model)
    rows = list(_json_rows(path))
    if n_examples is not None:
        rows = rows[:n_examples]
    sliced = rows[worker_index::n_workers]
    log.info("[exp23] manifest %s -> %d rows for worker %d/%d", path, len(sliced), worker_index, n_workers)
    return sliced


def _dataset_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id", row.get("record_id"))): row
        for row in load_dataset(path, n_examples=None)
    }


def _done_prompt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("prompt_id")) for row in _json_rows(path) if row.get("prompt_id") is not None}


def _prefix_ids_for_event(manifest_record: dict[str, Any], event: dict[str, Any]) -> list[int]:
    step = int(event["step"])
    free_runs = manifest_record.get("free_runs") or {}
    base = free_runs.get("A_pt_raw") or {}
    return [int(token_id) for token_id in (base.get("generated_token_ids") or [])[:step]]


def _unique_events(manifest_record: dict[str, Any], event_kinds: list[str]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    seen: dict[tuple[int, int, int], str] = {}
    for kind in event_kinds:
        event = (manifest_record.get("divergence_events") or {}).get(kind)
        if not isinstance(event, dict):
            continue
        pt_token = event.get("pt_token") or {}
        it_token = event.get("it_token") or {}
        if pt_token.get("token_id") is None or it_token.get("token_id") is None:
            continue
        key = (int(event["step"]), int(pt_token["token_id"]), int(it_token["token_id"]))
        if key in seen:
            out.append((kind, {"duplicate_of": seen[key], **event}))
            continue
        seen[key] = kind
        out.append((kind, event))
    return out


def _late_boundary(model_name: str) -> int:
    return int(DEPTH_ABLATION_WINDOWS[model_name]["late"][0])


def _token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _validate_tokenizers_and_tokens(
    *,
    model_name: str,
    prompt_mode: str,
    dataset_record: dict[str, Any],
    prefix_ids: list[int],
    y_pt: int,
    y_it: int,
    tokenizers: dict[str, Any],
    readouts: dict[str, ReadoutBundle],
) -> dict[str, Any]:
    if prompt_mode != "raw_shared":
        return {
            "ok": False,
            "reason": "residual_factorial_primary_requires_raw_shared",
        }

    prompt_pt = get_prompt_for_variant(
        dataset_record,
        variant="pt",
        tokenizer=tokenizers["pt"],
        apply_chat_template=False,
    )
    prompt_it = get_prompt_for_variant(
        dataset_record,
        variant="it",
        tokenizer=tokenizers["it"],
        apply_chat_template=False,
    )
    ids_pt = tokenizers["pt"].encode(prompt_pt, add_special_tokens=True)
    ids_it = tokenizers["it"].encode(prompt_it, add_special_tokens=True)
    if ids_pt != ids_it:
        return {
            "ok": False,
            "reason": "raw_shared_prompt_token_ids_differ",
            "pt_prompt_len": len(ids_pt),
            "it_prompt_len": len(ids_it),
        }

    for token_id in (y_pt, y_it, *prefix_ids):
        if int(token_id) < 0:
            return {"ok": False, "reason": "negative_token_id", "token_id": int(token_id)}
        pt_text = _token_text(tokenizers["pt"], int(token_id))
        it_text = _token_text(tokenizers["it"], int(token_id))
        if pt_text != it_text:
            return {
                "ok": False,
                "reason": "token_id_decodes_differ",
                "token_id": int(token_id),
                "pt_text": pt_text,
                "it_text": it_text,
            }
    for readout_name, bundle in readouts.items():
        vocab = int(bundle.lm_head.weight.shape[0])
        if y_pt >= vocab or y_it >= vocab:
            return {
                "ok": False,
                "reason": "target_id_out_of_readout_vocab",
                "readout": readout_name,
                "vocab": vocab,
                "y_pt": int(y_pt),
                "y_it": int(y_it),
            }
    return {
        "ok": True,
        "model": model_name,
        "prompt_len": len(ids_pt),
        "prefix_len": len(prefix_ids),
        "full_len": len(ids_pt) + len(prefix_ids),
        "pt_token_text": _token_text(tokenizers["pt"], y_pt),
        "it_token_text": _token_text(tokenizers["pt"], y_it),
    }


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    return torch.float32


def _logits_from_hidden(hidden: torch.Tensor, bundle: ReadoutBundle) -> torch.Tensor:
    norm_dtype = _module_dtype(bundle.final_norm)
    head_dtype = _module_dtype(bundle.lm_head)
    normed = bundle.final_norm(hidden.to(device=bundle.real_token_mask.device, dtype=norm_dtype).view(1, 1, -1))
    logits = bundle.lm_head(normed.view(1, -1).to(dtype=head_dtype)).float()[0]
    logits = logits.clone()
    logits[~bundle.real_token_mask.to(logits.device)] = float("-inf")
    return logits


def _logits_by_layer(residuals: list[torch.Tensor], bundle: ReadoutBundle) -> torch.Tensor:
    norm_dtype = _module_dtype(bundle.final_norm)
    head_dtype = _module_dtype(bundle.lm_head)
    hidden_stack = torch.stack(
        [hidden.to(device=bundle.real_token_mask.device, dtype=norm_dtype) for hidden in residuals],
        dim=0,
    )
    normed = bundle.final_norm(hidden_stack.view(len(residuals), 1, -1)).view(len(residuals), -1)
    logits = bundle.lm_head(normed.to(dtype=head_dtype)).float()
    logits = logits.clone()
    logits[:, ~bundle.real_token_mask.to(logits.device)] = float("-inf")
    return logits


def _rank(logits: torch.Tensor, token_id: int) -> int | None:
    if token_id < 0 or token_id >= logits.shape[-1]:
        return None
    return int((logits > logits[token_id]).sum().item()) + 1


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _trajectory_payload(residuals: list[torch.Tensor], bundle: ReadoutBundle) -> dict[str, Any]:
    logits = _logits_by_layer(residuals, bundle)
    arrays = distribution_arrays_from_logits(logits, top_k=5)
    return {
        "late_kl_mean": _safe_float(late_kl_mean(arrays["kl_to_final"])),
        "remaining_adj_js": _safe_float(remaining_adjacent_path(arrays["adjacent_js"], divergence="js")),
        "future_top1_flips": _safe_float(future_top1_flips(arrays["top1_ids"])),
        "top5_churn": _safe_float(top5_churn(arrays["top5_ids"])),
        "top1_path_metrics": {
            "future_top1_flips": _safe_float(future_top1_flips(arrays["top1_ids"])),
            "top5_churn": _safe_float(top5_churn(arrays["top5_ids"])),
        },
        "final_top1_id": int(arrays["top1_ids"][-1][0]),
    }


@torch.no_grad()
def _forward_cell(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    donor_boundary_state: torch.Tensor | None = None,
    collect_trajectories: bool,
) -> dict[str, Any]:
    patcher = None
    # Always capture layer outputs so common PT/IT readouts can be applied to
    # the actual final hidden state. ``collect_trajectories`` only controls
    # whether the whole residual stack is retained in the output metrics.
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        if donor_boundary_state is not None:
            patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        all_residuals = residual_capture.snapshot()
    finally:
        if patcher is not None:
            patcher.close()
        residual_capture.close()
    final_hidden = all_residuals[-1]
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": final_hidden,
        "residuals": all_residuals if collect_trajectories else [],
        "patch_n": patcher.n_patches if patcher is not None else 0,
        "patch_input_delta_max_abs": patcher.last_max_abs_input_delta if patcher is not None else None,
    }


@torch.no_grad()
def _baseline_forward_with_boundary(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    collect_trajectories: bool,
) -> dict[str, Any]:
    capture = BoundaryStateCapture(layers[boundary_layer])
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        boundary = capture.snapshot()
        all_residuals = residual_capture.snapshot()
    finally:
        capture.close()
        residual_capture.close()
    final_hidden = all_residuals[-1]
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": final_hidden,
        "residuals": all_residuals if collect_trajectories else [],
        "boundary_state": boundary,
    }


def _cell_readout_payload(
    *,
    cell_name: str,
    cell: dict[str, Any],
    host_variant: str,
    readouts: dict[str, ReadoutBundle],
    y_pt: int,
    y_it: int,
    collect_trajectories: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    native_readout_name = f"native_{host_variant}"
    for readout_name in ("common_it", "common_pt", native_readout_name):
        bundle = readouts[readout_name]
        if cell["final_hidden"] is None:
            logits = cell["logits"].clone().to(bundle.real_token_mask.device)
            logits[~bundle.real_token_mask.to(logits.device)] = float("-inf")
        else:
            logits = _logits_from_hidden(cell["final_hidden"], bundle)
        final_top1 = int(torch.argmax(logits).item())
        margin = float((logits[y_it] - logits[y_pt]).item())
        choice = "it" if final_top1 == y_it else ("pt" if final_top1 == y_pt else "other")
        payload = {
            "it_vs_pt_margin": margin,
            "it_logit": float(logits[y_it].item()),
            "pt_logit": float(logits[y_pt].item()),
            "final_top1_id": final_top1,
            "token_choice_class": choice,
            "it_rank": _rank(logits, y_it),
            "pt_rank": _rank(logits, y_pt),
        }
        if collect_trajectories and cell["residuals"]:
            payload["trajectory"] = _trajectory_payload(cell["residuals"], bundle)
        else:
            payload["trajectory"] = {
                "late_kl_mean": None,
                "remaining_adj_js": None,
                "future_top1_flips": None,
                "top5_churn": None,
                "top1_path_metrics": {},
            }
        out[readout_name] = payload
    out["cell"] = cell_name
    out["host_variant"] = host_variant
    out["patch_n"] = cell.get("patch_n", 0)
    out["patch_input_delta_max_abs"] = cell.get("patch_input_delta_max_abs")
    return out


def collect_manifest_record(
    *,
    model_name: str,
    manifest_record: dict[str, Any],
    dataset_record: dict[str, Any],
    prompt_mode: str,
    event_kinds: list[str],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    readouts: dict[str, ReadoutBundle],
    steering_adapter: Any,
    device: torch.device,
    collect_trajectories: bool,
    include_noop_patch: bool,
) -> dict[str, Any]:
    prompt_id = str(manifest_record.get("prompt_id"))
    boundary_layer = _late_boundary(model_name)
    pt_layers = steering_adapter.get_layers(models["pt"])
    it_layers = steering_adapter.get_layers(models["it"])
    layer_map = {"pt": pt_layers, "it": it_layers}
    adapter = steering_adapter.adapter
    raw_prompt = get_prompt_for_variant(
        dataset_record,
        variant="pt",
        tokenizer=tokenizers["pt"],
        apply_chat_template=False,
    )
    prompt_ids = tokenizers["pt"].encode(raw_prompt, add_special_tokens=True)

    events_out: dict[str, Any] = {}
    event_cache: dict[tuple[int, int, int], dict[str, Any]] = {}
    for kind, event in _unique_events(manifest_record, event_kinds):
        if "duplicate_of" in event:
            duplicate_of = str(event["duplicate_of"])
            events_out[kind] = {**events_out[duplicate_of], "duplicate_of": duplicate_of}
            continue

        y_pt = int(event["pt_token"]["token_id"])
        y_it = int(event["it_token"]["token_id"])
        prefix_ids = _prefix_ids_for_event(manifest_record, event)
        cache_key = (int(event["step"]), y_pt, y_it)
        if cache_key in event_cache:
            events_out[kind] = event_cache[cache_key]
            continue

        validation = _validate_tokenizers_and_tokens(
            model_name=model_name,
            prompt_mode=prompt_mode,
            dataset_record=dataset_record,
            prefix_ids=prefix_ids,
            y_pt=y_pt,
            y_it=y_it,
            tokenizers=tokenizers,
            readouts=readouts,
        )
        if not validation.get("ok"):
            payload = {
                "event": event,
                "valid": False,
                "validation": validation,
                "cells": {},
            }
            event_cache[cache_key] = payload
            events_out[kind] = payload
            continue

        full_ids = prompt_ids + prefix_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        baselines = {
            "pt": _baseline_forward_with_boundary(
                model=models["pt"],
                adapter=adapter,
                layers=pt_layers,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                collect_trajectories=collect_trajectories,
            ),
            "it": _baseline_forward_with_boundary(
                model=models["it"],
                adapter=adapter,
                layers=it_layers,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                collect_trajectories=collect_trajectories,
            ),
        }

        raw_cells: dict[str, dict[str, Any]] = {
            "U_PT__L_PT": baselines["pt"],
            "U_IT__L_IT": baselines["it"],
            "U_PT__L_IT": _forward_cell(
                model=models["it"],
                adapter=adapter,
                layers=it_layers,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                donor_boundary_state=baselines["pt"]["boundary_state"],
                collect_trajectories=collect_trajectories,
            ),
            "U_IT__L_PT": _forward_cell(
                model=models["pt"],
                adapter=adapter,
                layers=pt_layers,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                donor_boundary_state=baselines["it"]["boundary_state"],
                collect_trajectories=collect_trajectories,
            ),
        }
        if include_noop_patch:
            raw_cells["U_PT__L_PT_noop_patch"] = _forward_cell(
                model=models["pt"],
                adapter=adapter,
                layers=pt_layers,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                donor_boundary_state=baselines["pt"]["boundary_state"],
                collect_trajectories=False,
            )
            raw_cells["U_IT__L_IT_noop_patch"] = _forward_cell(
                model=models["it"],
                adapter=adapter,
                layers=it_layers,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                donor_boundary_state=baselines["it"]["boundary_state"],
                collect_trajectories=False,
            )

        cells = {
            name: _cell_readout_payload(
                cell_name=name,
                cell=payload,
                host_variant=CELL_SPECS[name].host_variant,
                readouts=readouts,
                y_pt=y_pt,
                y_it=y_it,
                collect_trajectories=collect_trajectories and name in RESIDUAL_CELLS,
            )
            for name, payload in raw_cells.items()
        }

        validations = {}
        if include_noop_patch:
            for base, noop in (
                ("U_PT__L_PT", "U_PT__L_PT_noop_patch"),
                ("U_IT__L_IT", "U_IT__L_IT_noop_patch"),
            ):
                check: dict[str, Any] = {
                    "patch_input_delta_max_abs": cells[noop].get("patch_input_delta_max_abs"),
                    "readouts": {},
                }
                for readout_name in ("common_it", "common_pt", f"native_{CELL_SPECS[base].host_variant}"):
                    base_readout = cells[base].get(readout_name)
                    noop_readout = cells[noop].get(readout_name)
                    if not isinstance(base_readout, dict) or not isinstance(noop_readout, dict):
                        continue
                    base_margin = base_readout["it_vs_pt_margin"]
                    noop_margin = noop_readout["it_vs_pt_margin"]
                    base_top1 = base_readout["final_top1_id"]
                    noop_top1 = noop_readout["final_top1_id"]
                    check["readouts"][readout_name] = {
                        "margin_abs_delta": abs(noop_margin - base_margin),
                        "top1_equal": bool(noop_top1 == base_top1),
                    }
                    if readout_name == "common_it":
                        # Backwards-compatible flat fields used by the analyzer.
                        check["common_it_margin_abs_delta"] = abs(noop_margin - base_margin)
                        check["common_it_top1_equal"] = bool(noop_top1 == base_top1)
                validations[f"{noop}_vs_{base}"] = check

        payload = {
            "event": event,
            "valid": True,
            "validation": validation,
            "boundary_layer": boundary_layer,
            "prefix_length": len(prefix_ids),
            "full_length": len(full_ids),
            "pt_token_id": y_pt,
            "it_token_id": y_it,
            "pt_token_text": validation.get("pt_token_text"),
            "it_token_text": validation.get("it_token_text"),
            "cells": cells,
            "noop_patch_checks": validations,
        }
        event_cache[cache_key] = payload
        events_out[kind] = payload

    return {
        "experiment": "exp23_midlate_interaction_suite",
        "part": "residual_factorial",
        "model": model_name,
        "prompt_id": prompt_id,
        "prompt_mode": prompt_mode,
        "events": events_out,
    }


def _make_readouts(
    *,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    steering_adapter: Any,
    real_token_masks: dict[str, torch.Tensor],
) -> dict[str, ReadoutBundle]:
    return {
        "common_it": ReadoutBundle(
            "common_it",
            "it",
            steering_adapter.get_final_norm(models["it"]),
            steering_adapter.get_lm_head(models["it"]),
            real_token_masks["it"],
        ),
        "common_pt": ReadoutBundle(
            "common_pt",
            "pt",
            steering_adapter.get_final_norm(models["pt"]),
            steering_adapter.get_lm_head(models["pt"]),
            real_token_masks["pt"],
        ),
        "native_pt": ReadoutBundle(
            "native_pt",
            "pt",
            steering_adapter.get_final_norm(models["pt"]),
            steering_adapter.get_lm_head(models["pt"]),
            real_token_masks["pt"],
        ),
        "native_it": ReadoutBundle(
            "native_it",
            "it",
            steering_adapter.get_final_norm(models["it"]),
            steering_adapter.get_lm_head(models["it"]),
            real_token_masks["it"],
        ),
    }


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = get_spec(args.model)
    steering_adapter = get_steering_adapter(args.model)
    pt_model, pt_tokenizer = load_model_and_tokenizer(model_id_for_variant(spec, "pt"), args.device, multi_gpu=spec.multi_gpu)
    it_model, it_tokenizer = load_model_and_tokenizer(model_id_for_variant(spec, "it"), args.device, multi_gpu=spec.multi_gpu)
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
    dataset_by_id = _dataset_lookup(args.dataset)
    manifest_rows = _load_manifest_records(
        exp20_root=args.exp20_root,
        exp20_fallback_root=args.exp20_fallback_root,
        prompt_mode=args.prompt_mode,
        model=args.model,
        n_examples=args.n_eval_examples,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"records_w{args.worker_index}.jsonl.gz"
    done = _done_prompt_ids(out_path)
    log.info("[exp23] residual %s/%s worker %d/%d resume=%d", args.prompt_mode, args.model, args.worker_index, args.n_workers, len(done))
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for idx, manifest_record in enumerate(manifest_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            if prompt_id in done:
                continue
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                log.warning("[exp23] missing dataset record for prompt_id=%s", prompt_id)
                continue
            try:
                result = collect_manifest_record(
                    model_name=args.model,
                    manifest_record=manifest_record,
                    dataset_record=dataset_record,
                    prompt_mode=args.prompt_mode,
                    event_kinds=args.event_kinds,
                    models=models,
                    tokenizers=tokenizers,
                    readouts=readouts,
                    steering_adapter=steering_adapter,
                    device=device,
                    collect_trajectories=args.collect_trajectories,
                    include_noop_patch=not args.no_noop_patch,
                )
                fout.write(json.dumps(result, separators=(",", ":")) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[exp23] residual prompt %s failed: %s", prompt_id, exc)
            if (idx + 1) % 5 == 0:
                log.info("[exp23] residual %s/%s %d/%d prompts", args.prompt_mode, args.model, idx + 1, len(manifest_rows))


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    seen: set[str] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(n_workers):
            path = out_dir / f"records_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp23] missing worker file %s", path)
                continue
            for row in _json_rows(path):
                prompt_id = str(row.get("prompt_id", ""))
                if prompt_id and prompt_id in seen:
                    continue
                if prompt_id:
                    seen.add(prompt_id)
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp23 residual-state x late-stack factorial records.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["raw_shared", "native"], default="raw_shared")
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=list(DEFAULT_EVENT_KINDS))
    parser.add_argument("--collect-trajectories", action="store_true")
    parser.add_argument("--no-noop-patch", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_workers(args.out_dir, args.n_workers)
        return
    run_worker(args)


if __name__ == "__main__":
    main()
