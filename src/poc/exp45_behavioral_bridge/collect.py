from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    CELL_SPECS,
    DEFAULT_EXP20_FALLBACK_ROOT,
    DEFAULT_EXP20_ROOT,
    DEFAULT_EVENT_KINDS,
    ReadoutBundle,
    _baseline_forward_with_boundary,
    _cell_readout_payload,
    _dataset_lookup,
    _find_manifest,
    _forward_cell,
    _json_rows,
    _late_boundary,
    _load_manifest_records,
    _logits_from_hidden,
    _rank,
    _token_text,
    _unique_events,
    _validate_tokenizers_and_tokens,
)
from src.poc.exp45_behavioral_bridge.metrics import CELL_ORDER, lexical_metrics


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _read_done_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    done: set[tuple[str, ...]] = set()
    for row in _json_rows(path):
        done.add(tuple(str(row.get(field, "")) for field in key_fields))
    return done


def _real_token_masks(steering_adapter: Any, tokenizers: dict[str, Any], models: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "pt": steering_adapter.real_token_mask(tokenizers["pt"], device, models["pt"]),
        "it": steering_adapter.real_token_mask(tokenizers["it"], device, models["it"]),
    }


def _make_readouts(
    *,
    models: dict[str, Any],
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


def _prefix_ids_for_event(manifest_record: dict[str, Any], event: dict[str, Any]) -> list[int]:
    step = int(event["step"])
    free_runs = manifest_record.get("free_runs") or {}
    base = free_runs.get("A_pt_raw") or {}
    return [int(token_id) for token_id in (base.get("generated_token_ids") or [])[:step]]


def _top_payload(logits: torch.Tensor, tokenizer: Any, y_pt: int, y_it: int, *, top_k: int = 20) -> dict[str, Any]:
    k = min(int(top_k), int(logits.numel()))
    vals, ids = torch.topk(logits, k=k)
    top_ids = [int(x) for x in ids.detach().cpu().tolist()]
    top_vals = [float(x) for x in vals.detach().cpu().tolist()]
    top1_id = top_ids[0]
    it_rank = _rank(logits, int(y_it))
    pt_rank = _rank(logits, int(y_pt))
    it_logit = float(logits[int(y_it)].item())
    pt_logit = float(logits[int(y_pt)].item())
    return {
        "margin_it_minus_pt": it_logit - pt_logit,
        "pairwise_it_win": bool(it_logit > pt_logit),
        "it_logit": it_logit,
        "pt_logit": pt_logit,
        "top1_id": top1_id,
        "top1_text": _token_text(tokenizer, top1_id),
        "top1_is_t_it": bool(top1_id == int(y_it)),
        "top1_is_t_pt": bool(top1_id == int(y_pt)),
        "top1_class": "it" if top1_id == int(y_it) else ("pt" if top1_id == int(y_pt) else "other"),
        "t_it_rank": it_rank,
        "t_pt_rank": pt_rank,
        "t_it_top5": bool(it_rank is not None and it_rank <= 5),
        "t_pt_top5": bool(pt_rank is not None and pt_rank <= 5),
        "t_it_gap_to_top1": float(it_logit - top_vals[0]),
        "top_ids": top_ids,
        "top_texts": [_token_text(tokenizer, token_id) for token_id in top_ids],
        "top_logits": top_vals,
    }


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = logits.detach().float().clone().to(mask.device)
    out[~mask.to(out.device)] = float("-inf")
    return out


@torch.no_grad()
def _one_step_cells(
    *,
    models: dict[str, Any],
    steering_adapter: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    collect_readouts: dict[str, ReadoutBundle],
) -> dict[str, dict[str, Any]]:
    adapter = steering_adapter.adapter
    pt_layers = steering_adapter.get_layers(models["pt"])
    it_layers = steering_adapter.get_layers(models["it"])
    baselines = {
        "pt": _baseline_forward_with_boundary(
            model=models["pt"],
            adapter=adapter,
            layers=pt_layers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
            collect_trajectories=False,
        ),
        "it": _baseline_forward_with_boundary(
            model=models["it"],
            adapter=adapter,
            layers=it_layers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
            collect_trajectories=False,
        ),
    }
    return {
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
            collect_trajectories=False,
        ),
        "U_IT__L_PT": _forward_cell(
            model=models["pt"],
            adapter=adapter,
            layers=pt_layers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
            donor_boundary_state=baselines["it"]["boundary_state"],
            collect_trajectories=False,
        ),
    }


def _cell_readout_metrics(
    *,
    cell_name: str,
    cell: dict[str, Any],
    readouts: dict[str, ReadoutBundle],
    tokenizers: dict[str, Any],
    y_pt: int,
    y_it: int,
    top_k: int,
) -> dict[str, Any]:
    host_variant = CELL_SPECS[cell_name].host_variant
    out: dict[str, Any] = {}
    for readout_name in ("common_it", "common_pt", f"native_{host_variant}"):
        bundle = readouts[readout_name]
        logits = _logits_from_hidden(cell["final_hidden"], bundle)
        out[readout_name] = _top_payload(logits, tokenizers[bundle.variant], y_pt, y_it, top_k=top_k)
    return out


@torch.no_grad()
def _next_logits_for_cell(
    *,
    cell_name: str,
    models: dict[str, Any],
    steering_adapter: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    real_token_masks: dict[str, torch.Tensor],
) -> torch.Tensor:
    spec = CELL_SPECS[cell_name]
    adapter = steering_adapter.adapter
    layer_map = {
        "pt": steering_adapter.get_layers(models["pt"]),
        "it": steering_adapter.get_layers(models["it"]),
    }
    if spec.patch_from_variant is None:
        outputs = models[spec.host_variant](input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return _masked_logits(outputs.logits[0, -1, :], real_token_masks[spec.host_variant])
    upstream = _baseline_forward_with_boundary(
        model=models[spec.upstream_variant],
        adapter=adapter,
        layers=layer_map[spec.upstream_variant],
        input_ids=input_ids,
        attention_mask=attention_mask,
        boundary_layer=boundary_layer,
        collect_trajectories=False,
    )
    patched = _forward_cell(
        model=models[spec.host_variant],
        adapter=adapter,
        layers=layer_map[spec.host_variant],
        input_ids=input_ids,
        attention_mask=attention_mask,
        boundary_layer=boundary_layer,
        donor_boundary_state=upstream["boundary_state"],
        collect_trajectories=False,
    )
    return _masked_logits(patched["logits"], real_token_masks[spec.host_variant])


@torch.no_grad()
def _rollout_cell(
    *,
    cell_name: str,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    steering_adapter: Any,
    start_ids: list[int],
    boundary_layer: int,
    real_token_masks: dict[str, torch.Tensor],
    y_pt: int,
    y_it: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    generated: list[int] = []
    per_step: list[dict[str, Any]] = []
    current = [int(x) for x in start_ids]
    late_variant = CELL_SPECS[cell_name].host_variant
    tokenizer = tokenizers[late_variant]
    eos_ids = set()
    for tok in (getattr(tokenizer, "eos_token_id", None), getattr(tokenizers["pt"], "eos_token_id", None), getattr(tokenizers["it"], "eos_token_id", None)):
        if tok is None:
            continue
        if isinstance(tok, (list, tuple, set)):
            eos_ids.update(int(x) for x in tok if x is not None)
        else:
            eos_ids.add(int(tok))

    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        logits = _next_logits_for_cell(
            cell_name=cell_name,
            models=models,
            steering_adapter=steering_adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
            real_token_masks=real_token_masks,
        )
        payload = _top_payload(logits, tokenizer, y_pt, y_it, top_k=5)
        next_id = int(payload["top1_id"])
        generated.append(next_id)
        current.append(next_id)
        per_step.append(
            {
                "step": step,
                "token_id": next_id,
                "token_text": _token_text(tokenizer, next_id),
                "margin_it_minus_pt": payload["margin_it_minus_pt"],
                "pairwise_it_win": payload["pairwise_it_win"],
                "top1_class": payload["top1_class"],
                "t_it_rank": payload["t_it_rank"],
                "t_pt_rank": payload["t_pt_rank"],
            }
        )
        if next_id in eos_ids:
            break

    continuation_text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    post_first_ids = generated[1:] if len(generated) > 1 else []
    post_first_text = tokenizer.decode(post_first_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {
        "generated_token_ids": generated,
        "generated_tokens": [_token_text(tokenizer, token_id) for token_id in generated],
        "generated_tokens_count": len(generated),
        "continuation_text": continuation_text,
        "post_first_continuation_text": post_first_text,
        "first_generated_token_id": generated[0] if generated else None,
        "first_generated_token_text": _token_text(tokenizer, generated[0]) if generated else None,
        "first_generated_is_t_it": bool(generated and generated[0] == int(y_it)),
        "first_generated_is_t_pt": bool(generated and generated[0] == int(y_pt)),
        "first_generated_class": (
            "it" if generated and generated[0] == int(y_it) else ("pt" if generated and generated[0] == int(y_pt) else "other")
        ),
        "per_step": per_step,
        **lexical_metrics(continuation_text, post_first_text=post_first_text),
    }


def _event_meta(
    *,
    model: str,
    prompt_mode: str,
    prompt_id: str,
    dataset_record: dict[str, Any],
    event_kind: str,
    event: dict[str, Any],
    boundary_layer: int,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt_mode": prompt_mode,
        "prompt_id": prompt_id,
        "category": dataset_record.get("category") or dataset_record.get("benchmark") or "",
        "event_kind": event_kind,
        "position": int(event.get("step", 0)),
        "position_ge_3": bool(int(event.get("step", 0)) >= 3),
        "boundary_layer": int(boundary_layer),
        "pt_token_id": int(event["pt_token"]["token_id"]),
        "it_token_id": int(event["it_token"]["token_id"]),
        "pt_token_text": validation.get("pt_token_text"),
        "it_token_text": validation.get("it_token_text"),
        "raw_prompt": get_prompt_for_variant(dataset_record, variant="pt", tokenizer=None, apply_chat_template=False),
    }


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = get_spec(args.model)
    steering_adapter = get_steering_adapter(args.model)
    pt_model, pt_tokenizer = load_model_and_tokenizer(model_id_for_variant(spec, "pt"), args.device, multi_gpu=spec.multi_gpu)
    it_model, it_tokenizer = load_model_and_tokenizer(model_id_for_variant(spec, "it"), args.device, multi_gpu=spec.multi_gpu)
    models = {"pt": pt_model, "it": it_model}
    tokenizers = {"pt": pt_tokenizer, "it": it_tokenizer}
    real_token_masks = _real_token_masks(steering_adapter, tokenizers, models, device)
    readouts = _make_readouts(models=models, steering_adapter=steering_adapter, real_token_masks=real_token_masks)
    dataset_by_id = _dataset_lookup(args.dataset)
    manifest_rows = _load_manifest_records(
        exp20_root=args.exp20_root,
        exp20_fallback_root=args.exp20_fallback_root,
        prompt_mode=args.prompt_mode,
        model=args.model,
        n_examples=args.n_prompts,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
    )
    boundary_layer = int(args.boundary_layer_override) if args.boundary_layer_override is not None else _late_boundary(args.model)
    prompt_path = _find_manifest(args.exp20_root, args.exp20_fallback_root, args.prompt_mode, args.model)
    log.info("[exp45] model=%s boundary=%d manifest=%s rows=%d", args.model, boundary_layer, prompt_path, len(manifest_rows))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    one_step_path = args.out_dir / f"one_step_records_w{args.worker_index}.jsonl.gz"
    rollout_path = args.out_dir / f"rollout_records_w{args.worker_index}.jsonl.gz"
    one_done = _read_done_keys(one_step_path, ("prompt_id", "event_kind"))
    rollout_done = _read_done_keys(rollout_path, ("prompt_id", "event_kind", "cell"))
    want_one = args.part in {"one_step", "both"}
    want_rollout = args.part in {"rollout", "both"}
    one_count = 0
    rollout_count = 0
    failures = 0
    with gzip.open(one_step_path, "at", encoding="utf-8") as one_f, gzip.open(rollout_path, "at", encoding="utf-8") as roll_f:
        event_seen = 0
        for idx, manifest_record in enumerate(manifest_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                log.warning("[exp45] missing dataset record prompt_id=%s", prompt_id)
                continue
            raw_prompt = get_prompt_for_variant(dataset_record, variant="pt", tokenizer=pt_tokenizer, apply_chat_template=False)
            prompt_ids = pt_tokenizer.encode(raw_prompt, add_special_tokens=True)
            for event_kind, event in _unique_events(manifest_record, args.event_kinds):
                if "duplicate_of" in event:
                    continue
                if event_seen >= args.max_events and args.max_events > 0:
                    break
                event_seen += 1
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
                    failures += 1
                    log.warning("[exp45] validation failed prompt=%s kind=%s reason=%s", prompt_id, event_kind, validation.get("reason"))
                    continue
                full_ids = prompt_ids + prefix_ids
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                meta = _event_meta(
                    model=args.model,
                    prompt_mode=args.prompt_mode,
                    prompt_id=prompt_id,
                    dataset_record=dataset_record,
                    event_kind=event_kind,
                    event=event,
                    boundary_layer=boundary_layer,
                    validation=validation,
                )
                meta["shared_prefix_token_ids"] = prefix_ids
                meta["shared_prefix_text"] = pt_tokenizer.decode(prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                try:
                    if want_one and (prompt_id, event_kind) not in one_done:
                        raw_cells = _one_step_cells(
                            models=models,
                            steering_adapter=steering_adapter,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            boundary_layer=boundary_layer,
                            collect_readouts=readouts,
                        )
                        cells = {
                            cell_name: _cell_readout_metrics(
                                cell_name=cell_name,
                                cell=raw_cells[cell_name],
                                readouts=readouts,
                                tokenizers=tokenizers,
                                y_pt=y_pt,
                                y_it=y_it,
                                top_k=args.top_k,
                            )
                            for cell_name in CELL_ORDER
                        }
                        one_f.write(json.dumps({**meta, "record_type": "one_step", "cells": cells}, separators=(",", ":")) + "\n")
                        one_f.flush()
                        one_count += 1
                    if want_rollout and event_seen <= args.rollout_events:
                        for cell_name in CELL_ORDER:
                            if (prompt_id, event_kind, cell_name) in rollout_done:
                                continue
                            rollout = _rollout_cell(
                                cell_name=cell_name,
                                models=models,
                                tokenizers=tokenizers,
                                steering_adapter=steering_adapter,
                                start_ids=full_ids,
                                boundary_layer=boundary_layer,
                                real_token_masks=real_token_masks,
                                y_pt=y_pt,
                                y_it=y_it,
                                max_new_tokens=args.max_new_tokens,
                                device=device,
                            )
                            roll_f.write(
                                json.dumps(
                                    {
                                        **meta,
                                        "record_type": "rollout",
                                        "cell": cell_name,
                                        "decode_mode": "greedy",
                                        "max_new_tokens": int(args.max_new_tokens),
                                        **rollout,
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            roll_f.flush()
                            rollout_count += 1
                except Exception as exc:
                    failures += 1
                    log.exception("[exp45] prompt=%s kind=%s failed: %s", prompt_id, event_kind, exc)
            if (idx + 1) % 5 == 0:
                log.info(
                    "[exp45] model=%s worker=%d/%d prompts=%d/%d one_step=%d rollouts=%d failures=%d",
                    args.model,
                    args.worker_index,
                    args.n_workers,
                    idx + 1,
                    len(manifest_rows),
                    one_count,
                    rollout_count,
                    failures,
                )
    log.info("[exp45] done model=%s one_step=%d rollouts=%d failures=%d", args.model, one_count, rollout_count, failures)


def merge_workers(out_dir: Path, n_workers: int) -> None:
    for stem, key_fields in (
        ("one_step_records", ("prompt_id", "event_kind")),
        ("rollout_records", ("prompt_id", "event_kind", "cell")),
    ):
        merged = out_dir / f"{stem}.jsonl.gz"
        seen: set[tuple[str, ...]] = set()
        with gzip.open(merged, "wt", encoding="utf-8") as fout:
            for worker_idx in range(int(n_workers)):
                path = out_dir / f"{stem}_w{worker_idx}.jsonl.gz"
                if not path.exists():
                    log.warning("[exp45] missing worker file %s", path)
                    continue
                for row in _json_rows(path):
                    key = tuple(str(row.get(field, "")) for field in key_fields)
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps(row, separators=(",", ":")) + "\n")
        log.info("[exp45] merged %s rows=%d -> %s", stem, len(seen), merged)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-prompts", type=int, default=600)
    parser.add_argument("--max-events", type=int, default=0, help="0 means no cap after prompt slicing")
    parser.add_argument("--rollout-events", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prompt-mode", choices=["raw_shared"], default="raw_shared")
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=["first_diff"])
    parser.add_argument("--part", choices=["one_step", "rollout", "both"], default="both")
    parser.add_argument("--boundary-layer-override", type=int, default=None)
    parser.add_argument("--merge-only", action="store_true")


def main(args: argparse.Namespace) -> None:
    if args.merge_only:
        merge_workers(args.out_dir, args.n_workers)
        return
    run_worker(args)

