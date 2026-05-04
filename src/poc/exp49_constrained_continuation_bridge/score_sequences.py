"""Teacher-force Exp49 candidate sequences through residual-factorial cells."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.utils import load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.boundary import BoundaryStateCapture, BoundaryStatePatch
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    CELL_SPECS,
    RESIDUAL_CELLS,
    ReadoutBundle,
    _late_boundary,
    _make_readouts,
    _module_dtype,
)
from src.poc.exp49_constrained_continuation_bridge import DEFAULT_EVENT_KIND
from src.poc.exp49_constrained_continuation_bridge.common import (
    EXP49_DEFAULT_ROOT,
    append_jsonl_gz,
    json_rows,
    stable_hash,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEQUENCE_KINDS = ("desc_primary", "base_primary", "base_forced_desc", "desc_shuffled")


@dataclass
class FullLayerOutputCapture:
    layer: torch.nn.Module
    adapter: Any

    def __post_init__(self) -> None:
        self.hidden: torch.Tensor | None = None
        self.handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, _module, _args, output) -> None:
        hidden = self.adapter.residual_from_output(output)
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise RuntimeError(f"Expected final hidden [batch, seq, d], got {type(hidden)}")
        self.hidden = hidden.detach().clone()

    def snapshot(self) -> torch.Tensor:
        if self.hidden is None:
            raise RuntimeError("Final hidden state was not captured")
        return self.hidden.detach().clone()

    def close(self) -> None:
        self.handle.remove()


@torch.no_grad()
def _baseline_forward_full(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
) -> dict[str, Any]:
    boundary_capture = BoundaryStateCapture(layers[boundary_layer])
    final_capture = FullLayerOutputCapture(layers[-1], adapter)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        boundary = boundary_capture.snapshot()
        final_hidden = final_capture.snapshot()
    finally:
        boundary_capture.close()
        final_capture.close()
    return {"boundary_state": boundary, "final_hidden": final_hidden}


@torch.no_grad()
def _patched_forward_full(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    donor_boundary_state: torch.Tensor,
) -> dict[str, Any]:
    patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state)
    final_capture = FullLayerOutputCapture(layers[-1], adapter)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        final_hidden = final_capture.snapshot()
    finally:
        patcher.close()
        final_capture.close()
    return {
        "final_hidden": final_hidden,
        "patch_n": patcher.n_patches,
        "patch_input_delta_max_abs": patcher.last_max_abs_input_delta,
    }


def _pad_batch(
    *,
    prefix_ids: list[int],
    sequences: dict[str, list[int]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[str, int]]:
    names = list(sequences)
    max_len = max(len(prefix_ids) + len(ids) for ids in sequences.values())
    input_ids = torch.full((len(names), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    name_to_idx: dict[str, int] = {}
    actual_lengths: dict[str, int] = {}
    for row_idx, name in enumerate(names):
        ids = prefix_ids + sequences[name]
        input_ids[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_idx, : len(ids)] = 1
        name_to_idx[name] = row_idx
        actual_lengths[name] = len(ids)
    return input_ids, attention_mask, name_to_idx, actual_lengths


def _rank(logits: torch.Tensor, token_id: int) -> int | None:
    if token_id < 0 or token_id >= logits.shape[-1]:
        return None
    return int((logits > logits[token_id]).sum().item()) + 1


@torch.no_grad()
def _score_candidate_sequences(
    *,
    hidden: torch.Tensor,
    bundle: ReadoutBundle,
    sequences: dict[str, list[int]],
    name_to_idx: dict[str, int],
    prefix_length_total: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    norm_dtype = _module_dtype(bundle.final_norm)
    head_dtype = _module_dtype(bundle.lm_head)
    mask = bundle.real_token_mask.to(hidden.device)
    for seq_name, token_ids in sequences.items():
        row_idx = name_to_idx[seq_name]
        positions = [prefix_length_total - 1 + j for j in range(len(token_ids))]
        selected = hidden[row_idx, positions, :].to(device=hidden.device, dtype=norm_dtype)
        normed = bundle.final_norm(selected.view(len(token_ids), 1, -1)).view(len(token_ids), -1)
        logits = bundle.lm_head(normed.to(dtype=head_dtype)).float()
        logits[:, ~mask] = float("-inf")
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropies = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)
        top1_ids = torch.argmax(logits, dim=-1)
        out[seq_name] = {
            "token_ids": [int(x) for x in token_ids],
            "target_logprobs": [float(log_probs[j, int(tok)].item()) for j, tok in enumerate(token_ids)],
            "target_logits": [float(logits[j, int(tok)].item()) for j, tok in enumerate(token_ids)],
            "target_ranks": [_rank(logits[j], int(tok)) for j, tok in enumerate(token_ids)],
            "top1_ids": [int(x) for x in top1_ids.detach().cpu().tolist()],
            "top1_matches_target": [bool(int(top1_ids[j].item()) == int(tok)) for j, tok in enumerate(token_ids)],
            "entropy": [float(x) for x in entropies.detach().cpu().tolist()],
        }
    return out


def _row_group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("recipe_group") or "unknown"),
        str(row.get("category") or "unknown"),
        str(row.get("token_category") or "unknown"),
    )


def _attach_shuffled_desc_candidates(rows: list[dict[str, Any]]) -> None:
    """Add ``desc_shuffled`` = same first token plus matched foreign descendant tail."""

    valid = [row for row in rows if row.get("valid") and isinstance(row.get("candidates"), dict)]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in valid:
        groups.setdefault(_row_group_key(row), []).append(row)
    global_pool = sorted(valid, key=lambda r: (stable_hash(str(r.get("prompt_id"))), str(r.get("prompt_id"))))
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda r: (stable_hash(str(r.get("prompt_id"))), str(r.get("prompt_id"))))
        pool = ordered if len(ordered) > 1 else global_pool
        for idx, row in enumerate(ordered):
            donor = pool[(idx + 1) % len(pool)] if pool else row
            if donor.get("prompt_id") == row.get("prompt_id") and len(pool) > 1:
                donor = pool[(idx + 2) % len(pool)]
            donor_tail = (donor["candidates"]["desc_primary"]["token_ids"] or [])[1:]
            shuffled = [int(row["it_token_id"])] + [int(x) for x in donor_tail]
            row["candidates"]["desc_shuffled"] = {
                "token_ids": shuffled,
                "token_texts": [],
                "text": None,
                "shuffle_donor_prompt_id": donor.get("prompt_id"),
                "shuffle_group_key": list(_row_group_key(row)),
                "first_eos_position": None,
                "has_non_eos_special": False,
                "non_eos_special_positions": [],
            }


def score_candidate_row(
    *,
    row: dict[str, Any],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    readouts: dict[str, ReadoutBundle],
    steering_adapter: Any,
    device: torch.device,
    boundary_layer: int,
) -> dict[str, Any]:
    if not row.get("valid"):
        return {**row, "scored": False, "score_reason": "candidate_invalid"}
    adapter = steering_adapter.adapter
    layers = {
        "pt": steering_adapter.get_layers(models["pt"]),
        "it": steering_adapter.get_layers(models["it"]),
    }
    prefix_ids = [int(x) for x in row["prompt_token_ids"]] + [int(x) for x in row["prefix_generated_ids"]]
    sequences = {
        name: [int(x) for x in row["candidates"][name]["token_ids"]]
        for name in SEQUENCE_KINDS
        if name in row.get("candidates", {})
    }
    if "desc_primary" not in sequences or "base_primary" not in sequences or "base_forced_desc" not in sequences:
        return {**row, "scored": False, "score_reason": "missing_required_candidates"}

    pad_id = tokenizers["pt"].pad_token_id
    if pad_id is None:
        pad_id = tokenizers["pt"].eos_token_id if tokenizers["pt"].eos_token_id is not None else 0
    input_ids, attention_mask, name_to_idx, actual_lengths = _pad_batch(
        prefix_ids=prefix_ids,
        sequences=sequences,
        pad_id=int(pad_id),
        device=device,
    )

    baselines = {
        "pt": _baseline_forward_full(
            model=models["pt"],
            adapter=adapter,
            layers=layers["pt"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
        ),
        "it": _baseline_forward_full(
            model=models["it"],
            adapter=adapter,
            layers=layers["it"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
        ),
    }
    raw_cells = {
        "U_PT__L_PT": baselines["pt"],
        "U_IT__L_IT": baselines["it"],
        "U_PT__L_IT": _patched_forward_full(
            model=models["it"],
            adapter=adapter,
            layers=layers["it"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
            donor_boundary_state=baselines["pt"]["boundary_state"],
        ),
        "U_IT__L_PT": _patched_forward_full(
            model=models["pt"],
            adapter=adapter,
            layers=layers["pt"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
            donor_boundary_state=baselines["it"]["boundary_state"],
        ),
    }

    cells: dict[str, Any] = {}
    for cell_name in RESIDUAL_CELLS:
        host_variant = CELL_SPECS[cell_name].host_variant
        readout_names = ("common_it", "common_pt", f"native_{host_variant}")
        cells[cell_name] = {
            "host_variant": host_variant,
            "patch_n": raw_cells[cell_name].get("patch_n", 0),
            "patch_input_delta_max_abs": raw_cells[cell_name].get("patch_input_delta_max_abs"),
            "readouts": {},
        }
        for readout_name in readout_names:
            cells[cell_name]["readouts"][readout_name] = _score_candidate_sequences(
                hidden=raw_cells[cell_name]["final_hidden"],
                bundle=readouts[readout_name],
                sequences=sequences,
                name_to_idx=name_to_idx,
                prefix_length_total=int(row["prefix_length_total"]),
            )

    return {
        "model": row["model"],
        "prompt_id": row["prompt_id"],
        "prompt_mode": row.get("prompt_mode", "raw_shared"),
        "event_kind": row.get("event_kind", DEFAULT_EVENT_KIND),
        "valid": True,
        "scored": True,
        "boundary_layer": int(boundary_layer),
        "prefix_length_total": int(row["prefix_length_total"]),
        "actual_lengths": actual_lengths,
        "pt_token_id": int(row["pt_token_id"]),
        "it_token_id": int(row["it_token_id"]),
        "pt_token_text": row.get("pt_token_text"),
        "it_token_text": row.get("it_token_text"),
        "category": row.get("category"),
        "source": row.get("source"),
        "recipe_group": row.get("recipe_group"),
        "token_category": row.get("token_category"),
        "position": row.get("position"),
        "position_ge_3": row.get("position_ge_3"),
        "position_ge_5": row.get("position_ge_5"),
        "slices": row.get("slices", ["full_1400"]),
        "candidates": row.get("candidates", {}),
        "cells": cells,
    }


def _done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("model")), str(row.get("prompt_id")))
        for row in json_rows(path)
        if row.get("model") and row.get("prompt_id")
    }


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    candidate_paths = sorted((Path(args.candidates_dir) / args.model).glob("candidate_sequences_w*.jsonl.gz"))
    if not candidate_paths:
        raise FileNotFoundError(f"No candidate files found under {Path(args.candidates_dir) / args.model}")
    rows = [row for path in candidate_paths for row in json_rows(path)]
    rows = rows[: args.n_examples] if args.n_examples is not None else rows
    _attach_shuffled_desc_candidates(rows)
    rows = rows[args.worker_index :: args.n_workers]
    out_path = Path(args.output_dir) / args.model / f"sequence_scores_w{args.worker_index}.jsonl.gz"
    done = _done_keys(out_path) if args.resume else set()

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
    boundary_layer = int(args.boundary_layer) if args.boundary_layer is not None else _late_boundary(args.model)
    log.info(
        "[exp49 score] model=%s worker=%d/%d rows=%d done=%d boundary=%d out=%s",
        args.model,
        args.worker_index,
        args.n_workers,
        len(rows),
        len(done),
        boundary_layer,
        out_path,
    )
    for idx, row in enumerate(rows):
        key = (str(row.get("model")), str(row.get("prompt_id")))
        if key in done:
            continue
        try:
            scored = score_candidate_row(
                row=row,
                models=models,
                tokenizers=tokenizers,
                readouts=readouts,
                steering_adapter=steering_adapter,
                device=device,
                boundary_layer=boundary_layer,
            )
        except Exception as exc:  # keep long jobs resumable while preserving the failure.
            log.exception("[exp49 score] failed model=%s prompt=%s", args.model, row.get("prompt_id"))
            scored = {
                "model": args.model,
                "prompt_id": row.get("prompt_id"),
                "valid": False,
                "scored": False,
                "score_reason": type(exc).__name__,
                "score_error": str(exc),
            }
        append_jsonl_gz(out_path, scored)
        if (idx + 1) % 20 == 0:
            log.info("[exp49 score] model=%s worker=%d scored %d/%d", args.model, args.worker_index, idx + 1, len(rows))
    log.info("[exp49 score] done model=%s worker=%d output=%s", args.model, args.worker_index, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates-dir", default=str(EXP49_DEFAULT_ROOT / "debug" / "candidates"))
    parser.add_argument("--output-dir", default=str(EXP49_DEFAULT_ROOT / "debug" / "scores"))
    parser.add_argument("--event-kind", default=DEFAULT_EVENT_KIND)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--boundary-layer", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_worker(args)


if __name__ == "__main__":
    main()

