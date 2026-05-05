"""Collect forced-token continuations for Exp52.

For each first-divergence event, this collector forces the IT-preferred token,
the PT-preferred token, and matched alternate controls, then continues with the
native IT model. It deliberately does not construct hybrid rollouts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer, LogitsProcessor

from src.poc.cross_model.config import DATASET_PATH, MODEL_REGISTRY, get_spec, model_id_for_variant, revision_for_model_id
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    DEFAULT_EXP20_FALLBACK_ROOT,
    DEFAULT_EXP20_ROOT,
    _find_manifest,
    _prefix_ids_for_event,
    _unique_events,
)
from src.poc.exp52_forced_token_consequence_bridge import DEFAULT_LENGTHS
from src.poc.exp52_forced_token_consequence_bridge.validators import score_views

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class RealTokenMaskProcessor(LogitsProcessor):
    def __init__(self, mask: torch.Tensor):
        self.mask = mask.detach().bool()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = self.mask.to(scores.device)
        scores = scores.clone()
        scores[:, ~mask] = float("-inf")
        return scores


@dataclass
class PreparedEvent:
    row: dict[str, Any]
    prefix_ids: list[int]
    y_pt: int
    y_it: int
    max_new_tokens: int


def _json_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _append_jsonl_gz(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def _read_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(str(r.get("prompt_id")), str(r.get("event_kind"))) for r in _json_rows(path)}


def _dataset_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id", row.get("record_id"))): row for row in load_dataset(path)}


def _load_effect_lookup(exp47_root: Path, event_kind: str) -> dict[tuple[str, str], dict[str, Any]]:
    effects_path = exp47_root / "analysis" / "effects.csv"
    if not effects_path.exists():
        return {}
    import csv

    out: dict[tuple[str, str], dict[str, Any]] = {}
    with effects_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_kind") != event_kind or row.get("slice") != "full_1400" or row.get("readout") != "common_it":
                continue
            key = (str(row.get("model")), str(row.get("prompt_id")))
            out[key] = {
                "P_event": _safe_float(row.get("P")),
                "M_event": _safe_float(row.get("M")),
                "C_event": _safe_float(row.get("C")),
                "portable_share": _safe_float(row.get("portable_share_P_over_M")),
                "interaction_over_native_shift": _safe_float(row.get("interaction_over_native_shift")),
                "recipe_group": row.get("recipe_group"),
                "slice": row.get("slice"),
                "token_category": row.get("token_category"),
                "position_bin": row.get("position_bin"),
            }
    return out


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _token_category(text: str) -> str:
    if text == "":
        return "empty"
    if text.isspace():
        return "whitespace"
    stripped = text.strip()
    if "\n" in text or "\r" in text:
        return "newline"
    if not stripped:
        return "whitespace"
    if all(ch.isdigit() for ch in stripped):
        return "digit"
    if all(ch.isalpha() for ch in stripped):
        return "alpha"
    if all(ch.isalnum() for ch in stripped):
        return "alnum"
    if all(not ch.isalnum() and not ch.isspace() for ch in stripped):
        return "punct"
    return "mixed"


def _length_bucket(text: str) -> str:
    n = len(text)
    if n <= 1:
        return "1"
    if n <= 3:
        return "2_3"
    if n <= 6:
        return "4_6"
    return "7_plus"


def _roundtrip_ok(tokenizer: Any, token_id: int) -> bool:
    text = _decode_token(tokenizer, token_id)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids) == 1 and int(ids[0]) == int(token_id)


def _validate_event_tokens(
    *,
    tokenizer: Any,
    real_token_mask: torch.Tensor,
    y_pt: int,
    y_it: int,
) -> dict[str, Any]:
    vocab = int(real_token_mask.numel())
    specials = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
    reasons: list[str] = []
    for name, token_id in (("pt", y_pt), ("it", y_it)):
        if token_id < 0 or token_id >= vocab:
            reasons.append(f"{name}_id_out_of_vocab")
            continue
        if not bool(real_token_mask[int(token_id)].item()):
            reasons.append(f"{name}_not_real_token")
        if int(token_id) in specials:
            reasons.append(f"{name}_special_token")
        if not _roundtrip_ok(tokenizer, int(token_id)):
            reasons.append(f"{name}_roundtrip_not_single_token")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "pt_token_text": _decode_token(tokenizer, y_pt) if 0 <= y_pt < vocab else None,
        "it_token_text": _decode_token(tokenizer, y_it) if 0 <= y_it < vocab else None,
    }


def _rank_from_logits(logits: torch.Tensor, token_id: int) -> int | None:
    if token_id < 0 or token_id >= logits.numel():
        return None
    return int((logits > logits[int(token_id)]).sum().item()) + 1


def _choose_alt_token(
    *,
    log_probs: torch.Tensor,
    real_token_mask: torch.Tensor,
    tokenizer: Any,
    target_token: int,
    exclude: set[int],
    require_same_class: bool = False,
) -> int | None:
    if target_token < 0 or target_token >= log_probs.numel():
        return None
    target_lp = log_probs[int(target_token)]
    if not torch.isfinite(target_lp):
        return None
    mask = real_token_mask.detach().bool().to(log_probs.device).clone()
    for token_id in exclude:
        if 0 <= int(token_id) < mask.numel():
            mask[int(token_id)] = False
    specials = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
    for token_id in specials:
        if 0 <= int(token_id) < mask.numel():
            mask[int(token_id)] = False
    if require_same_class:
        target_text = _decode_token(tokenizer, int(target_token))
        target_class = _token_category(target_text)
        target_bucket = _length_bucket(target_text)
        ids = torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().tolist()
        keep: list[int] = []
        for token_id in ids:
            text = _decode_token(tokenizer, int(token_id))
            if _token_category(text) == target_class and _length_bucket(text) == target_bucket:
                keep.append(int(token_id))
        if not keep:
            return None
        candidate_ids = torch.tensor(keep, device=log_probs.device, dtype=torch.long)
        diffs = torch.abs(log_probs[candidate_ids] - target_lp)
        return int(candidate_ids[int(torch.argmin(diffs).item())].item())
    diffs = torch.abs(log_probs - target_lp)
    diffs[~mask] = float("inf")
    idx = int(torch.argmin(diffs).item())
    return idx if math.isfinite(float(diffs[idx].item())) else None


def _pad_left(
    rows: list[list[int]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for idx, ids in enumerate(rows):
        offset = max_len - len(ids)
        input_ids[idx, offset:] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[idx, offset:] = 1
    return input_ids, attention_mask


@torch.no_grad()
def _prefix_logits_batch(
    *,
    model: Any,
    prefix_ids: list[list[int]],
    pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    input_ids, attention_mask = _pad_left(prefix_ids, pad_id=pad_id, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return outputs.logits[:, -1, :].detach().float()


@torch.no_grad()
def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    start_rows: list[list[int]],
    real_token_mask: torch.Tensor,
    max_new_tokens: int,
    pad_id: int,
    eos_ids: list[int],
    device: torch.device,
) -> list[list[int]]:
    input_ids, attention_mask = _pad_left(start_rows, pad_id=pad_id, device=device)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        logits_processor=[RealTokenMaskProcessor(real_token_mask)],
        eos_token_id=[int(x) for x in eos_ids] if eos_ids else None,
        pad_token_id=int(pad_id),
        use_cache=True,
    )
    tails: list[list[int]] = []
    start_len = int(input_ids.shape[1])
    eos_set = set(int(x) for x in eos_ids)
    for row in generated:
        ids = [int(x) for x in row[start_len:].detach().cpu().tolist()]
        trimmed: list[int] = []
        for token_id in ids:
            if token_id == int(pad_id) and trimmed and trimmed[-1] in eos_set:
                break
            trimmed.append(token_id)
            if token_id in eos_set:
                break
        tails.append(trimmed)
    return tails


def _branch_payload(
    *,
    tokenizer: Any,
    forced_id: int,
    suffix_ids: list[int],
    record: dict[str, Any],
) -> dict[str, Any]:
    forced_text = _decode_token(tokenizer, int(forced_id))
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    including_ids = [int(forced_id), *[int(x) for x in suffix_ids]]
    including_text = tokenizer.decode(including_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {
        "forced_token_id": int(forced_id),
        "forced_token_text": forced_text,
        "suffix_token_ids": [int(x) for x in suffix_ids],
        "including_forced_token_ids": including_ids,
        "suffix_token_count": len(suffix_ids),
        "including_token_count": len(including_ids),
        "suffix_text": suffix_text,
        "including_forced_token_text": including_text,
        "scores": score_views(record, including_forced_token=including_text, suffix_only=suffix_text),
    }


def _max_new_for_category(category: str, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    return int(DEFAULT_LENGTHS.get(str(category), 96))


def _prepare_rows(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    real_token_mask: torch.Tensor,
    dataset_by_id: dict[str, dict[str, Any]],
    effects: dict[tuple[str, str], dict[str, Any]],
) -> list[PreparedEvent]:
    manifest_path = _find_manifest(args.exp20_root, args.exp20_fallback_root, args.prompt_mode, args.model)
    rows = list(_json_rows(manifest_path))
    if args.n_prompts is not None:
        rows = rows[: int(args.n_prompts)]
    rows = rows[int(args.worker_index) :: int(args.n_workers)]
    prepared: list[PreparedEvent] = []
    for manifest_record in rows:
        prompt_id = str(manifest_record.get("prompt_id"))
        dataset_record = dataset_by_id.get(prompt_id)
        if dataset_record is None:
            continue
        category = str(dataset_record.get("category") or "")
        if args.category_filter and category not in set(args.category_filter):
            continue
        for event_kind, event in _unique_events(manifest_record, [args.event_kind]):
            if event_kind != args.event_kind or "duplicate_of" in event:
                continue
            y_pt = int(event["pt_token"]["token_id"])
            y_it = int(event["it_token"]["token_id"])
            token_validation = _validate_event_tokens(
                tokenizer=tokenizer,
                real_token_mask=real_token_mask,
                y_pt=y_pt,
                y_it=y_it,
            )
            raw_prompt = get_prompt_for_variant(dataset_record, variant="pt", tokenizer=tokenizer, apply_chat_template=False)
            prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=True)
            prefix_generated = _prefix_ids_for_event(manifest_record, event)
            full_prefix = [int(x) for x in prompt_ids + prefix_generated]
            row = {
                "model": args.model,
                "prompt_id": prompt_id,
                "prompt_mode": args.prompt_mode,
                "event_kind": event_kind,
                "category": category,
                "source": dataset_record.get("source"),
                "position": int(event.get("step", 0)),
                "position_ge_3": bool(int(event.get("step", 0)) >= 3),
                "pt_token_id": y_pt,
                "it_token_id": y_it,
                "pt_token_text": token_validation.get("pt_token_text"),
                "it_token_text": token_validation.get("it_token_text"),
                "token_validation": token_validation,
                "valid": bool(token_validation.get("ok")),
                "invalid_reason": ";".join(token_validation.get("reasons") or []) or None,
                "raw_prompt": raw_prompt,
                "shared_prefix_token_ids": prefix_generated,
                "shared_prefix_text": tokenizer.decode(prefix_generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                "prefix_length_total": len(full_prefix),
                "max_new_tokens": _max_new_for_category(category, args.max_new_tokens_override),
                "dataset_metadata": {
                    "expected_behavior": (dataset_record.get("metadata") or {}).get("expected_behavior")
                    or dataset_record.get("expected_behavior"),
                    "expected_answer": dataset_record.get("expected_answer"),
                    "compliance_criteria": dataset_record.get("compliance_criteria"),
                },
                "mechanism": effects.get((args.model, prompt_id), {}),
            }
            if row["valid"]:
                prepared.append(
                    PreparedEvent(
                        row=row,
                        prefix_ids=full_prefix,
                        y_pt=y_pt,
                        y_it=y_it,
                        max_new_tokens=int(row["max_new_tokens"]),
                    )
                )
    log.info("[exp52] manifest %s prepared %d valid candidate events", manifest_path, len(prepared))
    return prepared


def _plausibility_payload(logits: torch.Tensor, log_probs: torch.Tensor, y_pt: int, y_it: int) -> dict[str, Any]:
    return {
        "rank_IT_t_PT": _rank_from_logits(logits, y_pt),
        "rank_IT_t_IT": _rank_from_logits(logits, y_it),
        "logprob_IT_t_PT": float(log_probs[int(y_pt)].item()),
        "logprob_IT_t_IT": float(log_probs[int(y_it)].item()),
        "logprob_gap_IT_tIT_minus_tPT": float((log_probs[int(y_it)] - log_probs[int(y_pt)]).item()),
        "plausible_rank50_or_gap5": bool(
            (_rank_from_logits(logits, y_pt) is not None and int(_rank_from_logits(logits, y_pt) or 10**9) <= 50)
            or float((log_probs[int(y_it)] - log_probs[int(y_pt)]).item()) <= 5.0
        ),
    }


def _process_batch(
    *,
    batch: list[PreparedEvent],
    model: Any,
    tokenizer: Any,
    real_token_mask: torch.Tensor,
    pad_id: int,
    eos_ids: list[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    logits = _prefix_logits_batch(
        model=model,
        prefix_ids=[event.prefix_ids for event in batch],
        pad_id=pad_id,
        device=device,
    )
    mask = real_token_mask.to(logits.device).bool()
    logits[:, ~mask] = float("-inf")
    log_probs = torch.log_softmax(logits, dim=-1)
    rows: list[dict[str, Any]] = []
    starts_by_len: dict[int, list[tuple[int, str, int, list[int]]]] = {}
    for idx, event in enumerate(batch):
        lp = log_probs[idx]
        row_logits = logits[idx]
        y_pt = event.y_pt
        y_it = event.y_it
        alt_rank = _choose_alt_token(
            log_probs=lp,
            real_token_mask=mask,
            tokenizer=tokenizer,
            target_token=y_pt,
            exclude={y_pt, y_it},
            require_same_class=False,
        )
        alt_class = _choose_alt_token(
            log_probs=lp,
            real_token_mask=mask,
            tokenizer=tokenizer,
            target_token=y_pt,
            exclude={y_pt, y_it, alt_rank if alt_rank is not None else -1},
            require_same_class=True,
        )
        branches = {
            "it_branch": y_it,
            "pt_branch": y_pt,
            "it_rank_matched_alt": alt_rank,
            "token_class_matched_alt": alt_class,
        }
        event.row["branch_plausibility"] = _plausibility_payload(row_logits, lp, y_pt, y_it)
        event.row["branch_token_classes"] = {
            name: {
                "token_id": int(token_id) if token_id is not None else None,
                "token_text": _decode_token(tokenizer, int(token_id)) if token_id is not None else None,
                "token_category": _token_category(_decode_token(tokenizer, int(token_id))) if token_id is not None else None,
                "length_bucket": _length_bucket(_decode_token(tokenizer, int(token_id))) if token_id is not None else None,
                "rank_IT": _rank_from_logits(row_logits, int(token_id)) if token_id is not None else None,
                "logprob_IT": float(lp[int(token_id)].item()) if token_id is not None else None,
            }
            for name, token_id in branches.items()
        }
        event.row["_branch_token_ids"] = branches
        for branch_name, token_id in branches.items():
            if token_id is None:
                continue
            starts_by_len.setdefault(event.max_new_tokens, []).append(
                (idx, branch_name, int(token_id), event.prefix_ids + [int(token_id)])
            )

    generated_by_event: dict[tuple[int, str], list[int]] = {}
    for max_new, entries in starts_by_len.items():
        for start in range(0, len(entries), 32):
            sub = entries[start : start + 32]
            tails = _generate_batch(
                model=model,
                tokenizer=tokenizer,
                start_rows=[x[3] for x in sub],
                real_token_mask=mask,
                max_new_tokens=max_new,
                pad_id=pad_id,
                eos_ids=eos_ids,
                device=device,
            )
            for (idx, branch_name, _token_id, _start_ids), tail in zip(sub, tails):
                generated_by_event[(idx, branch_name)] = tail

    for idx, event in enumerate(batch):
        row = dict(event.row)
        branch_ids = row.pop("_branch_token_ids")
        record_scores: dict[str, Any] = {}
        # Reconstruct a minimal dataset-like record for validators.
        validator_record = {
            "category": row.get("category"),
            "source": row.get("source"),
            "expected_answer": row.get("dataset_metadata", {}).get("expected_answer"),
            "answer_aliases": [],
            "metadata": {
                "expected_behavior": row.get("dataset_metadata", {}).get("expected_behavior"),
            },
            "expected_behavior": row.get("dataset_metadata", {}).get("expected_behavior"),
            "compliance_criteria": row.get("dataset_metadata", {}).get("compliance_criteria"),
        }
        for branch_name, token_id in branch_ids.items():
            if token_id is None:
                record_scores[branch_name] = {"available": False, "reason": "no_matched_token"}
                continue
            suffix = generated_by_event.get((idx, branch_name), [])
            record_scores[branch_name] = {
                "available": True,
                **_branch_payload(tokenizer=tokenizer, forced_id=int(token_id), suffix_ids=suffix, record=validator_record),
            }
        row["branches"] = record_scores
        rows.append(row)
    return rows


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = get_spec(args.model)
    steering_adapter = get_steering_adapter(args.model)
    it_id = model_id_for_variant(spec, "it")
    model, tokenizer = load_model_and_tokenizer(it_id, args.device, multi_gpu=spec.multi_gpu)
    tokenizer.padding_side = "left"
    pt_id = model_id_for_variant(spec, "pt")
    pt_revision = revision_for_model_id(pt_id)
    try:
        pt_tokenizer = AutoTokenizer.from_pretrained(
            pt_id,
            revision=pt_revision,
            trust_remote_code=True,
        )
        pt_tokenizer_audit_error = None
    except Exception as exc:
        # The paper-primary operation is exact forcing in the IT tokenizer
        # space. Some base anchors are gated even when the descendant tokenizer
        # is already loaded and tokenizer-compatible; do not let this audit-only
        # fetch kill the run.
        log.warning("[exp52] PT tokenizer audit failed for %s; using IT tokenizer audit fallback: %s", pt_id, exc)
        pt_tokenizer = tokenizer
        pt_tokenizer_audit_error = f"{type(exc).__name__}: {exc}"
    real_token_mask = steering_adapter.real_token_mask(tokenizer, device, model).detach().bool()
    dataset_by_id = _dataset_lookup(args.dataset)
    effects = _load_effect_lookup(args.exp47_root, args.event_kind)
    prepared = _prepare_rows(
        args=args,
        tokenizer=tokenizer,
        real_token_mask=real_token_mask,
        dataset_by_id=dataset_by_id,
        effects=effects,
    )
    if args.max_events and args.max_events > 0:
        prepared = prepared[: int(args.max_events)]
    out_path = args.out_dir / args.model / f"forced_token_records_w{args.worker_index}.jsonl.gz"
    done = _read_done(out_path) if args.resume else set()
    prepared = [event for event in prepared if (str(event.row["prompt_id"]), str(event.row["event_kind"])) not in done]
    log.info(
        "[exp52] model=%s worker=%d/%d events=%d done=%d batch_size=%d",
        args.model,
        args.worker_index,
        args.n_workers,
        len(prepared),
        len(done),
        args.batch_size,
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    eos_ids = set(steering_adapter.eos_token_ids(tokenizer))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    # Check tokenizer identity cheaply. A mismatch does not invalidate exact IT
    # forcing, but it is important audit metadata.
    tokenizer_match = {
        "pt_vocab_size": len(pt_tokenizer),
        "it_vocab_size": len(tokenizer),
        "same_vocab_size": len(pt_tokenizer) == len(tokenizer),
        "pt_tokenizer_audit_error": pt_tokenizer_audit_error,
    }
    for start in range(0, len(prepared), int(args.batch_size)):
        batch = prepared[start : start + int(args.batch_size)]
        try:
            rows = _process_batch(
                batch=batch,
                model=model,
                tokenizer=tokenizer,
                real_token_mask=real_token_mask,
                pad_id=int(pad_id),
                eos_ids=sorted(eos_ids),
                device=device,
            )
        except Exception as exc:
            log.exception("[exp52] batch failed model=%s worker=%d start=%d", args.model, args.worker_index, start)
            rows = [
                {
                    **event.row,
                    "valid": False,
                    "invalid_reason": type(exc).__name__,
                    "error": str(exc),
                }
                for event in batch
            ]
        for row in rows:
            row["tokenizer_audit"] = tokenizer_match
            _append_jsonl_gz(out_path, row)
        if (start // int(args.batch_size) + 1) % 5 == 0:
            log.info("[exp52] model=%s worker=%d wrote %d/%d", args.model, args.worker_index, start + len(batch), len(prepared))
    log.info("[exp52] done model=%s worker=%d output=%s", args.model, args.worker_index, out_path)


def merge_workers(out_dir: Path, model: str, n_workers: int) -> None:
    merged = out_dir / model / "forced_token_records.jsonl.gz"
    seen: set[tuple[str, str]] = set()
    merged.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(merged, "wt", encoding="utf-8") as handle:
        for worker in range(int(n_workers)):
            path = out_dir / model / f"forced_token_records_w{worker}.jsonl.gz"
            if not path.exists():
                log.warning("[exp52] missing worker file %s", path)
                continue
            for row in _json_rows(path):
                key = (str(row.get("prompt_id")), str(row.get("event_kind")))
                if key in seen:
                    continue
                seen.add(key)
                handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    log.info("[exp52] merged model=%s rows=%d -> %s", model, len(seen), merged)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--exp47-root", type=Path, default=Path("results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-prompts", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--event-kind", default="first_diff")
    parser.add_argument("--prompt-mode", choices=["raw_shared"], default="raw_shared")
    parser.add_argument("--category-filter", nargs="*", default=None)
    parser.add_argument("--max-new-tokens-override", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")


def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_workers(args.out_dir, args.model, args.n_workers)
        return
    run_worker(args)
