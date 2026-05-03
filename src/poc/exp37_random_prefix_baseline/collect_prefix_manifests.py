"""Collect Exp20-compatible manifests for Exp37 matched-prefix baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer, read_done_ids
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp20_divergence_token_counterfactual.collect import _call_model, _decode_token, _masked_next_token
from src.poc.exp20_divergence_token_counterfactual.metrics import find_divergence_events
from src.poc.exp37_random_prefix_baseline import (
    ARM_SOURCE_KEYS,
    PREDIV_KEY,
    RANDOM_IT_KEY,
    RANDOM_PT_KEY,
    REFERENCE_KEY,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("id", record.get("record_id", "unknown")))


def _stable_rng(seed: int, *parts: str) -> random.Random:
    raw = "::".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _out_path(out_root: Path, key: str, model: str, worker_index: int) -> Path:
    return out_root / key / "raw_shared" / model / f"exp20_validation_records_w{worker_index}.jsonl"


def _qc_path(out_root: Path, model: str, worker_index: int) -> Path:
    return out_root / "_qc" / model / f"manifest_qc_w{worker_index}.jsonl"


def _done_by_key(out_root: Path, keys: list[str], model: str) -> dict[str, set[str]]:
    done: dict[str, set[str]] = {}
    for key in keys:
        model_dir = out_root / key / "raw_shared" / model
        ids: set[str] = set()
        for path in sorted(model_dir.glob("exp20_validation_records_w*.jsonl")):
            ids.update(read_done_ids(path))
        done[key] = ids
    return done


def _selected_keys(arms: list[str], prefix_sources: list[str]) -> list[str]:
    keys: list[str] = []
    if "first_diff_reference" in arms:
        keys.append(REFERENCE_KEY)
    if "random_local_disagreement" in arms:
        if "pt_rollout" in prefix_sources:
            keys.append(RANDOM_PT_KEY)
        if "it_rollout" in prefix_sources:
            keys.append(RANDOM_IT_KEY)
    if "prediv_future_pair" in arms and "shared_prediv" in prefix_sources:
        keys.append(PREDIV_KEY)
    return keys


@torch.no_grad()
def _generate_rollout(
    *,
    model_name: str,
    model: Any,
    tokenizer: Any,
    real_token_mask: torch.Tensor,
    steering_adapter: Any,
    prompt_ids: list[int],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    past_key_values = None
    total_len = len(prompt_ids)
    stop_ids = set(steering_adapter.eos_token_ids(tokenizer))
    generated_ids: list[int] = []
    generated_tokens: list[dict[str, Any]] = []
    for _ in range(max_new_tokens):
        outputs = _call_model(
            model_name,
            model,
            input_ids,
            attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_id = _masked_next_token(outputs.logits[0, -1, :], real_token_mask)
        generated_ids.append(int(next_id))
        generated_tokens.append(_decode_token(tokenizer, int(next_id)))
        if int(next_id) in stop_ids:
            break
        input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        total_len += 1
        attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)
    return {
        "generated_token_ids": generated_ids,
        "generated_tokens": generated_tokens,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "n_steps": len(generated_ids),
    }


def _logit_stats(logits: torch.Tensor, real_token_mask: torch.Tensor) -> dict[str, Any]:
    masked = logits.detach().float().clone()
    masked[~real_token_mask.to(masked.device)] = float("-inf")
    top_vals, top_ids = torch.topk(masked, k=2)
    log_probs = torch.log_softmax(masked, dim=-1)
    probs = torch.exp(log_probs)
    finite = torch.isfinite(log_probs)
    entropy = -(probs[finite] * log_probs[finite]).sum()
    return {
        "top1_token_id": int(top_ids[0].item()),
        "top2_token_id": int(top_ids[1].item()),
        "top1_logit": float(top_vals[0].item()),
        "top2_logit": float(top_vals[1].item()),
        "top1_confidence": float(probs[top_ids[0]].item()),
        "top1_top2_margin": float((top_vals[0] - top_vals[1]).item()),
        "entropy": float(entropy.item()),
    }


@torch.no_grad()
def _score_prefixes(
    *,
    model: Any,
    tokenizer: Any,
    real_token_mask: torch.Tensor,
    prompt_ids: list[int],
    prefixes: list[list[int]],
    device: torch.device,
) -> list[dict[str, Any]]:
    if not prefixes:
        return []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    sequences = [prompt_ids + [int(x) for x in prefix] for prefix in prefixes]
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    last_indices: list[int] = []
    for idx, seq in enumerate(sequences):
        input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[idx, : len(seq)] = 1
        last_indices.append(len(seq) - 1)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    out: list[dict[str, Any]] = []
    for idx, last_idx in enumerate(last_indices):
        out.append(_logit_stats(outputs.logits[idx, last_idx, :], real_token_mask))
    return out


def _free_run_payload(name: str, prefix_ids: list[int], tokenizer: Any, prompt_mode: str) -> dict[str, Any]:
    host_variant = "pt" if name == "A_pt_raw" else "it"
    return {
        "condition": name,
        "host_variant": host_variant,
        "donor_variant": None,
        "graft_kind": None,
        "graft_window": None,
        "prompt_mode": prompt_mode,
        "n_steps": len(prefix_ids),
        "generated_token_ids": [int(x) for x in prefix_ids],
        "generated_tokens": [_decode_token(tokenizer, int(x)) for x in prefix_ids],
        "generated_text": tokenizer.decode(prefix_ids, skip_special_tokens=True),
        "exp37_prefix_stub": True,
    }


def _event_with_metadata(
    *,
    kind: str,
    step: int,
    pt_token: dict[str, Any],
    it_token: dict[str, Any],
    baseline_kind: str,
    prefix_source: str,
    is_true_first_divergence: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "step": int(step),
        "pt_token": pt_token,
        "it_token": it_token,
        "baseline_kind": baseline_kind,
        "prefix_source": prefix_source,
        "is_true_first_divergence": bool(is_true_first_divergence),
        **metadata,
    }


def make_manifest_record(
    *,
    prompt_id: str,
    model_name: str,
    max_new_tokens: int,
    prefix_ids_for_exp23: list[int],
    event: dict[str, Any],
    tokenizer: Any,
    source_prefix_ids: list[int] | None = None,
) -> dict[str, Any]:
    prefix_ids_for_exp23 = [int(x) for x in prefix_ids_for_exp23]
    source_prefix_ids = [int(x) for x in (source_prefix_ids if source_prefix_ids is not None else prefix_ids_for_exp23)]
    return {
        "prompt_id": prompt_id,
        "model": model_name,
        "prompt_mode": "raw_shared",
        "max_new_tokens": int(max_new_tokens),
        "free_runs": {
            "A_pt_raw": _free_run_payload("A_pt_raw", prefix_ids_for_exp23, tokenizer, "raw_shared"),
            "C_it_chat": _free_run_payload("C_it_chat", source_prefix_ids, tokenizer, "raw_shared"),
        },
        "divergence_events": {"first_diff": event},
        "readouts": {},
        "validation_conditions": [],
        "exp37_manifest": True,
    }


def _first_diff_event(
    *,
    events: dict[str, Any],
    pt_scores: dict[str, Any],
    it_scores: dict[str, Any],
    tokenizer: Any,
    prompt_id: str,
) -> dict[str, Any] | None:
    event = events.get("first_diff")
    if not isinstance(event, dict):
        return None
    y_pt = int(event["pt_token"]["token_id"])
    y_it = int(event["it_token"]["token_id"])
    return _event_with_metadata(
        kind="first_diff",
        step=int(event["step"]),
        pt_token=_decode_token(tokenizer, y_pt),
        it_token=_decode_token(tokenizer, y_it),
        baseline_kind="first_diff_reference",
        prefix_source="shared_first_diff",
        is_true_first_divergence=True,
        metadata={
            "selection_rule": "true first shared-history PT/IT top1 disagreement",
            "original_prompt_id": prompt_id,
            "sampled_prefix_step": int(event["step"]),
            "source_position": int(event["step"]),
            "pt_entropy": pt_scores.get("entropy"),
            "it_entropy": it_scores.get("entropy"),
            "pt_confidence": pt_scores.get("top1_confidence"),
            "it_confidence": it_scores.get("top1_confidence"),
            "pt_top1_top2_margin": pt_scores.get("top1_top2_margin"),
            "it_top1_top2_margin": it_scores.get("top1_top2_margin"),
            "token_category_pair": (
                f"{event['pt_token'].get('token_category_collapsed')}->"
                f"{event['it_token'].get('token_category_collapsed')}"
            ),
        },
    )


def _score_single_prefix(
    *,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    prompt_ids: list[int],
    prefix_ids: list[int],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pt_score = _score_prefixes(
        model=models["pt"],
        tokenizer=tokenizers["pt"],
        real_token_mask=real_token_masks["pt"],
        prompt_ids=prompt_ids,
        prefixes=[prefix_ids],
        device=device,
    )[0]
    it_score = _score_prefixes(
        model=models["it"],
        tokenizer=tokenizers["it"],
        real_token_mask=real_token_masks["it"],
        prompt_ids=prompt_ids,
        prefixes=[prefix_ids],
        device=device,
    )[0]
    return pt_score, it_score


def _candidate_steps(source_ids: list[int], first_step: int | None, rng: random.Random, k: int) -> list[int]:
    candidates = list(range(1, len(source_ids) + 1))
    if first_step is not None:
        candidates = [step for step in candidates if step != int(first_step)]
    if not candidates:
        return []
    rng.shuffle(candidates)
    return candidates[: min(k, len(candidates))]


def _random_local_record(
    *,
    key: str,
    source_ids: list[int],
    first_step: int | None,
    prompt_id: str,
    model_name: str,
    max_new_tokens: int,
    prompt_ids: list[int],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    device: torch.device,
    rng: random.Random,
    k_candidates: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source_name = "pt_rollout" if key == RANDOM_PT_KEY else "it_rollout"
    steps = _candidate_steps(source_ids, first_step, rng, k_candidates)
    qc = {
        "key": key,
        "prompt_id": prompt_id,
        "status": "missing_no_local_disagreement",
        "n_candidates": len(steps),
    }
    if not steps:
        qc["status"] = "missing_too_short"
        return None, qc
    prefixes = [[int(x) for x in source_ids[:step]] for step in steps]
    pt_scores = _score_prefixes(
        model=models["pt"],
        tokenizer=tokenizers["pt"],
        real_token_mask=real_token_masks["pt"],
        prompt_ids=prompt_ids,
        prefixes=prefixes,
        device=device,
    )
    it_scores = _score_prefixes(
        model=models["it"],
        tokenizer=tokenizers["it"],
        real_token_mask=real_token_masks["it"],
        prompt_ids=prompt_ids,
        prefixes=prefixes,
        device=device,
    )
    for step, prefix_ids, pt_score, it_score in zip(steps, prefixes, pt_scores, it_scores, strict=True):
        y_pt = int(pt_score["top1_token_id"])
        y_it = int(it_score["top1_token_id"])
        if y_pt == y_it:
            continue
        pt_token = _decode_token(tokenizers["pt"], y_pt)
        it_token = _decode_token(tokenizers["pt"], y_it)
        event = _event_with_metadata(
            kind="first_diff",
            step=int(step),
            pt_token=pt_token,
            it_token=it_token,
            baseline_kind="random_local_disagreement",
            prefix_source=source_name,
            is_true_first_divergence=False,
            metadata={
                "selection_rule": "random matched prefix with local PT/IT top1 disagreement",
                "original_prompt_id": prompt_id,
                "source_position": int(step),
                "sampled_prefix_step": int(step),
                "sampled_generated_token_ids": [int(x) for x in prefix_ids],
                "pt_entropy": pt_score.get("entropy"),
                "it_entropy": it_score.get("entropy"),
                "pt_confidence": pt_score.get("top1_confidence"),
                "it_confidence": it_score.get("top1_confidence"),
                "pt_top1_top2_margin": pt_score.get("top1_top2_margin"),
                "it_top1_top2_margin": it_score.get("top1_top2_margin"),
                "token_category_pair": (
                    f"{pt_token.get('token_category_collapsed')}->{it_token.get('token_category_collapsed')}"
                ),
            },
        )
        record = make_manifest_record(
            prompt_id=prompt_id,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            prefix_ids_for_exp23=prefix_ids,
            event=event,
            tokenizer=tokenizers["pt"],
            source_prefix_ids=prefix_ids,
        )
        qc.update({"status": "ok", "sampled_prefix_step": int(step), "pt_token_id": y_pt, "it_token_id": y_it})
        return record, qc
    return None, qc


def _prediv_record(
    *,
    first_event: dict[str, Any] | None,
    pt_ids: list[int],
    prompt_id: str,
    model_name: str,
    max_new_tokens: int,
    prompt_ids: list[int],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    device: torch.device,
    rng: random.Random,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    qc = {"key": PREDIV_KEY, "prompt_id": prompt_id, "status": "missing_too_early"}
    if not isinstance(first_event, dict):
        qc["status"] = "missing_no_first_diff"
        return None, qc
    s = int(first_event["step"])
    if s < 3:
        return None, qc
    j = rng.choice(list(range(0, s - 1)))
    prefix_ids = [int(x) for x in pt_ids[:j]]
    pt_score, it_score = _score_single_prefix(
        models=models,
        tokenizers=tokenizers,
        real_token_masks=real_token_masks,
        prompt_ids=prompt_ids,
        prefix_ids=prefix_ids,
        device=device,
    )
    local_agree = int(pt_score["top1_token_id"]) == int(it_score["top1_token_id"])
    y_pt = int(first_event["pt_token"]["token_id"])
    y_it = int(first_event["it_token"]["token_id"])
    pt_token = _decode_token(tokenizers["pt"], y_pt)
    it_token = _decode_token(tokenizers["pt"], y_it)
    event = _event_with_metadata(
        kind="first_diff",
        step=int(j),
        pt_token=pt_token,
        it_token=it_token,
        baseline_kind="prediv_future_pair",
        prefix_source="shared_prediv",
        is_true_first_divergence=False,
        metadata={
            "selection_rule": "pre-divergence agreement prefix scored on future first-divergence token pair",
            "original_prompt_id": prompt_id,
            "sampled_prefix_step": int(j),
            "future_first_diff_step": int(s),
            "source_position": int(j),
            "local_agreement_token_id": int(pt_score["top1_token_id"]) if local_agree else None,
            "local_pt_top1": int(pt_score["top1_token_id"]),
            "local_it_top1": int(it_score["top1_token_id"]),
            "local_agreement_verified": bool(local_agree),
            "pt_entropy": pt_score.get("entropy"),
            "it_entropy": it_score.get("entropy"),
            "pt_confidence": pt_score.get("top1_confidence"),
            "it_confidence": it_score.get("top1_confidence"),
            "pt_top1_top2_margin": pt_score.get("top1_top2_margin"),
            "it_top1_top2_margin": it_score.get("top1_top2_margin"),
            "token_category_pair": (
                f"{pt_token.get('token_category_collapsed')}->{it_token.get('token_category_collapsed')}"
            ),
        },
    )
    record = make_manifest_record(
        prompt_id=prompt_id,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        prefix_ids_for_exp23=prefix_ids,
        event=event,
        tokenizer=tokenizers["pt"],
        source_prefix_ids=prefix_ids,
    )
    qc.update(
        {
            "status": "ok" if local_agree else "bad_local_disagreement",
            "sampled_prefix_step": int(j),
            "future_first_diff_step": int(s),
            "local_agreement_verified": bool(local_agree),
        }
    )
    return (record if local_agree else None), qc


def _write_jsonl_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def collect_worker(args: argparse.Namespace) -> None:
    keys = _selected_keys(args.arms, args.prefix_sources)
    if not keys:
        raise ValueError("No Exp37 manifest keys selected; check --arms and --prefix-sources")
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
    records = load_dataset(
        args.dataset,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        n_examples=args.n_eval_examples,
    )
    done = _done_by_key(args.out_root, keys, args.model)
    qc_file = _qc_path(args.out_root, args.model, args.worker_index)
    qc_file.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "[exp37] collect model=%s worker=%d/%d rows=%d keys=%s",
        args.model,
        args.worker_index,
        args.n_workers,
        len(records),
        ",".join(keys),
    )
    for idx, record in enumerate(records):
        prompt_id = _record_id(record)
        try:
            raw_prompt = get_prompt_for_variant(
                record,
                variant="pt",
                tokenizer=pt_tokenizer,
                apply_chat_template=False,
            )
            pt_prompt_ids = pt_tokenizer.encode(raw_prompt, add_special_tokens=True)
            it_prompt_ids = it_tokenizer.encode(raw_prompt, add_special_tokens=True)
            if pt_prompt_ids != it_prompt_ids:
                _write_jsonl_line(
                    qc_file,
                    {
                        "prompt_id": prompt_id,
                        "status": "bad_tokenization",
                        "pt_prompt_len": len(pt_prompt_ids),
                        "it_prompt_len": len(it_prompt_ids),
                    },
                )
                continue
            pt_rollout = _generate_rollout(
                model_name=args.model,
                model=pt_model,
                tokenizer=pt_tokenizer,
                real_token_mask=real_token_masks["pt"],
                steering_adapter=steering_adapter,
                prompt_ids=pt_prompt_ids,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            it_rollout = _generate_rollout(
                model_name=args.model,
                model=it_model,
                tokenizer=it_tokenizer,
                real_token_mask=real_token_masks["it"],
                steering_adapter=steering_adapter,
                prompt_ids=it_prompt_ids,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            events = find_divergence_events(pt_rollout["generated_tokens"], it_rollout["generated_tokens"])
            first_event = events.get("first_diff")
            first_step = int(first_event["step"]) if isinstance(first_event, dict) else None

            if REFERENCE_KEY in keys and prompt_id not in done.get(REFERENCE_KEY, set()) and isinstance(first_event, dict):
                prefix_ids = [int(x) for x in pt_rollout["generated_token_ids"][: int(first_event["step"])]]
                pt_score, it_score = _score_single_prefix(
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    prompt_ids=pt_prompt_ids,
                    prefix_ids=prefix_ids,
                    device=device,
                )
                event = _first_diff_event(
                    events=events,
                    pt_scores=pt_score,
                    it_scores=it_score,
                    tokenizer=pt_tokenizer,
                    prompt_id=prompt_id,
                )
                if event is not None:
                    manifest = make_manifest_record(
                        prompt_id=prompt_id,
                        model_name=args.model,
                        max_new_tokens=args.max_new_tokens,
                        prefix_ids_for_exp23=prefix_ids,
                        event=event,
                        tokenizer=pt_tokenizer,
                        source_prefix_ids=prefix_ids,
                    )
                    _write_jsonl_line(_out_path(args.out_root, REFERENCE_KEY, args.model, args.worker_index), manifest)
                    _write_jsonl_line(qc_file, {"key": REFERENCE_KEY, "prompt_id": prompt_id, "status": "ok"})

            random_specs = []
            if RANDOM_PT_KEY in keys and prompt_id not in done.get(RANDOM_PT_KEY, set()):
                random_specs.append((RANDOM_PT_KEY, pt_rollout["generated_token_ids"]))
            if RANDOM_IT_KEY in keys and prompt_id not in done.get(RANDOM_IT_KEY, set()):
                random_specs.append((RANDOM_IT_KEY, it_rollout["generated_token_ids"]))
            for key, source_ids in random_specs:
                rng = _stable_rng(args.seed, args.model, prompt_id, key)
                manifest, qc = _random_local_record(
                    key=key,
                    source_ids=source_ids,
                    first_step=first_step,
                    prompt_id=prompt_id,
                    model_name=args.model,
                    max_new_tokens=args.max_new_tokens,
                    prompt_ids=pt_prompt_ids,
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    device=device,
                    rng=rng,
                    k_candidates=args.k_candidate_prefixes,
                )
                if manifest is not None:
                    _write_jsonl_line(_out_path(args.out_root, key, args.model, args.worker_index), manifest)
                _write_jsonl_line(qc_file, qc)

            if PREDIV_KEY in keys and prompt_id not in done.get(PREDIV_KEY, set()):
                rng = _stable_rng(args.seed, args.model, prompt_id, PREDIV_KEY)
                manifest, qc = _prediv_record(
                    first_event=first_event,
                    pt_ids=pt_rollout["generated_token_ids"],
                    prompt_id=prompt_id,
                    model_name=args.model,
                    max_new_tokens=args.max_new_tokens,
                    prompt_ids=pt_prompt_ids,
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    device=device,
                    rng=rng,
                )
                if manifest is not None:
                    _write_jsonl_line(_out_path(args.out_root, PREDIV_KEY, args.model, args.worker_index), manifest)
                _write_jsonl_line(qc_file, qc)
        except Exception as exc:
            log.exception("[exp37] prompt %s failed: %s", prompt_id, exc)
            _write_jsonl_line(qc_file, {"prompt_id": prompt_id, "status": "exception", "error": str(exc)})
        if (idx + 1) % 10 == 0:
            log.info("[exp37] collect %s worker=%d %d/%d", args.model, args.worker_index, idx + 1, len(records))


def _merge_jsonl(paths: list[Path], out_path: Path, *, dedup_prompt_id: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for path in paths:
            if not path.exists():
                log.warning("[exp37] missing worker file %s", path)
                continue
            with path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    prompt_id = str(payload.get("prompt_id", ""))
                    if dedup_prompt_id and prompt_id and prompt_id in seen:
                        continue
                    if dedup_prompt_id and prompt_id:
                        seen.add(prompt_id)
                    fout.write(json.dumps(payload, separators=(",", ":")) + "\n")
                    n += 1
    return n


def merge_outputs(args: argparse.Namespace) -> None:
    keys = _selected_keys(args.arms, args.prefix_sources)
    for key in keys:
        paths = [
            _out_path(args.out_root, key, args.model, worker_idx)
            for worker_idx in range(args.n_workers)
        ]
        out_path = args.out_root / key / "raw_shared" / args.model / "exp20_validation_records.jsonl"
        n = _merge_jsonl(paths, out_path, dedup_prompt_id=True)
        log.info("[exp37] merged %s/%s -> %s rows=%d", key, args.model, out_path, n)
    qc_paths = [_qc_path(args.out_root, args.model, worker_idx) for worker_idx in range(args.n_workers)]
    qc_out = args.out_root / "_qc" / args.model / "manifest_qc.jsonl"
    n_qc = _merge_jsonl(qc_paths, qc_out, dedup_prompt_id=False)
    log.info("[exp37] merged qc %s rows=%d", qc_out, n_qc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--k-candidate-prefixes", type=int, default=8)
    parser.add_argument(
        "--arms",
        nargs="*",
        default=["first_diff_reference", "random_local_disagreement", "prediv_future_pair"],
        choices=["first_diff_reference", "random_local_disagreement", "prediv_future_pair"],
    )
    parser.add_argument(
        "--prefix-sources",
        nargs="*",
        default=["pt_rollout", "it_rollout", "shared_prediv"],
        choices=["pt_rollout", "it_rollout", "shared_prediv"],
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_outputs(args)
        return
    collect_worker(args)


if __name__ == "__main__":
    main()
