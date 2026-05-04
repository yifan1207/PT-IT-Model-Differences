"""Construct fixed candidate continuations for Exp49.

For each Exp47 first-divergence event we build three short sequences:

``desc_primary``
    ``[t_F]`` plus descendant-greedy tail.
``base_primary``
    ``[t_B]`` plus base-greedy tail.
``base_forced_desc``
    ``[t_F]`` plus base-greedy tail after the descendant token is forced.

The scorer later teacher-forces these sequences through the four residual
factorial cells. Candidate construction is separated so scoring can be retried
without recomputing generation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessor

from src.poc.cross_model.config import DATASET_PATH, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.residual_factorial import _prefix_ids_for_event
from src.poc.exp49_constrained_continuation_bridge import DEFAULT_EVENT_KIND, DEFAULT_MAX_TAIL
from src.poc.exp49_constrained_continuation_bridge.common import (
    EXP47_DEFAULT_ROOT,
    EXP49_DEFAULT_ROOT,
    append_jsonl_gz,
    find_exp20_manifest,
    json_rows,
    load_exp47_event_metadata,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class RealTokenMaskProcessor(LogitsProcessor):
    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = self.mask.to(device=scores.device)
        scores = scores.clone()
        scores[:, ~mask] = float("-inf")
        return scores


def _dataset_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id", row.get("record_id"))): row for row in load_dataset(path)}


def _done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(row.get("model")), str(row.get("prompt_id")))
        for row in json_rows(path)
        if row.get("model") and row.get("prompt_id")
    }


def _token_text(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [
        tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def _special_flags(tokenizer: Any, token_ids: list[int], eos_ids: set[int]) -> dict[str, Any]:
    specials = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
    eos_positions = [idx for idx, token_id in enumerate(token_ids) if int(token_id) in eos_ids]
    non_eos_special_positions = [
        idx for idx, token_id in enumerate(token_ids) if int(token_id) in specials and int(token_id) not in eos_ids
    ]
    return {
        "eos_positions": eos_positions,
        "first_eos_position": eos_positions[0] if eos_positions else None,
        "non_eos_special_positions": non_eos_special_positions,
        "has_non_eos_special": bool(non_eos_special_positions),
    }


@torch.no_grad()
def _generate_from_starts(
    *,
    model: Any,
    tokenizer: Any,
    start_ids_batch: list[list[int]],
    real_token_mask: torch.Tensor,
    max_tail: int,
    eos_ids: list[int],
    device: torch.device,
) -> list[list[int]]:
    max_len = max(len(ids) for ids in start_ids_batch)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    input_ids = torch.full((len(start_ids_batch), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, ids in enumerate(start_ids_batch):
        input_ids[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_idx, : len(ids)] = 1

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tail,
        do_sample=False,
        logits_processor=[RealTokenMaskProcessor(real_token_mask)],
        eos_token_id=[int(x) for x in eos_ids] if eos_ids else None,
        pad_token_id=int(pad_id),
        use_cache=True,
    )

    out: list[list[int]] = []
    for row_idx, start_ids in enumerate(start_ids_batch):
        continuation = generated[row_idx, len(start_ids) :].detach().cpu().tolist()
        # Drop generation padding after the first EOS; keep EOS itself as a real candidate token.
        trimmed: list[int] = []
        for token_id in continuation:
            if int(token_id) == int(pad_id) and trimmed and trimmed[-1] in eos_ids:
                break
            trimmed.append(int(token_id))
            if int(token_id) in eos_ids:
                break
        out.append(trimmed)
    return out


def _validate_raw_shared_prompt(
    *,
    prompt: str,
    tokenizers: dict[str, Any],
    prefix_generated_ids: list[int],
    pt_token_id: int,
    it_token_id: int,
) -> dict[str, Any]:
    ids_pt = tokenizers["pt"].encode(prompt, add_special_tokens=True)
    ids_it = tokenizers["it"].encode(prompt, add_special_tokens=True)
    if ids_pt != ids_it:
        return {"ok": False, "reason": "raw_prompt_token_ids_differ", "pt_len": len(ids_pt), "it_len": len(ids_it)}
    for token_id in [pt_token_id, it_token_id, *prefix_generated_ids]:
        pt_text = tokenizers["pt"].decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        it_text = tokenizers["it"].decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if pt_text != it_text:
            return {
                "ok": False,
                "reason": "token_id_decodes_differ",
                "token_id": int(token_id),
                "pt_text": pt_text,
                "it_text": it_text,
            }
    return {"ok": True, "prompt_token_ids": ids_pt}


def build_candidate_row(
    *,
    model_name: str,
    manifest_record: dict[str, Any],
    dataset_record: dict[str, Any],
    metadata: dict[str, Any] | None,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    max_tail: int,
    event_kind: str,
) -> dict[str, Any]:
    event = (manifest_record.get("divergence_events") or {}).get(event_kind)
    prompt_id = str(manifest_record.get("prompt_id"))
    if not isinstance(event, dict):
        return {"model": model_name, "prompt_id": prompt_id, "valid": False, "reason": "missing_event"}

    pt_token = event.get("pt_token") or {}
    it_token = event.get("it_token") or {}
    if pt_token.get("token_id") is None or it_token.get("token_id") is None:
        return {"model": model_name, "prompt_id": prompt_id, "valid": False, "reason": "missing_token_ids"}
    pt_token_id = int(pt_token["token_id"])
    it_token_id = int(it_token["token_id"])
    prefix_generated_ids = _prefix_ids_for_event(manifest_record, event)
    raw_prompt = get_prompt_for_variant(
        dataset_record,
        variant="pt",
        tokenizer=tokenizers["pt"],
        apply_chat_template=False,
    )
    validation = _validate_raw_shared_prompt(
        prompt=raw_prompt,
        tokenizers=tokenizers,
        prefix_generated_ids=prefix_generated_ids,
        pt_token_id=pt_token_id,
        it_token_id=it_token_id,
    )
    if not validation.get("ok"):
        return {
            "model": model_name,
            "prompt_id": prompt_id,
            "event_kind": event_kind,
            "valid": False,
            "reason": validation.get("reason"),
            "validation": validation,
        }

    prompt_token_ids = [int(x) for x in validation["prompt_token_ids"]]
    shared_prefix_ids = prompt_token_ids + prefix_generated_ids
    starts_it = [shared_prefix_ids + [it_token_id]]
    starts_pt = [shared_prefix_ids + [pt_token_id], shared_prefix_ids + [it_token_id]]
    eos_ids = {
        "pt": steering_adapter.eos_token_ids(tokenizers["pt"]),
        "it": steering_adapter.eos_token_ids(tokenizers["it"]),
    }
    desc_tail = _generate_from_starts(
        model=models["it"],
        tokenizer=tokenizers["it"],
        start_ids_batch=starts_it,
        real_token_mask=real_token_masks["it"],
        max_tail=max_tail,
        eos_ids=eos_ids["it"],
        device=device,
    )[0]
    base_tail, base_forced_desc_tail = _generate_from_starts(
        model=models["pt"],
        tokenizer=tokenizers["pt"],
        start_ids_batch=starts_pt,
        real_token_mask=real_token_masks["pt"],
        max_tail=max_tail,
        eos_ids=eos_ids["pt"],
        device=device,
    )

    candidates = {
        "desc_primary": [it_token_id] + desc_tail,
        "base_primary": [pt_token_id] + base_tail,
        "base_forced_desc": [it_token_id] + base_forced_desc_tail,
    }
    token_info = {
        name: {
            "token_ids": ids,
            "token_texts": _token_text(tokenizers["pt"], ids),
            "text": tokenizers["pt"].decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            **_special_flags(tokenizers["pt"], ids, set(eos_ids["pt"])),
        }
        for name, ids in candidates.items()
    }
    return {
        "model": model_name,
        "prompt_id": prompt_id,
        "prompt_mode": "raw_shared",
        "event_kind": event_kind,
        "valid": True,
        "max_tail": int(max_tail),
        "event_step": int(event["step"]),
        "prompt_token_ids": prompt_token_ids,
        "prefix_generated_ids": prefix_generated_ids,
        "prefix_length_total": len(shared_prefix_ids),
        "pt_token_id": pt_token_id,
        "it_token_id": it_token_id,
        "pt_token_text": pt_token.get("token_str"),
        "it_token_text": it_token.get("token_str"),
        "category": (metadata or {}).get("category"),
        "source": (metadata or {}).get("source"),
        "recipe_group": (metadata or {}).get("recipe_group"),
        "token_category": (metadata or {}).get("token_category"),
        "position": (metadata or {}).get("position"),
        "position_ge_3": (metadata or {}).get("position_ge_3"),
        "position_ge_5": (metadata or {}).get("position_ge_5"),
        "slices": (metadata or {}).get("slices", ["full_1400"]),
        "candidates": token_info,
    }


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    exp47_root = Path(args.exp47_root)
    out_path = Path(args.output_dir) / args.model / f"candidate_sequences_w{args.worker_index}.jsonl.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    dataset = _dataset_lookup(Path(args.dataset))
    event_metadata = load_exp47_event_metadata(exp47_root, event_kind=args.event_kind)
    manifest_path = find_exp20_manifest(exp47_root, args.model)
    rows = list(json_rows(manifest_path))
    if args.n_examples is not None:
        rows = rows[: int(args.n_examples)]
    rows = rows[args.worker_index :: args.n_workers]
    done = _done_keys(out_path) if args.resume else set()
    log.info(
        "[exp49 candidates] model=%s worker=%d/%d manifest=%s rows=%d done=%d",
        args.model,
        args.worker_index,
        args.n_workers,
        manifest_path,
        len(rows),
        len(done),
    )
    for idx, manifest_record in enumerate(rows):
        prompt_id = str(manifest_record.get("prompt_id"))
        if (args.model, prompt_id) in done:
            continue
        dataset_record = dataset.get(prompt_id)
        if dataset_record is None:
            row = {"model": args.model, "prompt_id": prompt_id, "valid": False, "reason": "dataset_record_missing"}
        else:
            row = build_candidate_row(
                model_name=args.model,
                manifest_record=manifest_record,
                dataset_record=dataset_record,
                metadata=event_metadata.get((args.model, prompt_id)),
                models=models,
                tokenizers=tokenizers,
                real_token_masks=real_token_masks,
                steering_adapter=steering_adapter,
                device=device,
                max_tail=int(args.max_tail),
                event_kind=args.event_kind,
            )
        append_jsonl_gz(out_path, row)
        if (idx + 1) % 25 == 0:
            log.info("[exp49 candidates] model=%s worker=%d wrote %d/%d", args.model, args.worker_index, idx + 1, len(rows))
    log.info("[exp49 candidates] done model=%s worker=%d output=%s", args.model, args.worker_index, out_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exp47-root", default=str(EXP47_DEFAULT_ROOT))
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--output-dir", default=str(EXP49_DEFAULT_ROOT / "debug" / "candidates"))
    parser.add_argument("--event-kind", default=DEFAULT_EVENT_KIND)
    parser.add_argument("--max-tail", type=int, default=DEFAULT_MAX_TAIL)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_worker(args)


if __name__ == "__main__":
    main()

