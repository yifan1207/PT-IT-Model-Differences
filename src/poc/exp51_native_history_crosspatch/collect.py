"""Collect Exp51 native-history local-disagreement cross-patching records.

Exp51 creates local PT/IT next-token disagreement events at fixed horizons on a
native greedy history, then reuses the Exp23 residual-state x late-stack
factorial scorer for the actual four-cell intervention.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessor

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    _late_boundary,
    _make_readouts,
    _rank,
    collect_manifest_record,
)
from src.poc.exp51_native_history_crosspatch import DEFAULT_HORIZONS

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


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _done_prompt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("prompt_id")) for row in _json_rows(path) if row.get("prompt_id") is not None}


def _dataset_rows(path: Path, n_examples: int | None, worker_index: int, n_workers: int) -> list[dict[str, Any]]:
    rows = load_dataset(path, n_examples=n_examples)
    return rows[worker_index::n_workers]


def _token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _token_class(text: str) -> str:
    if text == "":
        return "EMPTY"
    if text.isspace():
        return "WHITESPACE"
    stripped = text.strip()
    if stripped == "":
        return "WHITESPACE"
    if all(ch in ".,;:!?()[]{}<>-_/\\|\"'`~@#$%^&*+=\n\t " for ch in text):
        return "PUNCT_FORMAT"
    if stripped.startswith(("<|", "[", "{", "#", "-", "*")):
        return "FORMAT"
    if any(ch.isalpha() for ch in stripped):
        return "CONTENT_ALPHA"
    if any(ch.isdigit() for ch in stripped):
        return "CONTENT_NUMERIC"
    return "OTHER"


def _prefix_hash(token_ids: list[int]) -> str:
    payload = ",".join(str(int(x)) for x in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _stable_event_id(
    *,
    model_name: str,
    prompt_id: str,
    prompt_mode: str,
    history_source: str,
    horizon: int,
    prefix_hash: str,
    pt_token_id: int,
    it_token_id: int,
) -> str:
    payload = {
        "source": "exp51_native_history_crosspatch",
        "model": model_name,
        "prompt_id": prompt_id,
        "prompt_mode": prompt_mode,
        "history_source": history_source,
        "horizon": int(horizon),
        "prefix_hash": prefix_hash,
        "pt_token_id": int(pt_token_id),
        "it_token_id": int(it_token_id),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@torch.no_grad()
def _generate_history(
    *,
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    real_token_mask: torch.Tensor,
    eos_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        logits_processor=[RealTokenMaskProcessor(real_token_mask)],
        eos_token_id=[int(x) for x in eos_ids] if eos_ids else None,
        pad_token_id=int(pad_id),
        use_cache=True,
    )
    continuation = generated[0, len(prompt_ids) :].detach().cpu().tolist()
    out: list[int] = []
    for token_id in continuation:
        out.append(int(token_id))
        if int(token_id) in set(int(x) for x in eos_ids):
            break
    return out


@torch.no_grad()
def _native_logits(
    *,
    model: Any,
    input_ids: torch.Tensor,
    real_token_mask: torch.Tensor,
) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1, :].detach().float()
    logits = logits.clone()
    logits[~real_token_mask.to(logits.device)] = float("-inf")
    return logits


def _logprob(logits: torch.Tensor, token_id: int) -> float | None:
    if token_id < 0 or token_id >= logits.shape[-1] or not torch.isfinite(logits[token_id]):
        return None
    return float(torch.log_softmax(logits, dim=-1)[token_id].item())


def _validate_raw_shared_prompt(
    *,
    raw_prompt: str,
    tokenizers: dict[str, Any],
) -> dict[str, Any]:
    ids_pt = tokenizers["pt"].encode(raw_prompt, add_special_tokens=True)
    ids_it = tokenizers["it"].encode(raw_prompt, add_special_tokens=True)
    if ids_pt != ids_it:
        return {
            "ok": False,
            "reason": "raw_prompt_token_ids_differ",
            "pt_len": len(ids_pt),
            "it_len": len(ids_it),
        }
    return {"ok": True, "prompt_ids": [int(x) for x in ids_pt]}


def _horizon_status_template(horizon: int) -> dict[str, Any]:
    return {
        "horizon": int(horizon),
        "prefix_not_terminated": False,
        "local_disagreement": False,
        "valid_after_real_token_mask": False,
        "reason": None,
    }


def build_native_history_manifest(
    *,
    model_name: str,
    dataset_record: dict[str, Any],
    history_source: str,
    horizons: list[int],
    max_history_tokens: int,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_id = str(dataset_record.get("id", dataset_record.get("record_id")))
    raw_prompt = get_prompt_for_variant(
        dataset_record,
        variant="pt",
        tokenizer=tokenizers["pt"],
        apply_chat_template=False,
    )
    prompt_validation = _validate_raw_shared_prompt(raw_prompt=raw_prompt, tokenizers=tokenizers)
    if not prompt_validation.get("ok"):
        return (
            {"prompt_id": prompt_id, "divergence_events": {}, "free_runs": {"A_pt_raw": {"generated_token_ids": []}}},
            {
                "valid_prompt": False,
                "prompt_validation": prompt_validation,
                "horizon_status": {str(h): _horizon_status_template(h) for h in horizons},
            },
        )

    prompt_ids = [int(x) for x in prompt_validation["prompt_ids"]]
    eos_ids = {
        "pt": steering_adapter.eos_token_ids(tokenizers["pt"]),
        "it": steering_adapter.eos_token_ids(tokenizers["it"]),
    }
    generated_ids = _generate_history(
        model=models[history_source],
        tokenizer=tokenizers[history_source],
        prompt_ids=prompt_ids,
        real_token_mask=real_token_masks[history_source],
        eos_ids=eos_ids[history_source],
        max_new_tokens=max_history_tokens,
        device=device,
    )
    eos_set = set(int(x) for x in eos_ids[history_source])

    events: dict[str, Any] = {}
    horizon_status: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        status = _horizon_status_template(horizon)
        if horizon > len(generated_ids):
            status["reason"] = "history_shorter_than_horizon"
            horizon_status[str(horizon)] = status
            continue
        prefix_generated_ids = [int(x) for x in generated_ids[:horizon]]
        if any(int(x) in eos_set for x in prefix_generated_ids):
            status["reason"] = "prefix_already_terminated"
            horizon_status[str(horizon)] = status
            continue
        if horizon >= len(generated_ids):
            status["reason"] = "missing_next_history_token_for_consistency_check"
            horizon_status[str(horizon)] = status
            continue
        status["prefix_not_terminated"] = True
        full_ids = prompt_ids + prefix_generated_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        logits_pt = _native_logits(model=models["pt"], input_ids=input_ids, real_token_mask=real_token_masks["pt"])
        logits_it = _native_logits(model=models["it"], input_ids=input_ids, real_token_mask=real_token_masks["it"])
        t_pt = int(torch.argmax(logits_pt).item())
        t_it = int(torch.argmax(logits_it).item())
        expected_history_next = int(generated_ids[horizon])
        status.update(
            {
                "pt_token_id": t_pt,
                "it_token_id": t_it,
                "pt_token_text": _token_text(tokenizers["pt"], t_pt),
                "it_token_text": _token_text(tokenizers["pt"], t_it),
                "pt_token_class": _token_class(_token_text(tokenizers["pt"], t_pt)),
                "it_token_class": _token_class(_token_text(tokenizers["pt"], t_it)),
                "it_history_next_token_id": expected_history_next,
                "it_history_next_token_matches_top1": bool(
                    expected_history_next == (t_it if history_source == "it" else t_pt)
                ),
                "logprob_it_t_it": _logprob(logits_it, t_it),
                "logprob_it_t_pt": _logprob(logits_it, t_pt),
                "rank_it_t_it": _rank(logits_it, t_it),
                "rank_it_t_pt": _rank(logits_it, t_pt),
                "logprob_pt_t_pt": _logprob(logits_pt, t_pt),
                "logprob_pt_t_it": _logprob(logits_pt, t_it),
                "rank_pt_t_pt": _rank(logits_pt, t_pt),
                "rank_pt_t_it": _rank(logits_pt, t_it),
            }
        )
        if t_pt == t_it:
            status["reason"] = "pt_it_agree"
            horizon_status[str(horizon)] = status
            continue
        status["local_disagreement"] = True
        # Full target/prefix validation is repeated by Exp23's scorer, but we
        # perform the cheap token decode check here so support accounting has a
        # meaningful pre-scorer invalid bucket.
        for token_id in [t_pt, t_it, *prefix_generated_ids]:
            pt_text = _token_text(tokenizers["pt"], token_id)
            it_text = _token_text(tokenizers["it"], token_id)
            if pt_text != it_text:
                status["reason"] = "token_id_decodes_differ"
                status["mismatch_token_id"] = int(token_id)
                status["pt_text"] = pt_text
                status["it_text"] = it_text
                horizon_status[str(horizon)] = status
                break
        else:
            status["valid_after_real_token_mask"] = True
            prefix_sig = _prefix_hash(full_ids)
            event_key = f"{history_source}_h{int(horizon)}"
            events[event_key] = {
                "step": int(horizon),
                "horizon": int(horizon),
                "history_source": history_source,
                "event_id": _stable_event_id(
                    model_name=model_name,
                    prompt_id=prompt_id,
                    prompt_mode="raw_shared",
                    history_source=history_source,
                    horizon=int(horizon),
                    prefix_hash=prefix_sig,
                    pt_token_id=t_pt,
                    it_token_id=t_it,
                ),
                "prefix_hash": prefix_sig,
                "pt_token": {"token_id": t_pt, "text": status["pt_token_text"]},
                "it_token": {"token_id": t_it, "text": status["it_token_text"]},
                "native_readout": status,
            }
            horizon_status[str(horizon)] = status
            continue
        if str(horizon) not in horizon_status:
            horizon_status[str(horizon)] = status

    manifest = {
        "prompt_id": prompt_id,
        "divergence_events": events,
        # Exp23's scorer uses the A_pt_raw generated-token prefix by event step.
        # For Exp51 this is intentionally the selected native-history prefix.
        "free_runs": {"A_pt_raw": {"generated_token_ids": [int(x) for x in generated_ids]}},
    }
    metadata = {
        "valid_prompt": True,
        "prompt_validation": prompt_validation,
        "history_source": history_source,
        "generated_token_ids": [int(x) for x in generated_ids],
        "generated_token_texts": [_token_text(tokenizers[history_source], int(x)) for x in generated_ids],
        "horizon_status": horizon_status,
    }
    return manifest, metadata


def collect_dataset_record(
    *,
    model_name: str,
    dataset_record: dict[str, Any],
    history_source: str,
    horizons: list[int],
    max_history_tokens: int,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    readouts: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    include_noop_patch: bool,
    boundary_layer_override: int | None,
) -> dict[str, Any]:
    manifest, metadata = build_native_history_manifest(
        model_name=model_name,
        dataset_record=dataset_record,
        history_source=history_source,
        horizons=horizons,
        max_history_tokens=max_history_tokens,
        models=models,
        tokenizers=tokenizers,
        real_token_masks=real_token_masks,
        steering_adapter=steering_adapter,
        device=device,
    )
    prompt_id = str(dataset_record.get("id", dataset_record.get("record_id")))
    if not metadata.get("valid_prompt"):
        return {
            "part": "exp51_native_history_crosspatch",
            "model": model_name,
            "prompt_id": prompt_id,
            "prompt_mode": "raw_shared",
            "history_source": history_source,
            "native_history": metadata,
            "events": {},
            "valid": False,
        }
    event_kinds = list((manifest.get("divergence_events") or {}).keys())
    if event_kinds:
        residual = collect_manifest_record(
            model_name=model_name,
            manifest_record=manifest,
            dataset_record=dataset_record,
            prompt_mode="raw_shared",
            event_kinds=event_kinds,
            models=models,
            tokenizers=tokenizers,
            readouts=readouts,
            steering_adapter=steering_adapter,
            device=device,
            collect_trajectories=False,
            include_noop_patch=include_noop_patch,
            boundary_layer_override=boundary_layer_override,
            experiment_name="exp51_native_history_crosspatch",
            boundary_mode="full_late_native_history",
        )
        events = residual.get("events") or {}
        boundary_info = {
            "boundary_layer": residual.get("boundary_layer"),
            "boundary_source": residual.get("boundary_source"),
            "downstream_stack": residual.get("downstream_stack"),
        }
    else:
        events = {}
        boundary = int(boundary_layer_override) if boundary_layer_override is not None else _late_boundary(model_name)
        boundary_info = {
            "boundary_layer": boundary,
            "boundary_source": "cli_override" if boundary_layer_override is not None else "depth_ablation_windows",
            "downstream_stack": None,
        }
    return {
        "part": "exp51_native_history_crosspatch",
        "model": model_name,
        "prompt_id": prompt_id,
        "prompt_mode": "raw_shared",
        "history_source": history_source,
        "category": dataset_record.get("category") or dataset_record.get("benchmark") or dataset_record.get("task"),
        "native_history": metadata,
        "events": events,
        "valid": True,
        **boundary_info,
    }


def run_worker(args: argparse.Namespace) -> None:
    if args.prompt_mode != "raw_shared":
        raise ValueError("Exp51 v1 is raw_shared-only")
    if args.history_source not in {"it", "pt"}:
        raise ValueError("--history-source must be it or pt")
    device = torch.device(args.device)
    spec = get_spec(args.model)
    steering_adapter = get_steering_adapter(args.model)
    pt_model, pt_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "pt"), args.device, multi_gpu=spec.multi_gpu
    )
    it_model, it_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "it"), args.device, multi_gpu=spec.multi_gpu
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

    rows = _dataset_rows(args.dataset, args.n_eval_examples, args.worker_index, args.n_workers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"records_w{args.worker_index}.jsonl.gz"
    done = _done_prompt_ids(out_path)
    log.info(
        "[exp51] model=%s history=%s worker=%d/%d rows=%d resume=%d",
        args.model,
        args.history_source,
        args.worker_index,
        args.n_workers,
        len(rows),
        len(done),
    )
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for idx, dataset_record in enumerate(rows):
            prompt_id = str(dataset_record.get("id", dataset_record.get("record_id")))
            if prompt_id in done:
                continue
            try:
                result = collect_dataset_record(
                    model_name=args.model,
                    dataset_record=dataset_record,
                    history_source=args.history_source,
                    horizons=[int(x) for x in args.horizons],
                    max_history_tokens=args.max_history_tokens,
                    models=models,
                    tokenizers=tokenizers,
                    readouts=readouts,
                    real_token_masks=real_token_masks,
                    steering_adapter=steering_adapter,
                    device=device,
                    include_noop_patch=not args.no_noop_patch,
                    boundary_layer_override=args.boundary_layer_override,
                )
                fout.write(json.dumps(result, separators=(",", ":")) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[exp51] prompt %s failed: %s", prompt_id, exc)
                fout.write(
                    json.dumps(
                        {
                            "part": "exp51_native_history_crosspatch",
                            "model": args.model,
                            "prompt_id": prompt_id,
                            "prompt_mode": "raw_shared",
                            "history_source": args.history_source,
                            "valid": False,
                            "error": repr(exc),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                fout.flush()
            if (idx + 1) % 5 == 0:
                log.info("[exp51] %s/%s %d/%d prompts", args.history_source, args.model, idx + 1, len(rows))


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    seen: set[str] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(n_workers):
            path = out_dir / f"records_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp51] missing worker file %s", path)
                continue
            for row in _json_rows(path):
                prompt_id = str(row.get("prompt_id", ""))
                if prompt_id and prompt_id in seen:
                    continue
                if prompt_id:
                    seen.add(prompt_id)
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp51 native-history cross-patching records.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["raw_shared"], default="raw_shared")
    parser.add_argument("--history-source", choices=["it", "pt"], default="it")
    parser.add_argument("--horizons", nargs="*", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--max-history-tokens", type=int, default=64)
    parser.add_argument("--boundary-layer-override", type=int, default=None)
    parser.add_argument("--no-noop-patch", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args(argv)


def main_with_args(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.merge_only:
        merge_workers(args.out_dir, args.n_workers)
        return
    run_worker(args)


def main() -> None:
    main_with_args(None)


if __name__ == "__main__":
    main()
