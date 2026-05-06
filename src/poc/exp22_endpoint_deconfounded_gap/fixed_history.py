"""Fixed-history template audit collector for Exp22.

This extension separates prompt-template effects from free-generation length
effects. For each prompt it first obtains one greedy teacher continuation from
either IT-native or PT-raw, then replays exactly those continuation tokens
through PT raw, IT native-chat, and IT raw/no-template cells.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.utils import load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp22_endpoint_deconfounded_gap.collect import (
    DENSE5_MODELS,
    _load_readouts,
    _logits_by_layer,
    _prompt_for_regime,
    _read_done_ids,
    _record_id,
    merge_worker_outputs,
)
from src.poc.exp22_endpoint_deconfounded_gap.metrics import distribution_arrays_from_logits

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TEACHER_SOURCES = ("it_native", "pt_raw")
CELLS = ("pt_raw", "it_native", "it_raw")
CELL_TO_VARIANT = {
    "pt_raw": "pt",
    "it_native": "it",
    "it_raw": "it",
}
CELL_TO_PROMPT_REGIME = {
    "pt_raw": "native",
    "it_native": "native",
    "it_raw": "raw",
}


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _cell_prompt(
    record: dict[str, Any],
    *,
    cell: str,
    tokenizer,
) -> tuple[str, str]:
    if cell not in CELLS:
        raise ValueError(f"unknown fixed-history cell={cell!r}; choices={CELLS}")
    return _prompt_for_regime(
        record,
        variant=CELL_TO_VARIANT[cell],
        tokenizer=tokenizer,
        prompt_regime=CELL_TO_PROMPT_REGIME[cell],
    )


def _rank_from_logits(logits: torch.Tensor, token_id: int) -> int | None:
    if token_id < 0 or token_id >= int(logits.shape[-1]):
        return None
    target = logits[token_id]
    if not torch.isfinite(target):
        return None
    return int((logits > target).sum().item()) + 1


def _target_stats(masked_logits: torch.Tensor, token_id: int) -> dict[str, Any]:
    top1 = int(torch.argmax(masked_logits).item())
    if token_id < 0 or token_id >= int(masked_logits.shape[-1]) or not torch.isfinite(masked_logits[token_id]):
        return {
            "target_logprob": None,
            "target_rank": None,
            "top1_matches_forced": False,
            "final_top1_id": top1,
        }
    log_probs = torch.log_softmax(masked_logits.float(), dim=-1)
    return {
        "target_logprob": _finite_or_none(float(log_probs[token_id].item())),
        "target_rank": _rank_from_logits(masked_logits, token_id),
        "top1_matches_forced": bool(top1 == int(token_id)),
        "final_top1_id": top1,
    }


def _append_jsonl_gz(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _read_teacher_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_id = str(row.get("prompt_id"))
            if prompt_id:
                out[prompt_id] = row
    return out


@torch.no_grad()
def generate_teacher_continuation(
    *,
    record: dict[str, Any],
    prompt_id: str,
    teacher_source: str,
    model,
    tokenizer,
    adapter,
    steering_adapter,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt, prompt_mode = _cell_prompt(record, cell=teacher_source, tokenizer=tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    if input_ids.shape[1] < 1:
        raise ValueError(f"empty teacher prompt after tokenization for {prompt_id}")

    real_token_mask = steering_adapter.real_token_mask(tokenizer, device, model)
    stop_ids = set(adapter.stop_token_ids(tokenizer))
    forced_ids: list[int] = []
    past_key_values = None
    next_input_ids = input_ids
    for _step in range(max_new_tokens):
        outputs = model(
            input_ids=next_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        masked_final_logits = outputs.logits[0, -1, :].float().clone()
        masked_final_logits[~real_token_mask.to(masked_final_logits.device)] = float("-inf")
        next_id = int(torch.argmax(masked_final_logits).item())
        forced_ids.append(next_id)
        next_input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        if next_id in stop_ids:
            break

    teacher_text = tokenizer.decode(
        forced_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "prompt_id": prompt_id,
        "model": steering_adapter.model_name,
        "teacher_source": teacher_source,
        "prompt_mode": prompt_mode,
        "n_steps": len(forced_ids),
        "forced_ids": forced_ids,
        "teacher_generated_text": teacher_text,
    }


@torch.no_grad()
def replay_forced_continuation(
    *,
    record: dict[str, Any],
    prompt_id: str,
    teacher_source: str,
    forced_ids: list[int],
    teacher_generated_text: str,
    model,
    tokenizer,
    adapter,
    steering_adapter,
    cell: str,
    device: torch.device,
    readouts,
    top_k: int,
) -> dict[str, Any]:
    prompt, prompt_mode = _cell_prompt(record, cell=cell, tokenizer=tokenizer)
    variant = CELL_TO_VARIANT[cell]
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    if input_ids.shape[1] < 1:
        raise ValueError(f"empty replay prompt after tokenization for {prompt_id}/{cell}")

    layers = adapter.layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inp, output):
            h = adapter.residual_from_output(output)
            captured[layer_idx] = h[0, -1, :].detach()

        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(len(layers))]
    final_norm = adapter.final_norm(model)
    lm_head = adapter.lm_head(model)
    real_token_mask = steering_adapter.real_token_mask(tokenizer, device, model)
    probe_payloads: dict[str, dict[str, list[Any]]] = {
        family: {
            "kl_to_final": [],
            "entropy": [],
            "confidence": [],
            "top1_margin": [],
            "top1_ids": [],
            "top5_ids": [],
            "top5_logprobs": [],
            "adjacent_kl": [],
            "adjacent_js": [],
        }
        for family in readouts
    }
    target_logprob: list[float | None] = []
    target_rank: list[int | None] = []
    top1_matches_forced: list[bool] = []
    final_top1_ids: list[int] = []

    past_key_values = None
    next_input_ids = input_ids
    try:
        for token_id in forced_ids:
            captured.clear()
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            residuals = [captured[i] for i in range(len(layers))]
            masked_final_logits = outputs.logits[0, -1, :].float().clone()
            masked_final_logits[~real_token_mask.to(masked_final_logits.device)] = float("-inf")
            stats = _target_stats(masked_final_logits, int(token_id))
            target_logprob.append(stats["target_logprob"])
            target_rank.append(stats["target_rank"])
            top1_matches_forced.append(bool(stats["top1_matches_forced"]))
            final_top1_ids.append(int(stats["final_top1_id"]))

            for family, readout in readouts.items():
                logits = _logits_by_layer(
                    residuals=residuals,
                    final_norm=final_norm,
                    lm_head=lm_head,
                    real_token_mask=real_token_mask,
                    readout=readout,
                )
                arrays = distribution_arrays_from_logits(logits, top_k=top_k)
                for key, value in arrays.items():
                    probe_payloads[family][key].append(value)

            next_input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
    finally:
        for handle in handles:
            handle.remove()

    return {
        "prompt_id": prompt_id,
        "model": steering_adapter.model_name,
        "variant": variant,
        "cell": cell,
        "teacher_source": teacher_source,
        "prompt_mode": prompt_mode,
        "n_layers": len(layers),
        "n_steps": len(forced_ids),
        "forced_ids": [int(x) for x in forced_ids],
        "generated_ids": [int(x) for x in forced_ids],
        "teacher_generated_text": teacher_generated_text,
        "probe_families": sorted(readouts),
        "target_logprob": target_logprob,
        "target_rank": target_rank,
        "top1_matches_forced": top1_matches_forced,
        "final_top1_ids": final_top1_ids,
        "probes": probe_payloads,
    }


def _write_malformed(
    path: Path,
    *,
    prompt_id: str,
    model: str,
    teacher_source: str,
    cell: str | None,
    error: str,
) -> None:
    row = {
        "prompt_id": prompt_id,
        "model": model,
        "teacher_source": teacher_source,
        "malformed": True,
        "error": error,
    }
    if cell is not None:
        row["cell"] = cell
        row["variant"] = CELL_TO_VARIANT.get(cell)
    _append_jsonl_gz(path, row)


def _load_variant_context(args: argparse.Namespace, variant: str):
    spec = get_spec(args.model)
    device = torch.device(args.device)
    model_id = model_id_for_variant(spec, variant)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        device,
        dtype=getattr(torch, args.dtype),
        multi_gpu=spec.multi_gpu,
    )
    steering_adapter = get_steering_adapter(args.model)
    adapter = steering_adapter.adapter
    readouts = _load_readouts(
        probe_families=args.probe_families,
        tuned_lens_dir=args.tuned_lens_dir,
        model_name=args.model,
        variant=variant,
        d_model=spec.d_model,
        n_layers=spec.n_layers,
        device=device,
    )
    return model, tokenizer, adapter, steering_adapter, readouts


def _close_context(context) -> None:
    model, tokenizer, adapter, steering_adapter, readouts = context
    del model, tokenizer, adapter, steering_adapter, readouts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_collect(args: argparse.Namespace) -> None:
    if args.model not in DENSE5_MODELS and not args.allow_non_dense:
        raise ValueError(f"Exp22 fixed-history audit excludes non-dense model {args.model}; pass --allow-non-dense to override")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_dataset(
        args.dataset,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        n_examples=args.n_eval_examples,
    )
    teacher_variant = CELL_TO_VARIANT[args.teacher_source]
    prompt_ids = [_record_id(record, args.worker_index + local_idx * args.n_workers) for local_idx, record in enumerate(records)]
    teacher_dir = out_dir / f"teacher_{args.teacher_source}"
    teacher_path = teacher_dir / f"records_w{args.worker_index}.jsonl.gz"
    teacher_cache = _read_teacher_cache(teacher_path)
    missing_teacher = [
        (prompt_id, record)
        for prompt_id, record in zip(prompt_ids, records, strict=True)
        if prompt_id not in teacher_cache
    ]

    contexts: dict[str, Any] = {}
    if missing_teacher:
        log.info(
            "Generating %d missing %s teacher continuations for %s worker %d",
            len(missing_teacher),
            args.teacher_source,
            args.model,
            args.worker_index,
        )
        contexts[teacher_variant] = _load_variant_context(args, teacher_variant)
        model, tokenizer, adapter, steering_adapter, _readouts = contexts[teacher_variant]
        device = torch.device(args.device)
        for prompt_id, record in missing_teacher:
            try:
                teacher = generate_teacher_continuation(
                    record=record,
                    prompt_id=prompt_id,
                    teacher_source=args.teacher_source,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    steering_adapter=steering_adapter,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
                _append_jsonl_gz(teacher_path, teacher)
                teacher_cache[prompt_id] = teacher
            except Exception as exc:
                _write_malformed(
                    teacher_path,
                    prompt_id=prompt_id,
                    model=args.model,
                    teacher_source=args.teacher_source,
                    cell=None,
                    error=repr(exc),
                )
                log.exception("teacher generation failed prompt_id=%s model=%s", prompt_id, args.model)

    cells_by_variant: dict[str, list[str]] = {"it": [], "pt": []}
    for cell in args.cells:
        cells_by_variant[CELL_TO_VARIANT[cell]].append(cell)

    for variant in ("it", "pt"):
        cells = cells_by_variant[variant]
        if not cells:
            continue
        context = contexts.get(variant)
        if context is None:
            context = _load_variant_context(args, variant)
            contexts[variant] = context
        model, tokenizer, adapter, steering_adapter, readouts = context
        device = torch.device(args.device)
        for cell in cells:
            cell_dir = out_dir / cell
            worker_path = cell_dir / f"records_w{args.worker_index}.jsonl.gz"
            done_ids = _read_done_ids(worker_path)
            log.info(
                "Replaying fixed history model=%s cell=%s worker=%d/%d prompts=%d out=%s",
                args.model,
                cell,
                args.worker_index,
                args.n_workers,
                len(records),
                worker_path,
            )
            for prompt_id, record in zip(prompt_ids, records, strict=True):
                if prompt_id in done_ids:
                    continue
                teacher = teacher_cache.get(prompt_id)
                forced_ids = teacher.get("forced_ids") if teacher else None
                if not teacher or teacher.get("malformed") or not isinstance(forced_ids, list):
                    _write_malformed(
                        worker_path,
                        prompt_id=prompt_id,
                        model=args.model,
                        teacher_source=args.teacher_source,
                        cell=cell,
                        error="missing_or_malformed_teacher",
                    )
                    continue
                try:
                    out = replay_forced_continuation(
                        record=record,
                        prompt_id=prompt_id,
                        teacher_source=args.teacher_source,
                        forced_ids=[int(x) for x in forced_ids],
                        teacher_generated_text=str(teacher.get("teacher_generated_text") or ""),
                        model=model,
                        tokenizer=tokenizer,
                        adapter=adapter,
                        steering_adapter=steering_adapter,
                        cell=cell,
                        device=device,
                        readouts=readouts,
                        top_k=args.top_k,
                    )
                    _append_jsonl_gz(worker_path, out)
                except Exception as exc:
                    _write_malformed(
                        worker_path,
                        prompt_id=prompt_id,
                        model=args.model,
                        teacher_source=args.teacher_source,
                        cell=cell,
                        error=repr(exc),
                    )
                    log.exception("fixed replay failed prompt_id=%s model=%s cell=%s", prompt_id, args.model, cell)
    for context in contexts.values():
        _close_context(context)


def merge_outputs(out_dir: Path, n_workers: int, cells: list[str], teacher_source: str) -> None:
    merge_worker_outputs(out_dir / f"teacher_{teacher_source}", n_workers)
    for cell in cells:
        merge_worker_outputs(out_dir / cell, n_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--n-eval-examples", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--probe-families", nargs="+", choices=["raw", "tuned"], default=["raw", "tuned"])
    parser.add_argument("--tuned-lens-dir", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--teacher-source", choices=TEACHER_SOURCES, default="it_native")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--allow-non-dense", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_outputs(Path(args.out_dir), args.n_workers, args.cells, args.teacher_source)
        log.info("Merged fixed-history workers under %s", args.out_dir)
        return
    run_collect(args)


if __name__ == "__main__":
    main()
