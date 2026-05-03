"""Collect pre-late boundary commitment readouts for Exp23 first divergences.

Exp40 is a targeted audit for the possible tautology concern in the Exp23
upstream-state x late-stack factorial.  It replays the same first-divergence
prefixes, captures the residual state entering the late stack, and scores the
divergent IT-vs-PT token pair before any late-stack computation is applied.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.boundary import BoundaryStateCapture
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    DEFAULT_EXP20_FALLBACK_ROOT,
    DEFAULT_EXP20_ROOT,
    DEFAULT_EVENT_KINDS,
    ReadoutBundle,
    _dataset_lookup,
    _done_prompt_ids,
    _find_manifest,
    _late_boundary,
    _load_manifest_records,
    _logits_from_hidden,
    _make_readouts,
    _prefix_ids_for_event,
    _rank,
    _safe_float,
    _token_text,
    _unique_events,
    _validate_tokenizers_and_tokens,
)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

READOUT_NAMES = ("common_it", "common_pt")


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _finite_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _boundary_stats(state: torch.Tensor) -> dict[str, float | None]:
    full = state.detach().float()
    last = full[0, -1, :]
    return {
        "seq_rms": _finite_float(torch.sqrt(torch.mean(full.square())).item()),
        "last_rms": _finite_float(torch.sqrt(torch.mean(last.square())).item()),
        "seq_mean": _finite_float(full.mean().item()),
        "last_mean": _finite_float(last.mean().item()),
        "seq_std": _finite_float(full.std(unbiased=False).item()),
        "last_std": _finite_float(last.std(unbiased=False).item()),
    }


@torch.no_grad()
def _forward_boundary(
    *,
    model: Any,
    layer: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    capture = BoundaryStateCapture(layer)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        boundary_state = capture.snapshot()
    finally:
        capture.close()
    return {
        "boundary_state": boundary_state,
        "final_logits": outputs.logits[0, -1, :].detach().float(),
    }


def _score_logits(
    *,
    logits: torch.Tensor,
    bundle: ReadoutBundle,
    y_pt: int,
    y_it: int,
) -> dict[str, Any]:
    logits = logits.detach().float().clone().to(bundle.real_token_mask.device)
    logits[~bundle.real_token_mask.to(logits.device)] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    top_values, top_ids = torch.topk(logits, k=2)
    top1 = int(top_ids[0].item())
    margin = float((logits[y_it] - logits[y_pt]).item())
    choice = "it" if top1 == y_it else ("pt" if top1 == y_pt else "other")
    finite = torch.isfinite(logits)
    safe_probs = probs[finite]
    entropy = float((-(safe_probs * torch.log(safe_probs.clamp_min(1e-45)))).sum().item())
    return {
        "it_vs_pt_margin": _safe_float(margin),
        "it_logit": _safe_float(float(logits[y_it].item())),
        "pt_logit": _safe_float(float(logits[y_pt].item())),
        "top1_id": top1,
        "token_choice_class": choice,
        "top1_confidence": _safe_float(float(probs[top1].item())),
        "top1_top2_margin": _safe_float(float((top_values[0] - top_values[1]).item())),
        "entropy": _safe_float(entropy),
        "it_rank": _rank(logits, y_it),
        "pt_rank": _rank(logits, y_pt),
    }


def _score_boundary_state(
    *,
    state: torch.Tensor,
    readouts: dict[str, ReadoutBundle],
    y_pt: int,
    y_it: int,
) -> dict[str, Any]:
    hidden = state[0, -1, :]
    out: dict[str, Any] = {}
    for readout_name in READOUT_NAMES:
        bundle = readouts[readout_name]
        logits = _logits_from_hidden(hidden, bundle)
        out[readout_name] = _score_logits(logits=logits, bundle=bundle, y_pt=y_pt, y_it=y_it)
    return out


def _score_final_logits(
    *,
    logits: torch.Tensor,
    readouts: dict[str, ReadoutBundle],
    native_variant: str,
    y_pt: int,
    y_it: int,
) -> dict[str, Any]:
    # The model's native logits are already produced by its native readout.  We
    # store them only as a sanity check for the selected first-divergence event.
    native_name = f"native_{native_variant}"
    bundle = readouts[native_name]
    return _score_logits(logits=logits, bundle=bundle, y_pt=y_pt, y_it=y_it)


def _first_event(
    manifest_record: dict[str, Any],
    event_kinds: list[str],
) -> tuple[str, dict[str, Any]] | None:
    for kind, event in _unique_events(manifest_record, event_kinds):
        if isinstance(event, dict) and "duplicate_of" not in event:
            return kind, event
    return None


@torch.no_grad()
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
    boundary_layer_override: int | None,
) -> dict[str, Any]:
    prompt_id = str(manifest_record.get("prompt_id"))
    pt_layers = steering_adapter.get_layers(models["pt"])
    it_layers = steering_adapter.get_layers(models["it"])
    boundary_layer = (
        int(boundary_layer_override)
        if boundary_layer_override is not None
        else _late_boundary(model_name)
    )
    n_layers = min(len(pt_layers), len(it_layers))
    boundary_info = {
        "boundary_layer": int(boundary_layer),
        "boundary_source": "cli_override" if boundary_layer_override is not None else "depth_ablation_windows",
        "boundary_mode": "full_late" if boundary_layer_override is None else "override",
        "downstream_stack": f"layers_{int(boundary_layer)}_{int(n_layers) - 1}_plus_readout",
    }
    if boundary_layer < 0 or boundary_layer >= n_layers:
        raise ValueError(f"Boundary layer {boundary_layer} outside {model_name} layer range")

    event_pair = _first_event(manifest_record, event_kinds)
    if event_pair is None:
        return {
            "experiment": "exp40_prelate_commitment_control",
            "model": model_name,
            "prompt_id": prompt_id,
            "prompt_mode": prompt_mode,
            "valid": False,
            "validation": {"ok": False, "reason": "missing_requested_event"},
            **boundary_info,
        }
    event_kind, event = event_pair
    y_pt = int(event["pt_token"]["token_id"])
    y_it = int(event["it_token"]["token_id"])
    prefix_ids = _prefix_ids_for_event(manifest_record, event)
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
        return {
            "experiment": "exp40_prelate_commitment_control",
            "model": model_name,
            "prompt_id": prompt_id,
            "prompt_mode": prompt_mode,
            "event_kind": event_kind,
            "event": event,
            "valid": False,
            "validation": validation,
            **boundary_info,
        }

    raw_prompt = get_prompt_for_variant(
        dataset_record,
        variant="pt",
        tokenizer=tokenizers["pt"],
        apply_chat_template=False,
    )
    prompt_ids = tokenizers["pt"].encode(raw_prompt, add_special_tokens=True)
    full_ids = prompt_ids + prefix_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    forwards = {
        "pt": _forward_boundary(
            model=models["pt"],
            layer=pt_layers[boundary_layer],
            input_ids=input_ids,
            attention_mask=attention_mask,
        ),
        "it": _forward_boundary(
            model=models["it"],
            layer=it_layers[boundary_layer],
            input_ids=input_ids,
            attention_mask=attention_mask,
        ),
    }
    boundary_readouts = {
        variant: _score_boundary_state(
            state=payload["boundary_state"],
            readouts=readouts,
            y_pt=y_pt,
            y_it=y_it,
        )
        for variant, payload in forwards.items()
    }
    final_native = {
        variant: _score_final_logits(
            logits=payload["final_logits"],
            readouts=readouts,
            native_variant=variant,
            y_pt=y_pt,
            y_it=y_it,
        )
        for variant, payload in forwards.items()
    }
    boundary_stats = {
        variant: _boundary_stats(payload["boundary_state"])
        for variant, payload in forwards.items()
    }

    return {
        "experiment": "exp40_prelate_commitment_control",
        "model": model_name,
        "prompt_id": prompt_id,
        "prompt_mode": prompt_mode,
        "event_kind": event_kind,
        "event": event,
        "generated_position": int(event["step"]),
        "valid": True,
        "validation": validation,
        **boundary_info,
        "prefix_length": len(prefix_ids),
        "full_length": len(full_ids),
        "pt_token_id": y_pt,
        "it_token_id": y_it,
        "pt_token_text": validation.get("pt_token_text") or _token_text(tokenizers["pt"], y_pt),
        "it_token_text": validation.get("it_token_text") or _token_text(tokenizers["pt"], y_it),
        "boundary_readouts": boundary_readouts,
        "final_native": final_native,
        "boundary_stats": boundary_stats,
    }


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
    manifest_path = _find_manifest(args.exp20_root, args.exp20_fallback_root, args.prompt_mode, args.model)
    log.info(
        "[exp40] model=%s worker=%d/%d rows=%d done=%d manifest=%s out=%s",
        args.model,
        args.worker_index,
        args.n_workers,
        len(manifest_rows),
        len(done),
        manifest_path,
        out_path,
    )
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for idx, manifest_record in enumerate(manifest_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            if prompt_id in done:
                continue
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                log.warning("[exp40] missing dataset record prompt_id=%s", prompt_id)
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
                    boundary_layer_override=args.boundary_layer_override,
                )
                fout.write(json.dumps(result, separators=(",", ":")) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[exp40] prompt %s failed: %s", prompt_id, exc)
            if (idx + 1) % 10 == 0:
                log.info("[exp40] %s/%s %d/%d prompts", args.prompt_mode, args.model, idx + 1, len(manifest_rows))


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    seen: set[str] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(n_workers):
            path = out_dir / f"records_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp40] missing worker file %s", path)
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
    parser = argparse.ArgumentParser(description="Collect Exp40 pre-late commitment records.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["raw_shared"], default="raw_shared")
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=["first_diff"])
    parser.add_argument("--boundary-layer-override", type=int, default=None)
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
