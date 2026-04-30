"""Estimate PT-level residual-opposition ratios for Exp26 ``ptlevel_opp``."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
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
from src.poc.exp26_residual_opposition_mediation.hooks import LateMlpOppositionModifier
from src.poc.exp26_residual_opposition_mediation.variants import finite_mean


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@torch.no_grad()
def _diagnostic_forward(
    *,
    model: Any,
    steering_adapter: Any,
    late_layers: list[int],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    variant_name: str,
    prompt_id: str,
) -> dict[str, Any]:
    modifier = LateMlpOppositionModifier(
        model=model,
        steering_adapter=steering_adapter,
        late_layers=late_layers,
        variant="full",
        prompt_id=prompt_id,
        cell_name=f"{variant_name}_calibration",
    )
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return modifier.summary()
    finally:
        modifier.close()


def run_calibration(args: argparse.Namespace) -> None:
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
    dataset_by_id = _dataset_lookup(args.dataset)
    manifest_rows = _load_manifest_records(
        exp20_root=args.exp20_root,
        exp20_fallback_root=args.exp20_fallback_root,
        prompt_mode=args.prompt_mode,
        model=args.model,
        n_examples=args.n_eval_examples,
        worker_index=0,
        n_workers=1,
    )
    boundary_layer = _late_boundary(args.model)
    late_layers = list(range(boundary_layer, len(steering_adapter.get_layers(it_model))))
    values: dict[str, dict[int, list[float]]] = {
        "pt": defaultdict(list),
        "it": defaultdict(list),
    }
    kept_events = 0
    skipped = 0
    for idx, manifest_record in enumerate(manifest_rows):
        prompt_id = str(manifest_record.get("prompt_id"))
        dataset_record = dataset_by_id.get(prompt_id)
        if dataset_record is None:
            skipped += 1
            continue
        raw_prompt = get_prompt_for_variant(
            dataset_record,
            variant="pt",
            tokenizer=pt_tokenizer,
            apply_chat_template=False,
        )
        prompt_ids = pt_tokenizer.encode(raw_prompt, add_special_tokens=True)
        for event_kind, event in _unique_events(manifest_record, [args.event_kind]):
            if "duplicate_of" in event:
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
                skipped += 1
                continue
            full_ids = prompt_ids + prefix_ids
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            for variant in ("pt", "it"):
                summary = _diagnostic_forward(
                    model=models[variant],
                    steering_adapter=steering_adapter,
                    late_layers=late_layers,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    variant_name=variant,
                    prompt_id=prompt_id,
                )
                for layer, payload in (summary.get("by_layer") or {}).items():
                    value = payload.get("mean_opp_norm_frac")
                    if value is not None:
                        values[variant][int(layer)].append(float(value))
            kept_events += 1
        if (idx + 1) % 10 == 0:
            log.info("[exp26-calibrate] %s %d/%d rows kept_events=%d", args.model, idx + 1, len(manifest_rows), kept_events)

    alpha_by_layer: dict[str, float] = {}
    pt_means: dict[str, float | None] = {}
    it_means: dict[str, float | None] = {}
    for layer in late_layers:
        pt_mean = finite_mean(values["pt"][layer])
        it_mean = finite_mean(values["it"][layer])
        pt_means[str(layer)] = pt_mean
        it_means[str(layer)] = it_mean
        if pt_mean is None or it_mean is None or it_mean <= 1e-12:
            alpha = 0.0
        else:
            alpha = max(0.0, min(1.0, float(pt_mean / it_mean)))
        alpha_by_layer[str(layer)] = alpha

    payload = {
        "experiment": "exp26_ptlevel_opp_calibration",
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "event_kind": args.event_kind,
        "n_manifest_rows": len(manifest_rows),
        "kept_events": kept_events,
        "skipped": skipped,
        "late_layers": late_layers,
        "pt_mean_opp_norm_frac_by_layer": pt_means,
        "it_mean_opp_norm_frac_by_layer": it_means,
        "alpha_by_layer": alpha_by_layer,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"out_path": str(args.out_path), "kept_events": kept_events, "skipped": skipped}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Exp26 ptlevel_opp alpha by late layer.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-mode", choices=["raw_shared", "native"], default="raw_shared")
    parser.add_argument("--event-kind", choices=["first_diff", "first_nonformat_diff", "first_assistant_marker_diff"], default="first_diff")
    parser.add_argument("--n-eval-examples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    run_calibration(parse_args())


if __name__ == "__main__":
    main()

