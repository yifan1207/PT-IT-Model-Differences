"""Du 2025 truthfulness-direction extraction for the local six-model registry.

This implementation is intentionally simple and transparent:
  - it extracts last-prompt-token residual activations at every layer
  - it computes a layerwise difference-in-means direction for true vs false rows
  - it stores PT and IT outputs separately so cross-variant similarity can be
    measured later by analysis code

It is a local cross-model replication utility, not a claim that we have already
reproduced every paper-specific dataset and evaluation choice.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from src.poc.exp17_behavioral_direction_replication.shared import (
    VALID_MODELS,
    VALID_VARIANTS,
    accumulate_class_sums,
    build_prompts,
    load_exp17_model,
    normalize_vector,
    parse_binary_label,
    read_records,
    result_dir,
    write_json,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Du-style truthfulness directions.")
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--variant", required=True, choices=VALID_VARIANTS)
    parser.add_argument("--dataset", required=True, help="JSONL file with text and truth labels.")
    parser.add_argument("--text-field", default="statement")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def _load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    rows = read_records(args.dataset)
    if args.limit is not None:
        rows = rows[:args.limit]
    labels = [
        parse_binary_label(
            row[args.label_field],
            positive_values={"true", "t", "1", "yes", "positive"},
            negative_values={"false", "f", "0", "no", "negative"},
        )
        for row in rows
    ]
    mapped = ["true" if label == "positive" else "false" for label in labels]
    return rows, mapped


def main() -> None:
    args = parse_args()
    rows, labels = _load_rows(args)
    n_true = sum(label == "true" for label in labels)
    n_false = sum(label == "false" for label in labels)
    log.info(
        "exp17 truthfulness start model=%s variant=%s dataset=%s rows=%d true=%d false=%d batch_size=%d max_length=%d device=%s",
        args.model,
        args.variant,
        args.dataset,
        len(rows),
        n_true,
        n_false,
        args.batch_size,
        args.max_length,
        args.device,
    )
    if n_true == 0 or n_false == 0:
        raise ValueError(f"Dataset must contain both true and false rows, got true={n_true}, false={n_false}")

    loaded = load_exp17_model(
        args.model,
        variant=args.variant,
        device=args.device,
        dtype=_dtype_from_name(args.dtype),
    )
    prompts = build_prompts(
        rows,
        text_field=args.text_field,
        loaded=loaded,
        apply_chat_template=not args.no_chat_template,
        prefix=args.prefix,
        suffix=args.suffix,
    )

    sums, counts = accumulate_class_sums(
        loaded,
        prompts=prompts,
        labels=labels,
        class_names=("true", "false"),
        batch_size=args.batch_size,
        max_length=args.max_length,
        progress_desc=f"exp17 truthfulness {args.model}/{args.variant}",
    )

    out_dir = Path(args.out_dir) if args.out_dir else result_dir("du_truthfulness", args.model, args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": args.model,
        "variant": args.variant,
        "dataset": args.dataset,
        "text_field": args.text_field,
        "label_field": args.label_field,
        "apply_chat_template": not args.no_chat_template,
        "counts": counts,
        "directions": {},
        "true_mean": {},
        "false_mean": {},
        "raw_direction_norms": {},
    }

    peak_layer = None
    peak_norm = -1.0
    for layer_idx in range(loaded.spec.n_layers):
        true_mean = (sums["true"][layer_idx] / counts["true"]).to(torch.float32)
        false_mean = (sums["false"][layer_idx] / counts["false"]).to(torch.float32)
        direction, raw_norm = normalize_vector(true_mean - false_mean)
        payload["true_mean"][f"layer_{layer_idx}"] = true_mean
        payload["false_mean"][f"layer_{layer_idx}"] = false_mean
        payload["directions"][f"layer_{layer_idx}"] = direction
        payload["raw_direction_norms"][f"layer_{layer_idx}"] = raw_norm
        if raw_norm > peak_norm:
            peak_norm = raw_norm
            peak_layer = layer_idx

    torch.save(payload, out_dir / "directions.pt")
    write_json(
        out_dir / "summary.json",
        {
            "component": "du_truthfulness",
            "model": args.model,
            "variant": args.variant,
            "dataset": args.dataset,
            "text_field": args.text_field,
            "label_field": args.label_field,
            "counts": counts,
            "n_layers": loaded.spec.n_layers,
            "d_model": loaded.spec.d_model,
            "apply_chat_template": not args.no_chat_template,
            "peak_direction_norm_layer": peak_layer,
            "peak_direction_norm": peak_norm,
            "selected_layer": None,
            "selected_layer_note": "No paper-faithful best-layer selection yet; using per-layer extraction only.",
        },
    )
    log.info("exp17 truthfulness saved directions -> %s", out_dir)


if __name__ == "__main__":
    main()
