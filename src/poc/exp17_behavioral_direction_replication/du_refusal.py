"""Du 2025 refusal-direction candidate extraction for the local six-model registry.

Current scope:
  - load harmful and harmless prompt sets
  - capture last-prompt-token residuals at every layer
  - compute mean(harmful) - mean(harmless) candidate directions

This is a useful starting point for cross-model replication, but it is not yet
the full paper-faithful Arditi-style `(layer, token-position)` selection sweep.
The summary file records that limitation explicitly.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
from pathlib import Path
from typing import Any

import requests
import torch

from src.poc.exp17_behavioral_direction_replication.shared import (
    VALID_MODELS,
    VALID_VARIANTS,
    accumulate_class_sums,
    build_prompts,
    load_exp17_model,
    normalize_vector,
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
    parser = argparse.ArgumentParser(description="Extract candidate refusal directions.")
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--variant", required=True, choices=VALID_VARIANTS)
    parser.add_argument("--harmful-dataset", default="", help="JSONL/CSV harmful prompts.")
    parser.add_argument("--harmless-dataset", default="", help="JSONL/CSV harmless prompts.")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--keep-uneven", action="store_true")
    parser.add_argument(
        "--use-upstream-du-sources",
        action="store_true",
        help="Fetch the exact harmful/harmless instruction sources used in the upstream Du refusal code.",
    )
    parser.add_argument("--source-split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--n-inst-train",
        type=int,
        default=128,
        help="When using upstream Du sources, keep the first N instructions from the chosen split.",
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def _load_upstream_du_sources(split: str) -> tuple[list[str], list[str]]:
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    harmful_csv = list(csv.DictReader(io.StringIO(response.text)))
    harmful_instructions = [row["goal"] for row in harmful_csv if row.get("goal")]

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for --use-upstream-du-sources "
            "(upstream code uses load_dataset('tatsu-lab/alpaca'))."
        ) from exc

    try:
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for --use-upstream-du-sources "
            "(upstream code uses train_test_split(..., random_state=42))."
        ) from exc

    dataset = load_dataset("tatsu-lab/alpaca")
    harmless_instructions = [
        row["instruction"]
        for row in dataset["train"]
        if str(row.get("input", "")).strip() == ""
    ]

    harmful_train, harmful_test = train_test_split(harmful_instructions, test_size=0.2, random_state=42)
    harmless_train, harmless_test = train_test_split(harmless_instructions, test_size=0.2, random_state=42)
    if split == "train":
        return harmful_train, harmless_train
    return harmful_test, harmless_test


def _prepare_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    source_meta: dict[str, Any]
    if args.use_upstream_du_sources:
        harmful_instructions, harmless_instructions = _load_upstream_du_sources(args.source_split)
        harmful_rows = [{args.prompt_field: text} for text in harmful_instructions[:args.n_inst_train]]
        harmless_rows = [{args.prompt_field: text} for text in harmless_instructions[:args.n_inst_train]]
        source_meta = {
            "source_mode": "upstream_du_exact_sources",
            "source_split": args.source_split,
            "n_inst_train": args.n_inst_train,
            "harmful_source": "advbench harmful_behaviors.csv",
            "harmless_source": "tatsu-lab/alpaca (empty-input instructions)",
        }
    else:
        if not args.harmful_dataset or not args.harmless_dataset:
            raise ValueError(
                "Either provide --harmful-dataset and --harmless-dataset, "
                "or use --use-upstream-du-sources."
            )
        harmful_rows = read_records(args.harmful_dataset)
        harmless_rows = read_records(args.harmless_dataset)
        source_meta = {
            "source_mode": "local_files",
            "harmful_dataset": args.harmful_dataset,
            "harmless_dataset": args.harmless_dataset,
        }

    if args.limit_per_class is not None:
        harmful_rows = harmful_rows[:args.limit_per_class]
        harmless_rows = harmless_rows[:args.limit_per_class]
    if not args.keep_uneven:
        shared_n = min(len(harmful_rows), len(harmless_rows))
        harmful_rows = harmful_rows[:shared_n]
        harmless_rows = harmless_rows[:shared_n]

    rows = harmful_rows + harmless_rows
    labels = (["harmful"] * len(harmful_rows)) + (["harmless"] * len(harmless_rows))
    return rows, labels, source_meta


def main() -> None:
    args = parse_args()
    rows, labels, source_meta = _prepare_rows(args)
    n_harmful = sum(label == "harmful" for label in labels)
    n_harmless = sum(label == "harmless" for label in labels)
    if n_harmful == 0 or n_harmless == 0:
        raise ValueError(
            f"Need both harmful and harmless rows, got harmful={n_harmful}, harmless={n_harmless}"
        )

    loaded = load_exp17_model(
        args.model,
        variant=args.variant,
        device=args.device,
        dtype=_dtype_from_name(args.dtype),
    )
    prompts = build_prompts(
        rows,
        text_field=args.prompt_field,
        loaded=loaded,
        apply_chat_template=not args.no_chat_template,
        prefix=args.prefix,
        suffix=args.suffix,
    )

    sums, counts = accumulate_class_sums(
        loaded,
        prompts=prompts,
        labels=labels,
        class_names=("harmful", "harmless"),
        batch_size=args.batch_size,
        max_length=args.max_length,
        progress_desc=f"exp17 refusal {args.model}/{args.variant}",
    )

    out_dir = Path(args.out_dir) if args.out_dir else result_dir("du_refusal", args.model, args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": args.model,
        "variant": args.variant,
        "harmful_dataset": args.harmful_dataset or None,
        "harmless_dataset": args.harmless_dataset or None,
        "prompt_field": args.prompt_field,
        "apply_chat_template": not args.no_chat_template,
        "extraction_position": "last_prompt_token",
        "counts": counts,
        "source_meta": source_meta,
        "directions": {},
        "harmful_mean": {},
        "harmless_mean": {},
        "raw_direction_norms": {},
    }

    peak_layer = None
    peak_norm = -1.0
    for layer_idx in range(loaded.spec.n_layers):
        harmful_mean = (sums["harmful"][layer_idx] / counts["harmful"]).to(torch.float32)
        harmless_mean = (sums["harmless"][layer_idx] / counts["harmless"]).to(torch.float32)
        direction, raw_norm = normalize_vector(harmful_mean - harmless_mean)
        payload["harmful_mean"][f"layer_{layer_idx}"] = harmful_mean
        payload["harmless_mean"][f"layer_{layer_idx}"] = harmless_mean
        payload["directions"][f"layer_{layer_idx}"] = direction
        payload["raw_direction_norms"][f"layer_{layer_idx}"] = raw_norm
        if raw_norm > peak_norm:
            peak_norm = raw_norm
            peak_layer = layer_idx

    torch.save(payload, out_dir / "directions.pt")
    write_json(
        out_dir / "summary.json",
        {
            "component": "du_refusal",
            "model": args.model,
            "variant": args.variant,
            "harmful_dataset": args.harmful_dataset or None,
            "harmless_dataset": args.harmless_dataset or None,
            "prompt_field": args.prompt_field,
            "counts": counts,
            "n_layers": loaded.spec.n_layers,
            "d_model": loaded.spec.d_model,
            "apply_chat_template": not args.no_chat_template,
            "extraction_position": "last_prompt_token",
            "source_meta": source_meta,
            "peak_direction_norm_layer": peak_layer,
            "peak_direction_norm": peak_norm,
            "selected_layer": None,
            "selected_layer_note": (
                "Candidate extractor only. Full paper-faithful refusal replication "
                "still needs a token-position sweep and generation-based selection."
            ),
        },
    )
    log.info("Saved refusal candidate directions -> %s", out_dir)


if __name__ == "__main__":
    main()
