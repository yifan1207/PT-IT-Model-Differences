from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter


VALID_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b", "deepseek_v2_lite"]
DEFAULT_BATCH_SIZE = {
    "gemma3_4b": 96,
    "qwen3_4b": 96,
    "llama31_8b": 64,
    "mistral_7b": 64,
    "olmo2_7b": 64,
    "deepseek_v2_lite": 64,
}


class _RealTokenMaskProcessor(LogitsProcessor):
    def __init__(self, bad_token_mask: torch.Tensor) -> None:
        self.bad_token_mask = bad_token_mask

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[:, self.bad_token_mask.to(scores.device)] = float("-inf")
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a teacher-token manifest for exp16 replay.")
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--dataset", default="data/exp3_dataset.jsonl")
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--variant", default="pt", choices=["pt", "it"])
    parser.add_argument(
        "--prompt-mode",
        default=None,
        choices=["raw_format_b", "it_chat_template"],
        help="Defaults to raw_format_b for PT and it_chat_template for IT.",
    )
    parser.add_argument(
        "--pipeline-name",
        default=None,
        help="Defaults to A_pt_raw for PT and C_it_chat for IT.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _safe_decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _write_jsonl(path: Path, rows: list[dict], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = row.get("prompt_id") or row.get("id")
            if prompt_id is not None:
                done.add(str(prompt_id))
    return done


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def main() -> None:
    args = parse_args()
    _configure_reproducibility(args.seed)

    prompt_mode = args.prompt_mode or ("raw_format_b" if args.variant == "pt" else "it_chat_template")
    pipeline_name = args.pipeline_name or ("A_pt_raw" if args.variant == "pt" else "C_it_chat")
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.model]
    out_path = Path(args.out)

    manifest_rows = _load_jsonl(Path(args.prompt_manifest))
    if args.limit_prompts is not None:
        manifest_rows = manifest_rows[: args.limit_prompts]

    dataset_rows = load_dataset(args.dataset)
    dataset_by_id = {str(row["id"]): row for row in dataset_rows}
    ordered_records: list[dict] = []
    missing_ids: list[str] = []
    for row in manifest_rows:
        prompt_id = str(row.get("id") or row.get("prompt_id"))
        record = dataset_by_id.get(prompt_id)
        if record is None:
            missing_ids.append(prompt_id)
            continue
        ordered_records.append(record)
    if missing_ids:
        raise ValueError(
            f"Prompt manifest contains {len(missing_ids)} ids missing from dataset; first missing={missing_ids[:5]}"
        )

    if args.resume:
        done_ids = _load_done_ids(out_path)
        ordered_records = [record for record in ordered_records if str(record["id"]) not in done_ids]
    else:
        done_ids = set()
        if out_path.exists():
            out_path.unlink()

    if not ordered_records:
        print(f"[teacher-manifest] already complete -> {out_path}", flush=True)
        return

    spec = get_spec(args.model)
    model_id = model_id_for_variant(spec, args.variant)
    dtype = _dtype_from_name(args.dtype)
    model, tokenizer = load_model_and_tokenizer(model_id, args.device, dtype=dtype)
    steering_adapter = get_steering_adapter(args.model)
    _ensure_pad_token(tokenizer)

    device = next(model.parameters()).device
    real_token_mask = steering_adapter.real_token_mask(tokenizer, device, model_raw=model)
    bad_token_mask = (~real_token_mask).to(device)
    eos_token_ids = [int(tok) for tok in (tokenizer.eos_token_id,) if tok is not None]
    apply_chat_template = prompt_mode == "it_chat_template"

    prompt_text_by_id = {
        str(record["id"]): get_prompt_for_variant(
            record,
            variant=args.variant,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
        )
        for record in ordered_records
    }

    idx = 0
    current_batch_size = max(1, batch_size)
    original_padding_side = tokenizer.padding_side
    try:
        while idx < len(ordered_records):
            batch_records = ordered_records[idx : idx + current_batch_size]
            prompts = [prompt_text_by_id[str(record["id"])] for record in batch_records]
            tokenizer.padding_side = "left"
            try:
                encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()
                else:
                    attention_mask = attention_mask.to(device)

                logits_processor = LogitsProcessorList([_RealTokenMaskProcessor(bad_token_mask)])
                with torch.no_grad():
                    out_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        eos_token_id=eos_token_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        logits_processor=logits_processor,
                        use_cache=True,
                    )
            except RuntimeError as exc:
                if not _is_oom_error(exc) or current_batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                print(
                    f"[teacher-manifest] OOM for {args.model}, retrying with batch_size={current_batch_size}",
                    flush=True,
                )
                continue

            prompt_len = input_ids.shape[1]
            rows: list[dict] = []
            for row_idx, record in enumerate(batch_records):
                new_ids = [int(tok) for tok in out_ids[row_idx, prompt_len:].tolist() if tok != tokenizer.pad_token_id]
                rows.append(
                    {
                        "prompt_id": str(record["id"]),
                        "pipeline": pipeline_name,
                        "prompt_mode": prompt_mode,
                        "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
                        "token_ids": new_ids,
                        "generated_tokens": [
                            {"token_id": token_id, "token_str": _safe_decode(tokenizer, token_id)}
                            for token_id in new_ids
                        ],
                    }
                )
            _write_jsonl(out_path, rows, append=True)
            idx += len(batch_records)
            print(f"[teacher-manifest] {args.model} {idx}/{len(ordered_records)}", flush=True)
    finally:
        tokenizer.padding_side = original_padding_side

    print(f"[teacher-manifest] complete -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
