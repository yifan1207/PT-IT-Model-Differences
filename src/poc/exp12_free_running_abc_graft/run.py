from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import SteeringAdapter, get_steering_adapter


VALID_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b", "deepseek_v2_lite"]
DEFAULT_BATCH_SIZE = {
    "gemma3_4b": 64,
    "qwen3_4b": 64,
    "llama31_8b": 32,
    "mistral_7b": 32,
    "olmo2_7b": 32,
    "deepseek_v2_lite": 48,
}
DEFAULT_MAX_NEW_TOKENS = {
    "gemma3_4b": 512,
    "qwen3_4b": 512,
    "llama31_8b": 512,
    "mistral_7b": 512,
    "olmo2_7b": 512,
    "deepseek_v2_lite": 64,
}


class _RealTokenMaskProcessor(LogitsProcessor):
    """Suppress placeholder / reserved tokens during generation."""

    def __init__(self, bad_token_mask: torch.Tensor) -> None:
        self.bad_token_mask = bad_token_mask

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[:, self.bad_token_mask.to(scores.device)] = float("-inf")
        return scores


class _LateMLPGraft:
    """Swap PT late-layer MLP blocks with IT late-layer MLP blocks at runtime."""

    def __init__(
        self,
        *,
        pt_model_raw,
        it_model_raw,
        steering_adapter: SteeringAdapter,
        onset_layer: int,
    ) -> None:
        self.pt_model_raw = pt_model_raw
        self.it_model_raw = it_model_raw
        self.steering_adapter = steering_adapter
        self.onset_layer = onset_layer
        self._handles: list[Any] = []

    def __enter__(self) -> "_LateMLPGraft":
        pt_layers = self.steering_adapter.get_layers(self.pt_model_raw)
        it_layers = self.steering_adapter.get_layers(self.it_model_raw)
        for layer_idx in range(self.onset_layer, len(pt_layers)):
            pt_mlp = pt_layers[layer_idx].mlp
            it_mlp = it_layers[layer_idx].mlp

            def graft_hook(_module, args, _output, *, graft_mlp=it_mlp):
                with torch.no_grad():
                    return graft_mlp(args[0])

            self._handles.append(pt_mlp.register_forward_hook(graft_hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp12 free-running A/B/C raw-output comparison.")
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--n-prompts", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-seed", type=int, default=None)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--onset-layer", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="Subset of free-running pipelines to execute.",
    )
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


def _sample_prompts(records: list[dict], n_prompts: int, seed: int, categories: list[str] | None) -> list[dict]:
    if categories:
        allowed = set(categories)
        records = [r for r in records if r.get("category") in allowed]
    if n_prompts >= len(records):
        return sorted(records, key=lambda r: (r.get("category", ""), r["id"]))

    by_category: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_category[rec.get("category", "")].append(rec)

    rng = random.Random(seed)
    cats = sorted(by_category)
    total = sum(len(v) for v in by_category.values())
    targets = {cat: min(len(by_category[cat]), int(len(by_category[cat]) * n_prompts / total)) for cat in cats}
    allocated = sum(targets.values())
    remainders = sorted(
        (((len(by_category[cat]) * n_prompts / total) - targets[cat], cat) for cat in cats),
        reverse=True,
    )
    idx = 0
    while allocated < n_prompts and idx < len(remainders):
        _, cat = remainders[idx]
        if targets[cat] < len(by_category[cat]):
            targets[cat] += 1
            allocated += 1
        idx += 1
        if idx == len(remainders) and allocated < n_prompts:
            idx = 0

    selected: list[dict] = []
    for cat in cats:
        pool = list(by_category[cat])
        rng.shuffle(pool)
        selected.extend(sorted(pool[:targets[cat]], key=lambda r: r["id"]))
    return sorted(selected, key=lambda r: (r.get("category", ""), r["id"]))[:n_prompts]


def _random_subsample(records: list[dict], n_prompts: int, seed: int) -> list[dict]:
    if n_prompts >= len(records):
        return sorted(records, key=lambda r: r["id"])
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: r["id"])
    picked = rng.sample(pool, n_prompts)
    return sorted(picked, key=lambda r: r["id"])


def _apply_prompt_shard(prompts: list[dict], shard_index: int, num_shards: int) -> list[dict]:
    if num_shards <= 1:
        return prompts
    return prompts[shard_index::num_shards]


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _safe_decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _load_done_pairs(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done.add((row["condition"], row["record_id"]))
            except Exception:
                continue
    return done


def _write_jsonl(path: Path, rows: list[dict], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _generate_batch_rows(
    *,
    condition: str,
    pipeline: str,
    batch_records: list[dict],
    raw_prompt_by_id: dict[str, str],
    model_prompt_by_id: dict[str, str],
    tokenizer,
    model_raw,
    real_token_mask: torch.Tensor,
    eos_token_ids: list[int],
    max_new_tokens: int,
    model_name: str,
    prompt_mode: str,
    onset_layer: int | None,
    hook_context,
) -> list[dict]:
    device = next(model_raw.parameters()).device
    prompts = [model_prompt_by_id[record["id"]] for record in batch_records]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
        else:
            attention_mask = attention_mask.to(device)

        logits_processor = LogitsProcessorList([_RealTokenMaskProcessor((~real_token_mask).to(device))])

        with hook_context:
            with torch.no_grad():
                out_ids = model_raw.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    logits_processor=logits_processor,
                    use_cache=True,
                )
    finally:
        tokenizer.padding_side = original_padding_side

    prompt_len = input_ids.shape[1]
    rows: list[dict] = []
    for row_idx, record in enumerate(batch_records):
        new_ids = [int(tok) for tok in out_ids[row_idx, prompt_len:].tolist() if tok != tokenizer.pad_token_id]
        generated_tokens = [{"token_id": token_id, "token_str": _safe_decode(tokenizer, token_id)} for token_id in new_ids]
        rows.append(
            {
                "condition": condition,
                "pipeline": pipeline,
                "record_id": record["id"],
                "prompt_id": record["id"],
                "category": record.get("category", ""),
                "prompt": raw_prompt_by_id[record["id"]],
                "model_prompt": model_prompt_by_id[record["id"]],
                "prompt_mode": prompt_mode,
                "model_name": model_name,
                "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
                "generated_tokens": generated_tokens,
                "generated_token_ids": new_ids,
                "onset_layer": onset_layer,
            }
        )
    return rows


def _run_condition(
    *,
    condition: str,
    pipeline: str,
    prompts: list[dict],
    raw_prompt_by_id: dict[str, str],
    model_prompt_by_id: dict[str, str],
    tokenizer,
    model_raw,
    real_token_mask: torch.Tensor,
    eos_token_ids: list[int],
    max_new_tokens: int,
    out_path: Path,
    done_pairs: set[tuple[str, str]],
    model_name: str,
    prompt_mode: str,
    onset_layer: int | None,
    batch_size: int,
    hook_factory,
) -> int:
    remaining = [record for record in prompts if (condition, record["id"]) not in done_pairs]
    if not remaining:
        print(f"[exp12] {condition}: already complete", flush=True)
        return 0

    print(f"[exp12] {condition}: {len(remaining)} prompts", flush=True)
    idx = 0
    current_batch_size = max(1, batch_size)
    while idx < len(remaining):
        batch = remaining[idx : idx + current_batch_size]
        try:
            rows = _generate_batch_rows(
                condition=condition,
                pipeline=pipeline,
                batch_records=batch,
                raw_prompt_by_id=raw_prompt_by_id,
                model_prompt_by_id=model_prompt_by_id,
                tokenizer=tokenizer,
                model_raw=model_raw,
                real_token_mask=real_token_mask,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                model_name=model_name,
                prompt_mode=prompt_mode,
                onset_layer=onset_layer,
                hook_context=hook_factory(),
            )
            _write_jsonl(out_path, rows, append=True)
            for row in rows:
                done_pairs.add((row["condition"], row["record_id"]))
            idx += len(batch)
            print(f"[exp12] {condition}: {idx}/{len(remaining)}", flush=True)
        except RuntimeError as exc:
            if not _is_oom_error(exc) or current_batch_size == 1:
                raise
            torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
            print(f"[exp12] {condition}: OOM, retrying with batch_size={current_batch_size}", flush=True)
    return len(remaining)


def main() -> None:
    args = parse_args()
    _configure_reproducibility(args.seed)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    spec = get_spec(args.model)
    onset_layer = spec.corrective_onset if args.onset_layer is None else args.onset_layer
    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[args.model]
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.model]
    out_dir = Path(args.out_dir or f"results/exp12_free_running_abc_graft/{args.model}/abc_raw_eval_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_outputs_path = out_dir / "sample_outputs.jsonl"
    if not args.resume and sample_outputs_path.exists():
        sample_outputs_path.unlink()

    dtype = _dtype_from_name(args.dtype)
    dataset = load_dataset(args.dataset)
    if args.prompt_seed is not None:
        pool = dataset
        if args.categories:
            allowed = set(args.categories)
            pool = [record for record in dataset if record.get("category") in allowed]
        prompts = _random_subsample(pool, args.n_prompts, args.prompt_seed)
    else:
        prompts = _sample_prompts(dataset, args.n_prompts, args.seed, args.categories)
    prompts = _apply_prompt_shard(prompts, args.shard_index, args.num_shards)
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]

    pt_id = model_id_for_variant(spec, "pt")
    it_id = model_id_for_variant(spec, "it")
    pt_model, tokenizer_pt = load_model_and_tokenizer(pt_id, args.device, dtype=dtype)
    it_model, tokenizer_it = load_model_and_tokenizer(it_id, args.device, dtype=dtype)
    _ensure_pad_token(tokenizer_pt)
    _ensure_pad_token(tokenizer_it)

    steering_adapter = get_steering_adapter(args.model)
    pt_model.requires_grad_(False)
    it_model.requires_grad_(False)
    device = next(pt_model.parameters()).device
    real_token_mask_pt = steering_adapter.real_token_mask(tokenizer_pt, device, model_raw=pt_model)
    real_token_mask_it = steering_adapter.real_token_mask(tokenizer_it, device, model_raw=it_model)
    eos_ids_pt = steering_adapter.eos_token_ids(tokenizer_pt)
    eos_ids_it = steering_adapter.eos_token_ids(tokenizer_it)

    raw_prompt_by_id = {
        record["id"]: get_prompt_for_variant(
            record,
            variant="pt",
            tokenizer=tokenizer_pt,
            apply_chat_template=False,
        )
        for record in prompts
    }
    it_chat_prompt_by_id = {
        record["id"]: get_prompt_for_variant(
            record,
            variant="it",
            tokenizer=tokenizer_it,
            apply_chat_template=True,
        )
        for record in prompts
    }

    config = {
        "experiment": "exp12_abc_raw_eval_v1",
        "model": args.model,
        "n_layers": spec.n_layers,
        "pt_model_id": pt_id,
        "it_model_id": it_id,
        "dataset": args.dataset,
        "n_prompts_requested": args.n_prompts,
        "n_prompts_after_sharding": len(prompts),
        "seed": args.seed,
        "prompt_seed": args.prompt_seed,
        "categories": args.categories,
        "onset_layer": onset_layer,
        "max_new_tokens": max_new_tokens,
        "dtype": args.dtype,
        "batch_size_requested": batch_size,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "pipelines": args.pipelines,
        "conditions": {
            "A": {"condition": "A_pt_raw", "prompt_mode": "raw_format_b"},
            "B": {"condition": "B_graft_raw", "prompt_mode": "raw_format_b"},
            "C": {"condition": "C_it_chat", "prompt_mode": "it_chat_template"},
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    _write_jsonl(out_dir / "prompts.jsonl", prompts, append=False)

    done_pairs = _load_done_pairs(sample_outputs_path) if args.resume else set()
    total_written = 0

    if "A" in args.pipelines:
        total_written += _run_condition(
            condition="A_pt_raw",
            pipeline="A",
            prompts=prompts,
            raw_prompt_by_id=raw_prompt_by_id,
            model_prompt_by_id=raw_prompt_by_id,
            tokenizer=tokenizer_pt,
            model_raw=pt_model,
            real_token_mask=real_token_mask_pt,
            eos_token_ids=eos_ids_pt,
            max_new_tokens=max_new_tokens,
            out_path=sample_outputs_path,
            done_pairs=done_pairs,
            model_name=args.model,
            prompt_mode="raw_format_b",
            onset_layer=None,
            batch_size=batch_size,
            hook_factory=nullcontext,
        )
        torch.cuda.empty_cache()

    if "B" in args.pipelines:
        total_written += _run_condition(
            condition="B_graft_raw",
            pipeline="B",
            prompts=prompts,
            raw_prompt_by_id=raw_prompt_by_id,
            model_prompt_by_id=raw_prompt_by_id,
            tokenizer=tokenizer_pt,
            model_raw=pt_model,
            real_token_mask=real_token_mask_pt,
            eos_token_ids=eos_ids_pt,
            max_new_tokens=max_new_tokens,
            out_path=sample_outputs_path,
            done_pairs=done_pairs,
            model_name=args.model,
            prompt_mode="raw_format_b",
            onset_layer=onset_layer,
            batch_size=batch_size,
            hook_factory=lambda: _LateMLPGraft(
                pt_model_raw=pt_model,
                it_model_raw=it_model,
                steering_adapter=steering_adapter,
                onset_layer=onset_layer,
            ),
        )
        torch.cuda.empty_cache()

    if "C" in args.pipelines:
        total_written += _run_condition(
            condition="C_it_chat",
            pipeline="C",
            prompts=prompts,
            raw_prompt_by_id=raw_prompt_by_id,
            model_prompt_by_id=it_chat_prompt_by_id,
            tokenizer=tokenizer_it,
            model_raw=it_model,
            real_token_mask=real_token_mask_it,
            eos_token_ids=eos_ids_it,
            max_new_tokens=max_new_tokens,
            out_path=sample_outputs_path,
            done_pairs=done_pairs,
            model_name=args.model,
            prompt_mode="it_chat_template",
            onset_layer=None,
            batch_size=batch_size,
            hook_factory=nullcontext,
        )
        torch.cuda.empty_cache()

    counts_by_condition = Counter()
    if sample_outputs_path.exists():
        with open(sample_outputs_path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts_by_condition[row.get("condition", "")] += 1
    summary = {
        "model": args.model,
        "onset_layer": onset_layer,
        "n_prompt_records": len(prompts),
        "pipelines": args.pipelines,
        "counts_by_condition": dict(counts_by_condition),
        "sample_outputs_path": str(sample_outputs_path),
        "new_rows_written_this_run": total_written,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[exp12] done: {dict(counts_by_condition)}", flush=True)


if __name__ == "__main__":
    main()
