from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp11_matched_prefix_mlp_graft.run import DEPTH_ABLATION_WINDOWS
from src.poc.exp15_symmetric_behavioral_causality.dataset import (
    SUBSET_NAME,
    SUBSET_SEED,
    build_exp15_core_subset,
    build_human_audit_manifest,
    subset_summary,
    write_human_audit_csv,
    write_jsonl,
)
from src.poc.exp06_corrective_direction_steering.model_adapter import SteeringAdapter, get_steering_adapter


VALID_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b", "deepseek_v2_lite"]
ALL_PIPELINES = [
    "A_pt_raw",
    "B_early_raw",
    "B_mid_raw",
    "B_late_raw",
    "C_it_chat",
    "D_early_ptswap",
    "D_mid_ptswap",
    "D_late_ptswap",
]
DEFAULT_BATCH_SIZE = {
    "gemma3_4b": 64,
    "qwen3_4b": 64,
    "llama31_8b": 32,
    "mistral_7b": 32,
    "olmo2_7b": 32,
    "deepseek_v2_lite": 48,
}
FORCED_CHOICE_PROMPT_FORMAT = "C"
FORCED_CHOICE_MAX_NEW_TOKENS = 3


class _RealTokenMaskProcessor(LogitsProcessor):
    """Suppress placeholder / reserved tokens during generation."""

    def __init__(self, bad_token_mask: torch.Tensor) -> None:
        self.bad_token_mask = bad_token_mask

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[:, self.bad_token_mask.to(scores.device)] = float("-inf")
        return scores


class _WindowedMLPSwap:
    """Swap host MLPs with donor MLPs inside a contiguous layer window."""

    def __init__(
        self,
        *,
        host_model_raw,
        donor_model_raw,
        steering_adapter: SteeringAdapter,
        start_layer: int,
        end_layer_exclusive: int,
    ) -> None:
        self.host_model_raw = host_model_raw
        self.donor_model_raw = donor_model_raw
        self.steering_adapter = steering_adapter
        self.start_layer = start_layer
        self.end_layer_exclusive = end_layer_exclusive
        self._handles: list[Any] = []

    def __enter__(self) -> "_WindowedMLPSwap":
        for layer_idx in range(self.start_layer, self.end_layer_exclusive):
            host_mlp = self.steering_adapter.get_mlp(self.host_model_raw, layer_idx)
            donor_mlp = self.steering_adapter.get_mlp(self.donor_model_raw, layer_idx)

            def graft_hook(_module, args, _output, *, donor=donor_mlp):
                with torch.no_grad():
                    return donor(args[0])

            self._handles.append(host_mlp.register_forward_hook(graft_hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp15 free-running symmetric behavioral causality.")
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--run-name", default=SUBSET_NAME)
    parser.add_argument("--subset-seed", type=int, default=SUBSET_SEED)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=ALL_PIPELINES,
        choices=ALL_PIPELINES,
        help="Subset of Exp15 pipelines to execute.",
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


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _safe_decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _json_hash_from_rows(rows: list[dict]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_prompt_shard(prompts: list[dict], shard_index: int, num_shards: int) -> list[dict]:
    if num_shards <= 1:
        return prompts
    return prompts[shard_index::num_shards]


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _load_done_pairs(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as handle:
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
    with open(path, mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _assert_depth_windows(model: str, spec) -> dict[str, tuple[int, int]]:
    windows = DEPTH_ABLATION_WINDOWS[model]
    expected_width = spec.n_layers - spec.corrective_onset
    for pipeline_name, (start, end) in windows.items():
        if end - start != expected_width:
            raise ValueError(
                f"Depth window width mismatch for {model}/{pipeline_name}: {end - start} != {expected_width}"
            )
        if start < 0 or end > spec.n_layers or start >= end:
            raise ValueError(f"Invalid depth window for {model}/{pipeline_name}: [{start}, {end})")
    return windows


def _build_pipeline_manifest(model: str, spec) -> list[dict[str, Any]]:
    windows = _assert_depth_windows(model, spec)
    manifest = [
        {
            "condition": "A_pt_raw",
            "pipeline_family": "baseline",
            "host_variant": "pt",
            "donor_variant": None,
            "prompt_mode": "raw_format_b",
            "graft_start_layer": None,
            "graft_end_layer_exclusive": None,
        },
        {
            "condition": "B_early_raw",
            "pipeline_family": "sufficiency",
            "host_variant": "pt",
            "donor_variant": "it",
            "prompt_mode": "raw_format_b",
            "graft_start_layer": windows["B_early_raw"][0],
            "graft_end_layer_exclusive": windows["B_early_raw"][1],
        },
        {
            "condition": "B_mid_raw",
            "pipeline_family": "sufficiency",
            "host_variant": "pt",
            "donor_variant": "it",
            "prompt_mode": "raw_format_b",
            "graft_start_layer": windows["B_mid_raw"][0],
            "graft_end_layer_exclusive": windows["B_mid_raw"][1],
        },
        {
            "condition": "B_late_raw",
            "pipeline_family": "sufficiency",
            "host_variant": "pt",
            "donor_variant": "it",
            "prompt_mode": "raw_format_b",
            "graft_start_layer": windows["B_late_raw"][0],
            "graft_end_layer_exclusive": windows["B_late_raw"][1],
        },
        {
            "condition": "C_it_chat",
            "pipeline_family": "baseline",
            "host_variant": "it",
            "donor_variant": None,
            "prompt_mode": "it_chat_template",
            "graft_start_layer": None,
            "graft_end_layer_exclusive": None,
        },
        {
            "condition": "D_early_ptswap",
            "pipeline_family": "necessity",
            "host_variant": "it",
            "donor_variant": "pt",
            "prompt_mode": "it_chat_template",
            "graft_start_layer": windows["B_early_raw"][0],
            "graft_end_layer_exclusive": windows["B_early_raw"][1],
        },
        {
            "condition": "D_mid_ptswap",
            "pipeline_family": "necessity",
            "host_variant": "it",
            "donor_variant": "pt",
            "prompt_mode": "it_chat_template",
            "graft_start_layer": windows["B_mid_raw"][0],
            "graft_end_layer_exclusive": windows["B_mid_raw"][1],
        },
        {
            "condition": "D_late_ptswap",
            "pipeline_family": "necessity",
            "host_variant": "it",
            "donor_variant": "pt",
            "prompt_mode": "it_chat_template",
            "graft_start_layer": windows["B_late_raw"][0],
            "graft_end_layer_exclusive": windows["B_late_raw"][1],
        },
    ]
    return manifest


def _record_for_prompt_format(record: dict, prompt_format: str) -> dict:
    if prompt_format == "B":
        return record
    cloned = json.loads(json.dumps(record))
    formats = dict(cloned.get("formats", {}))
    formats["B"] = record.get("formats", {}).get(prompt_format, formats.get("B", ""))
    cloned["formats"] = formats
    return cloned


def _prompt_map(
    records: list[dict],
    *,
    variant: str,
    tokenizer,
    apply_chat_template: bool,
    prompt_format: str,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for record in records:
        prompt_record = _record_for_prompt_format(record, prompt_format)
        out[record["id"]] = get_prompt_for_variant(
            prompt_record,
            variant=variant,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
        )
    return out


def _generate_batch_rows(
    *,
    condition: str,
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
    host_variant: str,
    start_layer: int | None,
    end_layer_exclusive: int | None,
    hook_context,
    output_kind: str,
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
                "record_id": record["id"],
                "prompt_id": record["id"],
                "category": record.get("category", ""),
                "source": record.get("source", ""),
                "expected_behavior": (
                    record.get("expected_behavior")
                    or record.get("metadata", {}).get("expected_behavior")
                    or ""
                ),
                "exp15_selection_bucket": record.get("exp15_selection_bucket"),
                "exp15_assistant_facing": bool(record.get("exp15_assistant_facing")),
                "exp15_conversation_source": bool(record.get("exp15_conversation_source")),
                "exp15_safety_benign": bool(record.get("exp15_safety_benign")),
                "exp15_safety_harmful": bool(record.get("exp15_safety_harmful")),
                "prompt": raw_prompt_by_id[record["id"]],
                "model_prompt": model_prompt_by_id[record["id"]],
                "prompt_mode": prompt_mode,
                "host_variant": host_variant,
                "model_name": model_name,
                "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
                "generated_tokens": generated_tokens,
                "generated_token_ids": new_ids,
                "graft_start_layer": start_layer,
                "graft_end_layer_exclusive": end_layer_exclusive,
                "output_kind": output_kind,
            }
        )
    return rows


def _run_condition(
    *,
    condition: str,
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
    host_variant: str,
    start_layer: int | None,
    end_layer_exclusive: int | None,
    batch_size: int,
    hook_factory,
    output_kind: str,
) -> int:
    remaining = [record for record in prompts if (condition, record["id"]) not in done_pairs]
    if not remaining:
        print(f"[exp15] {condition}/{output_kind}: already complete", flush=True)
        return 0

    print(f"[exp15] {condition}/{output_kind}: {len(remaining)} prompts", flush=True)
    idx = 0
    current_batch_size = max(1, batch_size)
    while idx < len(remaining):
        batch = remaining[idx : idx + current_batch_size]
        try:
            rows = _generate_batch_rows(
                condition=condition,
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
                host_variant=host_variant,
                start_layer=start_layer,
                end_layer_exclusive=end_layer_exclusive,
                hook_context=hook_factory(),
                output_kind=output_kind,
            )
            _write_jsonl(out_path, rows, append=True)
            for row in rows:
                done_pairs.add((row["condition"], row["record_id"]))
            idx += len(batch)
            print(f"[exp15] {condition}/{output_kind}: {idx}/{len(remaining)}", flush=True)
        except RuntimeError as exc:
            if not _is_oom_error(exc) or current_batch_size == 1:
                raise
            torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
            print(
                f"[exp15] {condition}/{output_kind}: OOM, retrying with batch_size={current_batch_size}",
                flush=True,
            )
    return len(remaining)


def _count_rows_by_condition(path: Path) -> dict[str, int]:
    counts = Counter()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[row.get("condition", "")] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    _configure_reproducibility(args.generation_seed)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    spec = get_spec(args.model)
    steering_adapter = get_steering_adapter(args.model)
    depth_windows = _assert_depth_windows(args.model, spec)
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.model]
    max_new_tokens = args.max_new_tokens or steering_adapter.max_gen_tokens

    run_root = Path(
        args.out_dir
        or f"results/exp15_symmetric_behavioral_causality/data/{args.run_name}_{args.model}"
    )
    default_leaf = f"{args.run_name}_{args.model}"
    if args.num_shards > 1:
        default_leaf = f"{default_leaf}_shard{args.shard_index}of{args.num_shards}"
    out_dir = run_root / default_leaf
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = out_dir / "prompts_shard.jsonl"
    prompts_full_path = out_dir / "prompts_full.jsonl"
    pipeline_manifest_path = out_dir / "pipeline_manifest.json"
    sample_outputs_path = out_dir / "sample_outputs.jsonl"
    forced_choice_outputs_path = out_dir / "forced_choice_outputs.jsonl"
    audit_manifest_path = out_dir / "human_audit_manifest.jsonl"
    audit_template_path = out_dir / "human_audit_template.csv"
    config_path = out_dir / "config.json"
    summary_path = out_dir / "summary.json"

    all_records = load_dataset(args.dataset)
    selected_records = build_exp15_core_subset(
        all_records,
        seed=args.subset_seed,
        subset_name=args.run_name,
    )
    selected_shard = _apply_prompt_shard(selected_records, args.shard_index, args.num_shards)
    if args.limit_prompts is not None:
        selected_shard = selected_shard[: args.limit_prompts]

    audit_manifest = build_human_audit_manifest(selected_records, seed=args.subset_seed)
    audit_rows_with_blanks = []
    for row in audit_manifest:
        audit_rows_with_blanks.append(
            {
                **row,
                "question": "",
                "response": "",
            }
        )

    pipeline_manifest = _build_pipeline_manifest(args.model, spec)
    selected_pipelines = [row for row in pipeline_manifest if row["condition"] in set(args.pipelines)]

    write_jsonl(prompts_path, selected_shard)
    write_jsonl(prompts_full_path, selected_records)
    write_jsonl(audit_manifest_path, audit_manifest)
    write_human_audit_csv(audit_template_path, audit_rows_with_blanks)
    pipeline_manifest_path.write_text(json.dumps(selected_pipelines, indent=2), encoding="utf-8")

    config = {
        "experiment": "exp15_free_running_symmetric_behavioral_causality",
        "model": args.model,
        "pt_model_id": model_id_for_variant(spec, "pt"),
        "it_model_id": model_id_for_variant(spec, "it"),
        "dataset": args.dataset,
        "run_name": args.run_name,
        "subset_seed": args.subset_seed,
        "generation_seed": args.generation_seed,
        "max_new_tokens": max_new_tokens,
        "forced_choice_max_new_tokens": FORCED_CHOICE_MAX_NEW_TOKENS,
        "dtype": args.dtype,
        "batch_size_requested": batch_size,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "limit_prompts": args.limit_prompts,
        "n_prompts_full": len(selected_records),
        "n_prompts_shard": len(selected_shard),
        "pipelines": args.pipelines,
        "dataset_manifest_hash": _file_hash(prompts_full_path),
        "dataset_shard_hash": _file_hash(prompts_path),
        "pipeline_manifest_hash": _file_hash(pipeline_manifest_path),
        "human_audit_manifest_hash": _file_hash(audit_manifest_path),
        "subset_summary": subset_summary(selected_records, audit_manifest),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.prepare_only:
        summary = {
            "model": args.model,
            "prepare_only": True,
            "n_prompt_records": len(selected_shard),
            "n_prompt_records_full": len(selected_records),
            "pipelines": args.pipelines,
            "counts_by_condition": {},
            "forced_choice_counts_by_condition": {},
            "sample_outputs_path": str(sample_outputs_path),
            "forced_choice_outputs_path": str(forced_choice_outputs_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[exp15] prepared manifests in {out_dir}", flush=True)
        return

    if not args.resume and sample_outputs_path.exists():
        sample_outputs_path.unlink()
    if not args.resume and forced_choice_outputs_path.exists():
        forced_choice_outputs_path.unlink()

    dtype = _dtype_from_name(args.dtype)
    pt_model, tokenizer_pt = load_model_and_tokenizer(model_id_for_variant(spec, "pt"), args.device, dtype=dtype)
    it_model, tokenizer_it = load_model_and_tokenizer(model_id_for_variant(spec, "it"), args.device, dtype=dtype)
    _ensure_pad_token(tokenizer_pt)
    _ensure_pad_token(tokenizer_it)

    pt_model.requires_grad_(False)
    it_model.requires_grad_(False)
    device = next(pt_model.parameters()).device
    real_token_mask_pt = steering_adapter.real_token_mask(tokenizer_pt, device, model_raw=pt_model)
    real_token_mask_it = steering_adapter.real_token_mask(tokenizer_it, device, model_raw=it_model)
    eos_ids_pt = steering_adapter.eos_token_ids(tokenizer_pt)
    eos_ids_it = steering_adapter.eos_token_ids(tokenizer_it)

    raw_prompt_by_id = _prompt_map(
        selected_shard,
        variant="pt",
        tokenizer=tokenizer_pt,
        apply_chat_template=False,
        prompt_format="B",
    )
    it_chat_prompt_by_id = _prompt_map(
        selected_shard,
        variant="it",
        tokenizer=tokenizer_it,
        apply_chat_template=True,
        prompt_format="B",
    )
    pt_forced_choice_prompt_by_id = _prompt_map(
        [record for record in selected_shard if record.get("category") == "CONTENT-FACT"],
        variant="pt",
        tokenizer=tokenizer_pt,
        apply_chat_template=False,
        prompt_format=FORCED_CHOICE_PROMPT_FORMAT,
    )
    it_forced_choice_prompt_by_id = _prompt_map(
        [record for record in selected_shard if record.get("category") == "CONTENT-FACT"],
        variant="it",
        tokenizer=tokenizer_it,
        apply_chat_template=True,
        prompt_format=FORCED_CHOICE_PROMPT_FORMAT,
    )

    sample_done_pairs = _load_done_pairs(sample_outputs_path) if args.resume else set()
    forced_choice_done_pairs = _load_done_pairs(forced_choice_outputs_path) if args.resume else set()

    selected_pipeline_map = {row["condition"]: row for row in selected_pipelines}
    total_written = 0
    total_forced_choice_written = 0

    for condition in args.pipelines:
        meta = selected_pipeline_map[condition]
        host_variant = meta["host_variant"]
        donor_variant = meta["donor_variant"]
        start_layer = meta["graft_start_layer"]
        end_layer_exclusive = meta["graft_end_layer_exclusive"]
        is_pt_host = host_variant == "pt"
        tokenizer = tokenizer_pt if is_pt_host else tokenizer_it
        model_raw = pt_model if is_pt_host else it_model
        real_token_mask = real_token_mask_pt if is_pt_host else real_token_mask_it
        eos_token_ids = eos_ids_pt if is_pt_host else eos_ids_it
        prompt_by_id = raw_prompt_by_id if meta["prompt_mode"] == "raw_format_b" else it_chat_prompt_by_id

        if donor_variant is None:
            hook_factory = nullcontext
        else:
            donor_model_raw = it_model if donor_variant == "it" else pt_model
            hook_factory = lambda host=model_raw, donor=donor_model_raw, start=start_layer, end=end_layer_exclusive: _WindowedMLPSwap(
                host_model_raw=host,
                donor_model_raw=donor,
                steering_adapter=steering_adapter,
                start_layer=start,
                end_layer_exclusive=end,
            )

        total_written += _run_condition(
            condition=condition,
            prompts=selected_shard,
            raw_prompt_by_id=raw_prompt_by_id,
            model_prompt_by_id=prompt_by_id,
            tokenizer=tokenizer,
            model_raw=model_raw,
            real_token_mask=real_token_mask,
            eos_token_ids=eos_token_ids,
            max_new_tokens=max_new_tokens,
            out_path=sample_outputs_path,
            done_pairs=sample_done_pairs,
            model_name=args.model,
            prompt_mode=meta["prompt_mode"],
            host_variant=host_variant,
            start_layer=start_layer,
            end_layer_exclusive=end_layer_exclusive,
            batch_size=batch_size,
            hook_factory=hook_factory,
            output_kind="free_running",
        )
        torch.cuda.empty_cache()

        fact_prompts = [record for record in selected_shard if record.get("category") == "CONTENT-FACT"]
        if not fact_prompts:
            continue
        forced_choice_prompt_by_id = pt_forced_choice_prompt_by_id if is_pt_host else it_forced_choice_prompt_by_id
        total_forced_choice_written += _run_condition(
            condition=condition,
            prompts=fact_prompts,
            raw_prompt_by_id={record["id"]: record.get("formats", {}).get(FORCED_CHOICE_PROMPT_FORMAT, "") for record in fact_prompts},
            model_prompt_by_id=forced_choice_prompt_by_id,
            tokenizer=tokenizer,
            model_raw=model_raw,
            real_token_mask=real_token_mask,
            eos_token_ids=eos_token_ids,
            max_new_tokens=FORCED_CHOICE_MAX_NEW_TOKENS,
            out_path=forced_choice_outputs_path,
            done_pairs=forced_choice_done_pairs,
            model_name=args.model,
            prompt_mode=f"{meta['prompt_mode']}+format_{FORCED_CHOICE_PROMPT_FORMAT}",
            host_variant=host_variant,
            start_layer=start_layer,
            end_layer_exclusive=end_layer_exclusive,
            batch_size=batch_size,
            hook_factory=hook_factory,
            output_kind="forced_choice",
        )
        torch.cuda.empty_cache()

    summary = {
        "model": args.model,
        "n_prompt_records": len(selected_shard),
        "n_prompt_records_full": len(selected_records),
        "pipelines": args.pipelines,
        "counts_by_condition": _count_rows_by_condition(sample_outputs_path),
        "forced_choice_counts_by_condition": _count_rows_by_condition(forced_choice_outputs_path),
        "sample_outputs_path": str(sample_outputs_path),
        "forced_choice_outputs_path": str(forced_choice_outputs_path),
        "new_free_running_rows_written_this_run": total_written,
        "new_forced_choice_rows_written_this_run": total_forced_choice_written,
        "dataset_manifest_hash": config["dataset_manifest_hash"],
        "pipeline_manifest_hash": config["pipeline_manifest_hash"],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[exp15] done: {summary['counts_by_condition']}", flush=True)


if __name__ == "__main__":
    main()
