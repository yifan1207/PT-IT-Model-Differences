from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.utils import load_dataset, load_model_and_tokenizer
from src.poc.exp3.analysis.word_categories import classify_generated_tokens_by_word
from src.poc.exp6.model_adapter import get_steering_adapter
from src.poc.exp11.mlp_graft import (
    ArchitectureProbe,
    PipelineCapture,
    PipelineRun,
    PipelineStepRecord,
    compute_layer_metrics,
    first_stable_below_threshold,
    first_stable_true,
    logits_from_residuals,
    select_row_step_tensors,
)
from src.poc.exp11.structural_tokens import build_structural_token_masks


VALID_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
KL_THRESHOLDS = [0.05, 0.1, 0.2, 0.5, 1.0]
DEFAULT_BATCH_SIZE = {
    "gemma3_4b": 64,
    "qwen3_4b": 64,
    "llama31_8b": 32,
    "mistral_7b": 32,
    "olmo2_7b": 32,
}


@dataclass
class RequestState:
    prompt_id: str
    category: str
    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    generated_tokens: list[dict[str, Any]] = field(default_factory=list)
    step_records: list[PipelineStepRecord] = field(default_factory=list)
    baseline_cache: list[list[torch.Tensor]] = field(default_factory=list)


@dataclass
class FixedBatch:
    request_ids: list[str]
    next_input_ids: torch.Tensor
    valid_lengths: torch.Tensor
    cache_len: int
    past_key_values: DynamicCache
    finished_mask: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp11 PT vs PT+IT MLP graft.")
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onset-layer", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-prompts", type=int, default=None)
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


def _apply_prompt_shard(prompts: list[dict], shard_index: int, num_shards: int) -> list[dict]:
    if num_shards <= 1:
        return prompts
    return prompts[shard_index::num_shards]


def _safe_decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _apply_real_token_mask(logits: torch.Tensor, real_token_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    if masked.ndim == 1:
        masked[~real_token_mask] = float("-inf")
    elif masked.ndim == 2:
        masked[:, ~real_token_mask] = float("-inf")
    elif masked.ndim == 3:
        masked[:, :, ~real_token_mask] = float("-inf")
    else:
        raise ValueError(f"Unsupported logits rank for masking: {masked.ndim}")
    return masked


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _make_request_states(prompts: list[dict], tokenizer, done_ids: set[str]) -> list[RequestState]:
    requests: list[RequestState] = []
    for record in prompts:
        if record["id"] in done_ids:
            continue
        prompt = record["formats"]["B"]
        token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        requests.append(
            RequestState(
                prompt_id=record["id"],
                category=record.get("category", ""),
                prompt=prompt,
                prompt_token_ids=list(token_ids),
            )
        )
    return requests


def _clone_requests(requests: list[RequestState]) -> list[RequestState]:
    return [
        RequestState(
            prompt_id=request.prompt_id,
            category=request.category,
            prompt=request.prompt,
            prompt_token_ids=list(request.prompt_token_ids),
        )
        for request in requests
    ]


def _pad_left_input_batch(
    batch_token_ids: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch_token_ids)
    max_len = max(len(ids) for ids in batch_token_ids)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    valid_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
    for row_idx, token_ids in enumerate(batch_token_ids):
        tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        input_ids[row_idx, -len(token_ids) :] = tensor
        attention_mask[row_idx, -len(token_ids) :] = 1
        valid_lengths[row_idx] = len(token_ids)
    return input_ids, attention_mask, valid_lengths


def _build_decode_attention_mask(valid_lengths: torch.Tensor, cache_len: int) -> torch.Tensor:
    total_len = cache_len + 1
    positions = torch.arange(total_len, device=valid_lengths.device).unsqueeze(0)
    starts = (total_len - (valid_lengths + 1)).unsqueeze(1)
    return (positions >= starts).long()


def _ensure_dynamic_cache(past_key_values: Any) -> DynamicCache:
    if isinstance(past_key_values, DynamicCache):
        return past_key_values
    return DynamicCache.from_legacy_cache(past_key_values)


def _record_step(
    request: RequestState,
    token_id: int,
    token_str: str,
    metrics,
    row_residuals: list[torch.Tensor],
    *,
    store_baseline: bool,
) -> None:
    request.generated_token_ids.append(token_id)
    request.generated_tokens.append({"token_id": token_id, "token_str": token_str})
    request.step_records.append(PipelineStepRecord(token_id=token_id, token_str=token_str, metrics=metrics))
    if store_baseline:
        request.baseline_cache.append([tensor.detach().cpu() for tensor in row_residuals])


def _prefill_requests(
    *,
    new_requests: list[RequestState],
    requests_by_id: dict[str, RequestState],
    model_raw,
    capture: PipelineCapture,
    final_norm,
    lm_head,
    tokenizer,
    pad_token_id: int,
    real_token_mask: torch.Tensor,
    tier1_mask: torch.Tensor,
    tier12_mask: torch.Tensor,
    eos_token_ids: set[int],
    inactive_token_id: int,
    max_new_tokens: int,
    store_baseline: bool,
    baseline_lookup: dict[str, list[list[torch.Tensor]]] | None,
) -> FixedBatch | None:
    if not new_requests:
        return None

    device = next(model_raw.parameters()).device
    input_ids, attention_mask, valid_lengths = _pad_left_input_batch(
        [request.prompt_token_ids for request in new_requests],
        pad_token_id,
        device,
    )

    capture.reset_step()
    with torch.no_grad():
        outputs = model_raw(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = _ensure_dynamic_cache(outputs.past_key_values)
    step_tensors = capture.snapshot()
    logits = _apply_real_token_mask(outputs.logits[:, -1, :].float(), real_token_mask.to(device))
    layer_logits = _apply_real_token_mask(
        logits_from_residuals(final_norm, lm_head, step_tensors.residual_output),
        real_token_mask.to(device),
    )
    next_token_ids = logits.argmax(dim=-1)

    next_inputs: list[int] = []
    finished_mask = torch.zeros((len(new_requests),), dtype=torch.bool, device=device)
    for row_idx, request in enumerate(new_requests):
        token_id = int(next_token_ids[row_idx].item())
        token_str = _safe_decode(tokenizer, token_id)
        row_step_tensors = select_row_step_tensors(step_tensors, row_idx)
        metric_kwargs: dict[str, Any] = {}
        if baseline_lookup is not None and request.prompt_id in baseline_lookup and 0 < len(baseline_lookup[request.prompt_id]):
            baseline_residuals = [tensor.to(device) for tensor in baseline_lookup[request.prompt_id][0]]
            baseline_logits = _apply_real_token_mask(
                logits_from_residuals(final_norm, lm_head, baseline_residuals)[0],
                real_token_mask.to(device),
            )
            metric_kwargs["baseline_logits"] = baseline_logits
            metric_kwargs["baseline_residuals"] = baseline_residuals
        metrics = compute_layer_metrics(
            pipeline_logits=layer_logits[row_idx],
            step_tensors=row_step_tensors,
            chosen_token_id=token_id,
            tier1_mask=tier1_mask,
            tier12_mask=tier12_mask,
            **metric_kwargs,
        )
        _record_step(
            request,
            token_id,
            token_str,
            metrics,
            row_step_tensors.residual_output,
            store_baseline=store_baseline,
        )
        finished = token_id in eos_token_ids or len(request.generated_token_ids) >= max_new_tokens
        finished_mask[row_idx] = finished
        next_inputs.append(inactive_token_id if finished else token_id)

    if bool(finished_mask.all().item()):
        return None

    return FixedBatch(
        request_ids=[request.prompt_id for request in new_requests],
        next_input_ids=torch.tensor(next_inputs, dtype=torch.long, device=device).unsqueeze(1),
        valid_lengths=valid_lengths,
        cache_len=input_ids.shape[1],
        past_key_values=past_key_values,
        finished_mask=finished_mask,
    )


def _decode_fixed_batch(
    *,
    active: FixedBatch,
    requests_by_id: dict[str, RequestState],
    model_raw,
    capture: PipelineCapture,
    final_norm,
    lm_head,
    tokenizer,
    real_token_mask: torch.Tensor,
    tier1_mask: torch.Tensor,
    tier12_mask: torch.Tensor,
    eos_token_ids: set[int],
    inactive_token_id: int,
    max_new_tokens: int,
    store_baseline: bool,
    baseline_lookup: dict[str, list[list[torch.Tensor]]] | None,
) -> FixedBatch | None:
    device = next(model_raw.parameters()).device
    attention_mask = _build_decode_attention_mask(active.valid_lengths, active.cache_len)
    cache_position = torch.arange(active.cache_len, active.cache_len + 1, device=device)
    model_inputs = model_raw.prepare_inputs_for_generation(
        active.next_input_ids,
        past_key_values=active.past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        use_cache=True,
    )

    capture.reset_step()
    with torch.no_grad():
        outputs = model_raw(**model_inputs, return_dict=True)

    next_cache = _ensure_dynamic_cache(outputs.past_key_values)
    step_tensors = capture.snapshot()
    logits = _apply_real_token_mask(outputs.logits[:, -1, :].float(), real_token_mask.to(device))
    layer_logits = _apply_real_token_mask(
        logits_from_residuals(final_norm, lm_head, step_tensors.residual_output),
        real_token_mask.to(device),
    )
    next_token_ids = logits.argmax(dim=-1)

    next_inputs: list[int] = []
    updated_valid_lengths = active.valid_lengths + 1
    next_finished_mask = active.finished_mask.clone()
    for row_idx, prompt_id in enumerate(active.request_ids):
        if bool(active.finished_mask[row_idx].item()):
            next_inputs.append(inactive_token_id)
            continue

        request = requests_by_id[prompt_id]
        step_idx = len(request.generated_token_ids)
        token_id = int(next_token_ids[row_idx].item())
        token_str = _safe_decode(tokenizer, token_id)
        row_step_tensors = select_row_step_tensors(step_tensors, row_idx)
        metric_kwargs: dict[str, Any] = {}
        if baseline_lookup is not None:
            baseline_steps = baseline_lookup.get(prompt_id, [])
            if step_idx < len(baseline_steps):
                baseline_residuals = [tensor.to(device) for tensor in baseline_steps[step_idx]]
                baseline_logits = _apply_real_token_mask(
                    logits_from_residuals(final_norm, lm_head, baseline_residuals)[0],
                    real_token_mask.to(device),
                )
                metric_kwargs["baseline_logits"] = baseline_logits
                metric_kwargs["baseline_residuals"] = baseline_residuals
        metrics = compute_layer_metrics(
            pipeline_logits=layer_logits[row_idx],
            step_tensors=row_step_tensors,
            chosen_token_id=token_id,
            tier1_mask=tier1_mask,
            tier12_mask=tier12_mask,
            **metric_kwargs,
        )
        _record_step(
            request,
            token_id,
            token_str,
            metrics,
            row_step_tensors.residual_output,
            store_baseline=store_baseline,
        )
        finished = token_id in eos_token_ids or len(request.generated_token_ids) >= max_new_tokens
        next_finished_mask[row_idx] = finished
        next_inputs.append(inactive_token_id if finished else token_id)

    next_cache_len = active.cache_len + 1
    if bool(next_finished_mask.all().item()):
        return None

    return FixedBatch(
        request_ids=list(active.request_ids),
        next_input_ids=torch.tensor(next_inputs, dtype=torch.long, device=device).unsqueeze(1),
        valid_lengths=updated_valid_lengths,
        cache_len=next_cache_len,
        past_key_values=next_cache,
        finished_mask=next_finished_mask,
    )


def _run_pipeline_fixed_batches(
    *,
    requests: list[RequestState],
    model_raw,
    capture: PipelineCapture,
    final_norm,
    lm_head,
    tokenizer,
    pad_token_id: int,
    real_token_mask: torch.Tensor,
    tier1_mask: torch.Tensor,
    tier12_mask: torch.Tensor,
    eos_token_ids: set[int],
    max_new_tokens: int,
    inactive_token_id: int,
    batch_size: int,
    store_baseline: bool,
    baseline_lookup: dict[str, list[list[torch.Tensor]]] | None,
) -> dict[str, PipelineRun]:
    requests_by_id = {request.prompt_id: request for request in requests}
    for start in range(0, len(requests), batch_size):
        batch_requests = requests[start : start + batch_size]
        active = _prefill_requests(
            new_requests=batch_requests,
            requests_by_id=requests_by_id,
            model_raw=model_raw,
            capture=capture,
            final_norm=final_norm,
            lm_head=lm_head,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            real_token_mask=real_token_mask,
            tier1_mask=tier1_mask,
            tier12_mask=tier12_mask,
            eos_token_ids=eos_token_ids,
            inactive_token_id=inactive_token_id,
            max_new_tokens=max_new_tokens,
            store_baseline=store_baseline,
            baseline_lookup=baseline_lookup,
        )

        while active is not None:
            active = _decode_fixed_batch(
                active=active,
                requests_by_id=requests_by_id,
                model_raw=model_raw,
                capture=capture,
                final_norm=final_norm,
                lm_head=lm_head,
                tokenizer=tokenizer,
                real_token_mask=real_token_mask,
                tier1_mask=tier1_mask,
                tier12_mask=tier12_mask,
                eos_token_ids=eos_token_ids,
                inactive_token_id=inactive_token_id,
                max_new_tokens=max_new_tokens,
                store_baseline=store_baseline,
                baseline_lookup=baseline_lookup,
            )

    return {
        request.prompt_id: PipelineRun(
            generated_token_ids=request.generated_token_ids,
            generated_tokens=request.generated_tokens,
            generated_text=tokenizer.decode(request.generated_token_ids, skip_special_tokens=True),
            step_records=request.step_records,
            baseline_cache=request.baseline_cache,
        )
        for request in requests
    }


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _run_pipeline_with_batch_fallback(
    *,
    requests: list[RequestState],
    model_raw,
    capture_factory,
    final_norm,
    lm_head,
    tokenizer,
    pad_token_id: int,
    real_token_mask: torch.Tensor,
    tier1_mask: torch.Tensor,
    tier12_mask: torch.Tensor,
    eos_token_ids: set[int],
    max_new_tokens: int,
    inactive_token_id: int,
    batch_size: int,
    store_baseline: bool,
    baseline_lookup: dict[str, list[list[torch.Tensor]]] | None,
) -> tuple[dict[str, PipelineRun], int]:
    current_batch_size = max(1, batch_size)
    while True:
        capture = capture_factory()
        fresh_requests = _clone_requests(requests)
        try:
            runs = _run_pipeline_fixed_batches(
                requests=fresh_requests,
                model_raw=model_raw,
                capture=capture,
                final_norm=final_norm,
                lm_head=lm_head,
                tokenizer=tokenizer,
                pad_token_id=pad_token_id,
                real_token_mask=real_token_mask,
                tier1_mask=tier1_mask,
                tier12_mask=tier12_mask,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                inactive_token_id=inactive_token_id,
                batch_size=current_batch_size,
                store_baseline=store_baseline,
                baseline_lookup=baseline_lookup,
            )
            return runs, current_batch_size
        except RuntimeError as exc:
            if not _is_oom_error(exc) or current_batch_size == 1:
                raise
            torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
        finally:
            capture.close()


def _summarize_pipeline(run: PipelineRun) -> dict:
    tier1_ratio = 0.0
    tier12_ratio = 0.0
    word_categories = classify_generated_tokens_by_word(run.generated_tokens)
    if word_categories:
        tier1_ratio = sum(cat in {"STRUCTURAL", "PUNCTUATION"} for cat in word_categories) / len(word_categories)
        tier12_ratio = sum(cat in {"STRUCTURAL", "PUNCTUATION", "DISCOURSE"} for cat in word_categories) / len(word_categories)

    summary = {
        "generated_text_length": len(run.generated_token_ids),
        "structural_token_ratio_tier1_proxy": tier1_ratio,
        "structural_token_ratio_tier12_proxy": tier12_ratio,
    }
    if not run.step_records:
        return summary

    for threshold in KL_THRESHOLDS:
        per_step_commitment_layers = [
            first_stable_below_threshold(step.metrics.kl_to_own_final, threshold)
            for step in run.step_records
        ]
        valid = [x for x in per_step_commitment_layers if x is not None]
        summary[f"mean_commitment_layer_kl_{threshold:g}"] = (sum(valid) / len(valid)) if valid else None
        summary[f"final_step_commitment_layer_kl_{threshold:g}"] = per_step_commitment_layers[-1]
    per_step_top1_layers = [
        first_stable_true(step.metrics.top1_match_own_final)
        for step in run.step_records
    ]
    valid_top1 = [x for x in per_step_top1_layers if x is not None]
    summary["mean_commitment_layer_top1"] = (sum(valid_top1) / len(valid_top1)) if valid_top1 else None
    summary["final_step_commitment_layer_top1"] = per_step_top1_layers[-1]
    return summary


def _write_jsonl(path: Path, rows: list[dict], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    _configure_reproducibility(args.seed)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    spec = get_spec(args.model)
    onset_layer = spec.corrective_onset if args.onset_layer is None else args.onset_layer
    requested_batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.model]
    out_dir = Path(args.out_dir or f"results/exp11/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    dataset = load_dataset(args.dataset)
    prompts = _sample_prompts(dataset, args.n_prompts, args.seed, args.categories)
    prompts = _apply_prompt_shard(prompts, args.shard_index, args.num_shards)
    done_ids = set()
    if args.resume and (out_dir / "prompt_summaries.jsonl").exists():
        with open(out_dir / "prompt_summaries.jsonl") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["prompt_id"])

    pt_id = model_id_for_variant(spec, "pt")
    it_id = model_id_for_variant(spec, "it")
    pt_model, tokenizer = load_model_and_tokenizer(pt_id, args.device, dtype=dtype)
    it_model, _ = load_model_and_tokenizer(it_id, args.device, dtype=dtype)
    _ensure_pad_token(tokenizer)

    steering_adapter = get_steering_adapter(args.model)
    model_raw_pt = pt_model
    model_raw_it = it_model
    device = next(model_raw_pt.parameters()).device
    real_token_mask = steering_adapter.real_token_mask(tokenizer, device, model_raw=model_raw_pt)
    logit_dim = steering_adapter.get_lm_head(model_raw_pt).weight.shape[0]
    structural_masks = build_structural_token_masks(tokenizer, logit_dim, real_token_mask, device)
    final_norm = steering_adapter.get_final_norm(model_raw_pt)
    lm_head = steering_adapter.get_lm_head(model_raw_pt)
    eos_ids = set(steering_adapter.eos_token_ids(tokenizer))
    arch_probe = ArchitectureProbe()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have a pad token after _ensure_pad_token()")
    inactive_token_id = min(eos_ids) if eos_ids else pad_token_id

    prompts_to_run = [record for record in prompts if record["id"] not in done_ids]
    if args.limit_prompts is not None:
        prompts_to_run = prompts_to_run[: args.limit_prompts]
    requests_a = _make_request_states(prompts_to_run, tokenizer, set())
    runs_a, batch_size_a = _run_pipeline_with_batch_fallback(
        requests=requests_a,
        model_raw=model_raw_pt,
        capture_factory=lambda: PipelineCapture(
            model_raw=model_raw_pt,
            adapter=steering_adapter,
            arch_probe=arch_probe,
            onset_layer=onset_layer,
            graft_it_model_raw=None,
        ),
        final_norm=final_norm,
        lm_head=lm_head,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        real_token_mask=real_token_mask,
        tier1_mask=structural_masks.tier1,
        tier12_mask=structural_masks.tier12,
        eos_token_ids=eos_ids,
        max_new_tokens=args.max_new_tokens,
        inactive_token_id=inactive_token_id,
        batch_size=requested_batch_size,
        store_baseline=True,
        baseline_lookup=None,
    )
    baseline_lookup = {prompt_id: run.baseline_cache for prompt_id, run in runs_a.items()}

    requests_b = _make_request_states(prompts_to_run, tokenizer, set())
    runs_b, batch_size_b = _run_pipeline_with_batch_fallback(
        requests=requests_b,
        model_raw=model_raw_pt,
        capture_factory=lambda: PipelineCapture(
            model_raw=model_raw_pt,
            adapter=steering_adapter,
            arch_probe=arch_probe,
            onset_layer=onset_layer,
            graft_it_model_raw=model_raw_it,
        ),
        final_norm=final_norm,
        lm_head=lm_head,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        real_token_mask=real_token_mask,
        tier1_mask=structural_masks.tier1,
        tier12_mask=structural_masks.tier12,
        eos_token_ids=eos_ids,
        max_new_tokens=args.max_new_tokens,
        inactive_token_id=inactive_token_id,
        batch_size=batch_size_a,
        store_baseline=False,
        baseline_lookup=baseline_lookup,
    )

    config = {
        "model": args.model,
        "pt_model_id": pt_id,
        "it_model_id": it_id,
        "dataset": args.dataset,
        "n_prompts": args.n_prompts,
        "seed": args.seed,
        "onset_layer": onset_layer,
        "max_new_tokens": args.max_new_tokens,
        "categories": args.categories,
        "dtype": args.dtype,
        "batch_size_requested": requested_batch_size,
        "batch_size_pipeline_a": batch_size_a,
        "batch_size_pipeline_b": batch_size_b,
        "scheduler_mode": "fixed_batches",
        "continuous_admission": False,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "limit_prompts": args.limit_prompts,
        "n_prompts_sampled_after_sharding": len(prompts),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    _write_jsonl(out_dir / "prompts.jsonl", prompts, append=False)

    for record in prompts_to_run:
        run_a = runs_a[record["id"]]
        run_b = runs_b[record["id"]]

        divergence_step = None
        for idx, (tok_a, tok_b) in enumerate(zip(run_a.generated_token_ids, run_b.generated_token_ids, strict=False)):
            if tok_a != tok_b:
                divergence_step = idx
                break
        if divergence_step is None and len(run_a.generated_token_ids) != len(run_b.generated_token_ids):
            divergence_step = min(len(run_a.generated_token_ids), len(run_b.generated_token_ids))

        summary_row = {
            "prompt_id": record["id"],
            "category": record.get("category", ""),
            "prompt": record["formats"]["B"],
            "divergence_step": divergence_step,
            "pipeline_a": _summarize_pipeline(run_a),
            "pipeline_b": _summarize_pipeline(run_b),
        }
        _write_jsonl(out_dir / "prompt_summaries.jsonl", [summary_row], append=True)
        _write_jsonl(
            out_dir / "generated_texts.jsonl",
            [
                {
                    "prompt_id": record["id"],
                    "pipeline": "A",
                    "generated_text": run_a.generated_text,
                    "generated_tokens": run_a.generated_tokens,
                },
                {
                    "prompt_id": record["id"],
                    "pipeline": "B",
                    "generated_text": run_b.generated_text,
                    "generated_tokens": run_b.generated_tokens,
                },
            ],
            append=True,
        )

        step_rows: list[dict] = []
        for pipeline_name, run in (("A", run_a), ("B", run_b)):
            for step_idx, step in enumerate(run.step_records):
                step_rows.append(
                    {
                        "prompt_id": record["id"],
                        "pipeline": pipeline_name,
                        "step": step_idx,
                        "token_id": step.token_id,
                        "token_str": step.token_str,
                        "metrics": asdict(step.metrics),
                    }
                )
        _write_jsonl(out_dir / "step_metrics.jsonl", step_rows, append=True)


if __name__ == "__main__":
    main()
