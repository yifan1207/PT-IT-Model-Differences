from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

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
)
from src.poc.exp11.structural_tokens import build_structural_token_masks


VALID_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
KL_THRESHOLDS = [0.05, 0.1, 0.2, 0.5, 1.0]


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
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


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
        (
            ((len(by_category[cat]) * n_prompts / total) - targets[cat], cat)
            for cat in cats
        ),
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


def _safe_decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _apply_real_token_mask(logits: torch.Tensor, real_token_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked[~real_token_mask] = float("-inf")
    return masked


def _run_pipeline(
    *,
    prompt: str,
    model_raw,
    tokenizer,
    capture: PipelineCapture,
    final_norm,
    lm_head,
    real_token_mask: torch.Tensor,
    tier1_mask: torch.Tensor,
    tier12_mask: torch.Tensor,
    eos_token_ids: set[int],
    max_new_tokens: int,
    baseline_step_cache: list[tuple[torch.Tensor, list[torch.Tensor]]] | None = None,
) -> PipelineRun:
    device = next(model_raw.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    attention_mask = torch.ones_like(encoded, device=device)
    generated_token_ids: list[int] = []
    generated_tokens: list[dict] = []
    step_records: list[PipelineStepRecord] = []
    baseline_cache: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
    past_key_values = None
    current_input_ids = encoded
    current_attention_mask = attention_mask

    for step_idx in range(max_new_tokens):
        capture.reset_step()
        with torch.no_grad():
            outputs = model_raw(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        step_tensors = capture.snapshot()
        logits = outputs.logits[0, -1, :].float()
        logits = _apply_real_token_mask(logits, real_token_mask.to(logits.device))
        layer_logits = logits_from_residuals(final_norm, lm_head, step_tensors.residual_output)
        baseline_cache.append((layer_logits.detach(), [x.detach() for x in step_tensors.residual_output]))
        next_token_id = int(logits.argmax().item())

        metric_kwargs = {}
        if baseline_step_cache is not None and step_idx < len(baseline_step_cache):
            baseline_logits, baseline_residuals = baseline_step_cache[step_idx]
            metric_kwargs["baseline_logits"] = baseline_logits
            metric_kwargs["baseline_residuals"] = baseline_residuals

        metrics = compute_layer_metrics(
            pipeline_logits=layer_logits,
            step_tensors=step_tensors,
            chosen_token_id=next_token_id,
            tier1_mask=tier1_mask,
            tier12_mask=tier12_mask,
            **metric_kwargs,
        )

        token_str = _safe_decode(tokenizer, next_token_id)
        generated_token_ids.append(next_token_id)
        generated_tokens.append({"token_id": next_token_id, "token_str": token_str})
        step_records.append(PipelineStepRecord(token_id=next_token_id, token_str=token_str, metrics=metrics))

        if next_token_id in eos_token_ids:
            break

        next_token = torch.tensor([[next_token_id]], device=device)
        current_input_ids = next_token
        current_attention_mask = torch.ones((1, attention_mask.shape[1] + len(generated_token_ids)), device=device, dtype=attention_mask.dtype)
        past_key_values = outputs.past_key_values

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return PipelineRun(
        generated_token_ids=generated_token_ids,
        generated_tokens=generated_tokens,
        generated_text=generated_text,
        step_records=step_records,
        baseline_cache=baseline_cache,
    )


def _summarize_pipeline(run: PipelineRun) -> dict:
    tier1_ratio = 0.0
    tier12_ratio = 0.0
    word_categories = classify_generated_tokens_by_word(run.generated_tokens)
    if word_categories:
        tier1_ratio = sum(cat in {"STRUCTURAL", "PUNCTUATION"} for cat in word_categories) / len(word_categories)
        tier12_ratio = sum(cat in {"STRUCTURAL", "PUNCTUATION", "DISCOURSE"} for cat in word_categories) / len(word_categories)

    final_step = run.step_records[-1].metrics if run.step_records else None
    summary = {
        "generated_text_length": len(run.generated_token_ids),
        "structural_token_ratio_tier1_proxy": tier1_ratio,
        "structural_token_ratio_tier12_proxy": tier12_ratio,
    }
    if final_step is None:
        return summary

    for threshold in KL_THRESHOLDS:
        per_step_last_layer = [step.metrics.kl_to_own_final[-1] for step in run.step_records]
        summary[f"commitment_step_kl_{threshold:g}"] = first_stable_below_threshold(per_step_last_layer, threshold)
    per_step_top1_match = [step.metrics.top1_match_own_final[-1] for step in run.step_records]
    summary["commitment_step_top1"] = first_stable_true(per_step_top1_match)
    return summary


def _write_jsonl(path: Path, rows: list[dict], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    onset_layer = spec.corrective_onset if args.onset_layer is None else args.onset_layer
    out_dir = Path(args.out_dir or f"results/exp11/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    dataset = load_dataset(args.dataset)
    prompts = _sample_prompts(dataset, args.n_prompts, args.seed, args.categories)
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
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    _write_jsonl(out_dir / "prompts.jsonl", prompts, append=False)

    for record in prompts:
        if record["id"] in done_ids:
            continue
        prompt = record["formats"]["B"]
        capture_a = PipelineCapture(
            model_raw=model_raw_pt,
            adapter=steering_adapter,
            arch_probe=arch_probe,
            onset_layer=onset_layer,
            graft_it_model_raw=None,
        )
        try:
            run_a = _run_pipeline(
                prompt=prompt,
                model_raw=model_raw_pt,
                tokenizer=tokenizer,
                capture=capture_a,
                final_norm=final_norm,
                lm_head=lm_head,
                real_token_mask=real_token_mask,
                tier1_mask=structural_masks.tier1,
                tier12_mask=structural_masks.tier12,
                eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens,
            )
        finally:
            capture_a.close()

        capture_b = PipelineCapture(
            model_raw=model_raw_pt,
            adapter=steering_adapter,
            arch_probe=arch_probe,
            onset_layer=onset_layer,
            graft_it_model_raw=model_raw_it,
        )
        try:
            run_b = _run_pipeline(
                prompt=prompt,
                model_raw=model_raw_pt,
                tokenizer=tokenizer,
                capture=capture_b,
                final_norm=final_norm,
                lm_head=lm_head,
                real_token_mask=real_token_mask,
                tier1_mask=structural_masks.tier1,
                tier12_mask=structural_masks.tier12,
                eos_token_ids=eos_ids,
                max_new_tokens=args.max_new_tokens,
                baseline_step_cache=run_a.baseline_cache,
            )
        finally:
            capture_b.close()

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
            "prompt": prompt,
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
