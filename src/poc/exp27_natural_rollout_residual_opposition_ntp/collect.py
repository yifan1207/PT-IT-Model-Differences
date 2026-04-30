"""Collect Exp27 natural-rollout residual-opposition records."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp20_divergence_token_counterfactual.collect import DEPTH_ABLATION_WINDOWS
from src.poc.exp27_natural_rollout_residual_opposition_ntp import DEFAULT_VARIANTS, EXPERIMENT
from src.poc.exp27_natural_rollout_residual_opposition_ntp.hooks import SpanMlpOppositionModifier
from src.poc.exp27_natural_rollout_residual_opposition_ntp.scoring import (
    build_generated_source_mask,
    ensure_padding_token,
    generate_rollouts,
    pad_rollout_ids,
    score_variant_against_full,
)
from src.poc.exp27_natural_rollout_residual_opposition_ntp.variants import expand_variants


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _late_boundary(model_name: str) -> int:
    return int(DEPTH_ABLATION_WINDOWS[model_name]["late"][0])


def _record_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    seed = row.get("seed")
    return (
        str(row.get("model_variant")),
        str(row.get("prompt_id")),
        str(row.get("variant")),
        "none" if seed is None else str(seed),
    )


def _done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    return {_record_key(row) for row in _json_rows(path)}


def _prompt_tuple(record: dict[str, Any], *, model_variant: str, tokenizer: Any) -> tuple[str, str, str]:
    prompt_id = str(record.get("id", record.get("record_id")))
    category = str(record.get("category", "unknown"))
    prompt_text = get_prompt_for_variant(
        record,
        variant=model_variant,
        tokenizer=tokenizer,
        apply_chat_template=(model_variant == "it"),
    )
    return prompt_id, category, prompt_text


def _record_for_prompt_variant(
    *,
    model_name: str,
    model_variant: str,
    rollout,
    variant: str,
    seed: int | None,
    score: dict[str, Any],
    diagnostics: dict[str, Any],
    late_layers: list[int],
    include_boundary_source: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    opp_frac = diagnostics.get("mean_opp_norm_frac")
    nll_delta = score.get("nll_delta")
    true_logit_drop = score.get("true_logit_drop")
    return {
        "experiment": EXPERIMENT,
        "model": model_name,
        "model_variant": model_variant,
        "prompt_id": rollout.prompt_id,
        "category": rollout.category,
        "variant": variant,
        "seed": seed,
        "valid": score.get("n_positions", 0) > 0,
        "reason": None if score.get("n_positions", 0) > 0 else "no_measured_generated_source_positions",
        "max_new_tokens": int(max_new_tokens),
        "include_boundary_source": bool(include_boundary_source),
        "prompt_len": int(rollout.prompt_len),
        "generated_len": int(rollout.generated_len),
        "n_positions": int(score.get("n_positions", 0)),
        "late_layers": [int(layer) for layer in late_layers],
        "generated_token_ids": [int(token_id) for token_id in rollout.generated_ids],
        "generated_text": rollout.generated_text,
        "score": score,
        "diagnostics": diagnostics,
        "nll_delta_per_opp_frac": (
            None
            if nll_delta is None or opp_frac is None or abs(float(opp_frac)) < 1e-12
            else float(nll_delta) / float(opp_frac)
        ),
        "true_logit_drop_per_opp_frac": (
            None
            if true_logit_drop is None or opp_frac is None or abs(float(opp_frac)) < 1e-12
            else float(true_logit_drop) / float(opp_frac)
        ),
    }


@torch.no_grad()
def collect_model_variant(
    *,
    model_name: str,
    model_variant: str,
    dataset_records: list[dict[str, Any]],
    out_path: Path,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_tokens: int,
    variant_jobs: list[tuple[str, int | None]],
    include_boundary_source: bool,
) -> None:
    spec = get_spec(model_name)
    steering_adapter = get_steering_adapter(model_name)
    model, tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, model_variant),
        device,
        multi_gpu=spec.multi_gpu,
    )
    pad_id = ensure_padding_token(tokenizer)
    stop_ids = steering_adapter.eos_token_ids(tokenizer)
    late_start = _late_boundary(model_name)
    late_layers = list(range(late_start, len(steering_adapter.get_layers(model))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)
    prompts = [_prompt_tuple(record, model_variant=model_variant, tokenizer=tokenizer) for record in dataset_records]
    prompts = [
        prompt
        for prompt in prompts
        if not all((model_variant, prompt[0], variant, "none" if seed is None else str(seed)) in done for variant, seed in variant_jobs)
    ]
    log.info(
        "[exp27] collect model=%s variant=%s prompts=%d jobs=%s late_layers=%s",
        model_name,
        model_variant,
        len(prompts),
        variant_jobs,
        late_layers,
    )
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for batch_start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[batch_start : batch_start + batch_size]
            rollouts = generate_rollouts(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_batch,
                device=device,
                max_new_tokens=max_new_tokens,
                max_prompt_tokens=max_prompt_tokens,
                stop_token_ids=stop_ids,
            )
            input_ids, attention_mask, prompt_lens, generated_lens = pad_rollout_ids(
                rollouts,
                pad_id=pad_id,
                device=device,
            )
            source_mask = build_generated_source_mask(
                input_ids=input_ids,
                prompt_lens=prompt_lens,
                generated_lens=generated_lens,
                include_boundary_source=include_boundary_source,
            )
            row_ids = [rollout.prompt_id for rollout in rollouts]
            full_logits = None
            full_diagnostics: dict[str, Any] = {}
            batch_key = f"{model_name}:{model_variant}:{batch_start}:{'-'.join(row_ids)}"
            batch_has_pending_ablation = any(
                (model_variant, rollout.prompt_id, variant, "none" if seed is None else str(seed)) not in done
                for rollout in rollouts
                for variant, seed in variant_jobs
                if not (variant == "full" and seed is None)
            )
            for variant, seed in variant_jobs:
                key_seed = "none" if seed is None else str(seed)
                pending = [
                    rollout
                    for rollout in rollouts
                    if (model_variant, rollout.prompt_id, variant, key_seed) not in done
                ]
                should_run_for_scoring = (
                    variant == "full"
                    and seed is None
                    and full_logits is None
                    and batch_has_pending_ablation
                )
                if not pending and not should_run_for_scoring:
                    continue
                modifier = SpanMlpOppositionModifier(
                    model=model,
                    steering_adapter=steering_adapter,
                    late_layers=late_layers,
                    variant=variant,
                    seed=seed,
                    span_mask=source_mask,
                    row_ids=row_ids,
                    batch_key=batch_key,
                )
                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                finally:
                    modifier.close()
                logits = outputs.logits.detach()
                diagnostics_by_row = modifier.summary_by_row()
                if variant == "full" and seed is None:
                    full_logits = logits
                    full_diagnostics = diagnostics_by_row
                if full_logits is None:
                    raise RuntimeError("Exp27 variant order must run full before ablations")
                score_by_row = score_variant_against_full(
                    full_logits=full_logits,
                    variant_logits=logits,
                    input_ids=input_ids,
                    source_mask=source_mask,
                    row_ids=row_ids,
                    prompt_lens=prompt_lens,
                )
                for rollout in rollouts:
                    record_key = (model_variant, rollout.prompt_id, variant, key_seed)
                    if record_key in done:
                        continue
                    diagnostics = diagnostics_by_row.get(rollout.prompt_id, {})
                    if variant == "full" and seed is None and not diagnostics:
                        diagnostics = full_diagnostics.get(rollout.prompt_id, {})
                    record = _record_for_prompt_variant(
                        model_name=model_name,
                        model_variant=model_variant,
                        rollout=rollout,
                        variant=variant,
                        seed=seed,
                        score=score_by_row[rollout.prompt_id],
                        diagnostics=diagnostics,
                        late_layers=late_layers,
                        include_boundary_source=include_boundary_source,
                        max_new_tokens=max_new_tokens,
                    )
                    fout.write(json.dumps(record, separators=(",", ":")) + "\n")
                    done.add(record_key)
                fout.flush()
            if (batch_start // batch_size + 1) % 5 == 0:
                log.info(
                    "[exp27] %s/%s batch %d/%d",
                    model_name,
                    model_variant,
                    batch_start // batch_size + 1,
                    (len(prompts) + batch_size - 1) // batch_size,
                )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dataset_records = load_dataset(
        args.dataset,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        n_examples=args.n_eval_examples,
    )
    variant_jobs = expand_variants(args.variants, args.rand_seeds)
    if not variant_jobs or variant_jobs[0] != ("full", None):
        variant_jobs = [("full", None)] + [job for job in variant_jobs if job != ("full", None)]
    for model_variant in args.model_variants:
        out_path = args.out_dir / f"records_{model_variant}_w{args.worker_index}.jsonl.gz"
        collect_model_variant(
            model_name=args.model,
            model_variant=model_variant,
            dataset_records=dataset_records,
            out_path=out_path,
            device=device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            variant_jobs=variant_jobs,
            include_boundary_source=args.include_boundary_source,
        )


def merge_workers(out_dir: Path, n_workers: int, model_variants: list[str]) -> Path:
    merged = out_dir / "records.jsonl.gz"
    seen: set[tuple[str, str, str, str]] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for model_variant in model_variants:
            for worker_idx in range(n_workers):
                path = out_dir / f"records_{model_variant}_w{worker_idx}.jsonl.gz"
                if not path.exists():
                    log.warning("[exp27] missing worker file %s", path)
                    continue
                for row in _json_rows(path):
                    key = _record_key(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp27 natural-rollout residual-opposition NTP records.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--model-variants", nargs="+", choices=["pt", "it"], default=["pt", "it"])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--rand-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--include-boundary-source", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_workers(args.out_dir, args.n_workers, args.model_variants)
        return
    run_worker(args)


if __name__ == "__main__":
    main()
