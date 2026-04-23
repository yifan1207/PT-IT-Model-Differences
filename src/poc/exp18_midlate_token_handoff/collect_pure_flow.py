"""Pure PT/IT promote-suppress collector for Exp18.

This collector uses the cross-model HF adapters directly. It does not graft,
steer, or load transcoders. The output is intentionally compact: per generated
token it stores window-level support/suppression summaries instead of full logits.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.adapters import get_adapter
from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import (
    get_prompt_for_variant,
    load_dataset,
    load_model_and_tokenizer,
    read_done_ids,
)
from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import (
    classify_generated_tokens_by_word,
)
from src.poc.exp18_midlate_token_handoff.metrics import (
    COLLAPSED_CATEGORIES,
    RAW_CATEGORIES,
    WindowSpec,
    add_category_value,
    collapse_category,
    disjoint_windows,
    finalize_category_values,
    first_layer_in_topk,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_real_token_mask(tokenizer: Any, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Mask Gemma-style unused tokens while leaving normal vocabularies intact."""

    unused_re = re.compile(r"^<unused\d+>$")
    try:
        token_strs = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
        mask = [not unused_re.match(tok or "") for tok in token_strs]
    except Exception:
        mask = [True] * vocab_size
    return torch.tensor(mask, dtype=torch.bool, device=device)


def _safe_mean(values: list[float]) -> float | None:
    kept = [v for v in values if math.isfinite(v)]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _window_transition_values(
    layer_values: list[float],
    window: WindowSpec,
) -> float | None:
    values = [layer_values[layer] for layer in window.layers if 0 <= layer < len(layer_values)]
    return _safe_mean(values)


def _topk_firsts(top20_by_layer: list[list[int]], target_id: int) -> dict[str, int | None]:
    top1_by_layer = [[ids[0]] if ids else [] for ids in top20_by_layer]
    return {
        "first_top1_layer": first_layer_in_topk(top1_by_layer, target_id, k=1),
        "first_top5_layer": first_layer_in_topk(top20_by_layer, target_id, k=5),
        "first_top20_layer": first_layer_in_topk(top20_by_layer, target_id, k=20),
    }


def _adjacent_topk_movement(top20_by_layer: list[list[int]], window: WindowSpec) -> dict[str, float | None]:
    entries: list[float] = []
    exits: list[float] = []
    top1_changes: list[float] = []
    for layer in window.layers:
        if layer <= 0 or layer >= len(top20_by_layer):
            continue
        prev_ids = [int(x) for x in top20_by_layer[layer - 1]]
        curr_ids = [int(x) for x in top20_by_layer[layer]]
        if not prev_ids or not curr_ids:
            continue
        prev = set(prev_ids)
        curr = set(curr_ids)
        entries.append(float(len(curr - prev)))
        exits.append(float(len(prev - curr)))
        top1_changes.append(1.0 if prev_ids[0] != curr_ids[0] else 0.0)
    return {
        "top1_displacement": _safe_mean(top1_changes),
        "top20_entries": _safe_mean(entries),
        "top20_exits": _safe_mean(exits),
    }


def _compute_step_from_residuals(
    *,
    step_residuals: list[torch.Tensor | None],
    final_norm,
    lm_head_weight: torch.Tensor,
    real_token_mask: torch.Tensor,
    windows: dict[str, WindowSpec],
    tokenizer: Any,
    repulsion_top_k: int,
) -> dict[str, Any]:
    logits_by_layer: list[torch.Tensor] = []
    top20_by_layer: list[list[int]] = []
    for residual in step_residuals:
        if residual is None:
            logits_by_layer.append(torch.empty(0, device=real_token_mask.device))
            top20_by_layer.append([])
            continue
        h = residual.float().to(real_token_mask.device)
        normed = final_norm(h.unsqueeze(0).unsqueeze(0)).squeeze().float()
        logits = normed @ lm_head_weight
        masked = logits.clone()
        masked[~real_token_mask] = float("-inf")
        top20 = torch.topk(masked, k=min(20, masked.shape[-1])).indices.tolist()
        logits_by_layer.append(logits)
        top20_by_layer.append([int(x) for x in top20])

    target_id = int(top20_by_layer[-1][0])
    token_str = tokenizer.decode([target_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    token_category = classify_generated_tokens_by_word([{"token_str": token_str}])[0]

    support_by_layer = [float("nan")]
    repulsion_by_layer = [float("nan")]
    margin_by_layer = [float("nan")]
    for layer in range(1, len(logits_by_layer)):
        prev_logits = logits_by_layer[layer - 1]
        curr_logits = logits_by_layer[layer]
        if prev_logits.numel() == 0 or curr_logits.numel() == 0:
            support_by_layer.append(float("nan"))
            repulsion_by_layer.append(float("nan"))
            margin_by_layer.append(float("nan"))
            continue
        support = float((curr_logits[target_id] - prev_logits[target_id]).item())
        prev_masked = prev_logits.clone()
        prev_masked[~real_token_mask] = float("-inf")
        top_ids = torch.topk(prev_masked, k=min(repulsion_top_k + 1, prev_masked.shape[-1])).indices
        wrong_ids = top_ids[top_ids != target_id][:repulsion_top_k]
        if wrong_ids.numel() == 0:
            repulsion = float("nan")
        else:
            repulsion = float((curr_logits[wrong_ids] - prev_logits[wrong_ids]).mean().item())
        support_by_layer.append(support)
        repulsion_by_layer.append(repulsion)
        margin_by_layer.append(support - repulsion if math.isfinite(repulsion) else float("nan"))

    firsts = _topk_firsts(top20_by_layer, target_id)
    window_payload: dict[str, dict[str, float | None]] = {}
    for name, window in windows.items():
        movement = _adjacent_topk_movement(top20_by_layer, window)
        window_payload[name] = {
            "support_target_delta": _window_transition_values(support_by_layer, window),
            "repulsion_top10_delta": _window_transition_values(repulsion_by_layer, window),
            "margin_delta": _window_transition_values(margin_by_layer, window),
            "teacher_rank_gain": None,
            **movement,
        }

    mid = windows["mid_policy"]
    late = windows["late_reconciliation"]
    mid_selected = mid.contains(firsts["first_top20_layer"]) or mid.contains(firsts["first_top5_layer"])
    late_margin = window_payload["late_reconciliation"]["margin_delta"]
    late_reconciled = late_margin is not None and late_margin > 0

    return {
        "token_id": target_id,
        "token_str": token_str,
        "token_category": token_category,
        "token_category_collapsed": collapse_category(token_category),
        **firsts,
        "handoff": bool(mid_selected and late_reconciled),
        "mid_selected": bool(mid_selected),
        "late_reconciled": bool(late_reconciled),
        "windows": window_payload,
    }


def collect_prompt_pure_flow(
    *,
    prompt: str,
    prompt_id: str,
    model,
    tokenizer,
    adapter,
    spec,
    variant: str,
    device: torch.device,
    max_new_tokens: int,
    repulsion_top_k: int,
) -> dict[str, Any]:
    n_layers = spec.n_layers
    windows = disjoint_windows(
        n_layers=n_layers,
        phase_boundary=spec.phase_boundary,
        corrective_onset=spec.corrective_onset,
    )
    layer_modules = adapter.layers(model)
    step_residuals: list[torch.Tensor | None] = [None] * n_layers
    pending_steps: list[dict[str, Any]] = []
    final_norm = adapter.final_norm(model)
    logit_device = next(final_norm.parameters()).device
    lm_head_weight = adapter.lm_head(model).weight.detach().float().T.contiguous().to(logit_device)
    real_token_mask = build_real_token_mask(tokenizer, lm_head_weight.shape[-1], logit_device)

    def make_hook(layer_idx: int):
        def hook(_module, _inp, output):
            h = adapter.residual_from_output(output)
            step_residuals[layer_idx] = h[0, -1, :].detach()
            if layer_idx == n_layers - 1:
                pending_steps.append(
                    _compute_step_from_residuals(
                        step_residuals=step_residuals,
                        final_norm=final_norm,
                        lm_head_weight=lm_head_weight,
                        real_token_mask=real_token_mask,
                        windows=windows,
                        tokenizer=tokenizer,
                        repulsion_top_k=repulsion_top_k,
                    )
                )
        return hook

    handles = [layer_modules[idx].register_forward_hook(make_hook(idx)) for idx in range(n_layers)]
    stop_ids = adapter.stop_token_ids(tokenizer)
    generated_ids: list[int] = []
    past_key_values = None
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    total_seq_len = int(input_ids.shape[1])
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            for step_idx in range(max_new_tokens):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                if not pending_steps:
                    raise RuntimeError("No Exp18 step metrics were captured by hooks.")
                step_metrics = pending_steps[-1]
                step_metrics["step"] = step_idx
                next_token_id = int(step_metrics["token_id"])
                generated_ids.append(next_token_id)
                if next_token_id in stop_ids:
                    break
                input_ids = torch.tensor([[next_token_id]], device=device)
                total_seq_len += 1
                attention_mask = torch.ones((1, total_seq_len), dtype=torch.long, device=device)
    finally:
        for handle in handles:
            handle.remove()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "prompt_id": prompt_id,
        "model": spec.name,
        "variant": variant,
        "prompt_mode": "it_chat_template" if variant == "it" else "raw_format_b",
        "n_layers": n_layers,
        "n_steps": len(generated_ids),
        "generated_text": generated_text,
        "window_definitions": {name: window.to_json() for name, window in windows.items()},
        "steps": pending_steps[: len(generated_ids)],
    }


class PureSummaryAccumulator:
    def __init__(self) -> None:
        self.metrics: dict[str, dict[str, dict[str, dict[str, float | int | None]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0, "positive": 0}))
        )
        self.first_layers = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0, "positive": 0}))
        self.handoff = defaultdict(lambda: {"sum": 0.0, "count": 0, "positive": 0})
        self.n_steps = 0
        self.n_prompts = 0

    def add_record(self, record: dict[str, Any]) -> None:
        self.n_prompts += 1
        for step in record.get("steps", []):
            self.n_steps += 1
            raw_cat = step.get("token_category", "OTHER")
            for key in ["first_top1_layer", "first_top5_layer", "first_top20_layer"]:
                add_category_value(self.first_layers[key], raw_cat, step.get(key))
            add_category_value(self.handoff, raw_cat, 1.0 if step.get("handoff") else 0.0)
            for window, payload in step.get("windows", {}).items():
                for metric_name in [
                    "support_target_delta",
                    "repulsion_top10_delta",
                    "margin_delta",
                    "top1_displacement",
                    "top20_entries",
                    "top20_exits",
                ]:
                    add_category_value(self.metrics[window][metric_name], raw_cat, payload.get(metric_name))

    def finalize(self) -> dict[str, Any]:
        windows: dict[str, Any] = {}
        for window, by_metric in self.metrics.items():
            by_category = {cat: {} for cat in RAW_CATEGORIES + COLLAPSED_CATEGORIES}
            for metric_name, stats in by_metric.items():
                key = {
                    "support_target_delta": "support_target_delta",
                    "repulsion_top10_delta": "repulsion_top10_delta",
                    "margin_delta": "margin_delta",
                    "top1_displacement": "fraction_top1_displaced",
                    "top20_entries": "mean_top20_entries",
                    "top20_exits": "mean_top20_exits",
                }[metric_name]
                finalized = finalize_category_values(stats, mean_key=key)
                for cat, payload in finalized.items():
                    by_category.setdefault(cat, {})
                    by_category[cat].update(payload)
            first_finalized = {
                metric: finalize_category_values(stats, mean_key=f"mean_{metric}")
                for metric, stats in self.first_layers.items()
            }
            handoff_finalized = finalize_category_values(self.handoff, mean_key="handoff_rate")
            for cat in by_category:
                for metric, finalized in first_finalized.items():
                    by_category[cat].update(finalized.get(cat, {}))
                by_category[cat].update(handoff_finalized.get(cat, {}))
                by_category[cat].setdefault("teacher_rank_gain", None)
            windows[window] = {"by_token_category": by_category}
        return {
            "n_prompts": self.n_prompts,
            "n_steps": self.n_steps,
            "windows": windows,
        }


def merge_and_summarize(out_dir: Path, n_workers: int) -> dict[str, Any]:
    merged_path = out_dir / "pure_flow_results.jsonl"
    acc = PureSummaryAccumulator()
    with merged_path.open("w") as fout:
        for worker_idx in range(n_workers):
            path = out_dir / f"pure_flow_w{worker_idx}.jsonl"
            if not path.exists():
                log.warning("Missing worker file: %s", path)
                continue
            with path.open() as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    fout.write(json.dumps(record) + "\n")
                    acc.add_record(record)
    summary = acc.finalize()
    if merged_path.exists():
        summary["merged_jsonl"] = str(merged_path)
    (out_dir / "pure_flow_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_worker(
    *,
    model_name: str,
    variant: str,
    dataset_path: str,
    out_dir: Path,
    device_str: str,
    worker_index: int,
    n_workers: int,
    max_new_tokens: int,
    n_eval_examples: int | None,
    apply_chat_template: bool,
    repulsion_top_k: int,
) -> None:
    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    device = torch.device(device_str)
    model_id = model_id_for_variant(spec, variant)
    model, tokenizer = load_model_and_tokenizer(model_id, device_str, multi_gpu=spec.multi_gpu)
    records = load_dataset(
        dataset_path,
        worker_index=worker_index,
        n_workers=n_workers,
        n_examples=n_eval_examples,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pure_flow_w{worker_index}.jsonl"
    done_ids = read_done_ids(out_path)
    if done_ids:
        log.info("[w%d] resuming with %d done prompts", worker_index, len(done_ids))
    with out_path.open("a") as fout:
        for idx, record in enumerate(records):
            prompt_id = record.get("id", f"rec_{worker_index}_{idx}")
            if prompt_id in done_ids:
                continue
            prompt = get_prompt_for_variant(
                record,
                variant=variant,
                tokenizer=tokenizer,
                apply_chat_template=apply_chat_template,
            )
            try:
                result = collect_prompt_pure_flow(
                    prompt=prompt,
                    prompt_id=prompt_id,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    spec=spec,
                    variant=variant,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    repulsion_top_k=repulsion_top_k,
                )
                fout.write(json.dumps(result) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[w%d] prompt %s failed: %s", worker_index, prompt_id, exc)
            if (idx + 1) % 10 == 0:
                log.info("[w%d] %d/%d prompts done", worker_index, idx + 1, len(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp18 pure PT/IT token-flow collector.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--variant", required=True, choices=["pt", "it"])
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repulsion-top-k", type=int, default=10)
    parser.add_argument("--apply-chat-template", action="store_true", default=False)
    parser.add_argument("--no-chat-template", action="store_true", default=False)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    out_dir = args.out_dir or Path("results/exp18_midlate_token_handoff/pure_flow") / args.model / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        summary = merge_and_summarize(out_dir, args.n_workers)
        print(f"[exp18] wrote {out_dir / 'pure_flow_summary.json'} with {summary['n_steps']} steps")
        return
    use_chat_template = args.variant == "it" and not args.no_chat_template
    if args.apply_chat_template and not args.no_chat_template:
        use_chat_template = True
    run_worker(
        model_name=args.model,
        variant=args.variant,
        dataset_path=args.dataset,
        out_dir=out_dir,
        device_str=args.device,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        max_new_tokens=args.max_new_tokens,
        n_eval_examples=args.n_eval_examples,
        apply_chat_template=use_chat_template,
        repulsion_top_k=args.repulsion_top_k,
    )


if __name__ == "__main__":
    main()
