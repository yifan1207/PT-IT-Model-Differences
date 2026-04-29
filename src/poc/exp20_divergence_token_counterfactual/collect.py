"""Exp20 collector: free-run topology plus divergence-token counterfactual readouts."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer, read_done_ids
from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import (
    classify_generated_tokens_by_word,
)
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp11_matched_prefix_mlp_graft.mlp_graft import (
    ArchitectureProbe,
    PipelineCapture,
    ReadoutSpec,
    raw_logits_from_residuals,
)
from src.poc.exp18_midlate_token_handoff.metrics import first_layer_in_topk
from src.poc.exp20_divergence_token_counterfactual.metrics import (
    CONDITION_ORDER,
    classify_assistant_marker,
    find_divergence_events,
    pairwise_agreement,
    summarize_token_clusters,
    window_logit_summary,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


DEPTH_ABLATION_WINDOWS = {
    "gemma3_4b": {"early": (0, 14), "mid": (10, 24), "late": (20, 34)},
    "llama31_8b": {"early": (0, 13), "mid": (9, 22), "late": (19, 32)},
    "qwen3_4b": {"early": (0, 14), "mid": (11, 25), "late": (22, 36)},
    "qwen25_32b": {"early": (0, 26), "mid": (19, 45), "late": (38, 64)},
    "mistral_7b": {"early": (0, 13), "mid": (9, 22), "late": (19, 32)},
    "olmo2_7b": {"early": (0, 13), "mid": (9, 22), "late": (19, 32)},
    "olmo2_7b_pt_sft": {"early": (0, 13), "mid": (9, 22), "late": (19, 32)},
    "olmo2_7b_sft_dpo": {"early": (0, 13), "mid": (9, 22), "late": (19, 32)},
    "olmo2_7b_dpo_rlvr": {"early": (0, 13), "mid": (9, 22), "late": (19, 32)},
    "olmo2_32b": {"early": (0, 26), "mid": (19, 45), "late": (38, 64)},
    "deepseek_v2_lite": {"early": (0, 11), "mid": (8, 19), "late": (16, 27)},
}


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    host_variant: str
    donor_variant: str | None
    graft_kind: str | None


CONDITIONS = {
    "A_pt_raw": ConditionSpec("A_pt_raw", "pt", None, None),
    "B_early_raw": ConditionSpec("B_early_raw", "pt", "it", "early"),
    "B_mid_raw": ConditionSpec("B_mid_raw", "pt", "it", "mid"),
    "B_late_raw": ConditionSpec("B_late_raw", "pt", "it", "late"),
    "B_midlate_raw": ConditionSpec("B_midlate_raw", "pt", "it", "midlate"),
    "C_it_chat": ConditionSpec("C_it_chat", "it", None, None),
    "D_early_ptswap": ConditionSpec("D_early_ptswap", "it", "pt", "early"),
    "D_mid_ptswap": ConditionSpec("D_mid_ptswap", "it", "pt", "mid"),
    "D_late_ptswap": ConditionSpec("D_late_ptswap", "it", "pt", "late"),
    "D_midlate_ptswap": ConditionSpec("D_midlate_ptswap", "it", "pt", "midlate"),
}


def _window_defs(model_name: str, n_layers: int) -> dict[str, tuple[int, int]]:
    spec = get_spec(model_name)
    return {
        "early": (0, spec.phase_boundary),
        "mid_policy": (spec.phase_boundary, spec.corrective_onset),
        "late_reconciliation": (spec.corrective_onset, n_layers),
        "exp11_early": DEPTH_ABLATION_WINDOWS[model_name]["early"],
        "exp11_mid": DEPTH_ABLATION_WINDOWS[model_name]["mid"],
        "exp11_late": DEPTH_ABLATION_WINDOWS[model_name]["late"],
    }


def read_done_ids_any_worker(out_dir: Path) -> set[str]:
    """Return prompt ids completed by any worker file in this output directory.

    This keeps resume safe when an interrupted one-worker run is continued with
    multiple workers, or vice versa.
    """
    done: set[str] = set()
    for path in sorted(out_dir.glob("exp20_records_w*.jsonl")):
        done.update(read_done_ids(path))
    return done


def _prompt_for_condition(
    *,
    record: dict[str, Any],
    condition: ConditionSpec,
    tokenizer: Any,
    prompt_mode: str,
) -> str:
    apply_chat_template = prompt_mode == "native" and condition.host_variant == "it"
    return get_prompt_for_variant(
        record,
        variant=condition.host_variant,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
    )


def _graft_window(model_name: str, condition: ConditionSpec) -> tuple[int, int] | None:
    if condition.graft_kind is None:
        return None
    if condition.graft_kind == "earlymid":
        early_start, early_end = DEPTH_ABLATION_WINDOWS[model_name]["early"]
        mid_start, mid_end = DEPTH_ABLATION_WINDOWS[model_name]["mid"]
        return min(early_start, mid_start), max(early_end, mid_end)
    if condition.graft_kind == "midlate":
        mid_start, mid_end = DEPTH_ABLATION_WINDOWS[model_name]["mid"]
        late_start, late_end = DEPTH_ABLATION_WINDOWS[model_name]["late"]
        return min(mid_start, late_start), max(mid_end, late_end)
    return DEPTH_ABLATION_WINDOWS[model_name][condition.graft_kind]


def _make_capture(
    *,
    model_name: str,
    condition: ConditionSpec,
    models: dict[str, Any],
    steering_adapter: Any,
) -> PipelineCapture:
    graft_window = _graft_window(model_name, condition)
    donor = models[condition.donor_variant] if condition.donor_variant is not None else None
    start = graft_window[0] if graft_window else None
    end = graft_window[1] if graft_window else None
    return PipelineCapture(
        model_raw=models[condition.host_variant],
        adapter=steering_adapter,
        arch_probe=ArchitectureProbe(),
        graft_start_layer=start,
        graft_end_layer_exclusive=end,
        graft_it_model_raw=donor,
    )


def _decode_token(tokenizer: Any, token_id: int) -> dict[str, Any]:
    token_str = tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    raw_cat = classify_generated_tokens_by_word([{"token_str": token_str}])[0]
    collapsed = "FORMAT" if raw_cat in {"STRUCTURAL", "PUNCTUATION", "DISCOURSE"} else (
        "CONTENT" if raw_cat == "CONTENT" else "FUNCTION_OTHER"
    )
    return {
        "token_id": int(token_id),
        "token_str": token_str,
        "token_category": raw_cat,
        "token_category_collapsed": collapsed,
        "assistant_marker": classify_assistant_marker(token_str),
    }


def _call_model(model_name: str, model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, past_key_values=None, use_cache: bool = True):
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(**kwargs)


def _masked_next_token(logits: torch.Tensor, real_token_mask: torch.Tensor) -> int:
    masked = logits.float().clone()
    masked[~real_token_mask] = float("-inf")
    return int(masked.argmax(dim=-1).item())


def run_condition_free(
    *,
    model_name: str,
    condition: ConditionSpec,
    record: dict[str, Any],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    prompt_mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    tokenizer = tokenizers[condition.host_variant]
    model = models[condition.host_variant]
    prompt = _prompt_for_condition(record=record, condition=condition, tokenizer=tokenizer, prompt_mode=prompt_mode)
    capture = _make_capture(model_name=model_name, condition=condition, models=models, steering_adapter=steering_adapter)
    stop_ids = set(steering_adapter.eos_token_ids(tokenizer))
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    total_len = int(input_ids.shape[1])
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    past_key_values = None
    generated_ids: list[int] = []
    generated_tokens: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for _step in range(max_new_tokens):
                capture.reset_step()
                outputs = _call_model(
                    model_name,
                    model,
                    input_ids,
                    attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_id = _masked_next_token(outputs.logits[0, -1, :], real_token_masks[condition.host_variant])
                generated_ids.append(next_id)
                generated_tokens.append(_decode_token(tokenizer, next_id))
                if next_id in stop_ids:
                    break
                input_ids = torch.tensor([[next_id]], device=device)
                total_len += 1
                attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)
    finally:
        capture.close()
    return {
        "condition": condition.name,
        "host_variant": condition.host_variant,
        "donor_variant": condition.donor_variant,
        "graft_kind": condition.graft_kind,
        "graft_window": _graft_window(model_name, condition),
        "prompt_mode": prompt_mode,
        "n_steps": len(generated_ids),
        "generated_token_ids": generated_ids,
        "generated_tokens": generated_tokens,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
    }


def _rank_series(masked_logits: torch.Tensor, token_id: int) -> list[int | None]:
    if token_id < 0 or token_id >= masked_logits.shape[-1]:
        return [None for _ in range(masked_logits.shape[0])]
    target = masked_logits[:, token_id]
    return [int(x) + 1 for x in (masked_logits > target[:, None]).sum(dim=-1).tolist()]


def _first_rank_leq(ranks: list[int | None], k: int) -> int | None:
    topk = [[0] if rank is not None and rank <= k else [] for rank in ranks]
    return first_layer_in_topk(topk, 0, k=1)


def run_condition_readout(
    *,
    model_name: str,
    condition: ConditionSpec,
    record: dict[str, Any],
    prefix_token_ids: list[int],
    y_pt: int,
    y_it: int,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    prompt_mode: str,
) -> dict[str, Any]:
    tokenizer = tokenizers[condition.host_variant]
    model = models[condition.host_variant]
    prompt = _prompt_for_condition(record=record, condition=condition, tokenizer=tokenizer, prompt_mode=prompt_mode)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_ids = prompt_ids + [int(x) for x in prefix_token_ids]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    capture = _make_capture(model_name=model_name, condition=condition, models=models, steering_adapter=steering_adapter)
    try:
        with torch.no_grad():
            capture.reset_step()
            _call_model(model_name, model, input_ids, attention_mask, past_key_values=None, use_cache=False)
            snapshot = capture.snapshot()
            readout = ReadoutSpec(
                name=f"{condition.name}_raw",
                final_norm=steering_adapter.get_final_norm(model),
                lm_head=steering_adapter.get_lm_head(model),
                probes=None,
            )
            logits = raw_logits_from_residuals(readout, snapshot.residual_output)[0].detach().cpu()
    finally:
        capture.close()

    mask = real_token_masks[condition.host_variant].detach().cpu()
    masked_logits = logits.clone()
    masked_logits[:, ~mask] = float("-inf")
    y_it_logits = [float(x) for x in masked_logits[:, y_it].tolist()] if y_it < masked_logits.shape[-1] else []
    y_pt_logits = [float(x) for x in masked_logits[:, y_pt].tolist()] if y_pt < masked_logits.shape[-1] else []
    margin = [it - pt for it, pt in zip(y_it_logits, y_pt_logits, strict=False)]
    y_it_ranks = _rank_series(masked_logits, y_it)
    y_pt_ranks = _rank_series(masked_logits, y_pt)
    final_argmax = int(masked_logits[-1].argmax().item())
    winner = "it" if final_argmax == y_it else ("pt" if final_argmax == y_pt else "other")

    windows = _window_defs(model_name, get_spec(model_name).n_layers)
    graft_window = _graft_window(model_name, condition)
    if graft_window is not None:
        windows["condition_graft_window"] = graft_window
    window_payload = {
        name: {
            "y_it_logit": window_logit_summary(y_it_logits, window),
            "y_pt_logit": window_logit_summary(y_pt_logits, window),
            "it_minus_pt_margin": window_logit_summary(margin, window),
        }
        for name, window in windows.items()
    }
    return {
        "condition": condition.name,
        "host_variant": condition.host_variant,
        "donor_variant": condition.donor_variant,
        "graft_kind": condition.graft_kind,
        "graft_window": graft_window,
        "final_argmax_token_id": final_argmax,
        "winner": winner,
        "layerwise": {
            "y_it_logit": y_it_logits,
            "y_pt_logit": y_pt_logits,
            "it_minus_pt_margin": margin,
            "y_it_rank": y_it_ranks,
            "y_pt_rank": y_pt_ranks,
            "y_it_first_top1_layer": _first_rank_leq(y_it_ranks, 1),
            "y_it_first_top5_layer": _first_rank_leq(y_it_ranks, 5),
            "y_it_first_top20_layer": _first_rank_leq(y_it_ranks, 20),
            "y_pt_first_top1_layer": _first_rank_leq(y_pt_ranks, 1),
            "y_pt_first_top5_layer": _first_rank_leq(y_pt_ranks, 5),
            "y_pt_first_top20_layer": _first_rank_leq(y_pt_ranks, 20),
        },
        "windows": window_payload,
    }


def collect_prompt(
    *,
    model_name: str,
    record: dict[str, Any],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    prompt_mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt_id = str(record.get("id", record.get("record_id", "unknown")))
    free_runs = {
        name: run_condition_free(
            model_name=model_name,
            condition=CONDITIONS[name],
            record=record,
            models=models,
            tokenizers=tokenizers,
            real_token_masks=real_token_masks,
            steering_adapter=steering_adapter,
            device=device,
            prompt_mode=prompt_mode,
            max_new_tokens=max_new_tokens,
        )
        for name in CONDITION_ORDER
    }
    events = find_divergence_events(
        free_runs["A_pt_raw"]["generated_tokens"],
        free_runs["C_it_chat"]["generated_tokens"],
    )
    first_step = events["first_diff"]["step"] if events.get("first_diff") else None
    for event in events.values():
        if event is not None:
            event["shared_prefix_clean"] = event["step"] == first_step

    tokens_by_condition = {name: payload["generated_token_ids"] for name, payload in free_runs.items()}
    pairwise = {
        f"{a}__{b}": pairwise_agreement(tokens_by_condition[a], tokens_by_condition[b], max_len=max_new_tokens)
        for idx, a in enumerate(CONDITION_ORDER)
        for b in CONDITION_ORDER[idx + 1 :]
    }
    cluster_summary = summarize_token_clusters(tokens_by_condition, max_len=max_new_tokens)

    readouts: dict[str, Any] = {}
    shared_prefix_ids = free_runs["A_pt_raw"]["generated_token_ids"]
    for kind, event in events.items():
        if event is None or event.get("pt_token") is None or event.get("it_token") is None:
            readouts[kind] = None
            continue
        step = int(event["step"])
        prefix_ids = shared_prefix_ids[:step] if event.get("shared_prefix_clean") else shared_prefix_ids[:step]
        y_pt = int(event["pt_token"]["token_id"])
        y_it = int(event["it_token"]["token_id"])
        condition_token_at_step: dict[str, Any] = {}
        for name, payload in free_runs.items():
            token_ids = payload["generated_token_ids"]
            token_id = token_ids[step] if step < len(token_ids) else None
            if token_id is None:
                cls = "missing"
            elif int(token_id) == y_it:
                cls = "it"
            elif int(token_id) == y_pt:
                cls = "pt"
            else:
                cls = "other"
            condition_token_at_step[name] = {
                "token_id": token_id,
                "class": cls,
            }
        readouts[kind] = {
            "event": event,
            "condition_token_at_step": condition_token_at_step,
            "conditions": {
                name: run_condition_readout(
                    model_name=model_name,
                    condition=CONDITIONS[name],
                    record=record,
                    prefix_token_ids=prefix_ids,
                    y_pt=y_pt,
                    y_it=y_it,
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    steering_adapter=steering_adapter,
                    device=device,
                    prompt_mode=prompt_mode,
                )
                for name in CONDITION_ORDER
            },
        }

    return {
        "prompt_id": prompt_id,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "max_new_tokens": max_new_tokens,
        "free_runs": free_runs,
        "divergence_events": events,
        "pairwise_agreement": pairwise,
        "cluster_summary": cluster_summary,
        "readouts": readouts,
    }


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = get_spec(args.model)
    steering_adapter = get_steering_adapter(args.model)
    pt_model, pt_tokenizer = load_model_and_tokenizer(model_id_for_variant(spec, "pt"), args.device, multi_gpu=spec.multi_gpu)
    it_model, it_tokenizer = load_model_and_tokenizer(model_id_for_variant(spec, "it"), args.device, multi_gpu=spec.multi_gpu)
    models = {"pt": pt_model, "it": it_model}
    tokenizers = {"pt": pt_tokenizer, "it": it_tokenizer}
    real_token_masks = {
        "pt": steering_adapter.real_token_mask(pt_tokenizer, device, pt_model),
        "it": steering_adapter.real_token_mask(it_tokenizer, device, it_model),
    }
    records = load_dataset(
        args.dataset,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        n_examples=args.n_eval_examples,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"exp20_records_w{args.worker_index}.jsonl"
    done = read_done_ids_any_worker(args.out_dir)
    log.info(
        "[exp20] %s worker %d/%d resume: %d prompt ids already complete in %s",
        args.model,
        args.worker_index,
        args.n_workers,
        len(done),
        args.out_dir,
    )
    with out_path.open("a") as fout:
        for idx, record in enumerate(records):
            prompt_id = str(record.get("id", record.get("record_id", f"rec_{idx}")))
            if prompt_id in done:
                continue
            try:
                result = collect_prompt(
                    model_name=args.model,
                    record=record,
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    steering_adapter=steering_adapter,
                    device=device,
                    prompt_mode=args.prompt_mode,
                    max_new_tokens=args.max_new_tokens,
                )
                fout.write(json.dumps(result) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[exp20] prompt %s failed: %s", prompt_id, exc)
            if (idx + 1) % 5 == 0:
                log.info("[exp20] %s %d/%d prompts", args.model, idx + 1, len(records))


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "exp20_records.jsonl"
    seen: set[str] = set()
    with merged.open("w") as fout:
        for idx in range(n_workers):
            path = out_dir / f"exp20_records_w{idx}.jsonl"
            if not path.exists():
                log.warning("Missing worker file: %s", path)
                continue
            with path.open() as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        prompt_id = str(payload.get("prompt_id", ""))
                    except json.JSONDecodeError:
                        log.warning("Skipping malformed jsonl row in %s", path)
                        continue
                    if prompt_id and prompt_id in seen:
                        continue
                    if prompt_id:
                        seen.add(prompt_id)
                    fout.write(line)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp20 divergence-token counterfactual readouts.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-mode", choices=["native", "raw_shared"], default="native")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merged = merge_workers(args.out_dir, args.n_workers)
        print(f"[exp20] merged workers -> {merged}")
        return
    run_worker(args)


if __name__ == "__main__":
    main()
