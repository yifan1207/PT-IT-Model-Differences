"""Collect Exp21 productive-opposition projections at Exp20 divergence prefixes."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import (
    classify_generated_tokens_by_word,
)
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp11_matched_prefix_mlp_graft.mlp_graft import ArchitectureProbe, PipelineCapture
from src.poc.exp19_late_mlp_specificity_controls.controls import (
    CONTROL_MODE_RANDOM_RESPROJ,
    CONTROL_MODE_TRUE,
)
from src.poc.exp20_divergence_token_counterfactual.collect import DEPTH_ABLATION_WINDOWS
from src.poc.exp20_divergence_token_counterfactual.metrics import classify_assistant_marker
from src.poc.exp21_productive_opposition.metrics import (
    finite_mean,
    finite_rate,
    mlp_delta_cosine,
    negative_parallel_component,
    negative_parallel_norm,
    productive_opposition,
    summarize_categories,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


DEFAULT_EXP20_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early"
)
DEFAULT_EXP20_FALLBACK_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final"
)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    host_variant: str
    donor_variant: str | None = None
    graft_kind: str | None = None
    control_mode: str = CONTROL_MODE_TRUE
    control_seed: int = 0


CONDITIONS: dict[str, ConditionSpec] = {
    "A_pt_raw": ConditionSpec("A_pt_raw", "pt"),
    "B_early_raw": ConditionSpec("B_early_raw", "pt", "it", "early"),
    "B_mid_raw": ConditionSpec("B_mid_raw", "pt", "it", "mid"),
    "B_late_raw": ConditionSpec("B_late_raw", "pt", "it", "late"),
    "B_earlymid_raw": ConditionSpec("B_earlymid_raw", "pt", "it", "earlymid"),
    "B_midlate_raw": ConditionSpec("B_midlate_raw", "pt", "it", "midlate"),
    "C_it_chat": ConditionSpec("C_it_chat", "it"),
    "D_early_ptswap": ConditionSpec("D_early_ptswap", "it", "pt", "early"),
    "D_mid_ptswap": ConditionSpec("D_mid_ptswap", "it", "pt", "mid"),
    "D_late_ptswap": ConditionSpec("D_late_ptswap", "it", "pt", "late"),
    "D_earlymid_ptswap": ConditionSpec("D_earlymid_ptswap", "it", "pt", "earlymid"),
    "D_midlate_ptswap": ConditionSpec("D_midlate_ptswap", "it", "pt", "midlate"),
    "B_late_identity": ConditionSpec("B_late_identity", "pt", "pt", "late"),
    "D_late_identity": ConditionSpec("D_late_identity", "it", "it", "late"),
    "B_late_rand_resproj_s0": ConditionSpec(
        "B_late_rand_resproj_s0",
        "pt",
        "it",
        "late",
        CONTROL_MODE_RANDOM_RESPROJ,
        0,
    ),
    "D_late_rand_resproj_s0": ConditionSpec(
        "D_late_rand_resproj_s0",
        "it",
        "pt",
        "late",
        CONTROL_MODE_RANDOM_RESPROJ,
        0,
    ),
}

DEFAULT_CONDITIONS = list(CONDITIONS)
DEFAULT_EVENT_KINDS = ["first_diff", "first_nonformat_diff", "first_assistant_marker_diff"]


def _jsonl_rows(path: Path):
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw.decode("utf-8", errors="ignore"))


def _gzip_jsonl_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _find_exp20_manifest(root: Path, fallback_root: Path | None, prompt_mode: str, model: str) -> Path:
    candidates = [
        root / prompt_mode / model / "exp20_validation_records.jsonl",
        root / prompt_mode / model / "exp20_records.jsonl",
    ]
    if fallback_root is not None:
        candidates.extend(
            [
                fallback_root / prompt_mode / model / "exp20_validation_records.jsonl",
                fallback_root / prompt_mode / model / "exp20_records.jsonl",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No Exp20 manifest found for mode={prompt_mode} model={model}. Tried: "
        + ", ".join(str(x) for x in candidates)
    )


def _load_manifest_records(
    *,
    exp20_root: Path,
    exp20_fallback_root: Path | None,
    prompt_mode: str,
    model: str,
    n_examples: int | None,
    worker_index: int,
    n_workers: int,
) -> list[dict[str, Any]]:
    path = _find_exp20_manifest(exp20_root, exp20_fallback_root, prompt_mode, model)
    rows = list(_jsonl_rows(path))
    if n_examples is not None:
        rows = rows[:n_examples]
    sliced = rows[worker_index::n_workers]
    log.info("[exp21] manifest %s -> %d rows for worker %d/%d", path, len(sliced), worker_index, n_workers)
    return sliced


def _dataset_lookup(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_dataset(path, n_examples=None)
    return {str(row.get("id", row.get("record_id"))): row for row in rows}


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
    windows = DEPTH_ABLATION_WINDOWS[model_name]
    if condition.graft_kind == "earlymid":
        early = windows["early"]
        mid = windows["mid"]
        return min(early[0], mid[0]), max(early[1], mid[1])
    if condition.graft_kind == "midlate":
        mid = windows["mid"]
        late = windows["late"]
        return min(mid[0], late[0]), max(mid[1], late[1])
    return windows[condition.graft_kind]


def _window_defs(model_name: str) -> dict[str, tuple[int, int]]:
    spec = get_spec(model_name)
    n_layers = spec.n_layers
    depth = DEPTH_ABLATION_WINDOWS[model_name]
    return {
        "early": (0, spec.phase_boundary),
        "mid_policy": (spec.phase_boundary, spec.corrective_onset),
        "late_reconciliation": (spec.corrective_onset, n_layers),
        "exp11_early": depth["early"],
        "exp11_mid": depth["mid"],
        "exp11_late": depth["late"],
    }


def _make_capture(
    *,
    model_name: str,
    condition: ConditionSpec,
    models: dict[str, Any],
    steering_adapter: Any,
) -> PipelineCapture:
    graft_window = _graft_window(model_name, condition)
    donor = models[condition.donor_variant] if condition.donor_variant is not None else None
    return PipelineCapture(
        model_raw=models[condition.host_variant],
        adapter=steering_adapter,
        arch_probe=ArchitectureProbe(),
        graft_start_layer=graft_window[0] if graft_window else None,
        graft_end_layer_exclusive=graft_window[1] if graft_window else None,
        graft_it_model_raw=donor,
        graft_control_mode=condition.control_mode,
        graft_random_seed=condition.control_seed,
        graft_control_name=condition.name,
    )


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    return torch.float32


def _normed_hidden(final_norm: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    dtype = _module_dtype(final_norm)
    return final_norm(hidden.to(dtype=dtype).view(1, 1, -1)).view(-1).float()


def _full_logits(
    *,
    hidden: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    real_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    dtype = _module_dtype(lm_head)
    normed = _normed_hidden(final_norm, hidden).to(dtype=dtype)
    logits = lm_head(normed).float()
    if real_token_mask is not None:
        logits = logits.clone()
        logits[~real_token_mask.to(logits.device)] = float("-inf")
    return logits


def _selected_logits(
    *,
    hidden: torch.Tensor,
    token_ids: list[int],
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> dict[int, float]:
    valid = sorted({int(t) for t in token_ids if t is not None and 0 <= int(t) < lm_head.weight.shape[0]})
    if not valid:
        return {}
    normed = _normed_hidden(final_norm, hidden)
    ids = torch.tensor(valid, dtype=torch.long, device=lm_head.weight.device)
    weight = lm_head.weight.detach().float()[ids]
    values = weight @ normed.to(weight.device)
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        values = values + bias.detach().float()[ids]
    return {token_id: float(value) for token_id, value in zip(valid, values.tolist(), strict=False)}


def _token_payload(tokenizer: Any, token_id: int | None) -> dict[str, Any] | None:
    if token_id is None:
        return None
    token_str = tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    raw_category = classify_generated_tokens_by_word([{"token_str": token_str}])[0]
    collapsed = "FORMAT" if raw_category in {"STRUCTURAL", "DISCOURSE", "PUNCTUATION"} else (
        "CONTENT" if raw_category == "CONTENT" else "FUNCTION_OTHER"
    )
    return {
        "token_id": int(token_id),
        "token_str": token_str,
        "token_category": raw_category,
        "token_category_collapsed": collapsed,
        "assistant_marker": classify_assistant_marker(token_str),
    }


def _token_class(token_id: int | None, *, y_it: int, y_pt: int, pipeline_token: int | None) -> str:
    if token_id is None:
        return "missing"
    if int(token_id) == int(y_it):
        return "it"
    if int(token_id) == int(y_pt):
        return "pt"
    if pipeline_token is not None and int(token_id) == int(pipeline_token):
        return "pipeline"
    return "other"


def _tensor_at_batch0(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor[0].float()
    return tensor.float()


def _mean_for_window(layer_rows: list[dict[str, Any]], key: str, start: int, end: int) -> float | None:
    return finite_mean(row.get(key) for row in layer_rows[start:end])


def _rate_for_window(layer_rows: list[dict[str, Any]], key: str, start: int, end: int) -> float | None:
    return finite_rate(row.get(key) for row in layer_rows[start:end])


def _category_summary_for_window(layer_rows: list[dict[str, Any]], key: str, start: int, end: int) -> dict[str, int]:
    return summarize_categories([row.get(key) for row in layer_rows[start:end]])


def _summarize_windows(
    *,
    model_name: str,
    condition: ConditionSpec,
    layer_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    windows = _window_defs(model_name)
    graft_window = _graft_window(model_name, condition)
    if graft_window is not None:
        windows["condition_graft_window"] = graft_window
    metric_keys = [
        "delta_cosine_mlp",
        "negative_parallel_norm",
        "support_it_token",
        "support_pt_token",
        "margin_writein_it_vs_pt",
        "support_pipeline_token",
        "alt_top10_delta",
        "target_vs_alt_margin",
        "opposition_support_it_token",
        "opposition_support_pt_token",
        "opposition_margin_it_vs_pt",
        "opposition_support_pipeline_token",
        "opposition_alt_top10_delta",
        "opposition_target_vs_alt_margin",
        "remainder_margin_it_vs_pt",
        "remainder_target_vs_alt_margin",
    ]
    out: dict[str, Any] = {}
    for name, (start, end) in windows.items():
        payload = {key: _mean_for_window(layer_rows, key, start, end) for key in metric_keys}
        payload["productive_opposition_rate"] = _rate_for_window(layer_rows, "productive_opposition", start, end)
        payload["top_supported_category_counts"] = _category_summary_for_window(
            layer_rows,
            "top_supported_category",
            start,
            end,
        )
        payload["top_suppressed_category_counts"] = _category_summary_for_window(
            layer_rows,
            "top_suppressed_category",
            start,
            end,
        )
        payload["top_supported_token_class_counts"] = dict(
            sorted(
                {
                    cls: sum(1 for row in layer_rows[start:end] if row.get("top_supported_token_class") == cls)
                    for cls in ["it", "pt", "pipeline", "other", "missing"]
                }.items()
            )
        )
        payload["top_suppressed_token_class_counts"] = dict(
            sorted(
                {
                    cls: sum(1 for row in layer_rows[start:end] if row.get("top_suppressed_token_class") == cls)
                    for cls in ["it", "pt", "pipeline", "other", "missing"]
                }.items()
            )
        )
        out[name] = payload
    return out


def _layer_metrics(
    *,
    pre: torch.Tensor,
    update: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    real_token_mask: torch.Tensor,
    tokenizer: Any,
    y_it: int,
    y_pt: int,
    pipeline_token: int | None,
    top_k: int,
) -> dict[str, Any]:
    pre = pre.float()
    update = update.float()
    full = pre + update
    pre_logits = _full_logits(hidden=pre, final_norm=final_norm, lm_head=lm_head, real_token_mask=real_token_mask)
    full_logits = _full_logits(hidden=full, final_norm=final_norm, lm_head=lm_head, real_token_mask=real_token_mask)

    target_ids = [y_it, y_pt]
    if pipeline_token is not None:
        target_ids.append(pipeline_token)
    pre_selected = _selected_logits(hidden=pre, token_ids=target_ids, final_norm=final_norm, lm_head=lm_head)
    full_selected = _selected_logits(hidden=full, token_ids=target_ids, final_norm=final_norm, lm_head=lm_head)

    support_it = _delta_selected(pre_selected, full_selected, y_it)
    support_pt = _delta_selected(pre_selected, full_selected, y_pt)
    support_pipeline = _delta_selected(pre_selected, full_selected, pipeline_token)
    margin_it_pt = _diff_optional(support_it, support_pt)

    top_ids = [int(x) for x in torch.topk(pre_logits, k=min(top_k + 3, pre_logits.shape[-1])).indices.tolist()]
    alt_ids = [idx for idx in top_ids if pipeline_token is None or idx != int(pipeline_token)][:top_k]
    alt_delta = finite_mean(float((full_logits[idx] - pre_logits[idx]).item()) for idx in alt_ids)
    target_vs_alt = _diff_optional(support_pipeline, alt_delta)

    neg = negative_parallel_component(update, pre)
    rem = update - neg
    neg_selected = _selected_logits(hidden=pre + neg, token_ids=target_ids, final_norm=final_norm, lm_head=lm_head)
    rem_selected = _selected_logits(hidden=pre + rem, token_ids=target_ids, final_norm=final_norm, lm_head=lm_head)
    neg_support_it = _delta_selected(pre_selected, neg_selected, y_it)
    neg_support_pt = _delta_selected(pre_selected, neg_selected, y_pt)
    neg_support_pipeline = _delta_selected(pre_selected, neg_selected, pipeline_token)
    neg_alt_delta = _component_alt_delta(
        pre=pre,
        component=neg,
        alt_ids=alt_ids,
        final_norm=final_norm,
        lm_head=lm_head,
    )
    rem_margin_it_pt = _diff_optional(
        _delta_selected(pre_selected, rem_selected, y_it),
        _delta_selected(pre_selected, rem_selected, y_pt),
    )
    rem_alt_delta = _component_alt_delta(
        pre=pre,
        component=rem,
        alt_ids=alt_ids,
        final_norm=final_norm,
        lm_head=lm_head,
    )

    delta = full_logits - pre_logits
    masked_delta = delta.clone()
    masked_delta[~real_token_mask.to(masked_delta.device)] = float("nan")
    top_supported = _nan_arg(masked_delta, largest=True)
    top_suppressed = _nan_arg(masked_delta, largest=False)
    top_supported_payload = _token_payload(tokenizer, top_supported)
    top_suppressed_payload = _token_payload(tokenizer, top_suppressed)

    delta_cos = mlp_delta_cosine(update, pre)
    return {
        "delta_cosine_mlp": delta_cos,
        "negative_parallel_norm": negative_parallel_norm(update, pre),
        "support_it_token": support_it,
        "support_pt_token": support_pt,
        "margin_writein_it_vs_pt": margin_it_pt,
        "support_pipeline_token": support_pipeline,
        "alt_top10_delta": alt_delta,
        "target_vs_alt_margin": target_vs_alt,
        "productive_opposition": productive_opposition(delta_cos, margin_it_pt),
        "opposition_support_it_token": neg_support_it,
        "opposition_support_pt_token": neg_support_pt,
        "opposition_margin_it_vs_pt": _diff_optional(neg_support_it, neg_support_pt),
        "opposition_support_pipeline_token": neg_support_pipeline,
        "opposition_alt_top10_delta": neg_alt_delta,
        "opposition_target_vs_alt_margin": _diff_optional(neg_support_pipeline, neg_alt_delta),
        "remainder_margin_it_vs_pt": rem_margin_it_pt,
        "remainder_target_vs_alt_margin": _diff_optional(
            _delta_selected(pre_selected, rem_selected, pipeline_token),
            rem_alt_delta,
        ),
        "top_supported_token_id": top_supported,
        "top_supported_category": (top_supported_payload or {}).get("token_category_collapsed"),
        "top_supported_token_class": _token_class(top_supported, y_it=y_it, y_pt=y_pt, pipeline_token=pipeline_token),
        "top_suppressed_token_id": top_suppressed,
        "top_suppressed_category": (top_suppressed_payload or {}).get("token_category_collapsed"),
        "top_suppressed_token_class": _token_class(top_suppressed, y_it=y_it, y_pt=y_pt, pipeline_token=pipeline_token),
    }


def _delta_selected(before: dict[int, float], after: dict[int, float], token_id: int | None) -> float | None:
    if token_id is None:
        return None
    token = int(token_id)
    if token not in before or token not in after:
        return None
    return float(after[token] - before[token])


def _diff_optional(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _component_alt_delta(
    *,
    pre: torch.Tensor,
    component: torch.Tensor,
    alt_ids: list[int],
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> float | None:
    if not alt_ids:
        return None
    before = _selected_logits(hidden=pre, token_ids=alt_ids, final_norm=final_norm, lm_head=lm_head)
    after = _selected_logits(hidden=pre + component, token_ids=alt_ids, final_norm=final_norm, lm_head=lm_head)
    return finite_mean(_delta_selected(before, after, token_id) for token_id in alt_ids)


def _nan_arg(values: torch.Tensor, *, largest: bool) -> int | None:
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        return None
    work = values.clone()
    work[~finite] = float("-inf") if largest else float("inf")
    return int((torch.argmax(work) if largest else torch.argmin(work)).item())


def _run_condition_projection(
    *,
    model_name: str,
    condition: ConditionSpec,
    dataset_record: dict[str, Any],
    prompt_mode: str,
    prefix_token_ids: list[int],
    y_it: int,
    y_pt: int,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    top_k: int,
) -> dict[str, Any]:
    host = condition.host_variant
    tokenizer = tokenizers[host]
    model = models[host]
    prompt = _prompt_for_condition(
        record=dataset_record,
        condition=condition,
        tokenizer=tokenizer,
        prompt_mode=prompt_mode,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_ids = prompt_ids + [int(x) for x in prefix_token_ids]
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    capture = _make_capture(
        model_name=model_name,
        condition=condition,
        models=models,
        steering_adapter=steering_adapter,
    )
    try:
        with torch.no_grad():
            capture.reset_step()
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            snapshot = capture.snapshot()
    finally:
        capture.close()

    final_norm = steering_adapter.get_final_norm(model)
    lm_head = steering_adapter.get_lm_head(model)
    mask = real_token_masks[host]
    final_hidden = _tensor_at_batch0(snapshot.residual_output[-1])
    final_logits = _full_logits(hidden=final_hidden, final_norm=final_norm, lm_head=lm_head, real_token_mask=mask)
    pipeline_token = int(final_logits.argmax().item())
    winner_class = _token_class(pipeline_token, y_it=y_it, y_pt=y_pt, pipeline_token=pipeline_token)

    layer_rows = []
    for layer_idx, (pre_raw, update_raw) in enumerate(
        zip(snapshot.pre_mlp_residual, snapshot.mlp_output, strict=False)
    ):
        pre = _tensor_at_batch0(pre_raw)
        update = _tensor_at_batch0(update_raw)
        row = _layer_metrics(
            pre=pre,
            update=update,
            final_norm=final_norm,
            lm_head=lm_head,
            real_token_mask=mask,
            tokenizer=tokenizer,
            y_it=y_it,
            y_pt=y_pt,
            pipeline_token=pipeline_token,
            top_k=top_k,
        )
        row["layer"] = layer_idx
        layer_rows.append(row)

    return {
        "condition": condition.name,
        "host_variant": condition.host_variant,
        "donor_variant": condition.donor_variant,
        "graft_kind": condition.graft_kind,
        "control_mode": condition.control_mode,
        "control_seed": condition.control_seed,
        "graft_window": _graft_window(model_name, condition),
        "pipeline_token": _token_payload(tokenizer, pipeline_token),
        "winner_class": winner_class,
        "layers": layer_rows,
        "windows": _summarize_windows(model_name=model_name, condition=condition, layer_rows=layer_rows),
    }


def _unique_events(manifest_record: dict[str, Any], event_kinds: list[str]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    seen: dict[tuple[int, int, int], str] = {}
    events = manifest_record.get("divergence_events") or {}
    for kind in event_kinds:
        event = events.get(kind)
        if not isinstance(event, dict):
            continue
        pt_token = event.get("pt_token") or {}
        it_token = event.get("it_token") or {}
        if pt_token.get("token_id") is None or it_token.get("token_id") is None:
            continue
        key = (int(event["step"]), int(pt_token["token_id"]), int(it_token["token_id"]))
        if key in seen:
            out.append((kind, {"duplicate_of": seen[key], **event}))
        else:
            seen[key] = kind
            out.append((kind, event))
    return out


def _prefix_ids_for_event(manifest_record: dict[str, Any], event: dict[str, Any]) -> list[int]:
    step = int(event["step"])
    free_runs = manifest_record.get("free_runs") or {}
    base = free_runs.get("A_pt_raw") or {}
    return [int(x) for x in (base.get("generated_token_ids") or [])[:step]]


def collect_record(
    *,
    model_name: str,
    manifest_record: dict[str, Any],
    dataset_record: dict[str, Any],
    prompt_mode: str,
    conditions: list[str],
    event_kinds: list[str],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    top_k: int,
) -> dict[str, Any]:
    prompt_id = str(manifest_record.get("prompt_id"))
    out_events: dict[str, Any] = {}
    event_cache: dict[tuple[int, int, int], dict[str, Any]] = {}
    for kind, event in _unique_events(manifest_record, event_kinds):
        if "duplicate_of" in event:
            duplicate_of = str(event["duplicate_of"])
            out_events[kind] = {**out_events[duplicate_of], "duplicate_of": duplicate_of}
            continue
        y_pt = int(event["pt_token"]["token_id"])
        y_it = int(event["it_token"]["token_id"])
        key = (int(event["step"]), y_pt, y_it)
        prefix_ids = _prefix_ids_for_event(manifest_record, event)
        condition_payload = {}
        if key not in event_cache:
            for name in conditions:
                spec = CONDITIONS[name]
                condition_payload[name] = _run_condition_projection(
                    model_name=model_name,
                    condition=spec,
                    dataset_record=dataset_record,
                    prompt_mode=prompt_mode,
                    prefix_token_ids=prefix_ids,
                    y_it=y_it,
                    y_pt=y_pt,
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    steering_adapter=steering_adapter,
                    device=device,
                    top_k=top_k,
                )
            event_cache[key] = {
                "event": event,
                "prefix_length": len(prefix_ids),
                "condition_token_at_step": (
                    ((manifest_record.get("readouts") or {}).get(kind) or {}).get("condition_token_at_step")
                ),
                "conditions": condition_payload,
            }
        out_events[kind] = event_cache[key]

    return {
        "prompt_id": prompt_id,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "events": out_events,
    }


def _done_prompt_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for row in _gzip_jsonl_rows(out_path):
        prompt_id = row.get("prompt_id")
        if prompt_id:
            done.add(str(prompt_id))
    return done


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
    dataset_by_id = _dataset_lookup(args.dataset)
    manifest_rows = _load_manifest_records(
        exp20_root=args.exp20_root,
        exp20_fallback_root=args.exp20_fallback_root,
        prompt_mode=args.prompt_mode,
        model=args.model,
        n_examples=args.n_eval_examples,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"records_w{args.worker_index}.jsonl.gz"
    done = _done_prompt_ids(out_path)
    conditions = args.conditions or DEFAULT_CONDITIONS
    event_kinds = args.event_kinds or DEFAULT_EVENT_KINDS
    log.info("[exp21] %s/%s worker %d/%d resume=%d", args.prompt_mode, args.model, args.worker_index, args.n_workers, len(done))
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for idx, manifest_record in enumerate(manifest_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            if prompt_id in done:
                continue
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                log.warning("[exp21] missing dataset record for prompt_id=%s", prompt_id)
                continue
            try:
                result = collect_record(
                    model_name=args.model,
                    manifest_record=manifest_record,
                    dataset_record=dataset_record,
                    prompt_mode=args.prompt_mode,
                    conditions=conditions,
                    event_kinds=event_kinds,
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    steering_adapter=steering_adapter,
                    device=device,
                    top_k=args.top_k,
                )
                fout.write(json.dumps(result, separators=(",", ":")) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[exp21] prompt %s failed: %s", prompt_id, exc)
            if (idx + 1) % 5 == 0:
                log.info("[exp21] %s/%s %d/%d prompts", args.prompt_mode, args.model, idx + 1, len(manifest_rows))


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    seen: set[str] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for idx in range(n_workers):
            path = out_dir / f"records_w{idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp21] missing worker file %s", path)
                continue
            for row in _gzip_jsonl_rows(path):
                prompt_id = str(row.get("prompt_id", ""))
                if prompt_id and prompt_id in seen:
                    continue
                if prompt_id:
                    seen.add(prompt_id)
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp21 productive-opposition projections.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    # The Exp20 manifest determines the evaluation subset. Use the full lookup table
    # so fallback manifests from earlier runs (e.g. DeepSeek) do not appear missing.
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["native", "raw_shared"], default="native")
    parser.add_argument("--conditions", nargs="*", choices=list(CONDITIONS), default=None)
    parser.add_argument("--event-kinds", nargs="*", choices=DEFAULT_EVENT_KINDS, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_workers(args.out_dir, args.n_workers)
        return
    run_worker(args)


if __name__ == "__main__":
    main()
