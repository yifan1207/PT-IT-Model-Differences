from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import numpy as np
from transformers.cache_utils import DynamicCache

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.tuned_lens import _load_probes
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import classify_generated_tokens_by_word
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp11_matched_prefix_mlp_graft.mlp_graft import (
    ArchitectureProbe,
    GradientDirectionAccumulator,
    LayerStepMetrics,
    MechanismLayerMetrics,
    PipelineCapture,
    PipelineRun,
    PipelineStepRecord,
    ReadoutSpec,
    compute_batched_mechanism_metrics,
    compute_layer_metrics,
    first_stable_below_threshold,
    first_stable_true,
    gradient_direction_payload,
    init_gradient_direction_accumulator,
    logits_from_residuals,
    raw_logits_from_residuals,
    select_row_step_tensors,
)
from src.poc.exp11_matched_prefix_mlp_graft.structural_tokens import build_structural_token_masks


VALID_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b", "deepseek_v2_lite"]
KL_THRESHOLDS = [0.05, 0.1, 0.2, 0.5, 1.0]
DEFAULT_BATCH_SIZE = {
    "gemma3_4b": 64,
    "qwen3_4b": 64,
    "llama31_8b": 32,
    "mistral_7b": 32,
    "olmo2_7b": 32,
    "deepseek_v2_lite": 48,
}
GOVERNANCE_PATTERNS = {
    "paragraph_header_h1": r"\n\n# ",
    "paragraph_header_h2": r"\n\n## ",
    "bullet_dash": r"\n- ",
    "ordered_1": r"\n1\. ",
    "ordered_2": r"\n2\. ",
    "lead_heres_apostrophe": r"\bHere's\b",
    "lead_here_is": r"\bHere is\b",
    "lead_sure": r"\bSure\b",
    "lead_let_me": r"\bLet me\b",
    "lead_the": r"\bThe ",
}
_WORD_INITIAL_RE = re.compile(r"[A-Za-z']+")
TRAJECTORY_METRICS = [
    "delta_cosine",
    "kl_to_own_final",
    "structural_mass_prob_tier1",
    "entropy",
    "cross_kl",
    "kl_to_pt_final",
    "residual_cosine",
    "residual_divergence",
]
DEPTH_ABLATION_WINDOWS = {
    "gemma3_4b": {
        "B_early_raw": (0, 14),
        "B_mid_raw": (10, 24),
        "B_late_raw": (20, 34),
    },
    "llama31_8b": {
        "B_early_raw": (0, 13),
        "B_mid_raw": (9, 22),
        "B_late_raw": (19, 32),
    },
    "qwen3_4b": {
        "B_early_raw": (0, 14),
        "B_mid_raw": (11, 25),
        "B_late_raw": (22, 36),
    },
    "mistral_7b": {
        "B_early_raw": (0, 13),
        "B_mid_raw": (9, 22),
        "B_late_raw": (19, 32),
    },
    "olmo2_7b": {
        "B_early_raw": (0, 13),
        "B_mid_raw": (9, 22),
        "B_late_raw": (19, 32),
    },
    "deepseek_v2_lite": {
        "B_early_raw": (0, 11),
        "B_mid_raw": (8, 19),
        "B_late_raw": (16, 27),
    },
}
CAUSAL_MECHANISM_CORE_PIPELINES = {"A_prime_raw", "B_late_raw", "C_it_chat", "D_late_ptswap"}
CAUSAL_MECHANISM_AXES = ["anti_top1", "support_teacher", "anti_kl_final"]


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
    free_argmax_token_ids: list[int] = field(default_factory=list)


@dataclass
class FixedBatch:
    request_ids: list[str]
    next_input_ids: torch.Tensor
    valid_lengths: torch.Tensor
    cache_len: int
    past_key_values: DynamicCache
    finished_mask: torch.Tensor


class TrajectoryAccumulator:
    def __init__(self, *, metric_names: list[str], n_layers: int) -> None:
        self.metric_names = metric_names
        self.n_layers = n_layers
        self._stats: dict[str, dict[str, dict[str, list[float]]]] = {}

    def add(self, pipeline: str, metrics: LayerStepMetrics) -> None:
        pipeline_stats = self._stats.setdefault(
            pipeline,
            {
                metric: {
                    "sum": [0.0] * self.n_layers,
                    "count": [0.0] * self.n_layers,
                }
                for metric in self.metric_names
            },
        )
        metrics_dict = asdict(metrics)
        for metric in self.metric_names:
            values = metrics_dict.get(metric)
            if values is None:
                continue
            for layer_idx, value in enumerate(values):
                if value is None:
                    continue
                pipeline_stats[metric]["sum"][layer_idx] += float(value)
                pipeline_stats[metric]["count"][layer_idx] += 1.0

    def to_stats_dict(
        self,
        *,
        readout_name: str,
        readout_name_by_pipeline: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "readout_name": readout_name,
            "metrics": self._stats,
        }
        if readout_name_by_pipeline is not None:
            payload["readout_name_by_pipeline"] = dict(readout_name_by_pipeline)
        return payload


def _merge_trajectory_stats(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    if not existing:
        return update
    merged = {
        "readout_name": update.get("readout_name") or existing.get("readout_name"),
        "metrics": existing.get("metrics", {}),
    }
    if existing.get("readout_name_by_pipeline") or update.get("readout_name_by_pipeline"):
        merged["readout_name_by_pipeline"] = {
            **existing.get("readout_name_by_pipeline", {}),
            **update.get("readout_name_by_pipeline", {}),
        }
    for pipeline, pipeline_metrics in update.get("metrics", {}).items():
        dst_pipeline = merged["metrics"].setdefault(pipeline, {})
        for metric_name, metric_stats in pipeline_metrics.items():
            dst_metric = dst_pipeline.setdefault(
                metric_name,
                {
                    "sum": [0.0] * len(metric_stats.get("sum", [])),
                    "count": [0.0] * len(metric_stats.get("count", [])),
                },
            )
            for idx, value in enumerate(metric_stats.get("sum", [])):
                dst_metric["sum"][idx] += value
            for idx, value in enumerate(metric_stats.get("count", [])):
                dst_metric["count"][idx] += value
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp11 PT vs PT+IT MLP graft.")
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onset-layer", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Internal chunk size for memory management. Models loaded once, results saved per chunk.",
    )
    parser.add_argument(
        "--readout-mode",
        default="raw",
        choices=["raw", "tuned_pt_shared", "both"],
        help="Readout used for recorded layer metrics. Generation always uses true model logits.",
    )
    parser.add_argument(
        "--tuned-lens-dir",
        default=None,
        help="Root dir containing tuned-lens probes with v3 layout {model}/{variant}/probe_layer_*.pt.",
    )
    parser.add_argument(
        "--teacher-forced",
        action="store_true",
        help=(
            "Enable v11.1 teacher-forced mode: run IT free teacher (C), PT teacher-forced "
            "raw/template controls (A', A'_tmpl), and grafted raw/template branches "
            "(B, B2). Cross-pipeline metrics for teacher-forced "
            "branches are computed against C's residuals."
        ),
    )
    parser.add_argument(
        "--depth-ablation",
        action="store_true",
        help=(
            "Teacher-forced exp11.2 depth ablation: run C_it_chat, A_prime_raw, and "
            "equal-width early/mid/late raw graft branches. Requires --teacher-forced."
        ),
    )
    parser.add_argument(
        "--causal-combined",
        action="store_true",
        help=(
            "Teacher-forced combined Exp13/Exp14 run: run C_it_chat, A_prime_raw, "
            "B/D early-mid-late branches, and collect late-stage output-relevant "
            "mechanism summaries for core pipelines."
        ),
    )
    parser.add_argument(
        "--prompt-seed",
        type=int,
        default=None,
        help=(
            "If set, draw a uniform random subset of --n-prompts prompts from the full dataset "
            "using this seed (overrides the stratified _sample_prompts behavior)."
        ),
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


def _apply_prompt_shard(prompts: list[dict], shard_index: int, num_shards: int) -> list[dict]:
    if num_shards <= 1:
        return prompts
    return prompts[shard_index::num_shards]


def _random_subsample(records: list[dict], n_prompts: int, seed: int) -> list[dict]:
    if n_prompts >= len(records):
        return sorted(records, key=lambda r: r["id"])
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: r["id"])
    picked = rng.sample(pool, n_prompts)
    return sorted(picked, key=lambda r: r["id"])


def _safe_decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _safe_token_piece(tokenizer, token_id: int) -> str:
    piece = tokenizer.convert_ids_to_tokens(int(token_id))
    return piece if isinstance(piece, str) else _safe_decode(tokenizer, token_id)


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


def _assert_used_token_ids_compatible(
    tokenizer_src,
    tokenizer_dst,
    token_sequences: dict[str, list[int]] | list[list[int]],
    *,
    context: str,
    max_examples: int = 8,
) -> None:
    sequences = token_sequences.values() if isinstance(token_sequences, dict) else token_sequences
    seen: set[int] = set()
    problems: list[str] = []
    dst_vocab_size = len(tokenizer_dst)
    for seq in sequences:
        for token_id in seq:
            token_id = int(token_id)
            if token_id in seen:
                continue
            seen.add(token_id)
            if token_id < 0 or token_id >= dst_vocab_size:
                problems.append(f"id {token_id} out of range for target tokenizer (size={dst_vocab_size})")
            else:
                src_piece = _safe_token_piece(tokenizer_src, token_id)
                dst_piece = _safe_token_piece(tokenizer_dst, token_id)
                if src_piece != dst_piece:
                    problems.append(f"id {token_id}: src={src_piece!r} dst={dst_piece!r}")
            if len(problems) >= max_examples:
                break
        if len(problems) >= max_examples:
            break
    if problems:
        joined = "; ".join(problems)
        raise ValueError(f"Incompatible token IDs for {context}: {joined}")


def _is_depth_ablation(args: argparse.Namespace) -> bool:
    return bool(args.teacher_forced and args.depth_ablation)


def _is_causal_combined(args: argparse.Namespace) -> bool:
    return bool(args.teacher_forced and getattr(args, "causal_combined", False))


def _depth_windows_for_model(model: str, spec) -> dict[str, tuple[int, int]]:
    windows = DEPTH_ABLATION_WINDOWS[model]
    expected_width = spec.n_layers - spec.corrective_onset
    for pipeline_name, (start, end) in windows.items():
        if end - start != expected_width:
            raise ValueError(
                f"Depth-ablation window width mismatch for {model}/{pipeline_name}: "
                f"{end - start} != expected {expected_width}"
            )
        if start < 0 or end > spec.n_layers or start >= end:
            raise ValueError(
                f"Invalid depth-ablation window for {model}/{pipeline_name}: [{start}, {end})"
            )
    return windows


def _causal_windows_for_model(model: str, spec) -> dict[str, tuple[int, int]]:
    base = _depth_windows_for_model(model, spec)
    out = dict(base)
    out.update(
        {
            "D_early_ptswap": base["B_early_raw"],
            "D_mid_ptswap": base["B_mid_raw"],
            "D_late_ptswap": base["B_late_raw"],
        }
    )
    return out


def _make_request_states(
    prompts: list[dict],
    tokenizer,
    done_ids: set[str],
    *,
    prompt_text_by_id: dict[str, str] | None = None,
    prompt_token_ids_by_id: dict[str, list[int]] | None = None,
) -> list[RequestState]:
    requests: list[RequestState] = []
    for record in prompts:
        if record["id"] in done_ids:
            continue
        prompt = (
            prompt_text_by_id.get(record["id"], record["formats"]["B"])
            if prompt_text_by_id is not None
            else record["formats"]["B"]
        )
        token_ids = (
            list(prompt_token_ids_by_id[record["id"]])
            if prompt_token_ids_by_id is not None
            else tokenizer(prompt, add_special_tokens=True)["input_ids"]
        )
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
    free_argmax_token_id: int | None = None,
) -> None:
    request.generated_token_ids.append(token_id)
    request.generated_tokens.append({"token_id": token_id, "token_str": token_str})
    request.step_records.append(PipelineStepRecord(token_id=token_id, token_str=token_str, metrics=metrics))
    if store_baseline:
        request.baseline_cache.append([tensor.detach().cpu() for tensor in row_residuals])
    if free_argmax_token_id is not None:
        request.free_argmax_token_ids.append(free_argmax_token_id)


def _resolve_probe_dir(root: str | Path, model: str, variant: str) -> Path:
    root_path = Path(root)
    v3 = root_path / model / variant
    if v3.exists():
        return v3
    v2 = root_path / model / "tuned_lens" / variant
    if v2.exists():
        return v2
    raise FileNotFoundError(f"No tuned-lens probe dir found for {model}/{variant} under {root_path}")


def _load_readouts(
    *,
    args: argparse.Namespace,
    spec,
    steering_adapter,
    model_raw_pt,
    model_raw_it,
    device: torch.device,
) -> dict[str, ReadoutSpec]:
    final_norm_pt = steering_adapter.get_final_norm(model_raw_pt)
    lm_head_pt = steering_adapter.get_lm_head(model_raw_pt)
    final_norm_it = steering_adapter.get_final_norm(model_raw_it)
    lm_head_it = steering_adapter.get_lm_head(model_raw_it)
    bundle = {
        "raw_pt": ReadoutSpec(name="raw_logit_lens_pt", final_norm=final_norm_pt, lm_head=lm_head_pt),
        "raw_it": ReadoutSpec(name="raw_logit_lens_it", final_norm=final_norm_it, lm_head=lm_head_it),
    }
    if args.readout_mode == "raw":
        return bundle

    if args.tuned_lens_dir is None:
        raise ValueError("--tuned-lens-dir is required when --readout-mode is tuned_pt_shared or both")

    pt_probe_dir = _resolve_probe_dir(args.tuned_lens_dir, args.model, "pt")
    pt_probes = _load_probes(pt_probe_dir, spec.d_model, device)
    if len(pt_probes) != spec.n_layers:
        raise ValueError(
            f"Incomplete PT tuned-lens probes for {args.model}: expected {spec.n_layers}, found {len(pt_probes)}"
        )
    bundle["tuned_pt"] = ReadoutSpec(
        name="tuned_pt_shared",
        final_norm=final_norm_pt,
        lm_head=lm_head_pt,
        probes=pt_probes,
    )
    if args.readout_mode == "tuned_pt_shared":
        return bundle

    it_probe_dir = _resolve_probe_dir(args.tuned_lens_dir, args.model, "it")
    it_probes = _load_probes(it_probe_dir, spec.d_model, device)
    if len(it_probes) != spec.n_layers:
        raise ValueError(
            f"Incomplete IT tuned-lens probes for {args.model}: expected {spec.n_layers}, found {len(it_probes)}"
        )
    bundle["tuned_it"] = ReadoutSpec(
        name="tuned_native_it",
        final_norm=final_norm_it,
        lm_head=lm_head_it,
        probes=it_probes,
    )
    return bundle


def _maybe_secondary_logits(
    *,
    secondary_readout: ReadoutSpec | None,
    primary_readout: ReadoutSpec,
    residuals: list[torch.Tensor],
    primary_layer_logits: torch.Tensor,
    real_token_mask: torch.Tensor,
) -> torch.Tensor | None:
    if secondary_readout is None:
        return None
    if secondary_readout is primary_readout:
        return primary_layer_logits
    return _apply_real_token_mask(
        logits_from_residuals(secondary_readout, residuals),
        real_token_mask,
    )


def _prefill_requests(
    *,
    pipeline_name: str,
    new_requests: list[RequestState],
    requests_by_id: dict[str, RequestState],
    model_raw,
    capture: PipelineCapture,
    tokenizer,
    pad_token_id: int,
    real_token_mask: torch.Tensor,
    tier1_mask: torch.Tensor,
    eos_token_ids: set[int],
    inactive_token_id: int,
    max_new_tokens: int,
    store_baseline: bool,
    baseline_lookup: dict[str, list[list[torch.Tensor]]] | None,
    primary_readout: ReadoutSpec,
    secondary_readout: ReadoutSpec | None,
    baseline_primary_readout: ReadoutSpec | None,
    baseline_secondary_readout: ReadoutSpec | None,
    secondary_accumulator: TrajectoryAccumulator | None,
    mechanism_readout: ReadoutSpec | None = None,
    mechanism_baseline_lookup: dict[str, list[list[torch.Tensor]]] | None = None,
    mechanism_baseline_raw_readout: ReadoutSpec | None = None,
    mechanism_layer_indices: list[int] | None = None,
    gradient_accumulator: GradientDirectionAccumulator | None = None,
    teacher_tokens_by_prompt: dict[str, list[int]] | None = None,
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
    primary_layer_logits = _apply_real_token_mask(
        logits_from_residuals(primary_readout, step_tensors.residual_output),
        real_token_mask.to(device),
    )
    secondary_layer_logits = _maybe_secondary_logits(
        secondary_readout=secondary_readout,
        primary_readout=primary_readout,
        residuals=step_tensors.residual_output,
        primary_layer_logits=primary_layer_logits,
        real_token_mask=real_token_mask.to(device),
    )
    free_argmax_ids = logits.argmax(dim=-1)
    next_token_ids = free_argmax_ids.clone()
    forced_exhausted = [False] * len(new_requests)
    if teacher_tokens_by_prompt is not None:
        for row_idx, request in enumerate(new_requests):
            teacher_seq = teacher_tokens_by_prompt.get(request.prompt_id)
            step_idx = len(request.generated_token_ids)
            if teacher_seq is None or step_idx >= len(teacher_seq):
                forced_exhausted[row_idx] = True
            else:
                next_token_ids[row_idx] = teacher_seq[step_idx]

    mechanism_results: list[MechanismLayerMetrics] | None = None
    if mechanism_readout is not None and mechanism_layer_indices:
        step_indices = [len(request.generated_token_ids) for request in new_requests]
        baseline_raw_logits = _build_mechanism_baseline_raw_logits(
            request_ids=[request.prompt_id for request in new_requests],
            step_indices=step_indices,
            baseline_lookup=mechanism_baseline_lookup,
            baseline_raw_readout=mechanism_baseline_raw_readout,
            fallback_step_tensors=step_tensors,
            layer_indices=mechanism_layer_indices,
        )
        mechanism_results = compute_batched_mechanism_metrics(
            raw_readout=mechanism_readout,
            step_tensors=step_tensors,
            chosen_token_ids=next_token_ids,
            tokenizer=tokenizer,
            layer_indices=mechanism_layer_indices,
            baseline_raw_logits=baseline_raw_logits,
            gradient_accumulator=gradient_accumulator,
        )

    next_inputs: list[int] = []
    finished_mask = torch.zeros((len(new_requests),), dtype=torch.bool, device=device)
    for row_idx, request in enumerate(new_requests):
        if forced_exhausted[row_idx]:
            finished_mask[row_idx] = True
            next_inputs.append(inactive_token_id)
            continue
        token_id = int(next_token_ids[row_idx].item())
        token_str = _safe_decode(tokenizer, token_id)
        free_argmax_id = int(free_argmax_ids[row_idx].item()) if teacher_tokens_by_prompt is not None else None
        row_step_tensors = select_row_step_tensors(step_tensors, row_idx)
        metric_kwargs: dict[str, Any] = {}
        secondary_metric_kwargs: dict[str, Any] = {}
        if baseline_lookup is not None and request.prompt_id in baseline_lookup and len(baseline_lookup[request.prompt_id]) > 0:
            baseline_residuals = [tensor.to(device) for tensor in baseline_lookup[request.prompt_id][0]]
            if baseline_primary_readout is None:
                raise ValueError("baseline_primary_readout is required when baseline_lookup is provided")
            baseline_primary_logits = _apply_real_token_mask(
                logits_from_residuals(baseline_primary_readout, baseline_residuals)[0],
                real_token_mask.to(device),
            )
            metric_kwargs["baseline_logits"] = baseline_primary_logits
            metric_kwargs["baseline_residuals"] = baseline_residuals
            if baseline_secondary_readout is not None:
                baseline_secondary_logits = _apply_real_token_mask(
                    logits_from_residuals(baseline_secondary_readout, baseline_residuals)[0],
                    real_token_mask.to(device),
                )
                secondary_metric_kwargs["baseline_logits"] = baseline_secondary_logits
                secondary_metric_kwargs["baseline_residuals"] = baseline_residuals
        metrics = compute_layer_metrics(
            pipeline_logits=primary_layer_logits[row_idx],
            step_tensors=row_step_tensors,
            chosen_token_id=token_id,
            tier1_mask=tier1_mask,
            **metric_kwargs,
        )
        if mechanism_results is not None:
            _apply_mechanism_metrics(
                metrics,
                mechanism_results[row_idx],
                late_layers=mechanism_layer_indices or [],
            )
        _record_step(
            request,
            token_id,
            token_str,
            metrics,
            row_step_tensors.residual_output,
            store_baseline=store_baseline,
            free_argmax_token_id=free_argmax_id,
        )
        if secondary_accumulator is not None and secondary_layer_logits is not None:
            secondary_metrics = compute_layer_metrics(
                pipeline_logits=secondary_layer_logits[row_idx],
                step_tensors=row_step_tensors,
                chosen_token_id=token_id,
                tier1_mask=tier1_mask,
                **secondary_metric_kwargs,
            )
            secondary_accumulator.add(pipeline_name, secondary_metrics)
        finished = (
            token_id in eos_token_ids
            or len(request.generated_token_ids) >= max_new_tokens
            or forced_exhausted[row_idx]
        )
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
    pipeline_name: str,
    active: FixedBatch,
    requests_by_id: dict[str, RequestState],
    model_raw,
    capture: PipelineCapture,
    tokenizer,
    real_token_mask: torch.Tensor,
    tier1_mask: torch.Tensor,
    eos_token_ids: set[int],
    inactive_token_id: int,
    max_new_tokens: int,
    store_baseline: bool,
    baseline_lookup: dict[str, list[list[torch.Tensor]]] | None,
    primary_readout: ReadoutSpec,
    secondary_readout: ReadoutSpec | None,
    baseline_primary_readout: ReadoutSpec | None,
    baseline_secondary_readout: ReadoutSpec | None,
    secondary_accumulator: TrajectoryAccumulator | None,
    mechanism_readout: ReadoutSpec | None = None,
    mechanism_baseline_lookup: dict[str, list[list[torch.Tensor]]] | None = None,
    mechanism_baseline_raw_readout: ReadoutSpec | None = None,
    mechanism_layer_indices: list[int] | None = None,
    gradient_accumulator: GradientDirectionAccumulator | None = None,
    teacher_tokens_by_prompt: dict[str, list[int]] | None = None,
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
    primary_layer_logits = _apply_real_token_mask(
        logits_from_residuals(primary_readout, step_tensors.residual_output),
        real_token_mask.to(device),
    )
    secondary_layer_logits = _maybe_secondary_logits(
        secondary_readout=secondary_readout,
        primary_readout=primary_readout,
        residuals=step_tensors.residual_output,
        primary_layer_logits=primary_layer_logits,
        real_token_mask=real_token_mask.to(device),
    )
    free_argmax_ids = logits.argmax(dim=-1)
    next_token_ids = free_argmax_ids.clone()
    forced_exhausted: list[bool] = [False] * len(active.request_ids)
    if teacher_tokens_by_prompt is not None:
        for row_idx, prompt_id in enumerate(active.request_ids):
            if bool(active.finished_mask[row_idx].item()):
                continue
            request = requests_by_id[prompt_id]
            teacher_seq = teacher_tokens_by_prompt.get(prompt_id)
            step_idx = len(request.generated_token_ids)
            if teacher_seq is None or step_idx >= len(teacher_seq):
                forced_exhausted[row_idx] = True
            else:
                next_token_ids[row_idx] = teacher_seq[step_idx]

    mechanism_results: list[MechanismLayerMetrics] | None = None
    if mechanism_readout is not None and mechanism_layer_indices:
        step_indices = [
            len(requests_by_id[prompt_id].generated_token_ids)
            for prompt_id in active.request_ids
        ]
        baseline_raw_logits = _build_mechanism_baseline_raw_logits(
            request_ids=list(active.request_ids),
            step_indices=step_indices,
            baseline_lookup=mechanism_baseline_lookup,
            baseline_raw_readout=mechanism_baseline_raw_readout,
            fallback_step_tensors=step_tensors,
            layer_indices=mechanism_layer_indices,
        )
        mechanism_results = compute_batched_mechanism_metrics(
            raw_readout=mechanism_readout,
            step_tensors=step_tensors,
            chosen_token_ids=next_token_ids,
            tokenizer=tokenizer,
            layer_indices=mechanism_layer_indices,
            baseline_raw_logits=baseline_raw_logits,
            gradient_accumulator=gradient_accumulator,
        )

    next_inputs: list[int] = []
    updated_valid_lengths = active.valid_lengths.clone()
    next_finished_mask = active.finished_mask.clone()
    for row_idx, prompt_id in enumerate(active.request_ids):
        if bool(active.finished_mask[row_idx].item()):
            next_inputs.append(inactive_token_id)
            continue
        if forced_exhausted[row_idx]:
            next_finished_mask[row_idx] = True
            next_inputs.append(inactive_token_id)
            continue

        request = requests_by_id[prompt_id]
        step_idx = len(request.generated_token_ids)
        token_id = int(next_token_ids[row_idx].item())
        token_str = _safe_decode(tokenizer, token_id)
        free_argmax_id = int(free_argmax_ids[row_idx].item()) if teacher_tokens_by_prompt is not None else None
        row_step_tensors = select_row_step_tensors(step_tensors, row_idx)
        metric_kwargs: dict[str, Any] = {}
        secondary_metric_kwargs: dict[str, Any] = {}
        if baseline_lookup is not None:
            baseline_steps = baseline_lookup.get(prompt_id, [])
            if step_idx < len(baseline_steps):
                baseline_residuals = [tensor.to(device) for tensor in baseline_steps[step_idx]]
                if baseline_primary_readout is None:
                    raise ValueError("baseline_primary_readout is required when baseline_lookup is provided")
                baseline_primary_logits = _apply_real_token_mask(
                    logits_from_residuals(baseline_primary_readout, baseline_residuals)[0],
                    real_token_mask.to(device),
                )
                metric_kwargs["baseline_logits"] = baseline_primary_logits
                metric_kwargs["baseline_residuals"] = baseline_residuals
                if baseline_secondary_readout is not None:
                    baseline_secondary_logits = _apply_real_token_mask(
                        logits_from_residuals(baseline_secondary_readout, baseline_residuals)[0],
                        real_token_mask.to(device),
                    )
                    secondary_metric_kwargs["baseline_logits"] = baseline_secondary_logits
                    secondary_metric_kwargs["baseline_residuals"] = baseline_residuals
        metrics = compute_layer_metrics(
            pipeline_logits=primary_layer_logits[row_idx],
            step_tensors=row_step_tensors,
            chosen_token_id=token_id,
            tier1_mask=tier1_mask,
            **metric_kwargs,
        )
        if mechanism_results is not None:
            _apply_mechanism_metrics(
                metrics,
                mechanism_results[row_idx],
                late_layers=mechanism_layer_indices or [],
            )
        _record_step(
            request,
            token_id,
            token_str,
            metrics,
            row_step_tensors.residual_output,
            store_baseline=store_baseline,
            free_argmax_token_id=free_argmax_id,
        )
        if secondary_accumulator is not None and secondary_layer_logits is not None:
            secondary_metrics = compute_layer_metrics(
                pipeline_logits=secondary_layer_logits[row_idx],
                step_tensors=row_step_tensors,
                chosen_token_id=token_id,
                tier1_mask=tier1_mask,
                **secondary_metric_kwargs,
            )
            secondary_accumulator.add(pipeline_name, secondary_metrics)
        finished = (
            token_id in eos_token_ids
            or len(request.generated_token_ids) >= max_new_tokens
            or forced_exhausted[row_idx]
        )
        updated_valid_lengths[row_idx] += 1
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
    pipeline_name: str,
    requests: list[RequestState],
    model_raw,
    capture: PipelineCapture,
    tokenizer,
    pad_token_id: int,
    real_token_mask: torch.Tensor,
    tier1_mask: torch.Tensor,
    eos_token_ids: set[int],
    max_new_tokens: int,
    inactive_token_id: int,
    batch_size: int,
    store_baseline: bool,
    baseline_lookup: dict[str, list[list[torch.Tensor]]] | None,
    primary_readout: ReadoutSpec,
    secondary_readout: ReadoutSpec | None,
    baseline_primary_readout: ReadoutSpec | None,
    baseline_secondary_readout: ReadoutSpec | None,
    secondary_accumulator: TrajectoryAccumulator | None,
    mechanism_readout: ReadoutSpec | None = None,
    mechanism_baseline_lookup: dict[str, list[list[torch.Tensor]]] | None = None,
    mechanism_baseline_raw_readout: ReadoutSpec | None = None,
    mechanism_layer_indices: list[int] | None = None,
    gradient_accumulator: GradientDirectionAccumulator | None = None,
    teacher_tokens_by_prompt: dict[str, list[int]] | None = None,
) -> dict[str, PipelineRun]:
    requests_by_id = {request.prompt_id: request for request in requests}
    for start in range(0, len(requests), batch_size):
        batch_requests = requests[start : start + batch_size]
        active = _prefill_requests(
            pipeline_name=pipeline_name,
            new_requests=batch_requests,
            requests_by_id=requests_by_id,
            model_raw=model_raw,
            capture=capture,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            real_token_mask=real_token_mask,
            tier1_mask=tier1_mask,
            eos_token_ids=eos_token_ids,
            inactive_token_id=inactive_token_id,
            max_new_tokens=max_new_tokens,
            store_baseline=store_baseline,
            baseline_lookup=baseline_lookup,
            primary_readout=primary_readout,
            secondary_readout=secondary_readout,
            baseline_primary_readout=baseline_primary_readout,
            baseline_secondary_readout=baseline_secondary_readout,
            secondary_accumulator=secondary_accumulator,
            mechanism_readout=mechanism_readout,
            mechanism_baseline_lookup=mechanism_baseline_lookup,
            mechanism_baseline_raw_readout=mechanism_baseline_raw_readout,
            mechanism_layer_indices=mechanism_layer_indices,
            gradient_accumulator=gradient_accumulator,
            teacher_tokens_by_prompt=teacher_tokens_by_prompt,
        )

        while active is not None:
            active = _decode_fixed_batch(
                pipeline_name=pipeline_name,
                active=active,
                requests_by_id=requests_by_id,
                model_raw=model_raw,
                capture=capture,
                tokenizer=tokenizer,
                real_token_mask=real_token_mask,
                tier1_mask=tier1_mask,
                eos_token_ids=eos_token_ids,
                inactive_token_id=inactive_token_id,
                max_new_tokens=max_new_tokens,
                store_baseline=store_baseline,
                baseline_lookup=baseline_lookup,
                primary_readout=primary_readout,
                secondary_readout=secondary_readout,
                baseline_primary_readout=baseline_primary_readout,
                baseline_secondary_readout=baseline_secondary_readout,
                secondary_accumulator=secondary_accumulator,
                mechanism_readout=mechanism_readout,
                mechanism_baseline_lookup=mechanism_baseline_lookup,
                mechanism_baseline_raw_readout=mechanism_baseline_raw_readout,
                mechanism_layer_indices=mechanism_layer_indices,
                gradient_accumulator=gradient_accumulator,
                teacher_tokens_by_prompt=teacher_tokens_by_prompt,
            )

    return {
        request.prompt_id: PipelineRun(
            generated_token_ids=request.generated_token_ids,
            generated_tokens=request.generated_tokens,
            generated_text=tokenizer.decode(request.generated_token_ids, skip_special_tokens=True),
            step_records=request.step_records,
            baseline_cache=request.baseline_cache,
            free_argmax_token_ids=request.free_argmax_token_ids,
        )
        for request in requests
    }


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _run_pipeline_with_batch_fallback(
    *,
    pipeline_name: str,
    requests: list[RequestState],
    model_raw,
    capture_factory,
    tokenizer,
    pad_token_id: int,
    real_token_mask: torch.Tensor,
    tier1_mask: torch.Tensor,
    eos_token_ids: set[int],
    max_new_tokens: int,
    inactive_token_id: int,
    batch_size: int,
    store_baseline: bool,
    baseline_lookup: dict[str, list[list[torch.Tensor]]] | None,
    primary_readout: ReadoutSpec,
    secondary_readout: ReadoutSpec | None,
    baseline_primary_readout: ReadoutSpec | None,
    baseline_secondary_readout: ReadoutSpec | None,
    secondary_accumulator: TrajectoryAccumulator | None,
    mechanism_readout: ReadoutSpec | None = None,
    mechanism_baseline_lookup: dict[str, list[list[torch.Tensor]]] | None = None,
    mechanism_baseline_raw_readout: ReadoutSpec | None = None,
    mechanism_layer_indices: list[int] | None = None,
    gradient_accumulator: GradientDirectionAccumulator | None = None,
    teacher_tokens_by_prompt: dict[str, list[int]] | None = None,
) -> tuple[dict[str, PipelineRun], int]:
    current_batch_size = max(1, batch_size)
    while True:
        capture = capture_factory()
        fresh_requests = _clone_requests(requests)
        try:
            runs = _run_pipeline_fixed_batches(
                pipeline_name=pipeline_name,
                requests=fresh_requests,
                model_raw=model_raw,
                capture=capture,
                tokenizer=tokenizer,
                pad_token_id=pad_token_id,
                real_token_mask=real_token_mask,
                tier1_mask=tier1_mask,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                inactive_token_id=inactive_token_id,
                batch_size=current_batch_size,
                store_baseline=store_baseline,
                baseline_lookup=baseline_lookup,
                primary_readout=primary_readout,
                secondary_readout=secondary_readout,
                baseline_primary_readout=baseline_primary_readout,
                baseline_secondary_readout=baseline_secondary_readout,
                secondary_accumulator=secondary_accumulator,
                mechanism_readout=mechanism_readout,
                mechanism_baseline_lookup=mechanism_baseline_lookup,
                mechanism_baseline_raw_readout=mechanism_baseline_raw_readout,
                mechanism_layer_indices=mechanism_layer_indices,
                gradient_accumulator=gradient_accumulator,
                teacher_tokens_by_prompt=teacher_tokens_by_prompt,
            )
            return runs, current_batch_size
        except RuntimeError as exc:
            if not _is_oom_error(exc) or current_batch_size == 1:
                raise
            torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
        finally:
            capture.close()


def _compute_governance_metrics(run: PipelineRun) -> dict[str, Any]:
    text = run.generated_text or ""
    first_token_str = run.generated_tokens[0].get("token_str", "") if run.generated_tokens else ""
    first_word_match = _WORD_INITIAL_RE.search(text)
    first_word_str = first_word_match.group(0) if first_word_match else ""
    pattern_hits = {name: len(re.findall(pattern, text)) for name, pattern in GOVERNANCE_PATTERNS.items()}
    paragraph_count = text.count("\n\n")
    header_count = len(re.findall(r"(?m)^#{1,6} ", text))
    bullet_count = len(re.findall(r"(?m)^(?:[-*] |\d+\. )", text))
    return {
        "first_token_str": first_token_str,
        "first_word_str": first_word_str,
        "structure_pattern_hits": pattern_hits,
        "paragraph_count": paragraph_count,
        "header_count": header_count,
        "bullet_count": bullet_count,
    }


def _summarize_pipeline(run: PipelineRun) -> dict:
    tier1_ratio = 0.0
    word_categories = classify_generated_tokens_by_word(run.generated_tokens)
    if word_categories:
        tier1_ratio = sum(cat in {"STRUCTURAL", "PUNCTUATION"} for cat in word_categories) / len(word_categories)

    summary = {
        "generated_text_length": len(run.generated_token_ids),
        "structural_token_ratio_tier1_proxy": tier1_ratio,
    }
    summary.update(_compute_governance_metrics(run))
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


def _first_divergence_step(
    teacher_tokens: list[int],
    free_argmax_tokens: list[int],
) -> int | None:
    for idx, (teacher, free) in enumerate(zip(teacher_tokens, free_argmax_tokens, strict=False)):
        if teacher != free:
            return idx
    return None


def _first_divergence_vs_c(
    model_tokens: list[int],
    c_tokens: list[int],
) -> int | None:
    for idx, (m, c) in enumerate(zip(model_tokens, c_tokens, strict=False)):
        if m != c:
            return idx
    if len(model_tokens) != len(c_tokens):
        return min(len(model_tokens), len(c_tokens))
    return None


def _divergence_token_category(
    run: PipelineRun,
    divergence_step: int | None,
) -> str | None:
    if divergence_step is None or divergence_step >= len(run.generated_tokens):
        return None
    categories = classify_generated_tokens_by_word(run.generated_tokens)
    if divergence_step >= len(categories):
        return None
    return categories[divergence_step]


def _pipeline_summary_key(pipeline_name: str) -> str:
    return f"pipeline_{pipeline_name.lower()}"


def _write_jsonl(path: Path, rows: list[dict], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _collapsed_category(raw: str) -> str:
    if raw in {"STRUCTURAL", "DISCOURSE", "PUNCTUATION"}:
        return "FORMAT"
    if raw == "CONTENT":
        return "CONTENT"
    return "FUNCTION_OTHER"


def _teacher_token_category(token_str: str) -> str:
    raw = classify_generated_tokens_by_word([{"token_str": token_str}])[0]
    return raw if isinstance(raw, str) else "OTHER"


def _layer_range_for_final_region(spec) -> list[int]:
    start = max(0, math.floor(spec.n_layers * 0.8))
    return list(range(start, spec.n_layers))


def _late_mechanism_payload(
    *,
    mechanism: MechanismLayerMetrics,
    layer_indices: list[int],
    teacher_token_id: int,
    teacher_token_str: str,
) -> dict[str, Any]:
    teacher_category = _teacher_token_category(teacher_token_str)
    return {
        "layer_indices": list(layer_indices),
        "teacher_token_id": int(teacher_token_id),
        "teacher_token_str": teacher_token_str,
        "teacher_token_category": teacher_category,
        "teacher_token_category_collapsed": _collapsed_category(teacher_category),
        "teacher_token_rank": mechanism.local_teacher_rank,
        "teacher_token_rank_gain": mechanism.local_teacher_rank_gain,
        "current_top1_token_id": mechanism.local_top1_token,
        "current_top1_token_str": mechanism.local_top1_token_str,
        "current_top1_token_category": mechanism.local_top1_category,
        "current_top1_token_category_collapsed": [
            _collapsed_category(cat) for cat in mechanism.local_top1_category
        ],
        "anti_top1_proj": mechanism.anti_top1_proj,
        "anti_top1_cosine": mechanism.anti_top1_cosine,
        "support_teacher_proj": mechanism.support_teacher_proj,
        "support_teacher_cosine": mechanism.support_teacher_cosine,
        "anti_kl_final_proj": mechanism.anti_kl_final_proj,
        "anti_kl_final_cosine": mechanism.anti_kl_final_cosine,
        "orth_remainder_norm": mechanism.orth_remainder_norm,
    }


def _save_gradient_directions_npz(
    path: Path,
    *,
    payload: dict[str, Any],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    counts = payload.get("counts", {})
    for axis_name, layers in payload.items():
        if axis_name == "counts":
            continue
        for layer_key, values in layers.items():
            arrays[f"{axis_name}__{layer_key}"] = np.asarray(values, dtype=np.float32)
    for axis_name, layer_counts in counts.items():
        arrays[f"count__{axis_name}"] = np.asarray(layer_counts, dtype=np.int64)
    np.savez(path, **arrays)


def _build_mechanism_baseline_raw_logits(
    *,
    request_ids: list[str],
    step_indices: list[int],
    baseline_lookup: dict[str, list[list[torch.Tensor]]] | None,
    baseline_raw_readout: ReadoutSpec | None,
    fallback_step_tensors: BatchedStepTensors,
    layer_indices: list[int],
) -> torch.Tensor | None:
    if baseline_lookup is None or baseline_raw_readout is None or not layer_indices:
        return None
    per_layer: list[list[torch.Tensor]] = [[] for _ in layer_indices]
    for row_idx, prompt_id in enumerate(request_ids):
        steps = baseline_lookup.get(prompt_id, [])
        step_idx = step_indices[row_idx]
        if step_idx < len(steps):
            row_residuals = steps[step_idx]
            for late_idx, layer_idx in enumerate(layer_indices):
                per_layer[late_idx].append(row_residuals[layer_idx].float())
        else:
            for late_idx, layer_idx in enumerate(layer_indices):
                per_layer[late_idx].append(fallback_step_tensors.residual_output[layer_idx][row_idx : row_idx + 1].float().detach().cpu())
    stacked = [torch.cat(rows, dim=0).to(next(baseline_raw_readout.final_norm.parameters()).device) for rows in per_layer]
    return raw_logits_from_residuals(baseline_raw_readout, stacked)


def _apply_mechanism_metrics(
    metrics: LayerStepMetrics,
    mechanism: MechanismLayerMetrics,
    *,
    late_layers: list[int],
) -> None:
    n_layers = len(metrics.top1_token)
    index_map = {layer_idx: late_idx for late_idx, layer_idx in enumerate(late_layers)}
    def _fill_float(values: list[float]) -> list[float]:
        out = [0.0] * n_layers
        for layer_idx, late_idx in index_map.items():
            out[layer_idx] = float(values[late_idx])
        return out
    def _fill_int(values: list[int]) -> list[int]:
        out = [0] * n_layers
        for layer_idx, late_idx in index_map.items():
            out[layer_idx] = int(values[late_idx])
        return out
    def _fill_str(values: list[str]) -> list[str]:
        out = [""] * n_layers
        for layer_idx, late_idx in index_map.items():
            out[layer_idx] = str(values[late_idx])
        return out

    metrics.anti_top1_proj = _fill_float(mechanism.anti_top1_proj)
    metrics.anti_top1_cosine = _fill_float(mechanism.anti_top1_cosine)
    metrics.support_teacher_proj = _fill_float(mechanism.support_teacher_proj)
    metrics.support_teacher_cosine = _fill_float(mechanism.support_teacher_cosine)
    metrics.anti_kl_final_proj = _fill_float(mechanism.anti_kl_final_proj)
    metrics.anti_kl_final_cosine = _fill_float(mechanism.anti_kl_final_cosine)
    metrics.orth_remainder_norm = _fill_float(mechanism.orth_remainder_norm)
    metrics.local_teacher_rank = _fill_int(mechanism.local_teacher_rank)
    metrics.local_teacher_rank_gain = _fill_float(mechanism.local_teacher_rank_gain)
    metrics.local_top1_token = _fill_int(mechanism.local_top1_token)
    metrics.local_top1_token_str = _fill_str(mechanism.local_top1_token_str)
    metrics.local_top1_category = _fill_str(mechanism.local_top1_category)


def main() -> None:
    args = parse_args()
    _configure_reproducibility(args.seed)
    if args.depth_ablation and not args.teacher_forced:
        raise ValueError("--depth-ablation requires --teacher-forced")
    if args.causal_combined and not args.teacher_forced:
        raise ValueError("--causal-combined requires --teacher-forced")
    if args.depth_ablation and args.causal_combined:
        raise ValueError("--depth-ablation and --causal-combined are mutually exclusive")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    spec = get_spec(args.model)
    onset_layer = spec.corrective_onset if args.onset_layer is None else args.onset_layer
    depth_ablation = _is_depth_ablation(args)
    causal_combined = _is_causal_combined(args)
    depth_windows = _depth_windows_for_model(args.model, spec) if depth_ablation else {}
    causal_windows = _causal_windows_for_model(args.model, spec) if causal_combined else {}
    late_mechanism_layers = _layer_range_for_final_region(spec) if causal_combined else []
    requested_batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.model]
    out_dir = Path(args.out_dir or f"results/exp11_matched_prefix_mlp_graft/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    dataset = load_dataset(args.dataset)
    if args.prompt_seed is not None:
        pool = dataset
        if args.categories:
            allowed = set(args.categories)
            pool = [r for r in dataset if r.get("category") in allowed]
        prompts = _random_subsample(pool, args.n_prompts, args.prompt_seed)
    else:
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
    pt_model, tokenizer_pt = load_model_and_tokenizer(pt_id, args.device, dtype=dtype)
    it_model, tokenizer_it = load_model_and_tokenizer(it_id, args.device, dtype=dtype)
    _ensure_pad_token(tokenizer_pt)
    _ensure_pad_token(tokenizer_it)

    steering_adapter = get_steering_adapter(args.model)
    model_raw_pt = pt_model
    model_raw_it = it_model
    device = next(model_raw_pt.parameters()).device
    logit_dim_pt = steering_adapter.get_lm_head(model_raw_pt).weight.shape[0]
    logit_dim_it = steering_adapter.get_lm_head(model_raw_it).weight.shape[0]
    if logit_dim_pt != logit_dim_it:
        raise ValueError(
            f"PT/IT vocab mismatch in lm_head dim: {logit_dim_pt} != {logit_dim_it}; exp11 requires aligned vocab dims"
        )
    real_token_mask_pt = steering_adapter.real_token_mask(tokenizer_pt, device, model_raw=model_raw_pt)
    real_token_mask_it = steering_adapter.real_token_mask(tokenizer_it, device, model_raw=model_raw_it)
    structural_masks_pt = build_structural_token_masks(tokenizer_pt, logit_dim_pt, real_token_mask_pt, device)
    structural_masks_it = build_structural_token_masks(tokenizer_it, logit_dim_it, real_token_mask_it, device)
    readouts = _load_readouts(
        args=args,
        spec=spec,
        steering_adapter=steering_adapter,
        model_raw_pt=model_raw_pt,
        model_raw_it=model_raw_it,
        device=device,
    )
    raw_readout_pt = readouts["raw_pt"]
    raw_readout_it = readouts["raw_it"]
    tuned_readout_pt = readouts.get("tuned_pt", raw_readout_pt)
    tuned_readout_it = readouts.get("tuned_it", raw_readout_it)
    eos_ids_pt = set(steering_adapter.eos_token_ids(tokenizer_pt))
    eos_ids_it = set(steering_adapter.eos_token_ids(tokenizer_it))
    eos_ids_pt_or_it = eos_ids_pt | eos_ids_it
    arch_probe = ArchitectureProbe()
    pad_token_id_pt = tokenizer_pt.pad_token_id
    pad_token_id_it = tokenizer_it.pad_token_id
    if pad_token_id_pt is None or pad_token_id_it is None:
        raise ValueError("Tokenizers must have pad tokens after _ensure_pad_token()")
    inactive_token_id_pt = min(eos_ids_pt) if eos_ids_pt else pad_token_id_pt
    inactive_token_id_it = min(eos_ids_it) if eos_ids_it else pad_token_id_it

    prompts_to_run = [record for record in prompts if record["id"] not in done_ids]
    if args.limit_prompts is not None:
        prompts_to_run = prompts_to_run[: args.limit_prompts]

    raw_prompt_text_by_id = {
        record["id"]: get_prompt_for_variant(
            record,
            variant="pt",
            tokenizer=tokenizer_pt,
            apply_chat_template=False,
        )
        for record in prompts_to_run
    }
    raw_prompt_token_ids_pt_by_id = {
        prompt_id: tokenizer_pt(prompt_text, add_special_tokens=True)["input_ids"]
        for prompt_id, prompt_text in raw_prompt_text_by_id.items()
    }
    it_chat_prompt_text_by_id = {
        record["id"]: get_prompt_for_variant(
            record,
            variant="it",
            tokenizer=tokenizer_it,
            apply_chat_template=True,
        )
        for record in prompts_to_run
    }
    it_chat_prompt_token_ids_it_by_id = {
        prompt_id: tokenizer_it(prompt_text, add_special_tokens=True)["input_ids"]
        for prompt_id, prompt_text in it_chat_prompt_text_by_id.items()
    }
    if args.teacher_forced and not depth_ablation and not causal_combined:
        _assert_used_token_ids_compatible(
            tokenizer_it,
            tokenizer_pt,
            it_chat_prompt_token_ids_it_by_id,
            context="IT chat-template prompts on PT backbone",
        )

    chunk_size = args.chunk_size if args.chunk_size else len(prompts_to_run) or 1
    secondary_accumulator = None
    if args.readout_mode == "both" and not causal_combined:
        secondary_accumulator = TrajectoryAccumulator(metric_names=TRAJECTORY_METRICS, n_layers=spec.n_layers)
    mechanism_accumulators: dict[str, GradientDirectionAccumulator] = {}
    if causal_combined:
        for pipeline_name in CAUSAL_MECHANISM_CORE_PIPELINES:
            mechanism_accumulators[pipeline_name] = init_gradient_direction_accumulator(
                axis_names=CAUSAL_MECHANISM_AXES,
                layer_indices=late_mechanism_layers,
                d_model=spec.d_model,
            )

    # Write config and prompts once before chunk loop
    batch_size_a = requested_batch_size
    batch_size_b = requested_batch_size
    batch_size_by_pipeline: dict[str, int] = {}
    pipeline_primary_readouts: dict[str, str] = {}
    pipeline_raw_readouts: dict[str, str] = {}
    if causal_combined:
        pipelines = [
            "C_it_chat",
            "A_prime_raw",
            "B_early_raw",
            "B_mid_raw",
            "B_late_raw",
            "D_early_ptswap",
            "D_mid_ptswap",
            "D_late_ptswap",
        ]
        pipeline_prompt_modes = {
            "C_it_chat": "it_chat_template",
            "A_prime_raw": "raw_format_b",
            "B_early_raw": "raw_format_b",
            "B_mid_raw": "raw_format_b",
            "B_late_raw": "raw_format_b",
            "D_early_ptswap": "it_chat_template",
            "D_mid_ptswap": "it_chat_template",
            "D_late_ptswap": "it_chat_template",
        }
        for name in pipelines:
            if name.startswith("D_") or name == "C_it_chat":
                pipeline_primary_readouts[name] = tuned_readout_it.name
                pipeline_raw_readouts[name] = raw_readout_it.name
            else:
                pipeline_primary_readouts[name] = tuned_readout_pt.name
                pipeline_raw_readouts[name] = raw_readout_pt.name
        pipeline_aliases = None
        secondary_names = None
    elif depth_ablation:
        pipelines = ["C_it_chat", "A_prime_raw", "B_early_raw", "B_mid_raw", "B_late_raw"]
        pipeline_prompt_modes = {
            "C_it_chat": "it_chat_template",
            "A_prime_raw": "raw_format_b",
            "B_early_raw": "raw_format_b",
            "B_mid_raw": "raw_format_b",
            "B_late_raw": "raw_format_b",
        }
        for name in pipelines:
            pipeline_primary_readouts[name] = tuned_readout_pt.name if name != "C_it_chat" else tuned_readout_pt.name
            pipeline_raw_readouts[name] = raw_readout_it.name if name == "C_it_chat" else raw_readout_pt.name
        secondary_names = (
            {
                "C_it_chat": tuned_readout_it.name,
                "A_prime_raw": tuned_readout_pt.name,
                "B_early_raw": tuned_readout_it.name,
                "B_mid_raw": tuned_readout_it.name,
                "B_late_raw": tuned_readout_it.name,
            }
            if args.readout_mode == "both"
            else None
        )
        pipeline_aliases: dict[str, str] | None = None
    elif args.teacher_forced:
        pipelines = ["C", "A_prime", "B", "A_prime_tmpl", "B2"]
        pipeline_prompt_modes = {
            "C": "it_chat_template",
            "A_prime": "raw_format_b",
            "B": "raw_format_b",
            "A_prime_tmpl": "it_chat_template",
            "B2": "it_chat_template",
        }
        for name in pipelines:
            pipeline_primary_readouts[name] = tuned_readout_pt.name
            pipeline_raw_readouts[name] = raw_readout_it.name if name == "C" else raw_readout_pt.name
        secondary_names = (
            {
                "B": tuned_readout_it.name,
                "C": tuned_readout_it.name,
                "A_prime": tuned_readout_pt.name,
                "A_prime_tmpl": tuned_readout_pt.name,
                "B2": tuned_readout_it.name,
            }
            if args.readout_mode == "both"
            else None
        )
        pipeline_aliases = {
            "B": "B1_raw",
            "A_prime": "A_prime_raw",
            "B2": "B2_tmpl",
        }
    else:
        pipelines = ["A", "B"]
        pipeline_prompt_modes = {"A": "raw_format_b", "B": "raw_format_b"}
        for name in pipelines:
            pipeline_primary_readouts[name] = tuned_readout_pt.name
            pipeline_raw_readouts[name] = raw_readout_pt.name
        secondary_names = (
            {
                "A": tuned_readout_pt.name,
                "B": tuned_readout_it.name if args.readout_mode == "both" else tuned_readout_pt.name,
            }
            if args.readout_mode == "both"
            else None
        )
        pipeline_aliases = {}

    config = {
        "model": args.model,
        "n_layers": spec.n_layers,
        "pt_model_id": pt_id,
        "it_model_id": it_id,
        "dataset": args.dataset,
        "n_prompts": args.n_prompts,
        "seed": args.seed,
        "prompt_seed": args.prompt_seed,
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
        "chunk_size": args.chunk_size,
        "n_prompts_sampled_after_sharding": len(prompts),
        "readout_mode": args.readout_mode,
        "primary_readout_name": None if causal_combined else tuned_readout_pt.name,
        "secondary_readout_name": "mixed_by_pipeline" if secondary_names is not None else None,
        "secondary_readout_names_by_pipeline": secondary_names,
        "primary_readout_names_by_pipeline": pipeline_primary_readouts,
        "raw_readout_names_by_pipeline": pipeline_raw_readouts,
        "tuned_lens_dir": args.tuned_lens_dir,
        "teacher_forced": bool(args.teacher_forced),
        "depth_ablation": depth_ablation,
        "causal_combined": causal_combined,
        "pipelines": pipelines,
        "pipeline_prompt_modes": pipeline_prompt_modes,
        "pipeline_aliases": pipeline_aliases,
        "graft_windows_by_pipeline": (
            {
                name: {
                    "start_layer": start,
                    "end_layer_exclusive": end,
                    "display_range": f"{start}-{end - 1}",
                }
                for name, (start, end) in depth_windows.items()
            }
            if depth_ablation
            else {
                name: {
                    "start_layer": start,
                    "end_layer_exclusive": end,
                    "display_range": f"{start}-{end - 1}",
                }
                for name, (start, end) in causal_windows.items()
            }
            if causal_combined
            else None
        ),
        "late_mechanism_layers": late_mechanism_layers if causal_combined else None,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    _write_jsonl(out_dir / "prompts.jsonl", prompts, append=False)

    # Load existing secondary stats before chunk loop (for resume crash safety)
    existing_secondary_at_start: dict = {}
    if secondary_accumulator is not None and args.resume:
        secondary_path = out_dir / "secondary_trajectory_stats.json"
        if secondary_path.exists():
            existing_secondary_at_start = json.loads(secondary_path.read_text())

    total_to_run = len(prompts_to_run)
    n_done = 0
    print(f"[exp11] starting: {total_to_run} prompts, chunk_size={chunk_size}, batch_size={requested_batch_size}", flush=True)

    for chunk_start in range(0, total_to_run, chunk_size):
        chunk = prompts_to_run[chunk_start : chunk_start + chunk_size]

        runs_a: dict[str, PipelineRun] = {}
        runs_b: dict[str, PipelineRun] = {}
        runs_d_early: dict[str, PipelineRun] = {}
        runs_d_mid: dict[str, PipelineRun] = {}
        runs_d_late: dict[str, PipelineRun] = {}
        requests_a: list[RequestState] = []
        requests_b: list[RequestState] = []
        requests_d_early: list[RequestState] = []
        requests_d_mid: list[RequestState] = []
        requests_d_late: list[RequestState] = []
        if not args.teacher_forced:
            requests_a = _make_request_states(
                chunk,
                tokenizer_pt,
                set(),
                prompt_text_by_id=raw_prompt_text_by_id,
                prompt_token_ids_by_id=raw_prompt_token_ids_pt_by_id,
            )
            runs_a, batch_size_a = _run_pipeline_with_batch_fallback(
                pipeline_name="A",
                requests=requests_a,
                model_raw=model_raw_pt,
                capture_factory=lambda: PipelineCapture(
                    model_raw=model_raw_pt,
                    adapter=steering_adapter,
                    arch_probe=arch_probe,
                    onset_layer=onset_layer,
                    graft_it_model_raw=None,
                ),
                tokenizer=tokenizer_pt,
                pad_token_id=pad_token_id_pt,
                real_token_mask=real_token_mask_pt,
                tier1_mask=structural_masks_pt.tier1,
                eos_token_ids=eos_ids_pt,
                max_new_tokens=args.max_new_tokens,
                inactive_token_id=inactive_token_id_pt,
                batch_size=requested_batch_size,
                store_baseline=True,
                baseline_lookup=None,
                primary_readout=tuned_readout_pt,
                secondary_readout=tuned_readout_pt if args.readout_mode == "both" else None,
                baseline_primary_readout=None,
                baseline_secondary_readout=None,
                secondary_accumulator=secondary_accumulator,
            )
            batch_size_by_pipeline["A"] = batch_size_a

        runs_c: dict[str, PipelineRun] = {}
        runs_a_prime: dict[str, PipelineRun] = {}
        runs_b_early: dict[str, PipelineRun] = {}
        runs_b_mid: dict[str, PipelineRun] = {}
        runs_b_late: dict[str, PipelineRun] = {}
        runs_a_prime_tmpl: dict[str, PipelineRun] = {}
        runs_b2: dict[str, PipelineRun] = {}
        requests_c: list[RequestState] = []
        requests_a_prime: list[RequestState] = []
        requests_b_early: list[RequestState] = []
        requests_b_mid: list[RequestState] = []
        requests_b_late: list[RequestState] = []
        requests_a_prime_tmpl: list[RequestState] = []
        requests_b2: list[RequestState] = []
        c_baseline_lookup: dict[str, list[list[torch.Tensor]]] | None = None
        a_prime_baseline_lookup: dict[str, list[list[torch.Tensor]]] | None = None
        teacher_tokens: dict[str, list[int]] | None = None

        if args.teacher_forced:
            requests_c = _make_request_states(
                chunk,
                tokenizer_it,
                set(),
                prompt_text_by_id=it_chat_prompt_text_by_id,
                prompt_token_ids_by_id=it_chat_prompt_token_ids_it_by_id,
            )
            runs_c, batch_size_c = _run_pipeline_with_batch_fallback(
                pipeline_name="C_it_chat" if (depth_ablation or causal_combined) else "C",
                requests=requests_c,
                model_raw=model_raw_it,
                capture_factory=lambda: PipelineCapture(
                    model_raw=model_raw_it,
                    adapter=steering_adapter,
                    arch_probe=arch_probe,
                    onset_layer=onset_layer,
                    graft_it_model_raw=None,
                ),
                tokenizer=tokenizer_it,
                pad_token_id=pad_token_id_it,
                real_token_mask=real_token_mask_it,
                tier1_mask=structural_masks_it.tier1,
                eos_token_ids=eos_ids_it,
                max_new_tokens=args.max_new_tokens,
                inactive_token_id=inactive_token_id_it,
                batch_size=requested_batch_size,
                store_baseline=True,
                baseline_lookup=None,
                primary_readout=tuned_readout_it if causal_combined else tuned_readout_pt,
                secondary_readout=None if causal_combined else (tuned_readout_it if args.readout_mode == "both" else None),
                baseline_primary_readout=None,
                baseline_secondary_readout=None,
                secondary_accumulator=secondary_accumulator,
                mechanism_readout=raw_readout_it if causal_combined and "C_it_chat" in CAUSAL_MECHANISM_CORE_PIPELINES else None,
                mechanism_baseline_lookup=None,
                mechanism_baseline_raw_readout=None,
                mechanism_layer_indices=late_mechanism_layers if causal_combined and "C_it_chat" in CAUSAL_MECHANISM_CORE_PIPELINES else None,
                gradient_accumulator=mechanism_accumulators.get("C_it_chat"),
            )
            batch_size_by_pipeline[
                "C_it_chat" if (depth_ablation or causal_combined) else "C"
            ] = batch_size_c
            c_baseline_lookup = {pid: run.baseline_cache for pid, run in runs_c.items()}
            teacher_tokens = {pid: list(run.generated_token_ids) for pid, run in runs_c.items()}
            _assert_used_token_ids_compatible(
                tokenizer_it,
                tokenizer_pt,
                teacher_tokens,
                context="IT teacher-forced continuation tokens on PT backbone",
            )

            requests_a_prime = _make_request_states(
                chunk,
                tokenizer_pt,
                set(),
                prompt_text_by_id=raw_prompt_text_by_id,
                prompt_token_ids_by_id=raw_prompt_token_ids_pt_by_id,
            )
            runs_a_prime, batch_size_a_prime = _run_pipeline_with_batch_fallback(
                pipeline_name="A_prime_raw" if (depth_ablation or causal_combined) else "A_prime",
                requests=requests_a_prime,
                model_raw=model_raw_pt,
                capture_factory=lambda: PipelineCapture(
                    model_raw=model_raw_pt,
                    adapter=steering_adapter,
                    arch_probe=arch_probe,
                    onset_layer=onset_layer,
                    graft_it_model_raw=None,
                ),
                tokenizer=tokenizer_pt,
                pad_token_id=pad_token_id_pt,
                real_token_mask=real_token_mask_pt,
                tier1_mask=structural_masks_pt.tier1,
                eos_token_ids=eos_ids_pt_or_it,
                max_new_tokens=args.max_new_tokens,
                inactive_token_id=inactive_token_id_pt,
                batch_size=batch_size_a,
                store_baseline=causal_combined,
                baseline_lookup=c_baseline_lookup,
                primary_readout=tuned_readout_pt,
                secondary_readout=None if causal_combined else (tuned_readout_pt if args.readout_mode == "both" else None),
                baseline_primary_readout=tuned_readout_pt,
                baseline_secondary_readout=tuned_readout_it if args.readout_mode == "both" and not causal_combined else None,
                secondary_accumulator=secondary_accumulator,
                mechanism_readout=raw_readout_pt if causal_combined and "A_prime_raw" in CAUSAL_MECHANISM_CORE_PIPELINES else None,
                mechanism_baseline_lookup=None,
                mechanism_baseline_raw_readout=None,
                mechanism_layer_indices=late_mechanism_layers if causal_combined and "A_prime_raw" in CAUSAL_MECHANISM_CORE_PIPELINES else None,
                gradient_accumulator=mechanism_accumulators.get("A_prime_raw"),
                teacher_tokens_by_prompt=teacher_tokens,
            )
            batch_size_by_pipeline[
                "A_prime_raw" if (depth_ablation or causal_combined) else "A_prime"
            ] = batch_size_a_prime
            if causal_combined:
                a_prime_baseline_lookup = {pid: run.baseline_cache for pid, run in runs_a_prime.items()}

            if causal_combined:
                branch_requests_pt = {
                    name: _make_request_states(
                        chunk,
                        tokenizer_pt,
                        set(),
                        prompt_text_by_id=raw_prompt_text_by_id,
                        prompt_token_ids_by_id=raw_prompt_token_ids_pt_by_id,
                    )
                    for name in ("B_early_raw", "B_mid_raw", "B_late_raw")
                }
                branch_requests_it = {
                    name: _make_request_states(
                        chunk,
                        tokenizer_it,
                        set(),
                        prompt_text_by_id=it_chat_prompt_text_by_id,
                        prompt_token_ids_by_id=it_chat_prompt_token_ids_it_by_id,
                    )
                    for name in ("D_early_ptswap", "D_mid_ptswap", "D_late_ptswap")
                }
                branch_runs_pt: dict[str, dict[str, PipelineRun]] = {}
                branch_runs_it: dict[str, dict[str, PipelineRun]] = {}
                for pipeline_name, requests_branch in branch_requests_pt.items():
                    graft_start, graft_end = causal_windows[pipeline_name]
                    branch_runs_pt[pipeline_name], batch_size_by_pipeline[pipeline_name] = _run_pipeline_with_batch_fallback(
                        pipeline_name=pipeline_name,
                        requests=requests_branch,
                        model_raw=model_raw_pt,
                        capture_factory=lambda start=graft_start, end=graft_end: PipelineCapture(
                            model_raw=model_raw_pt,
                            adapter=steering_adapter,
                            arch_probe=arch_probe,
                            graft_start_layer=start,
                            graft_end_layer_exclusive=end,
                            graft_it_model_raw=model_raw_it,
                        ),
                        tokenizer=tokenizer_pt,
                        pad_token_id=pad_token_id_pt,
                        real_token_mask=real_token_mask_pt,
                        tier1_mask=structural_masks_pt.tier1,
                        eos_token_ids=eos_ids_pt_or_it,
                        max_new_tokens=args.max_new_tokens,
                        inactive_token_id=inactive_token_id_pt,
                        batch_size=batch_size_a,
                        store_baseline=False,
                        baseline_lookup=c_baseline_lookup,
                        primary_readout=tuned_readout_pt,
                        secondary_readout=None,
                        baseline_primary_readout=tuned_readout_pt,
                        baseline_secondary_readout=None,
                        secondary_accumulator=None,
                        mechanism_readout=raw_readout_pt if pipeline_name in CAUSAL_MECHANISM_CORE_PIPELINES else None,
                        mechanism_baseline_lookup=a_prime_baseline_lookup,
                        mechanism_baseline_raw_readout=raw_readout_pt,
                        mechanism_layer_indices=late_mechanism_layers if pipeline_name in CAUSAL_MECHANISM_CORE_PIPELINES else None,
                        gradient_accumulator=mechanism_accumulators.get(pipeline_name),
                        teacher_tokens_by_prompt=teacher_tokens,
                    )
                for pipeline_name, requests_branch in branch_requests_it.items():
                    graft_start, graft_end = causal_windows[pipeline_name]
                    branch_runs_it[pipeline_name], batch_size_by_pipeline[pipeline_name] = _run_pipeline_with_batch_fallback(
                        pipeline_name=pipeline_name,
                        requests=requests_branch,
                        model_raw=model_raw_it,
                        capture_factory=lambda start=graft_start, end=graft_end: PipelineCapture(
                            model_raw=model_raw_it,
                            adapter=steering_adapter,
                            arch_probe=arch_probe,
                            graft_start_layer=start,
                            graft_end_layer_exclusive=end,
                            graft_it_model_raw=model_raw_pt,
                        ),
                        tokenizer=tokenizer_it,
                        pad_token_id=pad_token_id_it,
                        real_token_mask=real_token_mask_it,
                        tier1_mask=structural_masks_it.tier1,
                        eos_token_ids=eos_ids_it,
                        max_new_tokens=args.max_new_tokens,
                        inactive_token_id=inactive_token_id_it,
                        batch_size=batch_size_a,
                        store_baseline=False,
                        baseline_lookup=c_baseline_lookup,
                        primary_readout=tuned_readout_it,
                        secondary_readout=None,
                        baseline_primary_readout=tuned_readout_it,
                        baseline_secondary_readout=None,
                        secondary_accumulator=None,
                        mechanism_readout=raw_readout_it if pipeline_name in CAUSAL_MECHANISM_CORE_PIPELINES else None,
                        mechanism_baseline_lookup=c_baseline_lookup,
                        mechanism_baseline_raw_readout=raw_readout_it,
                        mechanism_layer_indices=late_mechanism_layers if pipeline_name in CAUSAL_MECHANISM_CORE_PIPELINES else None,
                        gradient_accumulator=mechanism_accumulators.get(pipeline_name),
                        teacher_tokens_by_prompt=teacher_tokens,
                    )
                runs_b_early = branch_runs_pt["B_early_raw"]
                runs_b_mid = branch_runs_pt["B_mid_raw"]
                runs_b_late = branch_runs_pt["B_late_raw"]
                requests_b_early = branch_requests_pt["B_early_raw"]
                requests_b_mid = branch_requests_pt["B_mid_raw"]
                requests_b_late = branch_requests_pt["B_late_raw"]
                runs_d_early = branch_runs_it["D_early_ptswap"]
                runs_d_mid = branch_runs_it["D_mid_ptswap"]
                runs_d_late = branch_runs_it["D_late_ptswap"]
                requests_d_early = branch_requests_it["D_early_ptswap"]
                requests_d_mid = branch_requests_it["D_mid_ptswap"]
                requests_d_late = branch_requests_it["D_late_ptswap"]
            elif depth_ablation:
                branch_requests = {
                    name: _make_request_states(
                        chunk,
                        tokenizer_pt,
                        set(),
                        prompt_text_by_id=raw_prompt_text_by_id,
                        prompt_token_ids_by_id=raw_prompt_token_ids_pt_by_id,
                    )
                    for name in depth_windows
                }
                branch_runs: dict[str, dict[str, PipelineRun]] = {}
                branch_batch_sizes: dict[str, int] = {}
                for pipeline_name, requests_branch in branch_requests.items():
                    graft_start, graft_end = depth_windows[pipeline_name]
                    branch_runs[pipeline_name], branch_batch_sizes[pipeline_name] = _run_pipeline_with_batch_fallback(
                        pipeline_name=pipeline_name,
                        requests=requests_branch,
                        model_raw=model_raw_pt,
                        capture_factory=lambda start=graft_start, end=graft_end: PipelineCapture(
                            model_raw=model_raw_pt,
                            adapter=steering_adapter,
                            arch_probe=arch_probe,
                            graft_start_layer=start,
                            graft_end_layer_exclusive=end,
                            graft_it_model_raw=model_raw_it,
                        ),
                        tokenizer=tokenizer_pt,
                        pad_token_id=pad_token_id_pt,
                        real_token_mask=real_token_mask_pt,
                        tier1_mask=structural_masks_pt.tier1,
                        eos_token_ids=eos_ids_pt_or_it,
                        max_new_tokens=args.max_new_tokens,
                        inactive_token_id=inactive_token_id_pt,
                        batch_size=batch_size_a,
                        store_baseline=False,
                        baseline_lookup=c_baseline_lookup,
                        primary_readout=tuned_readout_pt,
                        secondary_readout=tuned_readout_it if args.readout_mode == "both" else None,
                        baseline_primary_readout=tuned_readout_pt,
                        baseline_secondary_readout=tuned_readout_it if args.readout_mode == "both" else None,
                        secondary_accumulator=secondary_accumulator,
                        teacher_tokens_by_prompt=teacher_tokens,
                    )
                    batch_size_by_pipeline[pipeline_name] = branch_batch_sizes[pipeline_name]
                runs_b_early = branch_runs["B_early_raw"]
                runs_b_mid = branch_runs["B_mid_raw"]
                runs_b_late = branch_runs["B_late_raw"]
                requests_b_early = branch_requests["B_early_raw"]
                requests_b_mid = branch_requests["B_mid_raw"]
                requests_b_late = branch_requests["B_late_raw"]
            else:
                requests_a_prime_tmpl = _make_request_states(
                    chunk,
                    tokenizer_it,
                    set(),
                    prompt_text_by_id=it_chat_prompt_text_by_id,
                    prompt_token_ids_by_id=it_chat_prompt_token_ids_it_by_id,
                )
                runs_a_prime_tmpl, batch_size_a_prime_tmpl = _run_pipeline_with_batch_fallback(
                    pipeline_name="A_prime_tmpl",
                    requests=requests_a_prime_tmpl,
                    model_raw=model_raw_pt,
                    capture_factory=lambda: PipelineCapture(
                        model_raw=model_raw_pt,
                        adapter=steering_adapter,
                        arch_probe=arch_probe,
                        onset_layer=onset_layer,
                        graft_it_model_raw=None,
                    ),
                    tokenizer=tokenizer_pt,
                    pad_token_id=pad_token_id_pt,
                    real_token_mask=real_token_mask_pt,
                    tier1_mask=structural_masks_pt.tier1,
                    eos_token_ids=eos_ids_pt_or_it,
                    max_new_tokens=args.max_new_tokens,
                    inactive_token_id=inactive_token_id_pt,
                    batch_size=batch_size_a,
                    store_baseline=False,
                    baseline_lookup=c_baseline_lookup,
                    primary_readout=tuned_readout_pt,
                    secondary_readout=tuned_readout_pt if args.readout_mode == "both" else None,
                    baseline_primary_readout=tuned_readout_pt,
                    baseline_secondary_readout=tuned_readout_it if args.readout_mode == "both" else None,
                    secondary_accumulator=secondary_accumulator,
                    teacher_tokens_by_prompt=teacher_tokens,
                )
                batch_size_by_pipeline["A_prime_tmpl"] = batch_size_a_prime_tmpl

                requests_b = _make_request_states(
                    chunk,
                    tokenizer_pt,
                    set(),
                    prompt_text_by_id=raw_prompt_text_by_id,
                    prompt_token_ids_by_id=raw_prompt_token_ids_pt_by_id,
                )
                runs_b, batch_size_b = _run_pipeline_with_batch_fallback(
                    pipeline_name="B",
                    requests=requests_b,
                    model_raw=model_raw_pt,
                    capture_factory=lambda: PipelineCapture(
                        model_raw=model_raw_pt,
                        adapter=steering_adapter,
                        arch_probe=arch_probe,
                        onset_layer=onset_layer,
                        graft_it_model_raw=model_raw_it,
                    ),
                    tokenizer=tokenizer_pt,
                    pad_token_id=pad_token_id_pt,
                    real_token_mask=real_token_mask_pt,
                    tier1_mask=structural_masks_pt.tier1,
                    eos_token_ids=eos_ids_pt_or_it,
                    max_new_tokens=args.max_new_tokens,
                    inactive_token_id=inactive_token_id_pt,
                    batch_size=batch_size_a,
                    store_baseline=False,
                    baseline_lookup=c_baseline_lookup,
                    primary_readout=tuned_readout_pt,
                    secondary_readout=tuned_readout_it if args.readout_mode == "both" else None,
                    baseline_primary_readout=tuned_readout_pt,
                    baseline_secondary_readout=tuned_readout_it if args.readout_mode == "both" else None,
                    secondary_accumulator=secondary_accumulator,
                    teacher_tokens_by_prompt=teacher_tokens,
                )
                batch_size_by_pipeline["B"] = batch_size_b

                requests_b2 = _make_request_states(
                    chunk,
                    tokenizer_it,
                    set(),
                    prompt_text_by_id=it_chat_prompt_text_by_id,
                    prompt_token_ids_by_id=it_chat_prompt_token_ids_it_by_id,
                )
                runs_b2, batch_size_b2 = _run_pipeline_with_batch_fallback(
                    pipeline_name="B2",
                    requests=requests_b2,
                    model_raw=model_raw_pt,
                    capture_factory=lambda: PipelineCapture(
                        model_raw=model_raw_pt,
                        adapter=steering_adapter,
                        arch_probe=arch_probe,
                        onset_layer=onset_layer,
                        graft_it_model_raw=model_raw_it,
                    ),
                    tokenizer=tokenizer_pt,
                    pad_token_id=pad_token_id_pt,
                    real_token_mask=real_token_mask_pt,
                    tier1_mask=structural_masks_pt.tier1,
                    eos_token_ids=eos_ids_pt_or_it,
                    max_new_tokens=args.max_new_tokens,
                    inactive_token_id=inactive_token_id_pt,
                    batch_size=batch_size_a,
                    store_baseline=False,
                    baseline_lookup=c_baseline_lookup,
                    primary_readout=tuned_readout_pt,
                    secondary_readout=tuned_readout_it if args.readout_mode == "both" else None,
                    baseline_primary_readout=tuned_readout_pt,
                    baseline_secondary_readout=tuned_readout_it if args.readout_mode == "both" else None,
                    secondary_accumulator=secondary_accumulator,
                    teacher_tokens_by_prompt=teacher_tokens,
                )
                batch_size_by_pipeline["B2"] = batch_size_b2
        else:
            baseline_lookup_b = {pid: run.baseline_cache for pid, run in runs_a.items()}
            requests_b = _make_request_states(
                chunk,
                tokenizer_pt,
                set(),
                prompt_text_by_id=raw_prompt_text_by_id,
                prompt_token_ids_by_id=raw_prompt_token_ids_pt_by_id,
            )
            runs_b, batch_size_b = _run_pipeline_with_batch_fallback(
                pipeline_name="B",
                requests=requests_b,
                model_raw=model_raw_pt,
                capture_factory=lambda: PipelineCapture(
                    model_raw=model_raw_pt,
                    adapter=steering_adapter,
                    arch_probe=arch_probe,
                    onset_layer=onset_layer,
                    graft_it_model_raw=model_raw_it,
                ),
                tokenizer=tokenizer_pt,
                pad_token_id=pad_token_id_pt,
                real_token_mask=real_token_mask_pt,
                tier1_mask=structural_masks_pt.tier1,
                eos_token_ids=eos_ids_pt,
                max_new_tokens=args.max_new_tokens,
                inactive_token_id=inactive_token_id_pt,
                batch_size=batch_size_a,
                store_baseline=False,
                baseline_lookup=baseline_lookup_b,
                primary_readout=tuned_readout_pt,
                secondary_readout=tuned_readout_it if args.readout_mode == "both" else None,
                baseline_primary_readout=tuned_readout_pt,
                baseline_secondary_readout=tuned_readout_pt if args.readout_mode == "both" else None,
                secondary_accumulator=secondary_accumulator,
                teacher_tokens_by_prompt=None,
            )
            batch_size_by_pipeline["B"] = batch_size_b

        for record in chunk:
            run_a = runs_a.get(record["id"])
            run_b = runs_b.get(record["id"])
            run_c = runs_c.get(record["id"])
            run_a_prime = runs_a_prime.get(record["id"])
            run_b_early = runs_b_early.get(record["id"])
            run_b_mid = runs_b_mid.get(record["id"])
            run_b_late = runs_b_late.get(record["id"])
            run_d_early = runs_d_early.get(record["id"])
            run_d_mid = runs_d_mid.get(record["id"])
            run_d_late = runs_d_late.get(record["id"])
            run_a_prime_tmpl = runs_a_prime_tmpl.get(record["id"])
            run_b2 = runs_b2.get(record["id"])

            summary_row: dict[str, Any] = {
                "prompt_id": record["id"],
                "category": record.get("category", ""),
                "prompt": raw_prompt_text_by_id[record["id"]],
                "prompt_raw": raw_prompt_text_by_id[record["id"]],
                "prompt_it_chat": it_chat_prompt_text_by_id[record["id"]],
            }
            if run_a is not None:
                divergence_step = None
                for idx, (tok_a, tok_b) in enumerate(zip(run_a.generated_token_ids, run_b.generated_token_ids, strict=False)):
                    if tok_a != tok_b:
                        divergence_step = idx
                        break
                if divergence_step is None and len(run_a.generated_token_ids) != len(run_b.generated_token_ids):
                    divergence_step = min(len(run_a.generated_token_ids), len(run_b.generated_token_ids))
                summary_row["divergence_step"] = divergence_step
                summary_row["pipeline_a"] = _summarize_pipeline(run_a)
                summary_row["pipeline_b"] = _summarize_pipeline(run_b)
            if causal_combined and run_c is not None and run_a_prime is not None:
                c_tokens = list(run_c.generated_token_ids)
                summary_row["pipeline_c_it_chat"] = _summarize_pipeline(run_c)
                summary_row["pipeline_a_prime_raw"] = _summarize_pipeline(run_a_prime)
                summary_row["pipeline_b_early_raw"] = _summarize_pipeline(run_b_early)
                summary_row["pipeline_b_mid_raw"] = _summarize_pipeline(run_b_mid)
                summary_row["pipeline_b_late_raw"] = _summarize_pipeline(run_b_late)
                summary_row["pipeline_d_early_ptswap"] = _summarize_pipeline(run_d_early)
                summary_row["pipeline_d_mid_ptswap"] = _summarize_pipeline(run_d_mid)
                summary_row["pipeline_d_late_ptswap"] = _summarize_pipeline(run_d_late)
                summary_row["divergence_step_a_prime_raw_vs_c"] = _first_divergence_step(
                    c_tokens, list(run_a_prime.free_argmax_token_ids)
                )
                for pipeline_name, run_branch in (
                    ("B_early_raw", run_b_early),
                    ("B_mid_raw", run_b_mid),
                    ("B_late_raw", run_b_late),
                    ("D_early_ptswap", run_d_early),
                    ("D_mid_ptswap", run_d_mid),
                    ("D_late_ptswap", run_d_late),
                ):
                    summary_row[f"divergence_step_{pipeline_name.lower()}_vs_c"] = _first_divergence_step(
                        c_tokens, list(run_branch.free_argmax_token_ids)
                    )
            elif depth_ablation and run_c is not None and run_a_prime is not None:
                c_tokens = list(run_c.generated_token_ids)
                summary_row["pipeline_c_it_chat"] = _summarize_pipeline(run_c)
                summary_row["pipeline_a_prime_raw"] = _summarize_pipeline(run_a_prime)
                summary_row["pipeline_b_early_raw"] = _summarize_pipeline(run_b_early)
                summary_row["pipeline_b_mid_raw"] = _summarize_pipeline(run_b_mid)
                summary_row["pipeline_b_late_raw"] = _summarize_pipeline(run_b_late)
                summary_row["divergence_step_a_prime_raw_vs_c"] = _first_divergence_step(
                    c_tokens, list(run_a_prime.free_argmax_token_ids)
                )
                for pipeline_name, run_branch in (
                    ("B_early_raw", run_b_early),
                    ("B_mid_raw", run_b_mid),
                    ("B_late_raw", run_b_late),
                ):
                    summary_row[f"divergence_step_{pipeline_name.lower()}_vs_c"] = _first_divergence_step(
                        c_tokens, list(run_branch.free_argmax_token_ids)
                    )
            elif args.teacher_forced and run_c is not None and run_a_prime is not None:
                c_tokens = list(run_c.generated_token_ids)
                div_b_vs_c = _first_divergence_step(c_tokens, list(run_b.free_argmax_token_ids))
                div_a_prime_vs_c = _first_divergence_step(
                    c_tokens, list(run_a_prime.free_argmax_token_ids)
                )
                div_a_prime_tmpl_vs_c = _first_divergence_step(
                    c_tokens, list(run_a_prime_tmpl.free_argmax_token_ids)
                ) if run_a_prime_tmpl is not None else None
                div_b2_vs_c = _first_divergence_step(
                    c_tokens, list(run_b2.free_argmax_token_ids)
                ) if run_b2 is not None else None
                summary_row["pipeline_c"] = _summarize_pipeline(run_c)
                summary_row["pipeline_a_prime"] = _summarize_pipeline(run_a_prime)
                if run_a_prime_tmpl is not None:
                    summary_row[_pipeline_summary_key("A_prime_tmpl")] = _summarize_pipeline(run_a_prime_tmpl)
                if run_b2 is not None:
                    summary_row[_pipeline_summary_key("B2")] = _summarize_pipeline(run_b2)
                summary_row["divergence_step_b_vs_c"] = div_b_vs_c
                summary_row["divergence_step_a_prime_vs_c"] = div_a_prime_vs_c
                summary_row["divergence_step_a_prime_tmpl_vs_c"] = div_a_prime_tmpl_vs_c
                summary_row["divergence_step_b2_vs_c"] = div_b2_vs_c
                summary_row["first_divergence_token_category_b_vs_c"] = (
                    _divergence_token_category(run_c, div_b_vs_c)
                )
                summary_row["first_divergence_token_category_a_prime_tmpl_vs_c"] = (
                    _divergence_token_category(run_c, div_a_prime_tmpl_vs_c)
                    if div_a_prime_tmpl_vs_c is not None
                    else None
                )
                summary_row["first_divergence_token_category_b2_vs_c"] = (
                    _divergence_token_category(run_c, div_b2_vs_c)
                    if div_b2_vs_c is not None
                    else None
                )

            _write_jsonl(out_dir / "prompt_summaries.jsonl", [summary_row], append=True)

            gen_rows: list[dict] = []
            if run_a is not None:
                gen_rows.append(
                    {
                        "prompt_id": record["id"],
                        "pipeline": "A",
                        "generated_text": run_a.generated_text,
                        "generated_tokens": run_a.generated_tokens,
                        "prompt_mode": "raw_format_b",
                    }
                )
            if not args.teacher_forced:
                gen_rows.append(
                    {
                        "prompt_id": record["id"],
                        "pipeline": "B",
                        "generated_text": run_b.generated_text,
                        "generated_tokens": run_b.generated_tokens,
                        "free_argmax_token_ids": list(run_b.free_argmax_token_ids),
                        "prompt_mode": "raw_format_b",
                    }
                )
            if causal_combined and run_c is not None and run_a_prime is not None:
                for pipeline_name, run_branch, prompt_mode in (
                    ("C_it_chat", run_c, "it_chat_template"),
                    ("A_prime_raw", run_a_prime, "raw_format_b"),
                    ("B_early_raw", run_b_early, "raw_format_b"),
                    ("B_mid_raw", run_b_mid, "raw_format_b"),
                    ("B_late_raw", run_b_late, "raw_format_b"),
                    ("D_early_ptswap", run_d_early, "it_chat_template"),
                    ("D_mid_ptswap", run_d_mid, "it_chat_template"),
                    ("D_late_ptswap", run_d_late, "it_chat_template"),
                ):
                    row = {
                        "prompt_id": record["id"],
                        "pipeline": pipeline_name,
                        "generated_text": run_branch.generated_text,
                        "generated_tokens": run_branch.generated_tokens,
                        "prompt_mode": prompt_mode,
                    }
                    if pipeline_name != "C_it_chat":
                        row["free_argmax_token_ids"] = list(run_branch.free_argmax_token_ids)
                    gen_rows.append(row)
            elif depth_ablation and run_c is not None and run_a_prime is not None:
                for pipeline_name, run_branch, prompt_mode in (
                    ("C_it_chat", run_c, "it_chat_template"),
                    ("A_prime_raw", run_a_prime, "raw_format_b"),
                    ("B_early_raw", run_b_early, "raw_format_b"),
                    ("B_mid_raw", run_b_mid, "raw_format_b"),
                    ("B_late_raw", run_b_late, "raw_format_b"),
                ):
                    row = {
                        "prompt_id": record["id"],
                        "pipeline": pipeline_name,
                        "generated_text": run_branch.generated_text,
                        "generated_tokens": run_branch.generated_tokens,
                        "prompt_mode": prompt_mode,
                    }
                    if pipeline_name != "C_it_chat":
                        row["free_argmax_token_ids"] = list(run_branch.free_argmax_token_ids)
                    gen_rows.append(row)
            elif args.teacher_forced and run_c is not None and run_a_prime is not None:
                gen_rows.append(
                    {
                        "prompt_id": record["id"],
                        "pipeline": "C",
                        "generated_text": run_c.generated_text,
                        "generated_tokens": run_c.generated_tokens,
                        "prompt_mode": "it_chat_template",
                    }
                )
                gen_rows.append(
                    {
                        "prompt_id": record["id"],
                        "pipeline": "A_prime",
                        "generated_text": run_a_prime.generated_text,
                        "generated_tokens": run_a_prime.generated_tokens,
                        "free_argmax_token_ids": list(run_a_prime.free_argmax_token_ids),
                        "prompt_mode": "raw_format_b",
                    }
                )
                if run_a_prime_tmpl is not None:
                    gen_rows.append(
                        {
                            "prompt_id": record["id"],
                            "pipeline": "A_prime_tmpl",
                            "generated_text": run_a_prime_tmpl.generated_text,
                            "generated_tokens": run_a_prime_tmpl.generated_tokens,
                            "free_argmax_token_ids": list(run_a_prime_tmpl.free_argmax_token_ids),
                            "prompt_mode": "it_chat_template",
                        }
                    )
                if run_b2 is not None:
                    gen_rows.append(
                        {
                            "prompt_id": record["id"],
                            "pipeline": "B2",
                            "generated_text": run_b2.generated_text,
                            "generated_tokens": run_b2.generated_tokens,
                            "free_argmax_token_ids": list(run_b2.free_argmax_token_ids),
                            "prompt_mode": "it_chat_template",
                        }
                    )
            _write_jsonl(out_dir / "generated_texts.jsonl", gen_rows, append=True)

            step_rows: list[dict] = []
            pipelines_to_log: list[tuple[str, PipelineRun]] = []
            if run_a is not None:
                pipelines_to_log.append(("A", run_a))
            if causal_combined and run_c is not None and run_a_prime is not None:
                pipelines_to_log.extend(
                    [
                        ("C_it_chat", run_c),
                        ("A_prime_raw", run_a_prime),
                        ("B_early_raw", run_b_early),
                        ("B_mid_raw", run_b_mid),
                        ("B_late_raw", run_b_late),
                        ("D_early_ptswap", run_d_early),
                        ("D_mid_ptswap", run_d_mid),
                        ("D_late_ptswap", run_d_late),
                    ]
                )
            elif depth_ablation and run_c is not None and run_a_prime is not None:
                pipelines_to_log.extend(
                    [
                        ("C_it_chat", run_c),
                        ("A_prime_raw", run_a_prime),
                        ("B_early_raw", run_b_early),
                        ("B_mid_raw", run_b_mid),
                        ("B_late_raw", run_b_late),
                    ]
                )
            else:
                pipelines_to_log.append(("B", run_b))
            if (
                args.teacher_forced
                and not depth_ablation
                and not causal_combined
                and run_c is not None
                and run_a_prime is not None
            ):
                pipelines_to_log.append(("C", run_c))
                pipelines_to_log.append(("A_prime", run_a_prime))
                if run_a_prime_tmpl is not None:
                    pipelines_to_log.append(("A_prime_tmpl", run_a_prime_tmpl))
                if run_b2 is not None:
                    pipelines_to_log.append(("B2", run_b2))
            for pipeline_name, run in pipelines_to_log:
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

            if causal_combined:
                mechanism_rows: list[dict[str, Any]] = []
                for pipeline_name, run in (
                    ("C_it_chat", run_c),
                    ("A_prime_raw", run_a_prime),
                    ("B_late_raw", run_b_late),
                    ("D_late_ptswap", run_d_late),
                ):
                    if run is None:
                        continue
                    for step_idx, step in enumerate(run.step_records):
                        if step.metrics.anti_top1_proj is None:
                            continue
                        mechanism_rows.append(
                            {
                                "prompt_id": record["id"],
                                "pipeline": pipeline_name,
                                "step": step_idx,
                                **_late_mechanism_payload(
                                    mechanism=MechanismLayerMetrics(
                                        anti_top1_proj=[step.metrics.anti_top1_proj[layer] for layer in late_mechanism_layers],
                                        anti_top1_cosine=[step.metrics.anti_top1_cosine[layer] for layer in late_mechanism_layers],
                                        support_teacher_proj=[step.metrics.support_teacher_proj[layer] for layer in late_mechanism_layers],
                                        support_teacher_cosine=[step.metrics.support_teacher_cosine[layer] for layer in late_mechanism_layers],
                                        anti_kl_final_proj=[step.metrics.anti_kl_final_proj[layer] for layer in late_mechanism_layers],
                                        anti_kl_final_cosine=[step.metrics.anti_kl_final_cosine[layer] for layer in late_mechanism_layers],
                                        orth_remainder_norm=[step.metrics.orth_remainder_norm[layer] for layer in late_mechanism_layers],
                                        local_teacher_rank=[step.metrics.local_teacher_rank[layer] for layer in late_mechanism_layers],
                                        local_teacher_rank_gain=[step.metrics.local_teacher_rank_gain[layer] for layer in late_mechanism_layers],
                                        local_top1_token=[step.metrics.local_top1_token[layer] for layer in late_mechanism_layers],
                                        local_top1_token_str=[step.metrics.local_top1_token_str[layer] for layer in late_mechanism_layers],
                                        local_top1_category=[step.metrics.local_top1_category[layer] for layer in late_mechanism_layers],
                                    ),
                                    layer_indices=late_mechanism_layers,
                                    teacher_token_id=step.token_id,
                                    teacher_token_str=step.token_str,
                                ),
                            }
                        )
                _write_jsonl(out_dir / "mechanism_metrics.jsonl", mechanism_rows, append=True)

        # Free baseline cache and generation state before next chunk
        if not args.teacher_forced:
            del runs_b, requests_b
            del runs_a, requests_a
        elif causal_combined:
            del runs_c, runs_a_prime, runs_b_early, runs_b_mid, runs_b_late, runs_d_early, runs_d_mid, runs_d_late
            del requests_c, requests_a_prime, requests_b_early, requests_b_mid, requests_b_late, requests_d_early, requests_d_mid, requests_d_late
            if c_baseline_lookup is not None:
                del c_baseline_lookup
            if a_prime_baseline_lookup is not None:
                del a_prime_baseline_lookup
            if teacher_tokens is not None:
                del teacher_tokens
        elif depth_ablation:
            del runs_c, runs_a_prime, runs_b_early, runs_b_mid, runs_b_late
            del requests_c, requests_a_prime, requests_b_early, requests_b_mid, requests_b_late
            if c_baseline_lookup is not None:
                del c_baseline_lookup
            if teacher_tokens is not None:
                del teacher_tokens
        else:
            del runs_c, runs_a_prime, runs_a_prime_tmpl, runs_b, runs_b2
            del requests_c, requests_a_prime, requests_a_prime_tmpl, requests_b, requests_b2
            if c_baseline_lookup is not None:
                del c_baseline_lookup
            if teacher_tokens is not None:
                del teacher_tokens
        if not args.teacher_forced:
            del baseline_lookup_b
        torch.cuda.empty_cache()

        n_done += len(chunk)
        print(f"[exp11] chunk done: {n_done}/{total_to_run} prompts", flush=True)

        # Write secondary trajectory stats after each chunk for crash safety
        if secondary_accumulator is not None:
            secondary_path = out_dir / "secondary_trajectory_stats.json"
            secondary_readout_names_by_pipeline = config.get("secondary_readout_names_by_pipeline")
            merged_secondary = _merge_trajectory_stats(
                existing_secondary_at_start,
                secondary_accumulator.to_stats_dict(
                    readout_name="mixed_by_pipeline",
                    readout_name_by_pipeline=secondary_readout_names_by_pipeline,
                ),
            )
            secondary_path.write_text(json.dumps(merged_secondary, indent=2))

        if causal_combined:
            gradient_payload = {
                pipeline_name: gradient_direction_payload(acc)
                for pipeline_name, acc in mechanism_accumulators.items()
            }
            (out_dir / "mechanism_gradient_directions.json").write_text(json.dumps(gradient_payload, indent=2))
            for pipeline_name, payload in gradient_payload.items():
                _save_gradient_directions_npz(
                    out_dir / f"mechanism_gradient_directions__{pipeline_name}.npz",
                    payload=payload,
                )

    # Update config with actual batch sizes from the last chunk
    config["batch_size_pipeline_a"] = batch_size_a if not args.teacher_forced else None
    config["batch_size_pipeline_b"] = batch_size_b
    config["batch_size_by_pipeline"] = batch_size_by_pipeline
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
