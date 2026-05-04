"""Shared utilities for Exp46 collectors and analysis."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from src.poc.cross_model.utils import get_raw_prompt, load_model_and_tokenizer
from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import (
    classify_generated_tokens_by_word,
)
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp20_divergence_token_counterfactual.metrics import classify_assistant_marker
from src.poc.exp23_midlate_interaction_suite.residual_factorial import ReadoutBundle
from src.poc.exp46_tulu_fixed_support_stage_sweep import MODEL_NAME, STAGES


def json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def token_payload(tokenizer: Any, token_id: int) -> dict[str, Any]:
    token_str = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    raw_cat = classify_generated_tokens_by_word([{"token_str": token_str}])[0]
    collapsed = (
        "FORMAT"
        if raw_cat in {"STRUCTURAL", "PUNCTUATION", "DISCOURSE"}
        else ("CONTENT" if raw_cat == "CONTENT" else "FUNCTION_OTHER")
    )
    return {
        "token_id": int(token_id),
        "text": token_str,
        "token_str": token_str,
        "token_category": raw_cat,
        "token_category_collapsed": collapsed,
        "assistant_marker": classify_assistant_marker(token_str),
    }


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("id", record.get("record_id", "unknown")))


def stage_key_for_label(label: str) -> str:
    label = label.upper()
    if label not in STAGES:
        raise ValueError(f"Unknown Tulu stage {label!r}; expected {list(STAGES)}")
    return label


def format_prompt(record: dict[str, Any], prompt_mode: str) -> str:
    """Return the exact shared prompt string for Exp46 fixed-support comparisons."""
    raw = get_raw_prompt(record)
    if prompt_mode == "raw_shared":
        return raw
    if prompt_mode == "tulu_shared_template":
        # Manual shared template keeps Base/SFT/DPO/Final token IDs comparable.
        return f"<|user|>\n{raw}\n<|assistant|>\n"
    raise ValueError("prompt_mode must be raw_shared or tulu_shared_template")


def load_stage(stage: str, device: str | torch.device, *, dtype: torch.dtype = torch.bfloat16):
    spec = STAGES[stage_key_for_label(stage)]
    return load_model_and_tokenizer(
        spec.repo_id,
        device,
        dtype=dtype,
        revision=spec.revision,
    )


def stage_adapter():
    return get_steering_adapter(MODEL_NAME)


def real_token_mask_for(model: Any, tokenizer: Any, device: torch.device) -> torch.Tensor:
    return stage_adapter().real_token_mask(tokenizer, device, model)


def make_readout_bundle(stage: str, model: Any, tokenizer: Any, device: torch.device) -> ReadoutBundle:
    adapter = stage_adapter()
    key = stage_key_for_label(stage)
    return ReadoutBundle(
        name=f"common_{key.lower()}",
        variant=key,
        final_norm=adapter.get_final_norm(model),
        lm_head=adapter.get_lm_head(model),
        real_token_mask=adapter.real_token_mask(tokenizer, device, model),
    )


def rank_from_logits(logits: torch.Tensor, token_id: int) -> int | None:
    if token_id < 0 or token_id >= logits.shape[-1]:
        return None
    return int((logits > logits[token_id]).sum().item()) + 1


def validate_full_prefix_boundary(tensor: torch.Tensor, *, expected_seq_len: int | None = None) -> None:
    if not torch.is_tensor(tensor) or tensor.ndim != 3:
        raise RuntimeError(
            "Exp46 boundary states must be full-prefix [batch, seq, d_model] tensors; "
            f"got {type(tensor)} shape={getattr(tensor, 'shape', None)}"
        )
    if expected_seq_len is not None and int(tensor.shape[1]) != int(expected_seq_len):
        raise RuntimeError(
            "Exp46 boundary sequence-length mismatch: "
            f"state={int(tensor.shape[1])} expected={int(expected_seq_len)}"
        )


def cell_name(upstream_stage: str, late_stage: str) -> str:
    return f"U_{stage_key_for_label(upstream_stage)}__L_{stage_key_for_label(late_stage)}"
