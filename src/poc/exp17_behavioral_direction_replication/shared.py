from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.poc.cross_model.adapters import ModelAdapter, get_adapter
from src.poc.cross_model.config import MODEL_REGISTRY, ModelSpec, get_spec, model_id_for_variant
from src.poc.cross_model.utils import load_model_and_tokenizer


log = logging.getLogger(__name__)

EXP17_RESULTS_ROOT = Path("results/exp17_behavioral_direction_replication")
VALID_MODELS = list(MODEL_REGISTRY.keys())
VALID_VARIANTS = ("pt", "it")


@dataclass(frozen=True)
class LoadedExp17Model:
    model_name: str
    variant: str
    spec: ModelSpec
    model_id: str
    model: Any
    tokenizer: Any
    adapter: ModelAdapter
    runtime_device: torch.device


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported dataset format for {path}; expected .jsonl or .csv")


def normalize_vector(vector: torch.Tensor) -> tuple[torch.Tensor, float]:
    norm = float(vector.norm().item())
    if norm == 0.0:
        return vector.clone(), norm
    return vector / norm, norm


def result_dir(component: str, model_name: str, variant: str) -> Path:
    return EXP17_RESULTS_ROOT / component / model_name / variant


def parse_binary_label(value: Any, *, positive_values: set[str], negative_values: set[str]) -> str:
    if isinstance(value, bool):
        return "positive" if value else "negative"
    if isinstance(value, int):
        if value == 1:
            return "positive"
        if value == 0:
            return "negative"
    text = str(value).strip().lower()
    if text in positive_values:
        return "positive"
    if text in negative_values:
        return "negative"
    raise ValueError(f"Unrecognized binary label: {value!r}")


def _ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def load_exp17_model(
    model_name: str,
    *,
    variant: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> LoadedExp17Model:
    spec = get_spec(model_name)
    model_id = model_id_for_variant(spec, variant)
    log.info(
        "exp17 loading model=%s variant=%s model_id=%s device=%s dtype=%s",
        model_name,
        variant,
        model_id,
        device,
        dtype,
    )
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        device=device,
        dtype=dtype,
        multi_gpu=spec.multi_gpu,
    )
    _ensure_pad_token(tokenizer)
    adapter = get_adapter(model_name)
    runtime_device = next(model.parameters()).device
    log.info(
        "exp17 loaded model=%s variant=%s runtime_device=%s layers=%d d_model=%d",
        model_name,
        variant,
        runtime_device,
        spec.n_layers,
        spec.d_model,
    )
    return LoadedExp17Model(
        model_name=model_name,
        variant=variant,
        spec=spec,
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        runtime_device=runtime_device,
    )


def build_prompts(
    rows: list[dict[str, Any]],
    *,
    text_field: str,
    loaded: LoadedExp17Model,
    apply_chat_template: bool,
    prefix: str = "",
    suffix: str = "",
) -> list[str]:
    log.info(
        "exp17 building prompts rows=%d text_field=%s apply_chat_template=%s prefix_len=%d suffix_len=%d",
        len(rows),
        text_field,
        apply_chat_template,
        len(prefix),
        len(suffix),
    )
    prompts: list[str] = []
    for row in rows:
        if text_field not in row:
            raise KeyError(f"Missing text field {text_field!r} in row keys={sorted(row.keys())}")
        text = f"{prefix}{row[text_field]}{suffix}"
        if loaded.variant == "it" and apply_chat_template:
            text = loaded.adapter.apply_template(loaded.tokenizer, text, is_it=True)
        prompts.append(text)
    log.info("exp17 built prompts rows=%d", len(prompts))
    return prompts


def accumulate_class_sums(
    loaded: LoadedExp17Model,
    *,
    prompts: list[str],
    labels: list[str],
    class_names: tuple[str, ...],
    batch_size: int,
    max_length: int,
    progress_desc: str,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, int]]:
    if len(prompts) != len(labels):
        raise ValueError(f"Prompt/label length mismatch: {len(prompts)} != {len(labels)}")

    n_layers = loaded.spec.n_layers
    d_model = loaded.spec.d_model
    sums = {
        class_name: {
            layer_idx: torch.zeros(d_model, dtype=torch.float64)
            for layer_idx in range(n_layers)
        }
        for class_name in class_names
    }
    counts = {class_name: 0 for class_name in class_names}

    layer_modules = loaded.adapter.layers(loaded.model)
    current_last_indices: torch.Tensor | None = None
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = loaded.adapter.residual_from_output(output)
            if current_last_indices is None:
                raise RuntimeError("current_last_indices not set before forward pass")
            row_idx = torch.arange(hidden.shape[0], device=hidden.device)
            captured[layer_idx] = hidden[row_idx, current_last_indices].detach().float().cpu()
        return hook

    handles = [
        layer_modules[layer_idx].register_forward_hook(make_hook(layer_idx))
        for layer_idx in range(n_layers)
    ]

    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc=progress_desc):
            batch_prompts = prompts[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]
            encoded = loaded.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(loaded.runtime_device)
            attention_mask = encoded["attention_mask"].to(loaded.runtime_device)
            current_last_indices = (attention_mask.sum(dim=1) - 1).to(torch.long)
            captured.clear()
            with torch.no_grad():
                loaded.model(input_ids=input_ids, attention_mask=attention_mask)

            for class_name in class_names:
                selected = [idx for idx, label in enumerate(batch_labels) if label == class_name]
                if not selected:
                    continue
                counts[class_name] += len(selected)
                selected_idx = torch.tensor(selected, dtype=torch.long)
                for layer_idx in range(n_layers):
                    sums[class_name][layer_idx] += captured[layer_idx][selected_idx].sum(dim=0, dtype=torch.float64)
    finally:
        for handle in handles:
            handle.remove()

    return sums, counts
