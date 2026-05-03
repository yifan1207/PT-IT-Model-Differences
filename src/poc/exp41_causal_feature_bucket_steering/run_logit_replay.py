"""Run Exp41 first-divergence logit replay with bucket edits."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp41_causal_feature_bucket_steering.bucket_hooks import (
    ActivationMode,
    BucketMlpEditor,
    ControlMode,
    stable_seed,
)
from src.poc.exp41_causal_feature_bucket_steering.config import (
    DATASET_PATH,
    DEFAULT_ALPHAS_SMOKE,
    EXP20_ROOT,
    MAIN_SMOKE_BUCKETS,
    PRIMARY_MODELS,
    TERMINAL_LAYERS,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class ReplayEvent:
    prompt_id: str
    prompt_category: str
    record: dict[str, Any]
    prefix_ids: list[int]
    step: int
    y_pt: int
    y_it: int
    pt_token: dict[str, Any]
    it_token: dict[str, Any]


@dataclass(frozen=True)
class CellSpec:
    name: str
    host_variant: str
    donor_variant: str | None


@dataclass(frozen=True)
class EditSpec:
    name: str
    source_bucket: str
    condition_kind: str
    control_mode: ControlMode
    rows: tuple[dict[str, str], ...]


CELLS = {
    "pt_pt": CellSpec("pt_pt", "pt", None),
    "pt_it": CellSpec("pt_it", "pt", "it"),
    "it_pt": CellSpec("it_pt", "it", "pt"),
    "it_it": CellSpec("it_it", "it", None),
}


class TerminalMlpGraft:
    """Swap donor MLP outputs into a host model for the terminal crosscoded layers."""

    def __init__(
        self,
        *,
        host_layers: list[Any],
        donor_layers: list[Any],
        target_layers: tuple[int, ...],
    ) -> None:
        self.host_layers = host_layers
        self.donor_layers = donor_layers
        self.target_layers = tuple(int(x) for x in target_layers)
        self.handles: list[Any] = []

    def __enter__(self) -> "TerminalMlpGraft":
        for layer_idx in self.target_layers:
            host_mlp = self.host_layers[layer_idx].mlp
            donor_mlp = self.donor_layers[layer_idx].mlp

            def hook(_module, args, _output, dm=donor_mlp):
                with torch.no_grad():
                    return dm(args[0])

            self.handles.append(host_mlp.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_dataset_map(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in _read_jsonl(path):
        out[str(row.get("id", row.get("record_id")))] = row
    return out


def _load_events(
    *,
    exp20_root: Path,
    model: str,
    prompt_mode: str,
    dataset_map: dict[str, dict[str, Any]],
    event_kind: str,
) -> list[ReplayEvent]:
    path = exp20_root / prompt_mode / model / "exp20_validation_records.jsonl"
    events: list[ReplayEvent] = []
    for payload in _read_jsonl(path):
        prompt_id = str(payload.get("prompt_id"))
        record = dataset_map.get(prompt_id)
        if record is None:
            log.warning("Skipping %s: missing dataset record", prompt_id)
            continue
        readout = (payload.get("readouts") or {}).get(event_kind)
        if not isinstance(readout, dict):
            continue
        event = readout.get("event")
        if not isinstance(event, dict):
            continue
        step = event.get("step")
        pt_token = event.get("pt_token")
        it_token = event.get("it_token")
        if not isinstance(step, int) or not isinstance(pt_token, dict) or not isinstance(it_token, dict):
            continue
        free_pt = ((payload.get("free_runs") or {}).get("A_pt_raw") or {}).get("generated_token_ids")
        if not isinstance(free_pt, list):
            continue
        events.append(
            ReplayEvent(
                prompt_id=prompt_id,
                prompt_category=str(record.get("category", "unknown")),
                record=record,
                prefix_ids=[int(x) for x in free_pt[:step]],
                step=int(step),
                y_pt=int(pt_token["token_id"]),
                y_it=int(it_token["token_id"]),
                pt_token=pt_token,
                it_token=it_token,
            )
        )
    return events


def _done_prompt_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line).get("prompt_id")))
            except json.JSONDecodeError:
                continue
    return done


def _token_bucket(text: str) -> str:
    if not text:
        return "other"
    stripped = text.strip()
    low = stripped.lower()
    if "\n" in text or stripped == "":
        return "newline_or_blankline"
    if any(ord(ch) > 127 for ch in text) and not re.search(r"[A-Za-z0-9]", text):
        return "rare_unicode_or_artifact"
    if stripped in {"-", "*", "•", "1.", "2.", "3.", "I.", "II.", "III."}:
        return "list_or_bullet_marker"
    if stripped in {":", "Answer:", "Question:", "Response:", "A:", "B:", "C:", "D:"}:
        return "field_label_or_colon"
    if re.fullmatch(r"[\(\[]?[A-Ha-h][\)\].:]?", stripped):
        return "mcq_option_marker"
    if any(ch in text for ch in ['"', "'", "(", ")", "[", "]", "{", "}"]):
        return "quote_or_parenthesis"
    if any(marker in low for marker in ("sorry", "can't", "cannot", "not able", "safe", "unsafe", "harm")):
        return "refusal_or_safety_phrase"
    if any(ch in text for ch in ("`", "<", ">", "/", "\\")):
        return "code_or_tool_syntax"
    if stripped and all(ch in ".,;:!?" for ch in stripped):
        return "punctuation_boundary"
    if stripped.startswith("▁") or stripped.startswith("Ġ") or len(stripped) <= 2 and not stripped.isalnum():
        return "surface_subword"
    if re.search(r"[A-Za-z0-9]", stripped):
        return "ordinary_content_word"
    return "other"


def _build_batch(
    *,
    events: list[ReplayEvent],
    tokenizer: Any,
    host_variant: str,
    prompt_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded: list[list[int]] = []
    for event in events:
        prompt = get_prompt_for_variant(
            event.record,
            variant=host_variant,
            tokenizer=tokenizer,
            apply_chat_template=(prompt_mode == "native" and host_variant == "it"),
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        encoded.append([int(x) for x in prompt_ids] + event.prefix_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    max_len = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    positions = torch.empty((len(encoded),), dtype=torch.long, device=device)
    for idx, ids in enumerate(encoded):
        n = len(ids)
        input_ids[idx, :n] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[idx, :n] = 1
        positions[idx] = n - 1
    return input_ids, attention_mask, positions


def _layer_latents(rows: tuple[dict[str, str], ...]) -> tuple[dict[int, list[int]], dict[int, Path], int]:
    by_layer: dict[int, list[int]] = {}
    roots: dict[int, Path] = {}
    count = 0
    for row in rows:
        layer = int(float(row["layer"]))
        latent = int(float(row["latent_id"]))
        by_layer.setdefault(layer, []).append(latent)
        roots.setdefault(layer, Path(row["result_root"]))
        count += 1
    return by_layer, roots, count


def _forward_cell(
    *,
    model_name: str,
    cell: CellSpec,
    events: list[ReplayEvent],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    layers_by_variant: dict[str, list[Any]],
    device: torch.device,
    prompt_mode: str,
    edit_spec: EditSpec | None,
    alpha: float,
    activation_mode: ActivationMode,
    crosscoder_dtype: torch.dtype,
    crosscoder_cache: dict[tuple[str, int], Any],
) -> dict[str, Any]:
    model = models[cell.host_variant]
    tokenizer = tokenizers[cell.host_variant]
    input_ids, attention_mask, positions = _build_batch(
        events=events,
        tokenizer=tokenizer,
        host_variant=cell.host_variant,
        prompt_mode=prompt_mode,
        device=device,
    )

    graft_ctx = None
    editor_ctx = None
    editor = None
    try:
        if cell.donor_variant is not None:
            graft_ctx = TerminalMlpGraft(
                host_layers=layers_by_variant[cell.host_variant],
                donor_layers=layers_by_variant[cell.donor_variant],
                target_layers=TERMINAL_LAYERS[model_name],
            )
            graft_ctx.__enter__()
        if edit_spec is not None and alpha != 0.0:
            layer_latents, layer_roots, _ = _layer_latents(edit_spec.rows)
            editor = BucketMlpEditor(
                model=model,
                layers=layers_by_variant[cell.host_variant],
                layer_latents=layer_latents,
                layer_roots=layer_roots,
                alpha=float(alpha),
                device=device,
                activation_mode=activation_mode,
                control_mode=edit_spec.control_mode,
                seed=stable_seed(model_name, edit_spec.name, alpha),
                crosscoder_dtype=crosscoder_dtype,
                target_positions=positions,
                crosscoder_cache=crosscoder_cache,
            )
            editor_ctx = editor
            editor_ctx.__enter__()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[torch.arange(input_ids.shape[0], device=device), positions, :].float()
    finally:
        if editor_ctx is not None:
            editor_ctx.__exit__(None, None, None)
        if graft_ctx is not None:
            graft_ctx.__exit__(None, None, None)

    mask = real_token_masks[cell.host_variant].to(device=logits.device)
    masked = logits.clone()
    masked[:, ~mask] = float("-inf")
    y_pt_logits: list[float] = []
    y_it_logits: list[float] = []
    margins: list[float] = []
    nll_it: list[float] = []
    nll_pt: list[float] = []
    argmax_ids: list[int] = []
    argmax_texts: list[str] = []
    log_probs = torch.log_softmax(masked, dim=-1)
    for idx, event in enumerate(events):
        y_pt = event.y_pt
        y_it = event.y_it
        if y_pt >= masked.shape[-1] or y_it >= masked.shape[-1]:
            y_pt_logits.append(float("nan"))
            y_it_logits.append(float("nan"))
            margins.append(float("nan"))
            nll_it.append(float("nan"))
            nll_pt.append(float("nan"))
        else:
            pt_val = float(masked[idx, y_pt].item())
            it_val = float(masked[idx, y_it].item())
            y_pt_logits.append(pt_val)
            y_it_logits.append(it_val)
            margins.append(it_val - pt_val)
            nll_it.append(float(-log_probs[idx, y_it].item()))
            nll_pt.append(float(-log_probs[idx, y_pt].item()))
        argmax_id = int(masked[idx].argmax(dim=-1).item())
        argmax_ids.append(argmax_id)
        argmax_texts.append(tokenizer.decode([argmax_id], skip_special_tokens=False, clean_up_tokenization_spaces=False))

    return {
        "cell": cell.name,
        "host_variant": cell.host_variant,
        "donor_variant": cell.donor_variant,
        "y_pt_logit": y_pt_logits,
        "y_it_logit": y_it_logits,
        "margin": margins,
        "nll_it": nll_it,
        "nll_pt": nll_pt,
        "argmax_token_id": argmax_ids,
        "argmax_token_text": argmax_texts,
        "diagnostics": editor.summary() if editor is not None else {},
    }


def _load_edit_specs(
    *,
    manifest_dir: Path,
    model: str,
    buckets: set[str],
    include_matched_random: bool,
    include_same_delta: bool,
) -> list[EditSpec]:
    feature_rows = [
        row
        for row in _read_csv(manifest_dir / "bucket_features.csv")
        if row.get("model") == model and row.get("bucket") in buckets
    ]
    control_rows = [
        row
        for row in _read_csv(manifest_dir / "bucket_controls.csv")
        if row.get("model") == model and row.get("source_bucket") in buckets
    ]
    by_bucket: dict[str, list[dict[str, str]]] = {}
    for row in feature_rows:
        by_bucket.setdefault(str(row["bucket"]), []).append(row)
    controls_by_bucket: dict[str, list[dict[str, str]]] = {}
    for row in control_rows:
        controls_by_bucket.setdefault(str(row["source_bucket"]), []).append(row)

    specs: list[EditSpec] = []
    for bucket in sorted(by_bucket):
        rows = tuple(sorted(by_bucket[bucket], key=lambda row: (int(float(row["layer"])), int(float(row["latent_id"])))))
        specs.append(
            EditSpec(
                name=bucket,
                source_bucket=bucket,
                condition_kind="feature_bucket",
                control_mode="feature",
                rows=rows,
            )
        )
        if include_matched_random:
            matched = tuple(
                sorted(
                    controls_by_bucket.get(bucket, []),
                    key=lambda row: (int(float(row["layer"])), int(float(row["latent_id"]))),
                )
            )
            if matched:
                specs.append(
                    EditSpec(
                        name=f"{bucket}__matched_random",
                        source_bucket=bucket,
                        condition_kind="matched_random",
                        control_mode="feature",
                        rows=matched,
                    )
                )
        if include_same_delta:
            specs.append(
                EditSpec(
                    name=f"{bucket}__same_delta_random",
                    source_bucket=bucket,
                    condition_kind="same_delta_random",
                    control_mode="same_delta_random",
                    rows=rows,
                )
            )
    return specs


def _nan_sub(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return a - b


def _interaction(m_pt_pt: float, m_pt_it: float, m_it_pt: float, m_it_it: float) -> float:
    if any(math.isnan(x) for x in (m_pt_pt, m_pt_it, m_it_pt, m_it_it)):
        return float("nan")
    return (m_it_it - m_it_pt) - (m_pt_it - m_pt_pt)


def _write_batch_rows(
    *,
    fout: Any,
    model_name: str,
    event_kind: str,
    prompt_mode: str,
    manifest_mode: str,
    activation_mode: str,
    events: list[ReplayEvent],
    edit_spec: EditSpec,
    alpha: float,
    baseline: dict[str, dict[str, Any]],
    edited_pt_it: dict[str, Any],
    edited_it_it: dict[str, Any],
) -> None:
    _, _, n_latents = _layer_latents(edit_spec.rows)
    pt_it_diag = edited_pt_it.get("diagnostics") or {}
    it_it_diag = edited_it_it.get("diagnostics") or {}
    for idx, event in enumerate(events):
        m_pt_pt = baseline["pt_pt"]["margin"][idx]
        m_pt_it = baseline["pt_it"]["margin"][idx]
        m_it_pt = baseline["it_pt"]["margin"][idx]
        m_it_it = baseline["it_it"]["margin"][idx]
        m_pt_it_edit = edited_pt_it["margin"][idx]
        m_it_it_edit = edited_it_it["margin"][idx]
        interaction_full = _interaction(m_pt_pt, m_pt_it, m_it_pt, m_it_it)
        interaction_after = _interaction(m_pt_pt, m_pt_it_edit, m_it_pt, m_it_it_edit)
        row = {
            "model": model_name,
            "prompt_id": event.prompt_id,
            "prompt_category": event.prompt_category,
            "prompt_mode": prompt_mode,
            "event_kind": event_kind,
            "step": event.step,
            "manifest_mode": manifest_mode,
            "activation_mode": activation_mode,
            "source_bucket": edit_spec.source_bucket,
            "edit_name": edit_spec.name,
            "condition_kind": edit_spec.condition_kind,
            "control_mode": edit_spec.control_mode,
            "alpha": float(alpha),
            "n_latents": n_latents,
            "terminal_layers": list(TERMINAL_LAYERS[model_name]),
            "y_pt_token_id": event.y_pt,
            "y_it_token_id": event.y_it,
            "y_pt_token_text": event.pt_token.get("token_str"),
            "y_it_token_text": event.it_token.get("token_str"),
            "pt_token_bucket": _token_bucket(str(event.pt_token.get("token_str", ""))),
            "it_token_bucket": _token_bucket(str(event.it_token.get("token_str", ""))),
            "baseline_pt_pt_margin": m_pt_pt,
            "baseline_pt_it_margin": m_pt_it,
            "baseline_it_pt_margin": m_it_pt,
            "baseline_it_it_margin": m_it_it,
            "edited_pt_it_margin": m_pt_it_edit,
            "edited_it_it_margin": m_it_it_edit,
            "drop_native": _nan_sub(m_it_it, m_it_it_edit),
            "drop_pt_upstream": _nan_sub(m_pt_it, m_pt_it_edit),
            "interaction_full": interaction_full,
            "interaction_after": interaction_after,
            "interaction_drop": _nan_sub(interaction_full, interaction_after),
            "baseline_it_it_nll_it": baseline["it_it"]["nll_it"][idx],
            "edited_it_it_nll_it": edited_it_it["nll_it"][idx],
            "native_nll_it_shift": _nan_sub(edited_it_it["nll_it"][idx], baseline["it_it"]["nll_it"][idx]),
            "edited_it_it_argmax_token_id": edited_it_it["argmax_token_id"][idx],
            "edited_it_it_argmax_token_text": edited_it_it["argmax_token_text"][idx],
            "edited_it_it_argmax_bucket": _token_bucket(str(edited_it_it["argmax_token_text"][idx])),
            "pt_it_mean_delta_norm_frac": pt_it_diag.get("mean_delta_norm_frac", 0.0),
            "it_it_mean_delta_norm_frac": it_it_diag.get("mean_delta_norm_frac", 0.0),
            "pt_it_max_delta_norm_frac": pt_it_diag.get("max_delta_norm_frac", 0.0),
            "it_it_max_delta_norm_frac": it_it_diag.get("max_delta_norm_frac", 0.0),
        }
        fout.write(json.dumps(row) + "\n")


def run_model(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    model_name = args.model
    spec = get_spec(model_name)
    steering_adapter = get_steering_adapter(model_name)
    dtype = _dtype_from_name(args.dtype)
    crosscoder_dtype = _dtype_from_name(args.crosscoder_dtype)

    log.info("[exp41] loading %s PT/IT on %s", model_name, device)
    pt_model, pt_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "pt"),
        args.device,
        dtype=dtype,
        multi_gpu=False,
    )
    it_model, it_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "it"),
        args.device,
        dtype=dtype,
        multi_gpu=False,
    )
    models = {"pt": pt_model, "it": it_model}
    tokenizers = {"pt": pt_tokenizer, "it": it_tokenizer}
    layers_by_variant = {
        "pt": steering_adapter.get_layers(pt_model),
        "it": steering_adapter.get_layers(it_model),
    }
    real_token_masks = {
        "pt": steering_adapter.real_token_mask(pt_tokenizer, device, pt_model),
        "it": steering_adapter.real_token_mask(it_tokenizer, device, it_model),
    }

    dataset_map = _load_dataset_map(args.dataset)
    events = _load_events(
        exp20_root=args.exp20_root,
        model=model_name,
        prompt_mode=args.prompt_mode,
        dataset_map=dataset_map,
        event_kind=args.event_kind,
    )
    events = events[: args.n_events] if args.n_events is not None else events
    events = events[args.worker_index :: args.n_workers]

    out_dir = args.out_dir / "logit_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"logit_records_{model_name}_w{args.worker_index}.jsonl"
    done = _done_prompt_ids(out_path)
    events = [event for event in events if event.prompt_id not in done]

    buckets = set(args.buckets)
    edit_specs = _load_edit_specs(
        manifest_dir=args.manifest_dir / args.manifest_mode,
        model=model_name,
        buckets=buckets,
        include_matched_random=not args.no_matched_random,
        include_same_delta=not args.no_same_delta,
    )
    if not edit_specs:
        raise RuntimeError(f"No edit specs for model={model_name} buckets={sorted(buckets)}")

    alphas = [float(x) for x in args.alphas]
    crosscoder_cache: dict[tuple[str, int], Any] = {}
    log.info(
        "[exp41] %s worker %d/%d: %d events, %d edit specs, alphas=%s, output=%s",
        model_name,
        args.worker_index,
        args.n_workers,
        len(events),
        len(edit_specs),
        alphas,
        out_path,
    )

    with out_path.open("a") as fout:
        for start in range(0, len(events), args.batch_size):
            batch = events[start : start + args.batch_size]
            baseline = {
                name: _forward_cell(
                    model_name=model_name,
                    cell=cell,
                    events=batch,
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    layers_by_variant=layers_by_variant,
                    device=device,
                    prompt_mode=args.prompt_mode,
                    edit_spec=None,
                    alpha=0.0,
                    activation_mode=args.activation_mode,
                    crosscoder_dtype=crosscoder_dtype,
                    crosscoder_cache=crosscoder_cache,
                )
                for name, cell in CELLS.items()
            }
            for edit_spec in edit_specs:
                for alpha in alphas:
                    if alpha == 0.0:
                        edited_pt_it = baseline["pt_it"]
                        edited_it_it = baseline["it_it"]
                    else:
                        edited_pt_it = _forward_cell(
                            model_name=model_name,
                            cell=CELLS["pt_it"],
                            events=batch,
                            models=models,
                            tokenizers=tokenizers,
                            real_token_masks=real_token_masks,
                            layers_by_variant=layers_by_variant,
                            device=device,
                            prompt_mode=args.prompt_mode,
                            edit_spec=edit_spec,
                            alpha=alpha,
                            activation_mode=args.activation_mode,
                            crosscoder_dtype=crosscoder_dtype,
                            crosscoder_cache=crosscoder_cache,
                        )
                        edited_it_it = _forward_cell(
                            model_name=model_name,
                            cell=CELLS["it_it"],
                            events=batch,
                            models=models,
                            tokenizers=tokenizers,
                            real_token_masks=real_token_masks,
                            layers_by_variant=layers_by_variant,
                            device=device,
                            prompt_mode=args.prompt_mode,
                            edit_spec=edit_spec,
                            alpha=alpha,
                            activation_mode=args.activation_mode,
                            crosscoder_dtype=crosscoder_dtype,
                            crosscoder_cache=crosscoder_cache,
                        )
                    _write_batch_rows(
                        fout=fout,
                        model_name=model_name,
                        event_kind=args.event_kind,
                        prompt_mode=args.prompt_mode,
                        manifest_mode=args.manifest_mode,
                        activation_mode=args.activation_mode,
                        events=batch,
                        edit_spec=edit_spec,
                        alpha=alpha,
                        baseline=baseline,
                        edited_pt_it=edited_pt_it,
                        edited_it_it=edited_it_it,
                    )
                    fout.flush()
            done_text = f"{min(start + len(batch), len(events))}/{len(events)}"
            if (start // args.batch_size + 1) % max(args.log_every_batches, 1) == 0:
                log.info("[exp41] %s worker %d progress %s", model_name, args.worker_index, done_text)

    summary = {
        "model": model_name,
        "worker_index": args.worker_index,
        "n_workers": args.n_workers,
        "n_events_remaining_at_start": len(events),
        "edit_specs": [spec.name for spec in edit_specs],
        "alphas": alphas,
        "out_path": str(out_path),
    }
    _write_json(out_dir / f"logit_replay_summary_{model_name}_w{args.worker_index}.json", summary)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--manifest-mode", default="strict_primary")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--exp20-root", type=Path, default=EXP20_ROOT)
    parser.add_argument("--prompt-mode", default="raw_shared", choices=["raw_shared", "native"])
    parser.add_argument("--event-kind", default="first_diff")
    parser.add_argument("--buckets", nargs="+", default=list(MAIN_SMOKE_BUCKETS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS_SMOKE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--crosscoder-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--activation-mode", choices=["mediation_topk", "raw_relu", "threshold"], default="mediation_topk")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-events", type=int, default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--no-matched-random", action="store_true")
    parser.add_argument("--no-same-delta", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model not in PRIMARY_MODELS:
        log.warning("%s is not a primary Exp41 model", args.model)
    path = run_model(args)
    print(f"[exp41] wrote {path}")


if __name__ == "__main__":
    main()
