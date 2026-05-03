"""Collect Exp44 middle-to-terminal feature handoff records.

Exp44 tests whether IT-shaped middle/preterminal computation drives the same
terminal crosscoder features that mediate the IT-token readout interaction.
For each first-divergence event, the collector builds upstream boundary states
from PT/IT hosts with optional MLP-window grafts, then runs the IT terminal
stack with and without selected terminal features ablated.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp20_divergence_token_counterfactual.collect import DEPTH_ABLATION_WINDOWS
from src.poc.exp23_midlate_interaction_suite.boundary import (
    BoundaryStateCapture,
    BoundaryStatePatch,
    LayerResidualCapture,
)
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    DEFAULT_EVENT_KINDS,
    DEFAULT_EXP20_FALLBACK_ROOT,
    DEFAULT_EXP20_ROOT,
    _dataset_lookup,
    _make_readouts,
    _prefix_ids_for_event,
    _unique_events,
    _validate_tokenizers_and_tokens,
)
from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import BatchTopKCrossCoder
from src.poc.exp28_late_mlp_crosscoder_mediation.mediation_hooks import CrosscoderMlpModifier, FeatureSelection
from src.poc.exp28_late_mlp_crosscoder_mediation.run_mediation import (
    LatentRow,
    _dtype_from_name,
    _forward_with_optional_edit,
    _interaction,
    _load_causal_rank_rows,
    _load_manifest_records_window,
    _margin,
    _matched_random,
    _readout_margin_vector,
    _to_selection,
)
from src.poc.exp42_terminal_feature_upstream_conditioning.collect import (
    FeatureActivationRecorder,
    _crosscoder_layers,
    _top_active_noncausal,
)
from src.poc.exp43_feature_rescue_handoff.collect import WindowMlpGraft

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


PRIMARY_MODELS = ("llama31_8b", "mistral_7b", "qwen3_4b")
DIAGNOSTIC_MODELS = ("gemma3_4b",)
DEFAULT_K_LIST = (20, 50, 100, 200)
DEFAULT_WINDOWS = ("early", "mid", "late_preterminal", "midlate_preterminal", "terminal_entry")


@dataclass(frozen=True)
class HandoffSpec:
    feature_set: str
    k: int
    control_seed: int | None
    control_mode: str
    rows: tuple[LatentRow, ...]

    @property
    def key(self) -> str:
        seed = "none" if self.control_seed is None else str(self.control_seed)
        return f"{self.feature_set}|{self.k}|{seed}|{self.control_mode}"

    @property
    def activation_key(self) -> str:
        seed = "none" if self.control_seed is None else str(self.control_seed)
        return f"{self.feature_set}|{self.k}|{seed}"

    @property
    def selection(self) -> FeatureSelection:
        # Force full dictionaries so matched-random/top-active controls can use
        # arbitrary latent IDs, not just sliced causal-feature checkpoints.
        return _to_selection(self.key, list(self.rows), mode="ablate_full")


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _done_event_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        "|".join([str(row.get("prompt_id")), str(row.get("event_kind"))])
        for row in _json_rows(path)
        if row.get("record_type") == "event_done"
    }


def _load_latent_rows_for_model(run_root: Path) -> tuple[list[LatentRow], list[LatentRow]]:
    return _load_causal_rank_rows(run_root / "feature_stats" / "causal_feature_scores.csv")


def _build_handoff_specs(
    *,
    causal_rows: list[LatentRow],
    pool_rows: list[LatentRow],
    k_list: list[int],
    random_seeds: list[int],
    include_top_active: bool,
    include_same_delta_random: bool,
) -> list[HandoffSpec]:
    max_k = max(k_list)
    causal_top_max = causal_rows[:max_k]
    fixed_random = {
        int(seed): _matched_random(pool_rows, causal_top_max, seed=int(seed))
        for seed in random_seeds
    }
    specs: list[HandoffSpec] = []
    for k in k_list:
        causal = tuple(causal_rows[: int(k)])
        specs.append(HandoffSpec("causal_top", int(k), None, "feature_ablate", causal))
        for seed in random_seeds:
            specs.append(
                HandoffSpec(
                    "causal_matched_random",
                    int(k),
                    int(seed),
                    "feature_ablate",
                    tuple(fixed_random[int(seed)][: int(k)]),
                )
            )
        if include_top_active:
            specs.append(
                HandoffSpec(
                    "top_active_noncausal",
                    int(k),
                    None,
                    "feature_ablate",
                    tuple(_top_active_noncausal(pool_rows, causal_rows, k=int(k))),
                )
            )
        if include_same_delta_random:
            for seed in random_seeds:
                specs.append(
                    HandoffSpec(
                        "causal_same_delta_random",
                        int(k),
                        int(seed),
                        "same_delta_random",
                        causal,
                    )
                )
    return specs


def _layer_feature_sets(specs: list[HandoffSpec]) -> dict[int, dict[str, list[int]]]:
    out: dict[int, dict[str, list[int]]] = defaultdict(dict)
    for spec in specs:
        by_layer: dict[int, list[int]] = defaultdict(list)
        for row in spec.rows:
            by_layer[int(row.layer)].append(int(row.latent_id))
        for layer, ids in by_layer.items():
            out[int(layer)][spec.activation_key] = ids
    return out


def _aggregate_activation(records: list[dict[str, Any]], spec: HandoffSpec) -> dict[str, float | int | None]:
    rows = [
        row
        for row in records
        if row.get("feature_set") == spec.feature_set
        and int(row.get("k", -1)) == int(spec.k)
        and (
            ("none" if row.get("control_seed") is None else str(row.get("control_seed")))
            == ("none" if spec.control_seed is None else str(spec.control_seed))
        )
    ]
    if not rows:
        return {
            "sum_activation": None,
            "decoder_margin_weighted_sum": None,
            "decoder_weighted_sum": None,
            "activation_rate": None,
            "selected_l0": None,
            "reconstruction_error_rel": None,
            "selected_union_mass_fraction": None,
            "n_activation_layers": 0,
        }

    def mean(field: str) -> float | None:
        vals = []
        for row in rows:
            try:
                val = float(row.get(field))
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                vals.append(val)
        return float(sum(vals) / len(vals)) if vals else None

    def total(field: str) -> float | None:
        vals = []
        for row in rows:
            try:
                val = float(row.get(field))
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                vals.append(val)
        return float(sum(vals)) if vals else None

    return {
        "sum_activation": total("sum_activation"),
        "decoder_margin_weighted_sum": total("decoder_margin_weighted_sum"),
        "decoder_weighted_sum": total("decoder_weighted_sum"),
        "activation_rate": mean("activation_rate"),
        "selected_l0": total("selected_feature_l0"),
        "reconstruction_error_rel": mean("reconstruction_error_rel"),
        "selected_union_mass_fraction": mean("selected_union_mass_fraction"),
        "n_activation_layers": len(rows),
    }


def _finite_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _window_defs(model_name: str, n_layers: int, boundary_layer: int) -> dict[str, tuple[int, int]]:
    base = DEPTH_ABLATION_WINDOWS[model_name]
    early = tuple(base["early"])
    mid = tuple(base["mid"])
    late = tuple(base["late"])
    late_start = min(max(int(late[0]), 0), int(boundary_layer))
    late_preterminal = (late_start, int(boundary_layer))
    if late_preterminal[1] - late_preterminal[0] <= 0:
        width = max(1, int(round(0.2 * n_layers)))
        late_preterminal = (max(0, int(boundary_layer) - width), int(boundary_layer))
    midlate_preterminal = (int(mid[0]), int(boundary_layer))
    return {
        "early": (int(early[0]), min(int(early[1]), int(boundary_layer))),
        "mid": (int(mid[0]), min(int(mid[1]), int(boundary_layer))),
        "late_preterminal": late_preterminal,
        "midlate_preterminal": midlate_preterminal,
        "terminal_entry": (int(boundary_layer), int(boundary_layer)),
    }


class SameDeltaRandomAblator:
    """Subtract a same-norm random vector at each selected terminal MLP layer."""

    def __init__(
        self,
        *,
        model: Any,
        steering_adapter: Any,
        run_root: Path,
        spec: HandoffSpec,
        device: torch.device,
        seed: int,
        crosscoder_cache: dict[int, Any],
        crosscoder_dtype: torch.dtype | None,
        use_threshold: bool,
    ) -> None:
        self.layers = steering_adapter.get_layers(model)
        self.run_root = run_root
        self.spec = spec
        self.device = device
        self.seed = int(seed)
        self.crosscoder_cache = crosscoder_cache
        self.crosscoder_dtype = crosscoder_dtype
        self.use_threshold = bool(use_threshold)
        self.handles: list[Any] = []
        self.diagnostics: list[dict[str, float | int | str]] = []

    def __enter__(self) -> "SameDeltaRandomAblator":
        by_layer: dict[int, list[int]] = defaultdict(list)
        for row in self.spec.rows:
            by_layer[int(row.layer)].append(int(row.latent_id))
        for layer_idx, latent_ids in sorted(by_layer.items()):
            cc = self._load_layer(layer_idx)
            latents = torch.tensor(latent_ids, dtype=torch.long, device=self.device)

            def hook(_module, _args, output, li=layer_idx, latent_tensor=latents, crosscoder=cc):
                if not torch.is_tensor(output):
                    raise RuntimeError(f"Expected tensor MLP output at layer {li}, got {type(output)}")
                update = output[:, -1, :]
                contrib, features = crosscoder.selected_branch_contribution(
                    update,
                    branch=1,
                    latent_ids=latent_tensor,
                    use_threshold=self.use_threshold,
                )
                random_delta = self._same_norm_random(contrib.float(), layer_idx=int(li), dtype=output.dtype)
                out = output.clone()
                out[:, -1, :] = (update.float() - random_delta.float()).to(dtype=output.dtype)
                base_norm = update.float().norm(dim=-1).mean().clamp_min(1e-8)
                self.diagnostics.append(
                    {
                        "layer": int(li),
                        "n_selected": int(latent_tensor.numel()),
                        "feature_l0": float((features > 0).float().sum(dim=-1).mean().item()),
                        "delta_norm": float(random_delta.float().norm(dim=-1).mean().item()),
                        "delta_norm_frac": float((random_delta.float().norm(dim=-1).mean() / base_norm).item()),
                    }
                )
                return out

            self.handles.append(self.layers[layer_idx].mlp.register_forward_hook(hook))
        return self

    def _load_layer(self, layer_idx: int) -> BatchTopKCrossCoder:
        if layer_idx not in self.crosscoder_cache:
            path = self.run_root / "dictionaries" / f"layer_{layer_idx}" / "crosscoder.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing crosscoder dictionary: {path}")
            cc = BatchTopKCrossCoder.load(path, device=self.device)
            if self.crosscoder_dtype is not None:
                cc = cc.to(dtype=self.crosscoder_dtype)
            cc.eval()
            self.crosscoder_cache[layer_idx] = cc
        return self.crosscoder_cache[layer_idx]

    def _same_norm_random(self, delta: torch.Tensor, *, layer_idx: int, dtype: torch.dtype) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + 1009 * int(layer_idx)) % (2**31 - 1))
        unit = torch.randn(delta.shape[-1], generator=generator, dtype=torch.float32).to(delta.device)
        unit = unit / unit.norm().clamp_min(1e-12)
        return (delta.float().norm(dim=-1, keepdim=True) * unit[None, :]).to(dtype=dtype)

    def summary(self) -> dict[str, Any]:
        if not self.diagnostics:
            return {"n_layer_calls": 0}
        return {
            "selection": self.spec.key,
            "mode": "same_delta_random",
            "n_layer_calls": len(self.diagnostics),
            "mean_delta_norm_frac": float(
                sum(float(row["delta_norm_frac"]) for row in self.diagnostics) / len(self.diagnostics)
            ),
            "mean_feature_l0": float(
                sum(float(row["feature_l0"]) for row in self.diagnostics) / len(self.diagnostics)
            ),
            "by_layer": {
                str(row["layer"]): {
                    "n": 1,
                    "mean_delta_norm_frac": row["delta_norm_frac"],
                    "mean_feature_l0": row["feature_l0"],
                    "n_selected": row["n_selected"],
                }
                for row in self.diagnostics
            },
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@torch.no_grad()
def _capture_boundary_with_graft(
    *,
    model: Any,
    layers: list[Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    graft: WindowMlpGraft | None = None,
) -> dict[str, Any]:
    capture = BoundaryStateCapture(layers[boundary_layer])
    try:
        if graft is None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            with graft:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        boundary = capture.snapshot()
    finally:
        capture.close()
    return {"boundary_state": boundary, "logits": outputs.logits[0, -1, :].detach().float()}


@torch.no_grad()
def _baseline_with_boundary_and_features(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    recorder: FeatureActivationRecorder | None,
) -> dict[str, Any]:
    capture = BoundaryStateCapture(layers[boundary_layer])
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        if recorder is None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            with recorder:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        boundary = capture.snapshot()
        residuals = residual_capture.snapshot()
    finally:
        capture.close()
        residual_capture.close()
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": residuals[-1],
        "boundary_state": boundary,
        "activation_records": list(recorder.records) if recorder is not None else [],
    }


@torch.no_grad()
def _patched_with_features(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    donor_boundary_state: torch.Tensor,
    recorder: FeatureActivationRecorder | None,
) -> dict[str, Any]:
    patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state)
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        if recorder is None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            with recorder:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        residuals = residual_capture.snapshot()
    finally:
        patcher.close()
        residual_capture.close()
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": residuals[-1],
        "patch_n": patcher.n_patches,
        "activation_records": list(recorder.records) if recorder is not None else [],
    }


@torch.no_grad()
def _forward_with_spec_ablation(
    *,
    model: Any,
    adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    donor_boundary_state: torch.Tensor,
    run_root: Path,
    steering_adapter: Any,
    spec: HandoffSpec,
    crosscoder_cache: dict[int, Any],
    crosscoder_dtype: torch.dtype | None,
    margin_vector: torch.Tensor,
    seed: int,
    use_threshold: bool,
) -> dict[str, Any]:
    if spec.control_mode != "same_delta_random":
        return _forward_with_optional_edit(
            model=model,
            adapter=adapter,
            layers=layers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
            donor_boundary_state=donor_boundary_state,
            run_root=run_root,
            steering_adapter=steering_adapter,
            selection=spec.selection,
            crosscoder_cache=crosscoder_cache,
            crosscoder_dtype=crosscoder_dtype,
            coverage_margin_vector=margin_vector,
        )

    patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state)
    modifier = SameDeltaRandomAblator(
        model=model,
        steering_adapter=steering_adapter,
        run_root=run_root,
        spec=spec,
        device=input_ids.device,
        seed=seed,
        crosscoder_cache=crosscoder_cache,
        crosscoder_dtype=crosscoder_dtype,
        use_threshold=use_threshold,
    )
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        with modifier:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        residuals = residual_capture.snapshot()
    finally:
        modifier.close()
        patcher.close()
        residual_capture.close()
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": residuals[-1],
        "hook_summary": modifier.summary(),
        "patch_n": patcher.n_patches,
    }


def _new_recorder(
    *,
    model: Any,
    steering_adapter: Any,
    crosscoder_root: Path,
    layer_sets: dict[int, dict[str, list[int]]],
    device: torch.device,
    margin_vector: torch.Tensor,
    crosscoder_cache: dict[int, Any],
    crosscoder_dtype: torch.dtype | None,
    no_threshold: bool,
    no_dictionary_diagnostics: bool,
) -> FeatureActivationRecorder:
    return FeatureActivationRecorder(
        model=model,
        steering_adapter=steering_adapter,
        run_root=crosscoder_root,
        layer_feature_sets=layer_sets,
        branch=1,
        device=device,
        margin_vector=margin_vector,
        crosscoder_cache=crosscoder_cache,
        crosscoder_dtype=crosscoder_dtype,
        use_threshold=not no_threshold,
        compute_diagnostics=not no_dictionary_diagnostics,
    )


def _write_jsonl(fout: Any, row: dict[str, Any]) -> None:
    fout.write(json.dumps(row, separators=(",", ":")) + "\n")


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
    readouts = _make_readouts(
        models=models,
        tokenizers=tokenizers,
        steering_adapter=steering_adapter,
        real_token_masks=real_token_masks,
    )
    readout_bundle = readouts[args.readout_name]
    adapter = steering_adapter.adapter
    pt_layers = steering_adapter.get_layers(pt_model)
    it_layers = steering_adapter.get_layers(it_model)
    n_layers = min(len(pt_layers), len(it_layers))

    dict_layers = _crosscoder_layers(args.crosscoder_root)
    if not dict_layers:
        raise FileNotFoundError(f"No crosscoder dictionary layers under {args.crosscoder_root / 'dictionaries'}")
    boundary_layer = int(args.boundary_layer_override) if args.boundary_layer_override is not None else int(min(dict_layers))
    if boundary_layer < 0 or boundary_layer >= n_layers:
        raise ValueError(f"Boundary layer {boundary_layer} outside shared layer range 0..{n_layers - 1}")
    terminal_scope = args.terminal_scope or f"final{n_layers - boundary_layer}"
    windows = _window_defs(args.model, n_layers, boundary_layer)
    requested_windows = [str(w) for w in args.windows]
    unknown = [w for w in requested_windows if w not in windows]
    if unknown:
        raise ValueError(f"Unknown windows for {args.model}: {unknown}; available={sorted(windows)}")

    causal_rows, pool_rows = _load_latent_rows_for_model(args.crosscoder_root)
    handoff_specs = _build_handoff_specs(
        causal_rows=causal_rows,
        pool_rows=pool_rows,
        k_list=[int(k) for k in args.k_list],
        random_seeds=[int(seed) for seed in args.random_seeds],
        include_top_active=not args.no_top_active_control,
        include_same_delta_random=not args.no_same_delta_random,
    )
    layer_sets = _layer_feature_sets(handoff_specs)
    log.info(
        "[exp44] model=%s boundary=%d scope=%s windows=%s specs=%d layers=%s",
        args.model,
        boundary_layer,
        terminal_scope,
        {w: windows[w] for w in requested_windows},
        len(handoff_specs),
        sorted(layer_sets),
    )

    dataset_by_id = _dataset_lookup(args.dataset)
    manifest_rows = _load_manifest_records_window(
        exp20_root=args.exp20_root,
        exp20_fallback_root=args.exp20_fallback_root,
        prompt_mode=args.prompt_mode,
        model=args.model,
        n_examples=args.n_prompts,
        skip_examples=args.skip_prompts,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
    )
    crosscoder_dtype = _dtype_from_name(args.crosscoder_dtype)
    crosscoder_cache: dict[int, Any] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"records_w{args.worker_index}.jsonl.gz"
    done = _done_event_keys(out_path)
    log.info(
        "[exp44] worker=%d/%d rows=%d resume_done=%d out=%s",
        args.worker_index,
        args.n_workers,
        len(manifest_rows),
        len(done),
        out_path,
    )

    n_events = 0
    n_written = 0
    n_fail = 0
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for row_idx, manifest_record in enumerate(manifest_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                log.warning("[exp44] missing dataset prompt_id=%s", prompt_id)
                continue
            raw_prompt = get_prompt_for_variant(
                dataset_record,
                variant="pt",
                tokenizer=pt_tokenizer,
                apply_chat_template=False,
            )
            prompt_ids = pt_tokenizer.encode(raw_prompt, add_special_tokens=True)
            for event_kind, event in _unique_events(manifest_record, list(args.event_kinds)):
                if "duplicate_of" in event:
                    continue
                event_key = "|".join([prompt_id, event_kind])
                if event_key in done:
                    continue
                y_pt = int(event["pt_token"]["token_id"])
                y_it = int(event["it_token"]["token_id"])
                prefix_ids = _prefix_ids_for_event(manifest_record, event)
                validation = _validate_tokenizers_and_tokens(
                    model_name=args.model,
                    prompt_mode=args.prompt_mode,
                    dataset_record=dataset_record,
                    prefix_ids=prefix_ids,
                    y_pt=y_pt,
                    y_it=y_it,
                    tokenizers=tokenizers,
                    readouts=readouts,
                )
                if not validation.get("ok"):
                    continue
                full_ids = prompt_ids + prefix_ids
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                try:
                    margin_vector = _readout_margin_vector(readout_bundle, y_pt=y_pt, y_it=y_it, device=device)
                    rec_it_native = _new_recorder(
                        model=it_model,
                        steering_adapter=steering_adapter,
                        crosscoder_root=args.crosscoder_root,
                        layer_sets=layer_sets,
                        device=device,
                        margin_vector=margin_vector,
                        crosscoder_cache=crosscoder_cache,
                        crosscoder_dtype=crosscoder_dtype,
                        no_threshold=args.no_threshold,
                        no_dictionary_diagnostics=args.no_dictionary_diagnostics,
                    )
                    rec_weak = _new_recorder(
                        model=it_model,
                        steering_adapter=steering_adapter,
                        crosscoder_root=args.crosscoder_root,
                        layer_sets=layer_sets,
                        device=device,
                        margin_vector=margin_vector,
                        crosscoder_cache=crosscoder_cache,
                        crosscoder_dtype=crosscoder_dtype,
                        no_threshold=args.no_threshold,
                        no_dictionary_diagnostics=args.no_dictionary_diagnostics,
                    )

                    base_pt = _baseline_with_boundary_and_features(
                        model=pt_model,
                        adapter=adapter,
                        layers=pt_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        recorder=None,
                    )
                    base_it = _baseline_with_boundary_and_features(
                        model=it_model,
                        adapter=adapter,
                        layers=it_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        recorder=rec_it_native,
                    )
                    weak = _patched_with_features(
                        model=it_model,
                        adapter=adapter,
                        layers=it_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        donor_boundary_state=base_pt["boundary_state"],
                        recorder=rec_weak,
                    )
                    opposite = _patched_with_features(
                        model=pt_model,
                        adapter=adapter,
                        layers=pt_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        donor_boundary_state=base_it["boundary_state"],
                        recorder=None,
                    )
                    full_cells = {
                        "U_PT__L_PT": _margin(base_pt, readout_bundle, y_pt, y_it),
                        "U_IT__L_IT": _margin(base_it, readout_bundle, y_pt, y_it),
                        "U_PT__L_IT": _margin(weak, readout_bundle, y_pt, y_it),
                        "U_IT__L_PT": _margin(opposite, readout_bundle, y_pt, y_it),
                    }
                    event_meta = {
                        "experiment": "exp44_middle_terminal_feature_handoff",
                        "model": args.model,
                        "prompt_id": prompt_id,
                        "prompt_mode": args.prompt_mode,
                        "event_kind": event_kind,
                        "event_step": int(event["step"]),
                        "position_ge_3": bool(int(event["step"]) >= 3),
                        "pt_token_id": y_pt,
                        "it_token_id": y_it,
                        "pt_token_text": validation.get("pt_token_text") or _token_text(pt_tokenizer, y_pt),
                        "it_token_text": validation.get("it_token_text") or _token_text(it_tokenizer, y_it),
                        "boundary_layer": int(boundary_layer),
                        "terminal_scope": terminal_scope,
                        "windows": {name: list(bounds) for name, bounds in windows.items()},
                        "full_cells": full_cells,
                        "interaction_full": _interaction(full_cells),
                    }
                    _write_jsonl(fout, {**event_meta, "record_type": "event_baseline"})
                    n_written += 1

                    base_ablate_cache: dict[tuple[str, str], dict[str, Any]] = {}
                    for spec_sel in handoff_specs:
                        for base_name, base_boundary in (
                            ("weak_hybrid", base_pt["boundary_state"]),
                            ("it_native", base_it["boundary_state"]),
                        ):
                            seed = _stable_seed(args.model, prompt_id, event_kind, base_name, spec_sel.key)
                            ablated = _forward_with_spec_ablation(
                                model=it_model,
                                adapter=adapter,
                                layers=it_layers,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                boundary_layer=boundary_layer,
                                donor_boundary_state=base_boundary,
                                run_root=args.crosscoder_root,
                                steering_adapter=steering_adapter,
                                spec=spec_sel,
                                crosscoder_cache=crosscoder_cache,
                                crosscoder_dtype=crosscoder_dtype,
                                margin_vector=margin_vector,
                                seed=seed,
                                use_threshold=not args.no_threshold,
                            )
                            base_ablate_cache[(base_name, spec_sel.key)] = {
                                "margin": _margin(ablated, readout_bundle, y_pt, y_it),
                                "hook_summary": ablated.get("hook_summary", {}),
                            }

                    for direction in args.directions:
                        for window_name in requested_windows:
                            start, end = windows[window_name]
                            if direction == "rescue":
                                base_name = "weak_hybrid"
                                base_margin = full_cells["U_PT__L_IT"]
                                base_activation_records = weak["activation_records"]
                                if window_name == "terminal_entry":
                                    boundary = base_it["boundary_state"]
                                else:
                                    graft = WindowMlpGraft(host_layers=pt_layers, donor_layers=it_layers, start=start, end=end)
                                    boundary = _capture_boundary_with_graft(
                                        model=pt_model,
                                        layers=pt_layers,
                                        input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        boundary_layer=boundary_layer,
                                        graft=graft,
                                    )["boundary_state"]
                            else:
                                base_name = "it_native"
                                base_margin = full_cells["U_IT__L_IT"]
                                base_activation_records = base_it["activation_records"]
                                if window_name == "terminal_entry":
                                    boundary = base_pt["boundary_state"]
                                else:
                                    graft = WindowMlpGraft(host_layers=it_layers, donor_layers=pt_layers, start=start, end=end)
                                    boundary = _capture_boundary_with_graft(
                                        model=it_model,
                                        layers=it_layers,
                                        input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        boundary_layer=boundary_layer,
                                        graft=graft,
                                    )["boundary_state"]

                            rec_perturbed = _new_recorder(
                                model=it_model,
                                steering_adapter=steering_adapter,
                                crosscoder_root=args.crosscoder_root,
                                layer_sets=layer_sets,
                                device=device,
                                margin_vector=margin_vector,
                                crosscoder_cache=crosscoder_cache,
                                crosscoder_dtype=crosscoder_dtype,
                                no_threshold=args.no_threshold,
                                no_dictionary_diagnostics=args.no_dictionary_diagnostics,
                            )
                            perturbed = _patched_with_features(
                                model=it_model,
                                adapter=adapter,
                                layers=it_layers,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                boundary_layer=boundary_layer,
                                donor_boundary_state=boundary,
                                recorder=rec_perturbed,
                            )
                            perturbed_margin = _margin(perturbed, readout_bundle, y_pt, y_it)
                            signed_margin_effect = (
                                perturbed_margin - base_margin
                                if direction == "rescue"
                                else base_margin - perturbed_margin
                            )
                            for spec_sel in handoff_specs:
                                base_agg = _aggregate_activation(base_activation_records, spec_sel)
                                perturbed_agg = _aggregate_activation(perturbed["activation_records"], spec_sel)
                                base_ablated = base_ablate_cache[(base_name, spec_sel.key)]
                                ablated = _forward_with_spec_ablation(
                                    model=it_model,
                                    adapter=adapter,
                                    layers=it_layers,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    boundary_layer=boundary_layer,
                                    donor_boundary_state=boundary,
                                    run_root=args.crosscoder_root,
                                    steering_adapter=steering_adapter,
                                    spec=spec_sel,
                                    crosscoder_cache=crosscoder_cache,
                                    crosscoder_dtype=crosscoder_dtype,
                                    margin_vector=margin_vector,
                                    seed=_stable_seed(args.model, prompt_id, event_kind, direction, window_name, spec_sel.key),
                                    use_threshold=not args.no_threshold,
                                )
                                perturbed_ablated_margin = _margin(ablated, readout_bundle, y_pt, y_it)
                                base_ablated_margin = float(base_ablated["margin"])
                                if direction == "rescue":
                                    mediated_gain = (perturbed_margin - base_margin) - (
                                        perturbed_ablated_margin - base_ablated_margin
                                    )
                                else:
                                    mediated_gain = (base_margin - perturbed_margin) - (
                                        base_ablated_margin - perturbed_ablated_margin
                                    )
                                row = {
                                    **event_meta,
                                    "record_type": "handoff",
                                    "direction": direction,
                                    "base_cell": base_name,
                                    "window": window_name,
                                    "window_start": int(start),
                                    "window_end": int(end),
                                    "feature_set": spec_sel.feature_set,
                                    "control_mode": spec_sel.control_mode,
                                    "k": int(spec_sel.k),
                                    "control_seed": spec_sel.control_seed,
                                    "n_features": len(spec_sel.rows),
                                    "base_margin": base_margin,
                                    "perturbed_margin": perturbed_margin,
                                    "perturbed_minus_base_margin": perturbed_margin - base_margin,
                                    "expected_positive_margin_effect": signed_margin_effect,
                                    "base_ablated_margin": base_ablated_margin,
                                    "perturbed_ablated_margin": perturbed_ablated_margin,
                                    "terminal_mediated_effect": mediated_gain,
                                    "terminal_mediated_fraction": mediated_gain / signed_margin_effect
                                    if abs(signed_margin_effect) > 1e-6
                                    else None,
                                    "base_decoder_margin_weighted_sum": base_agg["decoder_margin_weighted_sum"],
                                    "perturbed_decoder_margin_weighted_sum": perturbed_agg[
                                        "decoder_margin_weighted_sum"
                                    ],
                                    "activation_rescue_decoder_margin_weighted_sum": _finite_delta(
                                        perturbed_agg["decoder_margin_weighted_sum"],
                                        base_agg["decoder_margin_weighted_sum"],
                                    ),
                                    "base_sum_activation": base_agg["sum_activation"],
                                    "perturbed_sum_activation": perturbed_agg["sum_activation"],
                                    "activation_rescue_sum_activation": _finite_delta(
                                        perturbed_agg["sum_activation"],
                                        base_agg["sum_activation"],
                                    ),
                                    "base_activation_rate": base_agg["activation_rate"],
                                    "perturbed_activation_rate": perturbed_agg["activation_rate"],
                                    "base_reconstruction_error_rel": base_agg["reconstruction_error_rel"],
                                    "perturbed_reconstruction_error_rel": perturbed_agg["reconstruction_error_rel"],
                                    "base_selected_union_mass_fraction": base_agg["selected_union_mass_fraction"],
                                    "perturbed_selected_union_mass_fraction": perturbed_agg[
                                        "selected_union_mass_fraction"
                                    ],
                                    "base_hook_summary": base_ablated["hook_summary"],
                                    "perturbed_hook_summary": ablated.get("hook_summary", {}),
                                }
                                _write_jsonl(fout, row)
                                n_written += 1

                    _write_jsonl(
                        fout,
                        {
                            "experiment": "exp44_middle_terminal_feature_handoff",
                            "record_type": "event_done",
                            "model": args.model,
                            "prompt_id": prompt_id,
                            "event_kind": event_kind,
                            "event_step": int(event["step"]),
                            "n_specs": len(handoff_specs),
                            "n_windows": len(requested_windows),
                            "directions": list(args.directions),
                        },
                    )
                    fout.flush()
                    n_events += 1
                except Exception as exc:
                    n_fail += 1
                    log.exception("[exp44] prompt=%s event=%s failed: %s", prompt_id, event_kind, exc)

            if (row_idx + 1) % 5 == 0:
                log.info(
                    "[exp44] model=%s worker=%d/%d prompts=%d/%d events=%d rows=%d fail=%d",
                    args.model,
                    args.worker_index,
                    args.n_workers,
                    row_idx + 1,
                    len(manifest_rows),
                    n_events,
                    n_written,
                    n_fail,
                )
    log.info("[exp44] done model=%s events=%d rows=%d fail=%d", args.model, n_events, n_written, n_fail)


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    done_keys: set[str] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker in range(n_workers):
            path = out_dir / f"records_w{worker}.jsonl.gz"
            if not path.exists():
                log.warning("[exp44] missing worker file %s", path)
                continue
            current_event: tuple[str, str] | None = None
            buffer: list[dict[str, Any]] = []
            for row in _json_rows(path):
                row_type = row.get("record_type")
                if row_type == "event_done":
                    key = (str(row.get("prompt_id")), str(row.get("event_kind")))
                    if "|".join(key) not in done_keys:
                        for buffered in buffer:
                            fout.write(json.dumps(buffered, separators=(",", ":")) + "\n")
                        fout.write(json.dumps(row, separators=(",", ":")) + "\n")
                        done_keys.add("|".join(key))
                    buffer = []
                    current_event = None
                    continue
                key = (str(row.get("prompt_id")), str(row.get("event_kind")))
                if row_type == "event_baseline":
                    buffer = []
                    current_event = key
                elif current_event is None:
                    current_event = key
                elif key != current_event:
                    buffer = []
                    current_event = key
                buffer.append(row)
    if not done_keys:
        raise RuntimeError(f"Exp44 merge wrote no complete events to {merged}")
    return merged


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--prompt-mode", choices=["raw_shared"], default="raw_shared")
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=["first_diff"])
    parser.add_argument("--readout-name", default="common_it")
    parser.add_argument("--crosscoder-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-prompts", type=int, default=600)
    parser.add_argument("--skip-prompts", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--k-list", nargs="+", type=int, default=list(DEFAULT_K_LIST))
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--windows", nargs="+", choices=list(DEFAULT_WINDOWS), default=list(DEFAULT_WINDOWS))
    parser.add_argument("--directions", nargs="+", choices=["rescue", "degrade"], default=["rescue", "degrade"])
    parser.add_argument("--boundary-layer-override", type=int, default=None)
    parser.add_argument("--terminal-scope", default=None)
    parser.add_argument("--crosscoder-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-threshold", action="store_true")
    parser.add_argument("--no-top-active-control", action="store_true")
    parser.add_argument("--no-same-delta-random", action="store_true")
    parser.add_argument("--no-dictionary-diagnostics", action="store_true")
    parser.add_argument("--merge-only", action="store_true")


def main(args: argparse.Namespace) -> None:
    if args.merge_only:
        merge_workers(args.out_dir, args.n_workers)
        return
    run_worker(args)
