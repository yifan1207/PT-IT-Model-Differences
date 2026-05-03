"""Collect Exp43 feature-rescue and middle-to-terminal handoff records.

Exp43 is intentionally paired with Exp42.  Exp42 asks whether causally ranked
terminal features matter more in the IT-upstream cell than in the PT-upstream
cell.  Exp43 asks the sufficiency-style follow-up: if the PT-upstream/IT-late
cell under-activates those features, does patching their IT-upstream activation
pattern rescue IT-token margin?
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp20_divergence_token_counterfactual.collect import DEPTH_ABLATION_WINDOWS
from src.poc.exp23_midlate_interaction_suite.boundary import BoundaryStateCapture, BoundaryStatePatch
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
from src.poc.exp28_late_mlp_crosscoder_mediation.run_mediation import (
    LatentRow,
    _dtype_from_name,
    _interaction,
    _load_causal_rank_rows,
    _load_manifest_records_window,
    _margin,
    _matched_random,
    _readout_margin_vector,
)
from src.poc.exp42_terminal_feature_upstream_conditioning.collect import (
    _crosscoder_layers,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


DEFAULT_RESCUE_K_LIST = (20, 50, 100, 200)
DEFAULT_RESCUE_ALPHAS = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class RescueSpec:
    feature_set: str
    k: int
    control_seed: int | None
    control_mode: str
    rows: tuple[LatentRow, ...]

    @property
    def key(self) -> str:
        seed = "none" if self.control_seed is None else str(self.control_seed)
        return f"{self.feature_set}|{self.k}|{seed}|{self.control_mode}"


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


def _build_rescue_specs(
    *,
    causal_rows: list[LatentRow],
    pool_rows: list[LatentRow],
    k_list: list[int],
    random_seeds: list[int],
    include_same_delta_random: bool,
) -> list[RescueSpec]:
    max_k = max(k_list)
    causal_top_max = causal_rows[:max_k]
    fixed_random_orders = {
        int(seed): _matched_random(pool_rows, causal_top_max, seed=int(seed))
        for seed in random_seeds
    }
    specs: list[RescueSpec] = []
    for k in k_list:
        causal = tuple(causal_rows[:k])
        specs.append(RescueSpec("causal_top", int(k), None, "feature_delta", causal))
        for seed in random_seeds:
            specs.append(
                RescueSpec(
                    "causal_matched_random",
                    int(k),
                    int(seed),
                    "feature_delta",
                    tuple(fixed_random_orders[int(seed)][:k]),
                )
            )
        if include_same_delta_random:
            for seed in random_seeds:
                specs.append(
                    RescueSpec(
                        "causal_same_delta_random",
                        int(k),
                        int(seed),
                        "same_delta_random",
                        causal,
                    )
                )
    return specs


def _layer_feature_sets(specs: list[RescueSpec]) -> dict[int, dict[str, list[int]]]:
    out: dict[int, dict[str, list[int]]] = defaultdict(dict)
    for spec in specs:
        by_layer: dict[int, list[int]] = defaultdict(list)
        for row in spec.rows:
            by_layer[int(row.layer)].append(int(row.latent_id))
        for layer, ids in by_layer.items():
            out[layer][spec.key] = ids
    return out


class SelectedFeatureSnapshot:
    """Capture selected latent activations at terminal MLP layers."""

    def __init__(
        self,
        *,
        model: Any,
        steering_adapter: Any,
        run_root: Path,
        layer_feature_sets: dict[int, dict[str, list[int]]],
        branch: int,
        device: torch.device,
        margin_vector: torch.Tensor,
        crosscoder_cache: dict[int, Any],
        crosscoder_dtype: torch.dtype | None,
        use_threshold: bool,
    ) -> None:
        self.model = model
        self.layers = steering_adapter.get_layers(model)
        self.run_root = run_root
        self.layer_feature_sets = layer_feature_sets
        self.branch = int(branch)
        self.device = device
        self.margin_vector = margin_vector.detach().float().to(device)
        self.crosscoder_cache = crosscoder_cache
        self.crosscoder_dtype = crosscoder_dtype
        self.use_threshold = bool(use_threshold)
        self.handles: list[Any] = []
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.aggregates: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def __enter__(self) -> "SelectedFeatureSnapshot":
        for layer_idx in sorted(self.layer_feature_sets):
            crosscoder = self._load_layer(layer_idx)
            feature_sets = self.layer_feature_sets[layer_idx]
            union_ids = sorted({int(latent) for ids in feature_sets.values() for latent in ids})
            if not union_ids:
                continue
            union_tensor = torch.tensor(union_ids, dtype=torch.long, device=self.device)
            pos = {latent: idx for idx, latent in enumerate(union_ids)}
            set_positions = {
                key: torch.tensor([pos[int(latent)] for latent in ids], dtype=torch.long, device=self.device)
                for key, ids in feature_sets.items()
            }

            def hook(_module, _args, output, li=layer_idx, cc=crosscoder, latents=union_tensor, positions=set_positions):
                if not torch.is_tensor(output):
                    raise RuntimeError(f"Expected tensor MLP output at layer {li}, got {type(output)}")
                update = output[:, -1, :]
                _contrib, selected_features = cc.selected_branch_contribution(
                    update,
                    branch=self.branch,
                    latent_ids=latents,
                    use_threshold=self.use_threshold,
                )
                selected_features = selected_features.detach().float()
                decoder = cc.decoder[latents, self.branch, :].detach().float().to(update.device)
                decoder_margin = decoder @ self.margin_vector.to(update.device)
                self.snapshots[int(li)] = {
                    "latent_ids": latents.detach().cpu(),
                    "features": selected_features.detach().cpu(),
                }
                for key, idxs in positions.items():
                    feats = selected_features[:, idxs]
                    margins = decoder_margin[idxs]
                    self.aggregates[key]["sum_activation"] += float(feats.sum(dim=-1).mean().item())
                    self.aggregates[key]["decoder_margin_weighted_sum"] += float(
                        (feats * margins[None, :]).sum(dim=-1).mean().item()
                    )
                    self.aggregates[key]["selected_l0"] += float((feats > 0).float().sum(dim=-1).mean().item())
                    self.aggregates[key]["n_layers"] += 1.0
                return output

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

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class FeatureRescueEditor:
    """Patch selected IT-branch feature activations toward a target snapshot."""

    def __init__(
        self,
        *,
        model: Any,
        steering_adapter: Any,
        run_root: Path,
        spec: RescueSpec,
        target_snapshot: SelectedFeatureSnapshot,
        alpha: float,
        device: torch.device,
        crosscoder_cache: dict[int, Any],
        crosscoder_dtype: torch.dtype | None,
        use_threshold: bool,
        seed: int,
    ) -> None:
        self.layers = steering_adapter.get_layers(model)
        self.run_root = run_root
        self.spec = spec
        self.target_snapshot = target_snapshot
        self.alpha = float(alpha)
        self.device = device
        self.crosscoder_cache = crosscoder_cache
        self.crosscoder_dtype = crosscoder_dtype
        self.use_threshold = bool(use_threshold)
        self.seed = int(seed)
        self.handles: list[Any] = []
        self.diagnostics: list[dict[str, float | int | str]] = []

    def __enter__(self) -> "FeatureRescueEditor":
        by_layer: dict[int, list[int]] = defaultdict(list)
        for row in self.spec.rows:
            by_layer[int(row.layer)].append(int(row.latent_id))
        for layer_idx, ids in sorted(by_layer.items()):
            cc = self._load_layer(layer_idx)
            latent_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)

            def hook(_module, _args, output, li=layer_idx, latents=latent_tensor, crosscoder=cc):
                if not torch.is_tensor(output):
                    raise RuntimeError(f"Expected tensor MLP output at layer {li}, got {type(output)}")
                target = self._target_features(int(li), latents).to(device=output.device)
                update = output[:, -1, :]
                _current_contrib, current = crosscoder.selected_branch_contribution(
                    update,
                    branch=1,
                    latent_ids=latents,
                    use_threshold=self.use_threshold,
                )
                current = current.float()
                feature_delta = target.to(dtype=current.dtype) - current
                decoder = crosscoder.decoder[latents, 1, :].to(device=output.device, dtype=feature_delta.dtype)
                decoded_delta = feature_delta @ decoder
                if self.spec.control_mode == "same_delta_random":
                    decoded_delta = self._same_norm_random(decoded_delta, layer_idx=int(li), dtype=decoded_delta.dtype)
                delta = self.alpha * decoded_delta.float()
                out = output.clone()
                out[:, -1, :] = (update.float() + delta).to(dtype=output.dtype)
                base_norm = update.float().norm(dim=-1).mean().clamp_min(1e-12)
                self.diagnostics.append(
                    {
                        "layer": int(li),
                        "n_latents": int(latents.numel()),
                        "feature_delta_l1": float(feature_delta.abs().sum(dim=-1).mean().item()),
                        "decoded_delta_norm": float(decoded_delta.float().norm(dim=-1).mean().item()),
                        "applied_delta_norm_frac": float((delta.norm(dim=-1).mean() / base_norm).item()),
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

    def _target_features(self, layer_idx: int, latents: torch.Tensor) -> torch.Tensor:
        snap = self.target_snapshot.snapshots.get(layer_idx)
        if snap is None:
            return torch.zeros((1, latents.numel()), dtype=torch.float32)
        snap_ids = [int(x) for x in snap["latent_ids"].tolist()]
        pos = {latent: idx for idx, latent in enumerate(snap_ids)}
        indices = [pos.get(int(latent), -1) for latent in latents.detach().cpu().tolist()]
        if any(idx < 0 for idx in indices):
            missing = [int(latent) for latent, idx in zip(latents.detach().cpu().tolist(), indices, strict=False) if idx < 0]
            raise KeyError(f"Target snapshot missing latents at layer {layer_idx}: {missing[:8]}")
        return snap["features"][:, indices].float()

    def _same_norm_random(self, decoded_delta: torch.Tensor, *, layer_idx: int, dtype: torch.dtype) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + 1009 * int(layer_idx)) % (2**31 - 1))
        unit = torch.randn(decoded_delta.shape[-1], generator=generator, dtype=torch.float32).to(decoded_delta.device)
        unit = unit / unit.norm().clamp_min(1e-12)
        return decoded_delta.float().norm(dim=-1, keepdim=True) * unit[None, :].to(dtype=dtype)

    def summary(self) -> dict[str, float | int]:
        if not self.diagnostics:
            return {"n_hook_calls": 0}
        return {
            "n_hook_calls": len(self.diagnostics),
            "mean_decoded_delta_norm": float(sum(float(r["decoded_delta_norm"]) for r in self.diagnostics) / len(self.diagnostics)),
            "mean_applied_delta_norm_frac": float(
                sum(float(r["applied_delta_norm_frac"]) for r in self.diagnostics) / len(self.diagnostics)
            ),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class WindowMlpGraft:
    """Swap donor MLP outputs into a host model over a layer window."""

    def __init__(self, *, host_layers: list[Any], donor_layers: list[Any], start: int, end: int) -> None:
        self.host_layers = host_layers
        self.donor_layers = donor_layers
        self.start = int(start)
        self.end = int(end)
        self.handles: list[Any] = []

    def __enter__(self) -> "WindowMlpGraft":
        for layer_idx in range(self.start, self.end):
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


def _hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if torch.is_tensor(output):
        return output
    raise RuntimeError(f"Expected transformer layer tensor/tuple output, got {type(output)}")


@torch.no_grad()
def _baseline(
    *,
    model: Any,
    layers: list[Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    snapshot: SelectedFeatureSnapshot | None = None,
    graft: WindowMlpGraft | None = None,
) -> dict[str, Any]:
    capture = BoundaryStateCapture(layers[boundary_layer])
    final_hidden: torch.Tensor | None = None

    def final_hook(_module, _args, output):
        nonlocal final_hidden
        hidden = _hidden_from_layer_output(output)
        if hidden.ndim != 3:
            raise RuntimeError(f"Expected final hidden [batch, seq, d_model], got {tuple(hidden.shape)}")
        final_hidden = hidden[0, -1, :].detach().clone()

    final_handle = layers[-1].register_forward_hook(final_hook)
    try:
        contexts = []
        if graft is not None:
            contexts.append(graft)
        if snapshot is not None:
            contexts.append(snapshot)
        for ctx in contexts:
            ctx.__enter__()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        boundary = capture.snapshot()
    finally:
        for ctx in reversed(locals().get("contexts", [])):
            ctx.__exit__(None, None, None)
        final_handle.remove()
        capture.close()
    if final_hidden is None:
        raise RuntimeError("Final hidden state was not captured")
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": final_hidden,
        "boundary_state": boundary,
    }


@torch.no_grad()
def _patched(
    *,
    model: Any,
    layers: list[Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    donor_boundary_state: torch.Tensor,
    snapshot: SelectedFeatureSnapshot | None = None,
    rescue: FeatureRescueEditor | None = None,
) -> dict[str, Any]:
    patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state)
    final_hidden: torch.Tensor | None = None

    def final_hook(_module, _args, output):
        nonlocal final_hidden
        hidden = _hidden_from_layer_output(output)
        if hidden.ndim != 3:
            raise RuntimeError(f"Expected final hidden [batch, seq, d_model], got {tuple(hidden.shape)}")
        final_hidden = hidden[0, -1, :].detach().clone()

    final_handle = layers[-1].register_forward_hook(final_hook)
    try:
        contexts = []
        if snapshot is not None:
            contexts.append(snapshot)
        if rescue is not None:
            contexts.append(rescue)
        for ctx in contexts:
            ctx.__enter__()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for ctx in reversed(locals().get("contexts", [])):
            ctx.__exit__(None, None, None)
        final_handle.remove()
        patcher.close()
    if final_hidden is None:
        raise RuntimeError("Final hidden state was not captured")
    return {"logits": outputs.logits[0, -1, :].detach().float(), "final_hidden": final_hidden}


def _snapshot_metric(snapshot: SelectedFeatureSnapshot, spec: RescueSpec, field: str) -> float | None:
    agg = snapshot.aggregates.get(spec.key)
    if not agg:
        return None
    return float(agg.get(field, 0.0))


def _token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


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
    pt_layers = steering_adapter.get_layers(pt_model)
    it_layers = steering_adapter.get_layers(it_model)
    dict_layers = _crosscoder_layers(args.crosscoder_root)
    if not dict_layers:
        raise FileNotFoundError(f"No crosscoder dictionary layers under {args.crosscoder_root / 'dictionaries'}")
    boundary_layer = int(args.boundary_layer_override) if args.boundary_layer_override is not None else int(min(dict_layers))

    causal_rows, pool_rows = _load_latent_rows_for_model(args.crosscoder_root)
    rescue_specs = _build_rescue_specs(
        causal_rows=causal_rows,
        pool_rows=pool_rows,
        k_list=[int(k) for k in args.k_list],
        random_seeds=[int(seed) for seed in args.random_seeds],
        include_same_delta_random=not args.no_same_delta_random,
    )
    layer_sets = _layer_feature_sets(rescue_specs)
    middle_specs = [s for s in rescue_specs if s.feature_set == "causal_top" and s.k == int(args.middle_k)]
    if not middle_specs:
        middle_specs = [RescueSpec("causal_top", int(args.middle_k), None, "feature_delta", tuple(causal_rows[: int(args.middle_k)]))]
        for layer, ids in _layer_feature_sets(middle_specs).items():
            layer_sets[layer].update(ids)

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
    rescue_alphas = [float(x) for x in args.rescue_alphas]
    log.info(
        "[exp43] model=%s boundary=%d prompts=%d specs=%d alphas=%s output=%s",
        args.model,
        boundary_layer,
        len(manifest_rows),
        len(rescue_specs),
        rescue_alphas,
        out_path,
    )

    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for manifest_record in manifest_rows:
            prompt_id = str(manifest_record.get("prompt_id"))
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                continue
            raw_prompt = get_prompt_for_variant(dataset_record, variant="pt", tokenizer=pt_tokenizer, apply_chat_template=False)
            prompt_ids = pt_tokenizer.encode(raw_prompt, add_special_tokens=True)
            for event_kind, event in _unique_events(manifest_record, list(args.event_kinds)):
                if "duplicate_of" in event:
                    continue
                done_key = "|".join([prompt_id, event_kind])
                if done_key in done:
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
                margin_vector = _readout_margin_vector(readout_bundle, y_pt=y_pt, y_it=y_it, device=device)

                target_snapshot = SelectedFeatureSnapshot(
                    model=it_model,
                    steering_adapter=steering_adapter,
                    run_root=args.crosscoder_root,
                    layer_feature_sets=layer_sets,
                    branch=1,
                    device=device,
                    margin_vector=margin_vector,
                    crosscoder_cache=crosscoder_cache,
                    crosscoder_dtype=crosscoder_dtype,
                    use_threshold=not args.no_threshold,
                )
                current_snapshot = SelectedFeatureSnapshot(
                    model=it_model,
                    steering_adapter=steering_adapter,
                    run_root=args.crosscoder_root,
                    layer_feature_sets=layer_sets,
                    branch=1,
                    device=device,
                    margin_vector=margin_vector,
                    crosscoder_cache=crosscoder_cache,
                    crosscoder_dtype=crosscoder_dtype,
                    use_threshold=not args.no_threshold,
                )
                base_pt = _baseline(
                    model=pt_model,
                    layers=pt_layers,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    boundary_layer=boundary_layer,
                )
                base_it = _baseline(
                    model=it_model,
                    layers=it_layers,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    boundary_layer=boundary_layer,
                    snapshot=target_snapshot,
                )
                upt_lit = _patched(
                    model=it_model,
                    layers=it_layers,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    boundary_layer=boundary_layer,
                    donor_boundary_state=base_pt["boundary_state"],
                    snapshot=current_snapshot,
                )
                uit_lpt = _patched(
                    model=pt_model,
                    layers=pt_layers,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    boundary_layer=boundary_layer,
                    donor_boundary_state=base_it["boundary_state"],
                )
                full_cells = {
                    "U_PT__L_PT": _margin(base_pt, readout_bundle, y_pt, y_it),
                    "U_IT__L_IT": _margin(base_it, readout_bundle, y_pt, y_it),
                    "U_PT__L_IT": _margin(upt_lit, readout_bundle, y_pt, y_it),
                    "U_IT__L_PT": _margin(uit_lpt, readout_bundle, y_pt, y_it),
                }
                missing_margin = full_cells["U_IT__L_IT"] - full_cells["U_PT__L_IT"]
                event_base = {
                    "experiment": "exp43_feature_rescue_handoff",
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
                    "boundary_layer": boundary_layer,
                    "full_cells": full_cells,
                    "interaction_full": _interaction(full_cells),
                    "missing_margin_upt_lit_to_uit_lit": missing_margin,
                }
                fout.write(json.dumps({**event_base, "record_type": "event_baseline"}, separators=(",", ":")) + "\n")

                for spec_sel in rescue_specs:
                    for alpha in rescue_alphas:
                        if abs(float(alpha)) <= 1e-12:
                            rescued_margin = full_cells["U_PT__L_IT"]
                            rescue_summary: dict[str, float | int | bool] = {"n_hook_calls": 0, "skipped_noop": True}
                        else:
                            editor = FeatureRescueEditor(
                                model=it_model,
                                steering_adapter=steering_adapter,
                                run_root=args.crosscoder_root,
                                spec=spec_sel,
                                target_snapshot=target_snapshot,
                                alpha=float(alpha),
                                device=device,
                                crosscoder_cache=crosscoder_cache,
                                crosscoder_dtype=crosscoder_dtype,
                                use_threshold=not args.no_threshold,
                                seed=_stable_seed(args.model, prompt_id, event_kind, spec_sel.key),
                            )
                            rescued = _patched(
                                model=it_model,
                                layers=it_layers,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                boundary_layer=boundary_layer,
                                donor_boundary_state=base_pt["boundary_state"],
                                rescue=editor,
                            )
                            rescued_margin = _margin(rescued, readout_bundle, y_pt, y_it)
                            rescue_summary = editor.summary()
                        rescue_gain = rescued_margin - full_cells["U_PT__L_IT"]
                        after_cells = dict(full_cells)
                        after_cells["U_PT__L_IT"] = rescued_margin
                        payload = {
                            **event_base,
                            "record_type": "rescue",
                            "feature_set": spec_sel.feature_set,
                            "control_mode": spec_sel.control_mode,
                            "k": int(spec_sel.k),
                            "control_seed": spec_sel.control_seed,
                            "alpha": float(alpha),
                            "n_features": len(spec_sel.rows),
                            "target_sum_activation": _snapshot_metric(target_snapshot, spec_sel, "sum_activation"),
                            "current_sum_activation": _snapshot_metric(current_snapshot, spec_sel, "sum_activation"),
                            "target_decoder_margin_weighted_sum": _snapshot_metric(
                                target_snapshot, spec_sel, "decoder_margin_weighted_sum"
                            ),
                            "current_decoder_margin_weighted_sum": _snapshot_metric(
                                current_snapshot, spec_sel, "decoder_margin_weighted_sum"
                            ),
                            "rescued_upt_lit_margin": rescued_margin,
                            "rescue_gain": rescue_gain,
                            "rescue_fraction": rescue_gain / missing_margin if missing_margin > 1e-6 else None,
                            "missing_margin_positive": bool(missing_margin > 1e-6),
                            "interaction_after_rescue": _interaction(after_cells),
                            "interaction_drop_from_rescue": _interaction(full_cells) - _interaction(after_cells),
                            "rescue_summary": rescue_summary,
                        }
                        fout.write(json.dumps(payload, separators=(",", ":")) + "\n")

                if not args.no_middle_probe:
                    windows = DEPTH_ABLATION_WINDOWS[args.model]
                    for window_name in args.middle_windows:
                        start, end = windows[window_name]
                        probe_snapshot = SelectedFeatureSnapshot(
                            model=it_model,
                            steering_adapter=steering_adapter,
                            run_root=args.crosscoder_root,
                            layer_feature_sets=_layer_feature_sets(middle_specs),
                            branch=1,
                            device=device,
                            margin_vector=margin_vector,
                            crosscoder_cache=crosscoder_cache,
                            crosscoder_dtype=crosscoder_dtype,
                            use_threshold=not args.no_threshold,
                        )
                        graft = WindowMlpGraft(host_layers=it_layers, donor_layers=pt_layers, start=start, end=end)
                        probe = _baseline(
                            model=it_model,
                            layers=it_layers,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            boundary_layer=boundary_layer,
                            snapshot=probe_snapshot,
                            graft=graft,
                        )
                        probe_margin = _margin(probe, readout_bundle, y_pt, y_it)
                        for spec_sel in middle_specs:
                            payload = {
                                **event_base,
                                "record_type": "middle_probe",
                                "window": window_name,
                                "window_start": int(start),
                                "window_end": int(end),
                                "feature_set": spec_sel.feature_set,
                                "k": int(spec_sel.k),
                                "native_it_margin": full_cells["U_IT__L_IT"],
                                "probe_margin": probe_margin,
                                "margin_drop": full_cells["U_IT__L_IT"] - probe_margin,
                                "native_sum_activation": _snapshot_metric(target_snapshot, spec_sel, "sum_activation"),
                                "probe_sum_activation": _snapshot_metric(probe_snapshot, spec_sel, "sum_activation"),
                                "activation_drop": (_snapshot_metric(target_snapshot, spec_sel, "sum_activation") or 0.0)
                                - (_snapshot_metric(probe_snapshot, spec_sel, "sum_activation") or 0.0),
                                "native_decoder_margin_weighted_sum": _snapshot_metric(
                                    target_snapshot, spec_sel, "decoder_margin_weighted_sum"
                                ),
                                "probe_decoder_margin_weighted_sum": _snapshot_metric(
                                    probe_snapshot, spec_sel, "decoder_margin_weighted_sum"
                                ),
                            }
                            payload["decoder_margin_weighted_drop"] = (
                                (payload["native_decoder_margin_weighted_sum"] or 0.0)
                                - (payload["probe_decoder_margin_weighted_sum"] or 0.0)
                            )
                            fout.write(json.dumps(payload, separators=(",", ":")) + "\n")

                fout.write(
                    json.dumps(
                        {
                            "experiment": "exp43_feature_rescue_handoff",
                            "record_type": "event_done",
                            "model": args.model,
                            "prompt_id": prompt_id,
                            "event_kind": event_kind,
                            "event_step": int(event["step"]),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                fout.flush()


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    done_keys: set[str] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker in range(n_workers):
            path = out_dir / f"records_w{worker}.jsonl.gz"
            if not path.exists():
                log.warning("[exp43] missing worker file %s", path)
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
        raise RuntimeError(f"Exp43 merge wrote no complete events to {merged}")
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
    parser.add_argument("--k-list", nargs="+", type=int, default=list(DEFAULT_RESCUE_K_LIST))
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--rescue-alphas", nargs="+", type=float, default=list(DEFAULT_RESCUE_ALPHAS))
    parser.add_argument("--middle-k", type=int, default=200)
    parser.add_argument("--middle-windows", nargs="+", choices=["early", "mid", "late"], default=["early", "mid"])
    parser.add_argument("--boundary-layer-override", type=int, default=None)
    parser.add_argument("--crosscoder-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-threshold", action="store_true")
    parser.add_argument("--no-same-delta-random", action="store_true")
    parser.add_argument("--no-middle-probe", action="store_true")
    parser.add_argument("--merge-only", action="store_true")


def main(args: argparse.Namespace) -> None:
    if args.merge_only:
        merge_workers(args.out_dir, args.n_workers)
        return
    run_worker(args)
