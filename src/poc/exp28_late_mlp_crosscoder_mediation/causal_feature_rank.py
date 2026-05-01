"""Rank Exp28 crosscoder features by gradient-linearized factorial contribution.

The original Exp28 global selector ranked features by IT-biased activation and a
local margin proxy. That is useful for finding IT-looking features, but it is
not the same object as the Exp23 interaction. This phase ranks features by the
linearized contribution they make to the two IT-late cells of the 2x2:

    score_f = attr_f(U_IT, L_IT) - attr_f(U_PT, L_IT)

where attr is the gradient of the IT-vs-PT divergent-token margin with respect
to the last-token MLP output, dotted into the IT-branch crosscoder feature
contribution. The exact mediation phase then ablates the ranked features in
full forward passes on held-out prompts.
"""

from __future__ import annotations

import argparse
import csv
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
from src.poc.exp23_midlate_interaction_suite.boundary import (
    BoundaryStateCapture,
    BoundaryStatePatch,
)
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    DEFAULT_EVENT_KINDS,
    DEFAULT_EXP20_FALLBACK_ROOT,
    DEFAULT_EXP20_ROOT,
    _dataset_lookup,
    _late_boundary,
    _load_manifest_records,
    _prefix_ids_for_event,
    _unique_events,
    _validate_tokenizers_and_tokens,
)
from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import BatchTopKCrossCoder
from src.poc.exp28_late_mlp_crosscoder_mediation.run_mediation import _dtype_from_name

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class FeatureAccumulator:
    score_sum: float = 0.0
    score_pos_sum: float = 0.0
    score_abs_sum: float = 0.0
    native_attr_sum: float = 0.0
    ptup_attr_sum: float = 0.0
    active_native_count: int = 0
    active_ptup_count: int = 0
    active_union_count: int = 0


class MlpOutputGradientCapture:
    """Replace last-token MLP outputs with gradient leaves and keep their grads."""

    def __init__(self, *, layers: list[torch.nn.Module], target_layers: list[int]) -> None:
        self.layers = layers
        self.target_layers = sorted(set(int(layer) for layer in target_layers))
        self.leaves: dict[int, torch.Tensor] = {}
        self.handles = [
            self.layers[layer_idx].mlp.register_forward_hook(self._make_hook(layer_idx))
            for layer_idx in self.target_layers
        ]

    def _make_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            if not torch.is_tensor(output):
                raise RuntimeError(f"Expected tensor MLP output at layer {layer_idx}, got {type(output)}")
            leaf = output[:, -1, :].detach().clone().requires_grad_(True)
            out = output.clone()
            out[:, -1, :] = leaf
            self.leaves[layer_idx] = leaf
            return out

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _available_dictionary_layers(run_root: Path) -> list[int]:
    layers = sorted(
        int(path.parent.name.split("_")[1])
        for path in (run_root / "dictionaries").glob("layer_*/crosscoder.pt")
    )
    if not layers:
        raise FileNotFoundError(f"No crosscoder dictionaries found under {run_root / 'dictionaries'}")
    return layers


def _load_manifest_records_window(
    *,
    exp20_root: Path,
    exp20_fallback_root: Path | None,
    prompt_mode: str,
    model: str,
    n_examples: int | None,
    skip_examples: int,
    worker_index: int,
    n_workers: int,
) -> list[dict[str, Any]]:
    total = None if n_examples is None else int(skip_examples) + int(n_examples)
    rows = _load_manifest_records(
        exp20_root=exp20_root,
        exp20_fallback_root=exp20_fallback_root,
        prompt_mode=prompt_mode,
        model=model,
        n_examples=total,
        worker_index=0,
        n_workers=1,
    )
    if skip_examples:
        rows = rows[int(skip_examples):]
    if n_examples is not None:
        rows = rows[: int(n_examples)]
    sliced = rows[worker_index::n_workers]
    log.info(
        "[exp28-causal-rank] manifest window skip=%d n=%s -> %d rows for worker %d/%d",
        skip_examples,
        n_examples,
        len(sliced),
        worker_index,
        n_workers,
    )
    return sliced


def _load_crosscoder(
    *,
    run_root: Path,
    layer: int,
    device: torch.device,
    dtype: torch.dtype | None,
    cache: dict[int, BatchTopKCrossCoder],
) -> BatchTopKCrossCoder:
    if layer not in cache:
        path = run_root / "dictionaries" / f"layer_{layer}" / "crosscoder.pt"
        crosscoder = BatchTopKCrossCoder.load(path, device=device)
        if dtype is not None:
            crosscoder = crosscoder.to(dtype=dtype)
        crosscoder.eval()
        cache[layer] = crosscoder
    return cache[layer]


def _cell_feature_attrs(
    *,
    model: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    donor_boundary_state: torch.Tensor | None,
    target_layers: list[int],
    run_root: Path,
    crosscoder_cache: dict[int, BatchTopKCrossCoder],
    crosscoder_dtype: torch.dtype | None,
    y_pt: int,
    y_it: int,
    use_threshold: bool,
) -> dict[tuple[int, int], float]:
    patcher = None
    capture = None
    try:
        if donor_boundary_state is not None:
            patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state)
        capture = MlpOutputGradientCapture(layers=layers, target_layers=target_layers)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            margin = outputs.logits[0, -1, int(y_it)] - outputs.logits[0, -1, int(y_pt)]
            margin.backward()
        out: dict[tuple[int, int], float] = {}
        for layer_idx, leaf in capture.leaves.items():
            if leaf.grad is None:
                continue
            grad = leaf.grad.detach().float()
            update = leaf.detach()
            crosscoder = _load_crosscoder(
                run_root=run_root,
                layer=layer_idx,
                device=input_ids.device,
                dtype=crosscoder_dtype,
                cache=crosscoder_cache,
            )
            features = crosscoder.encode_branch(update, branch=1, use_threshold=use_threshold)
            active = torch.nonzero(features[0] > 0, as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            vals = features[0, active].float()
            decoder = crosscoder.decoder[active, 1, :].to(device=input_ids.device, dtype=torch.float32)
            attrs = vals * (decoder @ grad[0])
            for latent_id, attr in zip(active.tolist(), attrs.tolist(), strict=True):
                out[(int(layer_idx), int(latent_id))] = float(attr)
        return out
    finally:
        if capture is not None:
            capture.close()
        if patcher is not None:
            patcher.close()
        model.zero_grad(set_to_none=True)


def _baseline_boundary(
    *,
    model: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
) -> torch.Tensor:
    capture = BoundaryStateCapture(layers[boundary_layer])
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return capture.snapshot()
    finally:
        capture.close()


def _write_worker_outputs(
    *,
    out_dir: Path,
    worker_index: int,
    accum: dict[tuple[int, int], FeatureAccumulator],
    n_events: int,
    meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"causal_feature_scores_w{worker_index}.csv"
    fields = [
        "layer",
        "latent_id",
        "score_sum",
        "score_pos_sum",
        "score_abs_sum",
        "native_attr_sum",
        "ptup_attr_sum",
        "active_native_count",
        "active_ptup_count",
        "active_union_count",
        "n_events",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (layer, latent_id), row in sorted(accum.items()):
            writer.writerow(
                {
                    "layer": layer,
                    "latent_id": latent_id,
                    "score_sum": row.score_sum,
                    "score_pos_sum": row.score_pos_sum,
                    "score_abs_sum": row.score_abs_sum,
                    "native_attr_sum": row.native_attr_sum,
                    "ptup_attr_sum": row.ptup_attr_sum,
                    "active_native_count": row.active_native_count,
                    "active_ptup_count": row.active_ptup_count,
                    "active_union_count": row.active_union_count,
                    "n_events": n_events,
                }
            )
    meta_path = out_dir / f"causal_feature_scores_w{worker_index}.meta.json"
    payload = {**meta, "n_events": n_events, "n_feature_rows": len(accum)}
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("[exp28-causal-rank] wrote %s rows=%d n_events=%d", csv_path, len(accum), n_events)


def merge_workers(out_dir: Path, *, n_workers: int, top_n: int) -> Path:
    total_events = 0
    merged: dict[tuple[int, int], FeatureAccumulator] = defaultdict(FeatureAccumulator)
    meta_rows = []
    for worker in range(n_workers):
        meta_path = out_dir / f"causal_feature_scores_w{worker}.meta.json"
        if not meta_path.exists():
            log.warning("[exp28-causal-rank] missing worker metadata %s", meta_path)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_events = int(meta.get("n_events", 0))
        total_events += n_events
        meta_rows.append(meta)
        csv_path = out_dir / f"causal_feature_scores_w{worker}.csv"
        if not csv_path.exists():
            log.warning("[exp28-causal-rank] missing worker scores %s", csv_path)
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (int(row["layer"]), int(row["latent_id"]))
                acc = merged[key]
                acc.score_sum += float(row["score_sum"])
                acc.score_pos_sum += float(row["score_pos_sum"])
                acc.score_abs_sum += float(row["score_abs_sum"])
                acc.native_attr_sum += float(row["native_attr_sum"])
                acc.ptup_attr_sum += float(row["ptup_attr_sum"])
                acc.active_native_count += int(row["active_native_count"])
                acc.active_ptup_count += int(row["active_ptup_count"])
                acc.active_union_count += int(row["active_union_count"])
    if total_events <= 0:
        raise RuntimeError(f"No causal-rank events available to merge in {out_dir}")

    rows: list[dict[str, Any]] = []
    for (layer, latent_id), acc in merged.items():
        rows.append(
            {
                "layer": layer,
                "latent_id": latent_id,
                "score_sum": acc.score_sum,
                "score_mean": acc.score_sum / total_events,
                "score_pos_mean": acc.score_pos_sum / total_events,
                "score_abs_mean": acc.score_abs_sum / total_events,
                "native_attr_mean": acc.native_attr_sum / total_events,
                "ptup_attr_mean": acc.ptup_attr_sum / total_events,
                "active_native_count": acc.active_native_count,
                "active_ptup_count": acc.active_ptup_count,
                "active_union_count": acc.active_union_count,
                "active_union_rate": acc.active_union_count / total_events,
                "n_events": total_events,
            }
        )
    rows.sort(key=lambda row: float(row["score_mean"]), reverse=True)
    fields = list(rows[0].keys()) if rows else []
    final_path = out_dir / "causal_feature_scores.csv"
    with final_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    top_path = out_dir / "causal_top_features.csv"
    positive = [row for row in rows if float(row["score_mean"]) > 0.0][:top_n]
    with top_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(positive)

    summary = {
        "n_events": total_events,
        "n_features_with_nonzero_activity": len(rows),
        "n_positive_features": len([row for row in rows if float(row["score_mean"]) > 0.0]),
        "top_n_written": len(positive),
        "worker_metadata": meta_rows,
        "score_definition": "grad_attr(U_IT,L_IT) - grad_attr(U_PT,L_IT)",
    }
    (out_dir / "causal_feature_scores_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("[exp28-causal-rank] merged %d features over %d events -> %s", len(rows), total_events, final_path)
    return final_path


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = get_spec(args.model)
    steering_adapter = get_steering_adapter(args.model)
    pt_model, pt_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "pt"),
        args.device,
        multi_gpu=spec.multi_gpu,
    )
    it_model, it_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "it"),
        args.device,
        multi_gpu=spec.multi_gpu,
    )
    pt_model.requires_grad_(False)
    it_model.requires_grad_(False)
    models = {"pt": pt_model, "it": it_model}
    tokenizers = {"pt": pt_tokenizer, "it": it_tokenizer}
    real_token_masks = {
        "pt": steering_adapter.real_token_mask(pt_tokenizer, device, pt_model),
        "it": steering_adapter.real_token_mask(it_tokenizer, device, it_model),
    }
    # Validation expects the readout table, even though causal ranking uses the
    # IT model's native logits for the two IT-late cells.
    from src.poc.exp23_midlate_interaction_suite.residual_factorial import _make_readouts

    readouts = _make_readouts(
        models=models,
        tokenizers=tokenizers,
        steering_adapter=steering_adapter,
        real_token_masks=real_token_masks,
    )
    it_layers = steering_adapter.get_layers(it_model)
    pt_layers = steering_adapter.get_layers(pt_model)
    boundary_layer = _late_boundary(args.model)
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
    target_layers = [int(layer) for layer in args.layers] if args.layers else _available_dictionary_layers(args.run_root)
    target_layers = [layer for layer in target_layers if (args.run_root / "dictionaries" / f"layer_{layer}" / "crosscoder.pt").exists()]
    if not target_layers:
        raise RuntimeError("No requested causal-rank layers have crosscoder dictionaries")
    crosscoder_dtype = _dtype_from_name(args.crosscoder_dtype)
    crosscoder_cache: dict[int, BatchTopKCrossCoder] = {}
    accum: dict[tuple[int, int], FeatureAccumulator] = defaultdict(FeatureAccumulator)
    n_events = 0
    failures = 0
    log.info(
        "[exp28-causal-rank] worker=%d/%d prompts=%d layers=%s",
        args.worker_index,
        args.n_workers,
        len(manifest_rows),
        ",".join(str(layer) for layer in target_layers),
    )
    for row_idx, manifest_record in enumerate(manifest_rows):
        prompt_id = str(manifest_record.get("prompt_id"))
        dataset_record = dataset_by_id.get(prompt_id)
        if dataset_record is None:
            log.warning("[exp28-causal-rank] missing dataset record prompt_id=%s", prompt_id)
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
                pt_boundary = _baseline_boundary(
                    model=pt_model,
                    layers=pt_layers,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    boundary_layer=boundary_layer,
                )
                native_attrs = _cell_feature_attrs(
                    model=it_model,
                    layers=it_layers,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    boundary_layer=boundary_layer,
                    donor_boundary_state=None,
                    target_layers=target_layers,
                    run_root=args.run_root,
                    crosscoder_cache=crosscoder_cache,
                    crosscoder_dtype=crosscoder_dtype,
                    y_pt=y_pt,
                    y_it=y_it,
                    use_threshold=not args.no_threshold,
                )
                ptup_attrs = _cell_feature_attrs(
                    model=it_model,
                    layers=it_layers,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    boundary_layer=boundary_layer,
                    donor_boundary_state=pt_boundary,
                    target_layers=target_layers,
                    run_root=args.run_root,
                    crosscoder_cache=crosscoder_cache,
                    crosscoder_dtype=crosscoder_dtype,
                    y_pt=y_pt,
                    y_it=y_it,
                    use_threshold=not args.no_threshold,
                )
                keys = set(native_attrs) | set(ptup_attrs)
                for key in keys:
                    native = float(native_attrs.get(key, 0.0))
                    ptup = float(ptup_attrs.get(key, 0.0))
                    score = native - ptup
                    acc = accum[key]
                    acc.score_sum += score
                    acc.score_pos_sum += max(score, 0.0)
                    acc.score_abs_sum += abs(score)
                    acc.native_attr_sum += native
                    acc.ptup_attr_sum += ptup
                    acc.active_native_count += int(key in native_attrs)
                    acc.active_ptup_count += int(key in ptup_attrs)
                    acc.active_union_count += 1
                n_events += 1
            except Exception as exc:
                failures += 1
                log.exception("[exp28-causal-rank] prompt=%s event=%s failed: %s", prompt_id, event_kind, exc)
        if (row_idx + 1) % 5 == 0:
            log.info("[exp28-causal-rank] worker=%d processed %d/%d prompts", args.worker_index, row_idx + 1, len(manifest_rows))
    if n_events == 0:
        raise RuntimeError(f"Exp28 causal-rank worker {args.worker_index} found no valid events; failures={failures}")
    _write_worker_outputs(
        out_dir=args.out_dir,
        worker_index=args.worker_index,
        accum=accum,
        n_events=n_events,
        meta={
            "model": args.model,
            "dataset": str(args.dataset),
            "prompt_mode": args.prompt_mode,
            "event_kinds": list(args.event_kinds),
            "skip_prompts": int(args.skip_prompts),
            "n_prompts": int(args.n_prompts),
            "layers": target_layers,
            "worker_index": int(args.worker_index),
            "n_workers": int(args.n_workers),
            "failures": int(failures),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="llama31_8b")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--prompt-mode", choices=["raw_shared", "native"], default="raw_shared")
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=["first_diff"])
    parser.add_argument("--n-prompts", type=int, default=128)
    parser.add_argument("--skip-prompts", type=int, default=0)
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--crosscoder-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--no-threshold", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--top-n", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.run_root / "feature_stats"
    if not args.dataset.exists() and args.dataset.name == "eval_dataset_v2_holdout_0600_1199.jsonl":
        args.dataset = Path("data/eval_dataset_v2.jsonl")
    return args


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_workers(args.out_dir, n_workers=args.n_workers, top_n=args.top_n)
        return
    run_worker(args)


if __name__ == "__main__":
    main()
