"""Collect Exp26 residual-opposition mediation cells."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.boundary import BoundaryStatePatch, LayerResidualCapture
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    DEFAULT_EVENT_KINDS,
    DEFAULT_EXP20_FALLBACK_ROOT,
    DEFAULT_EXP20_ROOT,
    _baseline_forward_with_boundary,
    _cell_readout_payload,
    _dataset_lookup,
    _json_rows,
    _late_boundary,
    _load_manifest_records,
    _make_readouts,
    _prefix_ids_for_event,
    _unique_events,
    _validate_tokenizers_and_tokens,
)
from src.poc.exp26_residual_opposition_mediation import DEFAULT_VARIANTS, EXPERIMENT
from src.poc.exp26_residual_opposition_mediation.hooks import LateMlpOppositionModifier
from src.poc.exp26_residual_opposition_mediation.variants import expand_variants


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_alpha(path: Path | None, model: str) -> dict[int, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    model_payload = payload.get("models", {}).get(model, payload if payload.get("model") == model else {})
    by_layer = model_payload.get("alpha_by_layer") or payload.get("alpha_by_layer") or {}
    return {int(layer): float(value) for layer, value in by_layer.items()}


@torch.no_grad()
def _forward_variant_cell(
    *,
    model: Any,
    adapter: Any,
    steering_adapter: Any,
    layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    late_layers: list[int],
    variant: str,
    seed: int | None,
    prompt_id: str,
    cell_name: str,
    donor_boundary_state: torch.Tensor | None = None,
    alpha_by_layer: dict[int, float] | None = None,
) -> dict[str, Any]:
    patcher = None
    modifier = None
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    try:
        if donor_boundary_state is not None:
            patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state)
        modifier = LateMlpOppositionModifier(
            model=model,
            steering_adapter=steering_adapter,
            late_layers=late_layers,
            variant=variant,
            seed=seed,
            prompt_id=prompt_id,
            cell_name=cell_name,
            alpha_by_layer=alpha_by_layer,
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        all_residuals = residual_capture.snapshot()
    finally:
        if modifier is not None:
            modifier.close()
        if patcher is not None:
            patcher.close()
        residual_capture.close()
    return {
        "logits": outputs.logits[0, -1, :].detach().float(),
        "final_hidden": all_residuals[-1],
        "residuals": [],
        "patch_n": patcher.n_patches if patcher is not None else 0,
        "patch_input_delta_max_abs": patcher.last_max_abs_input_delta if patcher is not None else None,
        "diagnostics": modifier.summary() if modifier is not None else {},
    }


def _record_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("prompt_id")),
        str(row.get("event_kind")),
        str(row.get("late_target", "it")),
        str(row.get("variant")),
        "none" if row.get("seed") is None else str(row.get("seed")),
    )


def _done_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    if not path.exists():
        return set()
    return {_record_key(row) for row in _json_rows(path)}


def _diagnostics_summary(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cell_diags = {name: payload.get("diagnostics", {}) for name, payload in cells.items()}
    opp_vals = []
    update_vals = []
    post_vals = []
    for diag in cell_diags.values():
        if diag.get("mean_opp_norm_frac") is not None:
            opp_vals.append(diag["mean_opp_norm_frac"])
        if diag.get("mean_update_norm_ratio_after_hook") is not None:
            update_vals.append(diag["mean_update_norm_ratio_after_hook"])
        if diag.get("mean_postres_norm_ratio_after_hook") is not None:
            post_vals.append(diag["mean_postres_norm_ratio_after_hook"])
    from src.poc.exp26_residual_opposition_mediation.variants import finite_mean

    return {
        "cells": cell_diags,
        "mean_opp_norm_frac": finite_mean(opp_vals),
        "mean_update_norm_ratio_after_hook": finite_mean(update_vals),
        "mean_postres_norm_ratio_after_hook": finite_mean(post_vals),
        "patch_n": int(sum(int(payload.get("patch_n") or 0) for payload in cells.values())),
    }


def collect_manifest_record_variants(
    *,
    model_name: str,
    manifest_record: dict[str, Any],
    dataset_record: dict[str, Any],
    prompt_mode: str,
    event_kinds: list[str],
    variant_jobs: list[tuple[str, int | None]],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    readouts: dict[str, Any],
    steering_adapter: Any,
    device: torch.device,
    alpha_by_layer: dict[int, float],
    late_target: str,
) -> list[dict[str, Any]]:
    prompt_id = str(manifest_record.get("prompt_id"))
    boundary_layer = _late_boundary(model_name)
    late_layers = list(range(boundary_layer, len(steering_adapter.get_layers(models["it"]))))
    pt_layers = steering_adapter.get_layers(models["pt"])
    it_layers = steering_adapter.get_layers(models["it"])
    adapter = steering_adapter.adapter
    raw_prompt = get_prompt_for_variant(
        dataset_record,
        variant="pt",
        tokenizer=tokenizers["pt"],
        apply_chat_template=False,
    )
    prompt_ids = tokenizers["pt"].encode(raw_prompt, add_special_tokens=True)
    out: list[dict[str, Any]] = []

    for event_kind, event in _unique_events(manifest_record, event_kinds):
        if "duplicate_of" in event:
            continue
        y_pt = int(event["pt_token"]["token_id"])
        y_it = int(event["it_token"]["token_id"])
        prefix_ids = _prefix_ids_for_event(manifest_record, event)
        validation = _validate_tokenizers_and_tokens(
            model_name=model_name,
            prompt_mode=prompt_mode,
            dataset_record=dataset_record,
            prefix_ids=prefix_ids,
            y_pt=y_pt,
            y_it=y_it,
            tokenizers=tokenizers,
            readouts=readouts,
        )
        if not validation.get("ok"):
            for variant, seed in variant_jobs:
                out.append(
                    {
                        "experiment": EXPERIMENT,
                        "model": model_name,
                        "prompt_mode": prompt_mode,
                        "prompt_id": prompt_id,
                        "event_kind": event_kind,
                        "late_target": late_target,
                        "variant": variant,
                        "seed": seed,
                        "valid": False,
                        "validation": validation,
                        "event": event,
                        "cells": {},
                        "diagnostics": {},
                    }
                )
            continue

        full_ids = prompt_ids + prefix_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        pt_baseline = None
        it_baseline = None
        if late_target == "it":
            pt_baseline = _baseline_forward_with_boundary(
                model=models["pt"],
                adapter=adapter,
                layers=pt_layers,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                collect_trajectories=False,
            )
        elif late_target == "pt":
            it_baseline = _baseline_forward_with_boundary(
                model=models["it"],
                adapter=adapter,
                layers=it_layers,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                collect_trajectories=False,
            )
        else:
            raise ValueError(f"unsupported late_target={late_target}")
        for variant, seed in variant_jobs:
            if variant == "ptlevel_opp" and not alpha_by_layer:
                raise RuntimeError("ptlevel_opp requested but no alpha_by_layer values were provided")
            if late_target == "it":
                assert pt_baseline is not None
                raw_cells = {
                    "U_PT__L_IT_variant": _forward_variant_cell(
                        model=models["it"],
                        adapter=adapter,
                        steering_adapter=steering_adapter,
                        layers=it_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        late_layers=late_layers,
                        variant=variant,
                        seed=seed,
                        prompt_id=prompt_id,
                        cell_name="U_PT__L_IT_variant",
                        donor_boundary_state=pt_baseline["boundary_state"],
                        alpha_by_layer=alpha_by_layer if variant == "ptlevel_opp" else None,
                    ),
                    "U_IT__L_IT_variant": _forward_variant_cell(
                        model=models["it"],
                        adapter=adapter,
                        steering_adapter=steering_adapter,
                        layers=it_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        late_layers=late_layers,
                        variant=variant,
                        seed=seed,
                        prompt_id=prompt_id,
                        cell_name="U_IT__L_IT_variant",
                        donor_boundary_state=None,
                        alpha_by_layer=alpha_by_layer if variant == "ptlevel_opp" else None,
                    ),
                }
                host_variant = "it"
            else:
                assert it_baseline is not None
                raw_cells = {
                    "U_PT__L_PT_variant": _forward_variant_cell(
                        model=models["pt"],
                        adapter=adapter,
                        steering_adapter=steering_adapter,
                        layers=pt_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        late_layers=late_layers,
                        variant=variant,
                        seed=seed,
                        prompt_id=prompt_id,
                        cell_name="U_PT__L_PT_variant",
                        donor_boundary_state=None,
                        alpha_by_layer=alpha_by_layer if variant == "ptlevel_opp" else None,
                    ),
                    "U_IT__L_PT_variant": _forward_variant_cell(
                        model=models["pt"],
                        adapter=adapter,
                        steering_adapter=steering_adapter,
                        layers=pt_layers,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        boundary_layer=boundary_layer,
                        late_layers=late_layers,
                        variant=variant,
                        seed=seed,
                        prompt_id=prompt_id,
                        cell_name="U_IT__L_PT_variant",
                        donor_boundary_state=it_baseline["boundary_state"],
                        alpha_by_layer=alpha_by_layer if variant == "ptlevel_opp" else None,
                    ),
                }
                host_variant = "pt"
            cells = {
                cell_name: _cell_readout_payload(
                    cell_name=cell_name,
                    cell=cell_payload,
                    host_variant=host_variant,
                    readouts=readouts,
                    y_pt=y_pt,
                    y_it=y_it,
                    collect_trajectories=False,
                )
                for cell_name, cell_payload in raw_cells.items()
            }
            for cell_name, cell_payload in raw_cells.items():
                cells[cell_name]["diagnostics"] = cell_payload.get("diagnostics", {})
            out.append(
                {
                    "experiment": EXPERIMENT,
                    "model": model_name,
                    "prompt_mode": prompt_mode,
                    "prompt_id": prompt_id,
                    "event_kind": event_kind,
                    "late_target": late_target,
                    "variant": variant,
                    "seed": seed,
                    "valid": True,
                    "event": event,
                    "validation": validation,
                    "boundary_layer": boundary_layer,
                    "late_layers": late_layers,
                    "prefix_length": len(prefix_ids),
                    "full_length": len(full_ids),
                    "y_pt": y_pt,
                    "y_it": y_it,
                    "pt_token_text": validation.get("pt_token_text"),
                    "it_token_text": validation.get("it_token_text"),
                    "cells": cells,
                    "diagnostics": _diagnostics_summary(cells),
                }
            )
    return out


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
    variant_jobs = expand_variants(args.variants, args.rand_seeds)
    alpha_by_layer = _load_alpha(args.ptlevel_alpha_path, args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"records_w{args.worker_index}.jsonl.gz"
    done = _done_keys(out_path)
    log.info(
        "[exp26] %s/%s worker %d/%d rows=%d variants=%s resume=%d",
        args.prompt_mode,
        args.model,
        args.worker_index,
        args.n_workers,
        len(manifest_rows),
        variant_jobs,
        len(done),
    )
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for idx, manifest_record in enumerate(manifest_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                log.warning("[exp26] missing dataset record for prompt_id=%s", prompt_id)
                continue
            records = collect_manifest_record_variants(
                model_name=args.model,
                manifest_record=manifest_record,
                dataset_record=dataset_record,
                prompt_mode=args.prompt_mode,
                event_kinds=args.event_kinds,
                variant_jobs=variant_jobs,
                models=models,
                tokenizers=tokenizers,
                readouts=readouts,
                steering_adapter=steering_adapter,
                device=device,
                alpha_by_layer=alpha_by_layer,
                late_target=args.late_target,
            )
            for record in records:
                key = _record_key(record)
                if key in done:
                    continue
                fout.write(json.dumps(record, separators=(",", ":")) + "\n")
                done.add(key)
            fout.flush()
            if (idx + 1) % 5 == 0:
                log.info("[exp26] %s/%s %d/%d manifest rows", args.prompt_mode, args.model, idx + 1, len(manifest_rows))


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    seen: set[tuple[str, str, str, str, str]] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(n_workers):
            path = out_dir / f"records_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp26] missing worker file %s", path)
                continue
            for row in _json_rows(path):
                key = _record_key(row)
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp26 residual-opposition mediation records.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["raw_shared", "native"], default="raw_shared")
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=["first_diff"])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--rand-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--ptlevel-alpha-path", type=Path, default=None)
    parser.add_argument("--late-target", choices=["it", "pt"], default="it")
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
