"""Collect Exp36 off-manifold validation records.

Exp36 extends the Exp23 residual-state x late-stack factorial with a
PT-to-IT boundary interpolation path, same-norm label-destroyed random
controls, and host-state anomaly features.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    DEFAULT_EXP20_FALLBACK_ROOT,
    DEFAULT_EXP20_ROOT,
    DEFAULT_EVENT_KINDS,
    ReadoutBundle,
    _baseline_forward_with_boundary,
    _dataset_lookup,
    _done_prompt_ids,
    _find_manifest,
    _forward_cell,
    _late_boundary,
    _load_manifest_records,
    _logits_from_hidden,
    _make_readouts,
    _prefix_ids_for_event,
    _rank,
    _safe_float,
    _token_text,
    _trajectory_payload,
    _unique_events,
    _validate_tokenizers_and_tokens,
)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_MODELS = ("gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b")
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
READOUT_NAMES = ("common_it", "common_pt")
ANOMALY_FEATURES = ("seq_rms", "last_rms", "seq_std", "last_std")


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _alpha_label(alpha: float) -> str:
    return f"{float(alpha):.2f}"


def _cell_name(alpha: float, host_variant: str) -> str:
    return f"alpha_{_alpha_label(alpha)}_L_{host_variant.upper()}"


def _random_cell_name(seed_index: int, host_variant: str) -> str:
    return f"random_{seed_index}_L_{host_variant.upper()}"


def _stable_seed(*parts: object) -> int:
    joined = "\x1f".join(str(part) for part in parts)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).item())


def _finite_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _simple_boundary_stats(state: torch.Tensor) -> dict[str, float | None]:
    full = state.detach().float()
    last = full[0, -1, :]
    return {
        "seq_rms": _finite_float(float(torch.sqrt(torch.mean(full.square())).item())),
        "last_rms": _finite_float(float(torch.sqrt(torch.mean(last.square())).item())),
        "seq_mean": _finite_float(float(full.mean().item())),
        "last_mean": _finite_float(float(last.mean().item())),
        "seq_std": _finite_float(float(full.std(unbiased=False).item())),
        "last_std": _finite_float(float(last.std(unbiased=False).item())),
    }


def _boundary_anomaly_raw(
    *,
    state: torch.Tensor,
    host_native_state: torch.Tensor,
    u_pt: torch.Tensor,
    u_it: torch.Tensor,
) -> dict[str, float | None]:
    stats = _simple_boundary_stats(state)
    full = state.detach().float()
    host = host_native_state.detach().float()
    last = full[0, -1, :]
    host_last = host[0, -1, :]
    delta_last = last - host_last
    pt_it_last = u_it.detach().float()[0, -1, :] - u_pt.detach().float()[0, -1, :]
    pt_it_full = u_it.detach().float() - u_pt.detach().float()
    host_delta_full = full - host
    denom_last = _norm(pt_it_last)
    denom_full = _norm(pt_it_full)
    last_norm = _norm(last)
    host_last_norm = _norm(host_last)
    cosine = None
    if last_norm > 0 and host_last_norm > 0:
        cosine = float(torch.dot(last, host_last).item() / (last_norm * host_last_norm))
    stats.update(
        {
            "cos_to_host_native_last": _finite_float(cosine) if cosine is not None else None,
            "l2_to_host_native_last": _finite_float(_norm(delta_last)),
            "relative_l2_last_prompt": _finite_float(_norm(delta_last) / denom_last) if denom_last > 0 else None,
            "delta_norm_ratio": _finite_float(_norm(host_delta_full) / denom_full) if denom_full > 0 else None,
        }
    )
    return stats


def _logits_for_readout(cell: dict[str, Any], bundle: ReadoutBundle) -> torch.Tensor:
    if cell.get("final_hidden") is None:
        logits = cell["logits"].clone().to(bundle.real_token_mask.device).float()
        logits[~bundle.real_token_mask.to(logits.device)] = float("-inf")
        return logits
    return _logits_from_hidden(cell["final_hidden"], bundle)


def _readout_payload(
    *,
    cell: dict[str, Any],
    readouts: dict[str, ReadoutBundle],
    y_pt: int,
    y_it: int,
    collect_trajectories: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for readout_name in READOUT_NAMES:
        bundle = readouts[readout_name]
        logits = _logits_for_readout(cell, bundle)
        probs = torch.softmax(logits, dim=-1)
        top_values, top_ids = torch.topk(logits, k=2)
        final_top1 = int(top_ids[0].item())
        margin = float((logits[y_it] - logits[y_pt]).item())
        choice = "it" if final_top1 == y_it else ("pt" if final_top1 == y_pt else "other")
        safe_probs = probs[torch.isfinite(logits)]
        entropy = float((-(safe_probs * torch.log(safe_probs.clamp_min(1e-45)))).sum().item())
        payload: dict[str, Any] = {
            "it_vs_pt_margin": _safe_float(margin),
            "it_logit": _safe_float(float(logits[y_it].item())),
            "pt_logit": _safe_float(float(logits[y_pt].item())),
            "final_top1_id": final_top1,
            "token_choice_class": choice,
            "entropy": _safe_float(entropy),
            "top1_confidence": _safe_float(float(probs[final_top1].item())),
            "top1_top2_margin": _safe_float(float((top_values[0] - top_values[1]).item())),
            "it_rank": _rank(logits, y_it),
            "pt_rank": _rank(logits, y_pt),
        }
        if collect_trajectories and cell.get("residuals"):
            payload["trajectory"] = _trajectory_payload(cell["residuals"], bundle)
        else:
            payload["trajectory"] = {
                "late_kl_mean": None,
                "remaining_adj_js": None,
                "future_top1_flips": None,
                "top5_churn": None,
                "top1_path_metrics": {},
            }
        out[readout_name] = payload
    return out


def _decorate_cell(
    *,
    cell: dict[str, Any],
    host_variant: str,
    upstream_state: torch.Tensor,
    baselines: dict[str, dict[str, Any]],
    readouts: dict[str, ReadoutBundle],
    y_pt: int,
    y_it: int,
    collect_trajectories: bool,
    alpha: float | None = None,
    random_seed_index: int | None = None,
    random_control: str | None = None,
) -> dict[str, Any]:
    host_native_state = baselines[host_variant]["boundary_state"]
    u_pt = baselines["pt"]["boundary_state"]
    u_it = baselines["it"]["boundary_state"]
    payload: dict[str, Any] = {
        "host_variant": host_variant,
        "readouts": _readout_payload(
            cell=cell,
            readouts=readouts,
            y_pt=y_pt,
            y_it=y_it,
            collect_trajectories=collect_trajectories,
        ),
        "anomaly_raw": _boundary_anomaly_raw(
            state=upstream_state,
            host_native_state=host_native_state,
            u_pt=u_pt,
            u_it=u_it,
        ),
        "patch_n": cell.get("patch_n", 0),
        "patch_input_delta_max_abs": cell.get("patch_input_delta_max_abs"),
    }
    if alpha is not None:
        payload["alpha"] = float(alpha)
    if random_seed_index is not None:
        payload["random_seed_index"] = int(random_seed_index)
        payload["random_control"] = random_control
    return payload


@torch.no_grad()
def _run_host_cell(
    *,
    host_variant: str,
    upstream_state: torch.Tensor,
    models: dict[str, Any],
    adapter: Any,
    layer_map: dict[str, list[torch.nn.Module]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    collect_trajectories: bool,
    baselines: dict[str, dict[str, Any]],
    native_reuse: bool,
) -> dict[str, Any]:
    if native_reuse:
        return baselines[host_variant]
    return _forward_cell(
        model=models[host_variant],
        adapter=adapter,
        layers=layer_map[host_variant],
        input_ids=input_ids,
        attention_mask=attention_mask,
        boundary_layer=boundary_layer,
        donor_boundary_state=upstream_state,
        collect_trajectories=collect_trajectories,
    )


def _signed_permutation_control(
    *,
    u_pt: torch.Tensor,
    u_it: torch.Tensor,
    model_name: str,
    prompt_id: str,
    seed_index: int,
) -> torch.Tensor:
    delta = (u_it - u_pt).detach().float()
    d_model = int(delta.shape[-1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed("exp36", "signed_permutation", model_name, prompt_id, seed_index))
    perm = torch.randperm(d_model, generator=generator, device="cpu").to(delta.device)
    signs = torch.randint(0, 2, (d_model,), generator=generator, device="cpu").to(delta.device)
    signs = signs.to(dtype=delta.dtype).mul_(2.0).sub_(1.0)
    transformed = delta.index_select(dim=-1, index=perm) * signs.view(1, 1, -1)
    return u_pt.detach().float() + transformed


def _first_event(
    manifest_record: dict[str, Any],
    event_kinds: list[str],
) -> tuple[str, dict[str, Any]] | None:
    for kind, event in _unique_events(manifest_record, event_kinds):
        if isinstance(event, dict) and "duplicate_of" not in event:
            return kind, event
    return None


def collect_manifest_record(
    *,
    model_name: str,
    manifest_record: dict[str, Any],
    dataset_record: dict[str, Any],
    prompt_mode: str,
    event_kinds: list[str],
    alphas: list[float],
    n_random: int,
    random_control: str,
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    readouts: dict[str, ReadoutBundle],
    steering_adapter: Any,
    device: torch.device,
    collect_trajectories: bool,
) -> dict[str, Any]:
    prompt_id = str(manifest_record.get("prompt_id"))
    pt_layers = steering_adapter.get_layers(models["pt"])
    it_layers = steering_adapter.get_layers(models["it"])
    boundary_layer = _late_boundary(model_name)
    n_layers = min(len(pt_layers), len(it_layers))
    if boundary_layer < 0 or boundary_layer >= n_layers:
        raise ValueError(f"Boundary layer {boundary_layer} outside {model_name} layer range")
    layer_map = {"pt": pt_layers, "it": it_layers}
    adapter = steering_adapter.adapter
    event_pair = _first_event(manifest_record, event_kinds)
    if event_pair is None:
        return {
            "experiment": "exp36_offmanifold_validation",
            "model": model_name,
            "prompt_id": prompt_id,
            "prompt_mode": prompt_mode,
            "valid": False,
            "validation": {"ok": False, "reason": "missing_requested_event"},
            "cells": {},
        }
    event_kind, event = event_pair
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
    boundary_info = {
        "boundary_layer": int(boundary_layer),
        "downstream_stack": f"layers_{int(boundary_layer)}_{int(n_layers) - 1}_plus_readout",
        "boundary_source": "depth_ablation_windows",
        "boundary_mode": "full_late",
    }
    if not validation.get("ok"):
        return {
            "experiment": "exp36_offmanifold_validation",
            "model": model_name,
            "prompt_id": prompt_id,
            "prompt_mode": prompt_mode,
            "event_kind": event_kind,
            "event": event,
            "valid": False,
            "validation": validation,
            **boundary_info,
            "cells": {},
        }

    raw_prompt = get_prompt_for_variant(
        dataset_record,
        variant="pt",
        tokenizer=tokenizers["pt"],
        apply_chat_template=False,
    )
    prompt_ids = tokenizers["pt"].encode(raw_prompt, add_special_tokens=True)
    full_ids = prompt_ids + prefix_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    baselines = {
        "pt": _baseline_forward_with_boundary(
            model=models["pt"],
            adapter=adapter,
            layers=pt_layers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
            collect_trajectories=collect_trajectories,
        ),
        "it": _baseline_forward_with_boundary(
            model=models["it"],
            adapter=adapter,
            layers=it_layers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
            collect_trajectories=collect_trajectories,
        ),
    }
    u_pt = baselines["pt"]["boundary_state"].detach().float()
    u_it = baselines["it"]["boundary_state"].detach().float()

    cells: dict[str, Any] = {}
    for alpha in alphas:
        alpha = float(alpha)
        upstream = (1.0 - alpha) * u_pt + alpha * u_it
        for host_variant in ("pt", "it"):
            native_reuse = (host_variant == "pt" and math.isclose(alpha, 0.0)) or (
                host_variant == "it" and math.isclose(alpha, 1.0)
            )
            raw_cell = _run_host_cell(
                host_variant=host_variant,
                upstream_state=upstream,
                models=models,
                adapter=adapter,
                layer_map=layer_map,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                collect_trajectories=collect_trajectories,
                baselines=baselines,
                native_reuse=native_reuse,
            )
            cells[_cell_name(alpha, host_variant)] = _decorate_cell(
                cell=raw_cell,
                host_variant=host_variant,
                upstream_state=upstream,
                baselines=baselines,
                readouts=readouts,
                y_pt=y_pt,
                y_it=y_it,
                collect_trajectories=collect_trajectories,
                alpha=alpha,
            )

    if random_control != "signed_permutation":
        raise ValueError(f"Unsupported random control: {random_control}")
    for seed_index in range(n_random):
        upstream = _signed_permutation_control(
            u_pt=u_pt,
            u_it=u_it,
            model_name=model_name,
            prompt_id=prompt_id,
            seed_index=seed_index,
        )
        for host_variant in ("pt", "it"):
            raw_cell = _run_host_cell(
                host_variant=host_variant,
                upstream_state=upstream,
                models=models,
                adapter=adapter,
                layer_map=layer_map,
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_layer=boundary_layer,
                collect_trajectories=collect_trajectories,
                baselines=baselines,
                native_reuse=False,
            )
            cells[_random_cell_name(seed_index, host_variant)] = _decorate_cell(
                cell=raw_cell,
                host_variant=host_variant,
                upstream_state=upstream,
                baselines=baselines,
                readouts=readouts,
                y_pt=y_pt,
                y_it=y_it,
                collect_trajectories=collect_trajectories,
                random_seed_index=seed_index,
                random_control=random_control,
            )

    return {
        "experiment": "exp36_offmanifold_validation",
        "model": model_name,
        "prompt_id": prompt_id,
        "prompt_mode": prompt_mode,
        "event_kind": event_kind,
        "event": event,
        "generated_position": int(event["step"]),
        "valid": True,
        "validation": validation,
        **boundary_info,
        "prefix_length": len(prefix_ids),
        "full_length": len(full_ids),
        "pt_token_id": y_pt,
        "it_token_id": y_it,
        "pt_token_text": validation.get("pt_token_text") or _token_text(tokenizers["pt"], y_pt),
        "it_token_text": validation.get("it_token_text") or _token_text(tokenizers["pt"], y_it),
        "alphas": [float(alpha) for alpha in alphas],
        "n_random": int(n_random),
        "random_control": random_control,
        "native_boundary_stats": {
            "pt": _simple_boundary_stats(u_pt),
            "it": _simple_boundary_stats(u_it),
        },
        "cells": cells,
    }


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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"records_w{args.worker_index}.jsonl.gz"
    done = _done_prompt_ids(out_path)
    manifest_path = _find_manifest(args.exp20_root, args.exp20_fallback_root, args.prompt_mode, args.model)
    log.info(
        "[exp36] model=%s worker=%d/%d rows=%d done=%d manifest=%s out=%s",
        args.model,
        args.worker_index,
        args.n_workers,
        len(manifest_rows),
        len(done),
        manifest_path,
        out_path,
    )
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for idx, manifest_record in enumerate(manifest_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            if prompt_id in done:
                continue
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                log.warning("[exp36] missing dataset record prompt_id=%s", prompt_id)
                continue
            try:
                result = collect_manifest_record(
                    model_name=args.model,
                    manifest_record=manifest_record,
                    dataset_record=dataset_record,
                    prompt_mode=args.prompt_mode,
                    event_kinds=args.event_kinds,
                    alphas=args.alphas,
                    n_random=args.n_random,
                    random_control=args.random_control,
                    models=models,
                    tokenizers=tokenizers,
                    readouts=readouts,
                    steering_adapter=steering_adapter,
                    device=device,
                    collect_trajectories=not args.no_trajectories,
                )
                fout.write(json.dumps(result, separators=(",", ":")) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[exp36] prompt %s failed: %s", prompt_id, exc)
            if (idx + 1) % 5 == 0:
                log.info("[exp36] %s/%s worker=%d %d/%d prompts", args.prompt_mode, args.model, args.worker_index, idx + 1, len(manifest_rows))


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    seen: set[str] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(n_workers):
            path = out_dir / f"records_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp36] missing worker file %s", path)
                continue
            for row in _json_rows(path):
                prompt_id = str(row.get("prompt_id", ""))
                if prompt_id and prompt_id in seen:
                    continue
                if prompt_id:
                    seen.add(prompt_id)
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    log.info("[exp36] merged %d prompts -> %s", len(seen), merged)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp36 off-manifold validation records.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["raw_shared"], default="raw_shared")
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=["first_diff"])
    parser.add_argument("--alphas", nargs="*", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--random-control", choices=["signed_permutation"], default="signed_permutation")
    parser.add_argument("--no-trajectories", action="store_true")
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
