"""Low-rank boundary-state rescue for Exp48.

This is the held-out split version of the reviewer-facing question: how much of
the missing upstream-conditioned readout can be recovered by adding structured
components of the FT-minus-base boundary-state shift to the Base upstream state
before the descendant late stack?
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

import numpy as np
import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.boundary import BoundaryStatePatch
from src.poc.exp23_midlate_interaction_suite.residual_factorial import (
    DEFAULT_EVENT_KINDS,
    DEFAULT_EXP20_FALLBACK_ROOT,
    DEFAULT_EXP20_ROOT,
    _baseline_forward_with_boundary,
    _dataset_lookup,
    _find_manifest,
    _forward_cell,
    _load_manifest_records,
    _prefix_ids_for_event,
    _rank,
    _token_text,
    _unique_events,
)
from src.poc.exp48_static_chimera_sequence_validation.config import (
    DEFAULT_MODELS,
    RESCUE_ALPHAS,
    RESCUE_KS,
    prompt_split,
    wrong_descendant_for,
)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _stable_int(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _fit_path(out_dir: Path, model: str, boundary: int) -> Path:
    return out_dir / "rescue_fits" / model / f"boundary_{boundary}" / "fit.npz"


def _fit_meta_path(out_dir: Path, model: str, boundary: int) -> Path:
    return out_dir / "rescue_fits" / model / f"boundary_{boundary}" / "fit_meta.json"


def _event_rows(
    *,
    exp20_root: Path,
    exp20_fallback_root: Path | None,
    prompt_mode: str,
    model: str,
    n_examples: int | None,
    event_kinds: list[str],
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    rows = _load_manifest_records(
        exp20_root=exp20_root,
        exp20_fallback_root=exp20_fallback_root,
        prompt_mode=prompt_mode,
        model=model,
        n_examples=n_examples,
        worker_index=0,
        n_workers=1,
    )
    out = []
    for manifest_record in rows:
        for kind, event in _unique_events(manifest_record, event_kinds):
            if "duplicate_of" in event:
                continue
            out.append((manifest_record, kind, event))
    return out


def _load_pair(model: str, device: str) -> tuple[Any, Any, Any, Any, Any]:
    spec = get_spec(model)
    steering_adapter = get_steering_adapter(model)
    base_model, base_tokenizer = load_model_and_tokenizer(model_id_for_variant(spec, "pt"), device, multi_gpu=spec.multi_gpu)
    ft_model, ft_tokenizer = load_model_and_tokenizer(model_id_for_variant(spec, "it"), device, multi_gpu=spec.multi_gpu)
    return steering_adapter, base_model, base_tokenizer, ft_model, ft_tokenizer


@torch.no_grad()
def _state_for_event(
    *,
    model_name: str,
    manifest_record: dict[str, Any],
    dataset_record: dict[str, Any],
    event: dict[str, Any],
    boundary: int,
    steering_adapter: Any,
    base_model: Any,
    base_tokenizer: Any,
    ft_model: Any,
    device: torch.device,
) -> dict[str, Any] | None:
    raw_prompt = get_prompt_for_variant(dataset_record, variant="pt", tokenizer=base_tokenizer, apply_chat_template=False)
    prompt_ids = base_tokenizer.encode(raw_prompt, add_special_tokens=True)
    prefix_ids = _prefix_ids_for_event(manifest_record, event)
    full_ids = prompt_ids + prefix_ids
    if not full_ids:
        return None
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    adapter = steering_adapter.adapter
    base_layers = steering_adapter.get_layers(base_model)
    ft_layers = steering_adapter.get_layers(ft_model)
    base = _baseline_forward_with_boundary(
        model=base_model,
        adapter=adapter,
        layers=base_layers,
        input_ids=input_ids,
        attention_mask=attention_mask,
        boundary_layer=boundary,
        collect_trajectories=False,
    )
    ft = _baseline_forward_with_boundary(
        model=ft_model,
        adapter=adapter,
        layers=ft_layers,
        input_ids=input_ids,
        attention_mask=attention_mask,
        boundary_layer=boundary,
        collect_trajectories=False,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "base_boundary": base["boundary_state"],
        "ft_boundary": ft["boundary_state"],
        "base_logits": base["logits"],
        "ft_logits": ft["logits"],
        "pt_token_id": int(event["pt_token"]["token_id"]),
        "it_token_id": int(event["it_token"]["token_id"]),
        "prefix_len": len(prefix_ids),
        "full_len": len(full_ids),
    }


def fit_boundary_states(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    steering_adapter, base_model, base_tokenizer, ft_model, _ft_tokenizer = _load_pair(args.model, args.device)
    dataset_by_id = _dataset_lookup(args.dataset)
    event_rows = _event_rows(
        exp20_root=args.exp20_root,
        exp20_fallback_root=args.exp20_fallback_root,
        prompt_mode=args.prompt_mode,
        model=args.model,
        n_examples=args.n_examples,
        event_kinds=args.event_kinds,
    )
    deltas: list[np.ndarray] = []
    keys: list[str] = []
    splits: list[str] = []
    kept = 0
    for idx, (manifest_record, event_kind, event) in enumerate(event_rows):
        prompt_id = str(manifest_record.get("prompt_id"))
        dataset_record = dataset_by_id.get(prompt_id)
        if dataset_record is None:
            continue
        try:
            state = _state_for_event(
                model_name=args.model,
                manifest_record=manifest_record,
                dataset_record=dataset_record,
                event=event,
                boundary=args.boundary,
                steering_adapter=steering_adapter,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                ft_model=ft_model,
                device=device,
            )
            if state is None:
                continue
            delta = (state["ft_boundary"][:, -1, :] - state["base_boundary"][:, -1, :]).detach().float()[0].cpu().numpy()
            deltas.append(delta.astype(np.float32))
            keys.append(f"{prompt_id}:{event_kind}")
            splits.append(prompt_split(prompt_id))
            kept += 1
        except Exception as exc:
            log.warning("[exp48-rescue-fit] skip prompt=%s kind=%s: %s", prompt_id, event_kind, exc)
        if (idx + 1) % 25 == 0:
            log.info("[exp48-rescue-fit] model=%s boundary=%d events=%d kept=%d", args.model, args.boundary, idx + 1, kept)
    if not deltas:
        raise RuntimeError(f"No boundary deltas collected for {args.model} boundary={args.boundary}")
    delta_arr = np.stack(deltas, axis=0).astype(np.float32)
    split_arr = np.asarray(splits)
    train = delta_arr[split_arr == "train"]
    if train.shape[0] < 4:
        raise RuntimeError(f"Need at least 4 train deltas, got {train.shape[0]}")
    mean = train.mean(axis=0)
    centered = train - mean[None, :]
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    max_k = min(args.max_k, vt.shape[0])
    fit_path = _fit_path(args.out_dir, args.model, args.boundary)
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fit_path,
        deltas=delta_arr,
        keys=np.asarray(keys),
        splits=split_arr,
        mean=mean.astype(np.float32),
        components=vt[:max_k].astype(np.float32),
        singular_values=s.astype(np.float32),
    )
    explained = (s**2) / max(float(np.sum(s**2)), 1e-12)
    meta = {
        "experiment": "exp48_static_chimera_sequence_validation",
        "part": "structured_rescue_fit",
        "model": args.model,
        "boundary": int(args.boundary),
        "events_total": len(event_rows),
        "events_kept": kept,
        "train_events": int((split_arr == "train").sum()),
        "heldout_events": int((split_arr == "heldout").sum()),
        "d_model": int(delta_arr.shape[1]),
        "components_saved": int(max_k),
        "explained_variance_first_10": [float(x) for x in explained[:10]],
        "mean_delta_norm": float(np.linalg.norm(mean)),
        "delta_norm_mean": float(np.linalg.norm(delta_arr, axis=1).mean()),
        "delta_norm_std": float(np.linalg.norm(delta_arr, axis=1).std()),
        "fit_path": str(fit_path),
    }
    _fit_meta_path(args.out_dir, args.model, args.boundary).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, **meta}, indent=2))
    return fit_path


def _load_fit(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {
        "deltas": data["deltas"].astype(np.float32),
        "keys": [str(x) for x in data["keys"].tolist()],
        "splits": [str(x) for x in data["splits"].tolist()],
        "mean": data["mean"].astype(np.float32),
        "components": data["components"].astype(np.float32),
        "singular_values": data["singular_values"].astype(np.float32),
    }


def _parse_k(k: str, d_model: int, n_components: int) -> int:
    if k == "full":
        return int(d_model)
    return min(int(k), int(n_components))


def _project(delta: np.ndarray, mean: np.ndarray, components: np.ndarray, k: str) -> np.ndarray:
    if k == "full":
        return delta.astype(np.float32)
    kk = _parse_k(k, delta.shape[0], components.shape[0])
    if kk <= 0:
        return mean.astype(np.float32)
    basis = components[:kk]
    centered = delta - mean
    return (mean + centered @ basis.T @ basis).astype(np.float32)


def _unit_random(shape: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=shape).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        vec[0] = 1.0
        norm = 1.0
    return vec / norm


def _scaled(vec: np.ndarray, target_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec.astype(np.float32)
    return (vec * (float(target_norm) / norm)).astype(np.float32)


def _condition_vectors(
    *,
    key: str,
    delta: np.ndarray,
    fit: dict[str, Any],
    wrong_fit: dict[str, Any] | None,
    ks: list[str],
    alphas: list[float],
) -> list[tuple[str, str, float, np.ndarray]]:
    mean = fit["mean"]
    components = fit["components"]
    by_key = {k: i for i, k in enumerate(fit["keys"])}
    wrong_by_key = {k: i for i, k in enumerate(wrong_fit["keys"])} if wrong_fit is not None else {}
    rows: list[tuple[str, str, float, np.ndarray]] = []
    shuffled_idx = (_stable_int("shuffle", key) % max(1, len(fit["deltas"])))
    shuffled = fit["deltas"][shuffled_idx]
    wrong = None
    if wrong_fit is not None:
        wrong_idx = wrong_by_key.get(key)
        wrong = wrong_fit["deltas"][wrong_idx] if wrong_idx is not None else wrong_fit["mean"]
    for k in ks:
        paired = _project(delta, mean, components, k)
        shuffled_proj = _project(shuffled, mean, components, k)
        wrong_vec = _project(wrong, mean, components, k) if wrong is not None else np.zeros_like(delta)
        for alpha in alphas:
            rows.append(("paired_pca", k, float(alpha), (float(alpha) * paired).astype(np.float32)))
            rows.append(("mean_delta", k, float(alpha), (float(alpha) * mean).astype(np.float32)))
        # Controls at alpha=1.0 by default. The k sweep is retained so reviewers
        # can see whether controls only catch up when given the same subspace.
        target_norm = float(np.linalg.norm(paired))
        rows.extend(
            [
                ("shuffled_delta", k, 1.0, shuffled_proj.astype(np.float32)),
                ("sign_flip", k, 1.0, (-paired).astype(np.float32)),
                ("wrong_descendant", k, 1.0, wrong_vec.astype(np.float32)),
                (
                    "random_full",
                    k,
                    1.0,
                    _scaled(_unit_random(delta.shape[0], _stable_int("random_full", key, k)), target_norm),
                ),
                (
                    "gaussian",
                    k,
                    1.0,
                    _scaled(
                        _unit_random(delta.shape[0], _stable_int("gaussian", key, k)),
                        float(np.linalg.norm(delta)),
                    ),
                ),
            ]
        )
        if k != "full":
            kk = _parse_k(k, delta.shape[0], components.shape[0])
            coeff_rng = np.random.default_rng(_stable_int("span", key, k))
            coeff = coeff_rng.normal(size=kk).astype(np.float32)
            span = coeff @ components[:kk]
            rows.append(("random_delta_span", k, 1.0, _scaled(span, target_norm)))
    return rows


@torch.no_grad()
def _patched_ft_logits_batch(
    *,
    ft_model: Any,
    ft_layers: list[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary: int,
    donor_states: torch.Tensor,
) -> torch.Tensor:
    batch = donor_states.shape[0]
    patcher = BoundaryStatePatch(ft_layers[boundary], donor_states)
    try:
        out = ft_model(
            input_ids=input_ids.repeat(batch, 1),
            attention_mask=attention_mask.repeat(batch, 1),
            use_cache=False,
        )
    finally:
        patcher.close()
    return out.logits[:, -1, :].detach().float()


def _margin(logits: torch.Tensor, it_token: int, pt_token: int) -> float:
    if it_token >= logits.shape[-1] or pt_token >= logits.shape[-1]:
        return float("nan")
    return float((logits[int(it_token)] - logits[int(pt_token)]).detach().cpu().item())


def score_rescue(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    fit = _load_fit(_fit_path(args.fit_root or args.out_dir, args.model, args.boundary))
    wrong_model = args.wrong_model or wrong_descendant_for(args.model, tuple(args.model_set or DEFAULT_MODELS))
    wrong_path = _fit_path(args.fit_root or args.out_dir, wrong_model, args.boundary)
    wrong_fit = _load_fit(wrong_path) if wrong_path.exists() else None
    steering_adapter, base_model, base_tokenizer, ft_model, ft_tokenizer = _load_pair(args.model, args.device)
    dataset_by_id = _dataset_lookup(args.dataset)
    event_rows = _event_rows(
        exp20_root=args.exp20_root,
        exp20_fallback_root=args.exp20_fallback_root,
        prompt_mode=args.prompt_mode,
        model=args.model,
        n_examples=args.n_examples,
        event_kinds=args.event_kinds,
    )
    ft_layers = steering_adapter.get_layers(ft_model)
    out_dir = args.out_dir / "rescue_scores" / args.model / f"boundary_{args.boundary}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rescue_records.jsonl.gz"
    n = 0
    failures = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for idx, (manifest_record, event_kind, event) in enumerate(event_rows):
            prompt_id = str(manifest_record.get("prompt_id"))
            if prompt_split(prompt_id) != "heldout":
                continue
            dataset_record = dataset_by_id.get(prompt_id)
            if dataset_record is None:
                continue
            key = f"{prompt_id}:{event_kind}"
            try:
                state = _state_for_event(
                    model_name=args.model,
                    manifest_record=manifest_record,
                    dataset_record=dataset_record,
                    event=event,
                    boundary=args.boundary,
                    steering_adapter=steering_adapter,
                    base_model=base_model,
                    base_tokenizer=base_tokenizer,
                    ft_model=ft_model,
                    device=device,
                )
                if state is None:
                    continue
                y_pt = int(state["pt_token_id"])
                y_it = int(state["it_token_id"])
                delta = (state["ft_boundary"][:, -1, :] - state["base_boundary"][:, -1, :]).detach().float()[0].cpu().numpy()
                floor = _forward_cell(
                    model=ft_model,
                    adapter=steering_adapter.adapter,
                    layers=ft_layers,
                    input_ids=state["input_ids"],
                    attention_mask=state["attention_mask"],
                    boundary_layer=args.boundary,
                    donor_boundary_state=state["base_boundary"],
                    collect_trajectories=False,
                )
                base_margin = _margin(state["base_logits"], y_it, y_pt)
                floor_margin = _margin(floor["logits"], y_it, y_pt)
                ceiling_margin = _margin(state["ft_logits"], y_it, y_pt)
                base_payload = {
                    "experiment": "exp48_static_chimera_sequence_validation",
                    "record_type": "structured_rescue",
                    "model": args.model,
                    "wrong_model": wrong_model if wrong_fit is not None else None,
                    "prompt_id": prompt_id,
                    "event_kind": event_kind,
                    "category": dataset_record.get("category"),
                    "source": dataset_record.get("source"),
                    "boundary": int(args.boundary),
                    "position": int(event.get("step", 0)),
                    "pt_token_id": y_pt,
                    "it_token_id": y_it,
                    "pt_token_text": _token_text(ft_tokenizer, y_pt),
                    "it_token_text": _token_text(ft_tokenizer, y_it),
                    "base_margin": base_margin,
                    "floor_margin_U_base_L_ft": floor_margin,
                    "ceiling_margin_U_ft_L_ft": ceiling_margin,
                    "native_gap": ceiling_margin - floor_margin,
                    "boundary_state_scope": "last_token",
                }
                for label, margin in (
                    ("base_native", base_margin),
                    ("floor", floor_margin),
                    ("ceiling", ceiling_margin),
                ):
                    fout.write(
                        json.dumps(
                            {
                                **base_payload,
                                "condition": label,
                                "k": None,
                                "alpha": None,
                                "margin_it_minus_pt": margin,
                                "closure_fraction": None,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    n += 1
                vectors = _condition_vectors(
                    key=key,
                    delta=delta,
                    fit=fit,
                    wrong_fit=wrong_fit,
                    ks=list(args.ks),
                    alphas=[float(x) for x in args.alphas],
                )
                for chunk_start in range(0, len(vectors), max(1, args.rescue_batch_size)):
                    chunk = vectors[chunk_start : chunk_start + max(1, args.rescue_batch_size)]
                    donor_states = state["base_boundary"].repeat(len(chunk), 1, 1)
                    vec = torch.tensor(np.stack([v[3] for v in chunk]), device=device, dtype=donor_states.dtype)
                    donor_states[:, -1, :] = donor_states[:, -1, :] + vec
                    logits = _patched_ft_logits_batch(
                        ft_model=ft_model,
                        ft_layers=ft_layers,
                        input_ids=state["input_ids"],
                        attention_mask=state["attention_mask"],
                        boundary=args.boundary,
                        donor_states=donor_states,
                    )
                    for row_idx, (condition, k, alpha, _vec) in enumerate(chunk):
                        margin = _margin(logits[row_idx], y_it, y_pt)
                        gap = ceiling_margin - floor_margin
                        closure = (margin - floor_margin) / gap if math.isfinite(gap) and abs(gap) > 1e-8 else None
                        fout.write(
                            json.dumps(
                                {
                                    **base_payload,
                                    "condition": condition,
                                    "k": k,
                                    "alpha": float(alpha),
                                    "margin_it_minus_pt": margin,
                                    "closure_fraction": closure,
                                    "it_rank": _rank(logits[row_idx], y_it),
                                    "pt_rank": _rank(logits[row_idx], y_pt),
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        n += 1
                fout.flush()
            except Exception as exc:
                failures += 1
                log.exception("[exp48-rescue-score] prompt=%s kind=%s failed: %s", prompt_id, event_kind, exc)
            if (idx + 1) % 20 == 0:
                log.info("[exp48-rescue-score] model=%s boundary=%d events=%d rows=%d failures=%d", args.model, args.boundary, idx + 1, n, failures)
    print(json.dumps({"ok": True, "out": str(out_path), "rows": n, "failures": failures}, indent=2))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fit", "score"], required=True)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--model-set", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--wrong-model", choices=list(MODEL_REGISTRY), default=None)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--prompt-mode", choices=["raw_shared"], default="raw_shared")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--boundary", type=int, default=19)
    parser.add_argument("--n-examples", type=int, default=1400)
    parser.add_argument("--event-kinds", nargs="*", choices=list(DEFAULT_EVENT_KINDS), default=["first_diff"])
    parser.add_argument("--max-k", type=int, default=256)
    parser.add_argument("--ks", nargs="*", default=list(RESCUE_KS))
    parser.add_argument("--alphas", nargs="*", type=float, default=list(RESCUE_ALPHAS))
    parser.add_argument("--rescue-batch-size", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _find_manifest(args.exp20_root, args.exp20_fallback_root, args.prompt_mode, args.model)
    log.info("[exp48-rescue] mode=%s model=%s boundary=%d manifest=%s", args.mode, args.model, args.boundary, manifest)
    if args.mode == "fit":
        fit_boundary_states(args)
    else:
        score_rescue(args)


if __name__ == "__main__":
    main()
