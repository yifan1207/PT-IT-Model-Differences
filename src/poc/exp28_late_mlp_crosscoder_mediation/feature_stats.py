"""Rank Exp28 crosscoder latents and build matched feature sets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant, model_revision_for_variant
from src.poc.cross_model.utils import load_model_and_tokenizer
from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import BatchTopKCrossCoder

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


DEFAULT_EXP20_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early"
)
DEFAULT_EXP20_FALLBACK_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final"
)


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _find_manifest(root: Path, fallback_root: Path | None, prompt_mode: str, model: str) -> Path | None:
    candidates = [
        root / prompt_mode / model / "exp20_validation_records.jsonl",
        root / prompt_mode / model / "exp20_records.jsonl",
    ]
    if fallback_root is not None:
        candidates.extend(
            [
                fallback_root / prompt_mode / model / "exp20_validation_records.jsonl",
                fallback_root / prompt_mode / model / "exp20_records.jsonl",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _event_token_pairs(path: Path, *, event_kind: str, n_events: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for row in _json_rows(path):
        event = (row.get("divergence_events") or {}).get(event_kind)
        if not isinstance(event, dict):
            continue
        pt = event.get("pt_token") or {}
        it = event.get("it_token") or {}
        if pt.get("token_id") is None or it.get("token_id") is None:
            continue
        pairs.append((int(pt["token_id"]), int(it["token_id"])))
        if len(pairs) >= n_events:
            break
    return pairs


@torch.no_grad()
def _mean_margin_vector(
    *,
    model_name: str,
    prompt_mode: str,
    exp20_root: Path,
    exp20_fallback_root: Path | None,
    event_kind: str,
    n_events: int,
    device: torch.device,
) -> torch.Tensor | None:
    manifest = _find_manifest(exp20_root, exp20_fallback_root, prompt_mode, model_name)
    if manifest is None:
        log.warning("[exp28-stats] no Exp20 manifest found; local margin attribution disabled")
        return None
    pairs = _event_token_pairs(manifest, event_kind=event_kind, n_events=n_events)
    if not pairs:
        log.warning("[exp28-stats] no event token pairs found in %s", manifest)
        return None
    spec = get_spec(model_name)
    model, _tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "it"),
        str(device),
        multi_gpu=spec.multi_gpu,
    )
    weight = model.get_output_embeddings().weight.detach().float().to(device)
    diffs = []
    vocab = int(weight.shape[0])
    for pt_id, it_id in pairs:
        if 0 <= pt_id < vocab and 0 <= it_id < vocab:
            diffs.append(weight[it_id] - weight[pt_id])
    if not diffs:
        return None
    vec = torch.stack(diffs, dim=0).mean(dim=0).detach().cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vec


def _cache_tensor(run_root: Path, layer: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(run_root / "cache" / f"layer_{layer}.pt", map_location="cpu", weights_only=False)
    x = torch.stack([payload["pt_mlp"], payload["it_mlp"]], dim=1)
    is_val = payload.get("is_val")
    if is_val is None:
        is_val = torch.zeros(x.shape[0], dtype=torch.bool)
    return x, is_val.bool(), payload


@torch.no_grad()
def _activation_sums(
    *,
    model: BatchTopKCrossCoder,
    x: torch.Tensor,
    device: torch.device,
    batch_tokens: int,
) -> dict[str, torch.Tensor]:
    n = int(x.shape[0])
    dict_size = model.config.dict_size
    sum_pt = torch.zeros(dict_size, dtype=torch.float64)
    sum_it = torch.zeros(dict_size, dtype=torch.float64)
    fire_pt = torch.zeros(dict_size, dtype=torch.float64)
    fire_it = torch.zeros(dict_size, dtype=torch.float64)
    for start in range(0, n, batch_tokens):
        xb = x[start : start + batch_tokens].to(device=device, dtype=torch.float32)
        z_pt = model.encode_branch(xb[:, 0, :], branch=0, use_threshold=False)
        z_it = model.encode_branch(xb[:, 1, :], branch=1, use_threshold=False)
        sum_pt += z_pt.detach().double().sum(dim=0).cpu()
        sum_it += z_it.detach().double().sum(dim=0).cpu()
        fire_pt += (z_pt > 0).detach().double().sum(dim=0).cpu()
        fire_it += (z_it > 0).detach().double().sum(dim=0).cpu()
    denom = max(n, 1)
    return {
        "mean_pt": sum_pt / denom,
        "mean_it": sum_it / denom,
        "freq_pt": fire_pt / denom,
        "freq_it": fire_it / denom,
    }


def _feature_type(row: dict[str, float]) -> str:
    ratio = row["activation_ratio_it_pt"]
    scaling = row["latent_scaling_ratio"]
    if ratio >= 1.5 and scaling >= 1.25 and row["local_margin_attr"] > 0:
        if row["upstream_gating"] > 0:
            return "interaction_candidate"
        return "it_biased"
    if ratio <= (1.0 / 1.5) and scaling <= 0.8:
        return "pt_biased"
    if 0.75 <= ratio <= 1.33 and 0.75 <= scaling <= 1.33:
        return "shared"
    return "mixed"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_optional_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency on remotes
        log.warning("[exp28-stats] pyarrow unavailable, skipping parquet: %s", exc)
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _select_top_examples(
    *,
    run_root: Path,
    rows: list[dict[str, Any]],
    model_name: str,
    top_n_features: int,
    examples_per_feature: int,
    device: torch.device,
    batch_tokens: int,
) -> None:
    if top_n_features <= 0:
        return
    spec = get_spec(model_name)
    tok_kwargs: dict[str, Any] = {"trust_remote_code": True}
    revision = model_revision_for_variant(spec, "pt")
    if revision:
        tok_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id_for_variant(spec, "pt"), **tok_kwargs)
    selected = sorted(rows, key=lambda r: float(r["interaction_score"]), reverse=True)[:top_n_features]
    by_layer: dict[int, list[int]] = defaultdict(list)
    for row in selected:
        by_layer[int(row["layer"])].append(int(row["latent_id"]))
    out_path = run_root / "feature_stats" / "top_feature_examples.jsonl"
    with out_path.open("w", encoding="utf-8") as fout:
        for layer, latent_ids in sorted(by_layer.items()):
            model = BatchTopKCrossCoder.load(
                run_root / "dictionaries" / f"layer_{layer}" / "crosscoder.pt",
                device=device,
            )
            x, _is_val, payload = _cache_tensor(run_root, layer)
            wanted = torch.tensor(sorted(set(latent_ids)), dtype=torch.long, device=device)
            heaps: dict[int, list[tuple[float, int]]] = {int(i): [] for i in wanted.cpu().tolist()}
            for start in range(0, x.shape[0], batch_tokens):
                xb = x[start : start + batch_tokens].to(device=device, dtype=torch.float32)
                z_it = model.encode_branch(xb[:, 1, :], branch=1, use_threshold=False)
                vals = z_it[:, wanted].detach().cpu()
                for col, latent in enumerate(wanted.cpu().tolist()):
                    top = torch.topk(vals[:, col], k=min(examples_per_feature, vals.shape[0]))
                    for value, rel_idx in zip(top.values.tolist(), top.indices.tolist(), strict=False):
                        abs_idx = start + int(rel_idx)
                        heaps[int(latent)].append((float(value), abs_idx))
                        heaps[int(latent)] = sorted(heaps[int(latent)], reverse=True)[:examples_per_feature]
            for latent, examples in heaps.items():
                for value, idx in examples:
                    tok_id = int(payload["token_id"][idx].item())
                    fout.write(
                        json.dumps(
                            {
                                "layer": int(layer),
                                "latent_id": int(latent),
                                "activation": value,
                                "prompt_id": str(payload["prompt_id"][idx]),
                                "token_pos": int(payload["token_pos"][idx].item()),
                                "token_id": tok_id,
                                "token_text": tokenizer.decode(
                                    [tok_id],
                                    skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False,
                                ),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    layers = [int(x) for x in args.layers]
    margin_vec = None
    if not args.skip_logit_attr:
        margin_vec = _mean_margin_vector(
            model_name=args.model,
            prompt_mode=args.prompt_mode,
            exp20_root=args.exp20_root,
            exp20_fallback_root=args.exp20_fallback_root,
            event_kind=args.event_kind,
            n_events=args.n_margin_events,
            device=device,
        )
    rows: list[dict[str, Any]] = []
    for layer in layers:
        ckpt = args.run_root / "dictionaries" / f"layer_{layer}" / "crosscoder.pt"
        model = BatchTopKCrossCoder.load(ckpt, device=device)
        x, is_val, _payload = _cache_tensor(args.run_root, layer)
        train_x = x[~is_val] if bool((~is_val).any().item()) else x
        sums = _activation_sums(
            model=model,
            x=train_x,
            device=device,
            batch_tokens=args.batch_tokens,
        )
        dec = model.decoder.detach().cpu().float()
        norm_pt = dec[:, 0, :].norm(dim=1)
        norm_it = dec[:, 1, :].norm(dim=1)
        cosine = torch.nn.functional.cosine_similarity(dec[:, 0, :], dec[:, 1, :], dim=1)
        if margin_vec is not None:
            attr = dec[:, 1, :] @ margin_vec.float()
        else:
            attr = norm_it.clone()
        for latent in range(model.config.dict_size):
            mean_pt = float(sums["mean_pt"][latent].item())
            mean_it = float(sums["mean_it"][latent].item())
            npt = float(norm_pt[latent].item())
            nit = float(norm_it[latent].item())
            activation_ratio = (mean_it + args.eps) / (mean_pt + args.eps)
            scaling_ratio = ((mean_it * nit) + args.eps) / ((mean_pt * npt) + args.eps)
            gating = mean_it - mean_pt
            local_attr = float(attr[latent].item())
            score = max(gating, 0.0) * max(local_attr, 0.0) * math.log1p(max(scaling_ratio, 0.0))
            row = {
                "model": args.model,
                "layer": int(layer),
                "latent_id": int(latent),
                "mean_activation_pt": mean_pt,
                "mean_activation_it": mean_it,
                "freq_pt": float(sums["freq_pt"][latent].item()),
                "freq_it": float(sums["freq_it"][latent].item()),
                "activation_ratio_it_pt": activation_ratio,
                "latent_scaling_ratio": scaling_ratio,
                "decoder_norm_pt": npt,
                "decoder_norm_it": nit,
                "decoder_cosine_pt_it": float(cosine[latent].item()),
                "local_margin_attr": local_attr,
                "upstream_gating": gating,
                "interaction_score": score,
            }
            row["feature_type"] = _feature_type(row)  # type: ignore[arg-type]
            rows.append(row)
        log.info("[exp28-stats] layer=%d rows=%d", layer, model.config.dict_size)

    stats_dir = args.run_root / "feature_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(stats_dir / "all_latents.csv", rows)
    _write_optional_parquet(stats_dir / "all_latents.parquet", rows)
    top = sorted(rows, key=lambda r: float(r["interaction_score"]), reverse=True)[: args.top_n]
    _write_csv(stats_dir / "top_interaction_candidates.csv", top)
    selected = {
        "model": args.model,
        "layers": layers,
        "ranking": "interaction_score=(mean_it_only-mean_pt_only)_+ * local_margin_attr_+ * log1p(latent_scaling_ratio)",
        "top_interaction_candidates": [
            {"layer": int(r["layer"]), "latent_id": int(r["latent_id"]), "score": float(r["interaction_score"])}
            for r in top
        ],
    }
    (stats_dir / "selected_features.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    _select_top_examples(
        run_root=args.run_root,
        rows=rows,
        model_name=args.model,
        top_n_features=min(args.top_examples_features, len(top)),
        examples_per_feature=args.examples_per_feature,
        device=device,
        batch_tokens=args.batch_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="llama31_8b")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--prompt-mode", choices=["raw_shared", "native"], default="raw_shared")
    parser.add_argument("--event-kind", default="first_diff")
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--n-margin-events", type=int, default=600)
    parser.add_argument("--skip-logit-attr", action="store_true")
    parser.add_argument("--top-n", type=int, default=2000)
    parser.add_argument("--top-examples-features", type=int, default=100)
    parser.add_argument("--examples-per-feature", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
