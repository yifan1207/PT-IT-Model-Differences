"""Cache paired PT/IT late-MLP activations for Exp28 crosscoder training."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_raw_prompt, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _special_ids(tokenizer: Any) -> set[int]:
    ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    if tokenizer.pad_token_id is not None:
        ids.add(int(tokenizer.pad_token_id))
    return ids


def _tokenize_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in records:
        prompt_id = str(row.get("id", row.get("record_id", len(out))))
        ids = tokenizer.encode(get_raw_prompt(row), add_special_tokens=True)
        if max_seq_len > 0:
            ids = ids[:max_seq_len]
        if len(ids) < 2:
            continue
        out.append({"prompt_id": prompt_id, "input_ids": ids})
    return out


@torch.no_grad()
def _append_pt_greedy_tokens(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    max_seq_len: int,
    batch_size: int,
) -> None:
    if max_new_tokens <= 0:
        return
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if max_seq_len > max_new_tokens:
        prefix_budget = max(1, max_seq_len - max_new_tokens)
        for row in rows:
            row["input_ids"] = row["input_ids"][:prefix_budget]
    old_eos = getattr(model.generation_config, "eos_token_id", None)
    old_forced_eos = getattr(model.generation_config, "forced_eos_token_id", None)
    model.generation_config.eos_token_id = None
    model.generation_config.forced_eos_token_id = None
    try:
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            max_len = max(len(row["input_ids"]) for row in batch)
            padded: list[list[int]] = []
            masks: list[list[int]] = []
            pad_counts: list[int] = []
            for row in batch:
                ids = [int(x) for x in row["input_ids"]]
                n_pad = max_len - len(ids)
                pad_counts.append(n_pad)
                padded.append([int(pad_id)] * n_pad + ids)
                masks.append([0] * n_pad + [1] * len(ids))
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                pad_token_id=int(pad_id),
                eos_token_id=None,
                forced_eos_token_id=None,
            ).detach().cpu()
            for local_idx, row in enumerate(batch):
                seq = generated[local_idx].tolist()
                n_pad = pad_counts[local_idx]
                if n_pad:
                    seq = seq[n_pad:]
                if max_seq_len > 0:
                    seq = seq[:max_seq_len]
                row["input_ids"] = [int(x) for x in seq]
            done = min(start + len(batch), len(rows))
            if done % 25 == 0 or done == len(rows):
                log.info("[exp28-cache] generated prefixes for %d/%d prompts", done, len(rows))
    finally:
        model.generation_config.eos_token_id = old_eos
        model.generation_config.forced_eos_token_id = old_forced_eos


def _make_batch(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    device: torch.device,
    special_ids: set[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    tensors = [torch.tensor(row["input_ids"], dtype=torch.long) for row in rows]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    input_ids = pad_sequence(tensors, batch_first=True, padding_value=int(pad_id)).to(device)
    attention_mask = (input_ids != int(pad_id)).long()
    keep = attention_mask.bool()
    for tok in special_ids:
        keep &= input_ids != int(tok)
    prompt_ids: list[str] = []
    token_pos: list[int] = []
    token_ids: list[int] = []
    keep_cpu = keep.cpu()
    ids_cpu = input_ids.cpu()
    for batch_idx, row in enumerate(rows):
        positions = torch.nonzero(keep_cpu[batch_idx], as_tuple=False).flatten().tolist()
        for pos in positions:
            prompt_ids.append(str(row["prompt_id"]))
            token_pos.append(int(pos))
            token_ids.append(int(ids_cpu[batch_idx, pos].item()))
    return (
        input_ids,
        attention_mask,
        keep,
        prompt_ids,
        torch.tensor(token_pos, dtype=torch.int32),
        torch.tensor(token_ids, dtype=torch.int64),
    )


@torch.no_grad()
def _capture_variant(
    *,
    model: Any,
    layers: list[torch.nn.Module],
    layer_ids: list[int],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    keep_mask: torch.Tensor,
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _args, output):
            if not torch.is_tensor(output):
                raise RuntimeError(f"Expected tensor MLP output at layer {layer_idx}, got {type(output)}")
            captured[layer_idx] = output.detach()[keep_mask].to("cpu", dtype=torch.bfloat16)

        return hook

    try:
        for layer_idx in layer_ids:
            handles.append(layers[layer_idx].mlp.register_forward_hook(make_hook(layer_idx)))
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(layer_ids) - set(captured))
    if missing:
        raise RuntimeError(f"Missing MLP captures for layers {missing}")
    return captured


def _write_layer_cache(
    *,
    out_dir: Path,
    model_name: str,
    layer: int,
    pt_chunks: list[torch.Tensor],
    it_chunks: list[torch.Tensor],
    prompt_ids: list[str],
    token_pos: list[torch.Tensor],
    token_ids: list[torch.Tensor],
    val_fraction: float,
    seed: int,
) -> None:
    pt_mlp = torch.cat(pt_chunks, dim=0)
    it_mlp = torch.cat(it_chunks, dim=0)
    n_tokens = int(pt_mlp.shape[0])
    generator = torch.Generator().manual_seed(seed + int(layer))
    is_val = torch.rand(n_tokens, generator=generator) < float(val_fraction)
    payload = {
        "model": model_name,
        "layer": int(layer),
        "pt_mlp": pt_mlp,
        "it_mlp": it_mlp,
        "prompt_id": prompt_ids,
        "token_pos": torch.cat(token_pos, dim=0),
        "token_id": torch.cat(token_ids, dim=0),
        "is_val": is_val,
        "split": "train_recon_val",
    }
    path = out_dir / f"layer_{layer}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    log.info("[exp28-cache] wrote %s tokens=%d", path, n_tokens)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = get_spec(args.model)
    adapter = get_steering_adapter(args.model)
    layers = [int(x) for x in args.layers]
    if not layers:
        raise ValueError("At least one layer is required")
    bad = [layer for layer in layers if layer < 0 or layer >= spec.n_layers]
    if bad:
        raise ValueError(f"Invalid layers for {args.model}: {bad}")

    records = load_dataset(args.dataset, n_examples=args.n_prompts)

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

    rows = _tokenize_records(records, pt_tokenizer, max_seq_len=args.max_seq_len)
    if args.append_pt_greedy_tokens > 0:
        _append_pt_greedy_tokens(
            model=pt_model,
            tokenizer=pt_tokenizer,
            rows=rows,
            device=device,
            max_new_tokens=args.append_pt_greedy_tokens,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
        )

    pt_layers = adapter.get_layers(pt_model)
    it_layers = adapter.get_layers(it_model)
    special_ids = _special_ids(pt_tokenizer)
    out_dir = args.out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_chunks: dict[int, list[torch.Tensor]] = defaultdict(list)
    it_chunks: dict[int, list[torch.Tensor]] = defaultdict(list)
    prompt_ids_all: list[str] = []
    token_pos_all: list[torch.Tensor] = []
    token_ids_all: list[torch.Tensor] = []
    n_seen = 0

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        input_ids, attention_mask, keep, prompt_ids, token_pos, token_ids = _make_batch(
            batch_rows,
            tokenizer=pt_tokenizer,
            device=device,
            special_ids=special_ids,
        )
        if not prompt_ids:
            continue
        pt_caps = _capture_variant(
            model=pt_model,
            layers=pt_layers,
            layer_ids=layers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            keep_mask=keep,
        )
        it_caps = _capture_variant(
            model=it_model,
            layers=it_layers,
            layer_ids=layers,
            input_ids=input_ids,
            attention_mask=attention_mask,
            keep_mask=keep,
        )
        n_batch_tokens = len(prompt_ids)
        remaining = args.n_tokens - n_seen if args.n_tokens is not None else n_batch_tokens
        take = min(n_batch_tokens, remaining)
        if take <= 0:
            break
        prompt_ids_all.extend(prompt_ids[:take])
        token_pos_all.append(token_pos[:take])
        token_ids_all.append(token_ids[:take])
        for layer in layers:
            pt_chunks[layer].append(pt_caps[layer][:take])
            it_chunks[layer].append(it_caps[layer][:take])
        n_seen += take
        log.info("[exp28-cache] cached tokens=%d target=%s", n_seen, args.n_tokens or "all")
        if args.n_tokens is not None and n_seen >= args.n_tokens:
            break

    if n_seen == 0:
        raise RuntimeError("No activation tokens were cached")
    for layer in layers:
        _write_layer_cache(
            out_dir=out_dir,
            model_name=args.model,
            layer=layer,
            pt_chunks=pt_chunks[layer],
            it_chunks=it_chunks[layer],
            prompt_ids=prompt_ids_all,
            token_pos=token_pos_all,
            token_ids=token_ids_all,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    config = {
        "model": args.model,
        "layers": layers,
        "n_tokens": n_seen,
        "dataset": str(args.dataset),
        "n_prompts": args.n_prompts,
        "append_pt_greedy_tokens": args.append_pt_greedy_tokens,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "val_fraction": args.val_fraction,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="llama31_8b")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--n-prompts", type=int, default=600)
    parser.add_argument("--n-tokens", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--append-pt-greedy-tokens", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
