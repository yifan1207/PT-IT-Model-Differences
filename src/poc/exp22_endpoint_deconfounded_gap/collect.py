"""Collector for Exp22 endpoint-deconfounded convergence-gap records.

For each generated token, this collector computes raw and/or tuned-lens layer
distributions, immediately reduces them to scalar arrays plus top-5 summaries,
and writes strict compact JSONL.  It intentionally does not store full-vocab
logits on disk.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.tuned_lens import _load_probes
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp22_endpoint_deconfounded_gap.metrics import distribution_arrays_from_logits

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DENSE5_MODELS = ("gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b")


@dataclass
class ProbeReadout:
    name: str
    probes: dict[int, torch.nn.Module] | None = None


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    return torch.float32


def _resolve_probe_dir(root: str | Path, model: str, variant: str) -> Path:
    root_path = Path(root)
    candidates = [
        root_path / model / variant,
        root_path / model / "tuned_lens" / variant,
        root_path / "tuned_lens" / model / variant,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No tuned-lens probe dir found for {model}/{variant} under {root_path}; "
        f"checked: {', '.join(str(p) for p in candidates)}"
    )


def _load_readouts(
    *,
    probe_families: list[str],
    tuned_lens_dir: str | None,
    model_name: str,
    variant: str,
    d_model: int,
    n_layers: int,
    device: torch.device,
) -> dict[str, ProbeReadout]:
    readouts: dict[str, ProbeReadout] = {}
    if "raw" in probe_families:
        readouts["raw"] = ProbeReadout(name="raw")
    if "tuned" in probe_families:
        if tuned_lens_dir is None:
            raise ValueError("--tuned-lens-dir is required when --probe-families includes tuned")
        probe_dir = _resolve_probe_dir(tuned_lens_dir, model_name, variant)
        probes = _load_probes(probe_dir, d_model=d_model, device=device)
        if len(probes) != n_layers:
            raise ValueError(
                f"Incomplete tuned-lens probes for {model_name}/{variant}: "
                f"expected {n_layers}, found {len(probes)} at {probe_dir}"
            )
        readouts["tuned"] = ProbeReadout(name="tuned", probes=probes)
        log.info("Loaded %d tuned probes from %s", len(probes), probe_dir)
    return readouts


def _logits_by_layer(
    *,
    residuals: list[torch.Tensor],
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    real_token_mask: torch.Tensor,
    readout: ProbeReadout,
) -> torch.Tensor:
    logits: list[torch.Tensor] = []
    norm_dtype = _module_dtype(final_norm)
    head_dtype = _module_dtype(lm_head)
    for layer_idx, hidden in enumerate(residuals):
        h = hidden.to(device=real_token_mask.device)
        if readout.probes is not None:
            if layer_idx not in readout.probes:
                raise KeyError(f"missing tuned probe for layer {layer_idx}")
            h = readout.probes[layer_idx](h.to(dtype=norm_dtype).view(1, -1)).view(-1)
        normed = final_norm(h.to(dtype=norm_dtype).view(1, 1, -1)).view(-1).to(dtype=head_dtype)
        row = lm_head(normed).float()
        row = row.clone()
        row[~real_token_mask.to(row.device)] = float("-inf")
        logits.append(row)
    return torch.stack(logits, dim=0)


def _read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["prompt_id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    return done


def _record_id(record: dict[str, Any], fallback_index: int) -> str:
    return str(record.get("id") or record.get("record_id") or record.get("prompt_id") or fallback_index)


@torch.no_grad()
def collect_one_prompt(
    *,
    record: dict[str, Any],
    prompt_id: str,
    model,
    tokenizer,
    adapter,
    steering_adapter,
    variant: str,
    device: torch.device,
    max_new_tokens: int,
    readouts: dict[str, ProbeReadout],
    top_k: int,
) -> dict[str, Any]:
    prompt = get_prompt_for_variant(
        record,
        variant=variant,
        tokenizer=tokenizer,
        apply_chat_template=(variant == "it"),
    )
    prompt_mode = "native_chat" if variant == "it" else "raw"
    gen_device = device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(gen_device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(gen_device)
    if input_ids.shape[1] < 1:
        raise ValueError(f"empty prompt after tokenization for {prompt_id}")

    layers = adapter.layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            h = adapter.residual_from_output(output)
            captured[layer_idx] = h[0, -1, :].detach()

        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(len(layers))]
    final_norm = adapter.final_norm(model)
    lm_head = adapter.lm_head(model)
    real_token_mask = steering_adapter.real_token_mask(tokenizer, device, model)
    stop_ids = set(adapter.stop_token_ids(tokenizer))

    probe_payloads: dict[str, dict[str, list[Any]]] = {
        family: {
            "kl_to_final": [],
            "entropy": [],
            "confidence": [],
            "top1_margin": [],
            "top1_ids": [],
            "top5_ids": [],
            "top5_logprobs": [],
            "adjacent_kl": [],
            "adjacent_js": [],
        }
        for family in readouts
    }
    generated_ids: list[int] = []
    past_key_values = None
    next_input_ids = input_ids
    try:
        for _step in range(max_new_tokens):
            captured.clear()
            outputs = model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            residuals = [captured[i] for i in range(len(layers))]
            masked_final_logits = outputs.logits[0, -1, :].float().clone()
            masked_final_logits[~real_token_mask.to(masked_final_logits.device)] = float("-inf")
            next_id = int(torch.argmax(masked_final_logits).item())
            generated_ids.append(next_id)

            for family, readout in readouts.items():
                logits = _logits_by_layer(
                    residuals=residuals,
                    final_norm=final_norm,
                    lm_head=lm_head,
                    real_token_mask=real_token_mask,
                    readout=readout,
                )
                arrays = distribution_arrays_from_logits(logits, top_k=top_k)
                for key, value in arrays.items():
                    probe_payloads[family][key].append(value)

            next_input_ids = torch.tensor([[next_id]], dtype=torch.long, device=gen_device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=gen_device)],
                dim=1,
            )
            if next_id in stop_ids:
                break
    finally:
        for handle in handles:
            handle.remove()

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "prompt_id": prompt_id,
        "model": steering_adapter.model_name,
        "variant": variant,
        "prompt_mode": prompt_mode,
        "n_layers": len(layers),
        "n_steps": len(generated_ids),
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "probe_families": sorted(readouts),
        "probes": probe_payloads,
    }


def merge_worker_outputs(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "records.jsonl.gz"
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker in range(n_workers):
            path = out_dir / f"records_w{worker}.jsonl.gz"
            if not path.exists():
                log.warning("missing worker output: %s", path)
                continue
            with gzip.open(path, "rt", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
    return merged


def run_collect(args: argparse.Namespace) -> None:
    spec = get_spec(args.model)
    if args.model not in DENSE5_MODELS and not args.allow_non_dense:
        raise ValueError(f"Exp22 primary collector excludes non-dense model {args.model}; pass --allow-non-dense to override")
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_path = out_dir / f"records_w{args.worker_index}.jsonl.gz"
    done_ids = _read_done_ids(worker_path)

    model_id = model_id_for_variant(spec, args.variant)
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        device,
        dtype=getattr(torch, args.dtype),
        multi_gpu=spec.multi_gpu,
    )
    adapter = get_steering_adapter(args.model).adapter
    steering_adapter = get_steering_adapter(args.model)
    readouts = _load_readouts(
        probe_families=args.probe_families,
        tuned_lens_dir=args.tuned_lens_dir,
        model_name=args.model,
        variant=args.variant,
        d_model=spec.d_model,
        n_layers=spec.n_layers,
        device=device,
    )
    records = load_dataset(
        args.dataset,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        n_examples=args.n_eval_examples,
    )
    log.info(
        "Collecting Exp22 model=%s variant=%s worker=%d/%d prompts=%d out=%s",
        args.model,
        args.variant,
        args.worker_index,
        args.n_workers,
        len(records),
        worker_path,
    )
    with gzip.open(worker_path, "at", encoding="utf-8") as fout:
        for local_idx, record in enumerate(records):
            global_idx = args.worker_index + local_idx * args.n_workers
            prompt_id = _record_id(record, global_idx)
            if prompt_id in done_ids:
                continue
            try:
                out = collect_one_prompt(
                    record=record,
                    prompt_id=prompt_id,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    steering_adapter=steering_adapter,
                    variant=args.variant,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    readouts=readouts,
                    top_k=args.top_k,
                )
                fout.write(json.dumps(out, ensure_ascii=False, allow_nan=False) + "\n")
                fout.flush()
            except Exception as exc:
                err = {
                    "prompt_id": prompt_id,
                    "model": args.model,
                    "variant": args.variant,
                    "error": repr(exc),
                    "malformed": True,
                }
                fout.write(json.dumps(err, ensure_ascii=False, allow_nan=False) + "\n")
                fout.flush()
                log.exception("failed prompt_id=%s", prompt_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3_4b")
    parser.add_argument("--variant", choices=["pt", "it"], default="pt")
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--n-eval-examples", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--probe-families", nargs="+", choices=["raw", "tuned"], default=["raw", "tuned"])
    parser.add_argument("--tuned-lens-dir", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--allow-non-dense", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merged = merge_worker_outputs(Path(args.out_dir), args.n_workers)
        log.info("Merged workers into %s", merged)
        return
    run_collect(args)


if __name__ == "__main__":
    main()

