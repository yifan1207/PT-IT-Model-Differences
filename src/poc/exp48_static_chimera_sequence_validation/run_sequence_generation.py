"""Run Exp48 static chimera sequence generation.

This module evaluates real autoregressive hybrid checkpoints produced by
temporary module swaps. It never writes merged checkpoints: each worker loads a
base/descendant pair and swaps the requested late modules only while generating
one condition.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset, load_model_and_tokenizer
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp23_midlate_interaction_suite.boundary import BoundaryStateCapture, LayerResidualCapture
from src.poc.exp45_behavioral_bridge.metrics import lexical_metrics
from src.poc.exp48_static_chimera_sequence_validation.chimera_loader import (
    interpolated_late_context,
    static_chimera_context,
)
from src.poc.exp48_static_chimera_sequence_validation.config import (
    DEFAULT_MODELS,
    INTERPOLATION_ALPHAS,
    prompt_split,
    wrong_descendant_for,
)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_PAIR_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, Any, Any]] = {}
_WRONG_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_SUPPRESSED_ID_CACHE: dict[tuple[str, int], set[int]] = {}


class CommonVocabLogitsProcessor(LogitsProcessor):
    """Suppress ids that cannot be embedded by all participating models."""

    def __init__(self, allowed_max_exclusive: int, suppressed_ids: set[int]):
        self.allowed_max_exclusive = int(allowed_max_exclusive)
        self.suppressed_ids = {int(x) for x in suppressed_ids if int(x) >= 0}
        self._cache: dict[tuple[int, str], torch.Tensor] = {}

    def _mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        key = (int(vocab_size), str(device))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        mask = torch.zeros(int(vocab_size), dtype=torch.bool, device=device)
        mask[: min(int(vocab_size), self.allowed_max_exclusive)] = True
        for token_id in self.suppressed_ids:
            if 0 <= token_id < vocab_size:
                mask[token_id] = False
        self._cache[key] = mask
        return mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = self._mask(scores.shape[-1], scores.device)
        scores = scores.clone()
        scores[:, ~mask] = float("-inf")
        return scores


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _done_keys(path: Path) -> set[tuple[str, str, str, str, str, str]]:
    if not path.exists():
        return set()
    out = set()
    for row in _json_rows(path):
        out.add(
            (
                str(row.get("prompt_id")),
                str(row.get("cell")),
                str(row.get("component")),
                str(row.get("scenario")),
                str(row.get("boundary")),
                str(row.get("interpolation_alpha")),
            )
        )
    return out


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for idx in range(0, len(items), max(1, int(size))):
        yield items[idx : idx + max(1, int(size))]


def _reserved_ids(tokenizers: Iterable[Any]) -> set[int]:
    suppressed: set[int] = set()
    patterns = (
        re.compile(r"^<unused\d+>$"),
        re.compile(r"^<\|reserved_special_token_\d+\|>$"),
    )
    for tokenizer in tokenizers:
        key = (getattr(tokenizer, "name_or_path", type(tokenizer).__name__), len(tokenizer))
        cached = _SUPPRESSED_ID_CACHE.get(key)
        if cached is not None:
            suppressed.update(cached)
            continue
        local: set[int] = set()
        try:
            vocab_items = tokenizer.get_vocab().items()
        except Exception:
            vocab_items = []
        for tok, token_id in vocab_items:
            if tok is None or any(pat.match(str(tok)) for pat in patterns):
                local.add(int(token_id))
        _SUPPRESSED_ID_CACHE[key] = local
        suppressed.update(local)
    return suppressed


def _input_vocab(model: torch.nn.Module) -> int:
    return int(model.get_input_embeddings().weight.shape[0])


def _eos_ids(*tokenizers: Any, max_id: int | None = None) -> list[int]:
    out: set[int] = set()
    for tokenizer in tokenizers:
        value = getattr(tokenizer, "eos_token_id", None)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            out.update(int(x) for x in value if x is not None)
        else:
            out.add(int(value))
    if max_id is not None:
        out = {x for x in out if 0 <= x < max_id}
    return sorted(out)


def _decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _score_entropy(scores: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> float | None:
    if not scores:
        return None
    vals = []
    for score in scores:
        probs = torch.softmax(score.float(), dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1)
        vals.extend(float(x) for x in ent.detach().cpu().tolist())
    return float(sum(vals) / len(vals)) if vals else None


def _repeat_stats(ids: list[int]) -> dict[str, Any]:
    if not ids:
        return {
            "generated_tokens_count": 0,
            "unique_token_fraction": None,
            "max_3gram_repeat": 0,
            "has_long_loop": False,
        }
    unique_frac = len(set(ids)) / max(1, len(ids))
    grams: dict[tuple[int, ...], int] = {}
    for idx in range(max(0, len(ids) - 2)):
        gram = tuple(ids[idx : idx + 3])
        grams[gram] = grams.get(gram, 0) + 1
    max_repeat = max(grams.values(), default=0)
    return {
        "generated_tokens_count": len(ids),
        "unique_token_fraction": float(unique_frac),
        "max_3gram_repeat": int(max_repeat),
        "has_long_loop": bool(max_repeat >= 4 or unique_frac < 0.20),
    }


def _select_records(
    *,
    dataset: Path,
    n_prompts: int | None,
    split: str,
    category_filter: set[str] | None,
    source_filter: set[str] | None,
) -> list[dict[str, Any]]:
    rows = load_dataset(dataset, n_examples=n_prompts)
    out = []
    for row in rows:
        prompt_id = str(row.get("id", row.get("record_id")))
        if split != "all" and prompt_split(prompt_id) != split:
            continue
        if category_filter and str(row.get("category", "")) not in category_filter:
            continue
        if source_filter and str(row.get("source", "")) not in source_filter:
            continue
        out.append(row)
    return out


def _cell_models(
    *,
    cell: str,
    base_model: torch.nn.Module,
    ft_model: torch.nn.Module,
    wrong_model: torch.nn.Module | None,
    scenario: str,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if scenario == "wrong_descendant":
        if wrong_model is None:
            raise ValueError("wrong_descendant scenario requires wrong_model")
        return base_model, wrong_model
    if cell in {"BB", "BF"}:
        host = base_model
    elif cell in {"FB", "FF"}:
        host = ft_model
    else:
        raise ValueError(f"Unknown cell: {cell}")
    donor = ft_model if cell == "BF" else base_model
    return host, donor


def _readout_tokenizer(cell: str, scenario: str, tokenizers: dict[str, Any]) -> Any:
    if scenario == "wrong_descendant":
        return tokenizers["wrong"]
    if cell in {"BF", "FF"}:
        return tokenizers["ft"]
    return tokenizers["base"]


def _native_parent_label(cell: str, scenario: str) -> str:
    if scenario == "wrong_descendant":
        return "wrong"
    if cell in {"BF", "FF"}:
        return "ft"
    return "base"


@torch.no_grad()
def _last_position_logits(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1, :].detach().float()


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor, allowed: int) -> float:
    size = min(int(allowed), p_logits.shape[-1], q_logits.shape[-1])
    p_log = F.log_softmax(p_logits[..., :size].float(), dim=-1)
    q_log = F.log_softmax(q_logits[..., :size].float(), dim=-1)
    p = p_log.exp()
    val = (p * (p_log - q_log)).sum(dim=-1).mean()
    return float(val.detach().cpu().item())


@torch.no_grad()
def _teacher_forced_health(
    *,
    host: torch.nn.Module,
    native_base_logits: torch.Tensor,
    native_ft_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary: int,
    adapter: Any,
    allowed_vocab: int,
) -> dict[str, Any]:
    layers = adapter.get_layers(host)
    boundary_capture = BoundaryStateCapture(layers[boundary])
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter.adapter)
    try:
        logits = host(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        boundary_state = boundary_capture.snapshot()
        residuals = residual_capture.snapshot()
    finally:
        boundary_capture.close()
        residual_capture.close()

    labels = input_ids[:, 1:].contiguous()
    shifted = logits[:, :-1, : min(logits.shape[-1], allowed_vocab)].float()
    ok = labels < shifted.shape[-1]
    if ok.any():
        losses = F.cross_entropy(
            shifted.reshape(-1, shifted.shape[-1]),
            labels.clamp_max(shifted.shape[-1] - 1).reshape(-1),
            reduction="none",
        ).view_as(labels)
        nll = float(losses[ok].mean().detach().cpu().item())
    else:
        nll = None
    host_logits = logits[:, -1, :].detach().float()
    return {
        "prompt_nll": nll,
        "prompt_perplexity": float(math.exp(nll)) if nll is not None and nll < 30 else None,
        "boundary_last_norm": float(boundary_state[:, -1, :].float().norm(dim=-1).mean().detach().cpu().item()),
        "boundary_last_rms": float(boundary_state[:, -1, :].float().pow(2).mean(dim=-1).sqrt().mean().detach().cpu().item()),
        "final_last_norm": float(torch.stack(residuals)[-1].float().norm().detach().cpu().item()),
        "kl_to_base_last": _kl(host_logits, native_base_logits, allowed_vocab),
        "kl_to_ft_last": _kl(host_logits, native_ft_logits, allowed_vocab),
    }


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    wrong_model_name = args.wrong_model or wrong_descendant_for(args.model)

    log.info(
        "[exp48-seq] load model=%s scenario=%s cell=%s component=%s boundary=%s wrong=%s",
        args.model,
        args.scenario,
        args.cell,
        args.component,
        args.boundary,
        wrong_model_name,
    )
    pair_key = (args.model, args.device)
    if pair_key not in _PAIR_CACHE:
        spec = get_spec(args.model)
        steering_adapter = get_steering_adapter(args.model)
        base_model, base_tok = load_model_and_tokenizer(model_id_for_variant(spec, "pt"), args.device, multi_gpu=spec.multi_gpu)
        ft_model, ft_tok = load_model_and_tokenizer(model_id_for_variant(spec, "it"), args.device, multi_gpu=spec.multi_gpu)
        _PAIR_CACHE[pair_key] = (steering_adapter, base_model, base_tok, ft_model, ft_tok)
    else:
        steering_adapter, base_model, base_tok, ft_model, ft_tok = _PAIR_CACHE[pair_key]
    wrong_model = None
    wrong_tok = None
    if args.scenario == "wrong_descendant":
        wrong_key = (wrong_model_name, args.device)
        if wrong_key not in _WRONG_CACHE:
            wrong_spec = get_spec(wrong_model_name)
            wrong_model, wrong_tok = load_model_and_tokenizer(
                model_id_for_variant(wrong_spec, "it"), args.device, multi_gpu=wrong_spec.multi_gpu
            )
            _WRONG_CACHE[wrong_key] = (wrong_model, wrong_tok)
        else:
            wrong_model, wrong_tok = _WRONG_CACHE[wrong_key]

    tokenizers = {"base": base_tok, "ft": ft_tok}
    if wrong_tok is not None:
        tokenizers["wrong"] = wrong_tok
    for tok in tokenizers.values():
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

    allowed_vocab = min(_input_vocab(base_model), _input_vocab(ft_model))
    if wrong_model is not None:
        allowed_vocab = min(allowed_vocab, _input_vocab(wrong_model))
    suppressed = _reserved_ids(tokenizers.values())
    logits_processor = [CommonVocabLogitsProcessor(allowed_vocab, suppressed)]
    eos = _eos_ids(*tokenizers.values(), max_id=allowed_vocab)

    records = _select_records(
        dataset=args.dataset,
        n_prompts=args.n_prompts,
        split=args.prompt_split,
        category_filter=set(args.category_filter or []) or None,
        source_filter=set(args.source_filter or []) or None,
    )
    records = records[args.worker_index :: args.n_workers]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"sequence_records_w{args.worker_index}.jsonl.gz"
    done = _done_keys(out_path)
    log.info("[exp48-seq] selected prompts=%d resume=%d out=%s", len(records), len(done), out_path)

    cell = args.cell
    scenario = args.scenario
    component = args.component
    host_model, donor_model = _cell_models(
        cell=cell,
        base_model=base_model,
        ft_model=ft_model,
        wrong_model=wrong_model,
        scenario=scenario,
    )
    decode_tok = _readout_tokenizer(cell, scenario, tokenizers)
    prompt_tok = base_tok

    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for batch_idx, batch in enumerate(_chunks(records, args.batch_size)):
            batch = [
                row for row in batch
                if (
                    str(row.get("id", row.get("record_id"))),
                    cell,
                    component,
                    scenario,
                    str(args.boundary),
                    str(args.interpolation_alpha),
                )
                not in done
            ]
            if not batch:
                continue
            prompts = [
                get_prompt_for_variant(row, variant="pt", tokenizer=prompt_tok, apply_chat_template=False)
                for row in batch
            ]
            encoded = prompt_tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            native_base_logits = None
            native_ft_logits = None
            if args.health_every > 0 and batch_idx % args.health_every == 0:
                native_base_logits = _last_position_logits(base_model, input_ids, attention_mask)
                native_ft_logits = _last_position_logits(ft_model, input_ids, attention_mask)

            if scenario == "interpolated_late":
                ctx = interpolated_late_context(
                    host_model=base_model,
                    donor_model=ft_model,
                    boundary=args.boundary,
                    alpha=float(args.interpolation_alpha),
                    include_readout=True,
                )
                active_host = base_model
            else:
                ctx = static_chimera_context(
                    host_model=host_model,
                    donor_model=donor_model,
                    boundary=args.boundary,
                    component=component,
                    cell=cell,
                    permute_blocks=(scenario == "permuted_blocks"),
                )
                active_host = host_model

            with ctx:
                outputs = active_host.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=prompt_tok.pad_token_id,
                    eos_token_id=eos or None,
                    use_cache=True,
                    logits_processor=logits_processor,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                entropy = _score_entropy(outputs.scores)
                health_payloads: list[dict[str, Any] | None] = [None] * len(batch)
                if args.health_every > 0 and batch_idx % args.health_every == 0:
                    for j in range(len(batch)):
                        try:
                            one_ids = input_ids[j : j + 1]
                            one_mask = attention_mask[j : j + 1]
                            health_payloads[j] = _teacher_forced_health(
                                host=active_host,
                                native_base_logits=native_base_logits[j : j + 1],
                                native_ft_logits=native_ft_logits[j : j + 1],
                                input_ids=one_ids,
                                attention_mask=one_mask,
                                boundary=args.boundary,
                                adapter=steering_adapter,
                                allowed_vocab=allowed_vocab,
                            )
                        except Exception as exc:
                            health_payloads[j] = {"health_error": str(exc)}

            input_width = input_ids.shape[1]
            for j, row in enumerate(batch):
                seq = [int(x) for x in outputs.sequences[j].detach().cpu().tolist()]
                gen_ids = [int(x) for x in seq[input_width:] if int(x) != prompt_tok.pad_token_id]
                text = _decode(decode_tok, gen_ids)
                post_first = _decode(decode_tok, gen_ids[1:]) if len(gen_ids) > 1 else ""
                prompt_id = str(row.get("id", row.get("record_id")))
                payload = {
                    "experiment": "exp48_static_chimera_sequence_validation",
                    "record_type": "sequence",
                    "model": args.model,
                    "wrong_model": wrong_model_name if scenario == "wrong_descendant" else None,
                    "prompt_id": prompt_id,
                    "prompt_split": prompt_split(prompt_id),
                    "category": row.get("category"),
                    "source": row.get("source"),
                    "boundary": int(args.boundary),
                    "cell": cell,
                    "scenario": scenario,
                    "component": component,
                    "interpolation_alpha": float(args.interpolation_alpha) if scenario == "interpolated_late" else None,
                    "late_parent": _native_parent_label(cell, scenario),
                    "decode_mode": "greedy",
                    "max_new_tokens": int(args.max_new_tokens),
                    "prompt_token_count_padded": int(input_width),
                    "generated_token_ids": gen_ids,
                    "continuation_text": text,
                    "post_first_continuation_text": post_first,
                    "eos_emitted": bool(gen_ids and gen_ids[-1] in set(eos)),
                    "mean_step_entropy": entropy,
                    "invalid_output": bool((not text.strip()) or len(gen_ids) == 0),
                    **_repeat_stats(gen_ids),
                    **lexical_metrics(text, post_first_text=post_first),
                    "health": health_payloads[j] or {},
                }
                fout.write(json.dumps(payload, separators=(",", ":")) + "\n")
            fout.flush()
            if (batch_idx + 1) % 5 == 0:
                log.info("[exp48-seq] model=%s %s batches=%d", args.model, args.stem, batch_idx + 1)
    log.info("[exp48-seq] done %s", out_path)


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "sequence_records.jsonl.gz"
    seen: set[tuple[str, str, str, str, str, str]] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(int(n_workers)):
            path = out_dir / f"sequence_records_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp48-seq] missing worker file %s", path)
                continue
            for row in _json_rows(path):
                key = (
                    str(row.get("prompt_id")),
                    str(row.get("cell")),
                    str(row.get("component")),
                    str(row.get("scenario")),
                    str(row.get("boundary")),
                    str(row.get("interpolation_alpha")),
                )
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    log.info("[exp48-seq] merged rows=%d -> %s", len(seen), merged)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-prompts", type=int, default=1400)
    parser.add_argument("--prompt-split", choices=["train", "heldout", "all"], default="heldout")
    parser.add_argument("--boundary", type=int, default=19)
    parser.add_argument("--scenario", choices=["boundary_sweep", "component_variant", "wrong_descendant", "permuted_blocks", "interpolated_late", "decomposition"], default="boundary_sweep")
    parser.add_argument("--component", choices=["blocks_plus_head", "blocks_only", "head_only", "mlp_only", "attn_only"], default="blocks_plus_head")
    parser.add_argument("--cell", choices=["BB", "BF", "FB", "FF"], default="BF")
    parser.add_argument("--wrong-model", choices=list(MODEL_REGISTRY), default=None)
    parser.add_argument("--interpolation-alpha", type=float, default=1.0, choices=list(INTERPOLATION_ALPHAS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--health-every", type=int, default=8)
    parser.add_argument("--category-filter", nargs="*", default=None)
    parser.add_argument("--source-filter", nargs="*", default=None)
    parser.add_argument("--stem", default="sequence")
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
