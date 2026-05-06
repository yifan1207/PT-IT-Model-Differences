"""Evaluate Exp53 base vs merged checkpoints on held-out domain text."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from .common import BASE_MODEL_ID, BASE_REVISION, iter_jsonl, paths_for, write_json


def _lazy_imports():
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    return AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def _eval_model(
    *,
    model_id: str,
    revision: str | None,
    eval_path: Path,
    device: str,
    max_tokens: int,
    seq_len: int,
    hf_token: str | None,
) -> dict[str, Any]:
    AutoModelForCausalLM, AutoTokenizer = _lazy_imports()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
        device_map=device,
    )
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    rows = 0
    for row in iter_jsonl(eval_path):
        ids = tok.encode(str(row["text"]), add_special_tokens=False)
        if tok.eos_token_id is not None:
            ids.append(int(tok.eos_token_id))
        for start in range(0, max(0, len(ids) - 1), seq_len):
            chunk = ids[start : start + seq_len]
            if len(chunk) < 8:
                continue
            x = torch.tensor([chunk], dtype=torch.long, device=model.device)
            out = model(input_ids=x, labels=x)
            n = x.numel() - 1
            total_loss += float(out.loss.detach().cpu()) * n
            total_tokens += n
            if total_tokens >= max_tokens:
                break
        rows += 1
        if total_tokens >= max_tokens:
            break
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    nll = total_loss / max(1, total_tokens)
    return {
        "model_id": model_id,
        "revision": revision,
        "nll": nll,
        "ppl": float(torch.exp(torch.tensor(nll)).item()),
        "tokens": total_tokens,
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = paths_for(args.run_root, args.domain)
    base = _eval_model(
        model_id=args.base_model,
        revision=args.base_revision,
        eval_path=paths.eval_jsonl,
        device=args.device,
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
        hf_token=args.hf_token,
    )
    tuned = _eval_model(
        model_id=str(paths.merged_dir),
        revision=None,
        eval_path=paths.eval_jsonl,
        device=args.device,
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
        hf_token=args.hf_token,
    )
    payload = {
        "domain": args.domain,
        "base": base,
        "tuned": tuned,
        "nll_delta_tuned_minus_base": tuned["nll"] - base["nll"],
        "relative_nll_improvement": (base["nll"] - tuned["nll"]) / base["nll"] if base["nll"] else None,
        "passes_domain_gate": tuned["nll"] < base["nll"],
    }
    write_json(paths.eval_loss, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--domain", choices=["code", "biomed"], required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--seq-len", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

