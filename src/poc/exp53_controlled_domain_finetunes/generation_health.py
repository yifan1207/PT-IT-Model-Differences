"""Basic generation-health checks for Exp53 merged checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import BASE_MODEL_ID, BASE_REVISION, iter_jsonl, paths_for, write_json


def _lazy_imports():
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    return AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    AutoModelForCausalLM, AutoTokenizer = _lazy_imports()
    paths = paths_for(args.run_root, args.domain)
    tok = AutoTokenizer.from_pretrained(str(paths.merged_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(paths.merged_dir), torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=args.device)
    model.eval()
    rows = []
    degenerate = 0
    for i, row in enumerate(iter_jsonl(paths.eval_jsonl)):
        ids = tok.encode(str(row["text"]), add_special_tokens=False)[: args.prefix_tokens]
        if len(ids) < 8:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=model.device)
        out = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        new_ids = out[0, x.shape[1] :].tolist()
        text = tok.decode(new_ids, skip_special_tokens=False)
        uniq_ratio = len(set(new_ids)) / max(1, len(new_ids))
        is_bad = len(new_ids) >= args.max_new_tokens and uniq_ratio < 0.15
        degenerate += int(is_bad)
        rows.append({"idx": i, "new_tokens": len(new_ids), "unique_ratio": uniq_ratio, "degenerate": is_bad, "sample": text[:500]})
        if len(rows) >= args.n_examples:
            break
    payload = {
        "domain": args.domain,
        "n_examples": len(rows),
        "degenerate_count": degenerate,
        "degenerate_rate": degenerate / max(1, len(rows)),
        "ok": degenerate == 0,
        "examples": rows,
    }
    write_json(paths.generation_health, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(6)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--domain", choices=["code", "biomed"], required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-tokens", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-examples", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

