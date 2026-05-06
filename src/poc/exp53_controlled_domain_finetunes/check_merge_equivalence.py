"""Check PEFT adapter-loaded and merged Exp53 checkpoints agree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import BASE_MODEL_ID, BASE_REVISION, iter_jsonl, paths_for, write_json


def _lazy_imports():
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    return PeftModel, AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    PeftModel, AutoModelForCausalLM, AutoTokenizer = _lazy_imports()
    paths = paths_for(args.run_root, args.domain)
    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision, trust_remote_code=True, token=args.hf_token)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=args.hf_token,
        device_map=args.device,
    )
    adapter_model = PeftModel.from_pretrained(base, paths.adapter_dir)
    adapter_model.eval()
    merged = AutoModelForCausalLM.from_pretrained(str(paths.merged_dir), torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=args.device)
    merged.eval()
    max_abs = 0.0
    max_topk_abs = 0.0
    mean_abs = []
    kls = []
    top1_agree = 0
    adapter_top1_in_merged_topk = 0
    n = 0
    for row in iter_jsonl(paths.eval_jsonl):
        ids = tok.encode(str(row["text"]), add_special_tokens=False)[: args.seq_len]
        if len(ids) < 8:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=adapter_model.device)
        a = adapter_model(input_ids=x).logits[:, -1, :].float()
        b = merged(input_ids=x).logits[:, -1, :].float()
        diff = (a - b).abs()
        max_abs = max(max_abs, float(diff.max().cpu()))
        adapter_topk = a.topk(args.topk, dim=-1).indices
        merged_topk = b.topk(args.topk, dim=-1).indices
        max_topk_abs = max(max_topk_abs, float(diff.gather(-1, adapter_topk).max().cpu()))
        mean_abs.append(float(diff.mean().cpu()))
        logpa = torch.log_softmax(a, dim=-1)
        logpb = torch.log_softmax(b, dim=-1)
        kls.append(float((logpa.exp() * (logpa - logpb)).sum().cpu()))
        top1_agree += int(adapter_topk[0, 0].item() == merged_topk[0, 0].item())
        adapter_top1_in_merged_topk += int(adapter_topk[0, 0].item() in set(merged_topk[0].tolist()))
        n += 1
        if n >= args.n_examples:
            break
    mean_abs_value = sum(mean_abs) / max(1, len(mean_abs))
    mean_kl_value = sum(kls) / max(1, len(kls))
    top1_agreement = top1_agree / max(1, n)
    adapter_top1_in_merged_topk_rate = adapter_top1_in_merged_topk / max(1, n)
    ok = (
        max_abs <= args.max_abs_tolerance
        and max_topk_abs <= args.max_topk_abs_tolerance
        and mean_abs_value <= args.mean_abs_tolerance
        and mean_kl_value <= args.mean_kl_tolerance
        and adapter_top1_in_merged_topk_rate >= args.top1_in_topk_min
    )
    payload = {
        "domain": args.domain,
        "n_examples": n,
        "max_abs_logit_diff": max_abs,
        "max_topk_abs_logit_diff": max_topk_abs,
        "mean_abs_logit_diff": mean_abs_value,
        "mean_kl_adapter_to_merged": mean_kl_value,
        "top1_agreement": top1_agreement,
        f"adapter_top1_in_merged_top{args.topk}": adapter_top1_in_merged_topk_rate,
        "ok": ok,
        "max_abs_tolerance": args.max_abs_tolerance,
        "max_topk_abs_tolerance": args.max_topk_abs_tolerance,
        "mean_abs_tolerance": args.mean_abs_tolerance,
        "mean_kl_tolerance": args.mean_kl_tolerance,
        "top1_in_topk_min": args.top1_in_topk_min,
    }
    write_json(paths.merge_check, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(5)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--domain", choices=["code", "biomed"], required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-examples", type=int, default=32)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-abs-tolerance", type=float, default=6e-1)
    parser.add_argument("--max-topk-abs-tolerance", type=float, default=6e-1)
    parser.add_argument("--mean-abs-tolerance", type=float, default=5e-2)
    parser.add_argument("--mean-kl-tolerance", type=float, default=2e-3)
    parser.add_argument("--top1-in-topk-min", type=float, default=0.99)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
