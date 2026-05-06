"""Merge an Exp53 LoRA adapter into a BF16 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import BASE_MODEL_ID, BASE_REVISION, paths_for, write_json


def _lazy_imports():
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    return PeftModel, AutoModelForCausalLM, AutoTokenizer


def run(args: argparse.Namespace) -> dict:
    PeftModel, AutoModelForCausalLM, AutoTokenizer = _lazy_imports()
    paths = paths_for(args.run_root, args.domain)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=args.hf_token,
        device_map=args.device,
    )
    model = PeftModel.from_pretrained(base, paths.adapter_dir)
    model = model.merge_and_unload()
    paths.merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(paths.merged_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision, trust_remote_code=True, token=args.hf_token)
    tok.save_pretrained(paths.merged_dir)
    payload = {
        "domain": args.domain,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter_dir": str(paths.adapter_dir),
        "merged_dir": str(paths.merged_dir),
        "dtype": "bfloat16",
    }
    write_json(paths.merged_dir / "exp53_merge_manifest.json", payload)
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
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

