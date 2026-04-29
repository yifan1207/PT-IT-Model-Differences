"""Preflight checks for Exp25 OLMo stage-progression runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from src.poc.cross_model.config import get_spec, model_id_for_variant, model_revision_for_variant
from src.poc.cross_model.utils import get_raw_prompt, load_dataset
from src.poc.exp25_olmo_stage_progression import STAGE_MODELS


def _sha256(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _tokenizer_payload(repo_id: str, revision: str | None) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(repo_id, revision=revision, trust_remote_code=True)
    template = getattr(tok, "chat_template", None)
    return {
        "repo_id": repo_id,
        "configured_revision": revision,
        "tokenizer_class": tok.__class__.__name__,
        "vocab_size": len(tok),
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "chat_template_present": bool(template),
        "chat_template_sha256": _sha256(template),
        "chat_template_length": len(template or ""),
    }


def run_preflight(models: list[str], dataset: Path, n_examples: int) -> dict[str, Any]:
    api = HfApi()
    records = load_dataset(dataset, n_examples=n_examples)
    out: dict[str, Any] = {
        "models": models,
        "dataset": str(dataset),
        "n_examples_checked": len(records),
        "stage_pairs": {},
        "ok": True,
        "errors": [],
    }
    for model_name in models:
        spec = get_spec(model_name)
        pair_payload: dict[str, Any] = {"variants": {}, "raw_prompt_token_id_mismatches": []}
        tokenizers = {}
        for variant in ("pt", "it"):
            repo_id = model_id_for_variant(spec, variant)
            revision = model_revision_for_variant(spec, variant)
            info = api.model_info(repo_id, revision="main")
            payload = _tokenizer_payload(repo_id, revision)
            payload["resolved_main_sha"] = info.sha
            payload["configured_revision_equals_resolved_main"] = revision == info.sha
            pair_payload["variants"][variant] = payload
            tokenizers[variant] = AutoTokenizer.from_pretrained(
                repo_id,
                revision=revision,
                trust_remote_code=True,
            )

        pt_meta = pair_payload["variants"]["pt"]
        it_meta = pair_payload["variants"]["it"]
        for key in ("vocab_size", "eos_token_id", "pad_token_id"):
            if pt_meta.get(key) != it_meta.get(key):
                out["ok"] = False
                out["errors"].append(f"{model_name}: tokenizer {key} mismatch")

        for record in records:
            prompt_id = str(record.get("id", record.get("record_id", "unknown")))
            raw = get_raw_prompt(record)
            pt_ids = tokenizers["pt"].encode(raw, add_special_tokens=True)
            it_ids = tokenizers["it"].encode(raw, add_special_tokens=True)
            if pt_ids != it_ids:
                pair_payload["raw_prompt_token_id_mismatches"].append(prompt_id)
                if len(pair_payload["raw_prompt_token_id_mismatches"]) >= 5:
                    break
        if pair_payload["raw_prompt_token_id_mismatches"]:
            out["ok"] = False
            out["errors"].append(
                f"{model_name}: raw prompt token IDs mismatch for "
                f"{len(pair_payload['raw_prompt_token_id_mismatches'])}+ prompts"
            )

        out["stage_pairs"][model_name] = pair_payload
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=STAGE_MODELS)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/paper_synthesis/olmo_stage_preflight.json"))
    args = parser.parse_args()

    payload = run_preflight(list(args.models), args.dataset, args.n_examples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "ok": payload["ok"], "errors": payload["errors"]}, indent=2))
    if not payload["ok"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
