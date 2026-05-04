#!/usr/bin/env python3
"""Preflight checks for Exp47 same-base recipe specificity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from src.poc.cross_model.config import get_spec, model_id_for_variant, model_revision_for_variant
from src.poc.cross_model.utils import get_prompt_for_variant, load_dataset


DEFAULT_MODELS = (
    "llama31_meta_instruct",
    "llama31_tulu3_sft",
    "llama31_tulu3_dpo",
    "llama31_tulu3_final",
    "llama31_openmath2",
)

CONFIG_KEYS = (
    "model_type",
    "architectures",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "rope_scaling",
    "tie_word_embeddings",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_hashes(repo_id: str, revision: str | None, token: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in TOKENIZER_FILES:
        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    revision=revision,
                    filename=name,
                    token=token,
                )
            )
            out[name] = {"exists": True, "sha256": _sha256_path(path)}
        except Exception as exc:
            out[name] = {"exists": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def _model_payload(alias: str, variant: str, token: str | None, allow_main_drift: bool) -> dict[str, Any]:
    spec = get_spec(alias)
    repo_id = model_id_for_variant(spec, variant)
    pinned = model_revision_for_variant(spec, variant)
    api = HfApi(token=token)
    current_sha = None
    main_error = None
    try:
        current_sha = str(api.model_info(repo_id, revision="main", files_metadata=False).sha)
    except Exception as exc:
        main_error = f"{type(exc).__name__}: {exc}"
    if pinned and current_sha and pinned != current_sha and not allow_main_drift:
        raise RuntimeError(
            f"{alias}/{variant} {repo_id} main={current_sha} does not match pinned revision {pinned}. "
            "Re-audit or pass --allow-main-drift."
        )
    cfg = AutoConfig.from_pretrained(repo_id, revision=pinned, trust_remote_code=True, token=token)
    tok = AutoTokenizer.from_pretrained(repo_id, revision=pinned, trust_remote_code=True, token=token)
    return {
        "alias": alias,
        "variant": variant,
        "repo_id": repo_id,
        "pinned_revision": pinned,
        "current_main_sha": current_sha,
        "main_lookup_error": main_error,
        "current_main_matches_pin": bool((not pinned) or (current_sha == pinned)),
        "config": {key: _jsonable(getattr(cfg, key, None)) for key in CONFIG_KEYS},
        "tokenizer": {
            "class": tok.__class__.__name__,
            "vocab_size_len": len(tok),
            "eos_token_id": _jsonable(getattr(tok, "eos_token_id", None)),
            "pad_token_id": _jsonable(getattr(tok, "pad_token_id", None)),
            "bos_token_id": _jsonable(getattr(tok, "bos_token_id", None)),
            "chat_template_present": bool(getattr(tok, "chat_template", None)),
            "files": _file_hashes(repo_id, pinned, token),
        },
    }


def _compatibility(alias: str, pt: dict[str, Any], it: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    fatal_fields = {
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    }
    for key in CONFIG_KEYS:
        if pt["config"].get(key) != it["config"].get(key):
            mismatches.append({"field": key, "base": pt["config"].get(key), "descendant": it["config"].get(key)})
    fatal = [row for row in mismatches if row["field"] in fatal_fields]
    tokenizer_ok = pt["tokenizer"]["vocab_size_len"] == it["tokenizer"]["vocab_size_len"]
    return {
        "alias": alias,
        "ok": not fatal and tokenizer_ok,
        "tokenizer_vocab_ok": tokenizer_ok,
        "mismatches": mismatches,
        "fatal_mismatches": fatal,
    }


def _prompt_token_check(alias: str, records: list[dict[str, Any]], token: str | None) -> dict[str, Any]:
    spec = get_spec(alias)
    pt_tok = AutoTokenizer.from_pretrained(
        model_id_for_variant(spec, "pt"),
        revision=model_revision_for_variant(spec, "pt"),
        trust_remote_code=True,
        token=token,
    )
    it_tok = AutoTokenizer.from_pretrained(
        model_id_for_variant(spec, "it"),
        revision=model_revision_for_variant(spec, "it"),
        trust_remote_code=True,
        token=token,
    )
    mismatches = []
    for row in records:
        prompt_pt = get_prompt_for_variant(row, variant="pt", tokenizer=pt_tok, apply_chat_template=False)
        prompt_it = get_prompt_for_variant(row, variant="it", tokenizer=it_tok, apply_chat_template=False)
        ids_pt = pt_tok.encode(prompt_pt, add_special_tokens=True)
        ids_it = it_tok.encode(prompt_it, add_special_tokens=True)
        if ids_pt != ids_it:
            mismatches.append(
                {
                    "prompt_id": row.get("id", row.get("record_id")),
                    "pt_len": len(ids_pt),
                    "it_len": len(ids_it),
                }
            )
    return {
        "alias": alias,
        "n_checked": len(records),
        "ok": not mismatches,
        "n_mismatches": len(mismatches),
        "examples": mismatches[:10],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    records = load_dataset(args.dataset, n_examples=args.n_tokenizer_check_examples)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    aliases = list(args.models or DEFAULT_MODELS)
    model_manifest: dict[str, Any] = {}
    compat: dict[str, Any] = {}
    token_checks: dict[str, Any] = {}
    ok = True
    for alias in aliases:
        pt = _model_payload(alias, "pt", token, args.allow_main_drift)
        it = _model_payload(alias, "it", token, args.allow_main_drift)
        model_manifest[alias] = {"base": pt, "descendant": it}
        compat[alias] = _compatibility(alias, pt, it)
        token_checks[alias] = _prompt_token_check(alias, records, token)
        ok = ok and bool(compat[alias]["ok"]) and bool(token_checks[alias]["ok"])
    summary = {
        "experiment": "exp47_same_base_recipe_specificity",
        "models": aliases,
        "dataset": str(args.dataset),
        "n_tokenizer_check_examples": args.n_tokenizer_check_examples,
        "model_manifest": model_manifest,
        "compatibility": compat,
        "raw_shared_prompt_token_checks": token_checks,
        "ok": ok,
    }
    (args.out_dir / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "compatibility.json").write_text(json.dumps(compat, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "raw_shared_prompt_token_checks.json").write_text(json.dumps(token_checks, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not ok:
        raise SystemExit(4)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--n-tokenizer-check-examples", type=int, default=64)
    parser.add_argument("--allow-main-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({"experiment": summary["experiment"], "ok": summary["ok"], "models": summary["models"]}, indent=2))


if __name__ == "__main__":
    main()
