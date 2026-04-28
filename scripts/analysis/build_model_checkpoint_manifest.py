#!/usr/bin/env python3
"""Build a checkpoint/tokenizer manifest for paper reproducibility.

The registry records Hugging Face repo IDs plus immutable snapshot revisions
for reruns. Older experiment traces may only have stored repo IDs, so the
manifest also audits what the Hub currently resolves for `main`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.poc.cross_model.config import MODEL_REGISTRY, model_revision_for_variant


DEFAULT_MODELS = [
    "gemma3_4b",
    "llama31_8b",
    "qwen3_4b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]


def _sha256(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_info(api: HfApi, repo_id: str) -> dict[str, Any]:
    try:
        info = api.model_info(repo_id, revision="main")
        return {"resolved_main_sha": info.sha, "hub_error": None}
    except Exception as exc:  # pragma: no cover - network/auth dependent
        return {"resolved_main_sha": None, "hub_error": f"{type(exc).__name__}: {exc}"}


def _tokenizer_info(repo_id: str, revision: str | None) -> dict[str, Any]:
    try:
        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        tok = AutoTokenizer.from_pretrained(repo_id, **kwargs)
        chat_template = getattr(tok, "chat_template", None)
        return {
            "tokenizer_class": tok.__class__.__name__,
            "vocab_size": len(tok),
            "model_max_length": int(tok.model_max_length),
            "bos_token": tok.bos_token,
            "bos_token_id": tok.bos_token_id,
            "eos_token": tok.eos_token,
            "eos_token_id": tok.eos_token_id,
            "pad_token": tok.pad_token,
            "pad_token_id": tok.pad_token_id,
            "chat_template_present": bool(chat_template),
            "chat_template_sha256": _sha256(chat_template),
            "chat_template_length": len(chat_template) if chat_template else 0,
            "tokenizer_error": None,
        }
    except Exception as exc:  # pragma: no cover - network/auth dependent
        return {
            "tokenizer_class": None,
            "vocab_size": None,
            "model_max_length": None,
            "bos_token": None,
            "bos_token_id": None,
            "eos_token": None,
            "eos_token_id": None,
            "pad_token": None,
            "pad_token_id": None,
            "chat_template_present": None,
            "chat_template_sha256": None,
            "chat_template_length": None,
            "tokenizer_error": f"{type(exc).__name__}: {exc}",
        }


def build_manifest(models: list[str]) -> dict[str, Any]:
    api = HfApi()
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "important_scope_note": (
            "The project registry now pins immutable Hugging Face revisions "
            "for reproducible reruns. Historical result configs may only store "
            "repo IDs, so `resolved_main_sha` remains an audit field rather "
            "than proof of historical run-time pinning."
        ),
        "prompting_policy": {
            "pt": "raw prompt text from eval_dataset_v2 format B/A; no chat template",
            "it_native": (
                "tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], "
                "tokenize=False, add_generation_prompt=True)"
            ),
            "qwen_note": (
                "For Qwen IT native prompts, code passes enable_thinking=False "
                "when supported by the tokenizer."
            ),
            "raw_shared": (
                "First-divergence residual-state and raw_shared tests disable "
                "chat templates for both PT and IT and validate identical raw "
                "prompt token IDs before patching."
            ),
        },
        "models": {},
    }
    for name in models:
        spec = MODEL_REGISTRY[name]
        rows = {}
        for variant, repo_id in (("pt", spec.pt_id), ("it", spec.it_id)):
            configured_revision = model_revision_for_variant(spec, variant)
            rows[variant] = {
                "repo_id": repo_id,
                "configured_revision": configured_revision,
                "explicit_revision_pinned_in_current_registry": configured_revision is not None,
                "explicit_revision_pinned_in_historical_runs": False,
                **_model_info(api, repo_id),
                **_tokenizer_info(repo_id, configured_revision),
            }
            rows[variant]["configured_revision_equals_resolved_main"] = (
                configured_revision == rows[variant].get("resolved_main_sha")
                if configured_revision is not None and rows[variant].get("resolved_main_sha") is not None
                else None
            )
        manifest["models"][name] = {
            "n_layers": spec.n_layers,
            "d_model": spec.d_model,
            "n_heads": spec.n_heads,
            "n_kv_heads": spec.n_kv_heads,
            "is_moe": spec.is_moe,
            "is_sliding_window": spec.is_sliding_window,
            "pt": rows["pt"],
            "it": rows["it"],
            "same_vocab_size": (
                rows["pt"]["vocab_size"] == rows["it"]["vocab_size"]
                if rows["pt"]["vocab_size"] is not None and rows["it"]["vocab_size"] is not None
                else None
            ),
            "same_eos_token_id": (
                rows["pt"]["eos_token_id"] == rows["it"]["eos_token_id"]
                if rows["pt"]["eos_token_id"] is not None and rows["it"]["eos_token_id"] is not None
                else None
            ),
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/paper_synthesis/model_checkpoint_manifest.json"))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    payload = build_manifest(list(args.models))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "models": args.models}, indent=2))


if __name__ == "__main__":
    main()
