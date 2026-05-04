"""Preflight reproducibility and compatibility checks for Exp46."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer

from src.poc.cross_model.utils import load_dataset
from src.poc.exp46_tulu_fixed_support_stage_sweep import PROMPT_MODES, STAGE_ORDER, STAGES
from src.poc.exp46_tulu_fixed_support_stage_sweep.common import format_prompt, record_id


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
    "tie_word_embeddings",
)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    return str(value)


def _resolve_revisions(*, allow_main_drift: bool) -> dict[str, Any]:
    api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    rows: dict[str, Any] = {}
    for key in STAGE_ORDER:
        spec = STAGES[key]
        if spec.repo_id == "allenai/Llama-3.1-Tulu-3.1-8B":
            raise RuntimeError("Exp46 must use Tulu 3, not Tulu 3.1.")
        info = api.model_info(spec.repo_id, revision="main", files_metadata=False)
        current_sha = str(info.sha)
        ok = current_sha == spec.revision
        rows[key] = {
            "stage": key,
            "label": spec.label,
            "repo_id": spec.repo_id,
            "pinned_revision": spec.revision,
            "current_main_sha": current_sha,
            "current_main_matches_pin": ok,
        }
        if not ok and not allow_main_drift:
            raise RuntimeError(
                f"HF main for {spec.repo_id} resolved to {current_sha}, "
                f"but Exp46 pin is {spec.revision}. Re-audit or pass --allow-main-drift."
            )
    return rows


def _config_payload() -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for key in STAGE_ORDER:
        spec = STAGES[key]
        cfg = AutoConfig.from_pretrained(spec.repo_id, revision=spec.revision, trust_remote_code=True)
        configs[key] = {name: _as_jsonable(getattr(cfg, name, None)) for name in CONFIG_KEYS}
    ref = configs["B"]
    mismatches = []
    for key in STAGE_ORDER:
        for name in CONFIG_KEYS:
            if configs[key].get(name) != ref.get(name):
                mismatches.append(
                    {
                        "stage": key,
                        "field": name,
                        "base": ref.get(name),
                        "stage_value": configs[key].get(name),
                    }
                )
    fatal_fields = {
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    }
    fatal = [row for row in mismatches if row["field"] in fatal_fields]
    if fatal:
        raise RuntimeError(f"Architecture/token-vocab config mismatch across Tulu stages: {fatal[:5]}")
    vocab_sizes = {
        key: int(value["vocab_size"])
        for key, value in configs.items()
        if value.get("vocab_size") is not None
    }
    return {
        "configs": configs,
        "mismatches": mismatches,
        "shared_vocab_size": min(vocab_sizes.values()) if vocab_sizes else None,
        "max_vocab_size": max(vocab_sizes.values()) if vocab_sizes else None,
        "vocab_size_note": (
            "Tulu checkpoints may add special tokens beyond the Llama base vocab. "
            "This is allowed only because Exp46 validates shared prompt token IDs and "
            "the support collector rejects divergent target tokens outside the shared vocab."
        ),
        "ok": True,
    }


def _tokenizer_payload(dataset: Path, n_examples: int, prompt_modes: list[str]) -> dict[str, Any]:
    tokenizers = {
        key: AutoTokenizer.from_pretrained(
            STAGES[key].repo_id,
            revision=STAGES[key].revision,
            trust_remote_code=True,
        )
        for key in STAGE_ORDER
    }
    records = load_dataset(dataset, n_examples=n_examples)
    payload: dict[str, Any] = {"n_checked": len(records), "prompt_modes": {}, "ok": True}
    for prompt_mode in prompt_modes:
        mode_rows = []
        for record in records:
            prompt = format_prompt(record, prompt_mode)
            ids_by_stage = {
                key: tokenizers[key].encode(prompt, add_special_tokens=True)
                for key in STAGE_ORDER
            }
            ref = ids_by_stage["B"]
            bad = [key for key, ids in ids_by_stage.items() if ids != ref]
            if bad:
                mode_rows.append(
                    {
                        "prompt_id": record_id(record),
                        "mismatched_stages": bad,
                        "lengths": {key: len(ids) for key, ids in ids_by_stage.items()},
                    }
                )
        mode_ok = not mode_rows
        payload["prompt_modes"][prompt_mode] = {
            "ok": mode_ok,
            "n_mismatches": len(mode_rows),
            "examples": mode_rows[:10],
        }
        payload["ok"] = payload["ok"] and mode_ok
    if not payload["ok"]:
        raise RuntimeError(f"Tokenizer prompt-ID compatibility failed: {payload}")
    return payload


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.prompt_mode == "both":
        prompt_modes = list(PROMPT_MODES)
    else:
        prompt_modes = [args.prompt_mode]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    revisions = _resolve_revisions(allow_main_drift=args.allow_main_drift)
    configs = _config_payload()
    tokenizers = _tokenizer_payload(args.dataset, args.n_tokenizer_check_examples, prompt_modes)
    summary = {
        "experiment": "exp46_tulu_fixed_support_stage_sweep",
        "stage_revisions": revisions,
        "config_compatibility": configs,
        "tokenizer_compatibility": tokenizers,
        "prompt_modes_checked": prompt_modes,
        "ok": True,
    }
    (args.out_dir / "revisions.json").write_text(json.dumps(revisions, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "config_compatibility.json").write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "tokenizer_compatibility.json").write_text(json.dumps(tokenizers, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--n-tokenizer-check-examples", type=int, default=32)
    parser.add_argument("--prompt-mode", choices=[*PROMPT_MODES, "both"], default="raw_shared")
    parser.add_argument("--allow-main-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run_preflight(parse_args())
    print(json.dumps({"experiment": summary["experiment"], "ok": summary["ok"]}, indent=2))


if __name__ == "__main__":
    main()
