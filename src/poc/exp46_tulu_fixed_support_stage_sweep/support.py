"""Build fixed Base->Final first-divergence support for Exp46."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.poc.cross_model.utils import load_dataset
from src.poc.exp20_divergence_token_counterfactual.collect import _call_model, _masked_next_token
from src.poc.exp20_divergence_token_counterfactual.metrics import find_divergence_events
from src.poc.exp46_tulu_fixed_support_stage_sweep import MODEL_NAME, PROMPT_MODES, SUPPORT_FILE_NAME
from src.poc.exp46_tulu_fixed_support_stage_sweep.common import (
    format_prompt,
    json_rows,
    load_stage,
    real_token_mask_for,
    record_id,
    token_payload,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def support_file_name(target_stage: str) -> str:
    target_stage = target_stage.upper()
    if target_stage == "R":
        return SUPPORT_FILE_NAME
    if target_stage in {"S", "D"}:
        return f"base_{target_stage.lower()}_first_diff.jsonl.gz"
    raise ValueError("target_stage must be S, D, or R")


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["prompt_id"]) for row in json_rows(path) if row.get("prompt_id")}


@torch.no_grad()
def _generate_pair_until_first_diff(
    *,
    record: dict[str, Any],
    base_model: Any,
    base_tokenizer: Any,
    target_model: Any,
    target_tokenizer: Any,
    base_mask: torch.Tensor,
    target_mask: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    prompt_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    prompt = format_prompt(record, prompt_mode)
    base_prompt_ids = base_tokenizer.encode(prompt, add_special_tokens=True)
    target_prompt_ids = target_tokenizer.encode(prompt, add_special_tokens=True)
    if base_prompt_ids != target_prompt_ids:
        raise RuntimeError(
            f"Prompt token IDs differ for prompt_id={record_id(record)} prompt_mode={prompt_mode}"
        )

    state = {
        "B": {
            "model": base_model,
            "tokenizer": base_tokenizer,
            "mask": base_mask,
            "input_ids": torch.tensor([base_prompt_ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(base_prompt_ids)), dtype=torch.long, device=device),
            "past_key_values": None,
            "total_len": len(base_prompt_ids),
            "generated_ids": [],
            "generated_tokens": [],
            "stopped": False,
        },
        "R": {
            "model": target_model,
            "tokenizer": target_tokenizer,
            "mask": target_mask,
            "input_ids": torch.tensor([target_prompt_ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(target_prompt_ids)), dtype=torch.long, device=device),
            "past_key_values": None,
            "total_len": len(target_prompt_ids),
            "generated_ids": [],
            "generated_tokens": [],
            "stopped": False,
        },
    }

    first_event = None
    for _step in range(max_new_tokens):
        step_ids: dict[str, int | None] = {}
        for key, payload in state.items():
            if payload["stopped"]:
                step_ids[key] = None
                continue
            outputs = _call_model(
                MODEL_NAME,
                payload["model"],
                payload["input_ids"],
                payload["attention_mask"],
                past_key_values=payload["past_key_values"],
                use_cache=True,
            )
            payload["past_key_values"] = outputs.past_key_values
            next_id = _masked_next_token(outputs.logits[0, -1, :], payload["mask"])
            payload["generated_ids"].append(next_id)
            tok = token_payload(payload["tokenizer"], next_id)
            payload["generated_tokens"].append(tok)
            step_ids[key] = next_id
            if next_id == payload["tokenizer"].eos_token_id:
                payload["stopped"] = True
            else:
                payload["input_ids"] = torch.tensor([[next_id]], dtype=torch.long, device=device)
                payload["total_len"] += 1
                payload["attention_mask"] = torch.ones(
                    (1, payload["total_len"]), dtype=torch.long, device=device
                )
        if step_ids["B"] != step_ids["R"]:
            break
        if all(payload["stopped"] for payload in state.values()):
            break

    events = find_divergence_events(
        state["B"]["generated_tokens"],
        state["R"]["generated_tokens"],
    )
    event = events.get("first_diff")
    if isinstance(event, dict):
        first_event = event
    out = {}
    for key, payload in state.items():
        out[key] = {
            "generated_token_ids": payload["generated_ids"],
            "generated_tokens": payload["generated_tokens"],
            "generated_text": payload["tokenizer"].decode(payload["generated_ids"], skip_special_tokens=True),
            "n_steps": len(payload["generated_ids"]),
        }
    return out["B"], out["R"], first_event


def collect_support(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    target_stage = args.target_stage.upper()
    base_model, base_tokenizer = load_stage("B", device)
    target_model, target_tokenizer = load_stage(target_stage, device)
    base_mask = real_token_mask_for(base_model, base_tokenizer, device)
    target_mask = real_token_mask_for(target_model, target_tokenizer, device)
    records = load_dataset(
        args.dataset,
        n_examples=args.n_examples,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    file_name = support_file_name(target_stage)
    out_path = args.out_dir / f"{file_name.removesuffix('.jsonl.gz')}_w{args.worker_index}.jsonl.gz"
    done = _done_ids(out_path)
    log.info("[exp46] support worker %d/%d records=%d resume=%d", args.worker_index, args.n_workers, len(records), len(done))
    with gzip.open(out_path, "at", encoding="utf-8") as fout:
        for idx, record in enumerate(records):
            pid = record_id(record)
            if pid in done:
                continue
            try:
                base_run, rlvr_run, event = _generate_pair_until_first_diff(
                    record=record,
                    base_model=base_model,
                    base_tokenizer=base_tokenizer,
                    target_model=target_model,
                    target_tokenizer=target_tokenizer,
                    base_mask=base_mask,
                    target_mask=target_mask,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    prompt_mode=args.prompt_mode,
                )
                if not isinstance(event, dict):
                    payload = {
                        "prompt_id": pid,
                        "valid": False,
                        "reason": "no_first_diff",
                        "prompt_mode": args.prompt_mode,
                        "support_target_stage": target_stage,
                        "prompt_category": record.get("category"),
                        "base_run": base_run,
                        "rlvr_run": rlvr_run,
                        "final_run": rlvr_run,
                    }
                else:
                    prompt = format_prompt(record, args.prompt_mode)
                    prompt_ids = base_tokenizer.encode(prompt, add_special_tokens=True)
                    step = int(event["step"])
                    prefix_generated_ids = [int(x) for x in base_run["generated_token_ids"][:step]]
                    t_base = int(event["pt_token"]["token_id"])
                    t_rlvr = int(event["it_token"]["token_id"])
                    payload = {
                        "prompt_id": pid,
                        "valid": True,
                        "event_kind": "first_diff",
                        "prompt_text": prompt,
                        "prompt_mode": args.prompt_mode,
                        "support_target_stage": target_stage,
                        "prompt_category": record.get("category"),
                        "source": record.get("source"),
                        "divergence_step": step,
                        "prompt_token_ids": [int(x) for x in prompt_ids],
                        "prefix_generated_ids": prefix_generated_ids,
                        "full_input_ids": [int(x) for x in prompt_ids + prefix_generated_ids],
                        "t_base": token_payload(base_tokenizer, t_base),
                        "t_rlvr": token_payload(base_tokenizer, t_rlvr),
                        "t_final": token_payload(base_tokenizer, t_rlvr),
                        "base_run": base_run,
                        "rlvr_run": rlvr_run,
                        "final_run": rlvr_run,
                    }
                fout.write(json.dumps(payload, separators=(",", ":")) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[exp46] support prompt %s failed: %s", pid, exc)
            if (idx + 1) % 10 == 0:
                log.info("[exp46] support %d/%d", idx + 1, len(records))


def merge_support(out_dir: Path, n_workers: int, target_stage: str) -> Path:
    file_name = support_file_name(target_stage)
    merged = out_dir / file_name
    seen: set[str] = set()
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(n_workers):
            path = out_dir / f"{file_name.removesuffix('.jsonl.gz')}_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp46] missing support worker file %s", path)
                continue
            for row in json_rows(path):
                pid = str(row.get("prompt_id"))
                if pid in seen:
                    continue
                seen.add(pid)
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-examples", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-mode", choices=list(PROMPT_MODES), default="raw_shared")
    parser.add_argument("--target-stage", choices=["S", "D", "R"], default="R")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_support(args.out_dir, args.n_workers, args.target_stage)
        return
    collect_support(args)


if __name__ == "__main__":
    main()
