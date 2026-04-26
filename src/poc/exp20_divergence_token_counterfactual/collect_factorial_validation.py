"""Holdout validation collector for Exp20 first-divergence counterfactuals.

This collector intentionally does not free-run every graft branch. It free-runs
only pure PT and pure IT to identify the first divergent token, then performs
same-prefix one-step logit-lens readouts for the relevant graft branches. That
directly targets the identity-vs-margin claim while keeping the holdout rerun
small enough to repeat.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import torch

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.cross_model.utils import load_dataset, load_model_and_tokenizer, read_done_ids
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp20_divergence_token_counterfactual.collect import (
    CONDITIONS,
    ConditionSpec,
    _call_model,
    _decode_token,
    _masked_next_token,
    _prompt_for_condition,
    run_condition_free,
    run_condition_readout,
)
from src.poc.exp20_divergence_token_counterfactual.metrics import (
    DIVERGENCE_KINDS,
    find_divergence_events,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


VALIDATION_CONDITIONS: dict[str, ConditionSpec] = {
    **CONDITIONS,
    "B_earlymid_raw": ConditionSpec("B_earlymid_raw", "pt", "it", "earlymid"),
    "D_earlymid_ptswap": ConditionSpec("D_earlymid_ptswap", "it", "pt", "earlymid"),
}

FREE_CONDITIONS = ["A_pt_raw", "C_it_chat"]
READOUT_CONDITIONS = [
    "A_pt_raw",
    "B_early_raw",
    "B_mid_raw",
    "B_late_raw",
    "B_earlymid_raw",
    "B_midlate_raw",
    "C_it_chat",
    "D_early_ptswap",
    "D_mid_ptswap",
    "D_late_ptswap",
    "D_earlymid_ptswap",
    "D_midlate_ptswap",
]


def _read_done_ids_any_worker(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for path in sorted(out_dir.glob("exp20_validation_records_w*.jsonl")):
        done.update(read_done_ids(path))
    return done


def _event_prefix_and_targets(event: dict[str, Any], free_runs: dict[str, Any]) -> tuple[list[int], int, int] | None:
    pt_token = event.get("pt_token")
    it_token = event.get("it_token")
    step = event.get("step")
    if not isinstance(pt_token, dict) or not isinstance(it_token, dict) or not isinstance(step, int):
        return None
    a_tokens = free_runs["A_pt_raw"]["generated_token_ids"]
    return [int(x) for x in a_tokens[:step]], int(pt_token["token_id"]), int(it_token["token_id"])


def _class_for_token(token_id: int | None, y_pt: int, y_it: int) -> str:
    if token_id is None:
        return "missing"
    if int(token_id) == int(y_it):
        return "it"
    if int(token_id) == int(y_pt):
        return "pt"
    return "other"


def run_free_pair_until_first_diff(
    *,
    model_name: str,
    record: dict[str, Any],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    prompt_mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Generate pure PT/IT only until the first token disagreement.

    The validation analysis only uses the first divergent token and the shared
    prefix before it. Stopping here avoids spending most of the run budget on
    post-divergence continuations that are not read by the analysis.
    """

    state: dict[str, dict[str, Any]] = {}
    for name in FREE_CONDITIONS:
        condition = VALIDATION_CONDITIONS[name]
        tokenizer = tokenizers[condition.host_variant]
        prompt = _prompt_for_condition(
            record=record,
            condition=condition,
            tokenizer=tokenizer,
            prompt_mode=prompt_mode,
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        state[name] = {
            "condition": condition,
            "tokenizer": tokenizer,
            "model": models[condition.host_variant],
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long, device=device),
            "total_len": int(input_ids.shape[1]),
            "past_key_values": None,
            "stop_ids": set(steering_adapter.eos_token_ids(tokenizer)),
            "generated_ids": [],
            "generated_tokens": [],
            "stopped": False,
        }

    with torch.no_grad():
        for _step in range(max_new_tokens):
            step_ids: dict[str, int | None] = {}
            for name, payload in state.items():
                if payload["stopped"]:
                    step_ids[name] = None
                    continue
                outputs = _call_model(
                    model_name,
                    payload["model"],
                    payload["input_ids"],
                    payload["attention_mask"],
                    past_key_values=payload["past_key_values"],
                    use_cache=True,
                )
                payload["past_key_values"] = outputs.past_key_values
                variant = payload["condition"].host_variant
                next_id = _masked_next_token(outputs.logits[0, -1, :], real_token_masks[variant])
                payload["generated_ids"].append(next_id)
                payload["generated_tokens"].append(_decode_token(payload["tokenizer"], next_id))
                step_ids[name] = next_id
                if next_id in payload["stop_ids"]:
                    payload["stopped"] = True
                else:
                    payload["input_ids"] = torch.tensor([[next_id]], device=device)
                    payload["total_len"] += 1
                    payload["attention_mask"] = torch.ones(
                        (1, payload["total_len"]),
                        dtype=torch.long,
                        device=device,
                    )

            if step_ids["A_pt_raw"] != step_ids["C_it_chat"]:
                break
            if all(payload["stopped"] for payload in state.values()):
                break

    out = {}
    for name, payload in state.items():
        condition = payload["condition"]
        tokenizer = payload["tokenizer"]
        out[name] = {
            "condition": condition.name,
            "host_variant": condition.host_variant,
            "donor_variant": condition.donor_variant,
            "graft_kind": condition.graft_kind,
            "graft_window": None,
            "prompt_mode": prompt_mode,
            "n_steps": len(payload["generated_ids"]),
            "generated_token_ids": payload["generated_ids"],
            "generated_tokens": payload["generated_tokens"],
            "generated_text": tokenizer.decode(payload["generated_ids"], skip_special_tokens=True),
            "stopped_at_first_diff": True,
        }
    return out


def collect_prompt(
    *,
    model_name: str,
    record: dict[str, Any],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    prompt_mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt_id = str(record.get("id", record.get("record_id", "unknown")))
    if max_new_tokens <= 0:
        free_runs = {
            name: run_condition_free(
                model_name=model_name,
                condition=VALIDATION_CONDITIONS[name],
                record=record,
                models=models,
                tokenizers=tokenizers,
                real_token_masks=real_token_masks,
                steering_adapter=steering_adapter,
                device=device,
                prompt_mode=prompt_mode,
                max_new_tokens=max_new_tokens,
            )
            for name in FREE_CONDITIONS
        }
    else:
        free_runs = run_free_pair_until_first_diff(
            model_name=model_name,
            record=record,
            models=models,
            tokenizers=tokenizers,
            real_token_masks=real_token_masks,
            steering_adapter=steering_adapter,
            device=device,
            prompt_mode=prompt_mode,
            max_new_tokens=max_new_tokens,
        )

    events = find_divergence_events(
        free_runs["A_pt_raw"]["generated_tokens"],
        free_runs["C_it_chat"]["generated_tokens"],
    )

    readouts: dict[str, Any] = {}
    for kind in DIVERGENCE_KINDS:
        event = events.get(kind)
        if not isinstance(event, dict):
            readouts[kind] = None
            continue
        targets = _event_prefix_and_targets(event, free_runs)
        if targets is None:
            readouts[kind] = None
            continue
        prefix_ids, y_pt, y_it = targets
        condition_payloads = {
            name: run_condition_readout(
                model_name=model_name,
                condition=VALIDATION_CONDITIONS[name],
                record=record,
                prefix_token_ids=prefix_ids,
                y_pt=y_pt,
                y_it=y_it,
                models=models,
                tokenizers=tokenizers,
                real_token_masks=real_token_masks,
                steering_adapter=steering_adapter,
                device=device,
                prompt_mode=prompt_mode,
            )
            for name in READOUT_CONDITIONS
        }
        condition_token_at_step = {}
        for name, payload in condition_payloads.items():
            token_id = payload.get("final_argmax_token_id")
            tokenizer = tokenizers[payload["host_variant"]]
            condition_token_at_step[name] = {
                "token_id": token_id,
                "class": _class_for_token(token_id, y_pt, y_it),
                "decoded": _decode_token(tokenizer, int(token_id)) if token_id is not None else None,
            }
        readouts[kind] = {
            "event": event,
            "condition_token_at_step": condition_token_at_step,
            "conditions": condition_payloads,
        }

    return {
        "prompt_id": prompt_id,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "max_new_tokens": max_new_tokens,
        "free_runs": free_runs,
        "divergence_events": events,
        "readouts": readouts,
        "validation_conditions": READOUT_CONDITIONS,
    }


def run_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    spec = get_spec(args.model)
    steering_adapter = get_steering_adapter(args.model)
    pt_model, pt_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "pt"),
        args.device,
        multi_gpu=spec.multi_gpu,
    )
    it_model, it_tokenizer = load_model_and_tokenizer(
        model_id_for_variant(spec, "it"),
        args.device,
        multi_gpu=spec.multi_gpu,
    )
    models = {"pt": pt_model, "it": it_model}
    tokenizers = {"pt": pt_tokenizer, "it": it_tokenizer}
    real_token_masks = {
        "pt": steering_adapter.real_token_mask(pt_tokenizer, device, pt_model),
        "it": steering_adapter.real_token_mask(it_tokenizer, device, it_model),
    }
    records = load_dataset(
        args.dataset,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        n_examples=args.n_eval_examples,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"exp20_validation_records_w{args.worker_index}.jsonl"
    done = _read_done_ids_any_worker(args.out_dir)
    log.info(
        "[exp20-validation] %s/%s worker %d/%d resume: %d prompt ids already complete",
        args.prompt_mode,
        args.model,
        args.worker_index,
        args.n_workers,
        len(done),
    )
    with out_path.open("a") as fout:
        for idx, record in enumerate(records):
            prompt_id = str(record.get("id", record.get("record_id", f"rec_{idx}")))
            if prompt_id in done:
                continue
            try:
                result = collect_prompt(
                    model_name=args.model,
                    record=record,
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    steering_adapter=steering_adapter,
                    device=device,
                    prompt_mode=args.prompt_mode,
                    max_new_tokens=args.max_new_tokens,
                )
                fout.write(json.dumps(result) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[exp20-validation] prompt %s failed: %s", prompt_id, exc)
            if (idx + 1) % 10 == 0:
                log.info("[exp20-validation] %s %d/%d prompts", args.model, idx + 1, len(records))


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "exp20_validation_records.jsonl"
    seen: set[str] = set()
    with merged.open("w") as fout:
        for idx in range(n_workers):
            path = out_dir / f"exp20_validation_records_w{idx}.jsonl"
            if not path.exists():
                log.warning("Missing worker file: %s", path)
                continue
            with path.open("rb") as fin:
                for raw in fin:
                    if not raw.strip():
                        continue
                    try:
                        payload = json.loads(raw.decode("utf-8", errors="ignore"))
                        prompt_id = str(payload.get("prompt_id", ""))
                    except json.JSONDecodeError:
                        log.warning("Skipping malformed jsonl row in %s", path)
                        continue
                    if prompt_id and prompt_id in seen:
                        continue
                    if prompt_id:
                        seen.add(prompt_id)
                    fout.write(raw.decode("utf-8", errors="ignore"))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-mode", choices=["native", "raw_shared"], default="native")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merged = merge_workers(args.out_dir, args.n_workers)
        print(f"[exp20-validation] merged workers -> {merged}")
        return
    run_worker(args)


if __name__ == "__main__":
    main()
