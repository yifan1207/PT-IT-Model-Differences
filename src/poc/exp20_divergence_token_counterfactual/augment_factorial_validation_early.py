"""Augment Exp20 factorial-validation records with missing IT-host early swaps.

The holdout validation initially included IT-host mid, late, and mid+late PT
swaps. This add-on computes the missing early and early+mid PT-swap readouts at
the same first-divergence prefixes, without rerunning the completed pure PT/IT
free runs.
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
from src.poc.exp20_divergence_token_counterfactual.collect import run_condition_readout
from src.poc.exp20_divergence_token_counterfactual.collect_factorial_validation import (
    VALIDATION_CONDITIONS,
    _class_for_token,
)
from src.poc.exp20_divergence_token_counterfactual.metrics import DIVERGENCE_KINDS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EARLY_SWAP_CONDITIONS = ["D_early_ptswap", "D_earlymid_ptswap"]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw.decode("utf-8", errors="ignore"))


def _load_prompt_map(dataset: Path, n_examples: int | None) -> dict[str, dict[str, Any]]:
    records = load_dataset(dataset, n_examples=n_examples)
    return {
        str(record.get("id", record.get("record_id", f"rec_{idx}"))): record
        for idx, record in enumerate(records)
    }


def _baseline_rows(path: Path, worker_index: int, n_workers: int) -> list[dict[str, Any]]:
    rows = list(_iter_jsonl(path))
    return rows[worker_index::n_workers]


def _read_done_ids_any_worker(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for path in sorted(out_dir.glob("exp20_validation_early_records_w*.jsonl")):
        done.update(read_done_ids(path))
    return done


def _targets_for_event(record: dict[str, Any], event: dict[str, Any]) -> tuple[list[int], int, int] | None:
    pt_token = event.get("pt_token")
    it_token = event.get("it_token")
    step = event.get("step")
    if not isinstance(pt_token, dict) or not isinstance(it_token, dict) or not isinstance(step, int):
        return None
    a_tokens = ((record.get("free_runs") or {}).get("A_pt_raw") or {}).get("generated_token_ids")
    if not isinstance(a_tokens, list):
        return None
    return [int(x) for x in a_tokens[:step]], int(pt_token["token_id"]), int(it_token["token_id"])


def augment_prompt(
    *,
    model_name: str,
    baseline: dict[str, Any],
    record: dict[str, Any],
    models: dict[str, Any],
    tokenizers: dict[str, Any],
    real_token_masks: dict[str, torch.Tensor],
    steering_adapter: Any,
    device: torch.device,
    prompt_mode: str,
) -> dict[str, Any]:
    readouts: dict[str, Any] = {}
    events = baseline.get("divergence_events") or {}
    for kind in DIVERGENCE_KINDS:
        event = events.get(kind)
        if not isinstance(event, dict):
            readouts[kind] = None
            continue
        targets = _targets_for_event(baseline, event)
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
            for name in EARLY_SWAP_CONDITIONS
        }
        readouts[kind] = {
            "condition_token_at_step": {
                name: {
                    "token_id": payload.get("final_argmax_token_id"),
                    "class": _class_for_token(payload.get("final_argmax_token_id"), y_pt, y_it),
                }
                for name, payload in condition_payloads.items()
            },
            "conditions": condition_payloads,
        }
    return {
        "prompt_id": str(baseline["prompt_id"]),
        "model": model_name,
        "prompt_mode": prompt_mode,
        "readouts": readouts,
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
    prompt_map = _load_prompt_map(args.dataset, args.n_eval_examples)
    baselines = _baseline_rows(args.baseline_jsonl, args.worker_index, args.n_workers)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"exp20_validation_early_records_w{args.worker_index}.jsonl"
    done = _read_done_ids_any_worker(args.out_dir)
    log.info(
        "[exp20-validation-early] %s/%s worker %d/%d resume: %d prompt ids already complete",
        args.prompt_mode,
        args.model,
        args.worker_index,
        args.n_workers,
        len(done),
    )
    with out_path.open("a") as fout:
        for idx, baseline in enumerate(baselines):
            prompt_id = str(baseline.get("prompt_id", ""))
            if prompt_id in done:
                continue
            record = prompt_map.get(prompt_id)
            if record is None:
                log.warning("[exp20-validation-early] missing dataset record for prompt_id=%s", prompt_id)
                continue
            try:
                result = augment_prompt(
                    model_name=args.model,
                    baseline=baseline,
                    record=record,
                    models=models,
                    tokenizers=tokenizers,
                    real_token_masks=real_token_masks,
                    steering_adapter=steering_adapter,
                    device=device,
                    prompt_mode=args.prompt_mode,
                )
                fout.write(json.dumps(result) + "\n")
                fout.flush()
            except Exception as exc:
                log.exception("[exp20-validation-early] prompt %s failed: %s", prompt_id, exc)
            if (idx + 1) % 25 == 0:
                log.info("[exp20-validation-early] %s %d/%d prompts", args.model, idx + 1, len(baselines))


def merge_workers(out_dir: Path, n_workers: int) -> Path:
    merged = out_dir / "exp20_validation_early_records.jsonl"
    seen: set[str] = set()
    with merged.open("w") as fout:
        for idx in range(n_workers):
            path = out_dir / f"exp20_validation_early_records_w{idx}.jsonl"
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
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-eval-examples", type=int, default=600)
    parser.add_argument("--prompt-mode", choices=["native", "raw_shared"], default="native")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merged = merge_workers(args.out_dir, args.n_workers)
        print(f"[exp20-validation-early] merged workers -> {merged}")
        return
    run_worker(args)


if __name__ == "__main__":
    main()
