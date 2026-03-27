"""
Shared utilities for cross-model replication study.

Centralises model loading, dataset loading, prompt extraction, and
JSONL I/O so that all collection scripts (L1L2, L8, L9) stay DRY.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


# ── model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_id: str,
    device: str | torch.device,
    *,
    eager_attn: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load a HuggingFace CausalLM and its tokenizer.

    Args:
        model_id:    HuggingFace model identifier.
        device:      Target device string or torch.device (e.g. "cuda:0").
        eager_attn:  If True, load with attn_implementation="eager" (required for
                     output_attentions=True in L9).  Flash/SDPA attention does NOT
                     return attention weight matrices.
        dtype:       Weight dtype (default: bfloat16 — matches exp2/exp6 setup).

    Returns:
        (model, tokenizer) both ready for inference.
    """
    kwargs: dict = {
        "dtype": dtype,
        "device_map": str(device),
        "trust_remote_code": True,
    }
    if eager_attn:
        kwargs["attn_implementation"] = "eager"

    log.info("Loading model %s on %s (eager_attn=%s)", model_id, device, eager_attn)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


# ── dataset loading ────────────────────────────────────────────────────────────

def load_dataset(
    path: str | Path,
    *,
    worker_index: int = 0,
    n_workers: int = 1,
    n_examples: int | None = None,
) -> list[dict]:
    """Load JSONL dataset and return this worker's slice.

    Slicing mirrors exp2/exp6 pattern: worker i processes records at
    indices [i, i+n_workers, i+2*n_workers, ...] (round-robin).

    Args:
        path:         Path to .jsonl dataset file.
        worker_index: 0-based worker index.
        n_workers:    Total number of workers.
        n_examples:   If set, cap the total record count before slicing.
    """
    with open(path) as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    if n_examples is not None:
        all_records = all_records[:n_examples]

    return all_records[worker_index::n_workers]


# ── prompt extraction ─────────────────────────────────────────────────────────

def get_raw_prompt(record: dict) -> str:
    """Extract the raw prompt text from a dataset record.

    Uses format B (question + 'Answer:' suffix) if available, else format A
    (multiple-choice), else the 'prompt' field directly.

    IMPORTANT: Cross-model L1/L8/L9 collection always uses RAW text — no
    chat template for either PT or IT models.  The plan requires identical
    tokenisation across PT and IT so the only difference is model weights.
    Chat templates are only applied in exp6 governance steering experiments.
    """
    formats = record.get("formats", {})
    return formats.get("B") or formats.get("A") or record.get("prompt", "")


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def read_done_ids(path: Path) -> set[str]:
    """Return the set of prompt_ids already written to a JSONL file (for resume)."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    done.add(json.loads(line)["prompt_id"])
                except (KeyError, json.JSONDecodeError):
                    pass
    return done


def merge_worker_jsonls(
    out_dir: Path,
    n_workers: int,
    worker_prefix: str,
    merged_name: str,
) -> list[dict]:
    """Concatenate per-worker JSONL files into a single merged file.

    Returns list of all merged records.
    """
    merged_path = out_dir / merged_name
    all_records: list[dict] = []

    with open(merged_path, "w") as fout:
        for w in range(n_workers):
            worker_path = out_dir / f"{worker_prefix}_w{w}.jsonl"
            if not worker_path.exists():
                log.warning("Missing worker file: %s", worker_path)
                continue
            with open(worker_path) as fin:
                for line in fin:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            fout.write(line)
                            all_records.append(rec)
                        except json.JSONDecodeError:
                            log.warning("Skipping corrupt JSON line in %s", worker_path)

    log.info("Merged %d records → %s", len(all_records), merged_path)
    return all_records
