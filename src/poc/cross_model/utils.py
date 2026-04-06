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


# ── DynamicCache compatibility (transformers ≥4.38) ───────────────────────────
# DeepSeek-V2 uses internal cache attributes that were removed/renamed.
# Patch once at import time so all collectors benefit without code duplication.
def _patch_dynamic_cache() -> None:
    try:
        from transformers.cache_utils import DynamicCache as _DC
        if not hasattr(_DC, "seen_tokens"):
            _DC.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(_DC, "get_usable_length"):
            _DC.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
        if not hasattr(_DC, "get_max_length"):
            _DC.get_max_length = lambda self: None
    except Exception:
        pass


_patch_dynamic_cache()
# ─────────────────────────────────────────────────────────────────────────────


# ── model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_id: str,
    device: str | torch.device,
    *,
    eager_attn: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    multi_gpu: bool = False,
) -> tuple:
    """Load a HuggingFace CausalLM and its tokenizer.

    Args:
        model_id:    HuggingFace model identifier.
        device:      Target device string or torch.device (e.g. "cuda:0").
                     Ignored when multi_gpu=True.
        eager_attn:  If True, load with attn_implementation="eager" (required for
                     output_attentions=True in L9 and for DeepSeek V2 MLA).
        dtype:       Weight dtype (default: bfloat16 — matches exp2/exp6 setup).
        multi_gpu:   If True, spread the model across all visible GPUs using
                     device_map="sequential" (for models too large for one GPU).
                     Each GPU gets at most 72 GB to leave room for KV cache.

    Returns:
        (model, tokenizer) both ready for inference.
    """
    kwargs: dict = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    if multi_gpu:
        n_gpus = torch.cuda.device_count()
        kwargs["device_map"] = "sequential"
        kwargs["max_memory"] = {i: "72GB" for i in range(n_gpus)}
        kwargs["attn_implementation"] = "eager"  # required for DeepSeek MLA
        log.info("Loading model %s across %d GPUs (sequential device_map)", model_id, n_gpus)
    else:
        kwargs["device_map"] = str(device)
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

    Returns raw text without any chat template wrapping.
    """
    formats = record.get("formats", {})
    return formats.get("B") or formats.get("A") or record.get("prompt", "")


def get_prompt_for_variant(
    record: dict,
    *,
    variant: str,
    tokenizer=None,
    apply_chat_template: bool = True,
) -> str:
    """Get prompt text appropriate for the model variant.

    For PT models (or when apply_chat_template=False): returns raw text.
    For IT models (when apply_chat_template=True): wraps with the model's
    native chat template, which is the IT model's trained distribution.

    Args:
        record:              Dataset record with 'formats' and/or 'prompt' fields.
        variant:             'pt' or 'it'.
        tokenizer:           HuggingFace tokenizer (required for IT with chat template).
        apply_chat_template: If False, always returns raw text (ablation mode).
    """
    raw = get_raw_prompt(record)

    if variant == "pt" or not apply_chat_template:
        return raw

    # IT variant with chat template
    if tokenizer is None:
        raise ValueError("tokenizer required for IT variant with chat template")

    messages = [{"role": "user", "content": raw}]
    try:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return templated
    except Exception as e:
        log.warning("Chat template failed for record, falling back to raw: %s", e)
        return raw


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
