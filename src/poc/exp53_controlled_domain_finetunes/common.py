"""Shared utilities for Exp53 controlled domain fine-tunes."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B"
BASE_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

DOMAIN_TO_ALIAS = {
    "code": "llama31_code_cpt_lora",
    "biomed": "llama31_biomed_cpt_lora",
}

PERMISSIVE_CODE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "cc0-1.0",
    "unlicense",
}


@dataclass(frozen=True)
class DomainPaths:
    root: Path
    domain: str

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / self.domain

    @property
    def train_jsonl(self) -> Path:
        return self.data_dir / "train.jsonl.gz"

    @property
    def eval_jsonl(self) -> Path:
        return self.data_dir / "eval.jsonl.gz"

    @property
    def support_jsonl(self) -> Path:
        return self.data_dir / "support_dataset.jsonl"

    @property
    def data_manifest(self) -> Path:
        return self.data_dir / "data_manifest.json"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints" / DOMAIN_TO_ALIAS[self.domain]

    @property
    def adapter_dir(self) -> Path:
        return self.checkpoint_dir / "adapter"

    @property
    def merged_dir(self) -> Path:
        return self.root / "models" / f"{DOMAIN_TO_ALIAS[self.domain]}_merged"

    @property
    def train_log_jsonl(self) -> Path:
        return self.checkpoint_dir / "train_metrics.jsonl"

    @property
    def train_manifest(self) -> Path:
        return self.checkpoint_dir / "train_manifest.json"

    @property
    def eval_loss(self) -> Path:
        return self.checkpoint_dir / "eval_loss.json"

    @property
    def merge_check(self) -> Path:
        return self.checkpoint_dir / "merge_equivalence.json"

    @property
    def generation_health(self) -> Path:
        return self.checkpoint_dir / "generation_health.json"


def paths_for(root: Path, domain: str) -> DomainPaths:
    if domain not in DOMAIN_TO_ALIAS:
        raise ValueError(f"unknown Exp53 domain {domain!r}; expected {sorted(DOMAIN_TO_ALIAS)}")
    return DomainPaths(root=root, domain=domain)


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open_text(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "at") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def approx_tokens(text: str) -> int:
    # Conservative enough for Llama token budgeting without forcing a slow full
    # tokenization pass during data streaming.
    return max(1, len(text.encode("utf-8", errors="ignore")) // 4)

