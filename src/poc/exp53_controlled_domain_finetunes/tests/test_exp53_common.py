from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.poc.exp53_controlled_domain_finetunes.common import approx_tokens, paths_for, text_hash


def test_paths_for_aliases() -> None:
    paths = paths_for(Path("results/x"), "code")
    assert paths.adapter_dir.as_posix().endswith("checkpoints/llama31_code_cpt_lora/adapter")
    assert paths.merged_dir.as_posix().endswith("models/llama31_code_cpt_lora_merged")


def test_text_hash_is_stable() -> None:
    assert text_hash("abc") == text_hash("abc")
    assert text_hash("abc") != text_hash("abd")


def test_approx_tokens_positive() -> None:
    assert approx_tokens("") == 1
    assert approx_tokens("a" * 400) >= 100
