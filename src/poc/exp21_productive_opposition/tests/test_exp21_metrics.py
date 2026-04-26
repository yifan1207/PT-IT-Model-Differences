from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp21_productive_opposition.metrics import (
    collapse_category,
    mlp_delta_cosine,
    negative_parallel_component,
    negative_parallel_norm,
    productive_opposition,
    summarize_categories,
)


def test_delta_cosine_sign_convention() -> None:
    residual = torch.tensor([1.0, 0.0])
    assert mlp_delta_cosine(torch.tensor([-2.0, 0.0]), residual) == -1.0
    assert mlp_delta_cosine(torch.tensor([2.0, 0.0]), residual) == 1.0


def test_negative_parallel_component_only_keeps_opposing_projection() -> None:
    residual = torch.tensor([2.0, 0.0])
    update = torch.tensor([-3.0, 4.0])
    component = negative_parallel_component(update, residual)
    assert torch.allclose(component, torch.tensor([-3.0, 0.0]))
    assert negative_parallel_norm(update, residual) == 3.0
    assert torch.allclose(
        negative_parallel_component(torch.tensor([3.0, 4.0]), residual),
        torch.zeros(2),
    )


def test_productive_opposition_requires_negative_and_positive_margin() -> None:
    assert productive_opposition(-0.2, 1.5) is True
    assert productive_opposition(0.2, 1.5) is False
    assert productive_opposition(-0.2, -1.5) is False
    assert productive_opposition(None, 1.5) is None


def test_category_collapse_and_summary() -> None:
    assert collapse_category("STRUCTURAL") == "FORMAT"
    assert collapse_category("CONTENT") == "CONTENT"
    assert collapse_category("FUNCTION") == "FUNCTION_OTHER"
    assert summarize_categories(["STRUCTURAL", "CONTENT", "FUNCTION", "PUNCTUATION"]) == {
        "FORMAT": 2,
        "CONTENT": 1,
        "FUNCTION_OTHER": 1,
    }
