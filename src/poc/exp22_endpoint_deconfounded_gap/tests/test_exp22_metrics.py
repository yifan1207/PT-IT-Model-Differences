from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp22_endpoint_deconfounded_gap.metrics import (
    distribution_arrays_from_logits,
    exact_js_from_logprobs,
    exact_kl_from_logprobs,
    final_top1_stable_top5_entry,
    future_top1_flips,
    remaining_adjacent_path,
    top5_churn,
)


def test_distribution_arrays_exact_endpoint_metrics() -> None:
    logits = torch.tensor(
        [
            [0.0, 0.0, -2.0],
            [1.0, 0.0, -2.0],
            [2.0, 0.0, -2.0],
        ]
    )
    arrays = distribution_arrays_from_logits(logits, top_k=2)
    assert arrays["kl_to_final"][-1] == 0.0
    assert arrays["confidence"][-1] > arrays["confidence"][0]
    assert arrays["top1_margin"][-1] > arrays["top1_margin"][0]
    assert arrays["top5_ids"][-1][0] == 0
    assert arrays["adjacent_kl"][-1] is None
    assert arrays["adjacent_js"][-1] is None


def test_exact_kl_and_js_sign_conventions() -> None:
    logp = torch.log_softmax(torch.tensor([2.0, 0.0]), dim=-1)
    logq = torch.log_softmax(torch.tensor([0.0, 2.0]), dim=-1)
    assert exact_kl_from_logprobs(logp, logp).item() == 0.0
    assert exact_kl_from_logprobs(logp, logq).item() > 0.0
    assert math.isclose(
        exact_js_from_logprobs(logp, logq).item(),
        exact_js_from_logprobs(logq, logp).item(),
        rel_tol=1e-6,
    )


def test_endpoint_free_path_churn_and_flips() -> None:
    adjacent = [0.1] * 9 + [None]
    # Late start for 10 layers is 8, so only layer 8 has remaining path 0.1.
    assert math.isclose(remaining_adjacent_path(adjacent), 0.1, rel_tol=1e-6)

    top1 = [[0], [0], [0], [1], [1], [1], [2], [2], [2], [3]]
    assert future_top1_flips(top1) == 1.0

    stable = [[9, 1, 2, 3, 4] for _ in range(10)]
    changed = [row.copy() for row in stable]
    changed[8] = [8, 1, 2, 3, 4]
    assert top5_churn(changed) > 0.0


def test_stable_top5_entry_depth_for_final_top1() -> None:
    top1 = [[1], [2], [3], [4], [4]]
    top5 = [
        [9, 8, 7, 6, 5],
        [4, 8, 7, 6, 5],
        [4, 3, 2, 1, 0],
        [4, 3, 2, 1, 0],
        [4, 3, 2, 1, 0],
    ]
    assert final_top1_stable_top5_entry(top1, top5) == 0.25

