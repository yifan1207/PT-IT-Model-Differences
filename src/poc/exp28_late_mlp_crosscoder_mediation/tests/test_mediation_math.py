from __future__ import annotations

from src.poc.exp28_late_mlp_crosscoder_mediation.analyze import interaction_from_cells


def test_interaction_from_cells() -> None:
    cells = {
        "U_IT__L_IT": 4.0,
        "U_IT__L_PT": 1.5,
        "U_PT__L_IT": 2.0,
        "U_PT__L_PT": 1.0,
    }
    assert interaction_from_cells(cells) == 1.5

