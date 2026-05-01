from __future__ import annotations

import gzip
import json

from src.poc.exp28_late_mlp_crosscoder_mediation.analyze import interaction_from_cells
from src.poc.exp28_late_mlp_crosscoder_mediation.run_mediation import _done_keys, merge_workers


def test_interaction_from_cells() -> None:
    cells = {
        "U_IT__L_IT": 4.0,
        "U_IT__L_PT": 1.5,
        "U_PT__L_IT": 2.0,
        "U_PT__L_PT": 1.0,
    }
    assert interaction_from_cells(cells) == 1.5


def test_done_keys_and_merge_preserve_existing_compact_records(tmp_path) -> None:
    out_dir = tmp_path / "mediation"
    out_dir.mkdir()
    merged = out_dir / "records.jsonl.gz"
    old = {
        "prompt_id": "p0",
        "event_kind": "first_diff",
        "feature_set": "top_interaction",
        "k": 200,
        "control_seed": None,
    }
    new = {
        "prompt_id": "p1",
        "event_kind": "first_diff",
        "feature_set": "coverage_norm",
        "k": 90,
        "control_seed": None,
    }
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        fout.write(json.dumps(old) + "\n")
    worker = out_dir / "records_w0.jsonl.gz"
    with gzip.open(worker, "wt", encoding="utf-8") as fout:
        fout.write(json.dumps(new) + "\n")

    assert "p0|first_diff|top_interaction|200|None" in _done_keys(worker)
    merge_workers(out_dir, n_workers=1)
    with gzip.open(merged, "rt", encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin if line.strip()]
    assert rows == [old, new]
