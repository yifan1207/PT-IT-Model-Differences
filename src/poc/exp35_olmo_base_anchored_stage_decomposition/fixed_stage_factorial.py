"""Run the Exp35 fixed-support upstream-state x late-stage 4x4 factorial."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.poc.exp23_midlate_interaction_suite.boundary import BoundaryStatePatch, LayerResidualCapture
from src.poc.exp35_olmo_base_anchored_stage_decomposition import (
    DEFAULT_BOUNDARY_LAYER,
    STAGE_ORDER,
)
from src.poc.exp35_olmo_base_anchored_stage_decomposition.common import (
    cell_name,
    json_rows,
    load_stage,
    stage_adapter,
    stage_key_for_label,
    validate_full_prefix_boundary,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_support(path: Path, n_examples: int | None, worker_index: int, n_workers: int) -> list[dict[str, Any]]:
    rows = [row for row in json_rows(path) if row.get("valid")]
    if n_examples is not None:
        rows = rows[:n_examples]
    return rows[worker_index::n_workers]


def _load_stage_cache(cache_root: Path, stage: str) -> dict[str, dict[str, Any]]:
    stage = stage_key_for_label(stage)
    stage_dir = cache_root / stage
    out: dict[str, dict[str, Any]] = {}
    record_files = sorted(stage_dir.glob("records_w*.jsonl.gz"))
    if not record_files and (stage_dir / "records.jsonl.gz").exists():
        record_files = [stage_dir / "records.jsonl.gz"]
    state_cache: dict[str, dict[str, Any]] = {}
    for rec_path in record_files:
        for row in json_rows(rec_path):
            state_file = str(row["state_file"])
            if state_file not in state_cache:
                state_cache[state_file] = torch.load(stage_dir / state_file, map_location="cpu")
            payload = state_cache[state_file]
            idx = int(row["state_index"])
            key = str(row["key"])
            out[key] = {
                "record": row,
                "boundary_state": payload["boundary_states"][idx],
                "final_hidden": payload["final_hidden"][idx],
            }
    if not out:
        raise FileNotFoundError(f"No stage cache records loaded for {stage} from {stage_dir}")
    return out


@torch.no_grad()
def _forward_with_boundary(
    *,
    model: Any,
    layers: list[torch.nn.Module],
    adapter: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    boundary_layer: int,
    donor_boundary_state: torch.Tensor,
    noop_tol: float | None,
) -> dict[str, Any]:
    validate_full_prefix_boundary(donor_boundary_state, expected_seq_len=int(input_ids.shape[1]))
    residual_capture = LayerResidualCapture(layers=layers, adapter=adapter)
    patcher = BoundaryStatePatch(layers[boundary_layer], donor_boundary_state, noop_tol=noop_tol)
    try:
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        residuals = residual_capture.snapshot()
    finally:
        patcher.close()
        residual_capture.close()
    final_hidden = residuals[-1].detach().cpu().to(torch.bfloat16)
    return {
        "final_hidden": final_hidden,
        "patch_n": patcher.n_patches,
        "patch_input_delta_max_abs": patcher.last_max_abs_input_delta,
    }


@torch.no_grad()
def run_host_worker(args: argparse.Namespace) -> None:
    host_stage = stage_key_for_label(args.host_stage)
    device = torch.device(args.device)
    model, _tokenizer = load_stage(host_stage, device)
    adapter_bundle = stage_adapter()
    adapter = adapter_bundle.adapter
    layers = adapter_bundle.get_layers(model)
    boundary_layer = int(args.boundary_layer)
    if boundary_layer < 0 or boundary_layer >= len(layers):
        raise ValueError(f"boundary_layer={boundary_layer} outside stage={host_stage} layers={len(layers)}")

    support_rows = _load_support(args.support, args.n_examples, args.worker_index, args.n_workers)
    caches = {stage: _load_stage_cache(args.stage_cache, stage) for stage in STAGE_ORDER}

    out_dir = args.out_dir / host_stage
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"cells_w{args.worker_index}.jsonl.gz"
    hiddens_path = out_dir / f"final_hiddens_w{args.worker_index}.pt"

    hidden_rows: list[torch.Tensor] = []
    records_out: list[dict[str, Any]] = []
    log.info(
        "[exp35] fixed factorial host=%s worker=%d/%d rows=%d",
        host_stage,
        args.worker_index,
        args.n_workers,
        len(support_rows),
    )
    for idx, row in enumerate(support_rows):
        key = f"{row['prompt_id']}::{row.get('event_kind', 'first_diff')}"
        full_ids = [int(x) for x in row["full_input_ids"]]
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        for upstream_stage in STAGE_ORDER:
            upstream_cache = caches[upstream_stage].get(key)
            if upstream_cache is None:
                log.warning("[exp35] missing upstream cache stage=%s key=%s", upstream_stage, key)
                continue
            donor_boundary = upstream_cache["boundary_state"]
            noop_tol = float(args.noop_tol) if upstream_stage == host_stage else None
            try:
                cell = _forward_with_boundary(
                    model=model,
                    layers=layers,
                    adapter=adapter,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    boundary_layer=boundary_layer,
                    donor_boundary_state=donor_boundary,
                    noop_tol=noop_tol,
                )
                final_hidden = cell["final_hidden"]
                hidden_index = len(hidden_rows)
                hidden_rows.append(final_hidden)
                diag_delta = None
                if upstream_stage == host_stage:
                    direct = upstream_cache["final_hidden"].to(dtype=torch.bfloat16)
                    diag_delta = float((final_hidden.float() - direct.float()).abs().max().item())
                records_out.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "event_kind": row.get("event_kind", "first_diff"),
                        "key": key,
                        "upstream_stage": upstream_stage,
                        "late_stage": host_stage,
                        "cell": cell_name(upstream_stage, host_stage),
                        "hidden_file": hiddens_path.name,
                        "hidden_index": hidden_index,
                        "boundary_layer": boundary_layer,
                        "full_length": len(full_ids),
                        "divergence_step": row.get("divergence_step"),
                        "prompt_category": row.get("prompt_category"),
                        "source": row.get("source"),
                        "t_base_id": int(row["t_base"]["token_id"]),
                        "t_rlvr_id": int(row["t_rlvr"]["token_id"]),
                        "t_base_category": row["t_base"].get("token_category_collapsed"),
                        "t_rlvr_category": row["t_rlvr"].get("token_category_collapsed"),
                        "t_base_assistant_marker": row["t_base"].get("assistant_marker"),
                        "t_rlvr_assistant_marker": row["t_rlvr"].get("assistant_marker"),
                        "patch_n": cell["patch_n"],
                        "patch_input_delta_max_abs": cell["patch_input_delta_max_abs"],
                        "diagonal_final_hidden_max_abs_delta": diag_delta,
                    }
                )
            except Exception as exc:
                log.exception("[exp35] fixed cell host=%s upstream=%s key=%s failed: %s", host_stage, upstream_stage, key, exc)
        if (idx + 1) % 20 == 0:
            log.info("[exp35] fixed host=%s %d/%d", host_stage, idx + 1, len(support_rows))

    torch.save(
        {
            "host_stage": host_stage,
            "worker_index": args.worker_index,
            "boundary_layer": boundary_layer,
            "final_hidden": torch.stack(hidden_rows, dim=0) if hidden_rows else torch.empty(0),
        },
        hiddens_path,
    )
    with gzip.open(records_path, "wt", encoding="utf-8") as fout:
        for rec in records_out:
            fout.write(json.dumps(rec, separators=(",", ":")) + "\n")
    log.info("[exp35] fixed host=%s wrote cells=%d", host_stage, len(records_out))


def merge_host(out_dir: Path, host_stage: str, n_workers: int) -> Path:
    host_stage = stage_key_for_label(host_stage)
    host_dir = out_dir / host_stage
    merged = host_dir / "cells.jsonl.gz"
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(n_workers):
            path = host_dir / f"cells_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp35] missing fixed worker file %s", path)
                continue
            for row in json_rows(path):
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-stage", required=True, choices=list(STAGE_ORDER))
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--stage-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--boundary-layer", type=int, default=DEFAULT_BOUNDARY_LAYER)
    parser.add_argument("--noop-tol", type=float, default=1e-4)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_host(args.out_dir, args.host_stage, args.n_workers)
        return
    run_host_worker(args)


if __name__ == "__main__":
    main()
