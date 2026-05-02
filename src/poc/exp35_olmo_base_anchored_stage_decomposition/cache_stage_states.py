"""Cache OLMo stage full-prefix boundary states and last-token diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.poc.exp11_matched_prefix_mlp_graft.mlp_graft import ArchitectureProbe, PipelineCapture
from src.poc.exp23_midlate_interaction_suite.boundary import BoundaryStateCapture
from src.poc.exp35_olmo_base_anchored_stage_decomposition import DEFAULT_BOUNDARY_LAYER
from src.poc.exp35_olmo_base_anchored_stage_decomposition.common import (
    json_rows,
    load_stage,
    real_token_mask_for,
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


def _stack(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([value.detach().cpu().to(torch.bfloat16) for value in values], dim=0)


def _cosine_series(a: list[torch.Tensor], b: list[torch.Tensor]) -> list[float]:
    out = []
    for x, y in zip(a, b, strict=False):
        val = F.cosine_similarity(x.float().view(1, -1), y.float().view(1, -1), dim=-1)[0]
        out.append(float(val.item()))
    return out


@torch.no_grad()
def collect_stage_cache(args: argparse.Namespace) -> None:
    stage = stage_key_for_label(args.stage)
    device = torch.device(args.device)
    model, tokenizer = load_stage(stage, device)
    adapter = stage_adapter()
    layers = adapter.get_layers(model)
    boundary_layer = int(args.boundary_layer)
    if boundary_layer < 0 or boundary_layer >= len(layers):
        raise ValueError(f"boundary_layer={boundary_layer} outside stage={stage} layers={len(layers)}")
    real_mask = real_token_mask_for(model, tokenizer, device)
    rows = _load_support(args.support, args.n_examples, args.worker_index, args.n_workers)
    out_dir = args.out_dir / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"records_w{args.worker_index}.jsonl.gz"
    states_path = out_dir / f"states_w{args.worker_index}.pt"

    keys: list[str] = []
    boundary_states: list[torch.Tensor] = []
    final_hidden: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    pre_mlp: list[torch.Tensor] = []
    mlp_output: list[torch.Tensor] = []
    out_records: list[dict[str, Any]] = []

    log.info("[exp35] cache stage=%s worker=%d/%d rows=%d", stage, args.worker_index, args.n_workers, len(rows))
    for idx, row in enumerate(rows):
        key = f"{row['prompt_id']}::{row.get('event_kind', 'first_diff')}"
        try:
            full_ids = [int(x) for x in row["full_input_ids"]]
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            boundary_capture = BoundaryStateCapture(layers[boundary_layer])
            pipe_capture = PipelineCapture(
                model_raw=model,
                adapter=adapter,
                arch_probe=ArchitectureProbe(),
            )
            try:
                pipe_capture.reset_step()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                boundary = boundary_capture.snapshot().detach().cpu().to(torch.bfloat16)
                validate_full_prefix_boundary(boundary, expected_seq_len=len(full_ids))
                snap = pipe_capture.snapshot()
            finally:
                boundary_capture.close()
                pipe_capture.close()

            final = snap.residual_output[-1][0].detach().cpu().to(torch.bfloat16)
            logits = outputs.logits[0, -1, :].detach().float()
            logits[~real_mask] = float("-inf")
            top1 = int(torch.argmax(logits).item())
            t_base = int(row["t_base"]["token_id"])
            t_rlvr = int(row["t_rlvr"]["token_id"])
            margin = float((logits[t_rlvr] - logits[t_base]).item())
            state_index = len(keys)
            keys.append(key)
            boundary_states.append(boundary)
            final_hidden.append(final)
            res_stack = _stack([x[0] for x in snap.residual_output])
            pre_stack = _stack([x[0] for x in snap.pre_mlp_residual])
            mlp_stack = _stack([x[0] for x in snap.mlp_output])
            residuals.append(res_stack)
            pre_mlp.append(pre_stack)
            mlp_output.append(mlp_stack)
            delta_cosine = _cosine_series([x[0] for x in snap.mlp_output], [x[0] for x in snap.pre_mlp_residual])
            out_records.append(
                {
                    "prompt_id": row["prompt_id"],
                    "event_kind": row.get("event_kind", "first_diff"),
                    "key": key,
                    "stage": stage,
                    "state_file": states_path.name,
                    "state_index": state_index,
                    "full_length": len(full_ids),
                    "boundary_layer": boundary_layer,
                    "boundary_shape": list(boundary.shape),
                    "native_top1_id": top1,
                    "native_top1_class": "rlvr" if top1 == t_rlvr else ("base" if top1 == t_base else "other"),
                    "native_margin_rlvr_minus_base": margin,
                    "delta_cosine": delta_cosine,
                    "late_delta_cosine_mean": float(
                        sum(delta_cosine[boundary_layer:]) / max(1, len(delta_cosine[boundary_layer:]))
                    ),
                    "terminal_delta_cosine": float(delta_cosine[-1]),
                }
            )
        except Exception as exc:
            log.exception("[exp35] cache stage=%s key=%s failed: %s", stage, key, exc)
        if (idx + 1) % 20 == 0:
            log.info("[exp35] cache stage=%s %d/%d", stage, idx + 1, len(rows))

    torch.save(
        {
            "stage": stage,
            "boundary_layer": boundary_layer,
            "keys": keys,
            "boundary_states": boundary_states,
            "final_hidden": torch.stack(final_hidden, dim=0) if final_hidden else torch.empty(0),
            "residuals": torch.stack(residuals, dim=0) if residuals else torch.empty(0),
            "pre_mlp": torch.stack(pre_mlp, dim=0) if pre_mlp else torch.empty(0),
            "mlp_output": torch.stack(mlp_output, dim=0) if mlp_output else torch.empty(0),
        },
        states_path,
    )
    with gzip.open(records_path, "wt", encoding="utf-8") as fout:
        for rec in out_records:
            fout.write(json.dumps(rec, separators=(",", ":")) + "\n")
    log.info("[exp35] cache stage=%s wrote records=%d states=%s", stage, len(out_records), states_path)


def merge_stage_cache(out_dir: Path, stage: str, n_workers: int) -> Path:
    stage = stage_key_for_label(stage)
    stage_dir = out_dir / stage
    merged = stage_dir / "records.jsonl.gz"
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(n_workers):
            path = stage_dir / f"records_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp35] missing stage cache record file %s", path)
                continue
            for row in json_rows(path):
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["B", "S", "D", "R"])
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--boundary-layer", type=int, default=DEFAULT_BOUNDARY_LAYER)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_stage_cache(args.out_dir, args.stage, args.n_workers)
        return
    collect_stage_cache(args)


if __name__ == "__main__":
    main()

