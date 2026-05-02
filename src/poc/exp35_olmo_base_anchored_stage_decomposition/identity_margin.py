"""Fixed-support Exp20-style identity/margin decomposition across OLMo stages."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.poc.exp11_matched_prefix_mlp_graft.mlp_graft import ArchitectureProbe, PipelineCapture
from src.poc.exp20_divergence_token_counterfactual.collect import DEPTH_ABLATION_WINDOWS
from src.poc.exp35_olmo_base_anchored_stage_decomposition import MODEL_NAME, NON_BASE_STAGES
from src.poc.exp35_olmo_base_anchored_stage_decomposition.common import (
    json_rows,
    load_stage,
    real_token_mask_for,
    stage_adapter,
    stage_key_for_label,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_support(path: Path, n_examples: int | None, worker_index: int, n_workers: int) -> list[dict[str, Any]]:
    rows = [row for row in json_rows(path) if row.get("valid")]
    if n_examples is not None:
        rows = rows[:n_examples]
    return rows[worker_index::n_workers]


def _window(kind: str) -> tuple[int, int] | None:
    if kind == "none":
        return None
    if kind == "midlate":
        mid = DEPTH_ABLATION_WINDOWS[MODEL_NAME]["mid"]
        late = DEPTH_ABLATION_WINDOWS[MODEL_NAME]["late"]
        return min(mid[0], late[0]), max(mid[1], late[1])
    return DEPTH_ABLATION_WINDOWS[MODEL_NAME][kind]


@torch.no_grad()
def _condition_forward(
    *,
    condition: str,
    host_stage: str,
    donor_stage: str | None,
    graft_kind: str,
    models: dict[str, Any],
    masks: dict[str, torch.Tensor],
    adapter: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    t_base: int,
    t_rlvr: int,
) -> dict[str, Any]:
    graft = _window(graft_kind)
    capture = PipelineCapture(
        model_raw=models[host_stage],
        adapter=adapter,
        arch_probe=ArchitectureProbe(),
        graft_start_layer=graft[0] if graft else None,
        graft_end_layer_exclusive=graft[1] if graft else None,
        graft_it_model_raw=models[donor_stage] if donor_stage else None,
    )
    try:
        capture.reset_step()
        outputs = models[host_stage](input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        capture.close()
    logits = outputs.logits[0, -1, :].detach().float()
    logits[~masks[host_stage]] = float("-inf")
    top1 = int(torch.argmax(logits).item())
    return {
        "condition": condition,
        "host_stage": host_stage,
        "donor_stage": donor_stage,
        "graft_kind": graft_kind,
        "graft_window": list(graft) if graft else None,
        "final_top1_id": top1,
        "winner": "rlvr" if top1 == t_rlvr else ("base" if top1 == t_base else "other"),
        "margin_rlvr_minus_base": float((logits[t_rlvr] - logits[t_base]).item()),
        "rlvr_logit": float(logits[t_rlvr].item()),
        "base_logit": float(logits[t_base].item()),
    }


@torch.no_grad()
def run_identity_worker(args: argparse.Namespace) -> None:
    target_stage = stage_key_for_label(args.target_stage)
    if target_stage == "B":
        raise ValueError("target_stage must be S/D/R")
    device = torch.device(args.device)
    base_model, _base_tok = load_stage("B", device)
    target_model, _target_tok = load_stage(target_stage, device)
    models = {"B": base_model, target_stage: target_model}
    masks = {
        "B": real_token_mask_for(base_model, _base_tok, device),
        target_stage: real_token_mask_for(target_model, _target_tok, device),
    }
    adapter = stage_adapter()
    rows = _load_support(args.support, args.n_examples, args.worker_index, args.n_workers)
    out_dir = args.out_dir / target_stage
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"identity_margin_w{args.worker_index}.jsonl.gz"
    log.info("[exp35] identity target=%s worker=%d/%d rows=%d", target_stage, args.worker_index, args.n_workers, len(rows))
    conditions = [
        ("A_base", "B", None, "none"),
        ("C_stage", target_stage, None, "none"),
        ("B_mid", "B", target_stage, "mid"),
        ("B_late", "B", target_stage, "late"),
        ("B_midlate", "B", target_stage, "midlate"),
        ("D_mid", target_stage, "B", "mid"),
        ("D_late", target_stage, "B", "late"),
        ("D_midlate", target_stage, "B", "midlate"),
    ]
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for idx, row in enumerate(rows):
            full_ids = [int(x) for x in row["full_input_ids"]]
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            t_base = int(row["t_base"]["token_id"])
            t_rlvr = int(row["t_rlvr"]["token_id"])
            payload = {
                "prompt_id": row["prompt_id"],
                "event_kind": row.get("event_kind", "first_diff"),
                "target_stage": target_stage,
                "divergence_step": row.get("divergence_step"),
                "prompt_category": row.get("prompt_category"),
                "t_base_id": t_base,
                "t_rlvr_id": t_rlvr,
                "t_base_category": row["t_base"].get("token_category_collapsed"),
                "t_rlvr_category": row["t_rlvr"].get("token_category_collapsed"),
                "conditions": {},
            }
            for name, host, donor, graft_kind in conditions:
                try:
                    payload["conditions"][name] = _condition_forward(
                        condition=name,
                        host_stage=host,
                        donor_stage=donor,
                        graft_kind=graft_kind,
                        models=models,
                        masks=masks,
                        adapter=adapter,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        t_base=t_base,
                        t_rlvr=t_rlvr,
                    )
                except Exception as exc:
                    log.exception("[exp35] identity target=%s condition=%s prompt=%s failed: %s", target_stage, name, row["prompt_id"], exc)
                    payload["conditions"][name] = {"condition": name, "error": str(exc)}
            fout.write(json.dumps(payload, separators=(",", ":")) + "\n")
            fout.flush()
            if (idx + 1) % 20 == 0:
                log.info("[exp35] identity target=%s %d/%d", target_stage, idx + 1, len(rows))


def merge_identity(out_dir: Path, target_stage: str, n_workers: int) -> Path:
    target_stage = stage_key_for_label(target_stage)
    merged = out_dir / target_stage / "identity_margin.jsonl.gz"
    with gzip.open(merged, "wt", encoding="utf-8") as fout:
        for worker_idx in range(n_workers):
            path = out_dir / target_stage / f"identity_margin_w{worker_idx}.jsonl.gz"
            if not path.exists():
                log.warning("[exp35] missing identity file %s", path)
                continue
            for row in json_rows(path):
                fout.write(json.dumps(row, separators=(",", ":")) + "\n")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-stage", required=True, choices=list(NON_BASE_STAGES))
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_identity(args.out_dir, args.target_stage, args.n_workers)
        return
    run_identity_worker(args)


if __name__ == "__main__":
    main()

