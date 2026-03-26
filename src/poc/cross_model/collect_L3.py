"""
L3: Weight change localization — PT vs IT per-layer RMS weight diff.

For each model, loads both PT and IT state dicts on CPU and computes
the normalized Frobenius (RMS) weight difference per layer, decomposed
into attention / MLP / LayerNorm components.

No inference required. Fast (~1-3 min per model pair on CPU).

Output: results/cross_model/{model_name}/weight_diff.json
"""
from __future__ import annotations

import json
import math
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from src.poc.cross_model.config import get_spec, MODEL_REGISTRY, ModelSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rms_diff(sd_pt: dict, sd_it: dict, keys: list[str]) -> float:
    """Root-mean-square difference over a set of parameter tensors."""
    total_sq = 0.0
    total_n = 0
    for k in keys:
        if k not in sd_pt or k not in sd_it:
            continue
        diff = sd_it[k].float() - sd_pt[k].float()
        total_sq += diff.pow(2).sum().item()
        total_n += diff.numel()
    return math.sqrt(total_sq / total_n) if total_n > 0 else 0.0


def compute_weight_diff(spec: ModelSpec) -> dict:
    """Load PT and IT models on CPU; compute RMS weight diff per layer.

    Returns a dict with keys:
      n_layers, delta_w, delta_attn, delta_mlp, delta_norm,
      delta_embed, delta_final_norm, delta_lm_head
    """
    log.info("Loading PT model: %s", spec.pt_id)
    pt_model = AutoModelForCausalLM.from_pretrained(
        spec.pt_id, torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True,
    )
    log.info("Loading IT model: %s", spec.it_id)
    it_model = AutoModelForCausalLM.from_pretrained(
        spec.it_id, torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True,
    )

    pt_sd = pt_model.state_dict()
    it_sd = it_model.state_dict()
    del pt_model, it_model  # free model objects; keep state dicts

    result: dict = {
        "model": spec.name,
        "pt_id": spec.pt_id,
        "it_id": spec.it_id,
        "n_layers": spec.n_layers,
        "delta_w": [],
        "delta_attn": [],
        "delta_mlp": [],
        "delta_norm": [],
    }

    for layer_idx in range(spec.n_layers):
        prefix = f"model.layers.{layer_idx}."
        all_keys = [k for k in pt_sd if k.startswith(prefix)]

        # Attention: self_attn weights (q/k/v/o projections)
        attn_keys = [k for k in all_keys if "self_attn" in k or "attention" in k.lower()]
        # MLP: gate/up/down projections, also handles MoE expert weights
        mlp_keys  = [k for k in all_keys if "mlp" in k or "expert" in k]
        # LayerNorm/RMSNorm
        norm_keys = [k for k in all_keys if "norm" in k or "layernorm" in k.lower()]

        result["delta_w"].append(_rms_diff(pt_sd, it_sd, all_keys))
        result["delta_attn"].append(_rms_diff(pt_sd, it_sd, attn_keys))
        result["delta_mlp"].append(_rms_diff(pt_sd, it_sd, mlp_keys))
        result["delta_norm"].append(_rms_diff(pt_sd, it_sd, norm_keys))

        if (layer_idx + 1) % 8 == 0 or layer_idx == spec.n_layers - 1:
            log.info("  layer %d/%d done", layer_idx + 1, spec.n_layers)

    # Non-layer parameters
    result["delta_embed"] = _rms_diff(
        pt_sd, it_sd, [k for k in pt_sd if "embed_tokens" in k or k == "model.embed_tokens.weight"]
    )
    result["delta_final_norm"] = _rms_diff(
        pt_sd, it_sd, [k for k in pt_sd if k in {"model.norm.weight", "model.norm.bias"}]
    )
    result["delta_lm_head"] = _rms_diff(
        pt_sd, it_sd, [k for k in pt_sd if "lm_head" in k]
    )

    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="L3: Compute per-layer RMS weight diff between PT and IT models."
    )
    parser.add_argument(
        "--model", required=True,
        choices=list(MODEL_REGISTRY),
        help="Model name from MODEL_REGISTRY",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: results/cross_model/{model}/)",
    )
    args = parser.parse_args()

    spec = get_spec(args.model)
    out_dir = Path(args.out_dir) if args.out_dir else spec.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weight_diff.json"

    if out_path.exists():
        log.info("Output already exists: %s — skipping.", out_path)
        return

    log.info("=== L3 weight diff: %s ===", spec.name)
    result = compute_weight_diff(spec)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
