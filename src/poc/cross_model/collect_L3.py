"""
L3: Weight change localization — PT vs IT per-layer RMS weight diff.

For each model, loads PT and IT state dicts on CPU and computes the
normalized Frobenius (RMS) weight difference per layer, decomposed
into attention / MLP / LayerNorm components.

No inference required — purely a weight comparison.

═══ Memory-efficient design ══════════════════════════════════════════════
  Loading two 8B models in float32 would require ~64 GB CPU RAM.
  We avoid this by processing one layer at a time:
    1. Load PT state dict.
    2. Load IT state dict.
    3. Iterate layers, computing diffs on-the-fly.
    4. Free state dicts immediately after the loop.

  For the largest models (8B, ~32 GB float32 per SD) this still requires
  ~64 GB RAM if both SDs are in memory simultaneously — unavoidable for
  the per-layer-key comparison.  If RAM is tight, use bfloat16 (~16 GB
  per model) by passing --dtype bfloat16 (slightly less precise diff).

═══ MoE decomposition (DeepSeek-V2-Lite) ════════════════════════════════
  MoE layers have router / shared_expert / routed_expert sub-components
  (plan §4.L3 Step 3).  We detect them by key name patterns and report
  separately in delta_router, delta_shared_expert, delta_routed_experts.

═══ Output ═══════════════════════════════════════════════════════════════
  results/cross_model/{model}/weight_diff.json

  JSON format (plan §4.L3 Step 5):
    {
      "model": str, "pt_id": str, "it_id": str, "n_layers": int,
      "delta_w":     [float * n_layers],   # total RMS diff
      "delta_attn":  [float * n_layers],
      "delta_mlp":   [float * n_layers],
      "delta_norm":  [float * n_layers],
      "delta_embed": float,
      "delta_final_norm": float,
      "delta_lm_head": float,
      # DeepSeek only:
      "delta_router":          [float * n_layers],
      "delta_shared_expert":   [float * n_layers],
      "delta_routed_experts":  [float * n_layers]
    }
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
    """Root-mean-square element-wise diff over a set of parameter tensors."""
    total_sq = 0.0
    total_n  = 0
    for k in keys:
        if k not in sd_pt or k not in sd_it:
            continue
        diff = sd_it[k].float() - sd_pt[k].float()
        total_sq += diff.pow(2).sum().item()
        total_n  += diff.numel()
    return math.sqrt(total_sq / total_n) if total_n > 0 else 0.0


def _layer_keys(sd: dict, layer_idx: int) -> list[str]:
    """All parameter keys belonging to transformer layer layer_idx."""
    prefix = f"model.layers.{layer_idx}."
    return [k for k in sd if k.startswith(prefix)]


def _filter_keys(keys: list[str], *substrings: str) -> list[str]:
    """Return keys that contain any of the given substrings."""
    return [k for k in keys if any(s in k for s in substrings)]


# ── main computation ──────────────────────────────────────────────────────────

def compute_weight_diff(spec: ModelSpec, dtype: torch.dtype = torch.float32) -> dict:
    """Load PT and IT state dicts on CPU; return per-layer RMS diff dict."""
    log.info("Loading PT state dict: %s", spec.pt_id)
    pt_model = AutoModelForCausalLM.from_pretrained(
        spec.pt_id, torch_dtype=dtype, device_map="cpu", trust_remote_code=True,
    )
    pt_sd = pt_model.state_dict()
    del pt_model

    log.info("Loading IT state dict: %s", spec.it_id)
    it_model = AutoModelForCausalLM.from_pretrained(
        spec.it_id, torch_dtype=dtype, device_map="cpu", trust_remote_code=True,
    )
    it_sd = it_model.state_dict()
    del it_model

    result: dict = {
        "model":    spec.name,
        "pt_id":    spec.pt_id,
        "it_id":    spec.it_id,
        "n_layers": spec.n_layers,
        "delta_w":    [],
        "delta_attn": [],
        "delta_mlp":  [],
        "delta_norm": [],
    }

    # MoE-specific lists (only populated for spec.is_moe)
    if spec.is_moe:
        result["delta_router"]         = []
        result["delta_shared_expert"]  = []
        result["delta_routed_experts"] = []

    for layer_idx in range(spec.n_layers):
        all_keys  = _layer_keys(pt_sd, layer_idx)
        attn_keys = _filter_keys(all_keys, "self_attn", "attention")
        mlp_keys  = _filter_keys(all_keys, ".mlp.", "expert", "router", "gate")
        norm_keys = _filter_keys(all_keys, "layernorm", "layer_norm", ".norm.")

        result["delta_w"].append(_rms_diff(pt_sd, it_sd, all_keys))
        result["delta_attn"].append(_rms_diff(pt_sd, it_sd, attn_keys))
        result["delta_mlp"].append(_rms_diff(pt_sd, it_sd, mlp_keys))
        result["delta_norm"].append(_rms_diff(pt_sd, it_sd, norm_keys))

        if spec.is_moe:
            router_keys  = _filter_keys(all_keys, "router", "gate_proj")
            shared_keys  = _filter_keys(all_keys, "shared_expert")
            routed_keys  = _filter_keys(all_keys, "experts.")  # routed, not shared
            result["delta_router"].append(_rms_diff(pt_sd, it_sd, router_keys))
            result["delta_shared_expert"].append(_rms_diff(pt_sd, it_sd, shared_keys))
            result["delta_routed_experts"].append(_rms_diff(pt_sd, it_sd, routed_keys))

        if (layer_idx + 1) % 8 == 0 or layer_idx == spec.n_layers - 1:
            log.info("  layer %d/%d done", layer_idx + 1, spec.n_layers)

    # Non-layer parameters
    result["delta_embed"] = _rms_diff(
        pt_sd, it_sd,
        [k for k in pt_sd if "embed_tokens" in k],
    )
    result["delta_final_norm"] = _rms_diff(
        pt_sd, it_sd,
        [k for k in pt_sd if k in {"model.norm.weight", "model.norm.bias"}],
    )
    result["delta_lm_head"] = _rms_diff(
        pt_sd, it_sd,
        [k for k in pt_sd if k.startswith("lm_head")],
    )

    return result


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="L3: Per-layer RMS weight diff (PT → IT)."
    )
    parser.add_argument("--model",   required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "bfloat16"],
        help="Dtype for state-dict loading. bfloat16 halves RAM (~16 GB per 8B model).",
    )
    args = parser.parse_args()

    spec    = get_spec(args.model)
    dtype   = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    out_dir = Path(args.out_dir) if args.out_dir else spec.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weight_diff.json"

    if out_path.exists():
        log.info("Output already exists: %s — skipping.", out_path)
        return

    log.info("=== L3 weight diff: %s (dtype=%s) ===", spec.name, args.dtype)
    result = compute_weight_diff(spec, dtype=dtype)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
