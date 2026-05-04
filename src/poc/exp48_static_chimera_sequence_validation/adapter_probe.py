"""Preflight probes for Exp48 static chimeras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
from src.poc.exp48_static_chimera_sequence_validation.config import BOUNDARIES, DEFAULT_MODELS


def run(args: argparse.Namespace) -> dict:
    out = {"models": {}, "boundaries": list(args.boundaries)}
    for model in args.models:
        spec = get_spec(model)
        adapter = get_steering_adapter(model)
        problems = []
        if not spec.pt_id.startswith("meta-llama/Llama-3.1-8B"):
            problems.append("pt_not_llama31_8b_base")
        if spec.n_layers != 32:
            problems.append(f"unexpected_n_layers={spec.n_layers}")
        for boundary in args.boundaries:
            if boundary < 0 or boundary >= spec.n_layers:
                problems.append(f"boundary_out_of_range={boundary}")
        out["models"][model] = {
            "pt_id": model_id_for_variant(spec, "pt"),
            "it_id": model_id_for_variant(spec, "it"),
            "pt_revision": spec.pt_revision,
            "it_revision": spec.it_revision,
            "n_layers": spec.n_layers,
            "d_model": spec.d_model,
            "adapter": type(adapter.adapter).__name__,
            "problems": problems,
            "ok": not problems,
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "adapter_probe.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": all(v["ok"] for v in out["models"].values()), "out": str(path)}, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS), choices=list(MODEL_REGISTRY))
    parser.add_argument("--boundaries", nargs="*", type=int, default=list(BOUNDARIES))
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
