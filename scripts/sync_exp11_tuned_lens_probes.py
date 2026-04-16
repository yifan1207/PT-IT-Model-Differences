#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.cross_model.config import get_spec


GCS_ROOT = "gs://pt-vs-it-results/tuned_lens_probes_v3"
DEFAULT_CACHE = Path.home() / ".cache" / "exp11_tuned_lens_probes_v3"
DEFAULT_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b", "deepseek_v2_lite"]


def _download_probe(remote: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gsutil", "cp", remote, str(local_path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync exp11 tuned-lens probes from GCS into the local cache.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--variants", nargs="+", default=["pt", "it"])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    for model in args.models:
        spec = get_spec(model)
        for variant in args.variants:
            local_dir = args.cache_dir / model / variant
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"[sync-probes] ensuring {model}/{variant}")
            for layer_idx in range(spec.n_layers):
                local_path = local_dir / f"probe_layer_{layer_idx}.pt"
                if local_path.exists() and local_path.stat().st_size > 0:
                    continue
                remote = f"{GCS_ROOT}/{model}/{variant}/probe_layer_{layer_idx}.pt"
                print(f"[sync-probes] download {remote}")
                _download_probe(remote, local_path)


if __name__ == "__main__":
    main()
