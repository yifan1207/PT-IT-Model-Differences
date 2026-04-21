#!/usr/bin/env python3
"""Generate fixed random unit vectors for the A1_rand specificity control.

Creates random_directions.npz with one random unit vector per corrective layer
(20-33), using a fixed seed for full reproducibility.  Same file format as
corrective_directions.npz so it plugs straight into the existing
directional_remove pipeline.

Scientific purpose
------------------
A1_rand runs the identical α-sweep as A1 (layers 20-33, same formula) but
replaces the IT-PT corrective direction with a random unit vector.  If the
governance dose-response in A1 is direction-specific — i.e. the *content*
of the direction matters, not just the perturbation magnitude at those layers
— then A1_rand should be flat across all α values.

Usage:
    python scripts/generate_random_directions.py            # defaults
    python scripts/generate_random_directions.py --seed 7   # alt seed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

D_MODEL     = 2560                   # Gemma-3-4B hidden size
CORR_LAYERS = list(range(20, 34))    # layers 20-33 — A1 ablation target
DEFAULT_OUT  = "results/exp06_corrective_direction_steering/precompute/random_directions.npz"
DEFAULT_SEED = 42


def generate(out_path: str, seed: int) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print(f"[rand_dir] already exists, skipping: {out}")
        return

    rng = np.random.default_rng(seed)
    payload: dict[str, np.ndarray] = {}
    for li in CORR_LAYERS:
        v = rng.standard_normal(D_MODEL).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)   # unit vector
        payload[f"layer_{li}"] = v
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5, f"layer {li} not unit norm"

    np.savez_compressed(str(out), **payload)

    print(f"[rand_dir] saved {len(payload)} random unit vectors (seed={seed}) → {out}")
    for li in CORR_LAYERS:
        n = float(np.linalg.norm(payload[f"layer_{li}"]))
        print(f"  layer {li}: norm={n:.6f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",  default=DEFAULT_OUT,  help="Output npz path")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    args = p.parse_args()
    generate(args.out, args.seed)


if __name__ == "__main__":
    main()
