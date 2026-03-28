"""Bootstrap direction stability analysis (Exp7 0A) and matched-token comparison (0B).

0A mode: Loads merged per-record MLP activations from collect_per_record_acts.py,
computes bootstrap stability of the corrective direction.

0B (--matched-mode): Compares canonical (free-running) direction to direction
computed from matched token sequences (PT forced to follow IT's tokens).

Usage:

  # 0A: bootstrap stability
  uv run python -m src.poc.exp7.bootstrap_directions \\
      --acts-dir results/exp7/0A/acts/ \\
      --canonical-npz results/exp5/precompute_v2/precompute/corrective_directions.npz \\
      --n-bootstrap 50 --output-dir results/exp7/0A/

  # 0B: matched-token direction validation
  uv run python -m src.poc.exp7.bootstrap_directions \\
      --it-acts-dir results/exp7/0A/acts/ \\
      --pt-forced-acts-dir results/exp7/0B/acts/ \\
      --canonical-npz results/exp5/precompute_v2/precompute/corrective_directions.npz \\
      --matched-mode --output-dir results/exp7/0B/

  # Quick test
  uv run python -m src.poc.exp7.bootstrap_directions \\
      --acts-dir results/exp7/0A/acts/ \\
      --canonical-npz results/exp5/precompute_v2/precompute/corrective_directions.npz \\
      --n-bootstrap 10 --output-dir results/exp7/0A/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

ALL_LAYERS = list(range(1, 34))
# Subset sizes for convergence curve
CONVERGENCE_SIZES = [100, 200, 300, 400, 500, 600]
CONVERGENCE_DRAWS = 20   # random draws per subset size


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


def _load_acts(acts_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load merged.npz from acts_dir. Returns (record_ids, it_acts, pt_acts).

    it_acts[layer] = float32 [n_records, D_MODEL]
    """
    merged = acts_dir / "merged.npz"
    with np.load(merged, allow_pickle=True) as d:
        record_ids = d["record_ids"].tolist()
        it_acts = {li: d[f"it_acts_{li}"].astype(np.float64) for li in ALL_LAYERS}
        pt_acts = {li: d[f"pt_acts_{li}"].astype(np.float64) for li in ALL_LAYERS}
    return record_ids, it_acts, pt_acts


def _load_canonical(npz_path: Path) -> dict[int, np.ndarray]:
    """Load corrective_directions.npz → dict[layer → unit vector float64]."""
    with np.load(npz_path) as d:
        return {int(k.split("_", 1)[1]): d[k].astype(np.float64) for k in d.files if k.startswith("layer_")}


def run_bootstrap(
    it_acts: dict[int, np.ndarray],
    pt_acts: dict[int, np.ndarray],
    canonical: dict[int, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Run bootstrap direction stability analysis.

    Returns dict with per-layer stats and convergence curves.
    """
    rng = np.random.default_rng(seed)
    n_records = next(iter(it_acts.values())).shape[0]

    # ── Per-layer bootstrap ────────────────────────────────────────────────────
    layer_results: dict[int, dict] = {}

    for li in ALL_LAYERS:
        it = it_acts[li]  # [n_records, D]
        pt = pt_acts[li]

        bootstrap_dirs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_records, n_records, replace=True)
            d = _normalize(it[idx].mean(axis=0) - pt[idx].mean(axis=0))
            bootstrap_dirs.append(d)

        # Pairwise cosine similarity (upper triangle)
        pairwise = []
        for i in range(n_bootstrap):
            for j in range(i + 1, n_bootstrap):
                pairwise.append(_cosine(bootstrap_dirs[i], bootstrap_dirs[j]))
        pairwise = np.array(pairwise)

        # Cosine of each bootstrap direction to canonical
        canonical_cosines = []
        if li in canonical:
            for bd in bootstrap_dirs:
                canonical_cosines.append(_cosine(bd, canonical[li]))
        canonical_cosines = np.array(canonical_cosines) if canonical_cosines else np.array([float("nan")])

        layer_results[li] = {
            "pairwise_cosine_mean": float(pairwise.mean()),
            "pairwise_cosine_std": float(pairwise.std()),
            "pairwise_cosine_min": float(pairwise.min()),
            "canonical_cosine_mean": float(canonical_cosines.mean()),
            "canonical_cosine_std": float(canonical_cosines.std()),
            "canonical_cosine_min": float(canonical_cosines.min()),
        }

    # ── Convergence curve ─────────────────────────────────────────────────────
    convergence: dict[int, dict[int, dict]] = {}  # layer -> size -> stats
    for li in ALL_LAYERS:
        it = it_acts[li]
        pt = pt_acts[li]
        layer_conv: dict[int, dict] = {}
        for size in CONVERGENCE_SIZES:
            if size > n_records:
                continue
            cosines = []
            for _ in range(CONVERGENCE_DRAWS):
                idx = rng.choice(n_records, size, replace=False)
                d = _normalize(it[idx].mean(axis=0) - pt[idx].mean(axis=0))
                if li in canonical:
                    cosines.append(_cosine(d, canonical[li]))
            cosines_arr = np.array(cosines)
            layer_conv[size] = {
                "cosine_to_canonical_mean": float(cosines_arr.mean()),
                "cosine_to_canonical_std": float(cosines_arr.std()),
            }
        convergence[li] = layer_conv

    # ── Summary aggregates ────────────────────────────────────────────────────
    corrective_layers = list(range(20, 34))
    all_pairwise = [layer_results[li]["pairwise_cosine_mean"] for li in corrective_layers if li in layer_results]
    all_canonical = [layer_results[li]["canonical_cosine_mean"] for li in corrective_layers if li in layer_results]

    summary = {
        "pairwise_cosine_mean_corrective": float(np.mean(all_pairwise)),
        "pairwise_cosine_min_corrective": float(np.min(all_pairwise)),
        "canonical_cosine_mean_corrective": float(np.mean(all_canonical)),
        "canonical_cosine_min_corrective": float(np.min(all_canonical)),
        "n_records": n_records,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "convergence_sizes": CONVERGENCE_SIZES,
    }

    return {
        "summary": summary,
        "per_layer": layer_results,
        "convergence": {str(li): {str(s): v for s, v in conv.items()} for li, conv in convergence.items()},
    }


def run_matched_comparison(
    it_acts: dict[int, np.ndarray],
    pt_forced_acts: dict[int, np.ndarray],
    canonical: dict[int, np.ndarray],
) -> dict:
    """Compare canonical (free-running) direction to matched-token direction.

    matched_direction[L] = normalize(mean_IT_free[L] - mean_PT_forced[L])

    Returns per-layer cosine(matched, canonical).
    """
    results: dict[str, dict] = {}

    for li in ALL_LAYERS:
        it = it_acts[li]             # IT free-running acts [n, D]
        pt_f = pt_forced_acts[li]    # PT on IT's tokens [n, D]

        n = min(it.shape[0], pt_f.shape[0])
        matched_dir = _normalize(it[:n].mean(axis=0) - pt_f[:n].mean(axis=0))

        res: dict[str, float] = {}
        if li in canonical:
            res["cosine_matched_vs_canonical"] = _cosine(matched_dir, canonical[li])
        else:
            res["cosine_matched_vs_canonical"] = float("nan")

        results[str(li)] = res

    # Summary over corrective layers
    corrective = [float(results[str(li)]["cosine_matched_vs_canonical"]) for li in range(20, 34)]
    summary = {
        "cosine_matched_vs_canonical_mean_corrective": float(np.nanmean(corrective)),
        "cosine_matched_vs_canonical_min_corrective": float(np.nanmin(corrective)),
        "n_matched_records": min(
            next(iter(it_acts.values())).shape[0],
            next(iter(pt_forced_acts.values())).shape[0],
        ),
    }

    return {"summary": summary, "per_layer": results}


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap direction analysis (Exp7 0A/0B)")
    p.add_argument("--acts-dir", help="Path to merged acts dir (0A mode)")
    p.add_argument("--it-acts-dir", help="IT free-running acts dir (0B mode)")
    p.add_argument("--pt-forced-acts-dir", help="PT forced-decode acts dir (0B mode)")
    p.add_argument("--canonical-npz", required=True,
                   help="Path to corrective_directions.npz")
    p.add_argument("--n-bootstrap", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--matched-mode", action="store_true",
                   help="Run matched-token comparison (0B) instead of bootstrap (0A)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = _load_canonical(Path(args.canonical_npz))

    if args.matched_mode:
        if not args.it_acts_dir or not args.pt_forced_acts_dir:
            p.error("--matched-mode requires --it-acts-dir and --pt-forced-acts-dir")
        _, it_acts, _ = _load_acts(Path(args.it_acts_dir))
        _, pt_forced_acts, _ = _load_acts(Path(args.pt_forced_acts_dir))
        # pt_forced_acts has "it_acts" keys but these are actually PT-on-IT activations
        # force_decode_acts.py saves them under "it_acts_{L}" for the forced model
        results = run_matched_comparison(it_acts, pt_forced_acts, canonical)
        out_path = output_dir / "matched_cosines.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(
            f"[0B] matched direction cosine (corrective layers 20-33): "
            f"{results['summary']['cosine_matched_vs_canonical_mean_corrective']:.4f}",
            flush=True,
        )
        print(f"[0B] saved → {out_path}", flush=True)

    else:
        if not args.acts_dir:
            p.error("--acts-dir required in bootstrap mode (0A)")
        _, it_acts, pt_acts = _load_acts(Path(args.acts_dir))
        results = run_bootstrap(it_acts, pt_acts, canonical, args.n_bootstrap, args.seed)
        out_path = output_dir / "bootstrap_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        s = results["summary"]
        print(
            f"[0A] Bootstrap stability (corrective layers 20-33):\n"
            f"  Pairwise cosine: mean={s['pairwise_cosine_mean_corrective']:.4f}  "
            f"min={s['pairwise_cosine_min_corrective']:.4f}\n"
            f"  Canonical cosine: mean={s['canonical_cosine_mean_corrective']:.4f}  "
            f"min={s['canonical_cosine_min_corrective']:.4f}",
            flush=True,
        )
        print(f"[0A] saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
