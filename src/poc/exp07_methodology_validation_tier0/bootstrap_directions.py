"""Bootstrap direction stability analysis (Exp7 0A) and matched-token comparison (0B).

0A mode: Loads merged per-record MLP activations from collect_per_record_acts.py,
computes bootstrap stability of the corrective direction.

Reports stability separately for three layer groups:
  - Early (1–11): content/proposal layers
  - Mid (12–19): transition layers
  - Corrective (20–33): steering-relevant layers

0A-OOD mode: Compute direction from 600 out-of-distribution prompts (not from the
1,400 calibration set) and compare cosine to canonical. Tests whether the direction
is overfit to the calibration prompt distribution.

0B (--matched-mode): Compares canonical (free-running) direction to direction
computed from matched token sequences (PT forced to follow IT's tokens).
Supports two sub-modes:
  (a) governance-selected 600 — isolates token confound only
  (b) random 600 — tests both token confound AND prompt generalization

Usage:

  # 0A: bootstrap stability
  uv run python -m src.poc.exp07_methodology_validation_tier0.bootstrap_directions \\
      --acts-dir results/exp7/0A/acts/ \\
      --canonical-npz results/exp5/precompute_v2/precompute/corrective_directions.npz \\
      --n-bootstrap 50 --output-dir results/exp7/0A/

  # 0B: matched-token direction validation
  uv run python -m src.poc.exp07_methodology_validation_tier0.bootstrap_directions \\
      --it-acts-dir results/exp7/0A/acts/ \\
      --pt-forced-acts-dir results/exp7/0B/acts/ \\
      --canonical-npz results/exp5/precompute_v2/precompute/corrective_directions.npz \\
      --matched-mode --output-dir results/exp7/0B/

  # Quick test
  uv run python -m src.poc.exp07_methodology_validation_tier0.bootstrap_directions \\
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
# Layer groups (matching precompute_directions_v2.py Phase 4 split)
EARLY_LAYERS = list(range(1, 12))     # 1-11
MID_LAYERS = list(range(12, 20))      # 12-19
CORRECTIVE_LAYERS = list(range(20, 34))  # 20-33

LAYER_GROUPS = {
    "early": EARLY_LAYERS,
    "mid": MID_LAYERS,
    "corrective": CORRECTIVE_LAYERS,
}

# Subset sizes for convergence curve (extends to 1000/1400 when data available)
CONVERGENCE_SIZES = [100, 200, 300, 400, 500, 600, 1000, 1400]
CONVERGENCE_DRAWS = 20   # random draws per subset size


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


def _load_acts(acts_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load merged.npz from acts_dir. Returns (record_ids, it_acts, pt_acts).

    it_acts[layer] = float64 [n_records, D_MODEL]
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


def _group_summary(layer_results: dict[int, dict], group_layers: list[int], metric_key: str) -> dict:
    """Compute mean/min/std of a metric across a layer group."""
    vals = [layer_results[li][metric_key] for li in group_layers
            if li in layer_results and not np.isnan(layer_results[li].get(metric_key, float("nan")))]
    if not vals:
        return {"mean": float("nan"), "min": float("nan"), "std": float("nan")}
    arr = np.array(vals)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "std": float(arr.std())}


def run_bootstrap(
    it_acts: dict[int, np.ndarray],
    pt_acts: dict[int, np.ndarray],
    canonical: dict[int, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Run bootstrap direction stability analysis.

    Returns dict with per-layer stats, convergence curves, and layer-group summaries.
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

    # ── Layer-group summaries ────────────────────────────────────────────────
    group_summaries: dict[str, dict] = {}
    for group_name, group_layers in LAYER_GROUPS.items():
        group_summaries[group_name] = {
            "pairwise_cosine": _group_summary(layer_results, group_layers, "pairwise_cosine_mean"),
            "canonical_cosine": _group_summary(layer_results, group_layers, "canonical_cosine_mean"),
        }

    # ── Summary aggregates ────────────────────────────────────────────────────
    summary = {
        # Per-group means (the key addition for the paper)
        "corrective_mean_cosine": group_summaries["corrective"]["pairwise_cosine"]["mean"],
        "mid_mean_cosine": group_summaries["mid"]["pairwise_cosine"]["mean"],
        "early_mean_cosine": group_summaries["early"]["pairwise_cosine"]["mean"],
        "corrective_canonical_cosine": group_summaries["corrective"]["canonical_cosine"]["mean"],
        "mid_canonical_cosine": group_summaries["mid"]["canonical_cosine"]["mean"],
        "early_canonical_cosine": group_summaries["early"]["canonical_cosine"]["mean"],
        # Legacy aggregate stats
        "pairwise_cosine_mean_corrective": group_summaries["corrective"]["pairwise_cosine"]["mean"],
        "pairwise_cosine_min_corrective": group_summaries["corrective"]["pairwise_cosine"]["min"],
        "canonical_cosine_mean_corrective": group_summaries["corrective"]["canonical_cosine"]["mean"],
        "canonical_cosine_min_corrective": group_summaries["corrective"]["canonical_cosine"]["min"],
        "n_records": n_records,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "convergence_sizes": CONVERGENCE_SIZES,
    }

    return {
        "summary": summary,
        "layer_group_summaries": group_summaries,
        "per_layer": layer_results,
        "convergence": {str(li): {str(s): v for s, v in conv.items()} for li, conv in convergence.items()},
    }


def run_matched_comparison(
    it_acts: dict[int, np.ndarray],
    pt_forced_acts: dict[int, np.ndarray],
    canonical: dict[int, np.ndarray],
    mode_label: str = "governance_selected",
) -> dict:
    """Compare canonical (free-running) direction to matched-token direction.

    matched_direction[L] = normalize(mean_IT_free[L] - mean_PT_forced[L])

    Returns per-layer cosine(matched, canonical) with layer-group flagging.
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

    # Per-group summaries with corrective vs early flagging
    group_cosines: dict[str, dict] = {}
    for group_name, group_layers in LAYER_GROUPS.items():
        cosines = [float(results[str(li)]["cosine_matched_vs_canonical"])
                   for li in group_layers if str(li) in results
                   and not np.isnan(results[str(li)]["cosine_matched_vs_canonical"])]
        if cosines:
            arr = np.array(cosines)
            group_cosines[group_name] = {
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "std": float(arr.std()),
            }
        else:
            group_cosines[group_name] = {"mean": float("nan"), "min": float("nan"), "std": float("nan")}

    # Flag if corrective layers have lower cosine than early layers
    corr_mean = group_cosines["corrective"]["mean"]
    early_mean = group_cosines["early"]["mean"]
    drop_at_corrective = early_mean - corr_mean if not np.isnan(corr_mean) and not np.isnan(early_mean) else float("nan")

    summary = {
        "mode": mode_label,
        "cosine_matched_vs_canonical_mean_corrective": corr_mean,
        "cosine_matched_vs_canonical_min_corrective": group_cosines["corrective"]["min"],
        "cosine_by_group": group_cosines,
        "corrective_vs_early_drop": drop_at_corrective,
        "corrective_drop_flag": (
            "SIGNIFICANT: cosine drops at corrective layers — token confound is relevant for steering"
            if not np.isnan(drop_at_corrective) and drop_at_corrective > 0.05
            else "MINIMAL: cosine stable across layers — token confound is not layer-specific"
            if not np.isnan(drop_at_corrective)
            else "UNKNOWN"
        ),
        "n_matched_records": min(
            next(iter(it_acts.values())).shape[0],
            next(iter(pt_forced_acts.values())).shape[0],
        ),
    }

    return {"summary": summary, "per_layer": results}


def run_ood_direction_test(
    ood_acts_dir: Path,
    canonical: dict[int, np.ndarray],
) -> dict:
    """Compute direction from OOD prompts and compare to canonical.

    Tests whether the corrective direction generalises beyond the calibration set.
    Uses 600 out-of-distribution prompts (e.g. TriviaQA/ARC) not in eval_dataset_v2.
    """
    _, ood_it, ood_pt = _load_acts(ood_acts_dir)
    n_records = next(iter(ood_it.values())).shape[0]

    per_layer: dict[str, dict] = {}
    for li in ALL_LAYERS:
        ood_dir = _normalize(ood_it[li].mean(axis=0) - ood_pt[li].mean(axis=0))
        cos = _cosine(ood_dir, canonical[li]) if li in canonical else float("nan")
        per_layer[str(li)] = {"cosine_ood_vs_canonical": cos}

    # Group summaries
    group_cosines: dict[str, dict] = {}
    for group_name, group_layers in LAYER_GROUPS.items():
        cosines = [per_layer[str(li)]["cosine_ood_vs_canonical"]
                   for li in group_layers if str(li) in per_layer
                   and not np.isnan(per_layer[str(li)]["cosine_ood_vs_canonical"])]
        if cosines:
            arr = np.array(cosines)
            group_cosines[group_name] = {"mean": float(arr.mean()), "min": float(arr.min())}

    summary = {
        "n_ood_records": n_records,
        "cosine_by_group": group_cosines,
        "corrective_mean": group_cosines.get("corrective", {}).get("mean", float("nan")),
        "interpretation": (
            "direction generalises to OOD prompts (cosine > 0.90)"
            if group_cosines.get("corrective", {}).get("mean", 0) > 0.90
            else "direction may be overfit to calibration set (cosine < 0.90)"
        ),
    }

    return {"summary": summary, "per_layer": per_layer}


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
    p.add_argument("--matched-label", default="governance_selected",
                   help="Label for matched mode: 'governance_selected' or 'random_600'")
    p.add_argument("--ood-acts-dir", default=None,
                   help="Path to OOD acts dir for out-of-distribution direction test (0A)")
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
        results = run_matched_comparison(it_acts, pt_forced_acts, canonical, args.matched_label)
        out_path = output_dir / f"matched_cosines_{args.matched_label}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        s = results["summary"]
        print(
            f"[0B] Matched direction cosine ({args.matched_label}):\n"
            f"  Corrective (20-33): mean={s['cosine_matched_vs_canonical_mean_corrective']:.4f}\n"
            f"  Drop vs early: {s['corrective_vs_early_drop']:.4f}\n"
            f"  Flag: {s['corrective_drop_flag']}",
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
        gs = results["layer_group_summaries"]
        print(
            f"[0A] Bootstrap stability by layer group:\n"
            f"  Early    (1-11):  pairwise={gs['early']['pairwise_cosine']['mean']:.4f}  "
            f"canonical={gs['early']['canonical_cosine']['mean']:.4f}\n"
            f"  Mid      (12-19): pairwise={gs['mid']['pairwise_cosine']['mean']:.4f}  "
            f"canonical={gs['mid']['canonical_cosine']['mean']:.4f}\n"
            f"  Corrective (20-33): pairwise={gs['corrective']['pairwise_cosine']['mean']:.4f}  "
            f"canonical={gs['corrective']['canonical_cosine']['mean']:.4f}",
            flush=True,
        )
        print(f"[0A] saved → {out_path}", flush=True)

        # OOD direction test if provided
        if args.ood_acts_dir:
            ood_results = run_ood_direction_test(Path(args.ood_acts_dir), canonical)
            ood_path = output_dir / "ood_direction_cosine.json"
            with open(ood_path, "w") as f:
                json.dump(ood_results, f, indent=2)
            ood_s = ood_results["summary"]
            print(
                f"[0A-OOD] Direction from {ood_s['n_ood_records']} OOD prompts:\n"
                f"  Corrective cosine to canonical: {ood_s['corrective_mean']:.4f}\n"
                f"  {ood_s['interpretation']}",
                flush=True,
            )
            print(f"[0A-OOD] saved → {ood_path}", flush=True)


if __name__ == "__main__":
    main()
