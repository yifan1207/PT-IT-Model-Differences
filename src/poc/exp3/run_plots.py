"""
Regenerate all Exp3 plots from saved results.

Supports both JSON (legacy) and JSONL (new collect.py) result files.

Usage:
    uv run python -m src.poc.exp3.run_plots --variant it
    uv run python -m src.poc.exp3.run_plots --results path/to/results.jsonl
    uv run python -m src.poc.exp3.run_plots --variant it --pt-results path/to/pt_results.jsonl
    uv run python -m src.poc.exp3.run_plots --variant it --weight-shift path/to/weight_shift.json
    uv run python -m src.poc.exp3.run_plots --results results/it_run/results.jsonl \\
        --pt-results results/pt_run/results.jsonl --skip-e3
"""
import argparse
import json
from pathlib import Path


def _load_results(path: Path) -> list[dict]:
    """Load results from either a JSON array or a JSONL file."""
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
    if first_char == "[":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    # JSONL
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

from src.poc.exp3.config import Exp3Config
from src.poc.exp3.plots.plot1_transcoder_error import make_plot as plot1
from src.poc.exp3.plots.plot2_emergence import make_plot as plot2
from src.poc.exp3.plots.plot3_attraction_repulsion import make_plot as plot3
from src.poc.exp3.plots.plot4_token_stratification import make_plot as plot4
from src.poc.exp3.plots.plot5_alignment_tax import make_plot as plot5
from src.poc.exp3.plots.plot6_kl_trajectory import make_plot as plot6
from src.poc.exp3.plots.plot7_ablation_comparison import make_plot as plot7
from src.poc.exp3.plots.plot8_weight_norm import make_plot as plot8
from src.poc.exp3.plots.plot9_precorrection import make_plot as plot9
from src.poc.exp3.plots.plot10_generation_heatmap import make_plot as plot10
from src.poc.exp4.plots.plot1_jaccard_curve import make_plot as plot_jaccard
from src.poc.exp3.plots.plot_feature_populations import make_plot as plot_feat_pop
from src.poc.exp3.plots.plot_e3_3_raw_completion import make_plot as plot_e3_3
from src.poc.exp3.plots.plot_e3_4_matched_token import make_plot as plot_e3_4
from src.poc.exp3.plots.plot_e3_5_confidence_stratified import make_plot as plot_e3_5
from src.poc.exp3.plots.plot_e3_9_step_stability import make_plot as plot_e3_9
from src.poc.exp3.plots.plot_e3_10_mind_change import make_plot as plot_e3_10
from src.poc.exp3.plots.plot_e3_11_feature_importance import make_plot as plot_e3_11
from src.poc.exp3.plots.plot_e3_12_adjacent_layer_kl import make_plot as plot_e3_12
from src.poc.exp3.plots.plot_e3_13_candidate_reshuffling import make_plot as plot_e3_13


def _resolve_feature_npz(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "features.npz", run_dir / "exp3_results.npz"):
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Exp3 plots")
    parser.add_argument("--variant", choices=["pt", "it"], default="pt")
    parser.add_argument("--results", default=None,
                        help="Path to primary results JSON or JSONL (default: cfg.output_path)")
    parser.add_argument("--pt-results", default=None,
                        help="Path to PT results JSON/JSONL for comparison plots")
    parser.add_argument("--weight-shift", default="results/weight_shift/weight_shift.json",
                        help="Path to weight_shift.json (for plot 8)")
    parser.add_argument("--skip-e3", action="store_true",
                        help="Skip E3.3/E3.4/E3.5 confound-control plots (E3.9/10/11/12/13 always run)")
    parser.add_argument("--plot-dir", default=None,
                        help="Override output directory for plots")
    args = parser.parse_args()

    cfg = Exp3Config(model_variant=args.variant)

    # Resolve results path — accept both JSON and JSONL
    if args.results:
        results_path = Path(args.results)
    else:
        # Legacy default; also check for JSONL from new collect.py
        legacy = Path(cfg.output_path)
        jsonl  = legacy.parent / "results.jsonl"
        results_path = jsonl if jsonl.exists() else legacy

    plot_dir = args.plot_dir or str(results_path.parent / "plots")

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    print(f"Loading results from {results_path} ...")
    results = _load_results(results_path)
    print(f"  Loaded {len(results)} results")

    # Split counts summary
    from collections import Counter
    sc = Counter(r.get("split") or r.get("category", "?") for r in results)
    for s, n in sorted(sc.items()):
        print(f"    {s}: {n}")

    pt_results = None
    pt_path: Path | None = None
    if args.pt_results:
        pt_path = Path(args.pt_results)
        if pt_path.exists():
            pt_results = _load_results(pt_path)
            print(f"  Loaded {len(pt_results)} PT results for comparison")
        else:
            print(f"  [WARN] --pt-results path not found: {pt_path}")
    else:
        # Auto-detect PT results (JSONL first, then legacy JSON)
        pt_cfg = Exp3Config(model_variant="pt")
        pt_auto_dir = results_path.parent.parent / f"pt_{cfg.transcoder_variant.replace('width_','')}_t{cfg.max_gen_tokens}"
        for pt_auto_path in [pt_auto_dir / "results.jsonl", Path(pt_cfg.output_path)]:
            if pt_auto_path.exists():
                pt_results = _load_results(pt_auto_path)
                pt_path = pt_auto_path
                print(f"  Auto-loaded {len(pt_results)} PT results from {pt_auto_path}")
                break

    print(f"\nGenerating plots → {plot_dir}")
    plot1(results, plot_dir)
    plot2(results, plot_dir, pt_results=pt_results)
    plot3(results, plot_dir, pt_results=pt_results)
    plot4(results, plot_dir, pt_results=pt_results)
    plot5(results, plot_dir, pt_results=pt_results)
    plot6(results, plot_dir, pt_results=pt_results)
    plot7(results, plot_dir)
    plot8(args.weight_shift, plot_dir)
    plot9(results, plot_dir, pt_results=pt_results)
    plot10(results, plot_dir, pt_results=pt_results)

    # Feature-space dip analysis (E0b Jaccard + E1a feature populations)
    # Shared collector writes features.npz; legacy writes exp3_results.npz — try both.
    it_npz = _resolve_feature_npz(results_path.parent)
    pt_npz = _resolve_feature_npz(pt_path.parent) if pt_path else None
    pt_cfg = Exp3Config(model_variant="pt")
    if it_npz is not None and pt_npz is not None:
        plot_jaccard(
            pt_features_path=str(pt_npz),
            it_features_path=str(it_npz),
            output_dir=plot_dir,
            source="exp3",
        )
    else:
        print("  Skipped exp4 continuity plot — missing PT or IT feature NPZ.")
    if pt_results is not None and it_npz is not None and pt_npz is not None and pt_path is not None:
        plot_feat_pop(
            it_json=str(results_path),
            it_npz=str(it_npz),
            pt_json=str(pt_path),
            pt_npz=str(pt_npz),
            output_dir=plot_dir,
        )

    # ── E3 confound-control plots (skippable) ────────────────────────────────
    if not args.skip_e3:
        print("\nGenerating E3 confound-control plots …")
        plot_e3_3(results, plot_dir, pt_results=pt_results)
        plot_e3_4(results, plot_dir, pt_results=pt_results)
        plot_e3_5(results, plot_dir, pt_results=pt_results)

    # ── E3 main analyses (always run) ─────────────────────────────────────────
    print("\nGenerating E3 main analysis plots …")
    plot_e3_9(results, plot_dir, pt_results=pt_results)
    plot_e3_10(results, plot_dir, pt_results=pt_results)
    pt_summary_path = str(pt_path.parent / "feature_importance_summary.npz") if pt_path else None
    plot_e3_11(
        summary_path=str(results_path.parent / "feature_importance_summary.npz"),
        output_dir=plot_dir,
        pt_summary_path=pt_summary_path,
    )
    plot_e3_12(results, plot_dir, pt_results=pt_results)
    plot_e3_13(results, plot_dir, pt_results=pt_results)

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
