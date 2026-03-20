"""
Entry point for PT → IT weight shift analysis.

Usage:
    uv run python -m src.poc.exp2.run_weight_shift           # 8 GPUs (default)
    uv run python -m src.poc.exp2.run_weight_shift --gpus 1  # single GPU
    uv run python -m src.poc.exp2.run_weight_shift --output results/my_dir

Results are written to:
    <output>/weight_shift.json    per-parameter frob_shift + cos_dist
    <output>/plot_weight_shift.png  4-panel summary figure
"""
import argparse

from src.poc.exp2.weight_shift import compute_weight_shift, print_summary
from src.poc.exp2.plots.plot_weight_shift import make_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="PT → IT weight shift analysis")
    parser.add_argument(
        "--gpus", type=int, default=8,
        help="Number of GPUs for parallel metric computation (default: 8)",
    )
    parser.add_argument(
        "--output", default="results/weight_shift",
        help="Output directory for JSON and plot (default: results/weight_shift)",
    )
    parser.add_argument(
        "--top-n", type=int, default=40,
        help="Top-N parameters shown in bar chart (default: 40)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PT → IT weight shift: google/gemma-3-4b")
    print(f"  n_gpus : {args.gpus}")
    print(f"  output : {args.output}")
    print("=" * 60)

    results = compute_weight_shift(n_gpus=args.gpus, output_dir=args.output)
    print_summary(results, top_n=args.top_n)
    make_plot(results, output_dir=args.output, top_n=args.top_n)

    print("\nDone.")


if __name__ == "__main__":
    main()
