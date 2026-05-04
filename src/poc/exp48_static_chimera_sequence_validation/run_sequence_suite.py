"""Grouped Exp48 sequence jobs that reuse loaded model pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.poc.exp48_static_chimera_sequence_validation.config import INTERPOLATION_ALPHAS
from src.poc.exp48_static_chimera_sequence_validation.run_sequence_generation import run_worker


def _run_one(args: argparse.Namespace, *, boundary: int, scenario: str, component: str, cell: str, alpha: float | None = None) -> None:
    alpha_part = "" if alpha is None else f"__a{str(alpha).replace('.', 'p')}"
    stem = f"{args.model}__b{boundary}__{scenario}__{component}__{cell}{alpha_part}"
    out_dir = args.out_root / stem
    child = argparse.Namespace(
        model=args.model,
        dataset=args.dataset,
        out_dir=out_dir,
        device=args.device,
        worker_index=0,
        n_workers=1,
        n_prompts=args.n_prompts,
        prompt_split=args.prompt_split,
        boundary=int(boundary),
        scenario=scenario,
        component=component,
        cell=cell,
        wrong_model=args.wrong_model,
        interpolation_alpha=float(alpha) if alpha is not None else 1.0,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        health_every=args.health_every,
        category_filter=args.category_filter,
        source_filter=args.source_filter,
        stem=stem,
        merge_only=False,
    )
    run_worker(child)


def run_suite(args: argparse.Namespace) -> None:
    for boundary in args.boundaries:
        for cell in ("BB", "BF", "FB", "FF"):
            _run_one(args, boundary=boundary, scenario="boundary_sweep", component="blocks_plus_head", cell=cell)
        if boundary in set(args.component_boundaries):
            for component in ("blocks_only", "head_only"):
                for cell in ("BB", "BF", "FB", "FF"):
                    _run_one(args, boundary=boundary, scenario="component_variant", component=component, cell=cell)
            for component in ("mlp_only", "attn_only"):
                for cell in ("BB", "BF", "FF"):
                    _run_one(args, boundary=boundary, scenario="decomposition", component=component, cell=cell)
        if boundary in set(args.control_boundaries):
            for scenario in ("wrong_descendant", "permuted_blocks"):
                _run_one(args, boundary=boundary, scenario=scenario, component="blocks_plus_head", cell="BF")
            for alpha in args.interpolation_alphas:
                _run_one(
                    args,
                    boundary=boundary,
                    scenario="interpolated_late",
                    component="blocks_plus_head",
                    cell="BF",
                    alpha=float(alpha),
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-prompts", type=int, default=1400)
    parser.add_argument("--prompt-split", choices=["train", "heldout", "all"], default="heldout")
    parser.add_argument("--boundaries", nargs="*", type=int, default=[16, 19, 24, 29, 31])
    parser.add_argument("--component-boundaries", nargs="*", type=int, default=[19, 29, 31])
    parser.add_argument("--control-boundaries", nargs="*", type=int, default=[19, 29, 31])
    parser.add_argument("--interpolation-alphas", nargs="*", type=float, default=list(INTERPOLATION_ALPHAS))
    parser.add_argument("--wrong-model", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--health-every", type=int, default=8)
    parser.add_argument("--category-filter", nargs="*", default=None)
    parser.add_argument("--source-filter", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()

