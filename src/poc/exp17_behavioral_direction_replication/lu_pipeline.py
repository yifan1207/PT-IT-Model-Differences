"""Wrapper for Lu et al.'s public Assistant Axis pipeline.

This module does not reimplement the Assistant Axis internally. Instead it
launches the upstream `safety-research/assistant-axis` pipeline from a local
checkout and stores launch metadata under the repo's canonical exp17 paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
from pathlib import Path

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VALID_MODELS = list(MODEL_REGISTRY.keys())
VALID_VARIANTS = ("pt", "it")
DEFAULT_ASSISTANT_AXIS_REPO = Path("external/exp17_upstream/assistant-axis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lu Assistant Axis pipeline via upstream repo.")
    parser.add_argument(
        "--assistant-axis-repo",
        default=str(DEFAULT_ASSISTANT_AXIS_REPO),
        help="Local checkout of safety-research/assistant-axis",
    )
    parser.add_argument("--model", required=True, choices=VALID_MODELS)
    parser.add_argument("--variant", required=True, choices=VALID_VARIANTS)
    parser.add_argument("--step", default="all", choices=["generate", "activations", "judge", "vectors", "axis", "all"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--question-count", type=int, default=240)
    parser.add_argument("--roles-dir", default="")
    parser.add_argument("--questions-file", default="")
    parser.add_argument("--roles", nargs="*", default=[])
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def _base_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    return Path("results/exp17_behavioral_direction_replication/lu_replication") / args.model / args.variant


def _step_commands(args: argparse.Namespace, model_id: str, out_dir: Path) -> list[tuple[str, list[str]]]:
    repo = Path(args.assistant_axis_repo).expanduser().resolve()
    pipeline_dir = repo / "pipeline"
    responses_dir = out_dir / "responses"
    activations_dir = out_dir / "activations"
    scores_dir = out_dir / "scores"
    vectors_dir = out_dir / "vectors"
    axis_path = out_dir / "assistant_axis.pt"

    extra_generate: list[str] = []
    extra_activations: list[str] = []
    if args.roles_dir:
        extra_generate.extend(["--roles_dir", args.roles_dir])
    if args.questions_file:
        extra_generate.extend(["--questions_file", args.questions_file])
    if args.roles:
        extra_generate.extend(["--roles", *args.roles])
        extra_activations.extend(["--roles", *args.roles])

    generate_cmd = [
        "uv",
        "run",
        str(pipeline_dir / "1_generate.py"),
        "--model",
        model_id,
        "--output_dir",
        str(responses_dir),
        "--question_count",
        str(args.question_count),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        *extra_generate,
    ]
    activations_cmd = [
        "uv",
        "run",
        str(pipeline_dir / "2_activations.py"),
        "--model",
        model_id,
        "--responses_dir",
        str(responses_dir),
        "--output_dir",
        str(activations_dir),
        "--batch_size",
        str(args.batch_size),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        *extra_activations,
    ]
    judge_cmd = [
        "uv",
        "run",
        str(pipeline_dir / "3_judge.py"),
        "--responses_dir",
        str(responses_dir),
        "--output_dir",
        str(scores_dir),
    ]
    vectors_cmd = [
        "uv",
        "run",
        str(pipeline_dir / "4_vectors.py"),
        "--activations_dir",
        str(activations_dir),
        "--scores_dir",
        str(scores_dir),
        "--output_dir",
        str(vectors_dir),
    ]
    axis_cmd = [
        "uv",
        "run",
        str(pipeline_dir / "5_axis.py"),
        "--vectors_dir",
        str(vectors_dir),
        "--output",
        str(axis_path),
    ]
    return [
        ("generate", generate_cmd),
        ("activations", activations_cmd),
        ("judge", judge_cmd),
        ("vectors", vectors_cmd),
        ("axis", axis_cmd),
    ]


def _write_launch_metadata(out_dir: Path, args: argparse.Namespace, model_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "component": "lu_replication",
        "model": args.model,
        "variant": args.variant,
        "model_id": model_id,
        "assistant_axis_repo": str(Path(args.assistant_axis_repo).expanduser().resolve()),
        "step": args.step,
        "tensor_parallel_size": args.tensor_parallel_size,
        "batch_size": args.batch_size,
        "question_count": args.question_count,
        "roles_dir": args.roles_dir,
        "questions_file": args.questions_file,
        "roles": args.roles,
        "selected_layer": None,
        "selected_layer_note": "Populate after Lu-style layer selection / downstream analysis.",
    }
    with open(out_dir / "launch_config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    repo = Path(args.assistant_axis_repo).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"assistant-axis repo not found: {repo}")
    spec = get_spec(args.model)
    model_id = model_id_for_variant(spec, args.variant)
    out_dir = _base_output_dir(args).resolve()
    _write_launch_metadata(out_dir, args, model_id)

    step_names = [args.step] if args.step != "all" else ["generate", "activations", "judge", "vectors", "axis"]
    commands = _step_commands(args, model_id, out_dir)
    selected = [(name, cmd) for name, cmd in commands if name in step_names]

    for name, cmd in selected:
        rendered = shlex.join(cmd)
        log.info("[%s] %s", name, rendered)
        if args.print_only:
            continue
        subprocess.run(cmd, cwd=repo, check=True)

    if args.print_only:
        log.info("Print-only mode complete. Metadata recorded under %s", out_dir)
    else:
        log.info("Lu pipeline step(s) complete -> %s", out_dir)


if __name__ == "__main__":
    main()
