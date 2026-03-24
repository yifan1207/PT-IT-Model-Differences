# Structural Semantic Features

Research code for probing phase structure, feature populations, and ablation behavior in Gemma 3 PT vs IT models.

## What This Repo Contains

This repository is organized around a series of proof-of-concept experiments under `src/poc/`:

- `exp3`: PT vs IT generation-trace analysis, feature populations, and token/word-level behavior.
- `exp4`: adjacent-layer continuity, attention, intrinsic-dimension, and label-style analyses.
- `exp5`: intervention and ablation experiments over content, format, and corrective phases.
- `exp6`: newer governance / structural-feature experiments and dataset-building utilities.
- `shared`: reusable model loading, collection, and configuration components used across experiments.

The codebase is not a packaged library. It is an experiment repo with:

- Python modules in `src/`
- runnable helper scripts in `scripts/`
- datasets in `data/`
- generated outputs in `results/`
- supporting documentation in `docs/`

## Repository Layout

```text
.
├── data/           # experiment datasets and prompt sets
├── docs/           # research notes, issue writeups, and longer documentation
├── logs/           # local run logs
├── results/        # experiment outputs, figures, and merged runs
├── scripts/        # orchestration and utility scripts
├── tools/          # one-off local utilities and audit helpers
├── src/            # experiment code
├── main.py         # minimal entry stub
├── pyproject.toml  # dependencies / project metadata
└── README.md
```

## Quick Start

### Environment

This repo uses `uv` and Python `>=3.13`.

```bash
uv sync
```

Some experiments also require:

- Hugging Face access for gated Gemma checkpoints
- GPU execution for collection runs
- local auth for any Google Cloud Storage upload/download scripts

### Common Commands

Run an experiment entrypoint:

```bash
uv run python -m src.poc.exp3.run
uv run python -m src.poc.exp4.run
uv run python -m src.poc.exp5.run
```

Regenerate plots:

```bash
uv run python -m src.poc.exp3.run_plots --variant it
uv run python -m src.poc.exp4.run_plots
```

Run tests:

```bash
uv run pytest
uv run python tools/test_audit.py
```

## Documentation

Longer-form notes and issue writeups live under `docs/`:

- [POC pipeline notes](docs/poc-pipeline-notes.md)
- [Research notes v2](docs/research-notes-v2.md)
- [Circuit tracer nnsight issue](docs/circuit-tracer-nnsight-issue.md)
- [Exp6 steering design](docs/exp6-steering-design.md)

## Results And Artifacts

This repo intentionally keeps most large result artifacts out of git. In general:

- plots under selected experiment folders may be tracked
- large `.json`, `.jsonl`, and `.npz` run artifacts are usually ignored
- logs are local working artifacts and should not be treated as canonical outputs

## Notes On Repo Conventions

- `src/` is the source of truth for experiment code.
- `scripts/` is for orchestration and helper utilities, not core logic.
- `docs/` holds design notes and longer markdown documents so the repo root stays readable.
- `results/` reflects local experiment state and may contain partially merged or in-progress runs.

## License

This project is released under the [MIT License](LICENSE).
