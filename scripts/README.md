# Scripts Layout

The `scripts/` folder is now organized by function instead of keeping every entrypoint flat at the top level.

## Folders

- `scripts/run/`: experiment launchers and shell orchestration
- `scripts/plot/`: figure generation
- `scripts/analysis/`: summary and paper-facing analysis
- `scripts/precompute/`: direction extraction and preprocessing
- `scripts/eval/`: judge and evaluation entrypoints
- `scripts/merge/`: worker/shard merge helpers
- `scripts/scoring/`: rescoring utilities
- `scripts/infra/`: Modal/Lambda/remote orchestration helpers
- `scripts/data/`: dataset build/prep scripts

## Compatibility

Flat historical entrypoints such as `scripts/run_phase0_multimodel.sh` and `scripts/plot_exp7_tier0.py` are kept as symlinks where practical. New commands should prefer the grouped locations, for example:

- `bash scripts/run/run_phase0_multimodel.sh`
- `uv run python scripts/plot/plot_validation_tier0.py`
- `uv run python scripts/precompute/precompute_directions_multimodel.py`
- `uv run python scripts/infra/repo_doctor.py`

## Health check

For a lightweight project-scoped validation pass, run:

- `uv run python scripts/infra/repo_doctor.py`

Optional:

- `uv run python scripts/infra/repo_doctor.py --pytest`

This checks:

- Python syntax under `src/` and `scripts/`
- shell syntax for launchers
- broken script symlinks
- a few canonical `--help` smoke entrypoints

## Related docs

- Experiment map: [`docs/EXPERIMENT_REGISTRY.md`](../docs/EXPERIMENT_REGISTRY.md)
- Top-level overview: [`README.md`](../README.md)
