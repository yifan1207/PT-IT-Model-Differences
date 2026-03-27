# PT–IT Transformer Differences

Mechanistic interpretability research code for studying how instruction tuning restructures the computational pipeline of Gemma 3.

This repo centers on a concrete claim:

> pretrained and instruction-tuned models share much of the same early and mid-layer computation, but instruction tuning installs a late corrective/output-governance stage and sharpens a representational phase transition around the first third of the network.

The codebase is organized around a sequence of experiments that test that claim from several angles: observational tracing, cross-layer feature turnover, causal ablation, and steering.

## Research Focus

The current paper-facing story has two linked threads:

1. **Phase transition / pipeline restructuring**
   - PT and IT appear to differ less by wholesale rewriting of the whole model and more by a sharp computational boundary plus late corrective processing.
   - The main empirical signatures are layerwise feature-population shifts, adjacent-layer continuity breaks, altered attention/entropy trajectories, and a distinct late-stage output-shaping regime.

2. **Output governance as a corrective stage**
   - Late IT layers appear to regulate structure, formatting, discourse style, and conversational output form.
   - Exp5 established the need for a late-stage corrective story, but large phase ablations were often too destructive.
   - Exp6 is the more targeted causal follow-up: instead of removing whole phases, it steers specific corrective directions and feature sets.

The most relevant design notes for the current direction are:

- [Phase-transition hypothesis and experiment plan](docs/phase_transition_hypothesis_and_experiments.md)
- [Exp6 steering design](docs/exp6-steering-design.md)
- [Research notes v2](docs/research-notes-v2.md)

## At A Glance

| Experiment | Question | Main Output |
| --- | --- | --- |
| `exp3` | Where do PT and IT differ during generation? | layerwise trace metrics, feature populations, mind-change / KL plots |
| `exp4` | Is there a sharp cross-layer population transition? | adjacent-layer continuity, attention entropy, ID profiles |
| `exp5` | Can content / format / corrective phases be causally separated? | ablation sweeps, progressive skip, benchmark heatmaps |
| `exp6` | Can the corrective stage be steered directly? | corrective-direction steering, feature clamping, governance control plots |

## Key Figures

### Exp4: Adjacent-layer continuity around the dip

![Exp4 adjacent-layer continuity](results/exp4/plots/plot1_jaccard_curve.png)

This figure is the cleanest high-level view of the cross-layer population shift hypothesis: adjacent-layer continuity drops around the dip, and the dip region is treated as a real computational boundary rather than a smooth drift.

### Exp6 A1: remove the corrective direction from IT

![Exp6 A1 dose response](results/exp6/merged_A1_it/plots/A1_dose_response_v5.png)

This is the current headline causal figure. It tests whether the late corrective direction in IT can be removed smoothly rather than by deleting whole layers. The central question is whether governance and structure degrade before general capability collapses.

### Exp6 A5a: progressive skip inside the corrective stage

![Exp6 A5a progressive skip](results/exp6/merged_A5a_it_v1/plots/A5a_progressive_skip.png)

This is the cleaner successor to the older exp5-style progressive skip view. It asks how much of the late corrective stage is actually needed, and whether the effect looks gradual or cliff-like as more of the stage is removed.

## Repository Layout

```text
.
├── data/     # datasets and prompt collections used by the experiments
├── docs/     # paper notes, experiment plans, and issue writeups
├── logs/     # local run logs
├── results/  # generated outputs, merged runs, and tracked figures
├── scripts/  # orchestration, merging, plotting, and utility scripts
├── tools/    # local audit and one-off helper scripts
├── src/      # experiment code and shared runtime components
└── README.md
```

## Repo Guide

### `src/poc/shared`

Reusable loading, collection, and plotting helpers used across experiments. If you want to understand how model execution or tracing is wired, start here.

### `src/poc/exp3`

Generation-trace analysis for PT vs IT:

- emergence / KL / logit-lens behavior
- word-level token stratification
- feature-population analysis
- mind-change and candidate-reshuffling analyses

### `src/poc/exp4`

Cross-layer transition analysis:

- adjacent-layer continuity
- attention entropy
- intrinsic-dimension profiling
- feature-label distribution around the dip

### `src/poc/exp5`

Three-phase ablation experiments:

- content, format, and corrective phase interventions
- progressive skip / directional sweeps
- benchmark evaluation and checkpoint metric summaries

This experiment family is important for the historical arc of the project: it established the corrective-stage framing, but it also showed that broad phase ablations can be too destructive. That is one of the reasons Exp6 became central.

### `src/poc/exp6`

Steering experiments:

- corrective direction addition/removal
- feature clamping
- control feature comparisons
- layer-specific governance interventions

If you want the current strongest causal story in the repo, start here after reading the hypothesis note.

## Quick Start

### Environment

The repo uses `uv` with Python `>=3.13`.

```bash
uv sync
```

Most collection runs also need:

- Hugging Face access for gated Gemma checkpoints
- GPU execution
- local credentials if you use the GCS upload/download utilities

### Common entrypoints

Run core experiments:

```bash
uv run python -m src.poc.exp3.run
uv run python -m src.poc.exp4.run
uv run python -m src.poc.exp5.run
uv run python -m src.poc.exp6.run --help
```

Regenerate plots:

```bash
uv run python -m src.poc.exp3.run_plots --variant it
uv run python -m src.poc.exp4.run_plots
uv run python scripts/plot_exp6_B.py
```

Run tests:

```bash
uv run pytest
uv run python tools/test_audit.py
```

## How To Use The Code

### If you want to reproduce the current paper-style analyses

1. Generate or load the PT/IT trace artifacts in `results/exp3`.
2. Regenerate exp3 plots.
3. Use exp4 to quantify the transition around the dip.
4. Use exp5 to understand the broad phase-ablation results and their failure modes.
5. Use exp6 for the main causal tests of the corrective stage.

### If you are new to the repo

Best reading order:

1. [docs/phase_transition_hypothesis_and_experiments.md](docs/phase_transition_hypothesis_and_experiments.md)
2. [docs/exp6-steering-design.md](docs/exp6-steering-design.md)
3. [src/poc/exp4](src/poc/exp4)
4. [src/poc/exp3](src/poc/exp3)
5. [src/poc/exp5](src/poc/exp5)
6. [src/poc/exp6](src/poc/exp6)

### If you only need the current finalized figure folders

Tracked result plots are intentionally selective. At the moment, the most stable git-tracked figure folders are:

- `results/exp4/plots`
- `results/exp5/merged_progressive_it/plots`
- `results/exp6/merged_A1_it/plots`
- `results/exp6/merged_A5a_it_v1/plots`
- `results/exp6/plots_B`

Most large intermediate result artifacts remain ignored.

## Documentation

Project notes and design documents live in [docs](docs/README.md):

- [phase_transition_hypothesis_and_experiments.md](docs/phase_transition_hypothesis_and_experiments.md)
- [exp6-steering-design.md](docs/exp6-steering-design.md)
- [exp3_plan.md](docs/exp3_plan.md)
- [EVAL_REDESIGN_v1.md](docs/EVAL_REDESIGN_v1.md)
- [poc-pipeline-notes.md](docs/poc-pipeline-notes.md)
- [research-notes-v2.md](docs/research-notes-v2.md)
- [circuit-tracer-nnsight-issue.md](docs/circuit-tracer-nnsight-issue.md)

## Working Conventions

- `src/` is the source of truth for experiment code.
- `scripts/` is for orchestration and postprocessing, not core model logic.
- `results/` reflects local experiment state and is only partially tracked.
- `docs/` contains the evolving research narrative and experiment plans.

## License

Released under the [MIT License](LICENSE).
