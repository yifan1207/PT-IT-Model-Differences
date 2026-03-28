# PT–IT Transformer Differences

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="uv" src="https://img.shields.io/badge/package%20manager-uv-5C5CFF?style=flat-square">
  <img alt="Focus" src="https://img.shields.io/badge/main%20case-Gemma%203%204B-FF8C42?style=flat-square">
  <img alt="Scope" src="https://img.shields.io/badge/scope-cross--model%20replication-2E8B57?style=flat-square">
</p>

Mechanistic interpretability research code for studying how **instruction tuning changes transformer computation**.

The repo started as a deep dive into **Gemma 3 4B PT vs IT**, and that is still the main mechanistic case study. It now also includes **cross-model replication experiments** to test whether the same PT→IT signatures recur in other model families.

This is not just a benchmarking repo and not just a model-comparison repo. The central goal is to understand, mechanistically, what instruction tuning *does* inside a transformer:

- where pretrained and instruction-tuned models still share computation
- where they begin to diverge
- whether that divergence looks like a smooth drift or a real phase boundary
- whether late instruction-tuned layers act like a corrective stage that governs output form
- whether that picture is specific to Gemma 3 4B or recurs across model families

## Core Claim

> Pretrained and instruction-tuned models share much of the same early and mid-layer computation, but instruction tuning installs a late corrective / output-governance stage and sharpens a representational phase transition around the first third of the network.

In plainer language:

- **PT and IT are not different everywhere**
  - much of the early and middle network still looks shared
- **the biggest difference appears late**
  - especially in how the model shapes output structure, format, register, and conversational behavior
- **there is a boundary, not just a blur**
  - the repo’s observational experiments argue that PT→IT differences cluster around a sharp transition region
- **the late stage is causal**
  - the steering experiments test whether removing or injecting corrective directions changes governance-like behavior directly

## What This Repo Shows

<table>
  <tr>
    <td width="33%">
      <strong>1. Phase transition</strong><br>
      PT and IT do not differ by uniformly rewriting the whole model. The strongest change appears around a sharp cross-layer boundary.
    </td>
    <td width="33%">
      <strong>2. Corrective stage</strong><br>
      Late IT layers regulate output form: structure, formatting, conversational register, and governance.
    </td>
    <td width="33%">
      <strong>3. Cross-model generalization</strong><br>
      Gemma 3 4B is the main mechanistic focus, but the repo now also tests whether similar PT→IT signatures recur across other model families.
    </td>
  </tr>
</table>

## The Research Story

The work in this repo evolved in stages, and the code reflects that.

### Stage 1: observational evidence

The first question was simple:

> if you compare a pretrained model and its instruction-tuned counterpart token by token and layer by layer, where do they actually differ?

That led to the `exp3` and `exp4` pipeline:

- generation-trace collection
- logit-lens comparisons
- layerwise KL and commitment analyses
- active-feature and feature-population analyses
- adjacent-layer continuity / “dip” analysis
- attention entropy and intrinsic-dimension profiling

The outcome of that stage was a strong observational picture:

- PT and IT are not uniformly different
- there is a meaningful transition region around roughly the first third of the network
- after that point, late IT layers behave differently enough to motivate a “corrective” or “output-governance” interpretation

### Stage 2: broad causal tests

The next question was:

> can we causally isolate content, format, and corrective phases?

That is what `exp5` was built for.

The answer was mixed:

- broad ablations were informative
- but many phase-level interventions were too destructive to serve as clean causal evidence

That matters for the paper story: `exp5` is not wasted work. It is the point where the project learned that “delete a whole phase” is too coarse, and that a better causal test would need more targeted interventions.

### Stage 3: targeted steering of the corrective stage

That led to `exp6`, which is the current center of gravity for the causal claims.

Instead of ablating whole layer blocks, `exp6` asks:

- can we remove the late corrective direction from IT?
- can we inject it into PT?
- can we intervene on specific feature sets instead of whole layers?
- can we show dose-response behavior rather than brittle on/off collapse?

This is the part of the repo that most directly supports the current paper framing:

- the late stage is not just *correlated* with output governance
- it appears to be *causally involved* in it

### Stage 4: beyond one model family

A reasonable reviewer question is:

> is this just a Gemma 3 4B artifact?

That is what the newer cross-model work is for.

The cross-model results are not as mechanistically deep as the Gemma experiments, but they are important because they test whether the same high-level PT→IT structure appears elsewhere:

- similar transition depth in fractional terms
- similar commitment / entropy signatures
- similar evidence for a late-stage behavioral or corrective shift

So the repo now has two layers:

- **Gemma 3 4B** for the deepest mechanistic and causal analysis
- **cross-model replication** for the external-validity story

## Project Map

| Area | Question | Main role in the paper |
| --- | --- | --- |
| `exp3` | Where do PT and IT diverge during generation? | detailed trace analysis |
| `exp4` | Is there a real cross-layer transition around the dip? | main observational evidence |
| `exp5` | Can content / format / corrective phases be separated causally? | broad ablation attempt and failure analysis |
| `exp6` | Can the corrective stage be steered directly? | main causal evidence |
| `cross_model` | Does the same PT→IT structure appear outside Gemma? | generalization layer |

If you only want the shortest summary:

- `exp3`/`exp4` explain **what changes**
- `exp5` explains **why broad ablation was not enough**
- `exp6` tests **what the late stage actually does**
- `cross_model` asks **whether the story generalizes**

## Key Figures

### 1. Observational boundary in Gemma 3 4B

![Exp4 adjacent-layer continuity](results/exp4/plots/plot1_jaccard_curve.png)

This is the cleanest observational figure in the repo: adjacent-layer continuity drops around the dip, which supports a real computational boundary rather than a smooth drift.

Why it matters:

- if PT→IT differences were smeared smoothly across all layers, this figure would look much flatter
- instead, it supports the idea that the model crosses into a different computational regime

### 2. Main causal result: remove the corrective direction

![Exp6 A1 dose response](results/exp6/merged_A1_it/plots/A1_dose_response_v5.png)

This is the headline causal figure. It asks whether governance and output structure degrade when the late IT corrective direction is removed, while content remains comparatively stable.

Why it matters:

- this is stronger than saying “late layers look different”
- it asks whether the late-stage difference is functionally important
- the ideal outcome is selective degradation in governance-like behavior without immediate collapse of core content performance

### 3. How much corrective stage is actually needed?

![Exp6 A5a progressive skip](results/exp6/merged_A5a_it_v1/plots/A5a_progressive_skip.png)

This figure measures whether the corrective stage behaves like a gradual dose-response or a sharper cliff as more of the late stage is skipped.

Why it matters:

- it helps distinguish “one fragile magic layer” from a broader late-stage computation
- it also shows whether the corrective stage behaves more like a spread-out mechanism or a narrow bottleneck

### 4. Cross-model summary

![Cross-model summary](results/cross_model/plots/L1_summary.png)

This is the bridge from the Gemma case study to the broader claim: the repo now checks whether the same PT→IT signatures recur across model families, not just in Gemma 3 4B.

Why it matters:

- this is the main answer to the “is this Gemma-specific?” objection
- it makes the repo useful not just as one-model interpretability work, but as a proposal about instruction tuning more generally

### 5. Cross-model commitment-depth comparison

![Cross-model commitment](results/cross_model/plots/L2_commitment.png)

This asks whether the base→instruct change appears at a comparable **fractional depth** across models, which matters more than exact layer number.

Why it matters:

- different models have different layer counts
- the right comparison is often normalized depth, not literal layer index
- this is part of the argument that the corrective-stage story is architectural rather than accidental

## How To Read The Repo

There are two layers of evidence here:

1. **Gemma 3 4B as the main mechanistic case**
   - deepest tracing
   - strongest causal steering results
   - most detailed feature-level interpretation

2. **Cross-model replication as the generalization layer**
   - lighter-weight than the Gemma analysis
   - tests whether the same PT→IT structure shows up elsewhere

Short version:

- use **Gemma 3 4B** to understand the mechanism
- use **cross-model results** to argue the phenomenon is not Gemma-specific

If you are reading this as a paper/code reviewer, the intended interpretation is:

- **Gemma 3 4B** is the main mechanistic proof-of-concept
- **exp6** is the main causal section
- **cross_model** is the generalization section
- **exp5** is historically important because it explains why the final causal design moved away from coarse whole-phase ablation

## Quick Start

### Environment

```bash
uv sync
```

Typical requirements:

- Python `>=3.13`
- GPUs for collection and intervention runs
- Hugging Face access for gated checkpoints
- optional GCS credentials for result sync

### Common commands

Run the main experiment families:

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
uv run python scripts/plot_exp6_dose_response.py
uv run python scripts/plot_exp6_B.py
```

Those commands are the fastest way to get oriented:

- `exp3.run_plots` regenerates the detailed PT vs IT analysis figures
- `exp4.run_plots` regenerates the clean transition/dip figures
- `plot_exp6_dose_response.py` regenerates the main steering figures
- `plot_exp6_B.py` regenerates the feature-level steering and control figures

Inspect the main tracked figure folders:

```bash
ls results/exp4/plots
ls results/exp6/merged_A1_it/plots
ls results/exp6/merged_A5a_it_v1/plots
ls results/cross_model/plots
```

Run tests:

```bash
uv run pytest
uv run python tools/test_audit.py
```

## Recommended Reading Order

1. [docs/phase_transition_hypothesis_and_experiments.md](docs/phase_transition_hypothesis_and_experiments.md)
2. [docs/exp6-steering-design.md](docs/exp6-steering-design.md)
3. [docs/model_ablation.md](docs/model_ablation.md)
4. [src/poc/exp4](src/poc/exp4)
5. [src/poc/exp3](src/poc/exp3)
6. [src/poc/exp5](src/poc/exp5)
7. [src/poc/exp6](src/poc/exp6)

If you want a faster reading path:

1. look at the figures in `results/exp4/plots`
2. look at `results/exp6/merged_A1_it/plots`
3. look at `results/cross_model/plots`
4. then read the docs for the detailed framing

## Repo Layout

```text
.
├── data/         # datasets and prompt collections
├── docs/         # paper notes, plans, and design docs
├── logs/         # local run logs
├── results/      # merged runs, tracked figures, intermediate outputs
├── scripts/      # plotting, merging, orchestration, utilities
├── tools/        # one-off local helpers and audits
├── src/          # experiment code and shared runtime
└── README.md
```

## Important Folders

| Path | What it contains |
| --- | --- |
| `src/poc/shared` | shared model loading, collection, and utility code |
| `src/poc/exp3` | detailed PT vs IT generation-trace analysis |
| `src/poc/exp4` | cross-layer continuity / dip analysis |
| `src/poc/exp5` | broad phase-ablation experiments |
| `src/poc/exp6` | directional and feature-level steering |
| `results/cross_model` | cross-model replication figures and summaries |

Two practical notes:

- most of the heavy experimental state lives under `results/`
- only selected figure folders are tracked in git, so the checked-in plots represent the stable presentation layer rather than every intermediate artifact

## Stable Figure Folders

Tracked result plots are intentionally selective. The most useful git-tracked figure folders right now are:

- `results/exp4/plots`
- `results/exp5/merged_progressive_it/plots`
- `results/exp6/merged_A1_it/plots`
- `results/exp6/merged_A5a_it_v1/plots`
- `results/exp6/plots_B`
- `results/cross_model/plots`

Most large intermediate result artifacts remain ignored.

That is intentional. The repo is meant to stay readable even though the underlying experiments can generate very large trace and result files.

## Documentation

Main project notes live in [docs](docs/README.md):

- [phase_transition_hypothesis_and_experiments.md](docs/phase_transition_hypothesis_and_experiments.md)
- [exp6-steering-design.md](docs/exp6-steering-design.md)
- [model_ablation.md](docs/model_ablation.md)
- [exp3_plan.md](docs/exp3_plan.md)
- [EVAL_REDESIGN_v1.md](docs/EVAL_REDESIGN_v1.md)
- [poc-pipeline-notes.md](docs/poc-pipeline-notes.md)
- [research-notes-v2.md](docs/research-notes-v2.md)
- [circuit-tracer-nnsight-issue.md](docs/circuit-tracer-nnsight-issue.md)

You do not need to read those to understand the repo at a high level, but they are useful if you want the full paper context, design history, or implementation caveats.

## Working Conventions

- `src/` is the source of truth for experiment code.
- `scripts/` is for orchestration and postprocessing.
- `results/` reflects local experiment state and only selected figure folders are tracked.
- `docs/` contains the evolving research narrative.

## License

Released under the [MIT License](LICENSE).
