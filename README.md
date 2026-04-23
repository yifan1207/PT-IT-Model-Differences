# Instruction Tuning Creates a Broad Convergence-Gap Signature: Strongest Tested Leverage Lies in Late MLPs

### Cross-Family Localization and Behavioral Consequences Across Transformer Families

<p align="center">
  <img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.5+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="6 Model Families" src="https://img.shields.io/badge/models-6%20families-green?style=flat-square">

</p>

> **TL;DR** &mdash; The current paper story is paired: instruction tuning creates a **broad convergence-gap signature** under native decoding across six model families, and the strongest tested internal leverage on that signature lies in a **late MLP window** under matched-prefix control. The matched-prefix graft/swap experiments are the cross-model internal backbone, the free-running behavioral experiments show that the same late intervention family moves a real but partial slice of assistant behavior, and Gemma steering serves as a concrete single-model bridge rather than the main cross-family evidence base.

![Broad convergence gap across six model families](docs/assets/readme_broad_convergence_gap.png)

*Figure 1. Tuned-lens KL-to-own-final curves from the main cross-family observational suite. Across all six families, IT stays farther from its own final distribution than PT through much of the stack, making the broad convergence gap the primary cross-model signature.*

---

## Start Here

If you are new to the repo, these are the most useful entrypoints:

- [docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md): canonical experiment map and path conventions
- [scripts/README.md](scripts/README.md): grouped script layout and common commands
- `uv run python scripts/infra/repo_doctor.py`: lightweight repo health check
- [paper_draft/PAPER_DRAFT_v13.md](paper_draft/PAPER_DRAFT_v13.md): current paper framing

The repo has been reorganized into descriptive canonical paths:

- experiment code: `src/poc/exp##_descriptive_name/`
- results: `results/exp##_descriptive_name/`
- scripts: `scripts/run/`, `scripts/plot/`, `scripts/analysis/`, `scripts/infra/`, etc.

A few flat script aliases are still kept where practical, but results now live only under the descriptive canonical paths.

---

## Current Status

The current paper-facing story is best understood in three layers:

| Layer | Best current claim | Main evidence |
|---|---|---|
| Observational | Instruction tuning creates a broad convergence-gap signature under native decoding | `exp09` cross-model PT/IT analyses |
| Internal causal | The strongest tested leverage on the convergence gap is late-centered and MLP-heavy | `exp11` matched-prefix grafts + `exp14` symmetric sufficiency/necessity |
| Behavioral | The same late intervention family moves a real but partial slice of assistant behavior, strongest on the IT-side necessity test | `exp12` free-running A/B/C + `exp15` symmetric behavioral analysis |

What is strongest right now:

- broad IT-vs-PT convergence gap and delayed commitment across 6 families under both tuned and raw logit lenses
- a late-concentrated IT-vs-PT increase in residual opposition as a geometric companion, with architecture-dependent magnitude and spatial extent
- matched-prefix late graft/swap as the main cross-model internal backbone for late-window MLP localization of the convergence gap
- Gemma steering as the clearest single-direction causal bridge between convergence speed and governance behavior, consistent with that cross-family backbone
- `exp13A-lite` plus the exp13/14 mechanism summaries as evidence that the late stage is broader than a narrow formatting-token injector
- free-running A/B/C as a behavioral precision finding: late MLPs move anti-raw-continuation / anti-false-refusal more than polished structure

What remains intentionally careful:

- the free-running six-family observational curves are descriptive, not matched-history estimates
- `KL(layer || own final)` is useful but endpoint-sensitive
- dimensionality diagnostics are exploratory / mixed and are not part of the main claim
- late IT MLPs are the **strongest tested leverage window inside a broader middle-to-late circuit**, not a full assistantness module

![Late-window sufficiency and necessity under matched-prefix control](docs/assets/readme_exp14_causal_main.png)

*Figure 2. Symmetric matched-prefix exp13/14 summary. The late IT→PT graft is the strongest sufficiency window and the mirrored late PT→IT swap is the strongest necessity window on the primary late-region KL metric, supporting late-window localization with the strongest tested leverage at the end of the stack rather than a diffuse endpoint-only story.*

---

## Quickstart

### Setup

```bash
git clone <repo> && cd structral-semantic-features
uv sync
```

### Sanity-check the repo

```bash
uv run python scripts/infra/repo_doctor.py
```

Optional:

```bash
uv run python scripts/infra/repo_doctor.py --pytest
```

### Explore the main runnable entrypoints

```bash
# Canonical exp14 matched-prefix causal runner
uv run python -m src.poc.exp14_symmetric_matched_prefix_causality --help

# Canonical exp16 matched-prefix native-JS replay runner
uv run python -m src.poc.exp16_matched_prefix_js_gap --help

# Canonical exp15 free-running behavioral runner
uv run python -m src.poc.exp15_symmetric_behavioral_causality --help

# Local smoke for the exp13+14 causal stack
bash scripts/run/run_exp13_exp14_local.sh --mode smoke --model gemma3_4b --smoke-prompts 8
```

### Common analysis / plotting commands

```bash
# Current cross-model observational figures
uv run python -m src.poc.exp09_cross_model_observational_replication.plot_replication

# Exp13A-lite analysis + plots
uv run python scripts/analysis/analyze_exp13a_lite.py --help
uv run python scripts/plot/plot_exp13a_lite.py --help

# Exp13 full + Exp14 causal summary plots
uv run python scripts/analysis/analyze_exp13_full.py --help
uv run python scripts/plot/plot_exp13_full.py --help

# Exp16 native-JS replay analysis + plots
uv run python scripts/analysis/analyze_exp16.py --help
uv run python scripts/plot/plot_exp16.py --help
```

### Canonical run scripts

```bash
# Multi-model steering / phase 0
bash scripts/run/run_phase0_multimodel.sh --step precompute
bash scripts/run/run_phase0_multimodel.sh --step steer

# Exp13 + Exp14 local causal campaign
bash scripts/run/run_exp13_exp14_local.sh --mode full

# Exp16 local JS replay over the frozen exp14 teacher stream
bash scripts/run/run_exp16_js_replay_local.sh --mode smoke
```

---

## Models

| Model | Layers | d_model | Architecture | Pretraining / Post-training |
|-------|--------|---------|-------------|-----------------------------|
| **Gemma 3 4B** (primary) | 34 | 2560 | GQA, hybrid local/global (5:1) | Undisclosed pretraining / KD + supervised + preference + rule-based stages |
| **Llama 3.1 8B** | 32 | 4096 | GQA, all global | 15T-token pretraining / iterative supervised + preference optimization |
| **Qwen 3 4B** | 36 | 2560 | GQA, all global | 36T-token multilingual pretraining / multi-stage SFT + RL post-training |
| **Mistral 7B v0.3** | 32 | 4096 | GQA, sliding window (4096) | Undisclosed pretraining / instruct checkpoint |
| **DeepSeek-V2-Lite** | 27 | 2048 | MLA, MoE (2 shared + 64 routed, top-6) | 5.7T-token pretraining / **SFT-only** chat checkpoint |
| **OLMo 2 7B** | 32 | 4096 | MHA, all global | `OLMo-mix-1124` pretraining / T&uuml;lu 3-style SFT + DPO + RLVR |

OLMo 2 uses a staged base-model recipe with a late `Dolmino-mix-1124` curriculum, so the earlier single-dataset shorthand is inaccurate for this checkpoint. DeepSeek-V2-Lite-Chat is both the only MoE family here and an SFT-only chat checkpoint, so we treat it as a post-training outlier rather than as evidence for a uniform six-family IT recipe.

All main observational analyses use each IT model's native chat template and raw prompting for PT. Template-free conditions are treated as ablations rather than replacement primaries.

---

## Project structure

```
src/poc/
  cross_model/                                   # Shared multi-model infrastructure
  exp01_hierarchical_distributional_narrowing/
  exp02_ic_ooc_reasoning_mechanistic_comparison/
  exp03_corrective_stage_characterization/
  exp04_phase_transition_characterization/
  exp05_corrective_direction_ablation_cartography/
  exp06_corrective_direction_steering/
  exp07_methodology_validation_tier0/
  exp08_multimodel_steering_phase0/
  exp09_cross_model_observational_replication/
  exp10_contrastive_activation_patching/
  exp11_matched_prefix_mlp_graft/
  exp12_free_running_abc_graft/
  exp13_late_stage_token_support_analysis/
  exp14_symmetric_matched_prefix_causality/
  exp15_symmetric_behavioral_causality/
  exp16_matched_prefix_js_gap/

scripts/
  analysis/                                      # Post-hoc summaries, cross-checks, paper stats
  data/                                          # Dataset builders / data prep
  eval/                                          # Judge and evaluation entrypoints
  infra/                                         # Modal/Lambda/cloud helpers
  merge/                                         # Worker/shard merge utilities
  plot/                                          # Figure generation
  precompute/                                    # Direction extraction and preprocessing
  run/                                           # Main experiment launchers
  scoring/                                       # Rescoring utilities

results/
  cross_model/{model}/
  exp01_hierarchical_distributional_narrowing/
  ...
  exp15_symmetric_behavioral_causality/
```

Canonical experiment/result paths now use descriptive names. Source code now lives only in the canonical named experiment folders. Some legacy result and flat script aliases are still kept during the results/scripts migration so older commands keep working.

For a full index, see [docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md).

---

## Experiment index

### Observational (cross-model, 6/6)

| ID | Analysis | Key result |
|----|----------|------------|
| **L1** | &delta;-cosine profiles | IT adds more late residual opposition than PT in all 6 families, but with heterogeneous magnitude (&minus;0.021 to &minus;0.269 in the final 20%) |
| **L2** | Broad convergence gap + delayed commitment (5 metrics &times; 2 lenses) | IT stays farther from its own final distribution through much of the stack and commits later in all 6 families |
| **L3** | Weight change localization | Gemma: concentrated at corrective layers; others: uniform |
| **L8** | Geometry follow-up | Exploratory dimensionality / covariance diagnostics are mixed and not part of the core evidence chain |
| **L9** | Attention entropy divergence | Architecture-dependent |

### Causal steering (Gemma, extending to all 6)

| ID | Experiment | Key result |
|----|-----------|------------|
| **A1** | &alpha;-sweep on corrective layers | Governance dose-response, content flat |
| **A1_rand** | Random direction control | 3&times; less governance effect &mdash; direction specificity |
| **A1_notmpl** | No chat template | Dose-response preserved &mdash; weight-encoded |
| **A2** | Inject into PT | Noisy &mdash; PT lacks downstream circuitry |
| **A5a** | Progressive layer skipping | Final 3 layers: format; earlier: coherence |

### Matched-prefix Internal Causality

| ID | Experiment | Key result |
|----|-----------|------------|
| **exp11** | Matched-prefix late IT MLP graft | Late IT MLPs increase late KL-to-own-final and move PT internal predictions toward the IT teacher under shared token history |
| **exp13A-lite** | Descriptive token-support analysis | Late grafts broadly suppress raw-continuation-like `FUNCTION/OTHER` candidates and increase support for the eventual teacher token |
| **exp16** | Matched-prefix native-JS replay | Direct same-layer JS under frozen exp14 teacher histories removes unmatched-history and own-final-endpoint dependence from the main internal divergence readout |
| **exp14** | Symmetric sufficiency / necessity | Late IT→PT graft is the strongest sufficiency window and late PT→IT swap is the strongest necessity window across all 6 models on the primary late-region KL metric |

### Free-running Behavioral Causality

| ID | Experiment | Key result |
|----|-----------|------------|
| **exp12** | A/B/C free-running graft comparison | Late graft reduces benign false refusals in 6/6 families and improves assistant register in 4/6, but remains far from the full IT endpoint on polished structure |
| **exp15** | Symmetric behavioral causality | Current canonical behavioral estimate of the same late intervention family, with the clearest effects on the IT-side necessity test and weaker but real PT-side recovery |

### Methodology validation (Tier 0)

| ID | Test | Result |
|----|------|--------|
| **0A** | Direction bootstrap stability | cos > 0.993 by n=300 |
| **0B** | Matched-token direction | cos = 0.82 (primarily weight-driven) |
| **0C** | Projection-matched random | 3&times; less governance, identical content degradation |
| **0D** | Bootstrap 95% CIs | BCa intervals on all metrics |
| **0E** | Classifier robustness | Robust to all boundary perturbations |
| **0F** | Layer range sensitivity | Stable across 4 overlapping ranges |
| **0G** | Tuned-lens commitment | Primary commitment measurement (6 models &times; 2 variants) |
| **0H** | Calibration split | Three disjoint prompt sets &rarr; same dose-response |
| **0I** | Formula comparison | MLP projection only; attention/residual fail |
| **0J** | Onset threshold sensitivity | Robust across &sigma;-based and absolute thresholds |

### Contrastive activation patching (Exp10, in progress)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Forced-decoding paired data collection | Prototype complete |
| 2 | Ridge probes &rarr; convergence direction (d_conv) | Prototype complete |
| 3 | Causal activation patching (5 conditions) | Prototype complete |
| 4 | Steering with d_conv vs d_mean | Prototype: d_mean steers (11&ndash;19&times;), d_conv does not |

---

## Pipeline design

The steering pipeline is **architecture-agnostic**. It operates on raw MLP activations via a model-agnostic adapter system &mdash; no transcoders, SAEs, or model-specific decompositions required.

```
Direction Extraction          Steering                Evaluation
--------------------    --------------------    --------------------
IT model --+            IT model + hooks        LLM judge (G1/G2)
           |-- d_mean   h += (alpha-1)(d'h)d    Programmatic (STR)
PT model --+            per corrective layer    IFEval compliance
                                                MMLU / GSM8K / reasoning
```

The adapter system provides a uniform interface across all six architectures, including DeepSeek's MoE routing and Gemma's hybrid attention. Extending to a new model requires only registering its architecture in the adapter config.

---

## Citation

```bibtex
@article{anonymous2026corrective,
  title={Instruction Tuning Creates a Broad Convergence Gap: A Late-Centered Corrective Computation Across Transformer Families},
  author={Anonymous},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
