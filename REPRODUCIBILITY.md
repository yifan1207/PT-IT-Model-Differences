# Reproducibility Audit Guide

This repository exposes the paper claims at three audit levels.

| Audit level | Command | What it verifies | Expected cost |
|---|---|---|---|
| Summary audit | `bash scripts/reproduce/reproduce_claims_from_summaries.sh` | Recomputes headline manuscript numbers from committed JSON/CSV summaries. | CPU only, under 1 minute. |
| Minimal raw shard | `bash scripts/reproduce/reproduce_minimal.sh` | Validates a 20-prompt, one-family raw shard with per-layer logits, intervention outputs, first-divergence records, and expected summaries. | One 80GB A100/H100 for roughly 1-3 GPU-hours if regenerated; cached validation is CPU-only after download. |
| Full rerun | `bash scripts/run/...` plus the experiment-specific source packages | Regenerates full traces and summaries for the paper experiments. | Multi-GPU. Expect hundreds of GPU-hours and hundreds of GB to more than 1 TB transient disk if raw traces are retained. |

## Claim To Command Table

| Claim / number | Command | Expected artifact | Expected number |
|---|---|---|---|
| Six-family tuned final-half convergence gap | `bash scripts/reproduce/reproduce_claims_from_summaries.sh` | `results/exp09_cross_model_observational_replication/data/convergence_gap_values.json` | `0.398` |
| Six-family raw final-half convergence gap | same | same | `0.729` |
| Matched-prefix `JS(A', C)` before late window and final 20% | same | `results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/js_summary.json` | `0.106`, `0.169` |
| Depth-ablation late graft final-20% KL delta | same | `results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/depth_ablation_metrics.json` | `+0.341` nats |
| Symmetric late graft and late swap KL effects | same | `results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_summary.json` | `+0.338`, `-0.509` nats |
| Late random-control specificity | same | `results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/exp19B_summary_light.json` | true `+0.327`, random `+0.003`, margin `+0.324` |
| First-divergence mid vs late identity transfer | same | `results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early/validation_analysis/summary.json` | `26.0%` vs `17.6%`; mirror `31.2%` vs `20.8%` |
| Native late readout loss | same | same | `13.25` logits |
| Late MLP write-out and context gating | same | `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/summary.json`, `analysis/effects.csv` | late support `+0.789`; PT-host late `+0.0035`; upstream `+0.403`, late-weight `+0.148`, interaction `+0.288` |
| Behavioral bridge with human audit | same | `results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_behavior_summary.json`, `results/exp15_symmetric_behavioral_causality/human_eval/human_eval_summary.json` | LLM resolved G2: `56.3%`, `77.1%`; human resolved G2: `60.5%`, `70.6%` |

## Raw Data And Large Artifacts

Git contains paper-facing summaries and plots, not multi-GB raw traces or probe tensors. Large artifacts are mirrored under the project GCS bucket where available:

- `gs://pt-vs-it-results/tuned_lens_probes_v3/`
- `gs://pt-vs-it-results/results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421/`
- `gs://pt-vs-it-results/results/exp20_divergence_token_counterfactual/midlate_factorial_20260424_1116/`
- `gs://pt-vs-it-results/results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_dense5_midlate_factorial_20260424/`

The minimal audit shard is expected at:

```bash
gs://pt-vs-it-results/reproducibility/minimal_audit_shard/
```

Fetch it with:

```bash
mkdir -p results/reproducibility/minimal_audit_shard
gsutil -m rsync -r gs://pt-vs-it-results/reproducibility/minimal_audit_shard/ \
  results/reproducibility/minimal_audit_shard/
bash scripts/reproduce/reproduce_minimal.sh
```

If the shard is absent, the summary audit still verifies the manuscript numbers from committed artifacts; the shard is the reviewer-facing path for auditing raw per-layer records without downloading every full trace.

## Tuned-Lens Reproducibility

The main causal claims do not require tuned-lens checkpoints: matched-prefix JS, graft/swap KL summaries, first-divergence identity/margin, and MLP finite-difference write-out are auditable from committed summaries. Tuned-lens curves are used for the discovery visualization and are paired with raw-lens sensitivity results.

The trained probe archive is mirrored at `gs://pt-vs-it-results/tuned_lens_probes_v3/`. To retrain probes from scratch, use:

```bash
uv run python -m src.poc.cross_model.tuned_lens --model MODEL --variant pt --device cuda:0
uv run python -m src.poc.cross_model.tuned_lens --model MODEL --variant it --device cuda:0
```

The full six-family tuned-lens retrain is 12 runs. On 8x80GB H100/A100 GPUs, the expected wall time is roughly 4-6 hours using joint all-layer training.
