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
| Residual-state x late-stack factorial, common-IT readout | `bash scripts/reproduce/reproduce_claims_from_summaries.sh` | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/exp23_summary.json`; `analysis/exp23_effects.csv` | late from PT upstream `+0.572`; late from IT upstream `+3.207`; interaction `+2.635` logits over `2,983` prompt clusters |
| Gemma-removed and family checks | same | same | Gemma-removed interaction `+1.77`; family interactions all positive; median family `+1.85` |
| Label-swap null for the factorial | same | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/compatibility_permutation/` | observed interaction `+2.64`; null 99.9th percentile `+0.239`; `p=5.0e-5` |
| First-divergence position sensitivity | same | `results/paper_synthesis/exp23_position_sensitivity_table.csv`; `results/paper_synthesis/exp23_position_sensitivity_per_family.csv` | drop position 0 `+2.25`; position `>=3` `+1.52`; Gemma-removed position `>=3` `+0.79`; position `>=5` `+1.64` |
| Content/reasoning stress test | same | `results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/analysis/exp23_summary.json` | interaction `+1.81`; late from PT upstream `-1.18` |
| Qwen2.5-32B external-validity check | same | `results/exp24_32b_external_validity/exp24_qwen25_32b_full_eval_v21_20260427_194839/analysis/`; `results/paper_synthesis/exp24_32b_external_validity/` | interaction `+1.446`; late from PT upstream `+0.977`; late from IT upstream `+2.423`; position `>=3` interaction `+1.020`; raw-KL IT-side interaction `+0.465` |
| OLMo-2 stage-progression case study | same | `results/paper_synthesis/exp25_olmo_stage_full_20260428_0905/olmo_stage_progression_table.csv`; `olmo_stage_progression_summary.json`; `olmo_stage_preflight.json` | Base→SFT interaction `+0.782`; SFT→DPO `+0.135`; DPO→RLVR `+0.016`; Base→RLVR `+1.930` |
| First-divergence identity split | same | `results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early/validation_analysis/summary.json` | raw-shared middle vs late IT-token transfer `26.0%` vs `17.6%`; mirror `31.2%` vs `20.8%` |
| MLP write-out proxy | same | `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/summary.json`; `analysis/effects.csv` | late IT-token support `+0.789`; PT-host late `+0.0035`; residual-opposing component `-0.0046` |
| Dense-5 delayed-stabilization context | same | `results/exp09_cross_model_observational_replication/data/convergence_gap_values.json`; `results/paper_synthesis/exp22_endpoint_deconfounded_table.csv` | tuned final-half gap `0.410`; raw final-half gap `0.771`; endpoint-matched raw late gap `+0.425`; endpoint-matched tuned late gap `+0.762` |
| Matched-prefix late-window localization and random control | same | `results/exp11_matched_prefix_mlp_graft/.../depth_ablation_metrics.json`; `results/exp14_symmetric_matched_prefix_causality/.../exp13_full_summary.json`; `results/exp19_late_mlp_specificity_controls/.../exp19B_summary_light.json` | late graft `+0.341`; late swap `-0.509`; true late random-control contrast `+0.327` vs random `+0.003` |
| Behavioral sanity check and human audit | same | `results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_behavior_summary.json`; `results/exp15_symmetric_behavioral_causality/human_eval/human_eval_summary.json` | LLM resolved G2 `56.3%`, `77.1%`; human resolved G2 `60.2%`, `73.1%` |

## Raw Data And Large Artifacts

Git contains paper-facing summaries and plots, not multi-GB raw traces or probe tensors. Large artifacts are mirrored under the project GCS bucket where available:

- `gs://pt-vs-it-results/tuned_lens_probes_v3/`
- `gs://pt-vs-it-results/results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421/`
- `gs://pt-vs-it-results/results/exp20_divergence_token_counterfactual/midlate_factorial_20260424_1116/`
- `gs://pt-vs-it-results/results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_dense5_midlate_factorial_20260424/`
- `gs://pt-vs-it-results/results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/`
- `gs://pt-vs-it-results/results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/`
- `gs://pt-vs-it-results/results/exp21_productive_opposition/exp21_content_reasoning_20260427_0943_h100x8/`
- `gs://pt-vs-it-results/results/exp24_32b_external_validity/exp24_qwen25_32b_full_eval_v21_20260427_194839/`
- Exp25 OLMo-2 stage-progression raw and analysis artifacts are reproducible from the committed synthesis table plus `scripts/run/run_exp25_olmo_stage_progression_runpod.sh`; the paper-facing summaries live under `results/paper_synthesis/exp25_olmo_stage_full_20260428_0905/`.

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

The dense-5 tuned-lens retrain is 10 runs; including the DeepSeek side case is 12 runs. On 8x80GB H100/A100 GPUs, the expected wall time is roughly 4-6 hours using joint all-layer training.
