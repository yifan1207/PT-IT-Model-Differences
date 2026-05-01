# Experiment Registry

This repo now uses descriptive canonical paths for experiments, results, and script entrypoints.

- Canonical code paths live under `src/poc/exp##_descriptive_name/`.
- Canonical result roots live under `results/exp##_descriptive_name/`.
- Canonical scripts live under grouped folders such as `scripts/run/`, `scripts/plot/`, and `scripts/analysis/`.
- Source experiments now live only in the canonical named folders.
- Some legacy result aliases and flat `scripts/*` entrypoints are still kept during migration.

## Experiment Map

| ID | Canonical name | Code path | Results path | Notes |
|---|---|---|---|---|
| 01 | Hierarchical distributional narrowing | `src/poc/exp01_hierarchical_distributional_narrowing/` | `results/exp01_hierarchical_distributional_narrowing/` | Early attribution broadness / narrowing analyses |
| 02 | IC/OOC reasoning mechanistic comparison | `src/poc/exp02_ic_ooc_reasoning_mechanistic_comparison/` | `results/exp02_ic_ooc_reasoning_mechanistic_comparison/` | Template and reasoning comparison experiments |
| 03 | Corrective stage characterization | `src/poc/exp03_corrective_stage_characterization/` | `results/exp03_corrective_stage_characterization/` | Main corrective-stage observational analysis |
| 04 | Phase transition characterization | `src/poc/exp04_phase_transition_characterization/` | `results/exp04_phase_transition_characterization/` | Phase-transition geometry and feature alignment |
| 05 | Corrective-direction ablation cartography | `src/poc/exp05_corrective_direction_ablation_cartography/` | `results/exp05_corrective_direction_ablation_cartography/` | Precompute + ablation cartography feeding steering |
| 06 | Corrective-direction steering | `src/poc/exp06_corrective_direction_steering/` | `results/exp06_corrective_direction_steering/` | Core steering framework and A-series interventions |
| 07 | Methodology validation Tier 0 | `src/poc/exp07_methodology_validation_tier0/` | `results/exp07_methodology_validation_tier0/` | 0A–0J validation suite |
| 08 | Multimodel steering Phase 0 | `src/poc/exp08_multimodel_steering_phase0/` | `results/exp08_multimodel_steering_phase0/` | Cross-family steering extension and piggyback analyses |
| 09 | Cross-model observational replication | `src/poc/exp09_cross_model_observational_replication/` | `results/exp09_cross_model_observational_replication/` | Main PT/IT observational replication figures |
| 10 | Contrastive activation patching | `src/poc/exp10_contrastive_activation_patching/` | `results/exp10_contrastive_activation_patching/` | Paired-data probes and activation patching |
| 11 | Matched-prefix MLP graft | `src/poc/exp11_matched_prefix_mlp_graft/` | `results/exp11_matched_prefix_mlp_graft/` | Teacher-forced matched-prefix causal grafts |
| 12 | Free-running A/B/C graft | `src/poc/exp12_free_running_abc_graft/` | `results/exp12_free_running_abc_graft/` | Free-running behavioral follow-up to exp11 |
| 13 | Late-stage token-support analysis | `src/poc/exp13_late_stage_token_support_analysis/` | `results/exp13_late_stage_token_support_analysis/` | Analysis-only synthesis built from exp11/12 artifacts |
| 14 | Symmetric matched-prefix causality | `src/poc/exp14_symmetric_matched_prefix_causality/` | `results/exp14_symmetric_matched_prefix_causality/` | Symmetric late-window sufficiency/necessity causal runs; canonical module wrapper forwards to the shared matched-prefix engine |
| 15 | Symmetric behavioral causality | `src/poc/exp15_symmetric_behavioral_causality/` | `results/exp15_symmetric_behavioral_causality/` | Behavioral causal evaluation following exp14 |
| 16 | Matched-prefix JS gap replay | `src/poc/exp16_matched_prefix_js_gap/` | `results/exp16_matched_prefix_js_gap/` | Lightweight exp14 replay under frozen teacher tokens for native same-layer JS divergence |
| 17 | Behavioral-direction replication | `src/poc/exp17_behavioral_direction_replication/` | `results/exp17_behavioral_direction_replication/` | External-paper replication package for Lu 2026 Assistant Axis, Du 2025 truthfulness/refusal directions, and joint comparison against local convergence-gap summaries |
| 18 | Mid-late token handoff | `src/poc/exp18_midlate_token_handoff/` | `results/exp18_midlate_token_handoff/` | Mid-to-late promote/suppress and token-handoff analysis |
| 19 | Late MLP specificity controls | `src/poc/exp19_late_mlp_specificity_controls/` | `results/exp19_late_mlp_specificity_controls/` | Matched late-disruption controls for exp14/15: layer-permuted swaps and residual-projection-matched random output-delta controls. Plan: `docs/EXP19_LATE_MLP_SPECIFICITY_CONTROLS_PLAN.md` |
| 20 | Divergence-token counterfactual | `src/poc/exp20_divergence_token_counterfactual/` | `results/exp20_divergence_token_counterfactual/` | First-divergence identity/margin decomposition: middle-positioned substitutions transfer token identity more often than late-positioned substitutions |
| 21 | Productive opposition | `src/poc/exp21_productive_opposition/` | `results/exp21_productive_opposition/` | MLP write-out and residual-opposition component decomposition for first-divergent tokens |
| 22 | Endpoint-deconfounded gap | `src/poc/exp22_endpoint_deconfounded_gap/` | `results/paper_synthesis/exp22_endpoint_deconfounded_table.csv` | Endpoint-matched convergence-gap context and endpoint-free path checks |
| 23 | Mid-late interaction suite | `src/poc/exp23_midlate_interaction_suite/` | `results/exp23_midlate_interaction_suite/` and `results/paper_synthesis/exp23_dense6_core/` | Primary first-divergence upstream-state x late-stack factorial and Dense-6 synthesis |
| 23b | Mid-late KL factorial | `src/poc/exp23_midlate_kl_factorial/` | `results/exp23_midlate_interaction_suite/` | Raw-KL bridge/factorial support for the first-divergence interaction suite |
| 24 | 32B external validity | `src/poc/exp24_32b_external_validity/` | `results/exp24_32b_external_validity/` and `results/paper_synthesis/exp24_32b_external_validity/` | Qwen2.5-32B repetition of Exp20/21/23 and raw-KL bridge; included as the sixth Dense-6 core pair |
| 25 | OLMo stage progression | `src/poc/exp25_olmo_stage_progression/` | `results/paper_synthesis/exp25_olmo_stage_full_20260428_0905/` | Base/SFT/DPO/Instruct local-transition case study using the first-divergence factorial |
| 26 | Residual-opposition mediation | `src/poc/exp26_residual_opposition_mediation/` | `results/exp26_residual_opposition_mediation/` | Archived support check that removes/scales/flips/PT-levels/randomizes residual-opposing late-MLP components inside the Exp23 factorial; not part of the current paper spine |
| 27 | Natural-rollout residual-opposition NTP | `src/poc/exp27_natural_rollout_residual_opposition_ntp/` | `results/exp27_natural_rollout_residual_opposition_ntp/exp27_full_dense5_combined_20260430_2050/` | Current appendix mechanism check: natural greedy rollout follow-up measuring how residual-opposition ablations affect each model's own generated-token next-token prediction |

## Special Notes

- New exp13-lite outputs should go to `results/exp13_late_stage_token_support_analysis/`.
- New exp14 local or Lambda causal runs should go to `results/exp14_symmetric_matched_prefix_causality/`.
- New exp16 matched-prefix native-JS replays should go to `results/exp16_matched_prefix_js_gap/`.
- New exp19 specificity-control outputs should go to `results/exp19_late_mlp_specificity_controls/`.
- New exp23 paper-facing synthesis outputs should go to `results/paper_synthesis/exp23_dense6_core/`.
- Paper-facing markdown and PDF live directly under `paper_draft/`; older draft bundles are intentionally ignored.
- `results/cross_model/` remains a shared cross-family workspace because it spans multiple numbered experiments.
- Lightweight repo sanity checks live at `scripts/infra/repo_doctor.py`.

## Script Layout

| Folder | Purpose |
|---|---|
| `scripts/run/` | Main experiment launchers and orchestration shells |
| `scripts/plot/` | Plot generation for paper and appendix figures |
| `scripts/analysis/` | Post-hoc summaries, aggregation, and paper-facing analysis |
| `scripts/precompute/` | Direction extraction and preprocessing pipelines |
| `scripts/eval/` | Judge and metric evaluation entrypoints |
| `scripts/merge/` | Worker/shard merge utilities |
| `scripts/scoring/` | Rescoring helpers for generated outputs |
| `scripts/infra/` | Modal, Lambda, sync, and monitoring helpers |
| `scripts/data/` | Dataset construction and data prep utilities |

## Compatibility Policy

- Prefer canonical descriptive paths in new code, docs, and handoffs.
- Result paths are canonical-only; short numeric result aliases are no longer part of the repo layout.
- When unsure which location to use, choose the descriptive path shown in the table above.
