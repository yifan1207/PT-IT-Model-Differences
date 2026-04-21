"""
Exp8: Phase 0 — Multi-Model Causal Steering.

Extends the causal steering story from Gemma-only to all 6 model families.
Tests whether removing the corrective direction degrades governance/structural
formatting universally, not just for Gemma.

6 models: gemma3_4b, llama31_8b, qwen3_4b, mistral_7b, deepseek_v2_lite, olmo2_7b

Pipeline overview:
  1. Precompute corrective directions (IT-PT MLP activation differences)
  2. A1 alpha-sweep steering (14 alpha values + random controls)
  3. LLM judge evaluation (STR, FC, content quality)
  4. Piggybacked analyses: PCA rank-1, bootstrap stability, ID profiles, commitment

Scripts in this directory:
  plot_phase0.py          Cross-model comparison figures (dose-response, etc.)
  pca_rank1.py            1A: PCA of IT-PT direction — is the direction rank-1?
  direction_bootstrap.py  1A: Bootstrap stability of corrective directions
  direction_stability.py  0A analog: direction stability from worker subsets
  id_under_steering.py    1C: TwoNN intrinsic dimensionality at alpha=1,0,-1
  commitment_vs_alpha.py  1B: Commitment delay vs alpha (logit-lens post-processing)

Shared infrastructure (not duplicated here):
  scripts/precompute/precompute_directions_multimodel.py   Direction extraction (4-phase pipeline)
  scripts/run/run_phase0_multimodel.sh                     Local GPU orchestration
  scripts/infra/modal_phase0.py                            Modal cloud orchestration
  src/poc/exp06_corrective_direction_steering/run.py                           Core steering runner (--model-family flag)
  src/poc/exp06_corrective_direction_steering/model_adapter.py                 SteeringAdapter for multi-model hooks
  src/poc/cross_model/config.py                 MODEL_REGISTRY (6 models)
  src/poc/cross_model/adapters/                 Per-model architecture adapters

Results: results/exp08_multimodel_steering_phase0/plots/ (PNG)
         and results/exp08_multimodel_steering_phase0/plots/data/ (JSON/JSONL)

Quick start:
  # Full run on Modal:
  modal run --detach scripts/infra/modal_phase0.py

  # Or local GPUs:
  bash scripts/run/run_phase0_multimodel.sh --step precompute
  bash scripts/run/run_phase0_multimodel.sh --step steer
  bash scripts/run/run_phase0_multimodel.sh --step judge
  bash scripts/run/run_phase0_multimodel.sh --step pca
  bash scripts/run/run_phase0_multimodel.sh --step id-steering
  bash scripts/run/run_phase0_multimodel.sh --step commitment

  # Generate plots:
  uv run python -m src.poc.exp08_multimodel_steering_phase0.plot_phase0
"""
