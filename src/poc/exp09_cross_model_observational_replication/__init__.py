"""
Exp9: Cross-Model Replication — Observational Metrics.

Replicates 5 observational analyses across all 6 model families to test
whether the corrective computational stage is universal or Gemma-specific.

6 models: gemma3_4b, llama31_8b, qwen3_4b, mistral_7b, deepseek_v2_lite, olmo2_7b

Analyses:
  L1/L2  delta-cosine profiles + commitment delay (raw logit lens)
  L3     Weight change localization (MLP vs Attention, which layers changed most)
  L8     Intrinsic dimensionality profiles (TwoNN, IT vs PT)
  L9     Attention entropy divergence (IT - PT)
  0G     Tuned-lens commitment delay (trained affine probes, Belrose et al. 2023)

Scripts in this directory:
  plot_replication.py   Main plots: delta-cosine, commitment, weight diff, ID, entropy
  plot_commitment.py    All 7 commitment methods x raw/tuned logit-lens (11+ plots)
  plot_fixes.py         Improved plots with clear lens labels and validation

Data collection (shared infrastructure, not duplicated here):
  src/poc/cross_model/collect_L1L2.py   delta-cosine + raw commitment
  src/poc/cross_model/collect_L3.py     Weight change localization
  src/poc/cross_model/collect_L8.py     Intrinsic dimensionality (TwoNN)
  src/poc/cross_model/collect_L9.py     Attention entropy divergence
  src/poc/cross_model/tuned_lens.py     Tuned-lens probe training + eval (0G)
  src/poc/cross_model/config.py         MODEL_REGISTRY (6 models)
  src/poc/cross_model/adapters/         Per-model architecture adapters
  src/poc/cross_model/utils.py          Model loading, dataset loading

Modal scripts for tuned-lens (0G):
  scripts/infra/modal_tuned_lens_train.py      Probe training (12 jobs: 6 models x 2 variants)
  scripts/infra/modal_tuned_lens_eval.py       Commitment evaluation

Results:
  results/cross_model/{model}/{variant}/   Per-model raw data (JSONL, NPY)
  results/cross_model/{model}/tuned_lens/  Tuned-lens probes + commitment
  results/exp09_cross_model_observational_replication/plots/  PNG figures
  results/exp09_cross_model_observational_replication/data/   JSON exports for paper

Quick start:
  # Collect data (one model at a time, local GPU):
  uv run python -m src.poc.cross_model.collect_L1L2 --model gemma3_4b --device cuda:0
  uv run python -m src.poc.cross_model.collect_L8 --model gemma3_4b --device cuda:0
  # ... (see each collect_L*.py for CLI args)

  # Train tuned-lens probes (Modal):
  modal run scripts/infra/modal_tuned_lens_train.py

  # Generate plots:
  uv run python -m src.poc.exp09_cross_model_observational_replication.plot_replication
  uv run python -m src.poc.exp09_cross_model_observational_replication.plot_commitment
"""
