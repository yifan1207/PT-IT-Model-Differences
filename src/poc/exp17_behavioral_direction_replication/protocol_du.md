# Du 2025 Protocol Notes

Primary source:
- Hongzhe Du et al., "How Post-Training Reshapes LLMs: A Mechanistic View on
  Knowledge, Truthfulness, Refusal, and Confidence", arXiv:2504.02904.
- Public code repo: `HZD01/post-training-mechanistic-analysis`.
- Pinned upstream commit for repo replication: `4abef9d2d3d3bd7781dedfb09a48843d63c31f2c`.

Important update relative to older assumptions:

- As of April 22, 2026, there is a public repo with `Knowledge+Truthfulness/`
  and `Refusal/` directories. However, the released code is still oriented
  around the paper's original model setup and TransformerLens-centric plumbing,
  so cross-family replication on our six-model registry still requires local
  implementation work.
- Canonical bootstrap command:
  `bash scripts/infra/setup_exp17_upstreams.sh`
- Exact-source wiring currently used for local extensions:
  - truthfulness: upstream `Knowledge+Truthfulness/datasets/cities.csv`
  - refusal harmful prompts: upstream raw AdvBench CSV URL used by Du's repo
  - refusal harmless prompts: upstream `tatsu-lab/alpaca` dataset path used by
    Du's repo

Current package split:

1. Truthfulness
   - difference-in-means extraction from paired true/false statements
   - PT and IT processed separately

2. Refusal
   - harmful vs harmless contrastive direction extraction
   - current local implementation is a candidate extractor at the last prompt
     token, not yet the full paper-faithful selection sweep
