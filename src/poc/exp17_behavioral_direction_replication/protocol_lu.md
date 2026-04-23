# Lu 2026 Protocol Notes

Primary source:
- Christina Lu et al., "The Assistant Axis: Situating and Stabilizing the
  Default Persona of Language Models", arXiv:2601.10387.
- Public code: `safety-research/assistant-axis`.
- Pinned upstream commit for repo replication: `a98961956072224eaf244eb289d6c01700b63795`.

Repo-facing stance:

- We treat Lu as a replication target first and a synthesis input second.
- The preferred path is to run the upstream pipeline rather than silently
  reimplementing it inside this repo.
- Local code in `exp17` only wraps the upstream pipeline, records launch
  metadata, and maps outputs into the canonical repo result layout.
- Canonical bootstrap command:
  `bash scripts/infra/setup_exp17_upstreams.sh`

What should not be folded into the replication:

- replacing Lu's role set with local prompt pools
- merging Lu and Du prompts into one synthetic dataset
- reporting joint-analysis claims before first reproducing Lu-style outputs on
  our own model set
