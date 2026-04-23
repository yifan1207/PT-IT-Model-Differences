# Exp17: Behavioral-Direction Replication

`exp17` is the canonical package for external-paper replication work that we
want to connect back to the repo's main convergence-gap story without collapsing
the replications into a bespoke hybrid.

The package is intentionally split into three layers:

1. `Lu replication`
   Runs or wraps the published Assistant Axis pipeline as faithfully as
   possible.

2. `Du replication`
   Reproduces truthfulness and refusal direction extraction in a way that fits
   the repo's six-model PT/IT setup.

3. `Joint analysis`
   Compares the replicated layer-localized behavioral directions against the
   existing convergence-gap summaries from `exp09` and related late-window
   causal results.

## Design principles

- Replication and synthesis stay separate.
- Shared local infrastructure is allowed only for plumbing:
  model registry, chat-template handling, architecture adapters, result layout,
  and activation collection.
- New methodological shortcuts should not be silently introduced.
- Any place where the current implementation is a scaffold rather than a full
  paper-faithful reproduction is called out explicitly in code and docs.
