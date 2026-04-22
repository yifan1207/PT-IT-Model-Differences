# Exp15 Plot Notes

## What This Folder Contains

- `exp15_primary_bars.png`: Dense-5 pooled pointwise effects on the two main behavioral endpoints.
- `exp15_pairwise.png`: Dense-5 pooled pairwise preferences for late-vs-baseline comparisons.
- `exp15_per_model_deltas.png`: Model-by-model paired deltas with 95% bootstrap CIs across PT-side sufficiency and IT-side necessity.
- `exp15_internal_scatter.png`: Cross-condition link between matched-prefix internal deltas and free-running behavioral deltas.
- `exp15_g2_bucket_deltas.png`: Assistant-register deltas split by prompt subtype.
- `exp15_generation_diagnostics.png`: Assistant-facing output length and 512-token cap-rate diagnostics.
- `exp15_programmatic_deltas.png`: Programmatic cross-checks for structure/format/content.

## High-Level Read

- The cleanest behavioral result is on the IT-side necessity test: late PT-swaps hurt behavior most strongly on pooled `S2` and `G2`.
- PT-side late grafts are behaviorally real but not pointwise-maximal: `B_late_raw` beats `A_pt_raw` pairwise, yet `B_mid_raw` is strongest on pooled pointwise PT-side `S2` and `G2`.
- A likely reason is visible in `exp15_generation_diagnostics.png`: PT mid windows often shorten outputs and reduce cap saturation, while PT late windows frequently remain near the `512`-token cap.

## Plot-By-Plot Guide

### `exp15_primary_bars.png`

- Positive bars always mean movement in the expected causal direction.
- PT side: improvement relative to `A_pt_raw`.
- IT side: degradation relative to `C_it_chat`.
- Main message: late is strongest on the IT side, but mid is strongest on the PT-side pointwise bars.

### `exp15_pairwise.png`

- These plots compare only the late branch to its baseline under blind pairwise judging.
- PT side pooled late-vs-A preference:
  - `G2`: target preferred `49.2%` (95% CI `46.2%` to `52.0%`), other preferred `38.1%`, tie `12.7%`
  - `S2`: target preferred `51.7%` (95% CI `46.7%` to `56.8%`), other preferred `29.3%`, tie `18.9%`
- IT side pooled C-vs-Dlate preference:
  - `G2`: target preferred `75.3%` (95% CI `72.8%` to `77.9%`)
  - `S2`: target preferred `76.3%` (95% CI `72.0%` to `80.3%`)

### `exp15_g2_bucket_deltas.png`

- This explains where assistant-register effects come from.
- PT mid gains are concentrated especially on conversational-source prompts and extra conversational governance prompts.
- IT late losses are broader and remain strong on conversational and register-focused prompts.

### `exp15_generation_diagnostics.png`

- This is the main cautionary diagnostic for interpreting PT-side free-running sufficiency.
- Under capped `512`-token generation, a branch can look more assistant-like partly because it reaches a shorter, cleaner stopping point.
- The PT mid branch often reduces mean length and cap rate more than the PT late branch.

### `exp15_programmatic_deltas.png`

- These are non-judge cross-checks.
- They help separate structure/format changes from broad content collapse.
- In this run, PT mid is also stronger than PT late on several format-like programmatic views, while IT late remains the strongest broad necessity window behaviorally.

### `exp15_internal_scatter.png`

- Each point is a model-window condition.
- Colors are depth windows; labels use model abbreviations plus `E/M/L`.
- The IT-side correlations are materially cleaner than the PT-side ones:
  - PT `S2`: `r=0.19`
  - PT `G2`: `r=-0.25`
  - IT `S2`: `r=0.58`
  - IT `G2`: `r=0.43`

## Acceptance Snapshot

- Dense models available: `5`
- PT-side `S2` late-strongest count: `1/5`
- IT-side `S2` late-strongest count: `4/5`
- PT-side any-primary late-strongest count: `1/5`
- IT-side any-primary late-strongest count: `4/5`

## Recommended Paper Framing

- Use `exp15` primarily as a late-centered behavioral **necessity** result.
- Treat the PT-side behavioral sufficiency result as **real but partial**, with clear late pairwise gains but stronger mid pointwise gains under capped free-running decoding.
- Keep `exp11/13/14` as the main localization backbone, and use `exp15` to show that the same circuit matters under natural decoding rather than to claim a perfectly symmetric late-only behavioral story.
