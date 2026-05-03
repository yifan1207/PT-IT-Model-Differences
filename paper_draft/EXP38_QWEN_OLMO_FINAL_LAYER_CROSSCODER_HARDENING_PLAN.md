# Exp38 Plan: Qwen/OLMo Final-Layer Crosscoder Hardening

## Summary

Exp38 is a targeted follow-up to Exp34 for the two Dense-5 families whose final-three crosscoder results showed real causal signal but did not pass strict paper-quality gates:

- Qwen 3 4B: causal top-200 drop `+0.345`, matched random `-0.041`, but layer-33 `VE_IT=0.745` and `alive_fraction_max=0.305`.
- OLMo 2 7B: causal top-200 drop `+0.422`, matched random `-0.058`, but layer-30 `VE_IT=0.704`, `alive_fraction_max=0.208`, and unstable per-event mediation fractions.

The goal is not to rescue every family at all costs. The goal is to test a sharper hypothesis:

> Qwen and OLMo have terminal-readout features, but the current final-three dictionaries are too diffuse for clean feature-level mediation. Narrower final-layer or final-two-layer crosscoders may give cleaner sparse causal features.

If Exp38 succeeds, Qwen/OLMo can move from diagnostic appendix evidence into the cross-family feature-mediation result. If it fails, the paper should keep Gemma/Llama/Mistral as the clean crosscoder families and explicitly treat Qwen/OLMo as diffuse-feature cases.

## Current Evidence

### Exp31 terminal-depth factorial

Exp31 shows that the terminal blocks matter for both Qwen and OLMo, but final-one alone does not capture all of the terminal effect.

| Model | Boundary | Interaction | Retention vs full late |
|---|---:|---:|---:|
| Qwen 3 4B | final 1 | `+0.426` | `29%` |
| Qwen 3 4B | final 3 | `+1.022` | `70%` |
| OLMo 2 7B | final 1 | `+0.371` | `20%` |
| OLMo 2 7B | final 3 | `+0.850` | `46%` |

This supports a final-readout target, but not a final-one-only claim. The missing measurement is final-two.

### Residual-opposition geometry

The old residual-change curves do not support a broad "more late layers" first move:

- Qwen has a strong final-layer residual-opposing spike (`L35` IT delta-cosine about `-0.718`) but the preceding terminal layers are positive/non-opposing.
- OLMo has a weak final-layer residual-opposing turn (`L31` about `-0.080`) and mostly positive/non-opposing preceding terminal layers.
- In Llama, all-late crosscoder selection over layers 19-31 still concentrated causal score mass in layers 30-31, so broadening the layer set did not reveal a hidden earlier-late circuit.

So Exp38 should go narrower and cleaner before going wider.

## Main Questions

1. Can final-layer-only crosscoders pass strict quality gates for Qwen and OLMo?
2. Does final-two recover substantially more causal mediation than final-one while staying sparse?
3. Is the Exp34 final-three weakness caused by dictionary quality/diffuseness rather than absence of terminal causal features?
4. Are selected features robust under held-out mediation and matched-random controls?

## Targets

Use raw-shared first-divergence prompts and the same Exp30/Exp34 causal selector.

| Model | Layers | Final one | Final two | Final three reference |
|---|---:|---:|---:|---:|
| Qwen 3 4B | 36 | `35` | `34 35` | `33 34 35` |
| OLMo 2 7B | 32 | `31` | `30 31` | `29 30 31` |

Do not run Qwen2.5-32B in Exp38. This is a small-family hardening pass.

## Experimental Design

### Data

Training data:

```text
data/eval_dataset_v2.jsonl
data/exp3_dataset.jsonl
data/exp6_dataset.jsonl
```

Exclude paper holdout:

```text
data/eval_dataset_v2_holdout_0600_1199.jsonl
```

Evaluation data:

```text
data/eval_dataset_v2_holdout_0600_1199.jsonl
```

First-divergence events:

```text
results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early
```

Prompt mode:

```text
raw_shared
```

### Crosscoder object

For each target layer `l`, train BatchTopK crosscoders on paired PT/IT MLP outputs:

```text
[m_PT_l(x), m_IT_l(x)] -> z -> [recon_PT_l(x), recon_IT_l(x)]
```

Layer-local dictionaries are preferred. For final-two, train one crosscoder per layer and select features jointly across the two layer-local dictionaries. Do not concatenate layers into one dictionary unless layer-local training fails, because layer-local results are easier to quality-gate and interpret.

### Causal mediation measurement

Use the same terminal-readout margin as Exp30/34:

```text
Y = logit(t_IT) - logit(t_PT)
```

Rank features on a calibration split by their causal contribution to the terminal upstream-state x late-stack interaction. Mediate on held-out prompts only.

Primary effect:

```text
interaction_drop = interaction_full - interaction_ablate_selected_features
mediation_fraction = interaction_drop / interaction_full
```

Report `interaction_drop` as primary. Treat `mediation_fraction` as descriptive because it is denominator-sensitive.

Important scope: when the crosscoder is trained on final-one, ablate only features in final-one and compare against the final-one factorial interaction. When trained on final-two, ablate final-two features and compare against a final-two factorial reference. Do not claim final-one features mediate final-three effects unless final-three is also explicitly recomputed under that intervention.

Implementation requirement: the causal-rank and mediation commands must pass an explicit factorial boundary override matching the target window. Merely setting `CAUSAL_LAYERS` is not enough, because the default Exp30/34 mediation boundary is the full late-stack boundary.

| Scope | Qwen boundary override | OLMo boundary override |
|---|---:|---:|
| final-one | `35` | `31` |
| final-two | `34` | `30` |
| final-three reference | `33` | `29` |

Each output table must record `boundary_layer`, `downstream_stack`, and the selected crosscoder layers so the paper claim can be audited mechanically.

## Configuration Ladder

Run as a ladder, stopping early if a clean sparse pass is found.

### Stage A: Final-one pilot grid

For each model, final layer only:

Qwen `d_model=2560`:

| Config | Reason |
|---|---|
| `d81920_k64` | Exp34-size baseline but narrower layer set |
| `d131072_k64` | more capacity without increasing density |
| `d196608_k64` | high-capacity sparse test |
| `d196608_k48` | stricter sparsity / lower alive-fraction pressure |

OLMo `d_model=4096`:

| Config | Reason |
|---|---|
| `d131072_k64` | Exp30 Llama-style baseline |
| `d196608_k64` | higher capacity |
| `d262144_k64` | high-capacity sparse test |
| `d262144_k48` | stricter sparsity / lower alive-fraction pressure |

Pilot data:

- `500k` to `1M` training tokens.
- `8k` to `12k` steps.
- rank `80` prompts, mediate `120` prompts.
- `k_eval = 25, 50, 100, 200`.
- matched-random seeds `0,1,2`.

Choose one final-one config per model by quality and held-out causal signal, not by training reconstruction alone.

### Stage B: Final-two selected run

If final-one is clean but under-mediates, train final-two using the selected final-one config family:

- Qwen: `34 35`.
- OLMo: `30 31`.

Use full Exp30-style run:

- `2M` training tokens.
- `24k` steps.
- rank `160` prompts.
- mediate remaining `440` prompts.
- `k_eval = 25, 50, 100, 200, 500`.
- matched-random seeds `0,1,2`.

### Stage C: Optional final-three rerun

Only rerun final-three if final-one/final-two pass quality and suggest the Exp34 failure was mostly configuration-related. Otherwise do not spend more compute on broader dictionaries.

## Quality Gates

A result is paper-clean only if all of the following hold:

1. **Reconstruction health:** `VE_IT >= 0.78` for every included layer, or `>= 0.75` only if the causal effect and sparsity are exceptionally clean.
2. **Sparsity health:** `alive_fraction_max <= 0.15`; prefer `<= 0.10`.
3. **Causal direction:** selected-feature ablation reduces the terminal interaction by a positive amount on held-out prompts.
4. **Random control:** matched-random same-count features do not reproduce the drop; their mean should be near zero or negative. Controls should be matched within the same layer and by calibration activity/attribution mass, not only by feature count.
5. **Mass concentration:** top causal mass should not require hundreds of features. A rough gate: features needed for 50% of positive causal-score mass should be `<100` for a strong interpretability claim.
6. **Monotonicity:** `k=25,50,100,200` should show a broadly sensible curve. A single isolated top-200 win is not enough.
7. **No selection leakage:** rank prompts and mediation prompts must be disjoint.
8. **Primary estimand:** report interaction drop as the primary number. Report mediation fraction as secondary/descriptive, especially for low or sign-unstable denominator slices.

## Red-Team Concerns and Answers

| Concern | Design answer |
|---|---|
| "Final-one crosscoders are cherry-picking the terminal layer." | Report Exp31 final-one/final-three references and run final-two as the main hardening test. Final-one is a diagnostic, not the only target. |
| "You train on one layer but claim to explain the late stack." | Scope each claim to the factorial boundary being mediated: final-one mediates final-one, final-two mediates final-two. Compare to final-three only as retention context. |
| "Lower `k` just hides features and improves the alive metric artificially." | Require held-out causal drop and matched-random controls in addition to alive fraction. Report VE/sparsity/causal drop jointly. |
| "Top-200 is arbitrary." | Report the full `k` curve (`25,50,100,200,500`) plus causal mass coverage. Use top-200 only as a conventional comparison point. |
| "The features are not interpretable, only causal." | Do not claim interpretability until top activation examples, decoder/logit effects, and feature labels are analyzed. This pass is feature-mediation hardening. |
| "Qwen/OLMo are lower quality because the effect is diffuse." | If final-one/final-two still require many features or fail gates, that is the conclusion: terminal mediation exists but is diffuse in these families. |
| "The selector overfits the causal split." | Rank on one prompt split, mediate on disjoint prompts, and optionally swap rank/mediate splits for the final selected config. |
| "The implementation trained final-one features but measured the full late stack." | Every Exp38 command passes `--boundary-layer-override` and every record stores the boundary and downstream stack. |
| "Matched random was too weak." | Random controls are same-layer and matched by calibration active rate / attribution magnitude; report the whole random-seed distribution. |

## Failure Interpretation

Failure is informative and should not be hidden. If Qwen or OLMo still fail VE/alive gates, require hundreds of features for moderate drops, or show unstable fractions despite positive interaction drops, the conclusion is:

> terminal feature mediation exists in this family, but the effect is diffuse or dictionary-quality-limited under current sparse crosscoder training.

In that case, keep Gemma/Llama/Mistral as the clean feature-mediation result and use Qwen/OLMo as appendix diagnostics rather than forcing them into the headline.

## Output Artifacts

```text
results/exp38_qwen_olmo_final_layer_crosscoder_hardening/<run_name>/
  qwen3_4b/
    final1_<config>/
    final2_<config>/
  olmo2_7b/
    final1_<config>/
    final2_<config>/
  analysis/
    exp38_summary.json
    exp38_quality_table.csv
    exp38_mediation_curve.csv
    exp38_mediation_curve.png
    exp38_feature_mass_table.csv
    exp38_paper_note.md
```

Sync raw checkpoints/caches to GCS. Keep only compact analysis artifacts locally.

## Expected Runtime and Cost

Relative to Exp34, final-one is much cheaper because it caches/trains one layer instead of three.

Recommended execution:

1. Run Qwen final-one pilot and OLMo final-one pilot in parallel on one 8x A100/H100 pod.
2. Select one config per model.
3. Run final-two selected runs only for models whose final-one result is promising.

Rough estimate:

- Final-one pilots for both models: `2-5` wall-clock hours on 8x A100/H100 depending on cache reuse and token count.
- Final-two selected runs for both models: `3-7` additional hours.
- Storage: expect tens of GB per model if raw dictionaries/caches are retained; sync to GCS and keep compact local artifacts.

## Paper Integration Decision

Use Exp38 in the main paper only if:

- at least one of Qwen/OLMo becomes strict-pass, and
- the result improves the cross-family crosscoder story beyond the existing Gemma/Llama/Mistral evidence.

Otherwise:

- keep Gemma/Llama/Mistral as the clean feature-mediation result;
- mention Qwen/OLMo only in appendix as diffuse/quality-limited diagnostics;
- do not weaken the main paper with marginal feature-atlas evidence.
