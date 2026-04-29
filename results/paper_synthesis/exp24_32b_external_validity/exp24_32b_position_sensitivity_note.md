# Exp24 Qwen2.5-32B Position Sensitivity

CPU-only audit of the Qwen2.5-32B Exp23 raw-shared residual-state x late-stack records. Intervals bootstrap prompt clusters within the single 32B family.

## Position distribution

- Valid records: `1397` prompt clusters.
- Generated-position mean: `4.66`; median: `2`.
- Position bins: step 0 `542`; step >=3 `628`; step >=5 `427`; step >=10 `212`.
- Prompt categories: CONTENT-FACT 300 (21.5%), GOV-CONV 300 (21.5%), GOV-FORMAT 249 (17.8%), CONTENT-REASON 199 (14.2%), SAFETY 149 (10.7%), GOV-REGISTER 100 (7.2%), BASELINE-EASY 100 (7.2%).
- IT divergent-token categories: CONTENT 645 (46.2%), FUNCTION_OTHER 530 (37.9%), FORMAT 222 (15.9%).
- Assistant-marker events: `214` (`15.3%`).

## Interaction by position

| Position filter | Clusters | Interaction | 95% CI |
|---|---:|---:|---:|
| all positions | 1397 | `1.45` | `[1.33, 1.57]` |
| position 0 | 542 | `1.79` | `[1.60, 2.00]` |
| positions >=1 | 855 | `1.23` | `[1.08, 1.38]` |
| positions >=3 | 628 | `1.02` | `[0.85, 1.21]` |
| positions >=5 | 427 | `0.69` | `[0.52, 0.88]` |
| positions >=10 | 212 | `0.89` | `[0.59, 1.22]` |

## Prompt categories at later positions

| Position filter | Category | Clusters | Interaction | 95% CI |
|---|---|---:|---:|---:|
| all positions | CONTENT-FACT | 300 | `2.19` | `[1.88, 2.52]` |
| all positions | CONTENT-REASON | 199 | `0.90` | `[0.65, 1.20]` |
| positions >=3 | CONTENT-FACT | 174 | `1.47` | `[1.05, 1.93]` |
| positions >=3 | CONTENT-REASON | 71 | `1.58` | `[0.90, 2.33]` |
| positions >=5 | CONTENT-FACT | 87 | `0.21` | `[0.15, 0.27]` |
| positions >=5 | CONTENT-REASON | 59 | `1.52` | `[0.79, 2.36]` |

## Paper-facing read

The Qwen2.5-32B interaction is `1.45` logits overall and remains positive at generated position `>=3` (`1.02`, 95% CI `[0.85, 1.21]`), `>=5` (`0.69`, 95% CI `[0.52, 0.88]`), and `>=10` (`0.89`, 95% CI `[0.59, 1.22]`). This run is content/reasoning-heavy rather than governance-heavy, so its position profile is not directly comparable to the Dense-5 holdout profile.
At position `>=3`, CONTENT-FACT has `1.47` logits and CONTENT-REASON has `1.58` logits; both are positive, showing that the later-position 32B effect is not a single prompt-category residue.
