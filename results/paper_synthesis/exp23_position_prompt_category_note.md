# Exp23 Position x Prompt-Category Analysis

CPU-only stratification of the primary Exp23 residual-state x late-stack factorial. The metric is the common-IT upstream-state x late-stack interaction unless noted.

## Interaction by prompt category and position

| Position | Prompt category | Clusters | Models | Interaction | 95% CI |
|---|---|---:|---:|---:|---:|
| all positions | ALL | 2983 | 5 | 2.64 | [2.54, 2.73] |
| all positions | GOV-CONV | 1494 | 5 | 2.05 | [1.95, 2.16] |
| all positions | GOV-FORMAT | 745 | 5 | 3.61 | [3.36, 3.85] |
| all positions | SAFETY | 744 | 5 | 2.83 | [2.66, 3.01] |
| position 0 | ALL | 1499 | 5 | 3.01 | [2.88, 3.16] |
| position 0 | GOV-CONV | 556 | 5 | 2.70 | [2.53, 2.88] |
| position 0 | GOV-FORMAT | 421 | 5 | 2.30 | [2.02, 2.62] |
| position 0 | SAFETY | 522 | 5 | 3.24 | [3.04, 3.45] |
| positions >=1 | ALL | 1484 | 5 | 2.25 | [2.11, 2.39] |
| positions >=1 | GOV-CONV | 938 | 5 | 1.64 | [1.50, 1.77] |
| positions >=1 | GOV-FORMAT | 324 | 5 | 3.59 | [3.25, 3.97] |
| positions >=1 | SAFETY | 222 | 5 | 2.64 | [2.15, 3.18] |
| positions >=3 | ALL | 800 | 5 | 1.52 | [1.36, 1.68] |
| positions >=3 | GOV-CONV | 700 | 5 | 1.51 | [1.35, 1.68] |
| positions >=3 | GOV-FORMAT | 61 | 5 | 2.28 | [1.76, 2.85] |
| positions >=3 | SAFETY | 39 | 5 | 0.64 | [0.28, 0.95] |
| positions >=5 | ALL | 495 | 5 | 1.64 | [1.40, 1.90] |
| positions >=5 | GOV-CONV | 427 | 5 | 1.63 | [1.35, 1.91] |
| positions >=5 | GOV-FORMAT | 47 | 5 | 2.26 | [1.70, 2.82] |
| positions >=5 | SAFETY (thin) | 21 | 4 | 0.13 | NA |

## Main read

At generated position `>=3`, the pooled interaction is `1.52` logits. Most records are GOV-CONV, whose within-category interaction is `1.51` logits. GOV-FORMAT (`2.28`) and SAFETY (`0.64`) remain positive but are thin, so they are useful as direction checks rather than category-level headline estimates.
The position attenuation is therefore partly a composition shift toward later conversational/governance disagreements, but not only composition: GOV-CONV itself drops from the all-position estimate to the `>=3` estimate.
