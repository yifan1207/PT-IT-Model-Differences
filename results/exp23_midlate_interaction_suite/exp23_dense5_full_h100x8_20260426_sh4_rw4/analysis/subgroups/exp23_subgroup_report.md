# Exp23 Subgroup Analysis: Context-Gated Late Readout

This analysis stratifies the Exp23 residual-state x late-stack factorial without changing the estimand. The primary estimand is the `interaction` effect under the `common_it` readout.

## Data Profile

- Primary-readout units: `2983`
- `by_model`: `{"gemma3_4b": 600, "llama31_8b": 600, "mistral_7b": 597, "olmo2_7b": 586, "qwen3_4b": 600}`
- `by_prompt_category`: `{"GOV-CONV": 1494, "GOV-FORMAT": 745, "SAFETY": 744}`
- `by_event_kind`: `{"first_diff": 2983}`
- `by_it_token_category_collapsed`: `{"CONTENT": 1265, "FORMAT": 685, "FUNCTION_OTHER": 1033}`
- `assistant_marker_events`: `{"assistant_marker": 551, "non_assistant_marker": 2432}`

## Reportable Interaction Strata


### prompt_category

| value | n | models | interaction | 95% CI |
|---|---:|---:|---:|---:|
| GOV-CONV | 1494 | 5 | 2.05 | [1.95, 2.15] |
| GOV-FORMAT | 745 | 5 | 3.61 | [3.37, 3.87] |
| SAFETY | 744 | 5 | 2.83 | [2.67, 3.01] |

### it_token_category_collapsed

| value | n | models | interaction | 95% CI |
|---|---:|---:|---:|---:|
| CONTENT | 1265 | 5 | 2.50 | [2.35, 2.65] |
| FORMAT | 685 | 5 | 2.60 | [2.39, 2.82] |
| FUNCTION_OTHER | 1033 | 5 | 2.81 | [2.66, 2.97] |

### event_kind

| value | n | models | interaction | 95% CI |
|---|---:|---:|---:|---:|
| first_diff | 2983 | 5 | 2.64 | [2.54, 2.74] |

### it_margin_tercile_within_model

| value | n | models | interaction | 95% CI |
|---|---:|---:|---:|---:|
| high | 983 | 5 | 4.93 | [4.75, 5.11] |
| low | 1052 | 5 | 1.03 | [0.95, 1.11] |
| mid | 948 | 5 | 2.02 | [1.93, 2.12] |

### it_rank_bin

| value | n | models | interaction | 95% CI |
|---|---:|---:|---:|---:|
| rank_1 | 2920 | 5 | 2.66 | [2.56, 2.76] |
| rank_2_5 | 62 | 5 | 1.71 | [1.06, 2.40] |

### assistant_marker_event

| value | n | models | interaction | 95% CI |
|---|---:|---:|---:|---:|
| assistant_marker | 551 | 5 | 3.30 | [3.08, 3.54] |
| non_assistant_marker | 2432 | 5 | 2.48 | [2.37, 2.59] |

### divergence_step_bin

| value | n | models | interaction | 95% CI |
|---|---:|---:|---:|---:|
| step_0 | 1499 | 5 | 3.01 | [2.88, 3.15] |
| step_1 | 361 | 5 | 2.56 | [2.32, 2.82] |
| step_2_3 | 482 | 5 | 1.97 | [1.76, 2.19] |
| step_4_plus | 641 | 5 | 1.52 | [1.32, 1.72] |

## Interpretation Guardrails

- These are descriptive subgroup checks, not new headline claims.
- Bins are model-balanced where possible, but small strata are marked non-reportable in the CSV.
- Confidence bins use within-model terciles of the native IT baseline IT-vs-PT margin, not calibrated probabilities.
- The Exp23 holdout slice contains `GOV-CONV`, `GOV-FORMAT`, and `SAFETY`; factual/creative prompt strata require a new Exp23 run on a broader dataset slice.
