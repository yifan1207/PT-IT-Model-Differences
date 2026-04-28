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

| value | records | prompt clusters | models | interaction | 95% CI |
|---|---:|---:|---:|---:|---:|
| GOV-CONV | 1494 | 1494 | 5 | 2.05 | [1.95, 2.15] |
| GOV-FORMAT | 745 | 745 | 5 | 3.61 | [3.36, 3.88] |
| SAFETY | 744 | 744 | 5 | 2.83 | [2.64, 3.02] |

### it_token_category_collapsed

| value | records | prompt clusters | models | interaction | 95% CI |
|---|---:|---:|---:|---:|---:|
| CONTENT | 1265 | 1265 | 5 | 2.50 | [2.34, 2.65] |
| FORMAT | 685 | 685 | 5 | 2.60 | [2.39, 2.80] |
| FUNCTION_OTHER | 1033 | 1033 | 5 | 2.81 | [2.67, 2.97] |

### event_kind

| value | records | prompt clusters | models | interaction | 95% CI |
|---|---:|---:|---:|---:|---:|
| first_diff | 2983 | 2983 | 5 | 2.64 | [2.54, 2.73] |

### it_margin_tercile_within_model

| value | records | prompt clusters | models | interaction | 95% CI |
|---|---:|---:|---:|---:|---:|
| high | 983 | 983 | 5 | 4.93 | [4.75, 5.12] |
| low | 1052 | 1052 | 5 | 1.03 | [0.95, 1.11] |
| mid | 948 | 948 | 5 | 2.02 | [1.93, 2.12] |

### it_rank_bin

| value | records | prompt clusters | models | interaction | 95% CI |
|---|---:|---:|---:|---:|---:|
| rank_1 | 2920 | 2920 | 5 | 2.66 | [2.56, 2.76] |
| rank_2_5 | 62 | 62 | 5 | 1.71 | [1.06, 2.42] |

### assistant_marker_event

| value | records | prompt clusters | models | interaction | 95% CI |
|---|---:|---:|---:|---:|---:|
| assistant_marker | 551 | 551 | 5 | 3.30 | [3.07, 3.53] |
| non_assistant_marker | 2432 | 2432 | 5 | 2.48 | [2.38, 2.59] |

### divergence_step_bin

| value | records | prompt clusters | models | interaction | 95% CI |
|---|---:|---:|---:|---:|---:|
| step_0 | 1499 | 1499 | 5 | 3.01 | [2.87, 3.15] |
| step_1 | 361 | 361 | 5 | 2.56 | [2.31, 2.82] |
| step_2_3 | 482 | 482 | 5 | 1.97 | [1.74, 2.19] |
| step_4_plus | 641 | 641 | 5 | 1.52 | [1.33, 1.72] |

## Interpretation Guardrails

- These are descriptive subgroup checks, not new headline claims.
- Bins are model-balanced where possible, but small strata are marked non-reportable in the CSV.
- Confidence intervals resample prompt clusters within each model family, then average family estimates.
- Confidence bins use within-model terciles of the native IT baseline IT-vs-PT margin, not calibrated probabilities.
- Prompt-category coverage is exactly the categories observed in this run: `GOV-CONV, GOV-FORMAT, SAFETY`.
