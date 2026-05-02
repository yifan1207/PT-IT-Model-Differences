# Exp34 Dense-5 Final-Readout Crosscoder Synthesis

Models with paper-gate pass: `2/4`.

| Model | Quality | Paper | Top-200 drop | Top-200 frac | Matched-random drop | Final-layer mass |
|---|---|---|---:|---:|---:|---:|
| llama31_8b | pass | pass | 0.599 | 0.199 | -0.209 | 0.552 |
| qwen3_4b | fail | fail | 0.268 | 0.126 | -0.014 | 0.478 |
| mistral_7b | pass | pass | 0.684 | 0.336 | -0.100 | 0.690 |
| olmo2_7b | fail | fail | 0.497 | 0.554 | -0.058 | 0.486 |

## Interpretation Guardrail

Treat Exp34 as feature-level mediation evidence only when causal features beat same-layer matched random controls and dictionary health gates pass. The primary estimand is interaction drop in logits; mediation fraction is descriptive.
