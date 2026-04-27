# Exp23 Mid x Late KL Factorial

Run root: `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/part_a_mlp_kl`
Dense models present: gemma3_4b, qwen3_4b, llama31_8b, mistral_7b, olmo2_7b

## Dense-5 Effects

| Effect | Mean | 95% CI |
|---|---:|---:|
| E_mid_pt | -0.0310 | [-0.0365, -0.0258] |
| E_late_pt | 0.2175 | [0.2088, 0.2261] |
| E_midlate_pt | 0.1978 | [0.1875, 0.2082] |
| I_pt | 0.0114 | [0.0087, 0.0139] |
| L_given_M_pt | 0.2288 | [0.2209, 0.2374] |
| L_alone_pt | 0.2175 | [0.2090, 0.2268] |
| E_mid_it | -0.4863 | [-0.4992, -0.4727] |
| E_late_it | -0.8221 | [-0.8400, -0.8046] |
| E_midlate_it | -0.9865 | [-1.0086, -0.9648] |
| I_it | 0.3218 | [0.3127, 0.3314] |
| C_mid_it | 0.4863 | [0.4729, 0.4994] |
| C_late_it | 0.8221 | [0.8058, 0.8390] |
| C_midlate_it | 0.9865 | [0.9644, 1.0083] |
| I_collapse_it | -0.3218 | [-0.3319, -0.3119] |

## Validation

| Model | Missing conditions | PT common prompts | IT common prompts | Prompt-mode mismatches |
|---|---|---:|---:|---|
| gemma3_4b | none | 600 | 600 | none |
| qwen3_4b | none | 600 | 600 | none |
| llama31_8b | none | 600 | 600 | none |
| mistral_7b | none | 600 | 600 | none |
| olmo2_7b | none | 600 | 600 | none |
