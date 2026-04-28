# Exp23 Position-Threshold Category Mix

CPU-only audit of the primary Exp23 residual-state x late-stack raw-shared records. Counts use valid `first_diff` events only; in this primary holdout, records equal prompt clusters.

## all

- Records: `2983` across `5` families (gemma3_4b 600 (20.1%), qwen3_4b 600 (20.1%), llama31_8b 600 (20.1%), mistral_7b 597 (20.0%), olmo2_7b 586 (19.6%)).
- Prompt categories: GOV-CONV 1494 (50.1%), GOV-FORMAT 745 (25.0%), SAFETY 744 (24.9%).
- IT divergent-token categories: CONTENT 1265 (42.4%), FUNCTION_OTHER 1033 (34.6%), FORMAT 685 (23.0%).
- PT divergent-token categories: FUNCTION_OTHER 1283 (43.0%), CONTENT 1151 (38.6%), FORMAT 549 (18.4%).

## step_0

- Records: `1499` across `5` families (llama31_8b 356 (23.7%), olmo2_7b 352 (23.5%), gemma3_4b 315 (21.0%), qwen3_4b 292 (19.5%), mistral_7b 184 (12.3%)).
- Prompt categories: GOV-CONV 556 (37.1%), SAFETY 522 (34.8%), GOV-FORMAT 421 (28.1%).
- IT divergent-token categories: FUNCTION_OTHER 654 (43.6%), CONTENT 423 (28.2%), FORMAT 422 (28.2%).
- PT divergent-token categories: FUNCTION_OTHER 637 (42.5%), CONTENT 506 (33.8%), FORMAT 356 (23.7%).

## step_ge1

- Records: `1484` across `5` families (mistral_7b 413 (27.8%), qwen3_4b 308 (20.8%), gemma3_4b 285 (19.2%), llama31_8b 244 (16.4%), olmo2_7b 234 (15.8%)).
- Prompt categories: GOV-CONV 938 (63.2%), GOV-FORMAT 324 (21.8%), SAFETY 222 (15.0%).
- IT divergent-token categories: CONTENT 842 (56.7%), FUNCTION_OTHER 379 (25.5%), FORMAT 263 (17.7%).
- PT divergent-token categories: FUNCTION_OTHER 646 (43.5%), CONTENT 645 (43.5%), FORMAT 193 (13.0%).

## step_ge3

- Records: `800` across `5` families (qwen3_4b 211 (26.4%), mistral_7b 187 (23.4%), llama31_8b 168 (21.0%), olmo2_7b 159 (19.9%), gemma3_4b 75 (9.4%)).
- Prompt categories: GOV-CONV 700 (87.5%), GOV-FORMAT 61 (7.6%), SAFETY 39 (4.9%).
- IT divergent-token categories: CONTENT 476 (59.5%), FUNCTION_OTHER 199 (24.9%), FORMAT 125 (15.6%).
- PT divergent-token categories: CONTENT 389 (48.6%), FUNCTION_OTHER 320 (40.0%), FORMAT 91 (11.4%).

## step_ge5

- Records: `495` across `5` families (qwen3_4b 141 (28.5%), mistral_7b 120 (24.2%), llama31_8b 104 (21.0%), olmo2_7b 94 (19.0%), gemma3_4b 36 (7.3%)).
- Prompt categories: GOV-CONV 427 (86.3%), GOV-FORMAT 47 (9.5%), SAFETY 21 (4.2%).
- IT divergent-token categories: CONTENT 307 (62.0%), FUNCTION_OTHER 106 (21.4%), FORMAT 82 (16.6%).
- PT divergent-token categories: CONTENT 255 (51.5%), FUNCTION_OTHER 163 (32.9%), FORMAT 77 (15.6%).

## step_ge10

- Records: `140` across `5` families (qwen3_4b 50 (35.7%), mistral_7b 35 (25.0%), llama31_8b 30 (21.4%), olmo2_7b 17 (12.1%), gemma3_4b 8 (5.7%)).
- Prompt categories: GOV-CONV 111 (79.3%), GOV-FORMAT 25 (17.9%), SAFETY 4 (2.9%).
- IT divergent-token categories: CONTENT 72 (51.4%), FORMAT 37 (26.4%), FUNCTION_OTHER 31 (22.1%).
- PT divergent-token categories: CONTENT 57 (40.7%), FORMAT 44 (31.4%), FUNCTION_OTHER 39 (27.9%).

## Paper-facing summary

At generated position `>=3`, the subset is not a single-category residue: `800` records remain across all five dense families; prompt categories are GOV-CONV 700 (87.5%), GOV-FORMAT 61 (7.6%), SAFETY 39 (4.9%); IT divergent-token categories are CONTENT 476 (59.5%), FUNCTION_OTHER 199 (24.9%), FORMAT 125 (15.6%).
