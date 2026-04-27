# Exp23 Compatibility-Amplification Label Control

Run root: `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4`
Prompt mode: `raw_shared`
Readout: `common_it`

The compatibility amplification is algebraically the Exp23 interaction, but is read as an own-token compatibility test: IT late stack gains more from IT upstream than PT late stack gains from PT upstream.

| Model | N | IT compatibility boost | PT compatibility boost | IT-over-PT amplification |
|---|---:|---:|---:|---:|
| `gemma3_4b` | `600` | `+9.581` | `+3.503` | `+6.078` |
| `qwen3_4b` | `600` | `+3.349` | `+1.884` | `+1.464` |
| `llama31_8b` | `600` | `+4.564` | `+3.311` | `+1.253` |
| `mistral_7b` | `597` | `+5.824` | `+3.290` | `+2.534` |
| `olmo2_7b` | `586` | `+4.465` | `+2.618` | `+1.847` |
| **Dense mean** | `2983` | `+5.556` | `+2.921` | `+2.635` |

## Label-Swap Null

The null randomly swaps PT/IT label orientation within each model/prompt event. This preserves each event's four cell values and flips the sign of its compatibility amplification.

- Observed amplification: `+2.635` logits
- Null mean: `+0.0003` logits
- Null std: `0.0769` logits
- Null 99.9th percentile: `+0.239` logits
- One-sided permutation p-value: `4.99975e-05`
- Two-sided permutation p-value: `4.99975e-05`
