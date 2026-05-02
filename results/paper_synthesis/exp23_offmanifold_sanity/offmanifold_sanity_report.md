# Exp23 Off-Manifold Sanity Diagnostic

CPU-only diagnostic over the Dense-5 raw-record Exp23 first-divergence residual-factorial records.

## Summary

- Valid first-divergence events: `2983`; invalid events: `17`.
- Noop diagonal patch checks: `5966`; max patch-input delta `0.000`; max common-IT margin delta `0.000`; all common-IT top-1 equal: `True`.
- Mean common-IT interaction across Dense-5 models: `2.635` logits; mean common-PT interaction: `2.610` logits.
- Family-balanced bootstrap: common-IT interaction `2.635` [`2.537`, `2.739`] logits; common-PT interaction `2.610` [`2.515`, `2.708`] logits; common-IT minus common-PT `0.025` [`0.006`, `0.043`] logits.
- Across common-IT trajectory metrics, the worst off-diagonal mean is at most `1.074`x the worst diagonal mean; the largest absolute excess is `0.179`.
- Pooled common-IT top-1 choices for `U_PT__L_IT`: `0.526` PT, `0.226` IT, `0.249` other.
- Pooled common-IT top-1 choices for `U_IT__L_PT`: `0.241` PT, `0.447` IT, `0.313` other.

## Interpretation

These checks do not prove that off-diagonal hybrids are natural model states. They do show that the recorded hybrids are not numerical patching failures or degenerate collapsed distributions under the stored readouts: diagonal cells reconstruct exactly, common-IT and common-PT readouts agree on the interaction, off-diagonal trajectory metrics remain in the diagonal range or only slightly above it, and hybrid top-1 predictions move in graded ways rather than becoming arbitrary.

The remaining concern is semantic, not numerical: off-diagonal cells are still constructed counterfactuals. The main paper should therefore interpret the upstream x late interaction as a compatibility/readout estimand, not as full circuit recovery.

## Worst Off-Diagonal Trajectory Ratios

| Model | Metric | Common-IT offdiag / diag worst | Offdiag - diag worst |
|---|---|---:|---:|
| gemma3_4b | late_kl_mean | `0.864` | `-0.177` |
| gemma3_4b | remaining_adj_js | `0.928` | `-0.023` |
| gemma3_4b | future_top1_flips | `0.890` | `-0.139` |
| gemma3_4b | top5_churn | `0.932` | `-0.023` |
| llama31_8b | late_kl_mean | `0.983` | `-0.065` |
| llama31_8b | remaining_adj_js | `1.073` | `0.040` |
| llama31_8b | future_top1_flips | `0.907` | `-0.162` |
| llama31_8b | top5_churn | `1.033` | `0.015` |
| mistral_7b | late_kl_mean | `0.875` | `-0.405` |
| mistral_7b | remaining_adj_js | `0.910` | `-0.030` |
| mistral_7b | future_top1_flips | `0.963` | `-0.055` |
| mistral_7b | top5_churn | `0.975` | `-0.010` |
| olmo2_7b | late_kl_mean | `1.074` | `0.179` |
| olmo2_7b | remaining_adj_js | `1.045` | `0.017` |
| olmo2_7b | future_top1_flips | `1.071` | `0.100` |
| olmo2_7b | top5_churn | `1.061` | `0.026` |
| qwen3_4b | late_kl_mean | `0.925` | `-0.217` |
| qwen3_4b | remaining_adj_js | `0.942` | `-0.032` |
| qwen3_4b | future_top1_flips | `0.946` | `-0.092` |
| qwen3_4b | top5_churn | `0.967` | `-0.014` |

