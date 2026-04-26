# Exp20 Holdout + Early Add-on: Paper-Safe Interpretation

Run: `factorial_validation_holdout_fast_20260425_2009_with_early`

This add-on completes the reviewer-tight IT-host localization comparison:

- `C_it_chat`
- `D_early_ptswap`
- `D_mid_ptswap`
- `D_late_ptswap`
- `D_earlymid_ptswap`
- `D_midlate_ptswap`

The metric is the first-divergence IT-vs-PT token readout: at the prefix where pure PT and pure IT first choose different next tokens, evaluate each grafted pipeline by whether it chooses the IT token, PT token, or another token, and by the logit margin between the IT token and PT token. This is causal with respect to the layer-swap intervention and this proxy metric, but it is not by itself a direct behavioral or human-confidence measure.

## Primary Native IT-Host Result

Dense-5 pooled native results:

| Condition | IT-token fraction | PT-token fraction | Other fraction | IT-vs-PT margin |
|---|---:|---:|---:|---:|
| `C_it_chat` | 1.0000 | 0.0000 | 0.0000 | 24.5248 |
| `D_early_ptswap` | 0.6103 | 0.0333 | 0.3563 | 12.9913 |
| `D_mid_ptswap` | 0.5930 | 0.0160 | 0.3910 | 12.5162 |
| `D_late_ptswap` | 0.6767 | 0.0107 | 0.3127 | 11.2770 |
| `D_earlymid_ptswap` | 0.3477 | 0.0493 | 0.6030 | 8.5241 |
| `D_midlate_ptswap` | 0.5100 | 0.0233 | 0.4667 | 7.4268 |

Native margin drops relative to pure IT:

| Swap | Margin drop | 95% bootstrap CI |
|---|---:|---:|
| early | 11.5335 | [10.9897, 12.1129] |
| mid | 12.0087 | [11.3620, 12.6178] |
| late | 13.2479 | [12.5997, 13.9019] |
| early+mid | 16.0007 | [15.3789, 16.6819] |
| mid+late | 17.0980 | [16.3964, 17.7854] |

The pooled native pattern supports a readout-heavy role for late layers: replacing late IT layers with PT layers causes the largest single-window margin drop while preserving IT token identity better than early or mid swaps. That is the cleanest evidence for a late readout/reconciliation role.

## What This Does Not Prove

- It does not prove late layers uniquely cause IT behavior. Early and mid swaps also strongly reduce the IT-vs-PT margin.
- It does not prove full causal mediation from mid features to late readout. The metric is a first-divergence token proxy.
- It does not justify saying late layers are the birthplace of assistant behavior. Identity is distributed, and mid/early-mid effects remain important.
- It does not prove all families behave identically. Per-model single-window ranking varies.

## Best Main-Text Claim

Instruction tuning changes the trajectory by which models settle on final tokens. In native IT-host swaps, replacing any early, mid, or late IT window with its PT counterpart reduces the first-divergence IT-vs-PT margin. The late swap produces the largest pooled single-window margin loss while preserving more IT token identity than early or mid swaps, suggesting that late layers are comparatively readout-heavy rather than identity-originating. Multi-window swaps produce larger losses, so the mechanism is best described as a distributed mid-to-late policy-to-prediction handoff, not a late-only circuit.

## Reviewer-Facing Caveats

- Call this "causal evidence on a proxy metric," not a direct proof of behavioral confidence.
- Emphasize native IT-host results as primary because they keep the IT model in its trained chat-template distribution.
- Report raw-shared as an ablation; it is useful but less central because it changes the IT prompt distribution.
- Avoid universal late-only language because raw-shared results and per-model native results are heterogeneous.
- Keep convergence-gap localization separate from behavioral identity. Late MLPs can be strongest for delayed stabilization while behavioral token identity remains mid-to-late distributed.
