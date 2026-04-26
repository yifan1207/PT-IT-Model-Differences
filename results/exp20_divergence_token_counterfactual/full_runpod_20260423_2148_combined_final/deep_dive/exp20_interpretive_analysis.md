# Exp20 Interpretive Analysis

This note summarizes the scientific readout from Exp20 after final merge and local plotting.

## Data Quality

The final merged run passes the core quality checks:

- `merge_summary.json` reports `ok=true`, `missing_conditions=[]`, and no malformed rows.
- Both `native` and `raw_shared` analyses report zero duplicate prompt ids, zero malformed rows, and all six models present.
- Dense-model pooled analyses use the five dense models; `deepseek_v2_lite` is available in all-model outputs but should be discussed separately because it is MoE.

## Core Result

Exp20 supports the refined mid-to-late handoff story:

1. Mid/policy swaps are more diagnostic of discrete token identity.
2. Late/reconciliation layers are more diagnostic of final logit-margin sharpening.
3. The convergence gap can remain late-layer dominated even if the behavioral candidate is selected or exposed earlier.

The key paired effects are large relative to prompt-bootstrap uncertainty:

- Raw-shared `PT + IT mid` transfers the IT divergent token more than `PT + IT late` by +0.197 fraction points, 95% CI [0.177, 0.215].
- Raw-shared `IT + PT mid` transfers the PT divergent token more than `IT + PT late` by +0.186 fraction points, 95% CI [0.170, 0.202].
- Native pure IT late-minus-mid margin gain is +20.63 logits, 95% CI [20.07, 21.20].
- Raw-shared pure IT late-minus-mid margin gain is +3.75 logits, 95% CI [3.48, 4.00].
- Native pure IT late margin exceeds `IT + PT late` by +11.18 logits, 95% CI [10.89, 11.48].
- Raw-shared pure IT late margin exceeds `IT + PT late` by +2.52 logits, 95% CI [2.41, 2.63].

These CIs are prompt-level descriptive intervals, not model-cluster intervals. Model heterogeneity should still be shown separately.

The strongest claim is not that late layers create assistant behavior from scratch. The better claim is:

Instruction tuning appears to expose or select behavioral candidates in mid-to-late circuits, while late layers reconcile those candidates with next-token prediction by strongly increasing the IT-vs-PT token margin and suppressing the alternative token.

## Cleanest Identity Evidence: Raw Shared Prompt

The raw-shared setting is the cleaner identity test because PT and IT are evaluated under the same prompt format.

Dense5 token identity at the first PT/IT divergent token:

- `PT + IT mid` matches the IT divergent token in 33.5% of cases.
- `PT + IT late` matches the IT divergent token in 13.6% of cases.
- `IT + PT mid` matches the PT divergent token in 34.5% of cases.
- `IT + PT late` matches the PT divergent token in 15.8% of cases.

This is the main reason Exp20 argues against a simple late-only birthplace account. If late layers were the primary birthplace of IT behavior, late swaps should transfer token identity more strongly than mid swaps. They do not.

## Cleanest Margin Evidence: Native And Raw Shared

Late layers dominate margin sharpening.

Dense5 native IT:

- Mid/policy window IT-vs-PT margin delta: +1.29 logits.
- Late/reconciliation window IT-vs-PT margin delta: +21.92 logits.

Dense5 raw-shared IT:

- Mid/policy window IT-vs-PT margin delta: +0.61 logits.
- Late/reconciliation window IT-vs-PT margin delta: +4.36 logits.

The scale is larger in native mode because the IT model is in its trained chat-template distribution. The sign pattern is consistent across native and raw-shared mode.

## Key Counterfactual Signature

The best single signature is `IT + PT late` in native mode:

- It still matches the IT divergent token in 75.9% of cases.
- But its late IT-vs-PT margin drops from +21.92 in pure IT to +10.75.

That means replacing late IT layers with PT late layers often does not erase the already-selected IT token identity, but it does materially weaken the final readout/confidence of that token. This is exactly the "late reconciliation/readout cost" pattern.

## Pairwise Free-Running Agreement

Free-running sequence agreement remains low overall because once a first token differs, later autoregressive state diverges quickly.

Important dense5 values:

- Native PT vs IT agreement: 1.4%, first divergence mean step 0.30.
- Raw-shared PT vs IT agreement: 4.3%, first divergence mean step 1.61.
- Raw-shared PT vs `PT + IT mid` agreement: 16.8%.
- Raw-shared PT vs `PT + IT late` agreement: 17.7%.
- Raw-shared IT vs `IT + PT mid` agreement: 11.5%.
- Raw-shared IT vs `IT + PT late` agreement: 18.6%.

Sequence-level agreement is therefore useful as a descriptive sanity check, but the first-divergent-token identity and margin metrics are the cleaner mechanistic readouts.

## Token Category Readout

Native first divergences are heavily affected by chat-template distribution:

- Native PT divergent tokens are mostly format tokens: 64.4% format.
- Native IT divergent tokens are mostly function/assistant-style tokens: 55.8% function/other.

Raw-shared first divergences are more balanced:

- PT side: 44.9% content, 39.3% format, 15.9% function/other.
- IT side: 41.1% content, 41.7% format, 17.2% function/other.

This is why native mode is best interpreted as deployment-format behavior, while raw-shared mode is best interpreted as the cleaner model-difference control.

## What To Say In The Paper

Recommended wording:

"Layer-swap counterfactuals at the first PT/IT divergent token show a division between token identity and final readout. Under a shared raw prompt, mid-layer swaps transfer the opposite model's divergent token identity more often than late-layer swaps, suggesting that policy-relevant candidate selection is already visible before the final layers. In contrast, late windows dominate the IT-vs-PT token-margin change, especially under the native IT chat template. Thus, the convergence gap is best understood as a late readout/reconciliation cost of instruction-tuned behavior, not necessarily the layer where the behavior first originates."

Avoid claiming:

- "Late layers create assistant behavior."
- "Exp20 proves a causal mechanism by itself."
- "Mid layers alone explain the convergence gap."

Safer claim:

- Exp20 provides causal-intervention evidence for a mid-to-late division of labor: mid layers more strongly affect candidate identity, late layers more strongly affect final token-margin reconciliation.

## Remaining Caveats

- Native mode has a real chat-template confound, but that confound is scientifically relevant to deployed IT behavior.
- Raw-shared mode controls prompt format better, but it may understate native assistant behavior because IT models are outside their preferred prompt distribution.
- First-divergent-token analysis reduces autoregressive confounding, but it is still not a full causal circuit proof.
- Bootstrap CIs here resample prompts, not model families. They support within-run stability, while cross-family generality should be argued from per-model consistency.
- DeepSeek should be reported separately from dense5 because MoE routing may change the interpretation of layer swaps.
