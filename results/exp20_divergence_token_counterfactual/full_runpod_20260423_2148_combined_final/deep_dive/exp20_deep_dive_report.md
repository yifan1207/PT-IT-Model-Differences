# Exp20 Deep-Dive Report

Generated from `results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final`. Quality checks pass for both `native` and `raw_shared`: no malformed rows, no duplicate prompt ids, 600 prompts per model/mode.

## Main Read

Exp20 supports the refined story: mid/policy layers are more tied to discrete behavioral token selection, while late layers mainly act as a reconciliation/readout stage that strongly changes IT-vs-PT token margins. Late swaps often change confidence/margin more than they change the already-selected token identity.

## Key Dense5 Numbers

- Native first PT/IT divergence is almost immediate: mean step `0.304` over `3000` prompts. Raw-shared mean step is `1.608`, which is less template-dominated but still early.
- Raw-shared `PT + IT mid` matches the IT divergent token in `33.5%` of cases, versus `13.6%` for `PT + IT late`. This is the cleanest token-identity evidence for mid-layer policy transfer.
- Raw-shared `IT + PT mid` matches the PT divergent token in `34.5%` of cases, versus `15.8%` for `IT + PT late`. Again, mid swap changes identity more than late swap.
- Native `IT` late-window margin gain is `21.92` logits, compared with only `1.29` in the mid window. Raw-shared shows the same direction, smaller scale: late `4.36`, mid `0.61`.
- Swapping PT late layers into IT in native mode leaves the IT token identity mostly intact (`75.9%` IT-token match) but cuts the IT late margin from `21.92` to `10.75`. That is exactly the “late reconciliation/readout cost” signature, not a birthplace-of-policy signature.

## Interpretation

The strongest version is not “late layers create assistant behavior.” The better claim is: IT policy candidates are selected/exposed in mid-to-late circuits; late layers reconcile those candidates with next-token prediction by sharply boosting the final IT token margin and suppressing competing PT/raw alternatives. The convergence gap can therefore be late-readout dominated even when the behavioral choice first becomes separable earlier.

## Caveats

- Native mode is scientifically relevant because it uses the IT model’s deployment-format chat template, but it has a strong prompt-format confound. Use raw-shared mode for cleaner PT/IT layer-swap identity comparisons.
- Helper shards introduce duplicate raw rows by design, but final merge dedupes by `(prompt_mode, model, prompt_id)`. Final quality checks show zero duplicate prompt ids in the analyzed merged records.
