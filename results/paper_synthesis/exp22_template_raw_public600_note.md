# Exp22 Raw-Prompt Template-Regime Audit

Run: `exp22_template_raw_public600_20260505_132816`.

This audit reruns Exp22 on the two public dense families available without Hugging Face gated-model credentials (`qwen3_4b`, `olmo2_7b`) with PT and IT both receiving raw prompt text (`PROMPT_REGIME=raw`) and raw readouts only.

Quality gates pass: `600/600` records per branch, zero malformed records, minimum matched retention `0.998`, and maximum post-match SMD `0.083`.

The raw no-template endpoint-matched late-KL effect reverses on this public-family subset: `-0.275` nats, 95% CI `[-0.386, -0.170]`. The endpoint-free future-top-1-flip check remains positive: `0.063`, 95% CI `[0.044, 0.083]`, while remaining adjacent JS is near zero/slightly negative.

Generation lengths show that raw prompting is not a neutral wrapper removal. In this run, Qwen-IT hits the 128-token cap on every prompt, Qwen-PT on `79%`; OLMo-IT hits the cap on `36.5%`, OLMo-PT on `64.7%`. We therefore treat this as a template-regime limitation/audit, not as a replacement for the native IT endpoint-matched result.
