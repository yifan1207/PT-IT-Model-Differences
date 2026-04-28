# First-Divergence Factorial Diffing for Post-Trained Language Models

**Anonymous authors** | NeurIPS 2026 Submission

---

## Abstract

Layer-stage accounts describe late refinement, but they do not define a causal estimand for how post-training changes a paired forward pass under identical histories. We introduce first-divergence factorial diffing: at the first shared-history prefix where a pretrained checkpoint (PT) and its post-trained descendant (IT) prefer different next tokens, we cross upstream residual state with downstream late stack and measure the IT-vs-PT divergent-token margin. Across five released dense PT/IT families, conditional on these checkpoints, the same IT late stack shifts the margin by `+3.21` logits from an IT-shaped upstream state but only `+0.57` from a PT-shaped state, giving a family-balanced full-holdout interaction of `+2.64` logits (95% CI `[+2.54, +2.74]`). The sign is positive in every family, remains positive without Gemma (`+1.77`), and survives a label-swap null. The selected-event nature matters: the effect is strongest in early response formation, but at generated positions `>=3` the interaction remains `+1.52`, and the conjunction of generated positions `>=3` with Gemma removed remains positive (`+0.79`). A factual/reasoning stress test keeps the interaction positive (`+1.81`) while the PT-upstream late-only term flips negative (`-1.18`), showing that late IT computation is conditionally effective rather than a portable positive module. Supporting first-divergence tests separate candidate identity from margin: middle-positioned substitutions transfer token identity more often, while late-positioned substitutions more strongly affect margin/readout. The contribution is a reusable empirical methodology and measurement battery for natural first disagreements in paired dense PT/IT model diffing, not circuit recovery or a general mechanism for instruction following.

---

## 1. Introduction

Instruction tuning changes what models say, but a final output alone does not tell us where paired PT and post-trained forward passes first become different. Once two checkpoints generate different previous tokens, later internal differences can reflect different histories rather than different computation under the same history. We therefore study the first place where the paired checkpoints disagree while the history is still shared.

**Definition 1: first-divergence factorial diffing.** For a prompt, let the **first-divergence prefix** be the earliest shared-history position where a pretrained checkpoint (PT) and its post-trained descendant (IT) prefer different next tokens. Denote those tokens `t_PT` and `t_IT`.

At a pre-specified late boundary, `U_PT` and `U_IT` are the residual states obtained by running the PT and IT checkpoints, respectively, on that identical prefix up to the boundary; `L_PT` and `L_IT` are the PT and IT downstream transformer blocks from that boundary onward. The four cells are:

| Upstream state | PT late stack `L_PT` | IT late stack `L_IT` |
|---|---|---|
| PT upstream `U_PT` | PT baseline | Late-only |
| IT upstream `U_IT` | Upstream-only | Matched IT |

A readout `R` is specified separately; it consists of the final norm, `lm_head`, and real-token mask used to score the final hidden state. A **common-IT readout** scores every factorial cell with the IT final norm, `lm_head`, and token mask; a **common-PT readout** does the analogous scoring with the PT readout. The outcome is the divergent-token margin

`Y_R(U,L) = logit_{R(U,L)}(t_IT) - logit_{R(U,L)}(t_PT)`,

so positive values favor the IT token.

The primary estimand under a fixed readout `R` is the **upstream-late interaction**:

`[Y_R(U_IT,L_IT) - Y_R(U_IT,L_PT)] - [Y_R(U_PT,L_IT) - Y_R(U_PT,L_PT)]`.

For intuition, suppose PT and IT agree on every previous token, but PT next prefers `.` while IT next prefers `,`. First-divergence factorial diffing freezes that shared prefix, crosses the two upstream states with the two late stacks, and asks how the comma-versus-period margin changes.

**Late terminology.** The paper uses three related but distinct late objects:

| Term | Meaning | Main use |
|---|---|---|
| **Late stack** | Transformer blocks from the late boundary to the final hidden state; readout is specified separately. | First-divergence factorial in Section 3.1. |
| **Late MLP window** | Only the MLP sublayers inside the pre-specified late window. | Graft/swap, identity/margin, and write-out tests. |
| **Final-20% KL region** | The final normalized-depth segment used to summarize `KL(layer || own final)`. | Delayed-stabilization context in Section 3.2. |

The central measurement is simple. Across five released dense PT/IT families, conditional on these checkpoints, the same IT late stack shifts the IT-vs-PT margin by `+3.21` logits from an IT-shaped upstream state but only `+0.57` logits from a PT-shaped upstream state, giving a family-balanced interaction of `+2.64` logits (95% CI `[+2.54, +2.74]`). The interaction is positive in every family, with heterogeneous magnitudes (median family `+1.85`; Gemma-removed mean `+1.77`), survives a label-swap null, and persists on a factual/reasoning stress test where the simple late-only term changes sign. This is a selected-event claim about natural first disagreements, not a random-token statement; Section 3.1 treats Gemma as the largest-magnitude family and early response formation as the strongest regime, not as hidden assumptions.

A companion first-divergence decomposition explains why the interaction matters. Middle-positioned window substitutions transfer token identity more often, while late-positioned window substitutions more strongly affect margin/readout.

The novelty is twofold. First, the paper gives causal evidence for stage heuristics that were previously mostly descriptive: middle-positioned windows are more tied to divergent-token identity, late-positioned windows are more tied to margin/readout, and late post-training computation works in compatibility with upstream state. Second, it contributes the first-divergence factorial methodology that makes this test local and falsifiable. Layer-localization work can say where task information or output sharpening tends to appear; it does not test the counterfactual "run the same IT late stack from the PT upstream state at the exact token where PT and IT first disagree." The 2x2 factorial makes that counterfactual explicit. A state-independent late effect is an operational null for the measurement, not a claim that prior authors believed late layers ignore their inputs; the point is to measure how far the paired PT/IT contrast departs from that null, whether the sign is label-aligned, and how the magnitude changes under family, position, and prompt-regime stratification. Separately, the delayed-stabilization analysis checks a different failure mode: if late MLP effects were generic late-window fragility, matched random residual-projection controls should move the KL readout too; they do not. The observed pattern exposes a sharper conclusion: late IT computation is conditionally effective rather than portable.

Late-stage dependence is not a new intuition. Transformer-circuits work treats the residual stream as the shared channel that later layers read from and write to (Elhage et al., 2021), and work on FFN readouts, tuned lenses, DoLA, stages of inference, calibration, and layer localization uses a vocabulary of updating, refinement, sharpening, or correction that presupposes upstream-shaped state (Geva et al., 2022a,b; Belrose et al., 2023; Chuang et al., 2024; Lad et al., 2025; Joshi et al., 2025; Zhao, Ziser, and Cohen, 2024). First-divergent-token metrics also appear in model-compression work as a way to detect when a compressed model first departs from a reference model (Deiseroth et al., 2024). Our use is different: we condition on the first natural PT/IT disagreement to run paired interventions, not to score compression degradation. The ingredients are established; the contribution here is to turn them into a measured estimand at that disagreement: an upstream-state x late-stack interaction on the actual divergent-token margin, with separate token-identity and margin readouts, a label-swap falsifier for the factorial, and a matched random falsifier for the late-MLP KL control. Because the protocol only requires paired checkpoints and a shared-history next-token disagreement, it can be reused for other model-pair contrasts, including base-to-SFT, SFT-to-preference-tuned, reasoning-tuned, safety-tuned, or constitution-modified checkpoints.

The project began from a simpler delayed-stabilization observation: IT models remain farther from their own final next-token distribution until later layers. We use that observation to motivate looking at late windows and to check that the factorial result fits a broader layerwise pattern. We state the claim at the granularity directly supported by the interventions: a paired-checkpoint decomposition in which middle substitutions are more diagnostic of candidate identity, while late MLP and late-stack computation interact with upstream state to shape the measured final PT/IT margin.

We use `instruction-tuned` and `IT` as readable shorthand for instruction-following post-trained descendants of pretrained checkpoints released by the same model families. The recipes are heterogeneous, including supervised tuning and preference optimization, so our claim is about the PT-versus-post-trained contrast rather than isolation of a single training algorithm. The main paper reports pooled results for five dense PT/IT families: Gemma 3 4B, Llama 3.1 8B, Qwen 3 4B, Mistral 7B, and OLMo 2 7B.

---

## 2. Setup

### 2.1 Models

The main experiments use five dense PT/IT model families:

| Model | Layers | d_model | Attention | Pre-training data | Post-training |
|---|---|---|---|---|---|
| Gemma 3 4B | 34 | 2560 | GQA, hybrid local/global (5:1) | Undisclosed | Multi-stage post-training |
| Llama 3.1 8B | 32 | 4096 | GQA, all global | 15T tokens | Iterative supervised + preference optimization |
| Qwen 3 4B | 36 | 2560 | GQA, all global | 36T tokens, 119 languages | Multi-stage post-training |
| Mistral 7B v0.3 | 32 | 4096 | GQA, full attention (`sliding_window=null`) | Undisclosed | Instruct checkpoint |
| OLMo 2 7B | 32 | 4096 | MHA, all global | OLMo-mix-1124 | SFT + DPO + RLVR |

The exact Hugging Face checkpoint IDs used by the dense-family runs are:

| Family | PT checkpoint | IT checkpoint |
|---|---|---|
| Gemma 3 4B | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` |
| Qwen 3 4B | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B` |
| Mistral 7B v0.3 | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` |
| OLMo 2 7B | `allenai/OLMo-2-1124-7B` | `allenai/OLMo-2-1124-7B-Instruct` |

These checkpoint IDs and their pinned Hugging Face snapshot revisions are part of the estimand: the claim is about these released PT/post-trained descendants, not an abstract training algorithm. The native layerwise curves use each model in its native prompting regime; raw-shared first-divergence and residual-state runs force identical raw prompt token IDs before comparing residual states. Appendix G records the immutable revisions now used by the experiment loaders, tokenizer notes, chat-template behavior, and the historical limitation that earlier traces stored repo IDs rather than resolved snapshot hashes.

All main-text claims below therefore refer to the five dense families.

### 2.2 Readouts and interventions

The implementation is model-adapter based rather than feature-dictionary based. Matched-prefix graft/swap and first-divergence tests operate on raw MLP activations and residual-stream states through a shared adapter interface, and no core claim depends on transcoders, SAEs, or model-specific decomposition dictionaries. The reported residual interventions are restricted to paired PT/IT checkpoints with compatible residual dimensions, tokenizer IDs for the tested prompts/tokens, and block semantics; we do not patch residual states across unrelated architectures.

Definition 1 fixes the core notation. The setup section adds the implementation details needed to interpret the interventions:

| Object | Implementation detail | Role in the story |
|---|---|---|
| First-divergence factorial diffing | Patch upstream residual state at the late boundary, run PT or IT downstream blocks at the first-divergence prefix, then score with a specified readout. | Estimates the primary upstream-late interaction. |
| Identity/margin decomposition | Measure token transfer separately from the IT-vs-PT logit margin at the same shared prefix. | Separates candidate identity from final readout pressure. |
| Delayed-stabilization context | Track `KL(p_l || p_L)` from each layer's decoded distribution to the same model's final distribution. | Explains why late windows were tested causally. |

Prompting is matched to the question being asked:

| Analysis | Prompt regime | Why this regime is used |
|---|---|---|
| First-divergence factorial and identity/margin tests | **Raw-shared:** PT and IT receive identical raw prompt token IDs before the divergent-token comparison. | Isolates checkpoint computation under a shared history. |
| Native layerwise convergence curves | **Native:** PT uses raw prompt text; IT uses the checkpoint chat template. | Measures the deployment-style layerwise signature that motivated late-window interventions. |
| Matched-prefix graft/swap and JS replay | **Teacher-forced shared histories** under the run's specified raw or native prompt stream. | Tests late-window effects after removing generated-history divergence. |
| Behavioral sanity check | **Natural decoding:** PT/intervened PT and IT/intervened IT use their intended generation setup. | Checks output movement, not mechanism identification. |

**First-divergence position distribution.** First-divergence prefixes are selected events, not random token positions. In the primary holdout, the `2,983` dense-family valid events occur early: mean generated position `2.17`, median `0`, with `1,499` at the first generated token. Position `0` is still the next-token prediction after the full raw prompt under identical raw token IDs, not a BOS-only or chat-template comparison. The IT-side divergent tokens are mixed (`1,265` content, `685` format, `1,033` function/other), and `551/2,983` events contain an assistant-marker token. Because this early-position skew can affect magnitude, Section 3.1 reports position-stratified and category-mix checks; Appendix F gives the full distribution notes.

The late boundary in the first-divergence factorial is the first layer of the pre-specified late window: Gemma layer `20`, Llama layer `19`, Qwen layer `22`, Mistral layer `19`, and OLMo layer `19`. Appendix F lists the full early/middle/late windows and boundary-sensitivity checks.

The residual-state patch replaces the full `[batch, sequence, d_model]` hidden-state tensor entering the late-boundary transformer block, not only the final-token vector. Donor and host branches use the same raw prompt plus generated-prefix token IDs and attention mask, and the runs set `use_cache=False`, so no K/V cache is carried across branches; downstream attention and MLP computations are recomputed from the patched full-prefix residual sequence. The patch is at the transformer-block input, before that block's internal attention/MLP layer norms, and is not a separate attention-sublayer or MLP-sublayer patch.

Everything else is supporting machinery. The delayed-stabilization signature `KL(p_l || p_L)` supplies layerwise context for the first-divergence tests; same-history replay, endpoint matching, label-swap controls, matched random residual-projection controls for the MLP-graft KL analysis, and behavioral audits are introduced only where they protect a claim. Coarsened exact matching (Iacus, King, and Porro, 2012) balances token steps on final entropy, top-1 confidence, and top-1/top-2 margin; following the caution recommended for activation patching by Heimersheim and Nanda (2024), the late-MLP graft KL analysis is paired with a matched random residual-projection control.

Throughout, causal language refers to intervention effects on measured readouts within constructed hybrid forward passes. Cross-model grafts and residual-state patches can create states neither original model would naturally visit. Matched histories, the existing MLP-graft random control, label-swap controls, and raw-shared validation reduce this concern, but they do not turn the hybrids into complete natural-model explanations. We therefore phrase the central conclusion as intervention-scoped upstream-late interaction on the first-divergence margin.

Unless otherwise stated, uncertainty intervals are 95% percentile bootstrap intervals over prompt-cluster units. For first-divergence factorials, the bootstrap unit is `model x prompt_id`: if a prompt contributes multiple event kinds, we first average the relevant effect within that prompt, then resample prompt clusters within each dense family, recompute the family effect, and average family estimates. This keeps the Dense-5 interval from treating repeated event records from the same prompt or all families as independent draws from one pool. In the primary holdout, the `2,983` valid `first_diff` records are also `2,983` prompt clusters; in the factual/reasoning extension, `5,889` event records collapse to `2,983` prompt clusters before interval estimation. The same analysis reports per-family and leave-one-family-out intervals for the interaction term, because that interaction is the primary non-independence estimand.

Secondary diagnostics are demoted to appendices. Commitment summaries and raw-lens variants address metric/probe robustness; `δ`-cosine is a geometric marker of late revision rather than a mechanism by itself; Gemma feature-level analyses are supplementary.

### 2.3 Evidence sets

The main results use controlled-history first-divergence and matched-prefix runs on 400- and 600-prompt subsets from the same broad prompt pool. These runs freeze the token history, compare pure and intervened PT/IT branches, and supply the upstream-late factorial, identity/margin decomposition, same-history JS, graft/swap, and MLP write-out analyses.

Layerwise context comes from two supporting runs: a native free-running 2,936-prompt convergence analysis, and a dense-family 600-prompt endpoint-matched run with raw and tuned probes. The behavioral sanity check uses a frozen 600-prompt subset emphasizing conversational, register-sensitive, safety, and format-sensitive items; it checks aggregate output movement with LLM judging and a completed two-rater human audit, but it is not used to identify the internal mechanism.

### 2.4 Code and artifact availability

For double-blind review, code and paper-facing artifacts are released through an anonymized archive containing the model adapters, experiment packages, launch scripts, analysis scripts, prompt datasets, summary tables, bootstrap intervals, human-audit summaries, and final plots needed to audit the claims.

The release commits the summaries and plots from which manuscript numbers are read, plus a mechanical audit entrypoint, `bash scripts/reproduce/reproduce_claims_from_summaries.sh`, that checks the primary numbers against those artifacts. Large raw activation arrays, probe tensors, tuned-lens checkpoints, and per-token traces are omitted from git for size and mirrored separately where needed. Appendix G maps each main claim to commands, expected artifacts, expected numbers, and rerun costs.

Internal run identifiers and file-level provenance are kept in Appendix G and the anonymized artifact archive rather than in the main narrative.

---

## 3. Results

The results start with the primary first-divergence factorial: does the late-stack effect change with upstream residual state? We then give the delayed-stabilization context that led us to late windows, followed by the first-divergence identity/margin decomposition. A one-paragraph behavioral sanity check closes the section, with details in Appendix B.

### 3.1 Core estimate: late-stack effects depend on upstream state

**Core factorial.**

Using Definition 1 with a fixed readout `R`, the simple late effect from PT upstream is `Y_R(U_PT,L_IT) - Y_R(U_PT,L_PT)`. The simple late effect from IT upstream is `Y_R(U_IT,L_IT) - Y_R(U_IT,L_PT)`. Their difference is the upstream-late interaction, the paper's primary estimand.

In the common-IT readout variant, `L_PT` and `L_IT` denote PT/IT downstream transformer blocks, but all four cells are scored using the IT final norm, `lm_head`, and real-token mask. The common-PT variant analogously scores all cells with the PT final norm, `lm_head`, and mask; native readouts score a cell with its host checkpoint's own readout.

Under the common-IT readout, swapping in the IT late stack shifts the IT-vs-PT margin by `+0.572` logits (95% CI `[+0.494, +0.647]`) from a PT upstream state, but by `+3.207` logits (95% CI `[+3.095, +3.321]`) from an IT upstream state. The matched 2x2 decomposition gives a late-stack main effect of `+1.890` logits (95% CI `[+1.805, +1.975]`), a larger upstream-context effect of `+4.239` logits (95% CI `[+4.105, +4.376]`), and a positive interaction of `+2.635` logits (95% CI `[+2.538, +2.736]`). The common-PT readout agrees (`+0.609`, 95% CI `[+0.539, +0.677]`, versus `+3.218`, 95% CI `[+3.109, +3.333]`, for IT-late given PT versus IT upstream).

This is the core non-additivity estimate. If a state-independent late-effect account were adequate under this readout, the IT late stack should produce similar IT-vs-PT margin shifts from PT-shaped and IT-shaped upstream states. Instead, the effect from PT upstream is much smaller, and the 2x2 interaction is strongly positive: late IT computation is important, but its effect is much larger when the state entering it is already IT-shaped.

We lead with the full-holdout estimate because immediate response formation is part of the PT/IT contrast, not an artifact to discard. We do not treat it as the only effect size. The stricter conjunctions are smaller and still positive: after removing generated position 0, the interaction is `+2.25`; at generated positions `>=3`, it is `+1.52`; and at generated positions `>=3` with Gemma removed, it is `+0.79` (95% CI `[+0.70, +0.88]`). Thus stratification attenuates magnitude, but it does not change the estimand's sign or the conclusion that late-stack effects are state-conditioned.

![Figure 1: Core estimate: late-stack effects are non-additive with upstream state under this readout. First-divergence factorial diffing crosses upstream residual state with downstream late stack while holding the scoring readout fixed. The same IT late stack shifts the IT-vs-PT margin by `+0.57` logits from a PT upstream state but `+3.21` logits from an IT upstream state, a `5.6x` asymmetry. If the late effect were approximately state-independent, this ratio would be near `1x`; the observed asymmetry and `+2.64` logit interaction show a large departure from that additive late-only baseline under the measured first-divergence margin. The right panel shows family-level interaction CIs; interaction, not the IT/PT ratio, is the inferential quantity. A label-swap control preserving each prompt's four cells puts the observed interaction beyond the null 99.9th percentile (`+0.239` logits; `p=5.0e-5`).](../results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/exp23_midlate_interaction.png)

| Family | First-divergence records | IT late stack from PT upstream | IT late stack from IT upstream | IT/PT ratio | Interaction (95% CI) |
|---|---:|---:|---:|---:|---:|
| Gemma 3 4B | `600` | `+0.10` | `+6.18` | `60.5x` | `+6.08` `[+5.72, +6.44]` |
| Llama 3.1 8B | `600` | `+0.79` | `+2.05` | `2.6x` | `+1.25` `[+1.10, +1.42]` |
| Qwen 3 4B | `600` | `+0.59` | `+2.05` | `3.5x` | `+1.46` `[+1.32, +1.62]` |
| Mistral 7B | `597` | `+0.61` | `+3.15` | `5.1x` | `+2.53` `[+2.35, +2.72]` |
| OLMo 2 7B | `586` | `+0.76` | `+2.61` | `3.4x` | `+1.85` `[+1.67, +2.03]` |
| Dense-5 | `2,983` | `+0.57` | `+3.21` | `5.6x` | `+2.64` `[+2.54, +2.74]` |

**Family heterogeneity.** The interaction is positive in all five dense families, and each per-family interaction interval excludes zero. The `+2.64` headline is a family-balanced mean, not a claim that families share one scalar effect size. Magnitudes vary from Llama (`+1.25`) to Gemma (`+6.08`); the median family is OLMo at `+1.85`, and the trimmed family mean after dropping the smallest and largest values is `+1.95`. Excluding Gemma leaves a four-family mean interaction of `+1.77` logits (95% CI `[+1.69, +1.86]`) and a `3.57x` IT-upstream/PT-upstream ratio; all leave-one-family-out interaction estimates remain positive, ranging from `+1.77` to `+2.98` logits with intervals above zero. Even excluding both Gemma and Mistral leaves Llama/Qwen/OLMo with a descriptive mean of `+1.52` logits. The paper-level claim is therefore sign-consistent upstream-late non-additivity across released dense PT/IT descendants, with family-specific magnitude.

**Gemma as the largest-magnitude family.** Gemma is not used as the representative magnitude. It has the largest descriptive ratio (`60.5x`, mostly because the PT-upstream denominator is only `+0.10` logits), the smallest tuned-vs-raw lens improvement in the convergence-gap layerwise plots, and the most late-concentrated MLP weight-change profile (Tables A1-A3). These are diagnostics for why we report Gemma separately, not a causal explanation for Gemma's larger interaction. The first-divergence factorial itself is a raw-logit margin measurement, so Gemma's large interaction is not a tuned-lens artifact; establishing that the late-concentrated weight-change profile causes the larger Gemma effect would require a within-family mediation test or more families, not a correlation across five checkpoints. Ratios are kept descriptive because small PT-upstream denominators inflate them; the logit interaction is the inferential quantity.

**Position sensitivity.**

The interaction is not only an immediate-disagreement effect, and position-0 should not be read as a chat-template or BOS artifact. The primary factorial uses raw-shared prompts for both PT and IT branches, validates identical prompt token IDs, and evaluates the first generated token after the full prompt-conditioned residual state. What position 0 captures is a real part of the paired-checkpoint contrast: post-training often changes the first response token, especially for register, format, and safety-sensitive prompts. Still, first-divergence position selects different disagreement populations. Recomputing the same family-balanced prompt-cluster estimator after dropping all position-0 records gives `+2.25` logits (95% CI `[+2.11, +2.38]`; Gemma removed `+1.14` `[+1.05, +1.23]`). Restricting to first divergences at generated position `>=3` gives `+1.52` logits (95% CI `[+1.36, +1.68]`; Gemma removed `+0.79` `[+0.70, +0.88]`), with all five family-specific intervals above zero. A CPU-only mix audit shows that this `>=3` subset is shifted toward conversational/governance prompts but is not a single-category residue: it retains all three primary prompt categories and all three collapsed IT-token categories (Appendix F). At generated position `>=5`, the pooled estimate remains positive (`+1.64`, 95% CI `[+1.39, +1.88]`; Gemma removed `+0.83` `[+0.71, +0.94]`), but the cell is thinner (`495` records) and Llama's family-specific interval is compatible with zero. Thus position affects composition, magnitude, and family-level power; it does not make the pooled interaction a first-token-only result. We treat this as mechanism texture: upstream-late compatibility is strongest in early response formation and remains positive later, rather than being a static scalar independent of prompt and generation state.

**Label alignment.**

A label-swap control tests whether this compatibility is aligned with the PT/IT labels rather than an arbitrary sign convention. Re-scoring each cell by the token favored by its late stack, the IT late stack gains `+5.56` logits from IT-shaped rather than PT-shaped upstream state, while the PT late stack gains `+2.92` logits from PT-shaped rather than IT-shaped upstream state; their difference is the same `+2.64` interaction. A permutation that preserves each prompt's four factorial cell values but randomly swaps the PT/IT label orientation gives a null centered near zero (99.9th percentile `+0.239` logits), with the observed amplification far outside the null (`p=5.0e-5`, the minimum resolvable with `20,000` permutations). Thus the result is not merely that late layers depend on upstream context; the PT-to-IT contrast is label-aligned.

**Stress tests and subgroups.**

On a factual/reasoning stress test, the interaction remains positive (`+1.81` logits, 95% CI `[+1.72, +1.90]`) while the IT late stack from PT upstream moves against the IT token (`-1.18` logits, 95% CI `[-1.26, -1.09]`). That sign flip is the point of the stress test: the stable result is the upstream-late interaction, not a domain-invariant positive late-only effect. The primary holdout is dominated by governance, register, safety, and formatting prompts, where post-training-specific mid-to-late cooperation is expected to be strongest and PT upstream state can still be partly compatible with the IT late stack. The extension uses distinct prompt IDs from a content/reasoning slice; the repeated `2,983` prompt-cluster count is incidental, because both runs start from roughly 600 prompts per family, while the extension records up to three event kinds per prompt and averages them within `model x prompt_id` before bootstrapping. Factual/reasoning prompts are less dominated by assistant-register and safety-format behavior, so the simple late-only term need not keep the same sign; the interaction remains positive for both factual prompts (`+1.96`) and reasoning prompts (`+1.35`).

Within the primary holdout, subgroup checks preserve the same sign without changing the estimand. The interaction is positive for conversational/governance prompts (`+2.05` logits), format prompts (`+3.61`), and safety prompts (`+2.83`), and it is also positive when stratified by first-divergent IT-token type: content-like (`+2.50`), format (`+2.60`), and function/other (`+2.81`). These strata characterize where the primary estimate comes from; they are not additional headline claims.

**Scope-check summary.**

The primary scope checks stay on this estimand:

| Check | Readout | Result | Interpretation |
|---|---|---:|---|
| Dense-5 primary | Common-IT 2x2 interaction | `+2.64` logits `[+2.54, +2.74]` | Family-balanced prompt-cluster estimate. |
| Family median | Common-IT 2x2 interaction | `+1.85` logits | Typical-family magnitude summary; no family is used as representative. |
| Trimmed family mean | Common-IT 2x2 interaction | `+1.95` logits | Drops the smallest and largest family interactions. |
| Gemma removed | Common-IT 2x2 interaction | `+1.77` logits `[+1.69, +1.86]` | Not only a Gemma effect. |
| Drop generated position 0 | Common-IT 2x2 interaction | `+2.25` logits `[+2.11, +2.38]` | Not only an immediate-disagreement effect. |
| Generated position `>=3` | Common-IT 2x2 interaction | `+1.52` logits `[+1.36, +1.68]` | Later-position threshold with all five family intervals above zero. |
| Generated position `>=3`, Gemma removed | Common-IT 2x2 interaction | `+0.79` logits `[+0.70, +0.88]` | Main skeptical conjunction; attenuated but still positive. |
| Generated position `>=5` | Common-IT 2x2 interaction | `+1.64` logits `[+1.39, +1.88]` | Pooled later-position check; per-family power is thinner. |
| Generated position `>=5`, Gemma removed | Common-IT 2x2 interaction | `+0.83` logits `[+0.71, +0.94]` | Thinner later-position conjunction; Llama family interval is compatible with zero. |
| Factual/reasoning stress test | Common-IT 2x2 interaction | `+1.81` logits `[+1.72, +1.90]` | Interaction persists outside the governance/register-heavy holdout. |
| Label-swap null | Compatibility amplification | observed `+2.64`; null 99.9th percentile `+0.239`; `p=5.0e-5` | Interaction is PT/IT-label aligned. |

The factual/reasoning run also reports the PT-upstream late-only term (`-1.18` logits, 95% CI `[-1.26, -1.09]`) as a stress-test qualifier rather than an interaction estimate. The companion controls support the identity/margin decomposition and late-window context:

| Check | Readout | Result | Interpretation |
|---|---|---:|---|
| Raw-shared prompt control | PT-host IT-token transfer | middle `26.0%` `[24.5%, 27.7%]`; late `17.6%` `[16.2%, 18.9%]` | Identity split survives without native IT templates. |
| Native IT-template counterpart | IT-host margin drop | late `13.25` logits `[12.91, 13.61]`; middle `12.01` `[11.66, 12.35]` | Native prompting makes late margin/readout strongest. |
| Matched random residual projection | Final-20% KL effect | true late graft `+0.327` nats `[+0.298, +0.359]`; random `+0.003` `[-0.002, +0.008]` | Not a generic late-window perturbation. |

The factorial rows report the residual-state estimator or its simple late-only term. The other rows are companion controls, separated here so the reader can tell which evidence supports the primary interaction and which evidence supports the decomposition.

### 3.2 Layerwise context: delayed stabilization and late-window localization

The factorial result fits the layerwise pattern that first led us to inspect late windows. Under native free-running decoding, IT models remain farther from their own final next-token distribution than PT models do through much of the forward pass. Under the tuned lens, the dense-family IT-minus-PT `KL(layer || own final)` gap is positive in the early, middle, and late thirds of the network (`+0.62`, `+0.54`, and `+0.33` nats), and raw-lens variants preserve the qualitative ordering.

The intervals in this section come from separate layerwise and graft/swap analyses and are not always the same bootstrap object as the first-divergence factorial intervals in Section 3.1. The KL graft/swap depth-ablation intervals below are family-bootstrap intervals over dense-family means; random-control intervals are prompt-bootstrap intervals. We use them as layerwise context and localization support, not as directly comparable uncertainty estimates for the primary upstream-late interaction.

![Figure 2: Layerwise context: delayed stabilization. IT models stay farther from their own final-layer prediction than PT models do across much of the forward pass. Mean `KL(layer || final-layer distribution)` per layer, tuned-lens decoded. This layerwise pattern led us to test late-window effects under matched-prefix interventions and first-divergence factorials.](../results/exp09_cross_model_observational_replication/plots/L2_mean_kl_per_layer_tuned.png)

This metric is endpoint-relative, so we check it against the most direct endpoint and history confounds. After matching token steps on final entropy, final confidence, and final top-1/top-2 margin, the late IT-minus-PT gap remains positive under both raw (`+0.425` nats, 95% CI `[+0.356, +0.493]`) and tuned (`+0.762` nats, 95% CI `[+0.709, +0.814]`) probes. Under identical teacher-forced histories, same-layer PT/IT JS divergence is also positive and grows late (`0.121` pre-late to `0.196` final-20% under the prompt-mean regional estimator). Appendix A gives the full endpoint-matching, same-history replay, probe, commitment, per-family, and reverse-teacher views.

Under identical token histories, late MLP substitutions have the largest tested leverage on that delayed-stabilization side. On the dense-family mean, the final-20% KL effect is `+0.34` nats (family-bootstrap 95% CI `[+0.18, +0.50]`) for PT with late IT MLPs grafted in, versus `-0.03` (95% CI `[-0.10, +0.02]`) early and `-0.05` (95% CI `[-0.11, +0.02]`) middle. The raw late-graft effect is positive in all five dense families (`+0.115` to `+0.609` nats), and leave-one-family-out dense means remain positive (`+0.274` to `+0.398`), so the result is not driven by a single family.

The mirrored swap test asks whether removing late IT MLPs from an IT host also reduces the delay. It does: replacing late IT MLPs with PT MLPs produces the largest reduction of IT delayed stabilization, with a dense-family mean effect of `-0.51` nats (family-bootstrap 95% CI `[-0.83, -0.22]`) versus `-0.10` (95% CI `[-0.26, +0.03]`) early and `-0.23` (95% CI `[-0.37, -0.09]`) middle.

![Figure 3: Symmetric graft/swap localization. Left and center: late MLP substitutions have the largest convergence-gap effect in both directions, increasing the delay in a PT host and reducing it in an IT host. Right: output-relevant late-stage summaries predict the late KL shift better than the residual-opposing geometry alone. The matched random-control number in the text checks that this is not a generic late-window perturbation effect.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png)

A matched random-control follow-up argues against the simplest "late layers are fragile" explanation. Replacing the learned late IT-minus-PT MLP effect with matched random residual-projection controls gives a dense-family final-20% KL effect of `+0.003` nats (95% CI `[-0.002, +0.008]`), compared with `+0.327` (95% CI `[+0.298, +0.359]`) for the true late graft; the true-minus-random margin is `+0.324` nats (95% CI `[+0.294, +0.358]`). Thus late MLP substitutions are the only tested substitutions with a large delayed-stabilization effect, rather than merely perturbing a sensitive late window. The depth-ablation and symmetric-swap intervals above are family-bootstrap intervals over the five dense model-family means, while the random-control interval is prompt-bootstrap; Appendix A gives per-family, weight-change, geometry, and random-control details.

### 3.3 Supporting decomposition: middle-positioned substitutions transfer identity more often than late substitutions

The primary factorial estimates the upstream-late interaction; we now ask what the middle and late windows contribute to first-divergence token formation. Because these windows overlap, "middle" and "late" here mean middle-positioned and late-positioned windows, not disjoint subnetworks. At prefixes where PT and IT first choose different next tokens, which depth window changes the token identity and which changes the final margin? The first-divergence test evaluates intervened models at the shared prefix, and the companion MLP write-out test measures each window's finite-difference logit support for the PT token, the IT token, top alternatives, and residual-opposing components. All intervals in this subsection are 95% percentile bootstrap intervals over dense-family prompt clusters or prompt-level first-divergence records, depending on the source experiment.

The main measurements are summarized in one table so the identity and margin readouts are not conflated:

| Readout | Early MLP window | Middle MLP window | Late MLP window | What it says |
|---|---:|---:|---:|---|
| PT host: IT-token transfer | — | `26.0%` `[24.5%, 27.7%]` | `17.6%` `[16.2%, 18.9%]` | Middle substitutions transfer candidate identity more often. |
| IT host: PT-token transfer | — | `31.2%` `[29.6%, 32.9%]` | `20.8%` `[19.4%, 22.3%]` | Mirror direction gives the same identity pattern. |
| IT-host margin drop | `11.53` `[11.20, 11.88]` | `12.01` `[11.66, 12.35]` | `13.25` `[12.91, 13.61]` | Late substitutions have the largest margin effect. |
| Pure IT MLP support for `t_IT` | `-0.041` `[-0.049, -0.032]` | `+0.021` `[+0.011, +0.032]` | `+0.789` `[+0.754, +0.825]` | Native IT-token support is concentrated late. |
| PT-to-IT change in `t_IT` support | `+0.034` `[+0.027, +0.042]` | `+0.070` `[+0.059, +0.080]` | `+0.715` `[+0.683, +0.747]` | The learned write-out difference is also concentrated late. |

The first two rows give the identity side. The transfer rates are well below `50%`, so a single-window MLP substitution does not dominantly redirect token identity. The middle-over-late gap is a relative localization signal: middle windows are systematically more involved in candidate identity than late windows, not a claim that any one window alone controls the divergent token.

The last three rows show why the late window still matters. Late MLP windows have the largest effect on the IT-vs-PT margin and provide the strongest local support for the IT divergent token in native IT runs. The local MLP-only proxy also shows what late MLPs are not: adding late IT MLP updates to a PT host changes the same fixed-prefix IT-vs-PT margin by only `+0.0035` logits (95% CI `[-0.001, +0.009]`). Thus the native IT late write-out is not a portable plug-in module that works independently of the residual stream entering it.

![Figure 4: Candidate transfer and local late support. First-divergence tests show that middle substitutions transfer PT-vs-IT token identity more, while late IT MLP windows provide the strongest native IT-token support. The full late MLP update carries the IT-vs-PT margin write-in in this proxy; the residual-opposing component is a geometric marker rather than the margin vector.](../results/paper_synthesis/exp20_exp21_handoff_synthesis.png)

The synthesis is now visible from both sides. At the granularity tested here, middle windows are more diagnostic of which PT- or IT-like token candidate is exposed, while late MLP and late-stack windows interact with the upstream residual state to shape the measured final margin.

The implication is non-modularity under the tested interventions: in the primary holdout, the same late computation is positive yet much smaller from a PT upstream state and strongest when the residual stream is already IT-shaped; in the factual/reasoning stress test, the late-only term from PT upstream becomes negative even though the interaction remains positive. That pattern is expected if late computation is dynamically compatible with the upstream state and if the relevant post-training contrast differs by domain. Mid and late windows must be modeled jointly under this readout; neither in isolation captures the PT/IT next-token disagreement.

The delayed-stabilization pattern is therefore best read as the layerwise trace of a broader handoff: upstream state already encodes much of the PT/IT decision, and late computation changes the margin only in compatibility with that state. Appendix A gives the token-flow chronology and residual-opposing component decomposition, while Appendix G maps the subgroup and mid+late interaction analyses to their audit artifacts.

As a behavioral sanity check, the same late-window intervention family also moves natural-decoding outputs in the expected direction under LLM judging and a completed two-rater human audit. Because behavior is not used to identify the internal mechanism, Appendix B contains the full judge setup, resolved-vote rates, unresolved-label rates, and kappa caveats.

## 4. Related Work: From Stage Heuristics to Estimands

Prior work gives the vocabulary for late-stage dependence. Transformer-circuits work treats the residual stream as a communication channel through which later layers read information written earlier (Elhage et al., 2021). FFN analyses view layer outputs as memory- or vocabulary-space updates later refined through the residual stream (Geva et al., 2022a,b). Logit-lens and tuned-lens work make layerwise prediction refinement visible (nostalgebraist, 2020; Belrose et al., 2023); DoLA exploits differences between earlier and later layer logits (Chuang et al., 2024); Lad et al. (2025) frames late computation as residual sharpening after earlier candidate construction; Joshi et al. (2025) studies late confidence correction after decision certainty has emerged; and Zhao, Ziser, and Cohen (2024) locate transitions from general to task-oriented representations. These accepted or field-standard accounts make upstream dependence plausible and often implicit. We read our result as causal evidence for that stage picture under paired PT/IT counterfactuals, plus a methodology for measuring it, not as a competing stage theory or the surprising discovery that late layers use residual state.

The missing piece is the estimand and control stack. Existing work does not estimate a paired PT/IT upstream-state x late-computation interaction at the token where the checkpoints first disagree. This is a different object from a layerwise map. Zhao, Ziser, and Cohen can locate where task-oriented information appears across layers; it does not answer whether the same post-trained late stack has the same causal margin effect when driven by a PT-shaped versus IT-shaped upstream state for the same prompt prefix and token pair. Under first-divergence factorial diffing, the same IT late stack produces a `+3.21` logit shift from IT upstream state versus `+0.57` from PT upstream state in the primary holdout, while a factual/reasoning stress test preserves the interaction but flips the late-only PT-upstream term negative. The contribution is therefore not the discovery that late layers can use upstream state, nor an invariant positive late-only effect. It is the measurement battery: an explicit upstream-late interaction in hybrid passes, per-family intervals, a label-swap factorial falsifier, a matched random late-MLP KL falsifier, and supporting identity/margin readouts.

The residual-opposing geometry supports this interpretation, but only as interpretation. Late IT MLP updates have a substantial component opposing the residual stream entering them, a geometry consistent with revision or reconciliation rather than simple additive accumulation. However, the residual-opposing component is not itself the margin write-in under our proxy: it contributes `-0.0046` logits to the IT-vs-PT margin (95% CI `[-0.009, -0.001]`), while the full late MLP update contributes `+0.768` logits (95% CI `[+0.729, +0.805]`). We therefore do not claim that residual opposition is the mechanistic source of late sharpening. The geometry is consistent with revision, but does not by itself write the IT-vs-PT margin.

Post-training model-diffing and activation-patching papers ask nearby questions. Wu et al. (2024) studies instruction-conditioned behavioral shift, Du et al. (2025) compares base and post-trained models across knowledge, truthfulness, refusal, and confidence, and Prakash et al. (2024) uses cross-model activation patching to study entity tracking. Du et al. is the closest cross-family comparator for the portable-versus-state-dependent framing: their result that some directions transfer while others differ across base/post-trained checkpoints is directly aligned with the question asked here. Our addition is narrower and more local: at the natural PT/IT next-token disagreement under matched-prefix control, we estimate whether the same late stack has the same margin effect from PT-shaped and IT-shaped upstream states, and we pair that estimate with label-swap, position, family, and prompt-regime checks.

Sparse-crosscoder model diffing gives another complementary route into paired-checkpoint comparisons. Lindsey et al. (2024) and Minder et al. (2025) ask which shared or model-specific features are present at fixed layers; first-divergence factorial diffing instead asks how upstream state and late computation cooperate at the token where paired checkpoints first disagree. These are compatible rather than competing tools: a natural follow-up is to train a crosscoder at the late boundary and test whether IT-specific features mediate the upstream-late interaction.

Layer-localization work is especially close to our interpretation. Zhao, Ziser, and Cohen (2024) study where multi-task information appears across layers in instruction-tuned models, and Panigrahi et al. (2023) study task-specific skill localization in fine-tuned language models. Our middle/late identity-margin split deliberately echoes this literature: it should be read as a local causal version of a familiar stage picture, not as a qualitatively new taxonomy. The distinction is the estimand. By conditioning on the first natural PT/IT next-token disagreement, holding histories fixed, and crossing upstream state with late stack, the experiment separates three quantities that layer maps conflate: which divergent token candidate is exposed, how much final margin pressure is applied, and whether late margin pressure is portable across upstream states. The identity effect is deliberately phrased as relative localization (`26.0%` versus `17.6%`), not as a claim that middle MLP windows alone determine the divergent token.

Accepted activation-intervention and safety-localization papers explain why our interaction estimate should not be read as a behavior vector. Panickssery et al. (2024) steer behavior with contrastive activation additions, Arditi et al. (2024) identify a dominant refusal direction, Li et al. (2025) localize safety-relevant layers, and Jain et al. (2024) study what makes and breaks safety fine-tuning. Our first-divergence result is compatible with that picture rather than a foil to it: middle windows are more tied to which behaviorally relevant token candidate is selected, while the first-divergence factorial estimates how late-stack margin effects depend on upstream state.

Taken together, the measurement battery supports a depth-window account: the first-divergence factorial shows a positive upstream-late interaction, the identity/margin decomposition makes middle windows more tied to candidate transfer than late windows, and the delayed-stabilization analysis explains why late windows were a natural place to intervene. The paper does not claim to recover a named middle-layer feature feeding a named late write-in circuit. Its contribution is to validate the qualitative middle/late heuristic with local causal evidence and to provide a reusable paired-checkpoint protocol that turns an expected residual-stream dependency into a measured, label-aligned, stratified counterfactual estimate.

## 5. Limitations and Next Tests

The causal claim is window-level and intervention-scoped, not circuit-level or a complete natural-model explanation. MLP grafts and swaps estimate causal leverage on specified readouts in hybrid forward passes, and first-divergence factorial diffing tests non-additivity between an upstream state and downstream late computation, but neither identifies the underlying features, heads, or MLP directions. Hybrid grafts can create off-manifold states, so we treat matched histories, the late-MLP random control, label-swap controls, and raw-shared checks as safeguards rather than as proof that every intervened state is natural. The early/middle/late windows are coarse stage probes and can overlap; Appendix F reports the exact boundaries and explains why the result should not be read as a sharp layer-index boundary. The directly supported claim is candidate-transfer plus a positive upstream-late interaction under these windows, not a general mechanism for instruction following.

The delayed-stabilization analysis is endpoint-relative by design: it asks how far each layer is from the same model's eventual next-token prediction. Endpoint-matched controls show that the measured late KL gap remains after balancing final entropy, confidence, and top-1/top-2 margin, while same-history JS replay removes the own-final endpoint from a companion readout. These controls do not exhaust all final-distribution properties, which is why the first-divergence factorial is the primary estimand and the KL/JS analyses serve as layerwise context and support.

The empirical scope is five dense 4B-8B families. This breadth is unusual for mechanistic model diffing, but it is not frontier-scale validation and it excludes non-dense claims from the main pool. An additional non-dense side run is retained only in Appendix A.6 because those grafts also perturb routing and expert selection. Larger dense models and a second non-dense family are natural external-validity tests.

The behavioral evidence is a consistency check, not a standalone benchmark. LLM judging and the completed human audit agree directionally on the primary pairwise contrasts, but high tie/both-bad rates (`44.3-77.0%`) and heterogeneous κ keep the behavioral result from carrying the mechanistic claim. The internal intervention result does not depend on those labels.

First-divergence factorial diffing measures non-additivity but leaves the mechanism inside the upstream state unresolved. It shows that IT-shaped upstream residual state and IT late computation interact strongly on the first-divergence margin; it does not identify which upstream features, attention heads, or MLP directions carry that state. We would revise the upstream-gating interpretation if a held-out family showed the IT-late-from-PT-upstream effect approaching the IT-upstream effect while the interaction vanished, or if feature-level mediation showed a late subspace implementing the PT/IT margin independently of upstream state. The next circuit-level test is therefore not another coarse window swap, but a feature- or subspace-level mediation test inside the middle-to-late handoff, which we leave to future work.

We also do not run the sparse-crosscoder mediation test discussed in Related Work. A natural follow-up is to train a BatchTopK crosscoder at the late-window boundary and test whether IT-specific features mediate the upstream x late-stack interaction.

---

## References

Aghajanyan, A., et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL 2021*. arXiv:2012.13255.

Ansuini, A., et al. (2019). Intrinsic Dimension of Data Representations in Deep Neural Networks. *NeurIPS 2019*. arXiv:1905.12784.

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. *NeurIPS 2024*. arXiv:2406.11717.

Belrose, N., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. *COLM 2024*. arXiv:2303.08112.

Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models with Dictionary Learning. *Anthropic*.

Cheng, E., Doimo, D., Kervadec, C., Macocco, I., Yu, J., Laio, A., & Baroni, M. (2024). Emergence of a High-Dimensional Abstraction Phase in Language Transformers. *ICLR 2025*. arXiv:2405.15471.

Chuang, Y., et al. (2024). DoLA: Decoding by Contrasting Layers Improves Factuality. *ICLR 2024*. arXiv:2309.03883.

Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS 2023*. arXiv:2304.14997.

Deiseroth, B., Meuer, M., Gritsch, N., Eichenberg, C., Schramowski, P., Assenmacher, M., & Kersting, K. (2024). Divergent Token Metrics: Measuring Degradation to Prune Away LLM Components -- and Optimize Quantization. *NAACL 2024*. arXiv:2311.01544.

Du, H., Li, W., Cai, M., Saraipour, K., Zhang, Z., Lakkaraju, H., Sun, Y., & Zhang, S. (2025). How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence. *COLM 2025*. arXiv:2504.02904.

Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. B. (2024). Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. *COLM 2024*. arXiv:2404.04475.

Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic*.

Elhage, N., et al. (2022). Toy Models of Superposition. *Anthropic Transformer Circuits Thread*. arXiv:2209.10652.

Facco, E., et al. (2017). Estimating the Intrinsic Dimension of Datasets by a Minimal Neighborhood Information. *Scientific Reports*.

Friston, K. (2005). A Theory of Cortical Responses. *Philosophical Transactions of the Royal Society B*, 360(1456), 815–836.

Geva, M., Schuster, R., Berant, J., & Levy, O. (2022a). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2022*. arXiv:2012.14913.

Geva, M., Caciularu, A., Wang, K. R., & Goldberg, Y. (2022b). Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. *EMNLP 2022*. arXiv:2203.14680.

Gold, J. I., & Shadlen, M. N. (2007). The Neural Basis of Decision Making. *Annual Review of Neuroscience*, 30, 535–574.

Guerdan, L., Barocas, S., Holstein, K., Wallach, H., Wu, S., & Chouldechova, A. (2025). Validating LLM-as-a-Judge Systems under Rating Indeterminacy. *NeurIPS 2025*. OpenReview.

Hyland, K. (2005). *Metadiscourse: Exploring Interaction in Writing*. Continuum.

Iacus, S. M., King, G., & Porro, G. (2012). Causal Inference Without Balance Checking: Coarsened Exact Matching. *Political Analysis*, 20(1), 1-24.

Heimersheim, S., & Nanda, N. (2024). How to Use and Interpret Activation Patching. *arXiv:2404.15255*.

Jain, S., Lubana, E. S., Oksuz, K., Joy, T., Torr, P. H. S., Sanyal, A., & Dokania, P. K. (2024). What Makes and Breaks Safety Fine-tuning? A Mechanistic Study. *NeurIPS 2024*. arXiv:2407.10264.

Joshi, A., Ahmad, A., & Modi, A. (2025). Calibration Across Layers: Understanding Calibration Evolution in LLMs. *EMNLP 2025*. arXiv:2511.00280.

Lad, V., Lee, J. H., Gurnee, W., & Tegmark, M. (2025). The Remarkable Robustness of LLMs: Stages of Inference? *NeurIPS 2025*. arXiv:2406.19384.

Levelt, W. J. M. (1989). *Speaking: From Intention to Articulation*. MIT Press.

Li, S., Yao, L., Zhang, L., & Li, Y. (2025). Safety Layers in Aligned Large Language Models: The Key to LLM Security. *ICLR 2025*. arXiv:2408.17003.

Lin, B. Y., et al. (2024). The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning. *ICLR 2024*. arXiv:2312.01552.

Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., & Olah, C. (2024). Sparse Crosscoders for Cross-Layer Features and Model Diffing. *Transformer Circuits Thread*.

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment. *EMNLP 2023*. arXiv:2303.16634.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS 2022*. arXiv:2202.05262.

Minder, J., Dumas, C., Juang, C., Chughtai, B., & Nanda, N. (2025). Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning. *NeurIPS 2025*. arXiv:2504.02922.

Nanda, N., & Lieberum, T. (2022). A Mechanistic Interpretability Analysis of Grokking. *ICLR MATH-AI Workshop 2023*.

nostalgebraist. (2020). interpreting GPT: the logit lens. *LessWrong*.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*. arXiv:2203.02155.

Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M. (2024). Steering Llama 2 via Contrastive Activation Addition. *ACL 2024*. arXiv:2312.06681.

Panigrahi, A., Saunshi, N., Zhao, H., & Arora, S. (2023). Task-Specific Skill Localization in Fine-tuned Language Models. *ICML 2023*. arXiv:2302.06600.

Prakash, N., Shaham, T. R., Haklay, T., Belinkov, Y., & Bau, D. (2024). Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking. *ICLR 2024*. arXiv:2402.14811.

Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *NeurIPS 2023*. arXiv:2305.18290.

Saxe, A. M., et al. (2018). On the Information Bottleneck Theory of Deep Learning. *ICLR 2018*.

Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Anthropic*.

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019). Best Practices for the Human Evaluation of Automatically Generated Text. *INLG 2019*.

Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., & Sui, Z. (2023). Large Language Models Are Not Fair Evaluators. *ACL 2024*. arXiv:2305.17926.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small. *ICLR 2023*. arXiv:2211.00593.

Wu, X., Yao, W., Chen, J., Pan, X., Wang, X., Liu, N., & Yu, D. (2024). From Language Modeling to Instruction Following: Understanding the Behavior Shift in LLMs after Instruction Tuning. *NAACL 2024*. arXiv:2310.00492.

Xu, Z., et al. (2025). Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning. *NeurIPS 2025*. arXiv:2502.07154.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Li, D., Gonzalez, J. E., Xing, E. P., Zhang, H., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks*. arXiv:2306.05685.

Zhao, Z., Ziser, Y., & Cohen, S. B. (2024). Layer by Layer: Uncovering Where Multi-Task Learning Happens in Instruction-Tuned Large Language Models. *EMNLP 2024*. arXiv:2410.20008.

---

## Appendix Guide

The appendices are organized by purpose rather than by experiment ID. Appendix figures use local monotone labels (`Figure A1`-`A37` in Appendix A, `Figure B1` in Appendix B) rather than legacy run numbers; appendix tables use `Table A1`-`A4`. A reviewer checking a specific claim can use this map:

| Need to check | Where to go | What is there |
|---|---|---|
| Primary upstream-late interaction and audit trail | Appendix G | Claim-to-command-to-number table for Exp23, label-swap null, subgroup checks, and content/reasoning extension. |
| Why late windows were tested | Appendix A.3-A.4 | Probe/commitment robustness, endpoint matching, same-history JS, graft/swap localization, and matched random control. |
| Gemma magnitude caveat | Appendix A.1 and A.3 | Weight-change diagnostics, raw-vs-tuned lens sensitivity, and Gemma-specific layerwise plots. |
| Identity/margin decomposition | Appendix A.5 | Same-history JS, first-divergence token identity, margin flow, and candidate/amplification chronology. |
| Behavioral sanity check | Appendix B | LLM-judge design, human-audit resolved rates, unresolved-label rates, and kappa diagnostics. |
| Scope and boundary caveats | Appendix F | Window definitions, first-divergence position sensitivity, prompt formatting, pooling, and mechanism granularity. |
| Reproducibility details | Appendix G | Checkpoint manifest, artifact policy, commands, hardware costs, and code/artifact map. |

## Appendix A: Supplementary Evidence Map

This appendix keeps the full diagnostic surface, grouped by the role each artifact plays in the argument.

![Figure A1: First-divergence factorial diffing schematic. A shared prefix reaches the first PT/IT next-token disagreement; middle-window interventions primarily test candidate identity, late-window interventions test margin/readout, and the first-divergence factorial estimates the upstream-late interaction.](figures/first_divergence_decomposition.svg)

### A.1 Geometry and weight-change controls

[Figure A2: Generation-step × layer heatmap for Gemma 3 4B. Four panels showing δ-cosine stability across generation steps.](figures/it_plot10_generation_heatmap.png)
[Figure A3: Per-layer weight change localization (PT -> IT) across the five dense families plus a separate DeepSeek MoE side case. Gemma shows late-layer concentration; the other dense families show more diffuse changes.](../results/exp09_cross_model_observational_replication/plots/L3_weight_diff_6panel.png)

[Figure A4: δ-cosine profiles across the five dense families plus a separate DeepSeek MoE side case. IT (solid) vs PT (dashed). Gemma shows the largest late IT-vs-PT shift; Llama shows the weakest sustained shift because its PT variant already exhibits substantial late residual-opposing geometry.](../results/exp09_cross_model_observational_replication/plots/L1_delta_cosine_6panel.png)

**Table A1: depth-ablation effect normalized by window weight-change proxy.** `Δ KL` is the matched-prefix PT-side final-20% `B_window - A'` effect. `Mean ΔW` is the mean per-layer MLP weight-change proxy from Figure A3 over the same graft window. `Δ KL / Mean ΔW` is a descriptive scale-normalized diagnostic, not a formal parameter-efficiency estimand. Late is the largest raw effect and the largest normalized effect in every family, while the largest weight-change window is not consistently late.

| Family | Early ΔKL / Mean ΔW | Mid ΔKL / Mean ΔW | Late ΔKL / Mean ΔW | Largest ΔW window |
|---|---:|---:|---:|---|
| Gemma 3 4B | `-5.5` | `-14.0` | `146.6` | Mid |
| Llama 3.1 8B | `-236.7` | `-148.9` | `453.2` | Late |
| Qwen 3 4B | `-38.8` | `-60.1` | `184.4` | Late |
| Mistral 7B | `878.2` | `736.2` | `1754.4` | Mid |
| OLMo 2 7B | `0.6` | `27.1` | `130.3` | Mid |
| DeepSeek-V2-Lite | `-11.4` | `-55.1` | `234.7` | Late |

The dense-family mean raw effect is `+0.341` nats late versus `-0.035` early and `-0.045` mid, while mean window weight-change is nearly identical for middle and late (`0.00180` vs `0.00179`). Thus the late intervention effect is not explained by a systematically larger late MLP weight delta in the dense-family pool.

**Table A2: per-family raw late effect and leave-one-family-out sensitivity.** Mistral is extreme only on the weight-normalized ratio because its `Mean ΔW` denominator is unusually small; it is not an outlier in raw late effect and does not drive the dense-family mean.

| Family | Raw late ΔKL | Late Mean ΔW | Late ΔKL / Mean ΔW | Dense-4 mean if removed |
|---|---:|---:|---:|---:|
| Gemma 3 4B | `+0.609` | `0.004156` | `146.6` | `+0.274` |
| Llama 3.1 8B | `+0.310` | `0.000685` | `453.2` | `+0.349` |
| Qwen 3 4B | `+0.491` | `0.002661` | `184.4` | `+0.304` |
| Mistral 7B | `+0.115` | `0.000066` | `1754.4` | `+0.398` |
| OLMo 2 7B | `+0.181` | `0.001392` | `130.3` | `+0.381` |
| Dense-5 mean | `+0.341` | — | — | — |

The leave-one-family-out means remain positive in every case. Removing Mistral increases the dense-family mean, so the pooled raw late-graft effect is not Mistral-driven. The large Mistral normalized value instead reflects a small absolute PT→IT MLP RMS weight-change denominator (`6.6e-5`) paired with a modest positive raw effect.

[Figure A5: Cross-model δ-cosine heatmaps. Full layer x generation-step heatmaps for the five dense families plus the DeepSeek side case (PT and IT side by side), showing the distribution of residual-opposing MLP geometry across the full forward pass.](../results/exp09_cross_model_observational_replication/plots/L1_heatmaps_6x2.png)

### A.2 Feature-level supplements

[Figure A6: Feature importance analysis. Per-feature contribution to late post-training computation at layers 20–33, showing the distribution of importance across transcoder features.](../results/exp03_corrective_stage_characterization/plots/plot_e3_11_feature_importance.png)

[Figure A7: Feature population dynamics. Gini coefficient and N50 distributions for IT vs PT at late layers, quantifying the broadening of the active feature repertoire.](../results/exp03_corrective_stage_characterization/plots/plot_feature_populations.png)

### A.3 Probe and commitment robustness

This subsection supports the layerwise context claim: the delayed-stabilization pattern is visible under raw and tuned probes, survives endpoint matching, and is not a Gemma-only tuned-lens artifact. These figures explain why late windows were worth testing, but they are not the paper's primary causal estimand.

[Figure A8: Tuned-lens validation. KL(layer ell || final) for the five dense PT variants plus the DeepSeek side case. Red = tuned logit lens, blue = raw logit lens. The tuned lens substantially reduces KL at intermediate layers for Llama, Qwen, Mistral, and OLMo, with the DeepSeek side case behaving similarly. Gemma improves only modestly at comparable depth, indicating small tuned-vs-raw improvement rather than total probe failure. We therefore report both tuned and raw results throughout, and interpret Gemma's tuned-lens thresholded metrics with extra caution.](../results/exp09_cross_model_observational_replication/plots/tuned_lens_validation_kl_to_final.png)

**Table A3: raw-vs-tuned sensitivity for the convergence-gap context metric.** Values are mean IT-minus-PT `KL(layer || own final)` differences from the existing cross-family layerwise summaries. The raw lens does not remove the dense-family effect: the dense-5 final-half convergence gap is larger under raw lens (`0.771`) than tuned lens (`0.410`), and Gemma's own raw final-half gap (`1.008`) is larger than its tuned value (`0.351`). Excluding Gemma leaves the dense-family late-half raw gap positive (`0.712`), so the dense-family late stabilization result is not driven by Gemma's weak tuned-lens probe. The all-run row with DeepSeek is reported only as an appendix side case. This table is a sensitivity check for the layerwise readout, not a raw-only rerun of every matched-prefix KL intervention; the matched-prefix JS replay and first-divergence token projections are the non-tuned companion evidence for those later claims.

| Scope | Lens | Early third | Middle third | Late third | Final-half CG | Full-stack mean |
|---|---:|---:|---:|---:|---:|---:|
| Dense-5 + MoE side case | Tuned | `0.617` | `0.558` | `0.303` | `0.398` | `0.492` |
| Dense-5 + MoE side case | Raw | `0.330` | `0.598` | `0.638` | `0.729` | `0.526` |
| Dense-5 | Tuned | `0.616` | `0.536` | `0.329` | `0.410` | `0.493` |
| Dense-5 | Raw | `0.177` | `0.420` | `0.771` | `0.771` | `0.461` |
| Dense-5 excluding Gemma | Tuned | `0.487` | `0.545` | `0.324` | `0.425` | `0.453` |
| Dense-5 excluding Gemma | Raw | `-0.228` | `0.217` | `0.734` | `0.712` | `0.251` |
| Gemma only | Tuned | `1.133` | `0.500` | `0.350` | `0.351` | `0.652` |
| Gemma only | Raw | `1.797` | `1.231` | `0.922` | `1.008` | `1.305` |

[Figure A9: KL-to-final trajectories in Gemma 3 4B. IT (solid) shows elevated KL-to-final at late layers (20–33), converging to the 0.1 nat threshold later than PT (dashed).](../results/exp03_corrective_stage_characterization/plots/plot6_kl_trajectory.png)

[Figure A10: Mind-change analysis in Gemma 3 4B. Per-layer mind-change rates by token category. IT's late layers (20–33) show a sharp spike in mind-changes, with many targeting structural and discourse tokens.](../results/exp03_corrective_stage_characterization/plots/plot_e3_10_mind_change.png)

[Figure A11: Adjacent-layer KL divergence in Gemma 3 4B. IT (solid red) shows three discrete revision phases: early (layers 5–6), mid (15–17), and late (27–28), while PT (dashed blue) shows lower and more uniform prediction revision across layers.](../results/exp03_corrective_stage_characterization/plots/plot_e3_12_adjacent_layer_kl.png)

[Figure A12: Candidate reshuffling in Gemma 3 4B. Number of unique top-1 candidates encountered up to each layer. IT (red) shows rapid expansion in late layers; PT (blue) stabilizes earlier.](../results/exp03_corrective_stage_characterization/plots/plot_e3_13_candidate_reshuffling.png)

[Figure A13: Alignment tax localization in Gemma 3 4B. Fraction of total activation mass allocated to IT-amplified features by layer depth. Late layers (20–33) show 14–16% of activation mass at layers 28–33.](../results/exp03_corrective_stage_characterization/plots/plot5_alignment_tax.png)

[Figure A14: Raw vs tuned logit lens commitment scatter. Per-step top-1 commitment layer under raw (x-axis) vs tuned (y-axis) logit lens. Points below the diagonal indicate tuned lens commits earlier (i.e., the tuned lens reveals earlier convergence that the raw lens misses). For most models, the tuned lens detects commitment at earlier absolute layers — consistent with its more faithful intermediate predictions — while preserving the IT > PT ordering.](../results/exp09_cross_model_observational_replication/plots/L2_raw_vs_tuned_scatter.png)

[Figure A15: Alternative commitment definitions. Commitment delay under majority-vote (≥90% subsequent layers KL < 0.1) for tuned and raw logit lens. The delay pattern replicates under this more conservative definition.](../results/exp09_cross_model_observational_replication/plots/L2_commitment_tuned_majority_0.1.png)

[Figure A16: KL threshold sensitivity (full). Mean commitment vs KL threshold τ for both tuned (red) and raw (blue) lenses. The IT–PT gap is consistent across thresholds from 0.05 to 1.0 nats.](../results/exp09_cross_model_observational_replication/plots/L2_pure_kl_threshold_sensitivity.png)

[Figure A17: Cosine and entropy commitment. Commitment defined via cosine similarity (cos(h_ℓ, h_final) > 0.95) and entropy convergence (|H_ℓ − H_final| < 0.2). These representation-space metrics show minimal IT–PT difference, establishing that the convergence gap is a logit-space phenomenon.](../results/exp09_cross_model_observational_replication/plots/L2_commitment_cosine_0.95.png)

[Figure A18: Commitment CDF by normalized depth. Cumulative distribution of commitment layers for PT (dashed) and IT (solid), four methods. The rightward shift of IT CDFs is visible under KL-based metrics but absent under top-1 for some models — confirming the delay is distributional, not merely an argmax effect.](../results/exp09_cross_model_observational_replication/plots/L2_commitment_cdf_4methods.png)

[Figure A19: Endpoint-matched convergence-gap check. Token steps are matched within `model x probe_family` on final-layer entropy, final top-1 confidence, and final top-1/top-2 margin. IT retains a higher late `KL(layer || own final)` under raw and tuned probes, and endpoint-free path metrics also remain positive.](../results/paper_synthesis/exp22_endpoint_deconfounded_summary.png)

**Table A4: endpoint-matched convergence-gap control.** Dense-family endpoint-control run over 600 prompts per PT/IT branch. Matching is coarsened exact matching within `model x probe_family` on final entropy, final confidence, and final top-1/top-2 margin.

| Quantity | Estimate |
|---|---:|
| Raw-probe late `KL(layer || own final)`, IT - PT | `+0.425` nats, 95% CI `[+0.356, +0.493]` |
| Tuned-probe late `KL(layer || own final)`, IT - PT | `+0.762` nats, 95% CI `[+0.709, +0.814]` |
| Remaining adjacent JS after endpoint matching, IT - PT | `+0.052`, 95% CI `[+0.048, +0.057]` |
| Future top-1 flips after endpoint matching, IT - PT | `+0.203`, 95% CI `[+0.190, +0.215]` |
| Minimum matched-token retention across model/probe branches | `0.796` |
| Maximum post-match endpoint-covariate SMD | `0.057` |
| Maximum malformed branch rate | `0.000` |

### A.4 Matched-prefix localization and diagnostic behavioral figures

This subsection supports the late-window localization claim under identical token histories. The load-bearing artifacts are the graft/swap depth ablations and the matched random late-MLP KL control; the free-running behavior figures are consequence checks, not localization evidence.

[Figure A20: Matched-prefix MLP graft trajectories across the five dense families plus a separate DeepSeek-V2-Lite MoE case. The intact IT model generates freely; the PT teacher-forced control and grafted PT branches are then forced to follow the same continuation. Solid lines show the raw-prompt branch and dashed lines the chat-template branch. The graft consistently reduces cross-KL to the IT teacher while reproducing the residual-opposing δ-cosine signature only partially in the dense-model pool.](../results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_400rand_v11_teacherforced/overview_trajectories.png)

[Figure A21: PT-side graft depth ablation. Equal-width early, middle, and late IT-MLP grafts are compared under identical teacher-forced token histories in a PT host. The late graft is the only window that consistently induces the final-window convergence-gap increase across the five dense families.](../results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/depth_ablation_paper_main.png)

[Figure A22: Free-running A/B/C output evaluation overview across all four judged metrics. A = PT raw, B = PT + late IT MLP graft under the same raw prompt, C = full IT model with its native chat template. B moves consistently toward C on benign false-refusal reduction, more selectively on assistant register, and only weakly on broad structure and harmful-prompt refusal.](../results/exp12_free_running_abc_graft/plots/exp12_eval_v1_20260413_v3/exp12_scores_overview.png)

[Figure A23: Improvement relative to the PT baseline in the free-running A/B/C evaluation. Red = B − A, green = C − A. Positive bars indicate improvement for G1, G2, and S1; for S2, positive bars indicate a reduction in false refusal. The graft consistently captures part of the A→C gap, but remains well short of the full IT endpoint on most metrics.](../results/exp12_free_running_abc_graft/plots/exp12_eval_v1_20260413_v3/exp12_delta_vs_a.png)

[Figure A24: Cross-family descriptive token-type analysis of the matched-prefix late stage. Left: displaced vs supported token classes under `A' -> B_late`. Center: teacher-token rank gain by collapsed token type. Right: token-type rank gain under early, middle, and late graft windows on the subset with recoverable raw depth traces. The late stage broadly supports the eventual teacher token and suppresses `FUNCTION/OTHER` raw-continuation-style alternatives, with a secondary formatting/discourse component.](../results/exp13_late_stage_token_support_analysis/exp13A_lite_20260415_live/exp13a_lite_paper_main.png)

[Figure A25: Descriptive token-support appendix view. Per-model panels, candidate entry/exit distributions, and mind-change summaries for the matched-prefix token-type analysis.](../results/exp13_late_stage_token_support_analysis/exp13A_lite_20260415_live/exp13a_lite_appendix.png)

[Figure A26: Symmetric matched-prefix graft/swap summary. Left: PT-side late-region KL deltas for early, middle, and late IT-MLP grafts relative to `A'`. Center: IT-side late-region KL deltas for early, middle, and late PT-MLP swaps relative to `C`. Right: dense-family predictive correlations for output-relevant late-stage summaries (`support_teacher`, `anti_top1`, `anti_kl_final`) and `δ`-cosine.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png)

[Figure A27: Symmetric graft/swap appendix view. Per-model bidirectional window-effect panels and late-stage mechanism summaries for the matched-prefix graft/swap analysis.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_appendix.png)

[Figure A28: Matched random-control specificity check. Actual IT graft deltas are compared with matched random residual-projection controls on final-20% KL-to-own-final. The dense-family late true effect is large while the matched random late effect is near zero, ruling out a generic same-window perturbation account for the main late KL result.](../results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/exp19B_final20_kl_true_vs_random.png)

[Figure A29: Assistant-facing bucket deltas in the free-running LLM-judge behavioral follow-up. Dense-family pooled `G2` deltas by prompt bucket for PT-side grafts and IT-side swaps, highlighting that the largest late judge-rated degradation on the IT side concentrates on conversational and register-sensitive prompts.](../results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_paper_it_targeting.png)

### A.5 Same-history JS and candidate/amplification details

This subsection supports the identity/margin decomposition. Same-history JS shows that PT/IT distributions differ under matched histories; first-divergence token-transfer figures show the relative middle-over-late identity effect; margin-flow figures show why late windows still matter for readout.

[Figure A30: Matched-prefix native same-layer JS divergence under identical teacher tokens. Per-model `JS(A', C)` curves and dense-family pooled summaries show that broad PT↔IT output divergence is already present through much of the stack and amplifies late even when teacher histories are frozen.](../results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/exp16_js_appendix_models.png)

[Figure A31: Matched-prefix JS control view. PT-side and IT-side target-gap closure bars and host-local perturbation controls under matched prefix show that direct same-layer gap closure is more mid-to-late distributed than purely late, motivating the paper's “broad circuit, late delayed-stabilization window” synthesis rather than a strictly late-confined story.](../results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/exp16_js_appendix_controls.png)

[Figure A32: Reverse teacher-stream JS check. Replaying the same matched-prefix native-JS analysis with PT-generated continuations as teacher tokens again shows a broad dense-family PT↔IT same-layer JS gap. Late amplification is teacher-stream-dependent under token-step weighting, while prompt-mean aggregation still rises late; this supports the broad identical-history divergence claim while arguing against a strict teacher-stream-invariant late-confined interpretation. The Llama reverse replay excludes 11 empty PT-teacher continuations.](../results/exp16_matched_prefix_js_gap/exp16_js_reverse_pt_teacher_20260422_165259/plots/exp16_teacher_direction_comparison.png)

[Figure A33: Token identity at the first PT/IT divergent token. Under the raw-shared prompt, middle swaps transfer opposite-model token identity more than late swaps, while native prompting shows the deployment-format counterpart.](../results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/exp20_token_identity_dense5_ci.png)

[Figure A34: Mid-vs-late IT-token margin effects. Late windows dominate IT-vs-PT token-margin changes, especially under the native IT chat template.](../results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/exp20_mid_late_margin_dense5_ci.png)

[Figure A35: Raw-shared per-model token-transfer heatmap. Across dense families, middle swaps generally transfer token identity more than late swaps, with DeepSeek reported separately as the MoE case in all-model outputs.](../results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/exp20_raw_shared_model_transfer_heatmap.png)

[Figure A36: Native PT/IT final-token margin flow. Dense-5 pooled mid and late window margin deltas for the finally emitted token, plus dense-family mid-selected/late-helped rates and per-model IT `FORMAT` late-minus-mid margin. Late IT windows produce the strongest margin gains, especially for `FORMAT` and `CONTENT`, while PT is flatter.](figures/exp18_pure_flow_overview.png)

[Figure A37: Matched-prefix candidate/amplification chronology. Left: teacher-token rank gain by disjoint window under identical histories. Middle: strict rate at which a token is first selected in the middle window and then further helped late. Right: continuity view of `A' -> B_window` top-1 displacement. Format-like tokens become teacher-rank-positive only late, while content shows larger middle-window gains.](figures/exp18_handoff_summary.png)

### A.6 DeepSeek/MoE side case

DeepSeek-V2-Lite is retained only as a descriptive side case wherever the experiments were run. It is not pooled into the main dense-family intervention, behavioral, endpoint-control, or first-divergence claims. We did not add a second MoE model or analyze router/expert-selection mechanisms, so the paper makes no claim about MoE generalization. The reason is methodological rather than cosmetic: in a MoE checkpoint, an MLP graft can change both expert computation and routing/expert selection, which is not directly comparable to dense MLP substitutions without additional controls.

## Appendix B: Evaluation Methodology

![Figure B1: Diagnostic free-running behavioral check. The same late intervention family produces measurable LLM-judge behavioral movement under natural decoding, but the figure is appendix material because behavior is an aggregate consistency check rather than core internal localization evidence.](../results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_paper_behavior_main.png)

### B.1 LLM judge design and rubric definitions

The behavioral run uses four pointwise judge tasks and two blind pairwise judge tasks. `G1` is a 1-5 score for response structure and formatting, independent of factual correctness. `G2` is a 1-5 score for assistant-like conversational register versus raw web-text continuation, applied to assistant-facing prompts. `S1` classifies harmful-request responses as `REFUSE`, `COMPLY`, or `INCOHERENT`; incoherence is not counted as refusal. `S2` is a binary benign false-refusal metric, where 1 means the model refused, over-hedged, or failed to engage with a safe request. Pairwise `G2` and `S2` ask which of two anonymized responses better satisfies the same criterion, with randomized A/B ordering and a `TIE` option.

Pointwise judging uses a bulk model (`google/gemini-2.5-flash`) and a second model (`openai/gpt-4o-mini`) through the OpenRouter-compatible interface. The manifest also records an escalation model (`openai/gpt-4o`), parser versions, rubric hashes, dataset hash, pipeline hash, and output hash. Judge calls use temperature 0, are parsed into fixed schemas, and are retried on parser failure. Pointwise items are escalated if the two judge models differ by at least 2 points on `G1/G2`, disagree on `S1`, or disagree on `S2`. Pairwise judging uses the bulk model, is blind to model family and condition names, and uses randomized A/B presentation for `B_late` versus `A`, `C` versus `D_late`, and `C` versus `A`.

The rubrics intentionally separate surface structure from assistant register and safety behavior. `G1` asks only about structure/formatting; `G2` asks whether the response behaves like a helpful assistant addressing the user; `S1` asks whether harmful prompts are refused; and `S2` asks whether benign safety prompts are incorrectly refused or over-hedged. Pairwise `G2` and `S2` are the primary behavioral readouts because they are condition-blind, randomized, and closer to the human-preference setup used in LLM-as-judge work. We do not use the judge to score factual correctness or content quality in these claims.

Known LLM-judge failure modes motivate this design. Prior work finds that strong LLM judges can approximate human preferences but are sensitive to position, verbosity, self-enhancement, rating indeterminacy, and rubric design (Liu et al., 2023; Zheng et al., 2023; Wang et al., 2023; Dubois et al., 2024; Guerdan et al., 2025). We therefore use condition blinding, randomized order, a `TIE` option, fixed rubrics, schema-validated outputs, model-disagreement escalation, bootstrap uncertainty, and a completed blind human audit rather than treating the automated judge as ground truth.

### B.2 Human audit results

The current behavioral run materialized blind human-audit packs for all five dense families: 120 pointwise audit items per model, 600 total. The primary pairwise human packet contains 1,200 blinded comparisons per rater: 60 items per dense model for each of four primary contrasts, `C` versus `D_late` on `G2`, `C` versus `D_late` on `S2`, `B_late` versus `A` on `G2`, and `B_late` versus `A` on `S2`. Two independent raters completed the returned CSVs in `paper_draft/human_eval_survey/pointwise/` and `paper_draft/human_eval_survey/pairwise/`; the hidden keys are kept separately under `paper_draft/human_eval_survey/keys/`.

The human protocol follows standard NLG human-evaluation practice: independent blinded raters, unchanged row IDs, no access to condition or model labels, confidence ratings, optional notes for ambiguity, and frozen labels before unblinding (van der Lee et al., 2019). We do not adjudicate disagreements into a forced single label, because rating indeterminacy is itself informative for these outputs. Pairwise resolved rates therefore exclude `TIE` and `BOTH_BAD`, but those unresolved votes are reported explicitly. Confidence intervals bootstrap over `pair_id` clusters, with both rater votes included for each sampled item.

| Contrast | Criterion | Human resolved target win (95% CI) | Resolved votes | Unresolved vote rate | LLM same-sample resolved target win |
|---|---:|---:|---:|---:|---:|
| `C` over `D_late` | `G2` | `68.4%` `[62.5, 74.5]` | `244/600` | `59.3%` | `76.5%` |
| `C` over `D_late` | `S2` | `62.3%` `[54.1, 70.4]` | `138/600` | `77.0%` | `75.4%` |
| `B_late` over `A` | `G2` | `60.2%` `[53.9, 66.5]` | `334/600` | `44.3%` | `62.1%` |
| `B_late` over `A` | `S2` | `65.2%` `[58.2, 71.5]` | `287/600` | `52.2%` | `65.7%` |

The audit agrees directionally with the LLM judge on all four primary pairwise contrasts, and all four pooled human confidence intervals are above chance on resolved votes. The result should still be read conservatively. Ties and both-bad labels are common, especially for IT-vs-`D_late` benign-safety comparisons, and some per-family cells have small resolved counts or flip direction. The human audit supports the pooled behavioral sanity check; it is not a per-family behavioral localization claim.

| Contrast | Criterion | Raw agreement | Cohen's κ, four labels | Collapsed κ |
|---|---:|---:|---:|---:|
| `C` vs `D_late` | `G2` | `53.3%` | `0.17` | `0.17` |
| `C` vs `D_late` | `S2` | `68.7%` | `0.27` | `0.29` |
| `B_late` vs `A` | `G2` | `56.7%` | `0.38` | `0.45` |
| `B_late` vs `A` | `S2` | `67.3%` | `0.56` | `0.54` |

Collapsed κ merges `TIE` and `BOTH_BAD` into a single unresolved label. The agreement pattern is heterogeneous: the strongest reliability is the PT-side benign-safety comparison, while the IT-side assistant-register comparison has only low-to-fair reliability because both raters often use `TIE` but disagree on some resolved items. This motivates using the human audit as directional confirmation of the behavioral sanity check, not as a high-precision substitute for the automated judge.

Pointwise audit labels, on common/applicable filled fields, are used only as judge-calibration diagnostics. Weighted R1/R2 κ is `0.64` for `G1` and `0.32` for `G2`; categorical κ is `0.45` for `S1` and `0.33` for `S2`. Human-vs-judge κ is `0.60` for `G1`, `0.47` for `G2`, `0.58` for `S1`, and `0.31` for `S2`. These values are mixed rather than decisive, especially for `G2/S2`, which is why the behavioral check remains a resolved-vote aggregate plus LLM-judge effect-size diagnostic rather than a claim of high-agreement pointwise human scoring.

### B.3 Statistical testing approach

Pointwise behavioral summaries are prompt-bootstrap estimates over dense-family records. Pairwise summaries report target win rate, other win rate, tie rate, and resolved win rate, with bootstrap 95% confidence intervals. Dense-family claims pool Gemma, Llama, Qwen, Mistral, and OLMo; DeepSeek is excluded from the canonical behavioral pool because its MoE routing makes dense-family MLP graft/swap interpretation non-comparable. The main LLM-judge behavioral claim is deliberately asymmetric: IT-side late-window degradation is strong, while PT-side late grafting produces partial movement mainly supported by blind pairwise preferences.

For the completed human audit, the primary diagnostic quantities are pairwise resolved win rates and bootstrap confidence intervals for the same four primary contrasts shown in Figure B1. The rater packets randomize A/B order, expose no model or condition labels, and include `TIE`/`BOTH_BAD` to avoid forcing noisy preferences. The human audit was analyzed after the automated judge setup, endpoint definitions, and primary contrasts were fixed, so it functions as an external validation check rather than as a source of endpoint selection. We report same-sample LLM rates beside the human rates to show direction agreement directly.

### B.4 Prompt dataset construction

The behavioral run uses a frozen 600-prompt core subset of `eval_dataset_v2.jsonl`. The subset emphasizes conversational and register-sensitive prompts, benign safety prompts, harmful safety prompts, and format-sensitive items. Pointwise `G1` applies broadly; `G2` applies only to assistant-facing records; `S1` applies to harmful safety records; and `S2` applies to benign safety records. The judge manifest stores hashes for the dataset manifest, pipeline manifest, human-audit manifest, and sample outputs so reruns can detect accidental dataset or pipeline drift.

## Appendix C: Token Classifier Specification and Robustness

### C.1 Classifier specification

Five categories with priority order (first match wins): STRUCTURAL (regex-matched, 2.8% baseline), PUNCTUATION (1.2%), DISCOURSE (Hyland 2005 taxonomy, 0.5%), FUNCTION (closed-class, 34.9%), CONTENT (default, 59.9%).

### C.2 Perturbation robustness analysis

Four perturbation scenarios tested. Only the STRUCTURAL/DISCOURSE boundary produces measurable sensitivity (Δ = 0.145 when merged — expected, as both capture formatting). Content-side perturbations produce Δ < 0.00001. No core finding depends on precise boundary choices.

## Appendix D: Commitment Threshold Sensitivity

The convergence-gap claim does not depend on a single commitment threshold. The main text summarizes the threshold-free top-1 commitment view; Appendix Figures A15-A18 show majority-vote, KL-threshold, cosine, entropy, and normalized-depth CDF variants. Across KL thresholds from 0.05 to 1.0 nats, the IT-minus-PT delay remains qualitatively stable. Representation-space commitment summaries are weaker, which is consistent with the paper's narrower claim that the robust PT↔IT separation is in decoded next-token distributions rather than arbitrary residual-stream distances.

## Appendix E: Broader Literature Positioning

Table E1 is an orientation map, not a proof of priority. To keep the comparison disciplined, it includes accepted conference papers or field-standard sources that are directly used by the argument, rather than every adjacent arXiv preprint. Each ingredient in the table already appears somewhere in the literature: PT/post-training comparison, cross-model or cross-checkpoint patching, layer localization, behavioral intervention, first-divergent-token metrics, and conceptual accounts of late computation reading upstream state. The novelty claim is narrower: in this paper those ingredients are assembled into a measurable upstream-state x late-stack interaction at the first natural PT/IT next-token disagreement, with matched-history controls, separate identity/margin readouts, a label-swap permutation null for the factorial, and a matched random residual-projection control for the late-MLP KL analysis. The coarse labels below only indicate which ingredients are directly part of each paper's design (`Yes`), present in a narrower task-specific form (`Partial`), or outside that paper's main design (`No`).

| Paper | PT↔post-trained descendants? | Cross-family paired checkpoints? | Identical-history internal comparison? | Symmetric depth-localized intervention? | First-divergence candidate/margin? | State x late-stack factorial? | Natural-decoding consequence test? |
|---|---|---|---|---|---|---|---|
| Lad et al. (2025) | No | No | No | No | No | No | No |
| Joshi et al. (2025) | No | No | No | No | No | No | No |
| Wu et al. (2024) | Yes | No | No | No | No | No | No |
| Du et al. (2025) | Yes | Yes | No | No | No | No | No |
| Li et al. (2025) | No | No | No | No | No | No | No |
| Jain et al. (2024) | Yes | Partial | No | Partial | No | No | Yes |
| Prakash et al. (2024) | Yes | No | No | Partial | No | No | No |
| Deiseroth et al. (2024) | Partial | No | Yes | No | No | No | No |
| Zhao, Ziser, and Cohen (2024) | Yes | No | No | No | No | No | No |
| Panigrahi et al. (2023) | No | No | No | Partial | No | No | No |
| Panickssery et al. (2024) | No | No | No | No | No | No | Yes |
| Ours | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Lad et al. (2025) and Joshi et al. (2025) are the closest phenomenon-level comparisons. Lad gives a generic vocabulary for mid-to-late inference stages: prediction ensembling followed by residual sharpening. Joshi studies late confidence calibration after decision certainty has emerged. We do not treat late sharpening, late calibration, or late-stage confidence correction as new phenomena. The delayed-stabilization analysis is the aggregate background signature; the contribution is the paired PT/IT first-divergence analysis that asks which depth windows change the divergent token identity and how the final IT-vs-PT margin depends jointly on upstream state and late computation.

Du et al. (2025) overlaps with the paper's basic model-diffing premise: compare base and post-trained descendants and ask how knowledge, truthfulness, refusal, and confidence change internally. That overlap is real and useful. Our narrower addition is to condition on the first natural PT/IT next-token disagreement and run matched-history interventions at that exact prefix, so the readout is not only "post-training changes confidence/refusal" but "this upstream state and this late stack interact on this divergent-token margin."

Prakash et al. (2024) is the closest methodological precedent in this broader set: CMAP patches activations across related base/fine-tuned models to reveal improved mechanisms. Our use is different in scope and target: we apply a symmetric equal-width early/mid/late MLP graft/swap design across model families, and the localized target is not entity tracking but natural PT/IT next-token disagreement under matched-prefix control. Following the caution urged by activation-patching work such as Heimersheim and Nanda (2024), we describe these interventions as causal leverage on measured readouts, not as complete mechanism recovery.

Zhao, Ziser, and Cohen (2024) and Panigrahi et al. (2023) are important layer-localization precedents. Zhao, Ziser, and Cohen study where task-oriented representations appear in instruction-tuned models, and Panigrahi et al. study task-specific skill localization in fine-tuned language models. These papers make upstream dependence plausible rather than surprising. The distinction is not that we are the first to see layer structure after instruction tuning, and the paper should not be read as another qualitative layer-stage taxonomy. The distinction is the estimand: we estimate a window-level upstream-state x late-stack interaction at the first natural PT/IT token disagreement, with matched histories, separate candidate-identity and margin outcomes, and a PT/IT label-swap null. That design directly tests whether the late-stage effect is portable across upstream states; the factual/reasoning sign flip shows why this is not a cosmetic strengthening of the older heuristic.

Panickssery et al. (2024), Li et al. (2025), and Jain et al. (2024) are the main reason we do not frame the contribution as finding an instruction-following activation vector or safety layer. Those papers show that activation-space interventions or layer-localized analyses can affect behavior, including safety-relevant behavior. The paper-level contribution here is that paired PT/IT next-token disagreement is decomposed with graft/swap interventions under identical histories and first-divergence counterfactuals, rather than by extracting a behavioral vector.

The table should therefore be read as a map of overlap, not as a claim that novelty follows from a unique combination of checkmarks. The stronger claim is the estimand: prior ingredient-papers motivate pieces of the design, but they do not report an upstream-state x late-stack interaction on the actual first-divergent PT/IT token margin with the label-swap factorial control and matched random late-MLP KL control used here. Our paper asks a narrower and more operational question: what paired PT↔post-training forward-pass signature appears across dense families, how middle and late windows divide candidate selection from margin effects, and whether that same intervention family matters under natural decoding.

## Appendix F: Additional Scope Notes

The main limitations are stated in Section 5. This appendix keeps only the scope details that are useful for interpreting or reproducing the experiments.

The early/middle/late labels are coarse intervention windows, not discovered sharp phase boundaries. We use overlapping windows at comparable normalized depths so that the comparisons are not forced by a single brittle cutoff:

| Family | Early window | Middle window | Late window |
|---|---:|---:|---:|
| Gemma 3 4B | `0-13` | `10-23` | `20-33` |
| Llama 3.1 8B | `0-12` | `9-21` | `19-31` |
| Qwen 3 4B | `0-13` | `11-24` | `22-35` |
| Mistral 7B | `0-12` | `9-21` | `19-31` |
| OLMo 2 7B | `0-12` | `9-21` | `19-31` |

Earlier boundary checks support this conservative reading. A Gemma layer-range sensitivity run varied the late intervention range across `18-33`, `20-31`, `20-33`, and `22-33`; broad late-range behavior survived, while magnitudes changed. An onset-threshold sensitivity run found that estimated onset layers move with threshold and family. We therefore interpret the interventions as depth-window tests rather than evidence for a single exact start layer.

Prompt formatting is part of the estimand rather than a nuisance to erase. Native layerwise runs use each checkpoint in its natural deployment regime: PT branches receive raw prompts and IT branches receive the checkpoint chat template. Raw-shared first-divergence controls force identical raw prompt token IDs before comparing residual states, and they preserve the qualitative middle-identity/late-margin split. We therefore read templates as part of native IT operation, not as the sole source of the effect.

The first-divergence records are position-skewed, so we also stratify the primary Exp23 interaction by generated position. The estimate is largest at the first generated token and smaller later, but it remains positive in the later-position bins:

| Position stratum | Records | Dense-5 interaction, logits (95% CI) | Gemma-removed interaction, logits (95% CI) |
|---|---:|---:|---:|
| all positions | `2,983` | `+2.64` `[+2.54, +2.73]` | `+1.77` `[+1.69, +1.86]` |
| position `0` | `1,499` | `+3.01` `[+2.88, +3.16]` | `+2.38` `[+2.25, +2.52]` |
| positions `1-4` | `989` | `+2.42` `[+2.26, +2.58]` | `+1.29` `[+1.17, +1.41]` |
| positions `>=1` | `1,484` | `+2.25` `[+2.11, +2.38]` | `+1.14` `[+1.05, +1.23]` |
| positions `>=2` | `1,123` | `+1.77` `[+1.63, +1.93]` | `+1.06` `[+0.97, +1.15]` |
| positions `>=3` | `800` | `+1.52` `[+1.36, +1.68]` | `+0.79` `[+0.70, +0.88]` |
| positions `>=4` | `641` | `+1.52` `[+1.33, +1.71]` | `+0.80` `[+0.70, +0.91]` |
| position `>=5` | `495` | `+1.64` `[+1.39, +1.88]` | `+0.83` `[+0.71, +0.94]` |
| position `>=10` | `140` | `+1.49` `[+0.85, +2.20]` | `+1.05` `[+0.81, +1.29]` |

Per-family thresholding clarifies the power tradeoff. At generated position `>=3`, all five family-specific intervals are above zero, with Llama close to the boundary. At generated position `>=5`, the dense and Gemma-removed pooled estimates remain positive, but Llama's family-specific interval is compatible with zero (`+0.116`, 95% CI `[-0.047, +0.269]`). We therefore use `>=3` for the stronger family-level robustness statement and treat `>=5` and `>=10` as thinner later-position diagnostics.

A CPU-only category-mix audit checks that the `>=3` threshold is not simply one remaining category. It is composition-shifted, so it should be read as a later-position robustness check rather than a balanced prompt-category estimate: `800` records remain across all five dense families, with prompt categories `GOV-CONV` `700` (`87.5%`), `GOV-FORMAT` `61` (`7.6%`), and `SAFETY` `39` (`4.9%`). The divergent IT-token categories remain more mixed: `CONTENT` `476` (`59.5%`), `FUNCTION_OTHER` `199` (`24.9%`), and `FORMAT` `125` (`15.6%`).

Crossing position with prompt category gives the same cautious read. GOV-CONV dominates the later-position subset, but its own interaction also attenuates from `+2.05` logits over all positions to `+1.51` logits at position `>=3`. GOV-FORMAT remains larger at position `>=3` (`+2.28` logits, `61` clusters), while SAFETY is positive but thin (`+0.64` logits, `39` clusters). Thus the later-position drop is partly a category-composition effect and partly a within-category position effect.

Assistant-marker tokens are not the whole explanation. In the primary holdout, `551/2,983` events contain an assistant-marker token; the non-assistant-marker stratum remains strongly positive (`+2.48` logits, 95% CI `[+2.38, +2.59]`), while assistant-marker events are larger (`+3.30`, `[+3.07, +3.53]`). This is the pattern expected if response-opening conventions are one high-magnitude case of the same upstream-late compatibility effect, not the sole source of it.

The main convergence metric, `KL(layer || own final)`, is intentionally endpoint-relative: it measures stabilization toward a model's own eventual next-token distribution. Endpoint-matched controls balance final entropy, final top-1 confidence, and final top-1/top-2 margin, and matched-prefix JS gives a separate endpoint-free check. The first-divergence factorial remains the primary causal estimand; the KL and JS analyses provide layerwise context.

The primary first-divergence confidence intervals use a cluster bootstrap at the `model x prompt_id` level. This matters because the content/reasoning extension reports multiple event kinds for some prompts (`first_diff`, `first_nonformat_diff`, and `first_assistant_marker_diff`). For any subgroup or readout where multiple event records share a prompt, the analysis first averages the effect inside that prompt and then resamples prompt clusters within each dense family before averaging family estimates. The primary holdout is a useful sanity case: it analyzes only `first_diff`, so its `2,983` valid records are exactly `2,983` prompt clusters. The content/reasoning extension is the harder case: `5,889` event records reduce to `2,983` prompt clusters, and the manuscript reports the cluster-corrected intervals.

The remaining limitation is mechanism granularity. The evidence supports window-level causal leverage and upstream-late non-additivity in constructed hybrid forward passes, not a named circuit with identified features, heads, and subspaces. The natural next tests are feature- or subspace-level mediation inside the middle-to-late handoff, larger dense checkpoints, and a non-dense design that separates MoE routing from expert computation.

## Appendix G: Reproducibility and Artifact Map

**Public release.** During double-blind review, the project repository is provided through an anonymized artifact archive. It includes the current manuscript draft, all source experiment packages under `src/poc/`, shared cross-model infrastructure under `src/poc/cross_model/`, grouped script entrypoints under `scripts/`, and committed paper-facing artifacts under `results/`. The canonical prompt and evaluation datasets are committed under `data/`, including `eval_dataset_v2.jsonl`, `eval_dataset_v2_holdout_0600_1199.jsonl`, `exp3_dataset.jsonl`, `exp6_dataset.jsonl`, and `gold_standard_v1.csv`.

**Audit levels.** We expose the results at three levels, so a reviewer can check the main numbers without trusting prose summaries.

| Level | Command | What it verifies | Expected cost |
|---|---|---|---|
| Summary audit | `bash scripts/reproduce/reproduce_claims_from_summaries.sh` | Recomputes the primary paper numbers from committed JSON/CSV summaries. | CPU only; under 1 minute. |
| Minimal raw shard | `bash scripts/reproduce/reproduce_minimal.sh` after fetching the shard | Validates a 20-prompt, one-family raw shard containing cached per-layer logits, intervention outputs, first-divergence records, and expected summary JSONs. | CPU-only if cached logits are used; about 1-3 GPU-hours on one 80GB A100/H100 if regenerated. |
| Full rerun | Experiment launchers under `scripts/run/` and source packages under `src/poc/` | Regenerates full traces, summaries, and plots. | Multi-GPU; see estimates below. |

**Checkpoint and tokenizer manifest.** The current artifact pins Hugging Face model and tokenizer loading through immutable registry revisions in `src/poc/cross_model/config.py`; the shared loader passes those revisions to both `AutoTokenizer.from_pretrained(..., revision=SHA)` and `AutoModelForCausalLM.from_pretrained(..., revision=SHA)`. The manifest generated by `scripts/analysis/build_model_checkpoint_manifest.py` is committed at `results/paper_synthesis/model_checkpoint_manifest.json` and records, for each model, the repo ID, configured immutable revision, current Hub `main` resolution, tokenizer metadata, and chat-template hash when the tokenizer is accessible; all configured revisions in the manifest equal the audited current `main` SHA. The reported full-run artifacts were generated before this registry-wide pinning patch and their per-run configs stored repo IDs rather than resolved hashes, so they should be read as repo-ID historical runs whose closest recoverable and now-pinned rerun target is the audited revision below. For Qwen, Mistral, OLMo, and DeepSeek, local Hugging Face cache `refs/main` matched these audited SHAs; Gemma and Llama were gated in the local audit but resolved through the Hub metadata API.

| Family | PT repo and pinned/audited SHA | IT repo and pinned/audited SHA | Tokenizer and prompt-template note |
|---|---|---|---|
| Gemma 3 4B | `google/gemma-3-4b-pt` `cc012e0a6d0787b4adcc0fa2c4da74402494554d` | `google/gemma-3-4b-it` `093f9f388b31de276ce2de164bdc2081324b9767` | Gated tokenizer in this audit; code uses raw PT prompts and IT `apply_chat_template(..., add_generation_prompt=True)`, with `<end_of_turn>` as an additional IT stop token. |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` | `meta-llama/Llama-3.1-8B-Instruct` `0e9e39f249a16976918f6564b8830bc894c89659` | Gated tokenizer in this audit; code uses raw PT prompts and IT `apply_chat_template(..., add_generation_prompt=True)`, with `<|eot_id|>` as an additional IT stop token. |
| Qwen 3 4B | `Qwen/Qwen3-4B-Base` `906bfd4b4dc7f14ee4320094d8b41684abff8539` | `Qwen/Qwen3-4B` `1cfa9a7208912126459214e8b04321603b3df60c` | `Qwen2TokenizerFast`, shared vocab size `151669`; base EOS `<|endoftext|>` (`151643`), IT EOS `<|im_end|>` (`151645`); IT native prompts call `apply_chat_template` with `enable_thinking=False`. |
| Mistral 7B v0.3 | `mistralai/Mistral-7B-v0.3` `caa1feb0e54d415e2df31207e5f4e273e33509b1` | `mistralai/Mistral-7B-Instruct-v0.3` `c170c708c41dac9275d15a8fff4eca08d52bab71` | `LlamaTokenizerFast`, shared vocab size `32768`, shared EOS `</s>` (`2`); IT native prompts use the checkpoint chat template. |
| OLMo 2 7B | `allenai/OLMo-2-1124-7B` `7df9a82518afdecae4e8c026b27adccc8c1f0032` | `allenai/OLMo-2-1124-7B-Instruct` `470b1fba1ae01581f270116362ee4aa1b97f4c84` | `GPT2TokenizerFast`, shared vocab size `100278`, shared EOS `<|endoftext|>` (`100257`) and pad `<|pad|>` (`100277`); IT native prompts use the OLMo chat template. |
| DeepSeek-V2-Lite side case | `deepseek-ai/DeepSeek-V2-Lite` `604d5664dddd88a0433dbae533b7fe9472482de0` | `deepseek-ai/DeepSeek-V2-Lite-Chat` `85864749cd611b4353ce1decdb286193298f64c7` | MoE appendix side case only; `LlamaTokenizerFast`, shared vocab size `100002`, shared EOS (`100001`), shared chat-template hash in the audit. |

Across native runs, PT branches use raw prompt text and IT branches use the checkpoint's own chat template through `tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)`. Across `raw_shared` first-divergence and residual-state runs, both PT and IT branches are forced to raw text and the code validates identical raw prompt token IDs before comparing residual states.

**Claim to command to expected number.** The first command below is the most important reviewer audit path: it checks the manuscript numbers directly against committed artifacts.

| Claim or number | Command | Expected artifact | Expected number |
|---|---|---|---|
| Dense-5 final-half convergence gap | `bash scripts/reproduce/reproduce_claims_from_summaries.sh` | `results/exp09_cross_model_observational_replication/data/convergence_gap_values.json` | tuned `0.410`; raw `0.771`; raw excluding Gemma `0.712` |
| Endpoint-matched late convergence gap | same | `results/paper_synthesis/exp22_endpoint_deconfounded_table.csv` | raw `+0.425` `[+0.356, +0.493]`; tuned `+0.762` `[+0.709, +0.814]`; remaining adjacent JS `+0.052` `[+0.048, +0.057]`; future flips `+0.203` `[+0.190, +0.215]` |
| Endpoint-free matched-prefix JS | same | `results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/js_summary.json` | prompt-mean `JS(A', C)=0.121` `[0.119, 0.123]` pre-corrective and `0.196` `[0.193, 0.198]` in the final 20%; layer-weighted curve point estimates `0.106` and `0.169` |
| Matched-prefix depth ablation | same | `results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/depth_ablation_metrics.json` | family-bootstrap intervals over dense-family means: early `-0.035` `[-0.096, +0.019]`, middle `-0.045` `[-0.114, +0.023]`, late `+0.341` `[+0.181, +0.502]` nats |
| Symmetric late graft/swap | same | `results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_summary.json` | family-bootstrap intervals over dense-family means: PT-side late graft `+0.338` `[+0.182, +0.504]`; IT-side late PT-swap `-0.509` `[-0.828, -0.224]` nats |
| Late random-control specificity | same | `results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/exp19B_summary_light.json` | true `+0.327` `[+0.298, +0.359]`; matched random residual-projection `+0.003` `[-0.002, +0.008]`; true-minus-random `+0.324` `[+0.294, +0.358]` nats |
| First-divergence identity split | same | `results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early/validation_analysis/summary.json` | raw-shared `PT+IT mid` vs `PT+IT late`: `26.0%` `[24.5, 27.7]` vs `17.6%` `[16.2, 18.9]`; mirror `31.2%` `[29.6, 32.9]` vs `20.8%` `[19.4, 22.3]` |
| Native late readout loss | same | same | pure IT minus late PT-swap margin drop `13.25` logits `[12.91, 13.61]` |
| MLP write-out proxy | same | `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/summary.json`; `analysis/effects.csv` | late IT-token support `+0.789` `[+0.754, +0.825]`; MLP-only PT-host late graft `+0.0035` `[-0.001, +0.009]`; residual-opposing component margin `-0.0046` `[-0.009, -0.001]` |
| Residual-state x late-stack context gating | same | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/exp23_summary.json`; `analysis/exp23_effects.csv` | common-IT late given PT upstream `+0.572` `[+0.494, +0.647]`; late given IT upstream `+3.207` `[+3.095, +3.321]`; upstream `+4.239` `[+4.105, +4.376]`; late-stack `+1.890` `[+1.805, +1.975]`; interaction `+2.635` `[+2.538, +2.736]` over `2,983` prompt clusters; Gemma-removed interaction `+1.77` `[+1.69, +1.86]` |
| First-divergence position sensitivity | `python scripts/analysis/analyze_first_divergence_position_sensitivity.py --n-bootstrap 2000` | `results/paper_synthesis/exp23_position_sensitivity_table.csv`; `results/paper_synthesis/exp23_position_sensitivity_per_family.csv`; `results/paper_synthesis/exp23_position_sensitivity_note.md` | drop position 0 interaction `+2.25` `[+2.11, +2.38]`; position `>=3` interaction `+1.52` `[+1.36, +1.68]`, with all five family intervals above zero; position `>=5` interaction `+1.64` `[+1.39, +1.88]` with thinner per-family support |
| First-divergence `>=3` category mix | `python scripts/analysis/analyze_exp23_position_category_mix.py` | `results/paper_synthesis/exp23_position_category_mix.csv`; `results/paper_synthesis/exp23_position_category_mix_note.md` | `800` records across all five families; prompt mix `GOV-CONV 87.5%`, `GOV-FORMAT 7.6%`, `SAFETY 4.9%`; IT-token mix `CONTENT 59.5%`, `FUNCTION_OTHER 24.9%`, `FORMAT 15.6%` |
| First-divergence position x prompt category | `python scripts/analysis/analyze_exp23_position_prompt_category.py` | `results/paper_synthesis/exp23_position_prompt_category_effects.csv`; `results/paper_synthesis/exp23_position_prompt_category_note.md` | GOV-CONV interaction all positions `+2.05`, position `>=3` `+1.51`; GOV-FORMAT position `>=3` `+2.28`; SAFETY position `>=3` `+0.64` with only `39` clusters |
| Exp23 label-swap interaction control | `uv run python scripts/analysis/analyze_exp23_compatibility_permutation.py --run-root results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4 --n-permutations 20000` | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/compatibility_permutation/` | primary IT-late boost `+5.56`, PT-late boost `+2.92`, interaction `+2.64` over `2,983` prompt clusters; label-swap null 99.9th percentile `+0.239`, `p=5.0e-5` |
| Exp23 subgroup characterization | `python scripts/analysis/analyze_exp23_interaction_subgroups.py --run-root results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4 --n-bootstrap 2000` | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/subgroups/` | prompt-category interactions: `GOV-CONV +2.05`, `GOV-FORMAT +3.61`, `SAFETY +2.83`; IT-token-category interactions: `CONTENT +2.50`, `FORMAT +2.60`, `FUNCTION_OTHER +2.81`; assistant-marker `+3.30`, non-assistant-marker `+2.48`; primary records equal prompt clusters |
| Content/reasoning residual-state extension | `bash scripts/run/run_exp23_exp21_content_reasoning_extension.sh` | `results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/analysis/exp23_summary.json`; `analysis/subgroups/exp23_subgroup_report.md` | common-IT interaction `+1.81` `[+1.72, +1.90]` over `2,983` prompt clusters from `5,889` event records; prompt interactions: `CONTENT-FACT +1.96`, `CONTENT-REASON +1.35`, `GOV-FORMAT +2.28`; late IT from PT upstream `-1.18` `[-1.26, -1.09]` |
| Content/reasoning label-swap interaction control | `uv run python scripts/analysis/analyze_exp23_compatibility_permutation.py --run-root results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8 --n-permutations 20000` | `results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/analysis/compatibility_permutation/` | content/reasoning IT-late boost `+4.62`, PT-late boost `+2.81`, interaction `+1.81`; label-swap null 99.9th percentile `+0.178`, `p=5.0e-5` |
| Content/reasoning MLP write-out extension | same | `results/exp21_productive_opposition/exp21_content_reasoning_20260427_0943_h100x8/analysis/summary.json`; `analysis/effects.csv` | late-weight IT-vs-PT MLP margin `+0.042` `[+0.036, +0.048]`; residual-opposing component `-0.000006` `[-0.000041, +0.000029]`; remaining/token-specific component `+0.049` `[+0.043, +0.055]` |
| Behavioral sanity check and human audit | same | `results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_behavior_summary.json`; `results/exp15_symmetric_behavioral_causality/human_eval/human_eval_summary.json` | LLM resolved G2: `56.3%` `[53.3, 59.5]` and `77.1%` `[74.7, 79.6]`; human resolved G2: `60.2%` `[53.9, 66.5]` and `68.4%` `[62.5, 74.5]` |

**Committed versus regenerated artifacts.** The repository commits paper-facing summaries and plots used for manuscript numbers: JSON/CSV/MD summaries, bootstrap confidence intervals, human-evaluation summaries, and final PNG figures. It does not commit large regenerated intermediates such as raw activation arrays (`*.npy`, `*.npz`), model/probe tensors (`*.pt`, `*.safetensors`), tuned-lens checkpoints, or multi-gigabyte raw per-token trace JSONL/GZ files. These are size exclusions, not confidentiality exclusions. The corresponding collection scripts, analysis scripts, prompt manifests, and anonymized archive pointers are public within the review artifact.

**Large-artifact locations.** During review, raw traces, tuned-lens probes, endpoint-control records, late random-control traces, first-divergence archives, and the minimal audit shard are mirrored under the anonymized artifact archive. The public camera-ready release will replace these placeholders with stable permanent storage locations after the double-blind period.

**Hardware and runtime.** All full reruns use bf16 inference and deterministic greedy decoding unless a script states otherwise. The 4B-8B dense-family experiments fit on one 80GB A100/H100 per model job; DeepSeek-V2-Lite is reported separately because it is MoE/SFT-only. The full dense-5 intervention and endpoint-control suite (Exp11/14/16/20/21/22/23, including trace collection and analysis) should be budgeted as hundreds of H100 GPU-hours and roughly 0.5-1.5 TB of transient disk if raw traces and per-token records are retained. The minimal 20-prompt shard needs one 80GB A100/H100, about 1-3 GPU-hours if regenerated from checkpoints, and about 20-80 GB local disk depending on whether all per-layer logits are kept. The summary audit is CPU-only and reads only committed JSON/CSV files.

**Tuned-lens reproducibility.** The main intervention claims do not require tuned-lens checkpoints. The matched-prefix JS replay, graft/swap KL summaries, first-divergence identity/margin results, and finite-difference MLP write-out analyses are auditable from committed summaries and do not depend on trained probes. The tuned lens is used for the layerwise visualization and commitment diagnostics, and the paper reports raw-lens sensitivity alongside it. Probe checkpoints are included in the anonymized artifact archive; retraining uses `uv run python -m src.poc.cross_model.tuned_lens --model MODEL --variant {pt,it} --device cuda:N`. A complete dense-5 retrain is 10 PT/IT runs; including the DeepSeek side case is 12 runs. The full set takes about 4-6 wall-clock hours on 8x80GB A100/H100 GPUs with joint all-layer training.

**Main claim-to-artifact map.**

| Claim or analysis | Code entrypoints | Committed artifacts |
|---|---|---|
| Dense-5 convergence gap and commitment delay, with DeepSeek side case in appendix artifacts | `src/poc/exp09_cross_model_observational_replication/`; shared adapters in `src/poc/cross_model/` | `results/exp09_cross_model_observational_replication/data/`, `results/exp09_cross_model_observational_replication/plots/` |
| Endpoint-matched convergence gap | `src/poc/exp22_endpoint_deconfounded_gap/`; `scripts/run/run_exp22_endpoint_deconfounded_gap_runpod.sh`; `scripts/analysis/analyze_exp22_endpoint_deconfounded_gap.py`; `scripts/analysis/build_exp22_endpoint_deconfounded_synthesis.py` | `results/paper_synthesis/exp22_endpoint_deconfounded_table.csv`, `results/paper_synthesis/exp22_endpoint_deconfounded_summary.png`, raw mirror listed above |
| Gemma feature and Tier-0 validation controls | `src/poc/exp06_corrective_direction_steering/`; `src/poc/exp07_methodology_validation_tier0/`; `scripts/run/run_exp7_0*.sh`; `scripts/plot/plot_validation_tier0.py` | `results/exp06_corrective_direction_steering/plots/`, `results/exp07_methodology_validation_tier0/data/`, `results/exp07_methodology_validation_tier0/plots/` |
| Layer-range and onset sensitivity | `scripts/plot/plot_validation_tier0.py`; Tier-0 0F/0J runs | `results/exp07_methodology_validation_tier0/0F/layer_range_sensitivity_table.csv`, `results/exp07_methodology_validation_tier0/0J/onset_table.csv`, `results/exp07_methodology_validation_tier0/plots/0F_layer_range_sensitivity.png`, `results/exp07_methodology_validation_tier0/plots/0J_onset_sensitivity.png` |
| Matched-prefix graft depth ablation | `src/poc/exp11_matched_prefix_mlp_graft/` | `results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/` and selected summaries under `results/exp11_matched_prefix_mlp_graft/data/` |
| Symmetric graft/swap and late random-control specificity | `src/poc/exp14_symmetric_matched_prefix_causality/`; `src/poc/exp19_late_mlp_specificity_controls/` | `results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/`, `results/exp19_late_mlp_specificity_controls/` |
| Behavioral sanity check and human audit | `src/poc/exp15_symmetric_behavioral_causality/`; `scripts/eval/llm_judge.py`; human-audit materials under `paper_draft/human_eval_survey/` | `results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/`, `results/exp15_symmetric_behavioral_causality/human_eval/` |
| Endpoint-free matched-prefix JS replay | `src/poc/exp16_matched_prefix_js_gap/`; `scripts/analysis/analyze_exp16.py`; `scripts/plot/plot_exp16.py` | `results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/`, `results/exp16_matched_prefix_js_gap/exp16_js_reverse_pt_teacher_20260422_165259/` |
| Native final-token flow and matched-prefix chronology | `src/poc/exp18_midlate_token_handoff/`; `scripts/run/run_exp18_yanda_full.sh` | `results/exp18_midlate_token_handoff/full_runpod_20260423_095122/`, `results/exp18_midlate_token_handoff/matched_prefix_latest/` |
| First-divergence token identity and margin amplification | `src/poc/exp20_divergence_token_counterfactual/`; Exp20 analysis scripts under `scripts/analysis/` | `results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/`, `results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early/validation_analysis/` |
| MLP write-out, local context-gating proxy, content/reasoning extension, and synthesis figure | `src/poc/exp21_productive_opposition/`; `scripts/run/run_exp21_productive_opposition_runpod.sh`; `scripts/run/run_exp23_exp21_content_reasoning_extension.sh`; `scripts/analysis/analyze_exp21_productive_opposition.py`; `scripts/analysis/build_exp20_exp21_handoff_synthesis.py` | `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/`, `results/exp21_productive_opposition/exp21_content_reasoning_20260427_0943_h100x8/analysis/`, `results/paper_synthesis/` |
| Mid+late KL factorial, residual-state x late-stack interaction, subgroup/position characterization, label-swap control, and content/reasoning extension | `src/poc/exp23_midlate_interaction_suite/`; `src/poc/exp23_midlate_kl_factorial/`; `scripts/run/run_exp23_midlate_interaction_suite.sh`; `scripts/run/run_exp23_exp21_content_reasoning_extension.sh`; `scripts/analysis/analyze_exp23_midlate_interaction_suite.py`; `scripts/analysis/analyze_exp23_interaction_subgroups.py`; `scripts/analysis/analyze_first_divergence_position_sensitivity.py`; `scripts/analysis/analyze_exp23_position_category_mix.py`; `scripts/analysis/analyze_exp23_position_prompt_category.py`; `scripts/analysis/analyze_exp23_compatibility_permutation.py` | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/`, `results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/analysis/`, `results/paper_synthesis/exp23_position_sensitivity_table.csv`, `results/paper_synthesis/exp23_position_sensitivity_per_family.csv`, `results/paper_synthesis/exp23_position_category_mix.csv`, `results/paper_synthesis/exp23_position_prompt_category_effects.csv` |
