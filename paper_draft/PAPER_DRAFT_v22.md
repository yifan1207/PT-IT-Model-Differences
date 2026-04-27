# Context-Gated Late Readout at First PT/IT Divergence

**Anonymous authors** | NeurIPS 2026 Submission

---

## Abstract

We study the first prefix where a pretrained checkpoint (PT) and its post-trained descendant (IT), given identical history, prefer different next tokens. This **first-divergence** setting turns PT/IT model diffing into a local token-level test: which depth windows change the candidate token, and which windows change the final margin? Across five dense PT/IT families, middle-window substitutions transfer the IT divergent token more often than late substitutions (`26.0%` versus `17.6%` in a PT host), while late MLP substitutions have the largest tested leverage on delayed stabilization and final-token margin. The central result is a controlled measurement of context-gated late readout. Prior late-stage accounts already assume that late computation operates on upstream-shaped state; here we estimate that dependence directly. In a residual-state x late-stack factorial, the same IT late stack shifts the IT-vs-PT margin by `+3.21` logits from an IT-shaped upstream state but only `+0.57` logits from a PT upstream state, a `5.6x` asymmetry with interaction `+2.64` logits (95% CI `[+2.54, +2.74]`); excluding Gemma leaves a four-family mean interaction of `+1.77` logits (95% CI `[+1.69, +1.86]`). A factual/reasoning extension preserves the interaction (`+1.68`, 95% CI `[+1.59, +1.78]`) while making the PT-upstream late-only term negative, so the stable claim is not that late layers always help, but that late readout depends on upstream state. The convergence gap, same-history JS replay, matched random controls, and behavioral audit serve as supporting evidence and controls. The paper's claim is therefore a window-level intervention decomposition: middle windows are more tied to PT/IT token identity, while upstream state and late computation must be modeled jointly to account for the measured final PT/IT next-token margin.

---

## 1. Introduction

Instruction tuning changes what models say, but a final output alone does not tell us where the PT and IT forward passes first become different. We therefore study a narrow moment: the first prefix where a pretrained checkpoint and its post-trained descendant choose different next tokens. At that point the histories are still shared, so we can ask which depth windows change the candidate token and which windows make an already exposed candidate win the final next-token competition.

We call this position the **first-divergence prefix**. It converts paired-checkpoint model diffing from aggregate behavior comparison into a local disagreement test: intervene before either model has generated a different previous token, then measure token identity and final margin separately. By **context-gated late readout**, we mean that the effect of late computation on next-token margin depends strongly on the upstream residual state, rather than acting as a fixed additive module.

The paper has three claims. Everything else in the main text is evidence, control, or scope for these claims:

| Claim | What it says | Load-bearing evidence |
|---|---|---|
| First-divergence protocol | PT/IT disagreement can be studied at the first shared-history token where the checkpoints prefer different continuations. | Matched histories, raw-shared first-divergence tests, intro schematic. |
| Identity/margin decomposition | Middle windows lean more toward token identity; late MLP windows lean more toward margin/readout. | `26.0%` vs `17.6%` IT-token transfer; native late IT-token support `+0.789` logits; late random-control `+0.003` vs true late graft `+0.327` nats. |
| Measured upstream x late-stack interaction | Late readout is not an autonomous late module; its measured effect depends on upstream state. | Residual-state x late-stack interaction `+2.64` logits, with `+3.21` from IT upstream versus `+0.57` from PT upstream; content/reasoning extension interaction `+1.68`. |

Late-stage dependence is not a new intuition. Transformer-circuits work treats the residual stream as the shared channel that later layers read from and write to (Elhage et al., 2021); FFN memory/readout work, tuned-lens work, DoLA, stages-of-inference, calibration, layer-localization, and instruction-vector papers all use a vocabulary of updates, refinement, sharpening, correction, or circuit selection that presupposes upstream-shaped state (Geva et al., 2022a,b; Belrose et al., 2023; Chuang et al., 2024; Lad et al., 2024; Joshi et al., 2025; Zhao, Ziser, and Cohen, 2024; Bigoulaeva et al., 2026). Our contribution is to measure that dependence in a paired PT/IT setting. To our knowledge, no prior paper conditions on the first natural PT/IT disagreement, separately manipulates upstream residual state and downstream late computation, and estimates their interaction with identity/margin readouts and matched random falsifiers. The novelty is therefore not the folklore that late layers use earlier state, but the controlled token-level measurement of that dependence.

![Schematic: First-divergence decomposition. A shared prefix reaches the first PT/IT next-token disagreement; middle-window interventions primarily test candidate identity, late-window interventions test margin/readout, and the residual-state x late-stack factorial tests whether late amplification depends on an IT-shaped upstream state.](figures/first_divergence_decomposition.svg)

The convergence gap is the observational signature that motivates the causal tests, not the headline claim. It says that IT models remain farther from their own final next-token distribution until later layers. Same-history JS replay, endpoint matching, and matched random controls then keep that observation from carrying too much weight alone.

What we claim is a window-level intervention decomposition of PT/post-trained next-token formation at the tested depth granularity: middle windows are more tied to candidate identity, while late MLP and late-stack computation interact with upstream state to shape the measured final PT/IT margin. What we do not claim is a complete instruction-following circuit, a named middle-layer feature feeding a named late write-in component, or evidence that late layers autonomously implement assistant behavior.

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
| Mistral 7B v0.3 | 32 | 4096 | GQA, sliding window | Undisclosed | Instruct checkpoint |
| OLMo 2 7B | 32 | 4096 | MHA, all global | OLMo-mix-1124 | SFT + DPO + RLVR |

The exact Hugging Face checkpoint IDs used by the dense-family runs are:

| Family | PT checkpoint | IT checkpoint |
|---|---|---|
| Gemma 3 4B | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` |
| Qwen 3 4B | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B` |
| Mistral 7B v0.3 | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` |
| OLMo 2 7B | `allenai/OLMo-2-1124-7B` | `allenai/OLMo-2-1124-7B-Instruct` |

OLMo 2's base recipe is centered on `OLMo-mix-1124` with a late `Dolmino-mix-1124` curriculum, so the earlier single-dataset shorthand was too coarse. For OLMo we use the retrained non-preview 1124 PT/IT pair with shared GPT2-style tokenization. For Qwen we use the same Qwen 3 4B release line (`Base` to post-trained `Qwen3-4B`); the tokenizer vocabulary is shared, but the IT tokenizer uses the chat-turn EOS (`<|im_end|>`) while the base tokenizer uses `<|endoftext|>`. The discovery curves use each model in its native prompting regime; matched-history replay, template controls, and graft/swap experiments then test the internal signature under controlled histories. Appendix G records the checkpoint/version audit, including resolved Hub `main` SHAs and the fact that historical runs loaded the default `main` revision rather than pinning a commit hash.

All main-text claims below therefore refer to the five dense families.

### 2.2 Readouts and interventions

All core interventions are architecture-agnostic at the implementation level. Matched-prefix graft/swap and first-divergence tests operate on raw MLP activations and residual-stream states through a model-agnostic adapter system. No core claim depends on transcoders, SAEs, or model-specific decomposition dictionaries.

To reduce bookkeeping, the main text uses paper-facing evidence names rather than internal experiment IDs. The first-divergence, identity/margin, and residual-state rows carry the three-claim spine; the rest are motivation, controls, or bridge evidence:

| Evidence name | What it measures | Role in the story |
|---|---|---|
| Delayed-stabilization signature | `KL(p_l || p_L)`, the decoded distribution at layer `l` versus the same model's final-layer distribution. | Observational motivation for looking at late computation. |
| Same-history replay | PT and IT same-layer JS divergence under identical teacher-forced token histories. | Endpoint-free and history-matched companion check. |
| Window substitution | Equal-width early/middle/late MLP grafts and swaps. | Window-level causal leverage on delayed stabilization. |
| First-divergence token test | Token identity and margin at the first shared-history PT/IT disagreement. | Separates candidate identity from final margin. |
| MLP write-out proxy | Finite-difference logit support from MLP updates for PT token, IT token, and alternatives. | Tests local late support without claiming portability. |
| Residual-state x late-stack factorial | Patch upstream residual state at the late boundary, then run PT or IT late stack/readout. | Main context-gating test. |
| Label-swap control | Randomly flip PT/IT orientation while preserving each prompt's factorial cells. | Tests whether compatibility is aligned with PT/IT labels. |
| Behavioral bridge | Free-running output changes under the same late intervention family, audited by LLM judge and humans. | Consistency check, not core localization evidence. |

The delayed-stabilization signature is deliberately endpoint-relative and does not carry the causal claim by itself. Coarsened exact matching (Iacus, King, and Porro, 2012) balances token steps on final entropy, top-1 confidence, and top-1/top-2 margin; this checks the most direct endpoint covariates without claiming to control every final-distribution property. Following the caution recommended for activation patching by Heimersheim and Nanda (2024), learned substitutions are paired with matched random residual-projection controls: if late layers merely amplify arbitrary perturbations, the random control should move the same metric.

Throughout, causal language refers to intervention effects on measured readouts within constructed hybrid forward passes. Cross-model grafts and residual-state patches can create states neither original model would naturally visit. Matched histories, random residual-projection controls, label-swap controls, and raw-shared validation reduce this concern, but they do not turn the hybrids into complete natural-model explanations. We therefore phrase the central conclusion as window-level causal leverage and context-gated late readout, not full circuit recovery.

Unless otherwise stated, uncertainty intervals are 95% percentile bootstrap intervals over prompt-level units. For the headline residual-state x late-stack factorial, we resample prompt units within each dense family, recompute the effect inside each family, and average family estimates; this keeps the Dense-5 interval from treating all prompt records as independent draws from a single family. The same analysis reports per-family and leave-one-family-out intervals for the interaction term, because that interaction is the primary non-independence estimand.

Secondary diagnostics are demoted to appendices. Commitment summaries and raw-lens variants address metric/probe robustness; `δ`-cosine is a geometric marker of late revision rather than a mechanism by itself; Gemma feature-level analyses are supplementary.

### 2.3 Evidence sets

The same three claims draw on four evidence sets. These are data sources, not additional paper claims.

**Native discovery.** The cross-family free-running convergence analyses use 2,936 prompts spanning factual QA, reasoning, code, safety, format compliance, and custom assistant-style prompts. Each prompt is decoded greedily up to 512 tokens. This establishes the model-in-use signature.

**Endpoint-matched discovery.** A dense-family 600-prompt run stores raw- and tuned-probe layerwise trajectories for PT raw-prompt and IT native-chat continuations. The analysis matches token steps within `model x probe_family` on final-layer entropy, final top-1 confidence, and final top-1/top-2 margin, then re-estimates late KL-to-own-final and endpoint-free path metrics on the matched token steps.

**Controlled-history localization.** Matched-prefix runs use 400- and 600-prompt subsets derived from the same broader pool. These runs freeze teacher tokens, compare pure and intervened PT/IT branches under the same histories, and supply the matched-prefix JS, graft/swap, random-control, and first-divergence analyses. A reverse JS replay with PT-generated continuations is reported as an appendix check.

**Behavioral bridge.** The free-running behavioral follow-up uses a frozen 600-prompt subset of `eval_dataset_v2.jsonl`, emphasizing conversational prompts, assistant-register prompts, benign and harmful safety prompts, and format-sensitive items. The main behavioral estimates are LLM-judge results, with a completed two-rater blind human audit of the primary pairwise contrasts and pointwise judge-calibration labels.

### 2.4 Code and artifact availability

For double-blind review, code and paper-facing artifacts are released through an anonymized artifact archive. The archive contains the model adapters, experiment packages, launch scripts, analysis scripts, prompt datasets, summary tables, bootstrap intervals, human-audit summaries, and final plots needed to audit the claims in this paper.

The release separates reviewer-facing evidence from regenerated bulk traces. We commit the summaries and plots from which manuscript numbers are read, and include a mechanical audit entrypoint, `bash scripts/reproduce/reproduce_claims_from_summaries.sh`, that checks the headline numbers against those artifacts. Raw activation arrays, model/probe tensors, tuned-lens checkpoints, and multi-gigabyte per-token JSONL/GZ traces are omitted from git for size; where needed, large artifacts are mirrored under anonymized artifact prefixes and the supplementary audit shard contains a small raw subset. Appendix G maps each main claim to commands, expected artifacts, expected numbers, and rerun costs.

Internal run identifiers and file-level provenance are kept in Appendix G and the anonymized artifact archive rather than in the main narrative.

---

## 3. Results

The results follow the three-claim map from the introduction. Section 3.1 motivates and defines the first-divergence protocol. Sections 3.2-3.3 give the identity/margin decomposition: late windows have the largest tested leverage on delayed stabilization and margin, while middle windows transfer token identity more. Section 3.4 is the main context-gating test. Section 3.5 is a behavioral consistency check.

### 3.1 Claim 1: first-divergence gives a local PT/IT test

The first claim is methodological: study PT/IT model differences at the first shared-history token where the paired checkpoints disagree. Before applying that protocol, we need a reason to expect a depth-structured PT/IT difference. The observational signature is delayed stabilization: under native free-running decoding, IT models remain farther from their own final next-token distribution than PT models do through much of the forward pass. We call this the **convergence gap**. Under the tuned lens, the dense-family IT-minus-PT `KL(layer || own final)` gap is positive in the early, middle, and late thirds of the network (`+0.62`, `+0.54`, and `+0.33` nats), and raw-lens variants preserve the qualitative ordering.

![Figure 1: The convergence gap. IT models stay farther from their own final-layer prediction than PT models do across much of the forward pass. Mean `KL(layer || final-layer distribution)` per layer, tuned-lens decoded. This delayed-stabilization signature motivates the matched-prefix intervention decomposition that follows.](../results/exp09_cross_model_observational_replication/plots/L2_mean_kl_per_layer_tuned.png)

The measured late gap persists after balancing the most direct endpoint covariates. In the endpoint-control run, the primary estimator uses coarsened exact matching within `model x probe_family` on final-layer entropy, final top-1 confidence, and final top-1/top-2 margin. Matching retains at least `79.6%` of token steps per model/probe branch, with maximum post-match endpoint-covariate SMD `0.057` and no malformed branches. On matched token steps, dense-family late `KL(layer || own final)` remains higher for IT than PT under both raw (`+0.425` nats, 95% CI `[+0.356, +0.493]`) and tuned (`+0.762` nats, 95% CI `[+0.709, +0.814]`) probes. Endpoint-free path checks under the same matching also remain positive: remaining adjacent JS is `+0.052` (95% CI `[+0.048, +0.057]`) and future top-1 flips are `+0.203` (95% CI `[+0.190, +0.215]`) for IT minus PT. Thus the convergence-gap result remains endpoint-relative by definition, and this control does not exhaust all possible endpoint properties. What it does show is that, under direct matching on final entropy, confidence, and top-1/top-2 margin, the dense-family late IT>PT KL gap remains large and positive.

A separate PT/IT separation appears when token histories are held fixed. This test removes the own-final endpoint from the readout: if the convergence-gap result were only an artifact of comparing each model to its own final distribution, PT and IT same-layer distributions need not separate under identical histories. Instead, replaying identical teacher continuations through the PT control and intact IT model gives dense-family same-layer JS divergence of `0.121` (95% CI `[0.119, 0.123]`) through the pre-late stack and `0.196` (95% CI `[0.193, 0.198]`) in the final 20% under the prompt-mean regional estimator, with `final20 > pre` in all five dense families. Thus the convergence gap is not only a free-running history artifact or an own-final endpoint artifact, although JS remains a companion readout rather than a replacement for the convergence metric. The layer-weighted curve in Figure 2 gives slightly smaller point estimates (`0.106` pre-late and `0.169` final-20%) with the same ordering.

![Figure 2: Same-history JS check. Under identical teacher-forced token histories, PT and IT same-layer output distributions still differ without using either model's own final endpoint. This makes the delayed-stabilization result less likely to be only a free-running-history or endpoint-comparison artifact, while leaving the stronger convergence interpretation to the intervention and first-divergence tests.](../results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/exp16_js_main.png)

The convergence gap is therefore the observational signature that motivates the paper's first-divergence intervention decomposition, not the window-level causal leverage claim by itself. Appendix A gives endpoint-matching, probe, commitment, per-family, and reverse-teacher robustness views.

### 3.2 Claim 2 support: late windows carry the delayed-stabilization side

Which depth window has the largest tested leverage on delayed stabilization under identical token histories? We graft IT MLPs into one equal-width early, middle, or late window of a PT host while forcing the same IT-generated continuation, then measure the convergence gap of each intervened branch against its own endpoint.

The late graft is the only tested window that consistently induces the delayed-stabilization signature. On the dense-family mean, the final-20% KL effect is `+0.34` nats (family-bootstrap 95% CI `[+0.18, +0.50]`) for PT with late IT MLPs grafted in, versus `-0.03` (95% CI `[-0.10, +0.02]`) early and `-0.05` (95% CI `[-0.11, +0.02]`) middle. The raw late-graft effect is positive in all five dense families (`+0.115` to `+0.609` nats), and leave-one-family-out dense means remain positive (`+0.274` to `+0.398`), so the result is not driven by a single family. The raw-versus-template control gives nearly identical late effects, arguing that this is not only a chat-template surface artifact.

The mirrored swap test asks whether removing late IT MLPs from an IT host also reduces the delay. It does: replacing late IT MLPs with PT MLPs produces the largest reduction of IT delayed stabilization, with a dense-family mean effect of `-0.51` nats (family-bootstrap 95% CI `[-0.83, -0.22]`) versus `-0.10` (95% CI `[-0.26, +0.03]`) early and `-0.23` (95% CI `[-0.37, -0.09]`) middle.

![Figure 3: Symmetric graft/swap localization. Left and center: late MLP substitutions have the largest convergence-gap effect in both directions, increasing the delay in a PT host and reducing it in an IT host. Right: output-relevant late-stage summaries predict the late KL shift better than the residual-opposing geometry alone. The matched random-control number in the text checks that this is not a generic late-window perturbation effect.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png)

A matched random-control follow-up argues against the simplest "late layers are fragile" explanation. Replacing the learned late IT-minus-PT MLP effect with matched random residual-projection controls gives a dense-family final-20% KL effect of `+0.003` nats (95% CI `[-0.002, +0.008]`), compared with `+0.327` (95% CI `[+0.298, +0.359]`) for the true late graft; the true-minus-random margin is `+0.324` nats (95% CI `[+0.294, +0.358]`). Thus late MLP substitutions are the only tested substitutions with a large delayed-stabilization effect, rather than merely perturbing a sensitive late window. The depth-ablation and symmetric-swap intervals above are family-bootstrap intervals over the five dense model-family means, while the random-control interval is prompt-bootstrap; Appendix A gives per-family, weight-change, geometry, and random-control details.

### 3.3 Claim 2: middle leans identity, late leans margin

Section 3.2 shows that late windows have the largest tested leverage on the delayed-stabilization signature; we now ask what aspect of next-token formation they move. At prefixes where PT and IT first choose different next tokens, which depth window changes the token identity and which changes the final margin? The first-divergence test evaluates intervened models at the shared prefix, and the companion MLP write-out test measures each window's finite-difference logit support for the PT token, the IT token, top alternatives, and residual-opposing components. All intervals in this subsection are 95% percentile bootstrap intervals over dense-family prompt-level first-divergence records.

The token-choice test gives the identity side. Under raw-shared prompting, PT with middle IT MLPs grafted in matches the IT divergent token more often than PT with late IT MLPs grafted in (`26.0%`, 95% CI `[24.5%, 27.7%]`, versus `17.6%`, 95% CI `[16.2%, 18.9%]`). The mirror comparison points the same way: IT with middle PT MLPs swapped in transfers the PT token more often than IT with late PT MLPs swapped in (`31.2%`, 95% CI `[29.6%, 32.9%]`, versus `20.8%`, 95% CI `[19.4%, 22.3%]`). These transfer rates are well below `50%`, so a single-window MLP substitution does not dominantly redirect token identity. The gap is a relative localization signal: middle windows are systematically more involved in candidate identity than late windows, not a claim that any one window alone controls the divergent token.

The margin tests show why the late window still matters. In native IT-host swaps, replacing late IT MLPs with PT MLPs produces the largest single-window IT-vs-PT margin drop (`13.25` logits, 95% CI `[12.91, 13.61]`) relative to early (`11.53`, 95% CI `[11.20, 11.88]`) and middle (`12.01`, 95% CI `[11.66, 12.35]`) swaps. In pure IT runs, MLP support for the IT divergent token is overwhelmingly late: early `-0.041` logits (95% CI `[-0.049, -0.032]`), middle `+0.021` logits (95% CI `[+0.011, +0.032]`), late `+0.789` logits (95% CI `[+0.754, +0.825]`). The PT-to-IT change in that support is also concentrated late: early `+0.034` logits (95% CI `[+0.027, +0.042]`), middle `+0.070` logits (95% CI `[+0.059, +0.080]`), late `+0.715` logits (95% CI `[+0.683, +0.747]`).

The local MLP-only proxy also shows what late MLPs are not. Adding late IT MLP updates to a PT host changes the same fixed-prefix IT-vs-PT margin by only `+0.0035` logits (95% CI `[-0.001, +0.009]`). Thus the native IT late write-out is not a portable plug-in module that works independently of the residual stream entering it.

![Figure 4: Candidate transfer and local late support. First-divergence tests show that middle substitutions transfer PT-vs-IT token identity more, while late IT MLP windows provide the strongest native IT-token support. The full late MLP update carries the IT-vs-PT margin write-in in this proxy; the residual-opposing component is a geometric marker rather than the margin vector.](../results/paper_synthesis/exp20_exp21_handoff_synthesis.png)

These first-divergence results motivate a stronger question. If late support is weak in a PT host but strong in native IT runs, is late computation merely another independent additive effect, or does it depend on the state produced upstream? Section 3.4 tests that non-independence directly.

### 3.4 Claim 3: late readout depends on upstream state

The strongest intervention test patches the full residual state at the late-window boundary and runs the downstream late stack/readout under a common PT or IT readout. Under the common IT readout, swapping in the IT late stack shifts the IT-vs-PT margin by `+0.572` logits (95% CI `[+0.494, +0.647]`) from a PT upstream state, but by `+3.207` logits (95% CI `[+3.095, +3.321]`) from an IT upstream state. The matched 2x2 decomposition gives a late-stack main effect of `+1.890` logits (95% CI `[+1.805, +1.975]`), a larger upstream-context effect of `+4.239` logits (95% CI `[+4.105, +4.376]`), and a positive interaction of `+2.635` logits (95% CI `[+2.538, +2.736]`). The common PT readout agrees (`+0.609`, 95% CI `[+0.539, +0.677]`, versus `+3.218`, 95% CI `[+3.109, +3.333]`, for IT-late given PT versus IT upstream).

This is the main non-independence result. If late layers behaved as an autonomous instruction module under this readout, the IT late stack should produce a large IT-vs-PT margin even from a PT upstream state. Instead, the effect from PT upstream is much smaller, and the 2x2 interaction is strongly positive: late IT computation is important, but its effect is much larger when the state entering it is already IT-shaped. The pooled intervals resample prompt-level units within family and average family estimates; Appendix F gives the interaction details, per-family intervals, leave-one-family-out intervals, and scope cautions. The claim is not that every family has the same magnitude, that the exact boundary is a sharp circuit boundary, or that hybridized states fully explain the natural model trajectory.

| Family | First-divergence records | IT late stack from PT upstream | IT late stack from IT upstream | IT/PT ratio | Interaction (95% CI) |
|---|---:|---:|---:|---:|---:|
| Gemma 3 4B | `600` | `+0.10` | `+6.18` | `60.5x` | `+6.08` `[+5.72, +6.44]` |
| Llama 3.1 8B | `600` | `+0.79` | `+2.05` | `2.6x` | `+1.25` `[+1.10, +1.42]` |
| Qwen 3 4B | `600` | `+0.59` | `+2.05` | `3.5x` | `+1.46` `[+1.32, +1.62]` |
| Mistral 7B | `597` | `+0.61` | `+3.15` | `5.1x` | `+2.53` `[+2.35, +2.72]` |
| OLMo 2 7B | `586` | `+0.76` | `+2.61` | `3.4x` | `+1.85` `[+1.67, +2.03]` |
| Dense-5 | `2,983` | `+0.57` | `+3.21` | `5.6x` | `+2.64` `[+2.54, +2.74]` |

The interaction is positive in all five dense families, and each per-family interaction interval excludes zero. The ratio column is descriptive only: Gemma has the largest interaction (`+6.08`), but its `60.5x` ratio mostly reflects the unusually small PT-upstream effect (`+0.10` logits) in the denominator. The interaction term is therefore the primary non-independence measure. Excluding Gemma leaves a four-family mean interaction of `+1.77` logits (95% CI `[+1.69, +1.86]`) and a `3.57x` IT-upstream/PT-upstream ratio; all leave-one-family-out interaction estimates remain positive, ranging from `+1.77` to `+2.98` logits with intervals above zero. The structural direction is cross-family, while magnitude is family-dependent.

A label-swap control tests whether this compatibility is aligned with the PT/IT labels rather than an arbitrary sign convention. Re-scoring each cell by the token favored by its late stack, the IT late stack gains `+5.56` logits from IT-shaped rather than PT-shaped upstream state, while the PT late stack gains `+2.92` logits from PT-shaped rather than IT-shaped upstream state; their difference is the same `+2.64` interaction. A permutation that preserves each prompt's four factorial cell values but randomly swaps the PT/IT label orientation gives a null centered near zero (99.9th percentile `+0.239` logits), with the observed amplification far outside the null (`p=5.0e-5`, the minimum resolvable with `20,000` permutations). Thus the result is not merely that late layers depend on upstream context; the PT-to-IT contrast is label-aligned.

A descriptive subgroup pass leaves the interaction positive in every reportable stratum of the primary residual-state holdout slice. It is positive across the three prompt categories present in that slice (`GOV-CONV` `+2.05`, `GOV-FORMAT` `+3.61`, `SAFETY` `+2.83` logits), across collapsed IT-token categories (`CONTENT` `+2.50`, `FORMAT` `+2.60`, `FUNCTION_OTHER` `+2.81`), and for non-assistant-marker events (`+2.48`). A separate content/reasoning extension also preserves the central interaction (`+1.68` logits, 95% CI `[+1.59, +1.78]`), including factual prompts (`CONTENT-FACT` `+1.84`) and reasoning prompts (`CONTENT-REASON` `+1.34`). The decomposition is domain-dependent: in that extension, the IT late stack from PT upstream moves against the IT token (`-1.41` logits), so this result is consistent with context-gated non-autonomy rather than an invariant positive late-module effect.

The interaction term is not negligible relative to the main effects. Under the common-IT readout, dividing the three positive coefficient magnitudes (`4.24 + 1.89 + 2.64 = 8.77` logits), the upstream-context main effect accounts for about `48%` of the positive coefficient-magnitude mass, the late-stack main effect for `22%`, and the upstream-by-late interaction for `30%`. This is a coefficient-magnitude scale check, not a variance decomposition; the qualitative point is that the interaction is comparable in magnitude to the late-stack main effect. The result does not rule out compatibility learned jointly by upstream and late computation; it argues against analyzing the IT late stack as an independent module whose effect is approximately unchanged across upstream states. Distinguishing "late computation actively reasons about upstream state" from "late computation is approximately a function of upstream state" requires feature-level mediation inside the upstream state, which we leave to future work (Section 5).

![Figure 5: Headline result: late computation is not autonomous under this readout. Residual-state x late-stack factorial at first-divergence prefixes. The same IT late stack shifts the IT-vs-PT margin by `+0.57` logits from a PT upstream state but `+3.21` logits from an IT upstream state, a `5.6x` asymmetry. Under this window-level measurement, the observed asymmetry argues against a simple additive/autonomous account. Under common-IT readout, the upstream main effect is `+4.24`, the late-stack main effect is `+1.89`, and the interaction is `+2.64` logits. The right panel shows family-level interaction CIs; interaction, not the IT/PT ratio, is the inferential quantity. A label-swap control preserving each prompt's four cells puts the observed compatibility amplification beyond the null 99.9th percentile (`+0.239` logits; `p=5.0e-5`). The positive interaction quantifies the failure of additivity: upstream state and late computation have to be modeled jointly for this readout. Both common-PT and common-IT readouts agree.](../results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/exp23_midlate_interaction.png)

The synthesis is the paper's headline. At the granularity tested here, middle windows are more diagnostic of which PT- or IT-like token candidate is exposed, while late MLP and late-stack windows interact with the upstream residual state to shape the measured final margin.

The implication is non-modularity under the tested interventions: in the primary holdout, the same late computation is positive yet much smaller from a PT upstream state and strongest when the residual stream is already IT-shaped; in the factual/reasoning extension, the late-only term from PT upstream is negative even though the interaction remains positive. Mid and late windows must be modeled jointly under this readout; neither in isolation captures the PT/IT next-token disagreement.

The convergence gap is therefore not where we claim instruction-tuned behavior originates. It is the observational shadow of context-gated late readout operating on an upstream state that already encodes much of the IT-vs-PT decision. Appendices A and F give the token-flow chronology, residual-opposing component decomposition, and mid+late interaction details.

### 3.5 Behavioral consistency check

The internal intervention claim does not require behavioral support, but the same intervention family changes natural-decoding outputs in the predicted direction. Under LLM judging on the dense-family 600-prompt behavioral subset, replacing late IT MLPs with PT MLPs produces the strongest IT-side degradation on assistant register (`+0.42` worsening, 95% CI `[+0.27, +0.62]`) and a smaller degradation on benign false-refusal behavior (`+0.05`, 95% CI `[+0.02, +0.10]`). Grafting late IT MLPs into PT produces weaker but measurable movement: blind pairwise judging prefers PT with late IT MLPs grafted in over the PT baseline on `56.3%` of resolved assistant-register items (95% CI `[53.3%, 59.5%]`) and `63.8%` of resolved benign-safety items (95% CI `[58.6%, 69.1%]`). A completed two-rater blind human audit agrees directionally on the four primary pairwise contrasts, while high tie/both-bad rates and heterogeneous agreement keep the claim bounded: pairwise Cohen's kappa ranges from `0.01` to `0.52` across the four-label contrasts and from `0.00` to `0.48` after collapsing unresolved labels. Appendix B gives the judge design, human-audit tables, and kappa caveats.

## 4. Related Work and Discussion

Prior work gives the vocabulary for late-stage dependence. Transformer-circuits work treats the residual stream as a communication channel through which later layers read information written earlier (Elhage et al., 2021). FFN analyses view layer outputs as memory- or vocabulary-space updates later refined through the residual stream (Geva et al., 2022a,b). Logit-lens and tuned-lens work make layerwise prediction refinement visible (nostalgebraist, 2020; Belrose et al., 2023); DoLA exploits differences between earlier and later layer logits (Chuang et al., 2024); Lad et al. (2024) frames late computation as residual sharpening after earlier candidate construction; Joshi et al. (2025) studies late confidence correction after decision certainty has emerged; Zhao, Ziser, and Cohen (2024) locate transitions from general to task-oriented representations; and Bigoulaeva et al. (2026) describes later information pathways selected by earlier task representations. These accounts make upstream dependence plausible and often implicit. They do not, however, estimate a paired PT/IT upstream-state x late-computation interaction at the token where the checkpoints first disagree.

Our contribution is that controlled measurement. Under the first-divergence residual-state x late-stack factorial, the same IT late stack produces a `+3.21` logit shift from IT upstream state versus `+0.57` from PT upstream state in the primary holdout, while a factual/reasoning extension preserves the interaction but flips the late-only PT-upstream term negative. The contribution is therefore not the discovery that late layers can use upstream state, nor an invariant positive late-only module. It is the measurement that the late stage's effect is conditional on upstream state in these hybrid passes, with an explicit interaction term, per-family intervals, matched random falsifiers, and label-swap controls. Late post-training computation is functionally non-autonomous under this first-divergence measurement.

The residual-opposing geometry supports this interpretation, but only as interpretation. Late IT MLP updates have a substantial component opposing the residual stream entering them, a geometry consistent with revision or reconciliation rather than simple additive accumulation. However, the residual-opposing component is not itself the margin write-in under our proxy: it contributes `-0.0046` logits to the IT-vs-PT margin (95% CI `[-0.009, -0.001]`), while the full late MLP update contributes `+0.768` logits (95% CI `[+0.729, +0.805]`). We therefore do not claim that residual opposition is the mechanistic source of late sharpening. The geometry is consistent with revision, but does not by itself write the IT-vs-PT margin.

Post-training model-diffing and activation-patching papers ask nearby questions. Wu et al. (2024) studies instruction-conditioned behavioral shift, Du et al. (2025) compares base and post-trained models across knowledge, truthfulness, refusal, and confidence, and Prakash et al. (2024) uses cross-model activation patching to study entity tracking. We use related intervention logic, but the target is different: natural PT/IT next-token disagreement under matched-prefix control. This lets us ask whether post-training changes which token candidate is exposed, whether it changes the final margin, or whether the two effects interact.

Instruction-vector and layer-localization work is especially close to our interpretation. Bigoulaeva et al. (2026) identify instruction vectors and argue that later information pathways are selected conditional on earlier task representations. Zhao, Ziser, and Cohen (2024) study where multi-task information appears across layers in instruction-tuned models, while Nepal et al. (2025) show that layer importance for mathematical reasoning can remain stable across post-training methods. These results make it unsurprising that post-training effects are not confined to a single late module. Our contribution is narrower and more controlled: we condition on the first natural PT/IT next-token disagreement, hold histories fixed, and ask which depth interventions move token identity versus IT-vs-PT margin.

Behavioral-direction papers explain why the context-gated late readout effect should not be confused with a complete assistant module. Arditi et al. (2024), Lu et al. (2026), Stolfo et al. (2024), Turner et al. (2023), Zou et al. (2023), and Panickssery et al. (2024) show that activation-space directions can control refusal, assistant persona, instruction following, or high-level behavior. Li et al. (2025) and Chaudhury (2025) also find safety or preference effects in middle bands. Our first-divergence result is compatible with that picture rather than a foil to it: middle windows are more tied to which behaviorally relevant token candidate is selected, while late MLP and late-stack windows are where our readouts find the clearest context-gated late readout.

Taken together, the evidence supports a depth-window account: IT models stabilize later to their own final predictions, late MLP substitutions have the largest tested leverage on that delayed-stabilization signature, and first-divergence tests show middle-sensitive candidate transfer plus context-gated late readout. The paper does not claim to recover a named middle-layer feature feeding a named late write-in circuit. It identifies a cross-family route decomposition that future circuit work can make more surgical.

## 5. Limitations and Next Tests

The causal claim is window-level and intervention-scoped, not circuit-level or a complete natural-model explanation. MLP grafts and swaps estimate causal leverage on specified readouts in hybrid forward passes, and the residual-state x late-stack factorial tests non-additivity between an upstream state and downstream late computation, but neither identifies the underlying features, heads, or MLP directions. Hybrid grafts can create off-manifold states, so we treat matched histories, random controls, label-swap controls, and raw-shared checks as safeguards rather than as proof that every intervened state is natural. The early/middle/late windows are coarse stage probes and can overlap; Appendix F reports the exact boundaries and explains why the result should not be read as a sharp layer-index boundary. The directly supported claim is candidate-transfer plus context-gated late readout under these windows.

The convergence gap is endpoint-relative by design: it asks how far each layer is from the same model's eventual next-token prediction. Endpoint-matched controls show that the measured late KL gap remains after balancing final entropy, confidence, and top-1/top-2 margin, while same-history JS replay removes the own-final endpoint from a companion readout. These controls do not exhaust all final-distribution properties; the main claims rest on the conjunction of endpoint matching, same-history replay, graft/swap localization, matched random controls, and first-divergence token tests.

The empirical scope is five dense 4B-8B families. This breadth is unusual for mechanistic model diffing, but it is not frontier-scale validation and it excludes non-dense claims from the main pool. An additional non-dense side run is retained only in Appendix A6 because those grafts also perturb routing and expert selection. Larger dense models and a second non-dense family are natural external-validity tests.

The behavioral evidence is a consistency check, not a standalone benchmark. LLM judging and the completed human audit agree directionally on the primary pairwise contrasts, but high tie/both-bad rates and heterogeneous κ keep the behavioral result from carrying the mechanistic claim. The internal intervention result does not depend on those labels.

The residual-state x late-stack factorial measures non-additivity but leaves the mechanism inside the upstream state unresolved. It shows that IT-shaped upstream residual state and IT late computation interact strongly on the first-divergence margin; it does not identify which upstream features, attention heads, or MLP directions carry that state. The next circuit-level test is therefore not another coarse window swap, but a feature- or subspace-level mediation test inside the middle-to-late handoff, which we leave to future work.

We also do not compare directly to sparse-crosscoder model diffing (Lindsey et al., 2024; Minder et al., 2025). Crosscoders and our protocol address complementary questions: crosscoders ask which features differ between models, while first-divergence factorials ask how depth windows cooperate at the token where paired checkpoints first disagree. A natural follow-up is to train a BatchTopK crosscoder at the late-window boundary and test whether IT-specific features mediate the upstream x late-stack interaction.

---

## References

Aghajanyan, A., et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL 2021*. arXiv:2012.13255.

Ansuini, A., et al. (2019). Intrinsic Dimension of Data Representations in Deep Neural Networks. *NeurIPS 2019*. arXiv:1905.12784.

Arditi, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. *arXiv:2406.11717*.

Belrose, N., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. *COLM 2024*. arXiv:2303.08112.

Bigoulaeva, I., Rohweder, J., Dutta, S., & Gurevych, I. (2026). Patches of Nonlinearity: Instruction Vectors in Large Language Models. *arXiv:2602.07930*.

Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models with Dictionary Learning. *Anthropic*.

Chaudhury, A. (2025). Alignment is Localized: A Causal Probe into Preference Layers. *arXiv:2510.16167*.

Cheng, E., Doimo, D., Kervadec, C., Macocco, I., Yu, J., Laio, A., & Baroni, M. (2024). Emergence of a High-Dimensional Abstraction Phase in Language Transformers. *ICLR 2025*. arXiv:2405.15471.

Chuang, Y., et al. (2024). DoLA: Decoding by Contrasting Layers Improves Factuality. *ICLR 2024*. arXiv:2309.03883.

Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS 2023*. arXiv:2304.14997.

Cui, G., et al. (2025). The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models. *arXiv:2505.22617*.

Du, H., Li, W., Cai, M., Saraipour, K., Zhang, Z., Lakkaraju, H., Sun, Y., & Zhang, S. (2025). How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence. *COLM 2025*. arXiv:2504.02904.

Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. B. (2024). Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. *COLM 2024*. arXiv:2404.04475.

Dunefsky, J., et al. (2024). Transcoders Find Interpretable LLM Feature Circuits. *arXiv:2406.11944*.

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

Hewitt, J., Liu, N. F., Liang, P., & Manning, C. D. (2024). Instruction Following without Instruction Tuning. *arXiv:2409.14254*.

Jain, S., Lubana, E. S., Oksuz, K., Joy, T., Torr, P. H. S., Sanyal, A., & Dokania, P. K. (2024). What Makes and Breaks Safety Fine-tuning? A Mechanistic Study. *NeurIPS 2024*. arXiv:2407.10264.

Joad, F., et al. (2026). There Is More to Refusal in Large Language Models than a Single Direction. *arXiv:2602.02132*.

Joshi, A., Ahmad, A., & Modi, A. (2025). Calibration Across Layers: Understanding Calibration Evolution in LLMs. *EMNLP 2025*. arXiv:2511.00280.

Lad, F., et al. (2024). The Remarkable Robustness of LLMs: Stages of Inference? *arXiv:2406.19384*.

Levelt, W. J. M. (1989). *Speaking: From Intention to Articulation*. MIT Press.

Li, S., Yao, L., Zhang, L., & Li, Y. (2025). Safety Layers in Aligned Large Language Models: The Key to LLM Security. *ICLR 2025*. arXiv:2408.17003.

Lin, B. Y., et al. (2024). The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning. *ICLR 2024*. arXiv:2312.01552.

Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., & Olah, C. (2024). Sparse Crosscoders for Cross-Layer Features and Model Diffing. *Transformer Circuits Thread*.

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment. *EMNLP 2023*. arXiv:2303.16634.

Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. *arXiv:2601.10387*.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS 2022*. arXiv:2202.05262.

Minder, J., Dumas, C., Juang, C., Chughtai, B., & Nanda, N. (2025). Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning. *NeurIPS 2025*. arXiv:2504.02922.

Nanda, N., & Lieberum, T. (2022). A Mechanistic Interpretability Analysis of Grokking. *ICLR MATH-AI Workshop 2023*.

Nepal, A., Shrestha, S., Shrestha, A., Kim, M., Naghiyev, J., Shwartz-Ziv, R., & Ross, K. (2025). Layer Importance for Mathematical Reasoning is Forged in Pre-Training and Invariant after Post-Training. *arXiv:2506.22638*.

nostalgebraist. (2020). interpreting GPT: the logit lens. *LessWrong*.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*. arXiv:2203.02155.

Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M. (2024). Steering Llama 2 via Contrastive Activation Addition. *ACL 2024*. arXiv:2312.06681.

Panigrahi, A., Saunshi, N., Zhao, H., & Arora, S. (2023). Task-Specific Skill Localization in Fine-tuned Language Models. *ICML 2023*. arXiv:2302.06600.

Park, K., Choe, Y. J., & Veitch, V. (2023). The Linear Representation Hypothesis and the Geometry of Large Language Models. *arXiv:2311.03658*.

Prakash, N., Shaham, T. R., Haklay, T., Belinkov, Y., & Bau, D. (2024). Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking. *ICLR 2024*. arXiv:2402.14811.

Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *NeurIPS 2023*. arXiv:2305.18290.

Rocchetti, E., & Ferrara, A. (2026). How LLMs Follow Instructions: Skillful Coordination, Not a Universal Mechanism. *arXiv:2604.06015*.

Saxe, A. M., et al. (2018). On the Information Bottleneck Theory of Deep Learning. *ICLR 2018*.

Shwartz-Ziv, R., & Tishby, N. (2017). Opening the Black Box of Deep Neural Networks via Information. *arXiv:1703.00810*.

Singh, A., et al. (2024). Representation Surgery: Theory and Practice of Affine Steering in Language Models. *arXiv:2402.09631*.

Song, Y., et al. (2025). Bridging the Dimensional Chasm: Uncover Layer-wise Dimensional Reduction in Transformers through Token Correlation. *arXiv:2503.22547*.

Stolfo, A., et al. (2024). Improving Instruction-Following in Language Models through Activation Steering. *arXiv:2410.12877*.

Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Anthropic*.

Turner, A. M., et al. (2023). Steering Language Models With Activation Engineering. *arXiv:2308.10248*.

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019). Best Practices for the Human Evaluation of Automatically Generated Text. *INLG 2019*.

Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., & Sui, Z. (2023). Large Language Models Are Not Fair Evaluators. *ACL 2024*. arXiv:2305.17926.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small. *ICLR 2023*. arXiv:2211.00593.

Wei, R., Du, R., Yu, H., Tiwari, D., Li, J., Xu, Z., & Wang, H. (2026). The Diminishing Returns of Early-Exit Decoding in Modern LLMs. *arXiv:2603.23701*.

Wu, X., Yao, W., Chen, J., Pan, X., Wang, X., Liu, N., & Yu, D. (2024). From Language Modeling to Instruction Following: Understanding the Behavior Shift in LLMs after Instruction Tuning. *NAACL 2024*. arXiv:2310.00492.

Xu, Z., et al. (2025). Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning. *NeurIPS 2025*. arXiv:2502.07154.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Li, D., Gonzalez, J. E., Xing, E. P., Zhang, H., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks*. arXiv:2306.05685.

Zhao, Z., Ziser, Y., & Cohen, S. B. (2024). Layer by Layer: Uncovering Where Multi-Task Learning Happens in Instruction-Tuned Large Language Models. *EMNLP 2024*. arXiv:2410.20008.

Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.

---

## Appendix A: Supplementary Evidence Map

This appendix keeps the full diagnostic surface, grouped by the role each artifact plays in the argument.

### A1: Geometry and weight-change controls

[Figure S17: Generation-step × layer heatmap for Gemma 3 4B. Four panels showing δ-cosine stability across generation steps.](figures/it_plot10_generation_heatmap.png)
[Figure S18: Per-layer weight change localization (PT -> IT) across the five dense families plus a separate DeepSeek MoE side case. Gemma shows late-layer concentration; the other dense families show more diffuse changes.](../results/exp09_cross_model_observational_replication/plots/L3_weight_diff_6panel.png)

[Figure S20: δ-cosine profiles across the five dense families plus a separate DeepSeek MoE side case. IT (solid) vs PT (dashed). Gemma shows the largest late IT-vs-PT shift; Llama shows the weakest sustained shift because its PT variant already exhibits substantial late residual-opposing geometry.](../results/exp09_cross_model_observational_replication/plots/L1_delta_cosine_6panel.png)

**Table S18a: depth-ablation effect normalized by window weight-change proxy.** `Δ KL` is the matched-prefix PT-side final-20% `B_window - A'` effect. `Mean ΔW` is the mean per-layer MLP weight-change proxy from Figure S18 over the same graft window. `Δ KL / Mean ΔW` is a descriptive scale-normalized diagnostic, not a formal parameter-efficiency estimand. Late is the largest raw effect and the largest normalized effect in every family, while the largest weight-change window is not consistently late.

| Family | Early ΔKL / Mean ΔW | Mid ΔKL / Mean ΔW | Late ΔKL / Mean ΔW | Largest ΔW window |
|---|---:|---:|---:|---|
| Gemma 3 4B | `-5.5` | `-14.0` | `146.6` | Mid |
| Llama 3.1 8B | `-236.7` | `-148.9` | `453.2` | Late |
| Qwen 3 4B | `-38.8` | `-60.1` | `184.4` | Late |
| Mistral 7B | `878.2` | `736.2` | `1754.4` | Mid |
| OLMo 2 7B | `0.6` | `27.1` | `130.3` | Mid |
| DeepSeek-V2-Lite | `-11.4` | `-55.1` | `234.7` | Late |

The dense-family mean raw effect is `+0.341` nats late versus `-0.035` early and `-0.045` mid, while mean window weight-change is nearly identical for middle and late (`0.00180` vs `0.00179`). Thus the late intervention effect is not explained by a systematically larger late MLP weight delta in the dense-family pool.

**Table S18b: per-family raw late effect and leave-one-family-out sensitivity.** Mistral is extreme only on the weight-normalized ratio because its `Mean ΔW` denominator is unusually small; it is not an outlier in raw late effect and does not drive the dense-family mean.

| Family | Raw late ΔKL | Late Mean ΔW | Late ΔKL / Mean ΔW | Dense-4 mean if removed |
|---|---:|---:|---:|---:|
| Gemma 3 4B | `+0.609` | `0.004156` | `146.6` | `+0.274` |
| Llama 3.1 8B | `+0.310` | `0.000685` | `453.2` | `+0.349` |
| Qwen 3 4B | `+0.491` | `0.002661` | `184.4` | `+0.304` |
| Mistral 7B | `+0.115` | `0.000066` | `1754.4` | `+0.398` |
| OLMo 2 7B | `+0.181` | `0.001392` | `130.3` | `+0.381` |
| Dense-5 mean | `+0.341` | — | — | — |

The leave-one-family-out means remain positive in every case. Removing Mistral increases the dense-family mean, so the pooled raw late-graft effect is not Mistral-driven. The large Mistral normalized value instead reflects a small absolute PT→IT MLP RMS weight-change denominator (`6.6e-5`) paired with a modest positive raw effect.

[Figure S21: Cross-model δ-cosine heatmaps. Full layer x generation-step heatmaps for the five dense families plus the DeepSeek side case (PT and IT side by side), showing the distribution of residual-opposing MLP geometry across the full forward pass.](../results/exp09_cross_model_observational_replication/plots/L1_heatmaps_6x2.png)

### A2: Feature-level supplements

[Figure S23: Feature importance analysis. Per-feature contribution to late post-training computation at layers 20–33, showing the distribution of importance across transcoder features.](../results/exp03_corrective_stage_characterization/plots/plot_e3_11_feature_importance.png)

[Figure S24: Feature population dynamics. Gini coefficient and N50 distributions for IT vs PT at late layers, quantifying the broadening of the active feature repertoire.](../results/exp03_corrective_stage_characterization/plots/plot_feature_populations.png)

### A3: Probe and commitment robustness

[Figure S25: Tuned-lens validation. KL(layer ell || final) for the five dense PT variants plus the DeepSeek side case. Red = tuned logit lens, blue = raw logit lens. The tuned lens substantially reduces KL at intermediate layers for Llama, Qwen, Mistral, and OLMo, with the DeepSeek side case behaving similarly. Gemma improves only modestly at comparable depth, indicating weaker probe quality for its hybrid local/global attention architecture rather than total probe failure. We therefore report both tuned and raw results throughout, and interpret Gemma's tuned-lens thresholded metrics with extra caution.](../results/exp09_cross_model_observational_replication/plots/tuned_lens_validation_kl_to_final.png)

**Table S25a: raw-vs-tuned sensitivity for the convergence-gap discovery metric.** Values are mean IT-minus-PT `KL(layer || own final)` differences from the existing cross-family discovery summaries. The raw lens does not remove the dense-family headline effect: the dense-5 final-half convergence gap is larger under raw lens (`0.771`) than tuned lens (`0.410`), and Gemma's own raw final-half gap (`1.008`) is larger than its tuned value (`0.351`). Excluding Gemma leaves the dense-family late-half raw gap positive (`0.712`), so the dense-family late stabilization result is not driven by Gemma's weak tuned-lens probe. The all-run row with DeepSeek is reported only as an appendix side case. This table is a sensitivity check for the discovery readout, not a raw-only rerun of every matched-prefix KL intervention; the matched-prefix JS replay and first-divergence token projections are the non-tuned companion evidence for those later claims.

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

[Figure S26: KL-to-final trajectories in Gemma 3 4B. IT (solid) shows elevated KL-to-final at late layers (20–33), converging to the 0.1 nat threshold later than PT (dashed).](../results/exp03_corrective_stage_characterization/plots/plot6_kl_trajectory.png)

[Figure S27: Mind-change analysis in Gemma 3 4B. Per-layer mind-change rates by token category. IT's late layers (20–33) show a sharp spike in mind-changes, with many targeting structural and discourse tokens.](../results/exp03_corrective_stage_characterization/plots/plot_e3_10_mind_change.png)

[Figure S28: Adjacent-layer KL divergence in Gemma 3 4B. IT (solid red) shows three discrete revision phases: early (layers 5–6), mid (15–17), and late (27–28), while PT (dashed blue) shows lower and more uniform prediction revision across layers.](../results/exp03_corrective_stage_characterization/plots/plot_e3_12_adjacent_layer_kl.png)

[Figure S29: Candidate reshuffling in Gemma 3 4B. Number of unique top-1 candidates encountered up to each layer. IT (red) shows rapid expansion in late layers; PT (blue) stabilizes earlier.](../results/exp03_corrective_stage_characterization/plots/plot_e3_13_candidate_reshuffling.png)

[Figure S33: Alignment tax localization in Gemma 3 4B. Fraction of total activation mass allocated to IT-amplified features by layer depth. Late layers (20–33) show 14–16% of activation mass at layers 28–33.](../results/exp03_corrective_stage_characterization/plots/plot5_alignment_tax.png)

[Figure S33b: Raw vs tuned logit lens commitment scatter. Per-step top-1 commitment layer under raw (x-axis) vs tuned (y-axis) logit lens. Points below the diagonal indicate tuned lens commits earlier (i.e., the tuned lens reveals earlier convergence that the raw lens misses). For most models, the tuned lens detects commitment at earlier absolute layers — consistent with its more faithful intermediate predictions — while preserving the IT > PT ordering.](../results/exp09_cross_model_observational_replication/plots/L2_raw_vs_tuned_scatter.png)

[Figure S34: Alternative commitment definitions. Commitment delay under majority-vote (≥90% subsequent layers KL < 0.1) for tuned and raw logit lens. The delay pattern replicates under this more conservative definition.](../results/exp09_cross_model_observational_replication/plots/L2_commitment_tuned_majority_0.1.png)

[Figure S35: KL threshold sensitivity (full). Mean commitment vs KL threshold τ for both tuned (red) and raw (blue) lenses. The IT–PT gap is consistent across thresholds from 0.05 to 1.0 nats.](../results/exp09_cross_model_observational_replication/plots/L2_pure_kl_threshold_sensitivity.png)

[Figure S36: Cosine and entropy commitment. Commitment defined via cosine similarity (cos(h_ℓ, h_final) > 0.95) and entropy convergence (|H_ℓ − H_final| < 0.2). These representation-space metrics show minimal IT–PT difference, establishing that the convergence gap is a logit-space phenomenon.](../results/exp09_cross_model_observational_replication/plots/L2_commitment_cosine_0.95.png)

[Figure S37: Commitment CDF by normalized depth. Cumulative distribution of commitment layers for PT (dashed) and IT (solid), four methods. The rightward shift of IT CDFs is visible under KL-based metrics but absent under top-1 for some models — confirming the delay is distributional, not merely an argmax effect.](../results/exp09_cross_model_observational_replication/plots/L2_commitment_cdf_4methods.png)

[Figure S38: Endpoint-matched convergence-gap check. Token steps are matched within `model x probe_family` on final-layer entropy, final top-1 confidence, and final top-1/top-2 margin. IT retains a higher late `KL(layer || own final)` under raw and tuned probes, and endpoint-free path metrics also remain positive.](../results/paper_synthesis/exp22_endpoint_deconfounded_summary.png)

**Table S38: endpoint-matched convergence-gap control.** Dense-family endpoint-control run over 600 prompts per PT/IT branch. Matching is coarsened exact matching within `model x probe_family` on final entropy, final confidence, and final top-1/top-2 margin.

| Quantity | Estimate |
|---|---:|
| Raw-probe late `KL(layer || own final)`, IT - PT | `+0.425` nats, 95% CI `[+0.356, +0.493]` |
| Tuned-probe late `KL(layer || own final)`, IT - PT | `+0.762` nats, 95% CI `[+0.709, +0.814]` |
| Remaining adjacent JS after endpoint matching, IT - PT | `+0.052`, 95% CI `[+0.048, +0.057]` |
| Future top-1 flips after endpoint matching, IT - PT | `+0.203`, 95% CI `[+0.190, +0.215]` |
| Minimum matched-token retention across model/probe branches | `0.796` |
| Maximum post-match endpoint-covariate SMD | `0.057` |
| Maximum malformed branch rate | `0.000` |

### A4: Matched-prefix localization and behavioral consequence figures

[Figure S41: Matched-prefix MLP graft trajectories across the five dense families plus a separate DeepSeek-V2-Lite MoE case. The intact IT model generates freely; the PT teacher-forced control and grafted PT branches are then forced to follow the same continuation. Solid lines show the raw-prompt branch and dashed lines the chat-template branch. The graft consistently reduces cross-KL to the IT teacher while reproducing the residual-opposing δ-cosine signature only partially in the dense-model pool.](../results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_400rand_v11_teacherforced/overview_trajectories.png)

[Figure S42: PT-side graft depth ablation. Equal-width early, middle, and late IT-MLP grafts are compared under identical teacher-forced token histories in a PT host. The late graft is the only window that consistently induces the final-window convergence-gap increase across the five dense families.](../results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/depth_ablation_paper_main.png)

[Figure S43: Free-running A/B/C output evaluation overview across all four judged metrics. A = PT raw, B = PT + late IT MLP graft under the same raw prompt, C = full IT model with its native chat template. B moves consistently toward C on benign false-refusal reduction, more selectively on assistant register, and only weakly on broad structure and harmful-prompt refusal.](../results/exp12_free_running_abc_graft/plots/exp12_eval_v1_20260413_v3/exp12_scores_overview.png)

[Figure S44: Improvement relative to the PT baseline in the free-running A/B/C evaluation. Red = B − A, green = C − A. Positive bars indicate improvement for G1, G2, and S1; for S2, positive bars indicate a reduction in false refusal. The graft consistently captures part of the A→C gap, but remains well short of the full IT endpoint on most metrics.](../results/exp12_free_running_abc_graft/plots/exp12_eval_v1_20260413_v3/exp12_delta_vs_a.png)

[Figure S45: Cross-family descriptive token-type analysis of the matched-prefix late stage. Left: displaced vs supported token classes under `A' -> B_late`. Center: teacher-token rank gain by collapsed token type. Right: token-type rank gain under early, middle, and late graft windows on the subset with recoverable raw depth traces. The late stage broadly supports the eventual teacher token and suppresses `FUNCTION/OTHER` raw-continuation-style alternatives, with a secondary formatting/discourse component.](../results/exp13_late_stage_token_support_analysis/exp13A_lite_20260415_live/exp13a_lite_paper_main.png)

[Figure S46: Descriptive token-support appendix view. Per-model panels, candidate entry/exit distributions, and mind-change summaries for the matched-prefix token-type analysis.](../results/exp13_late_stage_token_support_analysis/exp13A_lite_20260415_live/exp13a_lite_appendix.png)

[Figure S47: Symmetric matched-prefix graft/swap summary. Left: PT-side late-region KL deltas for early, middle, and late IT-MLP grafts relative to `A'`. Center: IT-side late-region KL deltas for early, middle, and late PT-MLP swaps relative to `C`. Right: dense-family predictive correlations for output-relevant late-stage summaries (`support_teacher`, `anti_top1`, `anti_kl_final`) and `δ`-cosine.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png)

[Figure S48: Symmetric graft/swap appendix view. Per-model bidirectional window-effect panels and late-stage mechanism summaries for the matched-prefix graft/swap analysis.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_appendix.png)

[Figure S48b: Matched random-control specificity check. Actual IT graft deltas are compared with matched random residual-projection controls on final-20% KL-to-own-final. The dense-family late true effect is large while the matched random late effect is near zero, ruling out a generic same-window perturbation account for the main late KL result.](../results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/exp19B_final20_kl_true_vs_random.png)

[Figure S49: Assistant-facing bucket deltas in the free-running LLM-judge behavioral follow-up. Dense-family pooled `G2` deltas by prompt bucket for PT-side grafts and IT-side swaps, highlighting that the largest late judge-rated degradation on the IT side concentrates on conversational and register-sensitive prompts.](../results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_paper_it_targeting.png)

### A5: Same-history JS and candidate/amplification details

[Figure S50: Matched-prefix native same-layer JS divergence under identical teacher tokens. Per-model `JS(A', C)` curves and dense-family pooled summaries show that broad PT↔IT output divergence is already present through much of the stack and amplifies late even when teacher histories are frozen.](../results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/exp16_js_appendix_models.png)

[Figure S51: Matched-prefix JS control view. PT-side and IT-side target-gap closure bars and host-local perturbation controls under matched prefix show that direct same-layer gap closure is more mid-to-late distributed than purely late, motivating the paper's “broad circuit, late delayed-stabilization window” synthesis rather than a strictly late-confined story.](../results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/exp16_js_appendix_controls.png)

[Figure S52: Reverse teacher-stream JS check. Replaying the same matched-prefix native-JS analysis with PT-generated continuations as teacher tokens again shows a broad dense-family PT↔IT same-layer JS gap. Late amplification is teacher-stream-dependent under token-step weighting, while prompt-mean aggregation still rises late; this supports the broad identical-history divergence claim while arguing against a strict teacher-stream-invariant late-confined interpretation. The Llama reverse replay excludes 11 empty PT-teacher continuations.](../results/exp16_matched_prefix_js_gap/exp16_js_reverse_pt_teacher_20260422_165259/plots/exp16_teacher_direction_comparison.png)

[Figure S53: Token identity at the first PT/IT divergent token. Under the raw-shared prompt, middle swaps transfer opposite-model token identity more than late swaps, while native prompting shows the deployment-format counterpart.](../results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/exp20_token_identity_dense5_ci.png)

[Figure S54: Mid-vs-late IT-token margin effects. Late windows dominate IT-vs-PT token-margin changes, especially under the native IT chat template.](../results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/exp20_mid_late_margin_dense5_ci.png)

[Figure S55: Raw-shared per-model token-transfer heatmap. Across dense families, middle swaps generally transfer token identity more than late swaps, with DeepSeek reported separately as the MoE case in all-model outputs.](../results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/exp20_raw_shared_model_transfer_heatmap.png)

[Figure S56: Native PT/IT final-token margin flow. Dense-5 pooled mid and late window margin deltas for the finally emitted token, plus dense-family mid-selected/late-helped rates and per-model IT `FORMAT` late-minus-mid margin. Late IT windows produce the strongest margin gains, especially for `FORMAT` and `CONTENT`, while PT is flatter.](figures/exp18_pure_flow_overview.png)

[Figure S57: Matched-prefix candidate/amplification chronology. Left: teacher-token rank gain by disjoint window under identical histories. Middle: strict rate at which a token is first selected in the middle window and then further helped late. Right: continuity view of `A' -> B_window` top-1 displacement. Format-like tokens become teacher-rank-positive only late, while content shows larger middle-window gains.](figures/exp18_handoff_summary.png)

### A6: DeepSeek/MoE side case

DeepSeek-V2-Lite is retained only as a descriptive side case wherever the experiments were run. It is not pooled into the main dense-family intervention, behavioral, endpoint-control, or first-divergence claims. We did not add a second MoE model or analyze router/expert-selection mechanisms, so the paper makes no claim about MoE generalization. The reason is methodological rather than cosmetic: in a MoE checkpoint, an MLP graft can change both expert computation and routing/expert selection, which is not directly comparable to dense MLP substitutions without additional controls.

## Appendix B: Evaluation Methodology

![Figure B1: Free-running behavioral consequence check. The same late intervention family produces measurable LLM-judge behavioral movement under natural decoding, but the figure is appendix material because behavior is a consistency check rather than the core internal localization evidence.](../results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_paper_behavior_main.png)

### B1: LLM judge design and rubric definitions

The behavioral run uses four pointwise judge tasks and two blind pairwise judge tasks. `G1` is a 1-5 score for response structure and formatting, independent of factual correctness. `G2` is a 1-5 score for assistant-like conversational register versus raw web-text continuation, applied to assistant-facing prompts. `S1` classifies harmful-request responses as `REFUSE`, `COMPLY`, or `INCOHERENT`; incoherence is not counted as refusal. `S2` is a binary benign false-refusal metric, where 1 means the model refused, over-hedged, or failed to engage with a safe request. Pairwise `G2` and `S2` ask which of two anonymized responses better satisfies the same criterion, with randomized A/B ordering and a `TIE` option.

Pointwise judging uses a bulk model (`google/gemini-2.5-flash`) and a second model (`openai/gpt-4o-mini`) through the OpenRouter-compatible interface. The manifest also records an escalation model (`openai/gpt-4o`), parser versions, rubric hashes, dataset hash, pipeline hash, and output hash. Judge calls use temperature 0, are parsed into fixed schemas, and are retried on parser failure. Pointwise items are escalated if the two judge models differ by at least 2 points on `G1/G2`, disagree on `S1`, or disagree on `S2`. Pairwise judging uses the bulk model, is blind to model family and condition names, and uses randomized A/B presentation for `B_late` versus `A`, `C` versus `D_late`, and `C` versus `A`.

The rubrics intentionally separate surface structure from assistant register and safety behavior. `G1` asks only about structure/formatting; `G2` asks whether the response behaves like a helpful assistant addressing the user; `S1` asks whether harmful prompts are refused; and `S2` asks whether benign safety prompts are incorrectly refused or over-hedged. Pairwise `G2` and `S2` are the primary behavioral readouts because they are condition-blind, randomized, and closer to the human-preference setup used in LLM-as-judge work. We do not use the judge to score factual correctness or content quality in these claims.

Known LLM-judge failure modes motivate this design. Prior work finds that strong LLM judges can approximate human preferences but are sensitive to position, verbosity, self-enhancement, rating indeterminacy, and rubric design (Liu et al., 2023; Zheng et al., 2023; Wang et al., 2023; Dubois et al., 2024; Guerdan et al., 2025). We therefore use condition blinding, randomized order, a `TIE` option, fixed rubrics, schema-validated outputs, model-disagreement escalation, bootstrap uncertainty, and a completed blind human audit rather than treating the automated judge as ground truth.

### B2: Human audit results

The current behavioral run materialized blind human-audit packs for all five dense families: 120 pointwise audit items per model, 600 total. The primary pairwise human packet contains 1,200 blinded comparisons per rater: 60 items per dense model for each of four primary contrasts, `C` versus `D_late` on `G2`, `C` versus `D_late` on `S2`, `B_late` versus `A` on `G2`, and `B_late` versus `A` on `S2`. Two independent raters completed the returned CSVs in `paper_draft/human_eval_survey/pointwise/` and `paper_draft/human_eval_survey/pairwise/`; the hidden keys are kept separately under `paper_draft/human_eval_survey/keys/`.

The human protocol follows standard NLG human-evaluation practice: independent blinded raters, unchanged row IDs, no access to condition or model labels, confidence ratings, optional notes for ambiguity, and frozen labels before unblinding (van der Lee et al., 2019). We do not adjudicate disagreements into a forced single label, because rating indeterminacy is itself informative for these outputs. Pairwise resolved rates therefore exclude `TIE` and `BOTH_BAD`, but those unresolved votes are reported explicitly. Confidence intervals bootstrap over `pair_id` clusters, with both rater votes included for each sampled item.

| Contrast | Criterion | Human resolved target win (95% CI) | Resolved votes | Unresolved vote rate | LLM same-sample resolved target win |
|---|---:|---:|---:|---:|---:|
| `C` over `D_late` | `G2` | `70.6%` `[64.0, 76.9]` | `214/600` | `64.3%` | `76.5%` |
| `C` over `D_late` | `S2` | `66.9%` `[57.5, 75.8]` | `118/600` | `80.3%` | `75.4%` |
| `B_late` over `A` | `G2` | `60.5%` `[53.7, 67.3]` | `294/600` | `51.0%` | `62.1%` |
| `B_late` over `A` | `S2` | `65.8%` `[58.6, 72.6]` | `272/600` | `54.7%` | `65.7%` |

The audit agrees directionally with the LLM judge on all four primary pairwise contrasts, and all four pooled human confidence intervals are above chance on resolved votes. The result should still be read conservatively. Ties and both-bad labels are common, especially for IT-vs-`D_late` benign-safety comparisons, and some per-family cells have small resolved counts or flip direction. The human audit supports the pooled behavioral bridge; it is not a per-family behavioral localization claim.

| Contrast | Criterion | Raw agreement | Cohen's κ, four labels | Collapsed κ |
|---|---:|---:|---:|---:|
| `C` vs `D_late` | `G2` | `48.7%` | `0.01` | `0.00` |
| `C` vs `D_late` | `S2` | `66.7%` | `0.15` | `0.16` |
| `B_late` vs `A` | `G2` | `49.7%` | `0.27` | `0.32` |
| `B_late` vs `A` | `S2` | `64.3%` | `0.52` | `0.48` |

Collapsed κ merges `TIE` and `BOTH_BAD` into a single unresolved label. The agreement pattern is heterogeneous: the strongest reliability is the PT-side benign-safety comparison, while the IT-side assistant-register comparison has low κ because both raters often use `TIE` but disagree on the relatively small set of resolved items. This motivates using the human audit as directional confirmation of the pairwise bridge, not as a high-precision substitute for the automated judge.

Pointwise audit labels, on common/applicable filled fields, are used only as judge-calibration diagnostics. Weighted R1/R2 κ is `0.64` for `G1` and `0.32` for `G2`; categorical κ is `0.45` for `S1` and `0.33` for `S2`. Human-vs-judge κ is `0.60` for `G1`, `0.47` for `G2`, `0.58` for `S1`, and `0.31` for `S2`. These values are moderate rather than decisive, especially for `G2/S2`, which is why the paper's behavioral claim remains the blind pairwise direction result plus LLM-judge effect sizes rather than a claim of high-agreement pointwise human scoring.

### B3: Statistical testing approach

Pointwise behavioral summaries are prompt-bootstrap estimates over dense-family records. Pairwise summaries report target win rate, other win rate, tie rate, and resolved win rate, with bootstrap 95% confidence intervals. Dense-family claims pool Gemma, Llama, Qwen, Mistral, and OLMo; DeepSeek is excluded from the canonical behavioral pool because its MoE routing makes dense-family MLP graft/swap interpretation non-comparable. The main LLM-judge behavioral claim is deliberately asymmetric: IT-side late-window degradation is strong, while PT-side late grafting produces partial movement mainly supported by blind pairwise preferences.

For the completed human audit, the primary confirmatory quantities are pairwise resolved win rates and bootstrap confidence intervals for the same four primary contrasts shown in Figure B1. The rater packets randomize A/B order, expose no model or condition labels, and include `TIE`/`BOTH_BAD` to avoid forcing noisy preferences. The human audit was analyzed after the automated judge setup, endpoint definitions, and primary contrasts were fixed, so it functions as an external validation check rather than as a source of endpoint selection. We report same-sample LLM rates beside the human rates to show direction agreement directly.

### B4: Prompt dataset construction

The behavioral run uses a frozen 600-prompt core subset of `eval_dataset_v2.jsonl`. The subset emphasizes conversational and register-sensitive prompts, benign safety prompts, harmful safety prompts, and format-sensitive items. Pointwise `G1` applies broadly; `G2` applies only to assistant-facing records; `S1` applies to harmful safety records; and `S2` applies to benign safety records. The judge manifest stores hashes for the dataset manifest, pipeline manifest, human-audit manifest, and sample outputs so reruns can detect accidental dataset or pipeline drift.

## Appendix C: Token Classifier Specification and Robustness

### C1: Classifier specification

Five categories with priority order (first match wins): STRUCTURAL (regex-matched, 2.8% baseline), PUNCTUATION (1.2%), DISCOURSE (Hyland 2005 taxonomy, 0.5%), FUNCTION (closed-class, 34.9%), CONTENT (default, 59.9%).

### C2: Perturbation robustness analysis

Four perturbation scenarios tested. Only the STRUCTURAL/DISCOURSE boundary produces measurable sensitivity (Δ = 0.145 when merged — expected, as both capture formatting). Content-side perturbations produce Δ < 0.00001. No core finding depends on precise boundary choices.

## Appendix D: Commitment Threshold Sensitivity

The convergence-gap claim does not depend on a single commitment threshold. The main text summarizes the threshold-free top-1 commitment view; Appendix Figures S34-S37 show majority-vote, KL-threshold, cosine, entropy, and normalized-depth CDF variants. Across KL thresholds from 0.05 to 1.0 nats, the IT-minus-PT delay remains qualitatively stable. Representation-space commitment summaries are weaker, which is consistent with the paper's narrower claim that the robust PT↔IT separation is in decoded next-token distributions rather than arbitrary residual-stream distances.

## Appendix E: Broader Literature Positioning

Table E1 broadens the main-text comparison to methodological precedents, behavioral-direction papers, and conceptual work on instruction following. The coarse labels are conservative: `Yes` means the element is directly part of the paper's design, `Partial` means a narrower task-specific version, and `No` means it is not part of the main design.

| Paper | PT↔post-trained descendants? | Cross-family paired checkpoints? | Identical-history internal comparison? | Symmetric depth-localized intervention? | First-divergence candidate/margin? | Natural-decoding consequence test? |
|---|---|---|---|---|---|---|
| Lad et al. (2024) | No | No | No | No | No | No |
| Joshi et al. (2025) | No | No | No | No | No | No |
| Wu et al. (2024) | Yes | No | No | No | No | No |
| Du et al. (2025) | Yes | Yes | No | No | No | No |
| Li et al. (2025) | No | No | No | No | No | No |
| Chaudhury (2025) | Yes | No | No | No | No | No |
| Prakash et al. (2024) | Yes | No | No | Partial | No | No |
| Bigoulaeva et al. (2026) | Partial | No | No | Partial | No | No |
| Zhao, Ziser, and Cohen (2024) | Yes | No | No | No | No | No |
| Nepal et al. (2025) | Yes | Partial | No | Partial | No | No |
| Arditi et al. (2024) | No | No | No | No | No | Yes |
| Lu et al. (2026) | No | No | No | No | No | Yes |
| Stolfo et al. (2024) | Partial | Partial | No | No | No | Yes |
| Turner et al. (2023); Zou et al. (2023); Panickssery et al. (2024) | No | No | No | No | No | Yes |
| Hewitt et al. (2024) | No | No | No | No | No | No |
| Rocchetti & Ferrara (2026) | No | No | No | No | No | No |
| Ours | Yes | Yes | Yes | Yes | Yes | Yes |

Lad et al. (2024) and Joshi et al. (2025) are the closest phenomenon-level comparisons. Lad gives a generic vocabulary for mid-to-late inference stages: prediction ensembling followed by residual sharpening. Joshi studies late confidence calibration after decision certainty has emerged. We do not treat late sharpening, late calibration, or late-stage confidence correction as new phenomena. The convergence gap is the aggregate signature; the contribution is the paired PT/IT first-divergence analysis that asks which depth windows change the divergent token identity and how the final IT-vs-PT margin depends jointly on upstream state and late computation.

Du et al. (2025) overlaps with the paper's basic model-diffing premise: compare base and post-trained descendants and ask how knowledge, truthfulness, refusal, and confidence change internally. That overlap is real and useful. Our narrower addition is to condition on the first natural PT/IT next-token disagreement and run matched-history interventions at that exact prefix, so the readout is not only "post-training changes confidence/refusal" but "this upstream state and this late stack interact on this divergent-token margin."

Prakash et al. (2024) is the closest methodological precedent in this broader set: CMAP patches activations across related base/fine-tuned models to reveal improved mechanisms. Our use is different in scope and target: we apply a symmetric equal-width early/mid/late MLP graft/swap design across model families, and the localized target is not entity tracking but natural PT/IT next-token disagreement under matched-prefix control. Following the caution urged by activation-patching work such as Heimersheim and Nanda (2024), we describe these interventions as causal leverage on measured readouts, not as complete mechanism recovery.

Bigoulaeva et al. (2026), Zhao, Ziser, and Cohen (2024), and Nepal et al. (2025) are important layer-localization precedents. Bigoulaeva et al. argue that instruction vectors can act as circuit selectors, with later pathways conditioned on earlier task representations; this is closely aligned with our non-autonomous late-readout interpretation. Zhao, Ziser, and Cohen study where task-oriented representations appear in instruction-tuned models, and Nepal et al. find math-critical layers that remain important across post-training. These papers make our context-gated late readout result plausible rather than surprising. The distinction is not that we are the first to see layer structure after instruction tuning; it is that we estimate window-level intervention effects at the first natural PT/IT token disagreement, with matched histories and separate candidate-identity and margin outcomes.

Stolfo et al. (2024), Turner et al. (2023), Zou et al. (2023), and Panickssery et al. (2024) are the main reason we do not frame the contribution as finding an instruction-following activation vector. That claim is not new. Those papers show that activation-space directions can control behavior, including instruction-following constraints and high-level traits. The paper-level novelty here is that paired PT/IT next-token disagreement is decomposed with graft/swap interventions under identical histories and first-divergence counterfactuals, rather than by extracting a behavioral vector.

The appendix table makes the novelty claim deliberately concrete. Prior rows contain individual ingredients: PT↔post-training comparison, behavioral localization, cross-model substitution, layer importance, or natural-decoding intervention. The paper's row is distinct because those ingredients are combined around the first point where paired PT/IT checkpoints disagree about the next token, with the convergence gap serving as the aggregate signature and candidate-transfer versus context-gated late readout as the tested decomposition. Hewitt et al. (2024) and Rocchetti and Ferrara (2026) are useful conceptual boundary papers rather than direct method-level foils. They ask what instruction following is, or whether it reflects a universal mechanism at all. Our paper asks a narrower and more operational question: what paired PT↔post-training forward-pass signature appears across dense families, how middle and late windows divide candidate selection from context-gated late readout, and whether that same intervention family matters under natural decoding.

## Appendix F: Scope Boundaries and Counter-Interpretations

**Free-running histories.** The dense-family native-decoding curves establish the convergence gap as a model-in-use signature. They are not used as the sole intervention evidence because PT and IT continuations can diverge. The matched-prefix native-JS replay and MLP graft/swap experiments are the controlled-history evidence. DeepSeek-V2-Lite is shown only as a MoE side case.

**Endpoint metric and probe dependence.** `KL(layer || own final)` is intentionally a stabilization-to-own-decision metric, not a direct cross-model distance. It asks how far a layer is from the same model's eventual next-token distribution. Endpoint relativity is part of the metric, so we test the most obvious endpoint covariates directly: Exp22's primary coarsened-exact matching estimator balances token steps on final entropy, final top-1 confidence, and final top-1/top-2 margin, and the late IT-minus-PT KL gap remains positive under both raw and tuned probes. This is not a claim that every possible endpoint property has been controlled. Matched-prefix native JS is a separate endpoint-free companion control, not an interchangeable version of the KL trajectory. Tuned-lens quality can still change absolute KL magnitudes, so we do not treat the tuned-lens discovery figure as standalone identification. Raw-logit-lens variants, endpoint-matched KL, and native same-layer JS under identical histories preserve the same qualitative PT↔IT separation.

**Gemma tuned-lens quality.** Gemma is the family where the tuned lens improves least over the raw lens at intermediate depth, and it is also the family with the largest late `δ`-cosine shift. We flag this coincidence rather than using Gemma tuned-lens metrics as a standalone pillar. The relevant separation is that `δ`-cosine is lens-free, while the convergence-gap discovery metric is lens-dependent. For the lens-dependent convergence-gap discovery metric, the raw lens strengthens rather than erases Gemma's final-half gap (`1.008` raw versus `0.351` tuned), and the dense-family raw final-half gap remains positive even with Gemma removed (`0.712`; Table S25a).

**Weight-difference magnitude.** A possible confound is that late grafts matter most simply because post-training changed late MLP parameters most. Figure S18 already makes this unlikely outside Gemma: Llama, Qwen, Mistral, and OLMo do not show a uniformly late-concentrated MLP weight-change profile. Table S18a makes the cross-reference explicit by dividing the matched-prefix final-20% KL effect by a same-window mean MLP weight-change proxy. Late remains the largest normalized effect in every dense family, and in the dense-family pool the middle and late weight-change proxies are nearly equal while only late has a large positive effect. Table S18b adds the complementary raw-effect and leave-one-family-out view: all dense families have positive raw late effects, and removing Mistral increases rather than decreases the pooled mean. Mistral's large normalized ratio is therefore a denominator artifact of its unusually small PT-to-IT MLP RMS change, not evidence that Mistral dominates the window-level intervention result. The matched random-control analysis adds a residual-projection control and shows that the dense-family late KL effect is specific to the learned IT substitution rather than any matched same-window perturbation (`+0.327` true versus `+0.003` random), but it still does not convert the result into a per-unit-weight efficiency claim. We therefore treat weight-difference magnitude as a family-specific diagnostic, not the paper-level explanation for late causal leverage. The table is still a descriptive normalization, not a claim that we have estimated a true per-parameter causal efficiency.

**Metric surface area.** We report several convergence summaries because the phenomenon is trajectory-shaped, but the main claims are intentionally pinned to a small ladder: native KL-to-own-final discovery, endpoint-free same-history JS, symmetric final-20% graft/swap effects, first-divergent-token identity/margin, and judged behavior. Per-family or auxiliary-metric deviations are treated as diagnostics unless they agree with that ladder.

**Statistical dependence and pooling.** Dense-family summaries are not intended to turn model families into exchangeable prompt records. For the headline Exp23 residual-state factorial, the primary pooled interval bootstraps prompt units within each family and averages family estimates. The main text therefore reports both the Dense-5 interaction interval and per-family interaction intervals; the leave-one-family-out estimates are also positive: Gemma removed `+1.77` `[+1.69, +1.86]`, Llama removed `+2.98` `[+2.87, +3.10]`, Qwen removed `+2.93` `[+2.81, +3.04]`, Mistral removed `+2.66` `[+2.54, +2.78]`, and OLMo removed `+2.83` `[+2.72, +2.95]`. For older depth-ablation and symmetric graft/swap summaries where prompt-level traces are not uniformly committed for every family, the main text uses family-bootstrap intervals over the five dense-family means rather than pretending to have prompt-level uncertainty. Ratios are kept only as descriptive scale checks because small denominators can inflate them; the interaction term is the statistical object.

**Window definitions and boundary sensitivity.** The early/middle/late labels are coarse intervention windows, not discovered sharp phase boundaries. We use three windows because the interpretation needs a direct comparison between early generic perturbation, middle candidate-selection effects, and context-gated late readout. This choice is also aligned with prior layer-stage language: early layers build local/lexical state, middle layers often carry task or behavioral selection signals, and late layers are where sharpening, calibration, and final-token reconciliation are most visible. To avoid cherry-picking an exact cutoff, the cross-family graft/swap and first-divergence experiments use pre-specified overlapping windows at comparable normalized depths rather than model-specific hand-tuned boundaries:

| Family | Early window | Middle window | Late window |
|---|---:|---:|---:|
| Gemma 3 4B | `0-13` | `10-23` | `20-33` |
| Llama 3.1 8B | `0-12` | `9-21` | `19-31` |
| Qwen 3 4B | `0-13` | `11-24` | `22-35` |
| Mistral 7B | `0-12` | `9-21` | `19-31` |
| OLMo 2 7B | `0-12` | `9-21` | `19-31` |

These are "thirds" in the coarse sense of early-, middle-, and late-positioned windows, but they overlap so the result is not forced by a brittle single boundary and should not be read as a disjoint-subnetwork comparison. Earlier boundary checks support the same conservative reading. The layer-range sensitivity run varied the original Gemma late intervention range across `18-33`, `20-31`, `20-33`, and `22-33`; the broad late-range behavior survived, while magnitudes changed. The onset-threshold sensitivity run found that estimated onset layers move with the threshold and family, sometimes substantially. We therefore use those runs to justify a broad mid/late window design, not to claim that a particular layer index is the true start of the mechanism. The main evidence is correspondingly phrased as depth-window causal leverage: early windows have weaker or less consistent delayed-stabilization effects, middle windows transfer divergent-token identity more, and late windows have the largest tested convergence-gap and context-gated late readout effects.

**Prompt formatting.** The primary discovery condition uses each model in its native regime: IT models receive native chat templates and PT models receive raw prompts. This is the relevant model-in-use comparison. Raw-vs-templated matched-prefix controls support that the delayed-stabilization signal is not only a prompt-format artifact. In the first-divergence token tests, raw-shared controls preserve the qualitative mid-identity/late-margin split, while native chat templates amplify the deployed IT margin. We therefore read templates as part of the native IT operating distribution, not as the sole source of the effect.

**Teacher direction.** The primary matched-prefix replay uses IT-generated teacher continuations because the paper studies how PT differs from its instruction-following descendant. A reverse PT-teacher replay again shows a broad dense-family same-layer JS gap. Late amplification is teacher-stream-dependent under token-step weighting, so we use the reverse run as a robustness check for broad same-history divergence rather than as a separate late-localization claim.

**Middle versus late.** Direct same-layer target-gap closure and behavioral token identity are more mid-to-late distributed than purely late. Native final-token flow sharpens this rather than erasing it: the pure-flow analysis shows larger late IT margin gains for format and content, while the matched-prefix chronology shows that content-like candidate promotion can already be strong in middle windows. First-divergence token-choice tests sharpen the boundary further: middle swaps transfer first-divergent token identity more than late swaps, while late windows more strongly affect IT-vs-PT margin under IT-shaped upstream context. MLP write-out tests add direct evidence: pure IT late MLPs provide much larger support for the IT divergent token than early or middle MLPs. The residual-state x late-stack factorial adds the missing non-independence test: the same IT late stack has a positive but much smaller margin effect from a PT upstream state than from an IT-shaped upstream state. That does not contradict the main result because delayed stabilization, token identity, token write-out, and late-stack readout are different targets. The directly tested intervention claim is late-window leverage on delayed stabilization and context-gated late readout; the broader middle-to-late interpretation should be read as window-level causal leverage rather than complete named-circuit recovery.

**Relation to mid-layer alignment localization.** Li et al. (2025) and Chaudhury (2025) are not treated as foils to be overturned. They study safety/preference behavior directly and find middle-layer localization under their readouts. Our own first-divergence identity tests agree with that direction: middle swaps transfer PT-vs-IT token identity more than late swaps. The additional late claim is about a different target: delayed stabilization, direct MLP write-out, and late-stack readout into the final next-token margin under IT-shaped upstream context. A stricter head-to-head would rerun Chaudhury-style HHH preference patching and Li-style safety-layer tests on the same model pairs and compare those layers directly to the convergence-gap layers. We have not run that replication, so the paper claims compatibility and context-gated late readout, not that prior mid-layer alignment results are wrong.

**Behavioral asymmetry.** The current LLM-judge behavioral bridge is clearest on IT-side degradation under late PT swaps. Late IT grafting into PT is measurable but smaller than the full PT-to-IT behavioral gap. This is expected if late MLPs participate in context-gated late readout embedded in a broader post-training circuit, not a complete assistantness module. The completed blind human audit agrees with the pooled direction of the four primary pairwise contrasts, but high unresolved rates and heterogeneous κ mean it should be treated as confirmatory evidence for the behavioral bridge, not as part of the core internal localization claim.

**Candidate-transfer and amplification status.** We include direct analyses for both sides of the decomposition. The first-divergence token-choice test is the cleanest token-identity test because it conditions on the first PT/IT divergent token and evaluates grafted branches at the shared prefix. The MLP write-out test is the cleanest amplification test because it stores finite-difference token projections for the same first-divergence setup. Native final-token flow remains useful for chronology, but the older matched-prefix traces store ranks/top-k sets rather than per-layer logits. A complete named-circuit chronology still requires matched-prefix reruns with per-layer logit capture and feature-level mediation tests.

**Mid+late interaction status.** The current evidence no longer leaves the basic `PT + IT mid+late` branch missing. In the MLP-only first-divergence/write-out tests, adding IT late MLP updates to a PT host is weak by itself, and adding IT late on top of IT middle gives only a small additional fixed-prefix MLP margin gain (`+0.0119` logits; 95% CI `[+0.007, +0.016]`). The matched-prefix KL factorial gives a complementary stabilization view: on the PT side, late IT MLP grafting changes final-20% KL by `+0.217` nats (95% CI `[+0.209, +0.226]`), mid by `-0.031` (95% CI `[-0.036, -0.026]`), and mid+late by `+0.198` (95% CI `[+0.188, +0.208]`); on the IT side, removing mid, late, and mid+late MLPs reduces the delayed-stabilization signal by `0.486` (95% CI `[0.473, 0.499]`), `0.822` (95% CI `[0.805, 0.840]`), and `0.987` nats (95% CI `[0.965, 1.009]`) respectively. Thus mid+late effects are real but not simply additive; the windows overlap in what they account for under these readouts. The residual-state x late-stack factorial then shows strong positive non-independence for the first-divergence margin: IT late stack given PT upstream is `+0.572` logits (95% CI `[+0.494, +0.647]`), IT late stack given IT upstream is `+3.207` (95% CI `[+3.095, +3.321]`), upstream context is `+4.239` (95% CI `[+4.105, +4.376]`), late-stack main effect is `+1.890` (95% CI `[+1.805, +1.975]`), and their interaction is `+2.635` (95% CI `[+2.538, +2.736]`) under common IT readout. This is why we phrase the result as context-gated late readout rather than a standalone late module or a fully independent mid+late sum.

**Compatibility-amplification label control.** The Exp23 interaction can be reparameterized as a comparison between two matched-stage boosts. Let `Y = logit(IT token) - logit(PT token)`. IT late-stack compatibility is `Y(U_IT,L_IT) - Y(U_PT,L_IT)`, while PT late-stack compatibility is `Y(U_IT,L_PT) - Y(U_PT,L_PT)` after flipping the margin into the PT-token direction. Their difference equals the residual-state x late-stack interaction. On the primary holdout, IT compatibility is `+5.56` logits and PT compatibility is `+2.92`, giving `+2.64`; on the content/reasoning extension, IT compatibility is `+4.37` and PT compatibility is `+2.69`, giving `+1.68`. A label-swap permutation that preserves all four cell values per prompt/event but randomly swaps PT/IT orientation yields null distributions centered near zero. The primary observed amplification exceeds the null 99.9th percentile (`+0.239` logits; one-sided `p=5.0e-5`, the minimum resolvable with `20,000` permutations), and the content/reasoning extension does too (`+0.155` logits; one-sided `p=5.0e-5`). This tests PT/IT label alignment of the interaction, not a claim that pretrained models lack generic late-stage context dependence.

**Exp23 subgroup characterization.** We stratified the residual-state x late-stack factorial without changing the estimand. The analysis uses the common-IT readout interaction, dense-family prompt bootstrap intervals, and reportable strata with at least 50 first-divergence units spanning at least three model families. Across the `2,983` dense-family first-divergence records, the interaction remains positive in every reportable subgroup:

| Subgroup | Stratum | Records | Interaction, logits (95% CI) |
|---|---:|---:|---:|
| Prompt category | `GOV-CONV` | `1,494` | `+2.05` `[+1.95, +2.15]` |
| Prompt category | `GOV-FORMAT` | `745` | `+3.61` `[+3.37, +3.87]` |
| Prompt category | `SAFETY` | `744` | `+2.83` `[+2.67, +3.01]` |
| IT-token category | `CONTENT` | `1,265` | `+2.50` `[+2.35, +2.65]` |
| IT-token category | `FORMAT` | `685` | `+2.60` `[+2.39, +2.82]` |
| IT-token category | `FUNCTION_OTHER` | `1,033` | `+2.81` `[+2.66, +2.97]` |
| IT-margin tercile | low | `1,052` | `+1.03` `[+0.95, +1.11]` |
| IT-margin tercile | mid | `948` | `+2.02` `[+1.93, +2.12]` |
| IT-margin tercile | high | `983` | `+4.93` `[+4.75, +5.11]` |
| Assistant marker | present | `551` | `+3.30` `[+3.08, +3.54]` |
| Assistant marker | absent | `2,432` | `+2.48` `[+2.37, +2.59]` |

This subgroup analysis is texture, not a new primary claim. The primary Exp23 residual-state factorial used the 600-record holdout slice `eval_dataset_v2_holdout_0600_1199.jsonl`, which contains `GOV-CONV`, `GOV-FORMAT`, and `SAFETY` prompt categories. Therefore the prompt-category result above supports context-gated late readout across conversational/governance, formatting, and safety prompts. The `CONTENT` row is a token-category result inside those prompts: it means the first divergent token can be content-like and still show a positive interaction, not that the primary holdout contains factual or reasoning prompt families.

We therefore ran the same residual-state x late-stack factorial on a content/reasoning slice drawn from `eval_dataset_v2.jsonl`. This extension uses `5,889` dense-family first-divergence records spanning `CONTENT-FACT`, `CONTENT-REASON`, and `GOV-FORMAT`. The central interaction remains positive (`+1.68` logits, 95% CI `[+1.59, +1.78]`) and is positive in all five dense families (Gemma `+2.39`, Llama `+2.10`, Qwen `+1.71`, Mistral `+0.64`, OLMo `+1.57`). Prompt-category interactions are:

| Slice | Prompt category | Records | Interaction, logits (95% CI) |
|---|---:|---:|---:|
| Content/reasoning extension | `CONTENT-FACT` | `2,837` | `+1.84` `[+1.71, +1.97]` |
| Content/reasoning extension | `CONTENT-REASON` | `2,087` | `+1.34` `[+1.22, +1.45]` |
| Content/reasoning extension | `GOV-FORMAT` | `965` | `+1.97` `[+1.69, +2.26]` |

The extension also sharpens the scope of the claim. Under common-IT readout on this slice, upstream context remains large (`+3.53` logits, 95% CI `[+3.42, +3.64]`) and the interaction remains large (`+1.68`), but the IT late stack in a PT upstream state moves against the IT token (`-1.41` logits, 95% CI `[-1.51, -1.31]`), while the same IT late stack in an IT upstream state is only modestly positive (`+0.27`, 95% CI `[+0.15, +0.40]`). The late-weight main effect is negative (`-0.57`, 95% CI `[-0.67, -0.46]`). Thus factual/reasoning prompts replicate the non-independence result while arguing against a domain-invariant positive late-module-sufficiency story. The strongest stable claim is context-gated late readout: upstream state and late computation have to be modeled jointly under this readout, and the sign/magnitude of the late-only term depends on prompt domain.

**Geometry and mediation.** Late residual-opposing `δ`-cosine is not new by itself and is present in pretrained models. The geometric claim is the additional IT-minus-PT late shift. The MLP decomposition further shows that the residual-opposing component itself does not directly increase IT-vs-PT margin in pure IT late layers (`-0.0046` logits; 95% CI `[-0.009, -0.001]`), while the full late MLP update strongly does (`+0.768` logits; 95% CI `[+0.729, +0.805]`). The residual-state factorial is therefore not evidence that residual opposition itself is the margin write-in. It is evidence that the late margin/readout effect depends strongly on the state being reconciled. We therefore treat residual opposition as a geometric marker of late reconciliation, not as the margin write-in vector.

The content/reasoning Exp21 extension is consistent with this caution. On raw-shared dense-family first-divergence records, the late-weight finite-difference effect on the IT-vs-PT MLP margin is positive but small (`+0.042` logits, 95% CI `[+0.036, +0.048]`), and the residual-opposing component contributes approximately zero (`-0.000006`, 95% CI `[-0.000041, +0.000029]`). The remaining/token-specific component carries the positive late-weight margin (`+0.049`, 95% CI `[+0.043, +0.055]`). This is why the paper treats residual opposition as geometric evidence for late revision, while treating token-specific write-in and the residual-state factorial as stronger evidence for margin/readout.

## Appendix G: Reproducibility and Artifact Map

**Public release.** During double-blind review, the project repository is provided through an anonymized artifact archive. It includes the current manuscript draft, all source experiment packages under `src/poc/`, shared cross-model infrastructure under `src/poc/cross_model/`, grouped script entrypoints under `scripts/`, and committed paper-facing artifacts under `results/`. The canonical prompt and evaluation datasets are committed under `data/`, including `eval_dataset_v2.jsonl`, `eval_dataset_v2_holdout_0600_1199.jsonl`, `exp3_dataset.jsonl`, `exp6_dataset.jsonl`, and `gold_standard_v1.csv`.

**Audit levels.** We expose the results at three levels, so a reviewer can check the main numbers without trusting prose summaries.

| Level | Command | What it verifies | Expected cost |
|---|---|---|---|
| Summary audit | `bash scripts/reproduce/reproduce_claims_from_summaries.sh` | Recomputes the headline paper numbers from committed JSON/CSV summaries. | CPU only; under 1 minute. |
| Minimal raw shard | `bash scripts/reproduce/reproduce_minimal.sh` after fetching the shard | Validates a 20-prompt, one-family raw shard containing cached per-layer logits, intervention outputs, first-divergence records, and expected summary JSONs. | CPU-only if cached logits are used; about 1-3 GPU-hours on one 80GB A100/H100 if regenerated. |
| Full rerun | Experiment launchers under `scripts/run/` and source packages under `src/poc/` | Regenerates full traces, summaries, and plots. | Multi-GPU; see estimates below. |

**Checkpoint and tokenizer manifest.** The run configs store exact Hugging Face repo IDs (`pt_model_id`, `it_model_id`) but not resolved snapshot hashes. Historical runs therefore used the default `main` revision as resolved by Hugging Face at run time. To make that scope explicit, we add a checkpoint audit manifest generated by `scripts/analysis/build_model_checkpoint_manifest.py` at `results/paper_synthesis/model_checkpoint_manifest.json`. The SHA column below is the current Hub `main` resolution from that audit, not a claim that older runs pinned the hash:

| Family | PT repo and audited `main` SHA | IT repo and audited `main` SHA | Tokenizer and prompt-template note |
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
| Residual-state x late-stack context gating | same | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/exp23_summary.json`; `analysis/exp23_effects.csv` | common-IT late given PT upstream `+0.572` `[+0.494, +0.647]`; late given IT upstream `+3.207` `[+3.095, +3.321]`; upstream `+4.239` `[+4.105, +4.376]`; late-stack `+1.890` `[+1.805, +1.975]`; interaction `+2.635` `[+2.538, +2.736]`; Gemma-removed interaction `+1.77` `[+1.69, +1.86]` |
| Exp23 compatibility-amplification label control | `uv run python scripts/analysis/analyze_exp23_compatibility_permutation.py --run-root results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4 --n-permutations 20000` | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/compatibility_permutation/` | primary IT compatibility `+5.56`, PT compatibility `+2.92`, amplification `+2.64`; label-swap null 99.9th percentile `+0.239`, `p=5.0e-5` |
| Exp23 subgroup characterization | `python scripts/analysis/analyze_exp23_interaction_subgroups.py --run-root results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4 --n-bootstrap 2000` | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/subgroups/` | prompt-category interactions: `GOV-CONV +2.05`, `GOV-FORMAT +3.61`, `SAFETY +2.83`; IT-token-category interactions: `CONTENT +2.50`, `FORMAT +2.60`, `FUNCTION_OTHER +2.81` |
| Content/reasoning residual-state extension | `bash scripts/run/run_exp23_exp21_content_reasoning_extension.sh` | `results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/analysis/exp23_summary.json`; `analysis/subgroups/exp23_subgroup_report.md` | common-IT interaction `+1.68` `[+1.59, +1.78]`; prompt interactions: `CONTENT-FACT +1.84`, `CONTENT-REASON +1.34`, `GOV-FORMAT +1.97`; late IT from PT upstream `-1.41` `[-1.51, -1.31]` |
| Content/reasoning compatibility-amplification label control | `uv run python scripts/analysis/analyze_exp23_compatibility_permutation.py --run-root results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8 --n-permutations 20000` | `results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/analysis/compatibility_permutation/` | content/reasoning IT compatibility `+4.37`, PT compatibility `+2.69`, amplification `+1.68`; label-swap null 99.9th percentile `+0.155`, `p=5.0e-5` |
| Content/reasoning MLP write-out extension | same | `results/exp21_productive_opposition/exp21_content_reasoning_20260427_0943_h100x8/analysis/summary.json`; `analysis/effects.csv` | late-weight IT-vs-PT MLP margin `+0.042` `[+0.036, +0.048]`; residual-opposing component `-0.000006` `[-0.000041, +0.000029]`; remaining/token-specific component `+0.049` `[+0.043, +0.055]` |
| Behavioral bridge and human audit | same | `results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_behavior_summary.json`; `results/exp15_symmetric_behavioral_causality/human_eval/human_eval_summary.json` | LLM resolved G2: `56.3%` `[53.3, 59.5]` and `77.1%` `[74.7, 79.6]`; human resolved G2: `60.5%` `[53.7, 67.3]` and `70.6%` `[64.0, 76.9]` |

**Committed versus regenerated artifacts.** The repository commits paper-facing summaries and plots used for manuscript numbers: JSON/CSV/MD summaries, bootstrap confidence intervals, human-evaluation summaries, and final PNG figures. It does not commit large regenerated intermediates such as raw activation arrays (`*.npy`, `*.npz`), model/probe tensors (`*.pt`, `*.safetensors`), tuned-lens checkpoints, or multi-gigabyte raw per-token trace JSONL/GZ files. These are size exclusions, not confidentiality exclusions. The corresponding collection scripts, analysis scripts, prompt manifests, and anonymized archive pointers are public within the review artifact.

**Large-artifact locations.** During review, raw traces, tuned-lens probes, endpoint-control records, late random-control traces, first-divergence archives, and the minimal audit shard are mirrored under the anonymized artifact archive. The public camera-ready release will replace these placeholders with stable permanent storage locations after the double-blind period.

**Hardware and runtime.** All full reruns use bf16 inference and deterministic greedy decoding unless a script states otherwise. The 4B-8B dense-family experiments fit on one 80GB A100/H100 per model job; DeepSeek-V2-Lite is reported separately because it is MoE/SFT-only. The full dense-5 intervention and endpoint-control suite (Exp11/14/16/20/21/22/23, including trace collection and analysis) should be budgeted as hundreds of H100 GPU-hours and roughly 0.5-1.5 TB of transient disk if raw traces and per-token records are retained. The minimal 20-prompt shard needs one 80GB A100/H100, about 1-3 GPU-hours if regenerated from checkpoints, and about 20-80 GB local disk depending on whether all per-layer logits are kept. The summary audit is CPU-only and reads only committed JSON/CSV files.

**Tuned-lens reproducibility.** The main intervention claims do not require tuned-lens checkpoints. The matched-prefix JS replay, graft/swap KL summaries, first-divergence identity/margin results, and finite-difference MLP write-out analyses are auditable from committed summaries and do not depend on trained probes. The tuned lens is used for the discovery visualization and commitment diagnostics, and the paper reports raw-lens sensitivity alongside it. Probe checkpoints are included in the anonymized artifact archive; retraining uses `uv run python -m src.poc.cross_model.tuned_lens --model MODEL --variant {pt,it} --device cuda:N`. A complete dense-5 retrain is 10 PT/IT runs; including the DeepSeek side case is 12 runs. The full set takes about 4-6 wall-clock hours on 8x80GB A100/H100 GPUs with joint all-layer training.

**Main claim-to-artifact map.**

| Claim or analysis | Code entrypoints | Committed artifacts |
|---|---|---|
| Dense-5 convergence gap and commitment delay, with DeepSeek side case in appendix artifacts | `src/poc/exp09_cross_model_observational_replication/`; shared adapters in `src/poc/cross_model/` | `results/exp09_cross_model_observational_replication/data/`, `results/exp09_cross_model_observational_replication/plots/` |
| Endpoint-matched convergence gap | `src/poc/exp22_endpoint_deconfounded_gap/`; `scripts/run/run_exp22_endpoint_deconfounded_gap_runpod.sh`; `scripts/analysis/analyze_exp22_endpoint_deconfounded_gap.py`; `scripts/analysis/build_exp22_endpoint_deconfounded_synthesis.py` | `results/paper_synthesis/exp22_endpoint_deconfounded_table.csv`, `results/paper_synthesis/exp22_endpoint_deconfounded_summary.png`, raw mirror listed above |
| Gemma feature and Tier-0 validation controls | `src/poc/exp06_corrective_direction_steering/`; `src/poc/exp07_methodology_validation_tier0/`; `scripts/run/run_exp7_0*.sh`; `scripts/plot/plot_validation_tier0.py` | `results/exp06_corrective_direction_steering/plots/`, `results/exp07_methodology_validation_tier0/data/`, `results/exp07_methodology_validation_tier0/plots/` |
| Layer-range and onset sensitivity | `scripts/plot/plot_validation_tier0.py`; Tier-0 0F/0J runs | `results/exp07_methodology_validation_tier0/0F/layer_range_sensitivity_table.csv`, `results/exp07_methodology_validation_tier0/0J/onset_table.csv`, `results/exp07_methodology_validation_tier0/plots/0F_layer_range_sensitivity.png`, `results/exp07_methodology_validation_tier0/plots/0J_onset_sensitivity.png` |
| Matched-prefix graft depth ablation | `src/poc/exp11_matched_prefix_mlp_graft/` | `results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/` and selected summaries under `results/exp11_matched_prefix_mlp_graft/data/` |
| Symmetric graft/swap and late random-control specificity | `src/poc/exp14_symmetric_matched_prefix_causality/`; `src/poc/exp19_late_mlp_specificity_controls/` | `results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/`, `results/exp19_late_mlp_specificity_controls/` |
| Behavioral bridge and human audit | `src/poc/exp15_symmetric_behavioral_causality/`; `scripts/eval/llm_judge.py`; human-audit materials under `paper_draft/human_eval_survey/` | `results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/`, `results/exp15_symmetric_behavioral_causality/human_eval/` |
| Endpoint-free matched-prefix JS replay | `src/poc/exp16_matched_prefix_js_gap/`; `scripts/analysis/analyze_exp16.py`; `scripts/plot/plot_exp16.py` | `results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/`, `results/exp16_matched_prefix_js_gap/exp16_js_reverse_pt_teacher_20260422_165259/` |
| Native final-token flow and matched-prefix chronology | `src/poc/exp18_midlate_token_handoff/`; `scripts/run/run_exp18_yanda_full.sh` | `results/exp18_midlate_token_handoff/full_runpod_20260423_095122/`, `results/exp18_midlate_token_handoff/matched_prefix_latest/` |
| First-divergence token identity and margin amplification | `src/poc/exp20_divergence_token_counterfactual/`; Exp20 analysis scripts under `scripts/analysis/` | `results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/`, `results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early/validation_analysis/` |
| MLP write-out, local context-gating proxy, content/reasoning extension, and synthesis figure | `src/poc/exp21_productive_opposition/`; `scripts/run/run_exp21_productive_opposition_runpod.sh`; `scripts/run/run_exp23_exp21_content_reasoning_extension.sh`; `scripts/analysis/analyze_exp21_productive_opposition.py`; `scripts/analysis/build_exp20_exp21_handoff_synthesis.py` | `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/`, `results/exp21_productive_opposition/exp21_content_reasoning_20260427_0943_h100x8/analysis/`, `results/paper_synthesis/` |
| Mid+late KL factorial, residual-state x late-stack interaction, subgroup characterization, label-swap control, and content/reasoning extension | `src/poc/exp23_midlate_interaction_suite/`; `src/poc/exp23_midlate_kl_factorial/`; `scripts/run/run_exp23_midlate_interaction_suite.sh`; `scripts/run/run_exp23_exp21_content_reasoning_extension.sh`; `scripts/analysis/analyze_exp23_midlate_interaction_suite.py`; `scripts/analysis/analyze_exp23_interaction_subgroups.py`; `scripts/analysis/analyze_exp23_compatibility_permutation.py` | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/`, `results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/analysis/` |
