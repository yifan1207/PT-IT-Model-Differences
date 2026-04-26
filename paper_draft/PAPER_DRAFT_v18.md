# The Convergence Gap in Instruction-Tuned Language Models: Evidence for a Mid-to-Late Token-Identity/Readout Split.

**Anonymous authors** | NeurIPS 2026 Submission

---

## Abstract

Instruction tuning changes what language models say, but much less is known about how it changes the internal process by which they settle on a next token. We address this with a cross-family PT/IT identification package: six pretrained/instruction-tuned checkpoint pairs (five dense plus one MoE), endpoint-free same-history JS replay, symmetric matched-prefix MLP grafts and swaps at equal-width early/middle/late windows, matched random controls, and first-divergence-token counterfactuals. We apply this package to a within-model stabilization target, the **convergence gap**: the distance from each layer's prediction to the model's own final prediction. Across the six-family discovery run, IT models consistently have larger convergence gaps and commit later, suggesting that post-training changes the route by which final next-token predictions become stable rather than simply sharpening earlier. Under identical teacher-forced continuations, PT and IT same-layer output distributions still differ (dense-5 `JS(A', C)=0.106` before the late window and `0.169` in the final 20%). In dense-family graft/swap experiments, late IT→PT grafts recreate the delay (`+0.34` nats) and late PT→IT swaps collapse it (`-0.51` nats). A matched random residual-projection control on the same metric is near zero (`+0.003` versus `+0.327` for the true late graft), arguing against a generic late-perturbation account. First-divergence counterfactuals then separate token identity from final readout: middle layers transfer PT-vs-IT token identity more than late layers, while late IT MLPs are the strongest direct readout window in native IT runs (`+0.789` logits of IT-token support; 95% CI `[+0.754, +0.825]`) but are weak and compatible with zero when inserted into a PT upstream state (`+0.0035` logits; 95% CI `[-0.001, +0.009]`). The resulting picture is a **mid-to-late token-identity/readout split**: middle interventions are more tied to which PT- or IT-like token is exposed, and late MLPs are more tied to making that token win once the upstream state is IT-shaped. This reframes post-training as a change in prediction dynamics, not only final behavior, and gives mechanistic interpretability a protocol for studying how instruction-tuning effects become next-token probabilities beyond final scores or isolated behavioral vectors.

---

## 1. Introduction

Instruction tuning is normally evaluated at the surface: did the model follow the instruction, refuse the unsafe request, or speak in an assistant register? Those outputs matter, but they hide a mechanistic question. When a post-trained model produces a different token from its pretrained ancestor, did the whole stack gradually drift toward that token, or did a late computation turn an already-available candidate into the final prediction?

We study that question by turning "settling on an output" into a measurable trajectory. For every layer, we decode a next-token distribution and compare it to the model's own final distribution. The resulting **convergence gap** is a within-model stabilization metric: how far is this layer from the model's eventual next-token prediction? Across six paired pretrained (PT) and instruction-tuned (IT) families, IT models stay farther from their own final predictions for longer. The gap is broad under native decoding, is accompanied by endpoint-free PT↔IT separation under identical histories, and has its largest measured intervention effect under late MLP substitutions in matched-prefix graft/swap experiments.

The main contribution is not the observation that late layers sharpen logits. Prior work already shows late prediction sharpening, confidence correction, and compact activation-space correlates of behavior. Our claim is narrower and more useful: **post-training changes the layerwise route by which a model settles on its own next-token distribution**. Matched PT/IT grafts localize delayed stabilization most strongly to late MLPs; first-divergence tests separate mid-layer token identity from late-layer margin/readout.

This paper makes three claims:

1. **Instruction-tuned variants exhibit a convergence-gap signature.** Across six PT/IT pairs, IT models converge to their own final predictions later than PT models under native decoding.
2. **Late MLP substitutions have the largest measured effect on delayed stabilization.** Under identical token histories, late IT→PT grafts recreate the IT-like delay and late PT→IT swaps collapse it more than early or middle swaps.
3. **Token identity and final readout separate across depth.** First-divergence counterfactuals show that middle layers are more diagnostic of token identity, while late IT MLPs are most diagnostic of native IT-token margin and direct write-out.

We study six PT/IT pairs: Gemma 3 4B, Llama 3.1 8B, Qwen 3 4B, Mistral 7B, OLMo 2 7B, and DeepSeek-V2-Lite. We use `instruction tuning` and `IT` as readable shorthand for the instruction-following post-trained descendants of PT checkpoints; the recipes are heterogeneous, so the empirical comparison is PT-versus-post-training model diffing rather than isolation of one downstream training method.

The evidence ladder is:

| Step | Main question | Design | Main result |
|---|---|---|---|
| Discovery | Does post-training change stabilization dynamics? | Native decoding across six PT/IT families | IT stays farther from its own final distribution across the stack. |
| Matched-history check | Does PT↔IT separation persist under identical histories and endpoint-free readout? | Same teacher tokens, native same-layer JS | PT↔IT divergence persists without free-running-history or own-final endpoints. |
| Depth localization | Which MLP window most controls delayed stabilization? | Equal-width early/mid/late PT↔IT grafts and swaps, plus a matched random control | Late MLP substitutions have the largest measured effect on the convergence gap; the matched random control is near zero. |
| Identity/readout split | Do middle and late windows affect token choice and final margin differently? | First-divergence token tests and MLP write-out decomposition | Middle swaps transfer token identity more; late IT MLPs give the strongest native IT-token write-out, but only in IT-shaped upstream contexts. |

The reader should take away a single evidence-backed picture. Instruction tuning does not merely change the final token distribution; it changes when and how that distribution becomes stable. The convergence gap measures the delayed stabilization, matched-prefix interventions localize the largest tested leverage to late MLPs, and first-divergence analyses explain why this does not reduce assistant behavior to a late-only module: mid layers are more tied to candidate identity, late layers are more tied to making the selected candidate win.

**What is novel.** The paper's novelty is the conjunction of identification package, prediction-dynamics target, and identity/readout result.

| Contribution | What is new here | Main evidence |
|---|---|---|
| A cross-family PT/IT identification package | Matched histories, endpoint-free same-layer JS, symmetric equal-width early/middle/late MLP grafts and swaps, matched random controls, and first-divergence-token counterfactuals across paired checkpoints. This combination separates confounds prior work leaves coupled. | Matched-prefix graft/swap, native JS replay, random residual-projection control, and first-divergence counterfactuals. |
| A prediction-dynamics target for post-training | Convergence-to-own-final dynamics: a within-model stabilization trajectory that prior cross-model patching and behavioral-direction work has not directly localized. | Six-family native decoding, raw/tuned lens variants, commitment summaries. |
| A tested identity/readout split produced by the package | Candidate identity and final readout are separable under these readouts: mid layers transfer identity more; late IT MLPs write out IT-token margin more, but mainly in IT-shaped upstream contexts. | Native token-flow summaries, first-divergence token-choice tests, finite-difference MLP projections, and context-gating decomposition. |

We keep the main text focused on this positive story. Scope boundaries are still important, but they are concentrated in Appendix F: endpoint dependence, probe dependence, prompt formatting, weight-difference magnitude, the MoE case, and the fact that our interventions localize causal leverage on specified metrics rather than recovering a complete named feature circuit.

**Claim scope.** Pooled claims use the model set that was actually run and is interpretable for that design.

| Scope | Applies to | Use in the paper |
|---|---|---|
| Six-family claims | Gemma, Llama, Qwen, Mistral, OLMo, DeepSeek | Native convergence-gap discovery, commitment summaries, and broad geometric heterogeneity. |
| Dense-5 pooled claims | Gemma, Llama, Qwen, Mistral, OLMo | Main matched-prefix JS, symmetric graft/swap, token-flow/first-divergence tests, and judged behavior summaries. |
| DeepSeek separate MoE case | DeepSeek-V2-Lite | Reported where run, but not pooled into dense-family causal or behavioral averages because routing changes the intervention interpretation. |

Detailed scope boundaries and counter-interpretations are collected in Appendix F so the main text can follow the evidence chain directly without implying a complete mechanism.

---

## 2. Setup

### 2.1 Models

We compare PT/IT pairs from six families spanning dense and MoE architectures, hybrid and fully global attention, and a wide range of data and post-training regimes:

| Model | Layers | d_model | Attention | Pre-training data | Post-training |
|---|---|---|---|---|---|
| Gemma 3 4B | 34 | 2560 | GQA, hybrid local/global (5:1) | Undisclosed | Multi-stage post-training |
| Llama 3.1 8B | 32 | 4096 | GQA, all global | 15T tokens | Iterative supervised + preference optimization |
| Qwen 3 4B | 36 | 2560 | GQA, all global | 36T tokens, 119 languages | Multi-stage post-training |
| Mistral 7B v0.3 | 32 | 4096 | GQA, sliding window | Undisclosed | Instruct checkpoint |
| DeepSeek-V2-Lite | 27 | 2048 | MLA, MoE | 5.7T tokens | SFT-only chat checkpoint |
| OLMo 2 7B | 32 | 4096 | MHA, all global | OLMo-mix-1124 | SFT + DPO + RLVR |

OLMo 2's base recipe is centered on `OLMo-mix-1124` with a late `Dolmino-mix-1124` curriculum, so the earlier single-dataset shorthand was too coarse. DeepSeek-V2-Lite-Chat is an SFT-only chat checkpoint, which makes it a post-training outlier in addition to being the lone MoE family in our six-model set.

Accordingly, the paper's cross-family claim is about a shared PT↔IT phenotype across heterogeneous instruction-following post-training pipelines, not one homogeneous downstream recipe. The discovery curves use each model in its native prompting regime; matched-history replay, template controls, and graft/swap experiments then test the internal signature under controlled histories. For OLMo we use the retrained non-preview PT/IT checkpoints with shared tokenization. For DeepSeek we keep the MoE SFT-only case separate whenever architectural routing or post-training heterogeneity would make dense-family pooling misleading.

### 2.2 Architecture-agnostic pipeline and readouts

All core interventions are architecture-agnostic at the implementation level. Matched-prefix graft/swap and first-divergence tests operate on raw MLP activations and residual-stream states through a model-agnostic adapter system. No core claim depends on transcoders, SAEs, or model-specific decomposition dictionaries; Gemma transcoders appear only in a supplementary feature-level analysis.

The main target is the layerwise convergence trajectory `KL(p_l || p_L)`, where `p_l` is the decoded distribution at layer `l` and `p_L` is the same model's final-layer distribution. This is a within-model stabilization metric: it asks how far the current layer is from the model's eventual next-token prediction. It is symmetric across PT and IT because each model is evaluated against its own endpoint, but it is deliberately endpoint-relative.

We keep three companion readouts in the main story. First, matched-prefix native same-layer JS compares PT and IT output distributions at the same layer under the same teacher tokens, without using either model's final endpoint. Second, equal-width early/middle/late MLP grafts and swaps test which window has the largest causal leverage on delayed stabilization under identical histories, with a matched random residual-projection control for generic perturbation. Third, first-divergence token tests separate token identity from final margin/readout at prefixes where PT and IT would choose different next tokens.

Several secondary diagnostics are useful but not part of the reader's working-memory burden. Commitment summaries are robustness checks for the KL trajectory. Raw-lens variants and tuned-lens validation address probe dependence. `δ`-cosine, the cosine between each MLP update and the incoming residual stream, is a geometric marker of late revision rather than a mechanism by itself. Token-level summaries and finite-difference MLP projections are used only where the text asks a token-specific question. Appendix A lists the full diagnostic surface; the main claims are pinned to the four-step evidence ladder in the Introduction.

### 2.3 Data regimes

We use three data regimes, each tied to one role in the evidence ladder.

**Native discovery.** The cross-family free-running convergence analyses use 2,936 prompts spanning factual QA, reasoning, code, safety, format compliance, and custom assistant-style prompts. Each prompt is decoded greedily up to 512 tokens. This establishes the model-in-use signature.

**Controlled-history localization.** Matched-prefix runs use 400- and 600-prompt subsets derived from the same broader pool. These runs freeze teacher tokens, compare pure and intervened PT/IT branches under the same histories, and supply the matched-prefix JS, graft/swap, random-control, and first-divergence analyses. A reverse JS replay with PT-generated continuations is reported as an appendix check.

**Behavioral bridge.** The free-running behavioral follow-up uses a frozen 600-prompt subset of `eval_dataset_v2.jsonl`, emphasizing conversational prompts, assistant-register prompts, benign and harmful safety prompts, and format-sensitive items. The main behavioral estimates are LLM-judge results, with a completed two-rater blind human audit of the primary pairwise contrasts and pointwise judge-calibration labels.

Throughout, pooled paper claims are made on the five dense families. DeepSeek is used only as a separate MoE/SFT-only side case in experiments where we actually ran it, and it is not part of the canonical LLM-judged behavioral pool. Internal run identifiers and full file-level provenance are kept in the appendix and repository rather than in the main narrative.

### 2.4 Code and artifact availability

The code and paper-facing artifacts are publicly released at `https://github.com/yifan1207/structral-semantic-features`. The repository contains the model adapters, experiment packages, launch scripts, analysis scripts, prompt datasets, summary tables, bootstrap intervals, human-audit summaries, and final plots needed to audit the claims in this paper.

The release separates reviewer-facing evidence from regenerated bulk traces. We commit the summaries and plots from which manuscript numbers are read; raw activation arrays, model/probe tensors, tuned-lens checkpoints, and multi-gigabyte per-token JSONL/GZ traces are omitted from git for size, with their generation scripts and manifests included. Appendix G maps each main claim to the relevant source code, committed artifacts, and rerun entrypoints.

---

## 3. The Convergence-Gap Signature and a Mid-to-Late Identity/Readout Split

This section develops the central claim: **instruction-tuned variants exhibit a broad convergence-gap signature; matched PT/IT grafts localize delayed stabilization most strongly to late MLPs; and first-divergence tests separate mid-layer token identity from late-layer margin/readout**. Section 3.1 establishes the signature and adds an endpoint-free same-history JS companion check. Section 3.2 shows the late geometric shift that accompanies delayed stabilization. Section 3.3 gives the matched-prefix localization result: late MLPs have the largest tested effect on delayed stabilization. Section 3.4 separates token identity from readout before the behavioral section, so the behavioral effect sizes are interpreted as consequence evidence rather than as a claim that late layers create assistant behavior from scratch.

### 3.1 Broad convergence gap and delayed commitment across six families

The paper's first finding is simple: under native free-running decoding, instruction-tuned models remain farther from their own final output distribution than pretrained models do through much of the forward pass. We call this the **broad convergence gap**, and we treat delayed commitment as its discrete summary.

Figure 1 shows the clearest discovery visualization of the pattern, but not the whole evidential burden. Under the tuned lens, the pooled IT-minus-PT `KL(layer || own final)` gap is positive in the early, middle, and late thirds of the network (`+0.62`, `+0.56`, and `+0.30` nats). The raw lens tells the same qualitative story in five families and remains positive on late-half average in all six. We therefore use Figure 1 as a probe-based view of a broader signature rather than as standalone identification.

![Figure 1: Mean KL(layer ℓ ‖ final) per layer — PT (blue) vs IT (red). Tuned logit lens discovery view. In all six families, IT's curve lies above PT's across much of the stack, indicating a broad convergence gap whose later collapse appears as delayed commitment. Raw-lens and endpoint-free matched-prefix JS companions are reported in the text and appendix.](../results/exp09_cross_model_observational_replication/plots/L2_mean_kl_per_layer_tuned.png)

Commitment summaries tell the same story without needing another main figure: across five definitions, IT commits later than PT in all six families, and no definition shows an earlier IT commitment in any family. Raw-lens and threshold-sensitivity variants are reported in Appendix D and Appendix Figures S33-S37.

The matched-history check then freezes token history and changes the readout. We replay the same teacher continuations through PT and IT and compare **native same-layer** output distributions using symmetric JS divergence. On the dense-5 pool, matched-prefix `JS(A', C)` is `0.106` through the pre-late stack and `0.169` in the final 20%, with `final20 > pre` in all five dense families.

This does not directly validate `KL(layer || own final)`, but it shows that the PT↔IT separation is not confined to free-running histories or own-final endpoint comparisons. PT↔IT divergence survives identical histories and appears directly in same-layer native output distributions, so Figure 1 is not asked to carry the whole identification burden by itself. A reverse replay with PT-generated teacher continuations gives the same qualitative result and is reported in Appendix F.

Direct target-gap closure under the same replay is more mid-to-late distributed, which is exactly why we separate the claims: the PT↔IT distributional change is broad, token identity can shift before the end, and the largest tested effect on delayed stabilization lies late. Per-model curves and controls are in Appendix Figures S50-S52.

### 3.2 Geometric companion: late MLP opposition is a marker, not the mechanism

The convergence gap is the functional signature. The late geometric companion is a stronger IT-vs-PT shift toward MLP updates that oppose the accumulated residual stream. We measure this with `δ`-cosine, the cosine between each MLP update and the residual stream entering that layer. Across families, the shift is heterogeneous in size but consistently late-skewed: IT's MLP updates become more counter-aligned with the accumulated residual stream than PT's, and this geometric shift is more late-concentrated than the KL gap itself.

The logit lens decodes each layer's residual stream via `W_U · h_l`. PT models already exhibit late-layer negative `δ`-cosine, consistent with generic residual sharpening. The instruction-tuning effect is the additional late shift: IT updates are less aligned with the accumulated residual than PT updates, which is a plausible geometry for revising premature predictions. But this geometry is not automatically a token-specific mechanism. A residual-opposing vector can suppress a current direction, support a different token, or mostly move in directions irrelevant to the PT-vs-IT token pair. Section 3.4 therefore treats `δ`-cosine as a geometric marker and tests the actual token write-in directly.

This distinction matters empirically. The finite-difference MLP decomposition in §3.4 finds that pure IT late MLPs strongly increase the IT divergent token's logit through the full MLP update, but the negative-parallel component of that update does not itself increase the IT-vs-PT margin (`-0.0046` logits; 95% CI `[-0.009, -0.001]`). Increased late residual opposition is therefore a geometric marker of post-training's late readout shift; the IT-token support is carried by token-specific full-update write-in rather than by the negative-parallel component alone.

The full six-family `δ`-cosine profile is moved to Appendix Figure S20. The main-text claim is the additional IT-vs-PT increase in late-layer negativity, not the mere existence of negative late updates.

The late `δ`-cosine shift is directionally consistent in all six families and heterogeneous in magnitude. Pooled across families, the IT-minus-PT shift is near zero early, modest in the middle, and largest late. The broad convergence gap is the cross-family functional signature; increased late MLP opposition is its clearest geometric companion.

Families with sustained late opposition show a more distributed late adjustment, while families with terminal spikes show a sharper final adjustment. Detailed onset, weight-difference, and family-by-family geometry analyses are deferred to the appendix.

### 3.3 Causal localization: late stabilization within a broader mid-to-late pattern

The free-running results in §3.1–§3.2 establish the signature. Section 3.3 asks how it is implemented. The spine is cross-family matched-prefix evidence: a depth ablation shows that late MLPs most reliably recreate the *late delayed-convergence* signature, and a mirrored graft/swap follow-up shows that the same late window has the largest collapse effect in IT. The behavioral and JS evidence, however, is more mid-to-late distributed. We therefore treat the result as an identity/readout split rather than a late-confined account: middle layers are more implicated in PT-vs-IT token identity under first-divergence tests, while late layers are more implicated in margin/readout. This is the key internal localization step: it turns cross-model patching from a task-specific or single-pair diagnostic into a causal test of PT↔post-training prediction dynamics. Section 3.4 makes the identity/readout split explicit before we turn to free-running behavior.

#### 3.3.1 Cross-family matched-prefix depth ablation localizes the late stabilization effect

This experiment is the paper's first cross-family matched-prefix causal test and the first half of the main internal identification backbone. Weight grafting between base and post-trained variants is closely related to prior work on MLP editing, skill localization, and layer-scoped alignment interventions; our version asks whether IT MLP parameters at different depths recreate the convergence signature once token history is held fixed. For each prompt, `C` is the full IT model, `A'` is the PT backbone teacher-forced to `C`'s emitted continuation, and `B` is the same PT backbone with IT MLPs grafted into one equal-width window under that same continuation. We also run a raw-versus-template control, and we treat the five dense families as the main pooled set while reporting DeepSeek separately because its MoE swap also changes routing behavior.

The main result is simple for the convergence-gap readout. Under matched prefix, late IT grafting increases final-20% `KL(layer || own final)` relative to `A'` in all five dense families, with nearly identical raw and templated branches, and it also moves the PT backbone closer to the IT teacher under the shared PT readout. This raw-versus-template agreement matters because the discovery curves use native prompting: it argues that the late delayed-stabilization effect is not only a chat-template surface artifact. In this coarse window-intervention sense, late IT MLP weights can impose an IT-like delayed-convergence signature even when the teacher tokens are fixed. DeepSeek is directionally concordant on the same readouts but is kept separate as an MoE case.

The depth-specific pattern is sharp for delayed stabilization. Every branch is evaluated by the same within-model stabilization metric, and every grafted branch has its own endpoint, so the comparison is about which equal-width substitution produces the IT-like delay. On the dense-family mean, the final-20% KL effect is `+0.34` nats for the late graft versus `-0.03` early and `-0.05` mid. The same late window is also the only one that moves final-20% `δ`-cosine and threshold-free commitment in the consistently IT-like direction on average. Early and middle grafts perturb local dynamics and can improve teacher similarity, which is exactly the clue that PT↔IT behavioral change is broader than the final window. What they do not recreate is the *late-region delayed-convergence signature*. Appendix Table S18b reports per-family raw effects and leave-one-family-out means; the dense-family result is not driven by Mistral, whose raw late effect is the smallest in the dense set.

![Figure 2: Depth-specific matched-prefix graft ablation. Equal-width early, middle, and late IT-MLP grafts are compared under identical teacher-forced token histories. Early and middle grafts can perturb local dynamics, but only the late graft consistently recreates the late delayed-convergence signature on the primary final-20% KL metric across the five dense families; DeepSeek is reported separately as a concordant MoE case.](../results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/depth_ablation_paper_main.png)

The matched-prefix graft is the first internal localization result for delayed stabilization: late MLPs most strongly recreate delayed convergence under identical token histories. This motivates the next question, tested below: whether the same late window controls the identity of the instruction-tuned token, the margin by which it wins, or both.

#### 3.3.2 Symmetric matched-prefix graft/swap shows the largest late-window effect in both directions

The mirrored follow-up asks for the IT-side counterpart: if late IT MLPs can impose the delay on PT, do late PT MLP swaps also remove it from IT? We also add output-relevant late-stage summaries that go beyond `δ`-cosine alone.

![Figure 3: Symmetric matched-prefix graft/swap localization. Left: PT-side final-20% KL-to-own-final deltas for early, middle, and late IT-MLP grafts relative to `A'`. Center: IT-side deltas for early, middle, and late PT-MLP swaps relative to `C`. Right: dense-family correlations between late KL shifts and output-relevant late-stage summaries. The late window has the largest convergence-gap effect in both directions.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png)

Late has the largest bidirectional window effect on the convergence-gap metric in the dense-family pool, with DeepSeek reported separately as the MoE case. On the PT side, the dense-family mean final-20% KL effect is `+0.34` nats for `B_late` versus slightly negative early and mid values. On the IT side, the mirrored late PT swap produces the largest collapse of IT's delay, with a dense-family mean `D_late - C = -0.51` nats versus `-0.10` early and `-0.23` mid. The auxiliary late-geometry ordering points the same way. In this coarse window-intervention sense, late IT MLPs are sufficient to impose the measured delay, and late PT MLPs are sufficient to collapse it. This is not a claim that the late window is mechanistically necessary in isolation from upstream IT computation.

A matched random-control follow-up checks that this is not just "late layers are fragile." We replace the actual IT-minus-PT late MLP residual effect with matched random residual-projection controls and keep the same final-20% KL readout. In the dense-family pool, the true late graft effect is `+0.327` nats, while the matched random residual-projection effect is `+0.003` nats, for a true-minus-random specificity margin of `+0.324` nats. The random control does not make a weight-diff-normalized efficiency claim, but it does rule out the simplest generic same-window perturbation account for the main late KL effect.

The best output-relevant summaries are also broader than `δ`-cosine alone. Token-support and alternative-suppression summaries modestly outperform residual-opposition alone as predictors of later late-region KL. The mechanism picture is therefore sharper than pure anti-residual opposition: the late IT-like update supports the eventual teacher token while weakening premature commitment to the current dominant alternative. In the identity/readout interpretation, this is the late write-in step: once an IT-like candidate is available, late MLPs increase its margin against raw-continuation alternatives. Appendix F gives the full scalar-summary table and mediation boundary.

Taken together, the symmetric matched-prefix result is the paper's clearest causal localization evidence for the convergence gap. Late IT MLPs are the window whose installation most reliably recreates delayed convergence in PT and whose replacement most reliably collapses it in IT. That result localizes the late stabilization/readout effect in the tested window design; it leaves open, and in fact motivates, separate tests of token identity.

### 3.4 Token-level split: mid identity, late margin/readout

The results above localize delayed stabilization most strongly to late MLPs. We now ask what that late stage does at tokens where PT and IT would actually choose different continuations. The first-divergence token-choice test conditions on the first PT/IT divergent token and asks which grafted or swapped pipeline chooses the PT token, the IT token, or another token at that shared prefix. The companion MLP write-out test holds the same kind of prefix fixed and measures each MLP window's finite-difference logit write-in to the IT token, PT token, pipeline token, top alternatives, and residual-opposing component. This turns the broader story into two concrete readouts: token identity and token margin. All intervals in this subsection are 95% percentile bootstrap intervals over prompt-level dense-5 first-divergence records, resampling within model family and recomputing paired differences and source-decomposition terms inside each bootstrap sample.

The token-choice test gives the identity/readout split. The cleaner identity control is raw-shared prompting, where both variants receive the same surface format. In the dense-5 holdout, `PT + IT mid` matches the IT divergent token more often than `PT + IT late` (`26.0%`, 95% CI `[24.5%, 27.7%]`, versus `17.6%`, 95% CI `[16.2%, 18.9%]`). The mirror comparison points the same way: `IT + PT mid` transfers the PT token more often than `IT + PT late` (`31.2%`, 95% CI `[29.6%, 32.9%]`, versus `20.8%`, 95% CI `[19.4%, 22.3%]`). Middle swaps are therefore more diagnostic of divergent-token identity. Native IT-host swaps give the readout side: replacing IT late layers with PT late layers produces the largest single-window IT-vs-PT margin drop (`13.25` logits, 95% CI `[12.91, 13.61]`) relative to early (`11.53`, 95% CI `[11.20, 11.88]`) and middle (`12.01`, 95% CI `[11.66, 12.35]`) swaps, while the late swap preserves IT token identity better than early or middle swaps (`67.7%` IT-token matches, 95% CI `[66.2%, 69.2%]`, versus `61.0%`, 95% CI `[59.4%, 62.7%]`, and `59.3%`, 95% CI `[57.6%, 60.9%]`). Late layers are comparatively readout-heavy: they matter most for making an already IT-like token win by margin, not for being the unique origin of that token identity.

The MLP write-out test makes the late readout direct. In native dense-5 pure IT runs, the MLP support for the IT divergent token is overwhelmingly late: early `-0.041` logits (95% CI `[-0.049, -0.032]`), middle `+0.021` (95% CI `[+0.011, +0.032]`), late `+0.789` (95% CI `[+0.754, +0.825]`). The PT→IT change in that same support is also concentrated late: early `+0.034` (95% CI `[+0.027, +0.042]`), middle `+0.070` (95% CI `[+0.059, +0.080]`), late `+0.715` (95% CI `[+0.683, +0.747]`). This is the strongest evidence that late IT MLPs become especially important for shaping the final output distribution toward IT-specific tokens. But the effect is not a standalone late module. Removing IT late layers from an IT host reduces late-window MLP IT-vs-PT margin by `+0.292` logits (95% CI `[+0.278, +0.306]`), larger than early (`+0.222`, 95% CI `[+0.202, +0.240]`) or middle (`+0.160`, 95% CI `[+0.139, +0.179]`) swaps; adding IT late layers to a PT host changes the same margin by only `+0.0035` logits (95% CI `[-0.001, +0.009]`), which is compatible with zero, and adding IT late on top of IT middle changes it by `+0.0119` logits (95% CI `[+0.007, +0.016]`). The 2x2 decomposition says the same thing: upstream IT context (`+0.403`, 95% CI `[+0.360, +0.444]`) is larger than the late-weight main effect (`+0.148`, 95% CI `[+0.139, +0.157]`), with positive interaction (`+0.288`, 95% CI `[+0.277, +0.301]`). Late readout is real, but it is context-gated by earlier IT computation.

![Figure 4: First-divergence identity/readout synthesis with 95% bootstrap CIs. Panel A: native IT-host swaps show late is the largest single-window readout loss. Panel B: pure IT MLP support for the IT divergent token is overwhelmingly late. Panel C: source decomposition shows upstream IT context is larger than the late-weight main effect, with positive interaction. Panel D: full MLP update, not the negative-parallel component alone, carries IT-vs-PT margin write-in.](../results/paper_synthesis/exp20_exp21_handoff_synthesis.png)

The residual-opposition story must therefore be phrased carefully. Pure IT late MLPs are less residual-aligned than pure PT late MLPs (`δ`-cosine IT-minus-PT `-0.080`; 95% CI `[-0.081, -0.079]`), matching the paper's geometric signature. However, the negative-parallel component does not directly support the IT token against the PT alternative: in pure IT late layers, full-update IT-vs-PT margin write-in is `+0.768` logits (95% CI `[+0.729, +0.805]`), while the negative-parallel component's contribution is `-0.0046` logits (95% CI `[-0.009, -0.001]`). Increased late residual opposition is a geometric marker of post-training's late readout shift; the mechanism evidence is token-specific full-update write-in.

Native final-token flow supplies a complementary chronology view, but it is no longer asked to carry the strongest causal claim. On the dense-5 pool, late IT windows increase the eventual token's margin more than middle windows for both format-like and content-like tokens, while PT is much flatter. The matched-prefix chronology is lower-resolution because the older traces store ranks and top-k sets, not per-layer logits, so its plots are kept in Appendix Figures S56-S57. It remains useful as a chronology check: format-like tokens become teacher-rank-positive only late, while content-like tokens show more middle-window promotion. Strict mid-selected/late-helped rates are modest rather than universal, which is exactly why the paper claims a coarse identity/readout split rather than a single late-only circuit.

Taken together, these token-flow and first-divergence analyses support a genuine but qualified mid-to-late identity/readout split. Middle layers are more diagnostic of candidate exposure, reshuffling, and token identity transfer. Late IT MLPs are more diagnostic of final readout: in native IT contexts they provide the largest direct IT-token write-out and margin contribution, and the same late window produces the delayed-stabilization component of the convergence gap. The convergence gap is not the birthplace of instruction-tuned behavior; it is the readout cost of making instruction-tuned candidates dominate the final next-token distribution.

### 3.5 LLM-judge behavioral bridge with human audit: strongest late IT-side degradation and partial PT-side movement

The matched-prefix and first-divergence results establish the internal localization and identity/readout split. The behavioral follow-up asks whether the same intervention family changes actual assistant outputs under natural decoding, but it should be read as an **LLM-judge behavioral bridge with human audit**, not as a standalone human preference benchmark. We evaluate eight deterministic free-running pipelines on a frozen 600-prompt core subset of `eval_dataset_v2`, pooling the five dense families only. DeepSeek is not part of the canonical judged behavioral run, so it does not contribute to the pooled behavioral claims in this section.

![Figure 5: Free-running LLM-judge behavioral bridge. Left: IT-side pooled pointwise worsening on assistant register (G2) and benign false-refusal control (S2), where the late PT swap is the largest window on these two endpoints. Center: dense-family pooled blind pairwise judging for PT late graft vs PT baseline. Right: dense-family pooled blind pairwise judging for IT baseline vs late PT swap. The cleanest judge-based claim is strongest late-window degradation on the IT side; late grafting into PT shows smaller but reliable movement under pairwise judging.](../results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_paper_behavior_main.png)

The clearest late result is IT-side degradation under the LLM judge. On the dense-family pooled pointwise metrics, the IT-side ordering is `late > mid > early` for both primary endpoints: assistant register (`G2`) worsens by `+0.42` (95% CI `[+0.27, +0.62]`) for `D_late`, and benign false-refusal control (`S2`) worsens by `+0.05` (95% CI `[+0.02, +0.10]`). Late IT-side `G2` loss is positive in every dense family. Blind pairwise judging says the same thing more directly: the intact IT model is preferred over `D_late` on `75.3%` of `G2` comparisons (95% CI `[72.8%, 77.9%]`) and `76.3%` of `S2` comparisons (95% CI `[72.0%, 80.3%]`). The same late window whose replacement most strongly collapses IT's internal delay therefore also produces clear LLM-judge degradation under natural decoding.

The PT side shows measurable but partial movement toward IT behavior. In blind pairwise judging, `B_late` beats `A` on `56.3%` of resolved assistant-register items and `63.8%` of resolved benign-safety items. The direct PT-versus-IT baseline gives scale: in the same dense-family pairwise setup, intact `C` beats intact `A` on `98.0%` of `G2` comparisons (95% CI `[97.2%, 98.8%]`) and `96.8%` of `S2` comparisons (95% CI `[94.9%, 98.4%]`). Late grafting into PT therefore moves behavior in the IT direction without recovering the full PT-to-IT pairwise preference gap.

The completed blind human audit agrees with these directions on the four primary pairwise contrasts, but with many unresolved votes. On resolved human comparisons, intact `C` is preferred over `D_late` at `70.6%` for `G2` (95% CI `[64.0%, 76.9%]`) and `66.9%` for `S2` (95% CI `[57.5%, 75.8%]`); `B_late` is preferred over `A` at `60.5%` for `G2` (95% CI `[53.7%, 67.3%]`) and `65.8%` for `S2` (95% CI `[58.6%, 72.6%]`). Ties or both-bad labels account for `51-80%` of votes depending on contrast, and inter-rater agreement is heterogeneous, so the audit is directional confirmation rather than a replacement for the larger LLM-judge estimates.

The effect is targeted under the LLM judge: `D_late` harms assistant register most strongly on conversational and register-sensitive prompt buckets, especially `GOV-REGISTER` (Appendix Figure S49). The behavioral bridge is asymmetric but meaningful as consequence evidence: the late window is the strongest tested IT-side degradation window, and late grafting into PT produces measurable but partial movement. It supports "late readout contributes to assistant-like behavior," not "late MLPs recover assistant behavior." Appendix B gives the judge setup, bias controls, completed human-audit results, and remaining caveats.

## 4. Deeper Characterization

### 4.1 Geometry beyond the main result is mixed

We also tested whether the broad convergence gap has a simple representation-geometric counterpart beyond the late logit-space effect itself. Multi-estimator dimensionality, covariance-spectrum, and attention-entropy follow-ups are mixed across families, so they remain exploratory. The main result does not require a universal rank, width, or entropy story: it is a forward-pass prediction-dynamics signature with a tested mid-to-late identity/readout split.

### 4.2 Relation to known behavioral directions

Our results fit naturally with prior work on refusal directions, safety layers, and assistant-style control, while sharpening a different question. Refusal directions (Arditi et al., 2024) and the Assistant Axis (Lu et al., 2026) suggest that post-training leaves compact behavioral control signals in activation space. Du et al. (2025) separates post-training effects across knowledge, truthfulness, refusal, and confidence. Li et al. (2025) and Chaudhury (2025) are especially important because they localize safety or preference effects to middle bands under readouts centered on malicious-vs-normal discrimination and HHH-style preference behavior. We read those results as evidence about **where safety- or preference-relevant information can be represented or selected**. Our paper asks the next-token counterpart: once a PT- or IT-like token candidate is present, which layers make it dominate the final predictive distribution?

This is a real distinction, not just a definitional escape hatch. The first-divergence token-choice test directly checks token identity at the first PT/IT divergence and finds the mid-layer result that the prior literature would predict: middle swaps transfer opposite-model token identity more than late swaps. The companion MLP write-out test then asks a different question at the same kind of prefix and finds that pure IT late MLPs provide the largest direct support for the IT divergent token, with the PT→IT support increase concentrated late. The resulting synthesis is a broader middle-to-late window-level pattern with a late-skewed output reconciliation point. Middle layers may encode, select, or route safety/preference/persona information; late MLPs are where our matched-prefix interventions and token-margin analyses find the clearest final-token write-in. In the language of this paper, middle layers are more associated with instruction-tuned token identity, and late layers are more associated with making that token win. Appendix F states the remaining head-to-head boundary explicitly.

### 4.3 Feature-level results are supplementary

Gemma feature-level analyses are consistent with substantial late post-training reorganization: IT redistributes activation mass across a broader feature repertoire, and the fraction of activation mass attributable to IT-amplified features rises sharply in the late reconciliation region. Those findings are useful for intuition and for locating a layer-resolved alignment tax, but they depend on variant-matched transcoders and remain supplementary to the main architecture-agnostic evidence chain.

---

## 5. Discussion

### 5.1 What the current evidence shows

The evidence chain is protocol-led: the convergence gap gives the PT/IT prediction-dynamics target, and matched histories plus symmetric graft/swap plus first-divergence counterfactuals identify which coarse windows affect it. The first result is cross-family: IT predictions remain farther from their own final distribution for longer across all six families. The second result is causal localization: under matched prefix, late IT MLP grafts have the largest tested effect on delayed stabilization on the PT side, and late PT MLP swaps have the largest tested collapse effect on the IT side. The LLM-judge behavioral bridge provides consequence evidence without reducing assistant behavior to a self-contained late module, and the completed human audit directionally confirms the primary pairwise bridge while exposing high tie/both-bad rates. Native token-flow and first-divergence tests then supply the identity/readout bridge: middle layers more strongly affect candidate/token identity; late IT MLPs give the largest native IT-token write-out and margin contribution. Read together, these results support broad whole-stack change under native decoding, strongest tested late leverage on delayed stabilization, and a mid-to-late split between token identity and final margin/readout.

The novelty is a methodological package applied to a target prior work has not directly studied. The package is paired-checkpoint cross-family scope, matched-prefix MLP graft and swap interventions at equal-width early/middle/late windows, an endpoint-free same-history JS readout, matched random controls, and first-divergence-token counterfactuals separating identity from readout. This combination matters because it separates confounds that nearby work leaves coupled: Lad et al. (2024) and Joshi et al. (2025) characterize late prediction sharpening and confidence correction; Prakash et al. (2024) patches related base/fine-tuned models for entity tracking; Du et al. (2025) compares base and post-trained models across knowledge, truthfulness, refusal, and confidence; and Arditi et al. (2024), Stolfo et al. (2024), and Lu et al. (2026) show compact behavioral directions and instruction-following activation control. The target here is convergence-to-own-final prediction dynamics, and the result on that target is a tested mid-to-late split between PT/IT token identity and final-token margin/readout.

Put differently, the paper changes the question from "do late layers sharpen predictions?" to "can we causally separate how post-training changes histories, endpoints, token identity, and final readout?" The convergence gap is the target that makes this question measurable; the protocol is what makes it identifiable; the identity/readout split is the result the protocol exposes.

### 5.2 The identity/readout interpretation: mid token identity, late next-token reconciliation

A late-layer stabilization result is fully compatible with a gap that appears earlier in the stack. Our main observable is intentionally a stabilization-to-own-decision metric: if post-training shifts the candidate token distribution before the end, and late layers then increase the selected candidate's margin against raw or premature alternatives, earlier IT layers should be farther from their own endpoint until that readout step has run. That is the operational signature tested here: not a named latent variable, but a delayed-stabilization metric with its largest measured intervention effect late. The broad free-running gap and the late matched-prefix localization result are therefore complementary: the former describes the whole trajectory of the model in use, while the latter identifies where intervention has the largest effect on delayed stabilization.

The token analyses make this interpretation concrete. Native final-token flow shows that late IT windows raise the eventual token's margin more than middle windows for both format and content, whereas PT is flatter. The first-divergence token-choice test adds the identity check: under a shared raw prompt, mid swaps transfer first-divergent token identity more than late swaps, while native late swaps produce the largest IT-host margin loss. The MLP write-out test adds the direct readout check: pure IT late MLPs support the IT divergent token much more than early or middle MLPs, with bootstrap intervals reported in §3.4, but adding late IT layers to a PT host leaves the same fixed-prefix margin compatible with zero (`+0.0035` logits; 95% CI `[-0.001, +0.009]`). The clean synthesis is therefore not "mid causes behavior and late is irrelevant," nor "late causes assistant behavior." It is a coarse window-level split: mid layers are more tied to candidate identity; late layers are more tied to making the selected candidate dominate next-token prediction once the upstream state is IT-shaped.

This also explains why our result can coexist with mid-layer safety and preference localization. The PT-to-post-training change is broad, and middle layers can carry safety-, preference-, or persona-relevant information. The late result says something sharper and narrower: when the target is delayed stabilization to the model's own final output, or direct native IT-token write-out at a first-divergence prefix, the clearest tested control point is late. The convergence gap is not where instruction-tuned behavior begins; it is the readout cost of turning selected instruction-tuned candidates into final next-token predictions.

### 5.3 Scope boundaries

The evidence supports a window-level account: a broad convergence gap, late-MLP leverage on delayed stabilization, and a mid-to-late split between token identity and token margin. It does not yet identify a named middle-layer feature feeding a named late write-in direction. We also treat grafts and swaps as causal leverage on specified metrics, not as complete mechanism recovery. Appendix F collects the detailed boundaries: endpoint and probe dependence, prompt formatting, weight-difference magnitude, the DeepSeek MoE case, weak PT-host late effects, and the residual-opposition decomposition.

### 5.4 Most useful next causal tests

The token-flow and first-divergence analyses close an important gap: we now have direct token-flow, identity/readout evidence, and MLP-level write-in evidence rather than only an interpretation inferred from earlier late-stage summaries. The remaining causal question is no longer whether middle and late windows interact in the current design; they do, and the additional fixed-prefix effect of adding IT late on top of IT middle is small on the PT host (`+0.0119` logits; 95% CI `[+0.007, +0.016]`). The remaining question is why late IT readout is so strongly context-gated.

The most useful next experiment is therefore not another coarse middle/late factorial, but a higher-resolution context-gating test. Start from an IT host and selectively replace or interpolate the upstream residual state entering the late window while keeping late IT weights fixed. Conversely, start from a PT host and supply IT-like upstream residual states while keeping either PT or IT late weights fixed. The target is the interaction revealed by the source decomposition: upstream context (`+0.403`, 95% CI `[+0.360, +0.444]`) is larger than the late-weight main effect (`+0.148`, 95% CI `[+0.139, +0.157]`), with positive interaction (`+0.288`, 95% CI `[+0.277, +0.301]`). This experiment would test whether late IT MLPs require an IT-shaped residual state, a particular token candidate in the top-k set, or broader chat-template/context features to produce their write-out.

A second high-value follow-up is a higher-resolution matched-prefix rerun that stores per-layer logits rather than only ranks and top-k sets. The current matched-prefix chronology is supportive but limited by the older logging format. Re-running the same `A'`, `B_early`, `B_mid`, `B_late`, and `C` branches with per-layer logit capture would let us measure windowed target support, alternative suppression, and margin growth under identical histories rather than relying on rank-only proxies.

A third, more surgical follow-up is a finer-grained decomposition inside the late MLP update itself. The current MLP decomposition shows that the negative-parallel component does not directly carry IT-vs-PT margin, even though the full late MLP update strongly does. We can decompose the late update into token-supporting, alternative-suppressing, residual-opposing, and orthogonal components, then test which components recover delayed stabilization and IT-token margin under matched-norm controls. The context-gating test answers what late IT weights need; the higher-resolution matched-prefix rerun answers the chronology-under-identical-history question; this component test answers what part of the late update performs the readout.

---

## 6. Related Work

**Table 1: What prior work already shows, and what this paper adds.** The closest precedents are real and important; the novelty claim is the conjunction of identification package, prediction-dynamics target, and the mid-to-late token-identity/readout split.

| Paper | What prior work already shows | What this paper adds |
|---|---|---|
| Lad et al. (2024) | A generic depth-stage picture: prediction ensembling followed by residual sharpening. | A PT↔post-training prediction-dynamics claim: instruction-tuned descendants show delayed stabilization, with strongest tested late MLP leverage on that metric. |
| Joshi et al. (2025) | Late confidence correction and low-dimensional calibration directions after decision certainty. | A different target: convergence-to-final dynamics and candidate-to-readout transfer, tested across paired PT/IT families and localized with matched-prefix graft/swap. |
| Prakash et al. (2024) | CMAP-style cross-model activation patching can reveal improved mechanisms in related models. | Symmetric equal-width early/mid/late MLP grafts and swaps across paired families, aimed at the route from post-training features to next-token prediction rather than entity tracking. |
| Wu et al. (2024); Du et al. (2025) | Post-training changes behavior, knowledge/truthfulness/refusal directions, and confidence. | A prediction-dynamics target plus endpoint-free same-history replay, causal localization, and natural-decoding consequence tests. |
| Li et al. (2025); Chaudhury (2025) | Safety/preference behavior can localize to middle or mid-to-late layer bands. | A complementary readout: middle layers more strongly affect candidate identity, while late MLPs have the strongest tested leverage on delayed stabilization and margin/readout. |
| Arditi et al. (2024); Lu et al. (2026); Joad et al. (2026) | Refusal and assistant/persona behavior admit compact activation-space summaries, sometimes with multiple related directions. | A convergence-gap framing of how post-training features become next-token predictions, tested with paired PT/IT histories, swaps, and first-divergence counterfactuals. |
| Stolfo et al. (2024) | Instruction-following activation vectors can control output constraints such as format, length, and word inclusion, including transfer from instruction-tuned to base models. | We do not claim activation-vector control for instruction following is new. The paper-level novelty is the cross-family PT/IT identification package applied to convergence-to-own-final dynamics. |
| Turner et al. (2023); Zou et al. (2023); Panickssery et al. (2024) | Activation additions, representation engineering, and contrastive activation addition can control high-level behavior through directions in activation space. | A different target and evidence standard: paired PT/IT prediction trajectories, endpoint-free same-history JS, and window-level graft/swap causality rather than only inference-time behavioral control. |

Late sharpening is not new by itself. The logit-lens and tuned-lens line of work already makes late-layer prediction dynamics visible (nostalgebraist, 2020; Belrose et al., 2023), and Lad et al. (2024) explicitly frame late computation as residual sharpening after earlier candidate construction. Joshi et al. (2025) is the closest phenomenon-level neighbor: it studies how calibration evolves through layers and identifies an upper-layer confidence-correction phase. The open question left by this literature is not whether late layers sharpen, but what post-training changes about the route from candidate representations to final next-token prediction. Our contribution is to make that route identifiable in PT↔post-training pairs: define a stabilization target, show it across paired families, test it with endpoint-free same-history JS, and use symmetric interventions plus first-divergence counterfactuals to separate candidate identity from margin/readout.

Post-training model-diffing papers ask nearby questions. Wu et al. (2024) explains instruction-conditioned behavior shift with attribution, attention, and FFN analyses, while Du et al. (2025) compares base and post-trained models through knowledge, truthfulness, refusal, and confidence. These works establish that post-training reshapes internal mechanisms. We target a different observable: the layerwise path by which each model reaches its own final next-token prediction. That target is what makes the matched-prefix JS replay and the graft/swap localization meaningful.

Activation-patching and cross-model patching supply the methodological backdrop. Prakash et al. (2024) is the closest method precedent through CMAP-style activation patching across related models. We also follow Heimersheim and Nanda (2024) in treating patching-style evidence as causal leverage on a specified metric, not as automatic mechanism completion. Our design changes the unit of analysis: instead of a task-specific mechanism in one base/fine-tuned setting, we use equal-width early/mid/late swaps across paired model families to localize effects on convergence-to-final dynamics, then use first-divergence counterfactuals to separate token identity from margin/readout. The design makes the PT→IT route from divergent token candidates to next-token predictions experimentally testable.

Behavioral-vector papers explain why a late intervention can matter without being the whole assistant behavior. Arditi et al. (2024) shows a compact refusal direction, Lu et al. (2026) identifies an Assistant Axis, and Joad et al. (2026) argues refusal is not exhausted by a single direction even when linear activation control can act like a shared knob. Stolfo et al. (2024), Turner et al. (2023), Zou et al. (2023), and Panickssery et al. (2024) make the broader activation-control prior clear. Our claim is therefore not that behavioral vectors exist. In this version, the paper-level claim is the PT/IT identification package and the identity/readout split it exposes on convergence-to-own-final dynamics. Middle layers more strongly affect candidate identity, while late MLPs most strongly affect delayed stabilization and final margin/readout.

---

## 7. Conclusion

We introduced a cross-family symmetric matched-prefix protocol for studying how post-training changes prediction dynamics. Applied to six PT/IT checkpoint pairs, the protocol produces three results that prior tools did not directly establish together: post-training delays each model's stabilization to its own final prediction; this delay survives endpoint-free identical-history replay and has its largest measured intervention effect under late MLP substitutions with matched prefixes; and the late effect is a context-gated readout contribution rather than a self-contained assistant module.

The same late intervention family shows behavioral effects under LLM judging during natural decoding, with a completed human audit directionally confirming the primary pairwise comparisons while leaving the behavioral evidence bounded by high unresolved rates and heterogeneous agreement. The resulting story is compact: post-training changes the layerwise process by which models settle on their own outputs; middle layers more strongly affect instruction-tuned token identity; and late MLPs help selected candidates win the next-token competition by increasing teacher-token support and IT-token margin. Native token-flow and matched-prefix chronology analyses show category-dependent candidate exposure and late write-in. First-divergence token-choice tests add the clean split: mid swaps transfer token identity more, while late layers contribute more to margin/readout. MLP write-out tests make the late readout explicit: pure IT late layers provide the largest direct support for IT divergent tokens, but this write-out is context-gated and is not carried by the negative-parallel residual component alone. The next decisive step is to identify which upstream IT state features gate late readout and which late-update components carry token-specific support.

---

## References

Aghajanyan, A., et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL 2021*. arXiv:2012.13255.

Ansuini, A., et al. (2019). Intrinsic Dimension of Data Representations in Deep Neural Networks. *NeurIPS 2019*. arXiv:1905.12784.

Arditi, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. *arXiv:2406.11717*.

Belrose, N., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. *COLM 2024*. arXiv:2303.08112.

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

Geva, M., Schuster, R., Berant, J., & Levy, O. (2022). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2022*. arXiv:2012.14913.

Gold, J. I., & Shadlen, M. N. (2007). The Neural Basis of Decision Making. *Annual Review of Neuroscience*, 30, 535–574.

Guerdan, L., Barocas, S., Holstein, K., Wallach, H., Wu, S., & Chouldechova, A. (2025). Validating LLM-as-a-Judge Systems under Rating Indeterminacy. *NeurIPS 2025*. OpenReview.

Hyland, K. (2005). *Metadiscourse: Exploring Interaction in Writing*. Continuum.

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

Nanda, N., & Lieberum, T. (2022). A Mechanistic Interpretability Analysis of Grokking. *ICLR MATH-AI Workshop 2023*.

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

Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.

---

## Appendix A: Supplementary Figures

[Figure S17: Generation-step × layer heatmap for Gemma 3 4B. Four panels showing δ-cosine stability across generation steps.](figures/it_plot10_generation_heatmap.png)
[Figure S18: Per-layer weight change localization (PT → IT) across six families. Gemma shows late-layer concentration; others show more diffuse changes.](../results/exp09_cross_model_observational_replication/plots/L3_weight_diff_6panel.png)

[Figure S20: δ-cosine profiles across six model families. IT (solid) vs PT (dashed). Gemma shows the largest late IT-vs-PT shift; Llama shows the weakest sustained shift because its PT variant already exhibits substantial late opposition.](../results/exp09_cross_model_observational_replication/plots/L1_delta_cosine_6panel.png)

**Table S18a: depth-ablation effect normalized by window weight-change proxy.** `Δ KL` is the matched-prefix PT-side final-20% `B_window - A'` effect. `Mean ΔW` is the mean per-layer MLP weight-change proxy from Figure S18 over the same graft window. `Δ KL / Mean ΔW` is a descriptive scale-normalized diagnostic, not a formal parameter-efficiency estimand. Late is the largest raw effect and the largest normalized effect in every family, while the largest weight-change window is not consistently late.

| Family | Early ΔKL / Mean ΔW | Mid ΔKL / Mean ΔW | Late ΔKL / Mean ΔW | Largest ΔW window |
|---|---:|---:|---:|---|
| Gemma 3 4B | `-5.5` | `-14.0` | `146.6` | Mid |
| Llama 3.1 8B | `-236.7` | `-148.9` | `453.2` | Late |
| Qwen 3 4B | `-38.8` | `-60.1` | `184.4` | Late |
| Mistral 7B | `878.2` | `736.2` | `1754.4` | Mid |
| OLMo 2 7B | `0.6` | `27.1` | `130.3` | Mid |
| DeepSeek-V2-Lite | `-11.4` | `-55.1` | `234.7` | Late |

The dense-family mean raw effect is `+0.341` nats late versus `-0.035` early and `-0.045` mid, while mean window weight-change is nearly identical for middle and late (`0.00180` vs `0.00179`). Thus the late causal effect is not explained by a systematically larger late MLP weight delta in the dense-family pool.

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

[Figure S21: Cross-model δ-cosine heatmaps. Full layer × generation-step heatmaps for all six families (PT and IT side by side), showing the distribution of MLP opposition across the full forward pass.](../results/exp09_cross_model_observational_replication/plots/L1_heatmaps_6x2.png)

[Figure S23: Feature importance analysis. Per-feature contribution to late post-training computation at layers 20–33, showing the distribution of importance across transcoder features.](../results/exp03_corrective_stage_characterization/plots/plot_e3_11_feature_importance.png)

[Figure S24: Feature population dynamics. Gini coefficient and N50 distributions for IT vs PT at late layers, quantifying the broadening of the active feature repertoire.](../results/exp03_corrective_stage_characterization/plots/plot_feature_populations.png)

[Figure S25: Tuned-lens validation. KL(layer ℓ ‖ final) for all six models (PT variant). Red = tuned logit lens, blue = raw logit lens. The tuned lens substantially reduces KL at intermediate layers for five models (about 50–74% improvement at 60% depth for Llama, Qwen, Mistral, DeepSeek, and OLMo). Gemma improves only modestly at comparable depth (about 3%), indicating weaker probe quality for its hybrid local/global attention architecture rather than total probe failure. We therefore report both tuned and raw results throughout, and interpret Gemma's tuned-lens thresholded metrics with extra caution.](../results/exp09_cross_model_observational_replication/plots/tuned_lens_validation_kl_to_final.png)

**Table S25a: raw-vs-tuned sensitivity for the convergence-gap discovery metric.** Values are mean IT-minus-PT `KL(layer || own final)` differences from the existing cross-family discovery summaries. The raw lens does not remove the headline effect: the six-family final-half convergence gap is larger under raw lens (`0.729`) than tuned lens (`0.398`), and Gemma's own raw final-half gap (`1.008`) is larger than its tuned value (`0.351`). Excluding Gemma leaves the late-half raw gap positive (`0.712`), so the cross-family late stabilization result is not driven by Gemma's weak tuned-lens probe. This table is a sensitivity check for the discovery readout, not a raw-only rerun of every matched-prefix KL intervention; the matched-prefix JS replay and first-divergence token projections are the non-tuned companion evidence for those later claims.

| Scope | Lens | Early third | Middle third | Late third | Final-half CG | Full-stack mean |
|---|---:|---:|---:|---:|---:|---:|
| Six families | Tuned | `0.617` | `0.558` | `0.303` | `0.398` | `0.492` |
| Six families | Raw | `0.330` | `0.598` | `0.638` | `0.729` | `0.526` |
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

[Figure S41: Matched-prefix MLP graft trajectories across the five dense families plus a separate DeepSeek-V2-Lite MoE case. Pipeline C (IT) generates freely; PT control A' and grafted B are then forced to follow the same continuation. Solid lines show the raw-prompt branch and dashed lines the chat-template branch. The graft consistently reduces cross-KL to the IT teacher while reproducing the negative δ-cosine signature only partially in the dense-model pool.](../results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_400rand_v11_teacherforced/overview_trajectories.png)

[Figure S43: Free-running A/B/C output evaluation overview across all four judged metrics. A = PT raw, B = PT + late IT MLP graft under the same raw prompt, C = full IT model with its native chat template. B moves consistently toward C on benign false-refusal reduction, more selectively on assistant register, and only weakly on broad structure and harmful-prompt refusal.](../results/exp12_free_running_abc_graft/plots/exp12_eval_v1_20260413_v3/exp12_scores_overview.png)

[Figure S44: Improvement relative to the PT baseline in the free-running A/B/C evaluation. Red = B − A, green = C − A. Positive bars indicate improvement for G1, G2, and S1; for S2, positive bars indicate a reduction in false refusal. The graft consistently captures part of the A→C gap, but remains well short of the full IT endpoint on most metrics.](../results/exp12_free_running_abc_graft/plots/exp12_eval_v1_20260413_v3/exp12_delta_vs_a.png)

[Figure S45: Cross-family descriptive token-type analysis of the matched-prefix late stage. Left: displaced vs supported token classes under `A' -> B_late`. Center: teacher-token rank gain by collapsed token type. Right: token-type rank gain under early, middle, and late graft windows on the subset with recoverable raw depth traces. The late stage broadly supports the eventual teacher token and suppresses `FUNCTION/OTHER` raw-continuation-style alternatives, with a secondary formatting/discourse component.](../results/exp13_late_stage_token_support_analysis/exp13A_lite_20260415_live/exp13a_lite_paper_main.png)

[Figure S46: Descriptive token-support appendix view. Per-model panels, candidate entry/exit distributions, and mind-change summaries for the matched-prefix token-type analysis.](../results/exp13_late_stage_token_support_analysis/exp13A_lite_20260415_live/exp13a_lite_appendix.png)

[Figure S47: Symmetric matched-prefix graft/swap summary. Left: PT-side late-region KL deltas for early, middle, and late IT-MLP grafts relative to `A'`. Center: IT-side late-region KL deltas for early, middle, and late PT-MLP swaps relative to `C`. Right: dense-family predictive correlations for output-relevant late-stage summaries (`support_teacher`, `anti_top1`, `anti_kl_final`) and `δ`-cosine.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png)

[Figure S48: Symmetric graft/swap appendix view. Per-model bidirectional window-effect panels and late-stage mechanism summaries for the matched-prefix graft/swap analysis.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_appendix.png)

[Figure S48b: Matched random-control specificity check. Actual IT graft deltas are compared with matched random residual-projection controls on final-20% KL-to-own-final. The dense-family late true effect is large while the matched random late effect is near zero, ruling out a generic same-window perturbation account for the main late KL result.](../results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/exp19B_final20_kl_true_vs_random.png)

[Figure S49: Assistant-facing bucket deltas in the free-running LLM-judge behavioral follow-up. Dense-family pooled `G2` deltas by prompt bucket for PT-side grafts and IT-side swaps, highlighting that the largest late judge-rated degradation on the IT side concentrates on conversational and register-sensitive prompts.](../results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/exp15_paper_it_targeting.png)

[Figure S50: Matched-prefix native same-layer JS divergence under identical teacher tokens. Per-model `JS(A', C)` curves and dense-family pooled summaries show that broad PT↔IT output divergence is already present through much of the stack and amplifies late even when teacher histories are frozen.](../results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/exp16_js_appendix_models.png)

[Figure S51: Matched-prefix JS control view. PT-side and IT-side target-gap closure bars and host-local perturbation controls under matched prefix show that direct same-layer gap closure is more mid-to-late distributed than purely late, motivating the paper's “broad circuit, late delayed-stabilization window” synthesis rather than a strictly late-confined story.](../results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/exp16_js_appendix_controls.png)

[Figure S52: Reverse teacher-stream JS check. Replaying the same matched-prefix native-JS analysis with PT-generated continuations as teacher tokens again shows a broad dense-family PT↔IT same-layer JS gap. Late amplification is teacher-stream-dependent under token-step weighting, while prompt-mean aggregation still rises late; this supports the broad identical-history divergence claim while arguing against a strict teacher-stream-invariant late-confined interpretation. The Llama reverse replay excludes 11 empty PT-teacher continuations.](../results/exp16_matched_prefix_js_gap/exp16_js_reverse_pt_teacher_20260422_165259/plots/exp16_teacher_direction_comparison.png)

[Figure S53: Token identity at the first PT/IT divergent token. Under the raw-shared prompt, middle swaps transfer opposite-model token identity more than late swaps, while native prompting shows the deployment-format counterpart.](../results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/exp20_token_identity_dense5_ci.png)

[Figure S54: Mid-vs-late IT-token margin effects. Late windows dominate IT-vs-PT token-margin changes, especially under the native IT chat template.](../results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/exp20_mid_late_margin_dense5_ci.png)

[Figure S55: Raw-shared per-model token-transfer heatmap. Across dense families, middle swaps generally transfer token identity more than late swaps, with DeepSeek reported separately as the MoE case in all-model outputs.](../results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/exp20_raw_shared_model_transfer_heatmap.png)

[Figure S56: Native PT/IT final-token margin flow. Dense-5 pooled mid and late window margin deltas for the finally emitted token, plus dense-family mid-selected/late-helped rates and per-model IT `FORMAT` late-minus-mid margin. Late IT windows produce the strongest margin gains, especially for `FORMAT` and `CONTENT`, while PT is flatter.](figures/exp18_pure_flow_overview.png)

[Figure S57: Matched-prefix identity/readout chronology. Left: teacher-token rank gain by disjoint window under identical histories. Middle: strict rate at which a token is first selected in the middle window and then further helped late. Right: continuity view of `A' -> B_window` top-1 displacement. Format-like tokens become teacher-rank-positive only late, while content shows larger middle-window gains.](figures/exp18_handoff_summary.png)

## Appendix B: Evaluation Methodology

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

For the completed human audit, the primary confirmatory quantities are pairwise resolved win rates and bootstrap confidence intervals for the same four primary contrasts used in Figure 5. The rater packets randomize A/B order, expose no model or condition labels, and include `TIE`/`BOTH_BAD` to avoid forcing noisy preferences. The human audit was analyzed after the automated judge setup, endpoint definitions, and primary contrasts were fixed, so it functions as an external validation check rather than as a source of endpoint selection. We report same-sample LLM rates beside the human rates to show direction agreement directly.

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

| Paper | PT↔post-trained descendants? | Cross-family paired checkpoints? | Identical-history internal comparison? | Symmetric depth-localized intervention? | Natural-decoding consequence test? |
|---|---|---|---|---|---|
| Lad et al. (2024) | No | No | No | No | No |
| Joshi et al. (2025) | No | No | No | No | No |
| Wu et al. (2024) | Yes | No | No | No | No |
| Du et al. (2025) | Yes | Yes | No | No | No |
| Li et al. (2025) | No | No | No | No | No |
| Chaudhury (2025) | Yes | No | No | No | No |
| Prakash et al. (2024) | Yes | No | No | Partial | No |
| Arditi et al. (2024) | No | No | No | No | Yes |
| Lu et al. (2026) | No | No | No | No | Yes |
| Stolfo et al. (2024) | Partial | Partial | No | No | Yes |
| Turner et al. (2023); Zou et al. (2023); Panickssery et al. (2024) | No | No | No | No | Yes |
| Hewitt et al. (2024) | No | No | No | No | No |
| Rocchetti & Ferrara (2026) | No | No | No | No | No |
| Ours | Yes | Yes | Yes | Yes | Yes |

Lad et al. (2024) and Joshi et al. (2025) are the closest phenomenon-level comparisons. Lad gives a generic vocabulary for mid-to-late inference stages: prediction ensembling followed by residual sharpening. Joshi studies late confidence calibration after decision certainty has emerged. We do not treat late sharpening or late calibration as new phenomena. The contribution is the PT↔post-training identification package around how instruction-tuned candidates become final next-token predictions: a convergence-to-final target, endpoint-free same-history companion readout, symmetric depth-localized intervention, and first-divergent-token identity/readout split.

Prakash et al. (2024) is the closest methodological precedent in this broader set: CMAP patches activations across related base/fine-tuned models to reveal improved mechanisms. Our use is different in scope and target: we apply a symmetric equal-width early/mid/late MLP graft/swap design across model families, and the localized target is not entity tracking but PT↔post-training convergence-to-final dynamics. Following the caution urged by activation-patching work such as Heimersheim and Nanda (2024), we describe these interventions as causal leverage on measured readouts, not as complete mechanism recovery. Arditi et al. (2024) and Lu et al. (2026) show that post-training leaves low-dimensional behavioral control signals, which is compatible with the broader mid-to-late identity/readout picture but does not identify the same package of confounds: history, endpoint, bidirectional coarse interventions, and identity/readout.

Stolfo et al. (2024), Turner et al. (2023), Zou et al. (2023), and Panickssery et al. (2024) are the main reason we do not frame the contribution as finding an instruction-following activation vector. That claim is not new. Those papers show that activation-space directions can control behavior, including instruction-following constraints and high-level traits. The paper-level novelty here is that a prediction-dynamics target is localized with paired PT↔IT graft/swap interventions under identical histories and first-divergence counterfactuals, rather than by extracting a behavioral vector.

The appendix table makes the novelty claim deliberately concrete. Prior rows contain individual ingredients: PT↔post-training comparison, behavioral localization, cross-model substitution, or natural-decoding intervention. The paper's row is distinct because those ingredients are combined into an identification package around convergence-to-final dynamics under PT↔post-training model diffing, interpreted through a directly tested candidate-identity versus margin/readout split. Hewitt et al. (2024) and Rocchetti and Ferrara (2026) are useful conceptual boundary papers rather than direct method-level foils. They ask what instruction following is, or whether it reflects a universal mechanism at all. Our paper asks a narrower and more operational question: what paired PT↔post-training forward-pass signature appears across families, how middle and late layers divide candidate selection from final-token reconciliation, and whether that same intervention family matters under natural decoding.

## Appendix F: Scope Boundaries and Counter-Interpretations

**Free-running histories.** The six-family native-decoding curves establish the convergence gap as a model-in-use signature. They are not used as the sole causal evidence because PT and IT continuations can diverge. The matched-prefix native-JS replay and MLP graft/swap experiments are the controlled-history evidence.

**Endpoint metric and probe dependence.** `KL(layer || own final)` is intentionally a stabilization-to-own-decision metric, not a direct cross-model distance. It asks how far a layer is from the same model's eventual next-token distribution. Endpoint relativity is part of the metric; endpoint dependence is tested by an endpoint-free same-history readout. Matched-prefix native JS is therefore a companion control, not an interchangeable version of the KL trajectory. The remaining concern is estimator dependence: tuned-lens quality can change absolute KL magnitudes. We therefore do not treat the tuned-lens discovery figure as standalone identification. Raw-logit-lens variants and native same-layer JS under identical histories are the required companion readouts, and they preserve the same qualitative PT↔IT separation.

**Gemma tuned-lens quality.** Gemma is the family where the tuned lens improves least over the raw lens at intermediate depth, and it is also the family with the largest late `δ`-cosine shift. We flag this coincidence rather than using Gemma tuned-lens metrics as a standalone pillar. The relevant separation is that `δ`-cosine is lens-free, while the convergence-gap discovery metric is lens-dependent. For the lens-dependent convergence-gap discovery metric, the raw lens strengthens rather than erases Gemma's final-half gap (`1.008` raw versus `0.351` tuned), and the dense-family raw final-half gap remains positive even with Gemma removed (`0.712`; Table S25a).

**Weight-difference magnitude.** A possible confound is that late grafts matter most simply because post-training changed late MLP parameters most. Figure S18 already makes this unlikely outside Gemma: Llama, Qwen, Mistral, OLMo, and DeepSeek do not show a uniformly late-concentrated MLP weight-change profile. Table S18a makes the cross-reference explicit by dividing the matched-prefix final-20% KL effect by a same-window mean MLP weight-change proxy. Late remains the largest normalized effect in every family, and in the dense-family pool the middle and late weight-change proxies are nearly equal while only late has a large positive effect. Table S18b adds the complementary raw-effect and leave-one-family-out view: all dense families have positive raw late effects, and removing Mistral increases rather than decreases the pooled mean. Mistral's large normalized ratio is therefore a denominator artifact of its unusually small PT→IT MLP RMS change, not evidence that Mistral dominates the causal result. The matched random-control analysis adds a residual-projection control and shows that the dense-family late KL effect is specific to the learned IT substitution rather than any matched same-window perturbation (`+0.327` true versus `+0.003` random), but it still does not convert the result into a per-unit-weight efficiency claim. We therefore treat weight-difference magnitude as a family-specific diagnostic, not the paper-level explanation for late causal leverage. The table is still a descriptive normalization, not a claim that we have estimated a true per-parameter causal efficiency.

**Metric surface area.** We report several convergence summaries because the phenomenon is trajectory-shaped, but the main claims are intentionally pinned to a small ladder: native KL-to-own-final discovery, endpoint-free same-history JS, symmetric final-20% graft/swap effects, first-divergent-token identity/margin, and judged behavior. Per-family or auxiliary-metric deviations are treated as diagnostics unless they agree with that ladder.

**Prompt formatting.** The primary discovery condition uses each model in its native regime: IT models receive native chat templates and PT models receive raw prompts. This is the relevant model-in-use comparison. Raw-vs-templated matched-prefix controls support that the delayed-stabilization signal is not only a prompt-format artifact. In the first-divergence token tests, raw-shared controls preserve the qualitative mid-identity/late-margin split, while native chat templates amplify the deployed IT margin. We therefore read templates as part of the native IT operating distribution, not as the sole source of the effect.

**Teacher direction.** The primary matched-prefix replay uses IT-generated teacher continuations because the paper studies how PT differs from its instruction-following descendant. A reverse PT-teacher replay again shows a broad dense-family same-layer JS gap. Late amplification is teacher-stream-dependent under token-step weighting, so we use the reverse run as a robustness check for broad same-history divergence rather than as a separate late-localization claim.

**Middle versus late.** Direct same-layer target-gap closure and behavioral token identity are more mid-to-late distributed than purely late. Native final-token flow sharpens this rather than erasing it: the pure-flow analysis shows larger late IT margin gains for format and content, while the matched-prefix chronology shows that content-like candidate promotion can already be strong in middle windows. First-divergence token-choice tests sharpen the boundary further: middle swaps transfer first-divergent token identity more than late swaps, while late layers more strongly affect IT-vs-PT margin/readout. MLP write-out tests add direct evidence: pure IT late MLPs provide much larger support for the IT divergent token than early or middle MLPs. That does not contradict the main result because delayed stabilization, token identity, and token write-out are different targets. The directly tested causal claim is late MLP leverage on delayed stabilization and native IT-token readout; the broader middle-to-late identity/readout split is supported by token-flow evidence but remains an interpretation at the complete named-circuit level.

**Relation to mid-layer alignment localization.** Li et al. (2025) and Chaudhury (2025) are not treated as foils to be overturned. They study safety/preference behavior directly and find middle-layer localization under their readouts. Our own first-divergence identity tests agree with that direction: middle swaps transfer PT-vs-IT token identity more than late swaps. The new late claim is about a different target: delayed stabilization and MLP write-out into the final next-token margin. A stricter head-to-head would rerun Chaudhury-style HHH preference patching and Li-style safety-layer tests on the same model pairs and compare those layers directly to the convergence-gap layers. We have not run that replication, so the paper claims compatibility and a tested identity/readout split, not that prior mid-layer alignment results are wrong.

**Behavioral asymmetry.** The current LLM-judge behavioral bridge is clearest on IT-side degradation under late PT swaps. Late IT grafting into PT is measurable but smaller than the full PT-to-IT behavioral gap. This is expected if late MLPs are a privileged final-token reconciliation window embedded in a broader post-training circuit, not a complete assistantness module. The completed blind human audit agrees with the pooled direction of the four primary pairwise contrasts, but high unresolved rates and heterogeneous κ mean it should be treated as confirmatory evidence for the behavioral bridge, not as part of the core internal localization claim.

**Identity/readout status.** We now include direct identity/readout analyses. The first-divergence token-choice test is the cleanest token-identity test because it conditions on the first PT/IT divergent token and evaluates grafted pipelines at the shared prefix. The MLP write-out test is the cleanest readout test because it stores finite-difference token projections for the same first-divergence setup. Native final-token flow remains useful for chronology, but the older matched-prefix traces store ranks/top-k sets rather than per-layer logits. A complete named-circuit chronology still requires matched-prefix reruns with per-layer logit capture and feature-level mediation tests.

**Mid+late interaction status.** The current evidence no longer leaves the basic `PT + IT mid+late` branch missing. In the first-divergence/write-out tests, adding IT late layers to a PT host is weak by itself, and adding IT late on top of IT middle gives only a small additional fixed-prefix MLP margin gain (`+0.0119` logits; 95% CI `[+0.007, +0.016]`). This weak PT-host result does not refute late readout; it sharpens it. Late IT MLPs are strongest when the upstream state is already IT-like, as shown by the larger IT-host late-swap loss and by the source decomposition where upstream context exceeds the late-weight main effect.

**Geometry and mediation.** Negative late `δ`-cosine is not new by itself and is present in pretrained models. The geometric claim is the additional IT-minus-PT late shift. The MLP decomposition further shows that the residual-opposing component itself does not directly increase IT-vs-PT margin in pure IT late layers (`-0.0046` logits; 95% CI `[-0.009, -0.001]`), while the full late MLP update strongly does (`+0.768` logits; 95% CI `[+0.729, +0.805]`). We therefore treat negative residual opposition as a geometric marker of late reconciliation, not as the causal write-in vector.

## Appendix G: Reproducibility and Artifact Map

**Public release.** The project repository is `https://github.com/yifan1207/structral-semantic-features`. It includes the current manuscript draft, all source experiment packages under `src/poc/`, shared cross-model infrastructure under `src/poc/cross_model/`, grouped script entrypoints under `scripts/`, and committed paper-facing artifacts under `results/`. The canonical prompt and evaluation datasets are committed under `data/`, including `eval_dataset_v2.jsonl`, `eval_dataset_v2_holdout_0600_1199.jsonl`, `exp3_dataset.jsonl`, `exp6_dataset.jsonl`, and `gold_standard_v1.csv`.

**Environment and reruns.** The Python environment is specified by `pyproject.toml` and `uv.lock`. GPU-heavy reruns require local access to the listed Hugging Face checkpoints and sufficient memory for the 4B-8B model pairs. Standard reruns use `uv run python -m ...` for source packages and `bash scripts/run/...` for orchestration scripts; worker outputs are merged with `scripts/merge/merge_steering_workers.py` where applicable. Unless a script states otherwise, reported generation traces use deterministic greedy decoding.

**Committed versus regenerated artifacts.** The repository commits the paper-facing summaries and plots used for manuscript numbers: JSON/CSV/MD summaries, bootstrap confidence intervals, human-evaluation summaries, and final PNG figures. It does not commit large regenerated intermediates such as raw activation arrays (`*.npy`, `*.npz`), model/probe tensors (`*.pt`, `*.safetensors`), tuned-lens checkpoints, or multi-gigabyte raw per-token trace JSONL/GZ files. These are size exclusions, not confidentiality exclusions; the corresponding collection scripts, analysis scripts, and prompt manifests are public.

**Main claim-to-artifact map.**

| Claim or analysis | Code entrypoints | Committed artifacts |
|---|---|---|
| Six-family convergence gap and commitment delay | `src/poc/exp09_cross_model_observational_replication/`; shared adapters in `src/poc/cross_model/` | `results/exp09_cross_model_observational_replication/data/`, `results/exp09_cross_model_observational_replication/plots/` |
| Gemma steering and Tier-0 validation controls | `src/poc/exp06_corrective_direction_steering/`; `src/poc/exp07_methodology_validation_tier0/`; `scripts/run/run_exp7_0*.sh`; `scripts/plot/plot_validation_tier0.py` | `results/exp06_corrective_direction_steering/plots/`, `results/exp07_methodology_validation_tier0/data/`, `results/exp07_methodology_validation_tier0/plots/` |
| Matched-prefix graft depth ablation | `src/poc/exp11_matched_prefix_mlp_graft/` | `results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/` and selected summaries under `results/exp11_matched_prefix_mlp_graft/data/` |
| Symmetric graft/swap and late random-control specificity | `src/poc/exp14_symmetric_matched_prefix_causality/`; `src/poc/exp19_late_mlp_specificity_controls/` | `results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/`, `results/exp19_late_mlp_specificity_controls/` |
| Behavioral bridge and human audit | `src/poc/exp15_symmetric_behavioral_causality/`; `scripts/eval/llm_judge.py`; human-audit materials under `paper_draft/human_eval_survey/` | `results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/`, `results/exp15_symmetric_behavioral_causality/human_eval/` |
| Endpoint-free matched-prefix JS replay | `src/poc/exp16_matched_prefix_js_gap/`; `scripts/analysis/analyze_exp16.py`; `scripts/plot/plot_exp16.py` | `results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/`, `results/exp16_matched_prefix_js_gap/exp16_js_reverse_pt_teacher_20260422_165259/` |
| Native final-token flow and matched-prefix chronology | `src/poc/exp18_midlate_token_handoff/`; `scripts/run/run_exp18_yanda_full.sh` | `results/exp18_midlate_token_handoff/full_runpod_20260423_095122/`, `results/exp18_midlate_token_handoff/matched_prefix_latest/` |
| First-divergence token identity and margin/readout | `src/poc/exp20_divergence_token_counterfactual/`; Exp20 analysis scripts under `scripts/analysis/` | `results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final/deep_dive/`, `results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early/validation_analysis/` |
| MLP write-out, context-gating decomposition, and synthesis figure | `src/poc/exp21_productive_opposition/`; `scripts/analysis/analyze_exp21_productive_opposition.py`; `scripts/analysis/build_exp20_exp21_handoff_synthesis.py` | `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/`, `results/paper_synthesis/` |
