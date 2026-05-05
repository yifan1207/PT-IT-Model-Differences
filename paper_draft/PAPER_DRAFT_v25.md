# First-Divergence Model Diffing Reveals Upstream-Dependent Late-Layer Effects in Base-to-Instruct Models

**Anonymous authors** | NeurIPS 2026 Submission

---

## Abstract

Late-layer patching is often read as localizing what a post-trained model changed. But a late stack can look causal simply because it is paired with the upstream residual state it was trained to read. We test this portability directly in paired base/instruct models. At the first generated token where the base and instruct checkpoints disagree, we cross the upstream state and late stack from each model, then measure the instruct-token over base-token logit margin. Across five dense families on matched holdout support, the instruct late stack is only partly portable: `31.1%` of its matched-context effect transfers to base upstream state, while the matched instruct-upstream effect is `3.2x` larger, giving a `+1.68` logit interaction. Controls rule out broken hybrids, arbitrary token selection, and pre-late commitment; same-base recipe and released-stage checks show the pattern is not generic fine-tune compatibility. Constrained continuation shows the split persists beyond the selected token, and terminal crosscoders connect it to sparse MLP features that partially mediate the interaction. The result is a methodological warning and a model-diffing tool: late patches should not be interpreted as portable late-only mechanisms unless portability across upstream states is tested.

---

## 1. Introduction

Late-layer patching can answer where a base-to-instruct difference is expressed, but not whether that late computation ports across upstream states. When an instruction-following checkpoint first chooses a different next token from its pretrained base, where in the forward pass does that difference become a logit? Paired base/instruct checkpoints make this question unusually clean: architecture and tokenizer are shared, but the released descendants differ after instruction tuning, preference optimization, reinforcement-style training, or a mixture of post-training stages. The object of study is therefore a **paired-checkpoint model diff**.

The tempting answer is "late layers." Late transformer computation is close to the unembedding, and prior work already makes late-stage refinement plausible: feed-forward layers promote vocabulary-space concepts (Geva et al., 2022b), late layers sharpen or calibrate predictions (Lad et al., 2025; Joshi et al., 2025), and instruction-tuned models show layer-structured task information (Zhao, Ziser, and Cohen, 2024). But ordinary late-layer patching leaves a crucial ambiguity. A late stack can look causal partly because it is paired with the upstream residual state it was trained to read, not because its effect would transfer unchanged as a late-only update.

We test portability directly with **first-divergence model diffing**. For each prompt, we find the earliest generated position where PT and IT prefer different next tokens under the same generated history. Let those tokens be `t_PT` and `t_IT`. At that prefix, we cross upstream residual state (`U_PT` or `U_IT`) with downstream late stack (`L_PT` or `L_IT`) and measure `logit(t_IT) - logit(t_PT)`. The key estimand is a difference-in-differences: how much larger is the IT-minus-PT late-stack replacement effect when the upstream state is IT-shaped rather than PT-shaped?

The answer is a partially portable but mostly upstream-conditioned late readout. Across the five-family core set, the IT late stack has a positive effect even from PT upstream, but the portable share is centered at `31.1%` family-balanced (`29.2%` median, `19.5-44.3%` range). The matched-context effect is `3.2x` larger, producing a `+1.68` logit interaction. The effect remains positive at generated positions `>=3`, under a common-PT readout, on a 32B family, under label-swap nulls, and under controls for pre-late token commitment, hybrid-state mismatch, and selected-token support.

This matters for model diffing practice. A late patch should not be interpreted as a fully portable late-layer mechanism unless the late effect is tested under both native and foreign upstream states.

The paper has three contributions.

1. **A paired-checkpoint first-divergence factorial.** We introduce a local counterfactual estimand at the exact token where PT and IT first disagree. It decomposes the IT-minus-PT late-stack replacement effect into a portable component that transfers to PT upstream and an upstream-conditioned component that appears under IT-shaped upstream state.
2. **A validation and specificity ladder for the estimand.** We test the practical artifact explanations: broken hybrids, arbitrary selected-token support, pre-late commitment, readout choice, label orientation, position support, family heterogeneity, generic fine-tune compatibility, and whether released stage lineages show a coherent profile. The main text gives the decisive checks; the appendices give provenance.
3. **A sparse terminal-feature and structured-state bridge.** Terminal crosscoders connect the window-level result to sparse MLP features that partially mediate the readout interaction, matter more under IT-shaped upstream state, partially rescue the weak hybrid, and respond to preterminal computation patches. Depth anatomy supports the candidate-to-margin handoff, and a separate boundary-state rescue shows that a train-fit medium-rank descendant-minus-base residual shift recovers much of the missing terminal margin. This is a partial bridge, not full circuit reconstruction.

Scope is intentionally local. We use **IT** as shorthand for instruction-following post-trained descendants, but the recipes are heterogeneous; the empirical claim is about released dense base/instruct checkpoint contrasts, not one training algorithm. Causal language is readout-scoped: replacing an upstream state, late stack, or MLP component changes a specified next-token readout in a constructed forward pass. We do not claim a universal instruction-following circuit or full circuit reconstruction.

---

## 2. Setup

### 2.1 Model Sets and Statistical Reporting

The main factorial uses five dense PT/IT pairs: Llama 3.1 8B, Qwen 3 4B, Mistral 7B, OLMo 2 7B, and Qwen2.5 32B. We call this the **Core-5 set**. Several support analyses were run before the 32B extension and use the four smaller families: Llama, Qwen 3, Mistral, and OLMo. We call this the **Core-small support set**.

Prompt-bootstrap intervals quantify conditional precision on the sampled prompts and released checkpoints. They are not estimates over all possible model families, post-training recipes, or prompt distributions. For headline family-generalization claims, we therefore report the Core-5 mean and family-level range or median where it helps interpretation.

**Table 1: Minimal reproducibility snapshot.** The table below gives the fixed ingredients needed to audit the main factorial. Revision prefixes are unique abbreviations of the pinned Hugging Face revisions; Appendix A lists the full 40-character pins.

| Family | Checkpoint pair (`PT -> IT`) | Prompt set | Prompts -> valid events | First-div. position `0 / >=3 / >=5` | Late stack |
|---|---|---|---:|---:|---|
| Llama 3.1 8B | `Llama-3.1-8B@d04e592 -> Instruct@0e9e39f` | holdout-600 | `600 -> 600` | `59.3% / 28.0% / 17.3%` | layers `19-31` |
| Qwen 3 4B | `Qwen3-4B-Base@906bfd4 -> Qwen3-4B@1cfa9a7` | holdout-600 | `600 -> 600` | `48.7% / 35.2% / 23.5%` | layers `22-35` |
| Mistral 7B | `Mistral-7B-v0.3@caa1feb -> Instruct@c170c70` | holdout-600 | `600 -> 597` | `30.8% / 31.3% / 20.1%` | layers `19-31` |
| OLMo 2 7B | `OLMo-2-1124-7B@7df9a82 -> Instruct@470b1fb` | holdout-600 | `600 -> 586` | `60.1% / 27.1% / 16.0%` | layers `19-31` |
| Qwen2.5 32B | `Qwen2.5-32B@1818d35 -> Instruct@5ede1c9` | holdout-600 | `600 -> 599` | `46.4% / 39.4% / 28.9%` | layers `38-63` |

All rows use raw-shared prompts, greedy top-1 first-divergence search with a real-token mask, and at most `128` generated tokens. Bootstrap unit is the prompt cluster within family. Appendix A gives prompt mixes and full revision pins; Appendix B gives the main effect audit and Appendix I gives artifact roots.

### 2.2 First-Divergence Factorial

For each prompt, generate with PT and IT until the first shared-history prefix where their top-1 next tokens differ. Let those tokens be `t_PT` and `t_IT`, and define

`Y(U,L) = logit(t_IT) - logit(t_PT)`.

Larger `Y` means the hybrid forward pass favors the IT divergent token. At a pre-specified late boundary, run the four cells below:

| Upstream state | PT late stack `L_PT` | IT late stack `L_IT` |
|---|---:|---:|
| PT upstream `U_PT` | `Y(U_PT,L_PT)` | `Y(U_PT,L_IT)` |
| IT upstream `U_IT` | `Y(U_IT,L_PT)` | `Y(U_IT,L_IT)` |

The primary estimand is:

`[Y(U_IT,L_IT) - Y(U_IT,L_PT)] - [Y(U_PT,L_IT) - Y(U_PT,L_PT)]`.

Equivalently: first measure the IT-minus-PT late-stack replacement effect under each upstream state; then compare those two effects. This subtracts the PT-late-stack baseline within each upstream row, so the result is not merely a native-stack-vs-foreign-stack comparison. Common-IT and common-PT readouts score all four cells with one fixed final norm, `lm_head`, and real-token mask. Unless stated otherwise, main factorial numbers use common-IT readout.

All raw-shared first-divergence and residual-state runs force both PT and IT branches to raw text and validate identical raw prompt token IDs before comparing residual states. Position 0 is therefore the first generated token after the full raw prompt, not a chat-template artifact.

![First-divergence schematic and token examples](../results/paper_synthesis/first_divergence_schematic_examples.png)

**Figure 1: First-divergence schematic and token examples.** Panel A shows the four hybrid passes used to estimate the upstream x late interaction. Panel B gives illustrative divergent-token pairs; all quantitative claims use the full support in Table 1.

---

## 3. Results

### 3.1 Main First-Divergence Factorial

Prior work leaves three hypotheses open. **H1: late-only transfer story** -- the IT-minus-PT late-stack effect should be about the same from PT and IT upstream (`~1x` amplification). **H2: no portable component** -- the IT late stack should do little from PT upstream. **H3: partially portable, upstream-conditioned readout** -- both effects should be positive, but the matched-context effect should be much larger. The factorial distinguishes these hypotheses directly.

At the first natural PT/IT disagreement, the Core-5 result supports H3. Replacing the PT late stack with the IT late stack shifts the IT-token margin by `+0.76` logits from PT upstream, but by `+2.44` logits from IT upstream. Thus the family-balanced portable share is `31.1%`, with family values from `19.5%` to `44.3%`; most of the matched-context effect is exposed only under IT-shaped upstream state. The `+1.68` logit gap means the matched-context effect is `3.2x` larger.

The sign pattern is consistent across families: every Core-5 interaction is positive, ranging from `+1.25` to `+2.53` logits. The amplification scale is the cleanest magnitude reference because it compares the same late-stack replacement under two upstream states; Appendix B reports native-shift shares and family-level ranges.

The interaction is also not confined to response openings: generated position `>=3` and `>=5` subsets remain positive, though smaller and thinner.

![Figure 2: Core-5 first-divergence interaction by family. The upstream x late interaction is positive in every core dense family, including the 32B scale check.](../results/paper_synthesis/exp23_core5/exp23_core5_interaction.png)

| Scope/readout | Late effect from PT upstream | Late effect from IT upstream | Portable share | Interaction | Amplification |
|---|---:|---:|---:|---:|---:|
| Core-5, common-IT | `+0.76` `[+0.68, +0.84]` | `+2.44` `[+2.34, +2.53]` | `31.1%` (`29.2%` median; `19.5-44.3%` range) | `+1.68` `[+1.60, +1.76]` | `3.2x` |
| Core-5, common-PT | `+0.78` `[+0.72, +0.85]` | `+2.49` `[+2.40, +2.58]` | `31.5%` | `+1.70` `[+1.63, +1.78]` | `3.2x` |
| Qwen2.5-32B only | `+1.04` `[+0.89, +1.17]` | `+2.34` `[+2.14, +2.55]` | `44.3%` | `+1.30` `[+1.14, +1.49]` | `2.3x` |

On the local token-margin scale, `+1.68` logits is substantial: it multiplies the odds of `t_IT` over `t_PT` by about `5.4x` within the constructed contrast, and it is `33.7%` of the native PT->IT diagonal margin shift on the same support. This is still a token-level mechanistic quantity, not a deployment-level behavior estimate.

This is the paper's central claim: at natural base/instruct disagreement points, the IT late stack is not a self-contained module whose full effect ports to any upstream state. It has a measurable portable component, but most of its matched-context effect is upstream-conditioned. On a factual/reasoning stress test, for example, the late-only effect from PT upstream is negative (`-1.18`), while the interaction remains positive (`+1.81`).

### 3.2 Validation Ladder

The first-divergence factorial intentionally selects a high-signal disagreement point and constructs hybrid forward passes. No patching experiment makes those hybrids literally natural trajectories. The validation question is narrower: do the main artifact explanations make the estimand uninformative? Three classes of checks answer no; Appendix C gives the full audit trail.

| Concern | Main evidence | Takeaway |
|---|---|---|
| The hybrid state is broken or generically off-manifold. | Native diagonals reconstruct exactly; PT->IT boundary interpolation is smooth and positive; signed permutations recover only a minority of the effect. | The effect behaves like a structured PT->IT direction, not a patching cliff or generic hybrid perturbation. |
| First divergence is an arbitrary selected token pair. | Random local disagreements from the same prompts retain only `56%` of the first-divergence interaction; pre-divergence prefixes scored on the future token pair are near zero (`3%`). | First divergence is a meaningful high-signal support, not just any PT/IT token contrast. |
| The upstream state has already decided before the late stack. | The interaction remains positive when the IT boundary readout does not yet favor `t_IT` and in the lowest boundary-margin tercile. | The late stack still contributes after controlling for pre-late commitment. |

The selected-support control is important enough to visualize directly. First divergence is high-signal, but it is not a one-off cherry-picked token pair: random later PT/IT disagreements from the same rollouts show the same positive sign at reduced magnitude (`56%` in the source-balanced comparison), while scoring the future divergent token pair before the models actually diverge is near zero (`3%`). This is the pattern we would expect if first divergence is a principled support-selection device: it concentrates the interaction, but related local disagreements retain a weaker version of it.

![Figure 3: Selection baselines for first divergence. Random local disagreements from the same rollouts preserve the sign of the upstream x late interaction at reduced magnitude, while pre-divergence future-token scoring is near zero. Values are Core-small family-balanced means.](../results/paper_synthesis/exp37_core_small_selection_baseline/selection_baselines_core_small.png)

Secondary checks agree: common-PT readout gives the same answer, later-position subsets remain positive, the label-swap null passes, and every core family has a positive interaction. Together, these controls make the practical artifact explanations unlikely. The exact logit magnitude remains intervention-scoped, but the upstream-conditioned late-readout pattern is robust under the tests that would otherwise explain it away.

### 3.3 Recipe and Stage Specificity

The Core-5 result compares released base checkpoints to released instruction-following descendants. Two extra checks narrow the interpretation. First, a same-base recipe comparison asks whether upstream-conditioned late readout is merely generic fine-tune compatibility. On instruction/format supports from Llama-3.1-8B descendants, instruction-like checkpoints are positive while an OpenMath2 domain-specialized checkpoint is not; after matching on prompt category, generated-position bin, and token category, the instruction-like-minus-OpenMath contrast is `+1.335` logits. OpenMath also does not show a positive interaction on math-domain support (`-0.154` `[-0.450, +0.128]`), despite large late effects, while the instruction-like mean remains positive (`+1.670`). The instruction-like orientation passes a sign-flip null (`p=5e-5`).

| Same Llama-3.1-8B base descendant, instruction/format support | Interaction | Read |
|---|---:|---|
| Meta Instruct | `+1.053` `[+0.920, +1.201]` | instruction-like, positive |
| Tulu SFT | `+0.287` `[+0.239, +0.337]` | smaller but positive |
| Tulu DPO | `+1.131` `[+1.028, +1.242]` | larger |
| Tulu Final/RLVR | `+1.365` `[+1.236, +1.499]` | largest instruction-like checkpoint |
| OpenMath2 | `-0.358` `[-0.534, -0.181]` | domain-specialized control does not reproduce it |

A constrained-continuation bridge shows that the recipe-specific split is not confined to the first token. After forcing the descendant-preferred divergent token, we teacher-force short native descendant/base continuations through the same four cells. For instruction-like descendants, the common-IT interaction grows from `+1.50` at `N=0` to `+2.71` over `N=8`, with `+0.193` logits/token tail-only persistence. A same-forced-descendant-tail control remains positive (`+2.46` at `N=8`), while shuffled descendant tails are much smaller (`+0.87`). OpenMath again behaves differently and is readout-sensitive at longer horizons, so we use this bridge for the instruction-like persistence claim rather than as a universal sequence-level recipe taxonomy.

Second, we ask whether the interaction appears all at once at the final checkpoint or is already visible along released post-training paths. We apply the same fixed-support factorial to two multi-stage lineages: Tulu-3 on Llama-3.1-8B and OLMo-2. In each case, the support is the Base->Final first-divergence set, and intermediate checkpoints are scored on the same `t_Base`/`t_Final` token contrast. This is a stage diagnostic, not causal attribution to a training algorithm.

| Lineage, fixed Base->Final support | SFT interaction | DPO interaction | Final interaction | Read |
|---|---:|---:|---:|---|
| Tulu-3 / Llama-3.1-8B | `+0.419` `[+0.349, +0.491]` (`29%`) | `+1.216` `[+1.090, +1.341]` (`84%`) | `+1.455` `[+1.316, +1.606]` | partly present at SFT; largely present by DPO |
| OLMo-2 | `+0.773` `[+0.674, +0.873]` (`40%`) | `+1.629` `[+1.473, +1.793]` (`85%`) | `+1.924` `[+1.747, +2.104]` | same qualitative profile |

Tulu is the cleanest second-lineage check because it uses the same Llama-3.1-8B base architecture as one Core-5 family. Common and native readouts agree, and the final Tulu interaction passes the same label-swap orientation test as the main factorial (`+1.455` observed vs `+0.296` null 99.9th percentile). Base->SFT and Base->DPO support reruns give the same qualitative profile, with final interactions `+1.322` and `+1.436`. Together with OLMo, this suggests the upstream-conditioned readout is not a one-off final-checkpoint accident: it is already measurable after SFT and mostly present by the preference-trained checkpoints in two released lineages.

We phrase this carefully. The recipe control is a token-factorial specificity check, not a behavior claim. The stage percentages are fixed-support checkpoint scores: they do not imply that DPO causally contributes exactly `84-85%` of the mechanism, nor that every post-training recipe has this profile.

### 3.4 From Depth Anatomy to Sparse Terminal Features

**Window anatomy.** The interaction has a consistent depth anatomy, which motivates the terminal feature tests. Middle-positioned MLP substitutions transfer divergent-token identity more often than late substitutions, while late windows dominate margin/readout. Terminal-depth audits sharpen the same story: the last few blocks preserve a large readout subcomponent but transfer token identity poorly. The handoff is therefore operational rather than a complete circuit: middle windows are relatively more candidate/identity-selective, while late and terminal windows are more margin/readout-sensitive.

| Evidence | Key result | Interpretation |
|---|---:|---|
| Middle vs late identity transfer | `25.6%` mid vs `18.8%` late in PT host; `28.2%` mid vs `21.5%` late in IT host | Middle windows are relatively more candidate/identity-selective. |
| Native IT MLP margin support | late `+0.986`; middle `+0.136`; early `-0.085` | Native IT-token margin support is late-concentrated. |
| PT-host late insertion | `+0.004` | Late MLP updates alone are near zero from PT upstream state. |
| Terminal-depth audit | final-three blocks retain `52%`; final block retains `23%` | Terminal layers carry a substantial readout subcomponent. |

![Figure 4: Window-level identity/margin handoff. Middle substitutions transfer divergent-token identity more often, while late MLPs dominate native IT-token support.](../results/paper_synthesis/exp20_exp21_handoff_core_small.png)

**Sparse terminal mediation.** Terminal crosscoders make this handoff visible at feature level. We train paired PT/IT BatchTopK crosscoders with a shared latent dictionary and separate PT/IT decoder branches on terminal MLP outputs, rank features by held-out causal effect, and ablate their IT-branch decoder contribution inside the terminal IT stack. The fixed top-200 causal subset accounts for `26-48%` of the terminal readout interaction, while matched-random sets have the wrong sign or near-zero effect. Appendix E gives reconstruction, sparsity, and training details.

![Figure 5: Terminal crosscoder mediation. Ablating the top causally ranked terminal features reduces the upstream x late interaction in each clean terminal-crosscoder family, while matched random features do not reproduce the effect. Percent labels show the top-200 share of the family interaction.](../results/paper_synthesis/exp34_core_feature_mediation/terminal_crosscoder_core3_mediation.png)

| Terminal feature bridge test | Result | Meaning |
|---|---:|---|
| Feature mediation | `26-48%` | Sparse terminal features carry a concentrated part. |
| Upstream-conditioned ablation | causal gate beats matched random | Same features matter more under IT upstream. |
| Feature rescue | `+0.49` | Selected features partially restore the weak hybrid. |
| Upstream patch | `+1.71`, `+0.13` mediated | Preterminal computation drives these features. |
| Reverse patch | `+3.57`, `+0.53` mediated | Removing IT preterminal state suppresses them. |

**Upstream drives terminal features.** The same features inherit the upstream conditioning seen at window level. Ablating the top-200 causal terminal features hurts the `U_IT,L_IT` readout much more than the `U_PT,L_IT` readout, and patching their native `U_IT,L_IT` activations into the weak `U_PT,L_IT` hybrid rescues `+0.49` logits. Both effects beat matched-random controls. The rescue is deliberately harder than ablation and is not full reconstruction, but it shows that the upstream-conditioned terminal features recover a measurable slice of the missing IT-token margin.

Finally, a direct handoff test perturbs upstream computation and re-measures the same terminal features. Injecting IT mid-to-preterminal computation into the weak `U_PT,L_IT` hybrid rescues `+1.71` logits of IT-token margin, with `+0.13` logits mediated by the selected terminal features. The reverse intervention, replacing the same preterminal computation in native IT with PT computation, causes a `+3.57` logit drop, with `+0.53` mediated. A terminal-entry patch gives the expected upper bound (`+5.15` total, `+0.71` mediated). Thus the feature bridge is not only terminal: preterminal state changes drive a measurable part of the terminal sparse-feature readout.

**Structured boundary-state rescue.** A held-out boundary-state rescue gives the same conclusion without crosscoder features. On Llama-3.1 descendants, we fit PCA directions to train-split descendant-minus-base boundary-state shifts and inject held-out projections into the weak base-upstream/descendant-late hybrid. At terminal boundary 31, a rank-256 projection closes `0.71` of the missing IT-token margin; the full held-out delta gives the expected upper bound (`0.97`). Matched Gaussian and random full directions are near zero, and sign-flipped directions go negative. We use this as structured residual-state support for upstream-conditioned readout, not as recipe-specific or completion-level evidence.

Held-out autointerp makes the mediated feature set readable but not load-bearing. Across `225` interpreted features from the clean terminal-crosscoder families, validation reaches mean AUROC `0.886`; we use these labels descriptively while the causal evidence comes from feature edits. As a targeted semantic check, a predeclared `structure_readout` bucket gives a monotone positive edit across clean crosscoder families; Appendix E reports the full dose response and controls.

### 3.5 What This Measures

The first-divergence event is a distributional fact: the released PT and IT checkpoints prefer different next tokens at that prefix. The factorial asks a different question: once the token contrast is fixed, how much of the IT late-stack effect ports across upstream states? The answer is: some, but not most. The IT late stack can add IT-token margin from PT upstream, but most of its matched-context readout effect appears only when the upstream state is IT-shaped.

This distinction is why the result is more than "PT and IT differ." Random local disagreements, pre-divergence future-token scoring, and pre-late commitment controls all reduce or preserve the interaction in the predicted directions. The feature bridge then shows that part of the window-level interaction is carried by terminal sparse features that are more causally important under IT-shaped upstream state, can partially rescue the weak PT-upstream hybrid, and respond when upstream/preterminal computation is patched.

A supporting late-signature analysis stays in the appendix. Endpoint-matched late-refinement signatures explain why late/terminal windows were a good target: after matching on final entropy, confidence, and margin, IT late predictions remain farther from their own final readout than PT late predictions under both raw and tuned probes (`+0.425` and `+0.762` nats; Appendix F). This is a localization signature, not the mechanism itself.

Contemporaneous token-level RLVR work reaches a complementary conclusion from the outside: Sparse but Critical shows that a small set of shifted token decisions can carry large downstream effects under cross-sampling interventions (Meng et al., 2026). Our question is internal and paired-checkpoint: at the selected token where behavior first changes, how does the IT-token preference become a logit inside the model?

---

## 4. Related Work

**Late refinement and FFN readout.** Feed-forward layers promote vocabulary-space concepts and progressively refine predictions (Geva et al., 2022a,b). Layerwise intervention studies describe late residual sharpening (Lad et al., 2025), calibration analyses find an upper-layer confidence-adjustment phase (Joshi et al., 2025), and tuned lenses operationalize layerwise prediction refinement (nostalgebraist, 2020; Belrose et al., 2023). These works establish late refinement as plausible. Our contribution is to measure how it differs across paired PT/IT checkpoints at natural first-divergence tokens.

**Post-training model diffs.** Wu et al. (2024) study behavioral shifts from language modeling to instruction following. Du et al. (2025) compare base and post-trained checkpoints mechanistically across knowledge, truthfulness, refusal, and confidence, finding both transferable and weakly transferable directions. Zhao, Ziser, and Cohen (2024) show layer-structured task information in instruction-tuned models. Chaudhury (2025) finds preference-tuning effects localized to a mid-layer subspace in Llama-3.2-1B. Sparse but Critical analyzes RLVR as sparse token-level distribution shifts whose cross-sampled substitutions affect reasoning trajectories (Meng et al., 2026). We add an internal paired-checkpoint counterfactual: at the first token where PT and IT disagree, how much of the IT-token margin comes from a portable late-stack effect, and how much requires IT-shaped upstream state? This complements mid-layer localization results: our late effect is a readout of IT-shaped upstream state, not a claim that alignment information lives only in late layers.

**Activation patching and feature-level model diffing.** Activation patching requires care because metric choice, intervention direction, and off-manifold hybrids affect interpretation (Heimersheim and Nanda, 2024). We therefore report intervention-scoped readout effects and validate them with readout swaps, diagonal reconstruction, interpolation, low-anomaly filtering, label-swap nulls, signed-permutation controls, and random-disagreement baselines. Cross-model activation patching across base and fine-tuned variants is the closest methodological precedent (Prakash et al., 2024): their target is an entity-tracking mechanism and whether fine-tuning enhances an existing task circuit, while ours is the first natural PT/IT next-token disagreement and how much the IT late readout margin ports across upstream states. Sparse crosscoders provide a complementary route for model diffing (Lindsey et al., 2024), with known sparsity-artifact pitfalls (Minder et al., 2025). Unlike global sparse model-diff work, we use crosscoders after defining a paired-checkpoint causal estimand and ask which terminal features mediate that estimand: causally ranked terminal features partially mediate the interaction, matter more under IT-shaped upstream state, and partially rescue the weak PT-upstream hybrid.

**Automated feature interpretation.** LLM-based neuron interpretation commonly follows a generate-and-score pattern: propose a natural-language feature hypothesis, then test it on held-out examples (Bills et al., 2023; Huang et al., 2023). We follow that pattern for terminal crosscoder features. Labels are descriptive evidence; feature ablations provide the causal evidence.

**Novelty.** Several ingredients have precedents: late refinement, FFN vocabulary promotion, activation patching, global base/instruct diffing, and sparse model-diff features. The qualitative possibility of upstream/late co-adaptation is also not surprising given prior fine-tuning and transferability work. The new object is the paired-checkpoint first-divergence factorial estimand. It measures the portable-vs-conditioned split in the IT-minus-PT late-stack replacement effect at the natural token where PT and IT first disagree. The feature-level contribution is the bridge from this window-level interaction to terminal sparse features: the same causally ranked features mediate part of the readout interaction, are selectively necessary under IT-shaped upstream state, partially rescue the missing margin when inserted into the PT-upstream hybrid, and are partly driven by upstream/preterminal computation.

---

## 5. Discussion, Scope, and Next Tests

### 5.1 Interpretation

The interpretation suggested by these results is that released base-to-instruct transitions reshape how existing computation is read out, rather than adding a fully portable late instruction module. This is compatible with a broader view of post-training as selecting, amplifying, or consolidating reachable behaviors from the pretrained model, but our evidence is narrower: it concerns dense base/instruct checkpoint pairs at first-divergence token readouts. RLVR analyses such as Sparse but Critical are complementary context because they study token-level post-training shifts from outside the model; our factorial asks where the selected token preference becomes a logit inside paired checkpoints.

The same-base recipe check is informative because OpenMath2 is not a weak descendant. It is a math-specialized SFT of the same Llama-3.1-8B base, trained on OpenMathInstruct-2, and was released because it improves strongly on math benchmarks (Toshniwal et al., 2024). On our math-domain support, its matched-context late effect is also large (`+3.275`), but the same late effect is already portable from base upstream (`+3.430`), leaving no positive upstream-conditioned interaction (`-0.154`). Instruction-like descendants remain upstream-conditioned on the same math support (`+1.670` mean).

This makes OpenMath2 more than a foil. It suggests a recipe-structured distinction: a domain math SFT can create a mostly portable late computation for its token contrast, while instruction-following descendants more often require native upstream state to expose the same late readout. That pattern is compatible with task-specific skill-localization results, where fine-tuned skills can be grafted through small parameter subsets (Panigrahi et al., 2023), but our unit is activation-state portability rather than parameter sparsity. Within this same-base comparison, the factorial distinguishes portable task computation from co-adapted upstream-to-late readout.

The Tulu-3 and OLMo-2 lineages illustrate how the same estimand can be reused beyond final base/instruct pairs. In both released lineages, the measured interaction is already partly present after SFT and largely present by the DPO/preference-trained checkpoint, with the final checkpoint strongest. These are token-factorial checkpoint comparisons, not universal stage attributions.

### 5.2 Scope

The scope is narrow by design. First, the estimand is local to first-divergence next-token readouts: it targets the earliest point where a released PT/IT pair changes preference under shared history, not an average over all model behavior. Position-stratified and domain stress tests show that the interaction persists beyond the earliest generated positions, while its magnitude varies with prompt domain and generated position.

Second, the interventions are window-level compatibility tests. Hybrid-state validation makes practical artifact explanations unlikely, constrained continuation shows short-horizon likelihood persistence rather than free-running behavior, and terminal crosscoders expose a partial, terminal, quality-gated sparse-feature trace of the interaction. These checks provide concentrated feature-level mediation, rescue, and upstream-handoff evidence, but not the complete circuit or an on-manifold natural-trajectory effect size. The headline logit margins are dimensionalized as odds and native-shift shares in §3.1; they should not be read as direct estimates of completion-level behavior.

Third, the empirical scope is five dense core PT/IT pairs, one same-base recipe control, and two released dense stage lineages. DeepSeek-V2-Lite stays appendix-only because MoE routing and expert swaps require different controls. Architecture and MoE generalization are therefore next-step questions, not claims made by the core dense-family result. The recipe and stage-lineage results are fixed-support token-factorial case studies, not universal training-stage attributions.

### 5.3 Practical Implications and Next Tests

The immediate practical implication is methodological: late-patching, steering, and model surgery studies on paired checkpoints should test whether late effects are portable across upstream states before treating them as standalone late-layer mechanisms.

The most direct next test is to make the upstream side sparse as well: train or interpret middle/preterminal features directly and test which of them drive the upstream-conditioned terminal features.

---

## 6. Conclusion

First-divergence model diffing turns a vague question -- "do late layers explain the base-to-instruct difference?" -- into a paired-checkpoint counterfactual. At the first token where released PT and IT checkpoints disagree, the IT-minus-PT late-stack replacement effect is partially portable but mostly upstream-conditioned: the IT late stack has a positive effect from PT upstream, yet its effect is much larger from IT-shaped upstream state. The result is positive across the five dense core families, stronger for instruction-like descendants than a same-base domain fine-tune, coherent along two released stage lineages, and robust to targeted checks for readout choice, selected-token support, pre-late commitment, label orientation, and hybrid-state failure.

The mechanistic bridge is also graded rather than all-or-nothing. Middle windows are relatively more identity-selective, late and terminal windows are more margin/readout-sensitive, and terminal sparse features carry a concentrated part of the readout interaction. Those features are more causally important under IT-shaped upstream state, their native-IT activation pattern partially rescues the weak PT-upstream hybrid, and upstream/preterminal patches drive a measurable part of their mediated effect. Together with structured boundary-state rescue, the picture is a portable-vs-conditioned late-readout decomposition in released dense base/instruct checkpoint contrasts, not a fully portable late-only update.

---

## References

Aghajanyan, A., et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL 2021*.

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. *NeurIPS 2024*.

Belrose, N., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112.

Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., Sutskever, I., Leike, J., Wu, J., & Saunders, W. (2023). Language Models Can Explain Neurons in Language Models. OpenAI.

Chuang, Y., et al. (2024). DoLA: Decoding by Contrasting Layers Improves Factuality. *ICLR 2024*.

Chaudhury, A. (2025). Alignment is Localized: A Causal Probe into Preference Layers. arXiv:2510.16167.

Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS 2023*.

Deiseroth, B., Meuer, M., Gritsch, N., Eichenberg, C., Schramowski, P., Assenmacher, M., & Kersting, K. (2024). Divergent Token Metrics: Measuring Degradation to Prune Away LLM Components -- and Optimize Quantization. *NAACL 2024*.

Du, H., Li, W., Cai, M., Saraipour, K., Zhang, Z., Lakkaraju, H., Sun, Y., & Zhang, S. (2025). How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence. *COLM 2025*.

Geva, M., Schuster, R., Berant, J., & Levy, O. (2022a). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2022*.

Geva, M., Caciularu, A., Wang, K. R., & Goldberg, Y. (2022b). Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. *EMNLP 2022*.

Heimersheim, S., & Nanda, N. (2024). How to Use and Interpret Activation Patching. arXiv:2404.15255.

Huang, J., Geiger, A., D'Oosterlinck, K., Wu, Z., & Potts, C. (2023). Rigorously Assessing Natural Language Explanations of Neurons. *BlackboxNLP 2023*.

Joshi, A., Ahmad, A., & Modi, A. (2025). Calibration Across Layers: Understanding Calibration Evolution in LLMs. *EMNLP 2025*.

Lad, V., Lee, J. H., Gurnee, W., & Tegmark, M. (2025). The Remarkable Robustness of LLMs: Stages of Inference? *NeurIPS 2025*.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., et al. (2025). Tulu 3: Pushing Frontiers in Open Language Model Post-Training. *COLM 2025*.

Lin, B. Y., et al. (2024). The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning. *ICLR 2024*.

Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., & Olah, C. (2024). Sparse Crosscoders for Cross-Layer Features and Model Diffing. *Transformer Circuits Thread*.

Makelov, A., Lange, G., & Nanda, N. (2024). Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control. arXiv:2405.08366.

Meng, H., Huang, K., Wei, S., Ma, C., Yang, S., Wang, X., Wang, G., & Ding, B. (2026). Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs. arXiv:2603.22446.

Minder, J., Dumas, C., Juang, C., Chughtai, B., & Nanda, N. (2025). Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning. *NeurIPS 2025*.

Panigrahi, A., Saunshi, N., Zhao, H., & Arora, S. (2023). Task-Specific Skill Localization in Fine-tuned Language Models. *ICML 2023*.

Prakash, N., Shaham, T. R., Haklay, T., Belinkov, Y., & Bau, D. (2024). Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking. *ICLR 2024*.

Team OLMo, Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., et al. (2025). 2 OLMo 2 Furious. *COLM 2025*.

Toshniwal, S., Du, W., Moshkov, I., Kisacanin, B., Ayrapetyan, A., & Gitman, I. (2024). OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data. arXiv:2410.01560.

Wu, X., Yao, W., Chen, J., Pan, X., Wang, X., Liu, N., & Yu, D. (2024). From Language Modeling to Instruction Following: Understanding the Behavior Shift in LLMs after Instruction Tuning. *NAACL 2024*.

Zhao, Z., Ziser, Y., & Cohen, S. B. (2024). Layer by Layer: Uncovering Where Multi-Task Learning Happens in Instruction-Tuned Large Language Models. *EMNLP 2024*.

---

## Appendix Roadmap

The main text is written around stable claim names. For details, start with Appendix B for the main four-cell factorial, Appendix C for validation controls, Appendix D for depth anatomy, and Appendix E for sparse terminal features. Numeric run IDs appear only in file paths or script names; they are provenance labels, not concepts the reader needs to parse.

| Claim | Main location | Appendix | Artifact/script pointer |
|---|---|---|---|
| Minimal reproducibility snapshot | §2.1 | A, B, I | model registry, dataset manifests, raw first-divergence records |
| Core-5 first-divergence interaction, amplification scale, and portable share | §3.1 | B | `results/paper_synthesis/exp23_core5/`; `scripts/analysis/build_exp23_core5_synthesis.py` |
| Validation ladder | §3.2 | C | hybrid-state validation, random-disagreement baselines, token-support audit, pre-late commitment control |
| Recipe, continuation, and released stage-lineage checks | §3.3 | G | same-base Llama recipe control; constrained continuation bridge; fixed-support Tulu-3 and OLMo-2 Base/SFT/DPO/Final stage sweeps |
| Depth and terminal anatomy | §3.4 | D | identity/margin handoff, terminal-depth audit, terminal MLP audit |
| Terminal feature mediation, upstream-conditioning, rescue, structured boundary-state rescue, handoff, and structure-bucket validation | §3.4 | E | terminal crosscoder synthesis, hardening runs, upstream-conditioning audit, feature rescue, structured boundary-state rescue, preterminal handoff, autointerp taxonomy, structure-readout edit |
| Late refinement/readout signatures | §3.5 | F | endpoint-matched KL, late MLP random controls |
| Architecture and MoE scope | §5 | H | dense/MoE scope note |

Prompt-bootstrap CIs in the main text are conditional precision estimates over sampled prompts and released checkpoints. They are paired with family-level summaries or family ranges where a claim could otherwise be mistaken for a population-level model-family generalization.

| Appendix | Supports | Does not prove |
|---|---|---|
| B | Core-5 interaction magnitude, amplification, portable share, family consistency, and readout robustness. | Population-level generalization over all dense or post-trained models. |
| C | Main artifact explanations are unlikely: broken hybrids, arbitrary token support, and pre-late commitment. | Hybrid passes are natural deployment trajectories or completion-level behavior estimates. |
| D | Depth anatomy: middle windows are relatively more identity-selective while late/terminal windows are more margin-sensitive. | A complete circuit or a unique layer boundary. |
| E | Sparse terminal features and structured boundary-state shifts partially mediate, gate, and rescue the terminal readout interaction. | Full mechanism recovery, recipe-unique boundary directions, or feature monosemanticity. |
| F | Late/terminal windows were a motivated target. | The upstream-conditioned mechanism itself. |
| G | Same-base recipe control, constrained continuation scoring, and two released dense lineages show specificity beyond generic fine-tune compatibility, a one-token artifact, or one final checkpoint accident. | Isolated causal attribution to SFT, DPO, or RLVR algorithms, or natural-rollout behavior for every recipe. |
| H | Dense-family scope and MoE limitations are explicit. | MoE generalization. |
| I | Reviewer-facing reproduction levels and artifact roots. | That full raw GPU reruns are cheap. |

---

## Appendix A: Model Scope and Statistical Reporting

**Claim supported.** The paper uses fixed released checkpoint pairs, fixed prompt supports, and a consistent statistical reporting convention.

**Primary evidence.** Full checkpoint revisions, prompt mixes, and the exact scale definitions used in §3.1.

**What this does not prove.** The released checkpoints are not controlled training-recipe ablations.

**Where to audit.** Model registry, dataset manifests, and artifact roots are consolidated in Appendix I.

**Core-5 set.** Llama 3.1 8B, Qwen 3 4B, Mistral 7B, OLMo 2 7B, and Qwen2.5 32B. Qwen2.5 32B is included as the scale check in the core first-divergence synthesis.

**Core-small support set.** Llama 3.1 8B, Qwen 3 4B, Mistral 7B, and OLMo 2 7B. Supporting identity/margin, terminal MLP, crosscoder, and KL analyses use this smaller-family scope unless explicitly marked otherwise. The Qwen2.5 32B pair is included in the main factorial and omitted from these support analyses for compute.

**Prompt set.** The Core-5 factorial uses the holdout-600 prompt mix: `GOV-CONV/GOV-FORMAT/SAFETY = 300/150/150`. The Qwen2.5 32B raw run was collected on a larger support, but the Core-5 synthesis restricts it to the same holdout mix (`599` valid events because one safety prompt has no first divergence).

| Family | PT checkpoint | PT revision | IT checkpoint | IT revision | Notes |
|---|---|---|---|---|---|
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | Dense GQA. |
| Qwen 3 4B | `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | Dense GQA. |
| Mistral 7B | `mistralai/Mistral-7B-v0.3` | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | Dense attention in v0.3 config. |
| OLMo 2 7B | `allenai/OLMo-2-1124-7B` | `7df9a82518afdecae4e8c026b27adccc8c1f0032` | `allenai/OLMo-2-1124-7B-Instruct` | `470b1fba1ae01581f270116362ee4aa1b97f4c84` | Released stage lineage available. |
| Qwen2.5 32B | `Qwen/Qwen2.5-32B` | `1818d35814b8319459f4bd55ed1ac8709630f003` | `Qwen/Qwen2.5-32B-Instruct` | `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` | Core-5 scale check. |

The Core-5 synthesis combines stored per-family prompt-bootstrap estimates for the five families in §2.1. The amplification ratio reported in §3.1 is:

`late-stack amplification = [Y(U_IT,L_IT) - Y(U_IT,L_PT)] / [Y(U_PT,L_IT) - Y(U_PT,L_PT)]`.

This is the main scale reference because it compares the same IT late-stack replacement under the two upstream states. We also report a secondary native-shift scale, computed inside the same 2x2:

`native PT->IT diagonal margin shift = Y(U_IT,L_IT) - Y(U_PT,L_PT) = late_weight_effect + upstream_context_effect`.

The reported interaction share is `interaction / native diagonal margin shift`. This is a scale reference, not a claim that the interaction linearly decomposes all behavioral difference.

---

## Appendix B: Main First-Divergence Factorial Audit Trail

**Claim supported.** The Core-5 first-divergence interaction is positive across families, partially portable, and mostly upstream-conditioned under both common readouts.

**Primary evidence.** Core-5 common-IT/common-PT four-cell summaries, family-level interaction ranges, and the content/reasoning stress test.

**What this does not prove.** The prompt-bootstrap intervals are conditional on sampled prompts and released checkpoints; they are not population-level uncertainty over all post-training recipes.

**Where to audit.** Core synthesis artifacts, raw first-divergence mirrors, and Qwen2.5 32B support are listed in Appendix I.2.

Main Core-5 effects:

| Scope/readout | PT-up late effect | IT-up late effect | Interaction | Amplification | Portable share | Native shift | Interaction share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Core-5, common-IT | `+0.759` `[+0.682, +0.835]` | `+2.439` `[+2.344, +2.533]` | `+1.680` `[+1.604, +1.756]` | `3.2x` | `31.1%` | `+4.991` | `33.7%` |
| Core-5, common-PT | `+0.784` `[+0.717, +0.850]` | `+2.488` `[+2.397, +2.578]` | `+1.704` `[+1.628, +1.780]` | `3.2x` | `31.5%` | `+5.051` | `33.7%` |

Core-5 family-level common-IT interactions:

| Family | Interaction | Native diagonal shift | Interaction share |
|---|---:|---:|---:|
| Llama 3.1 8B | `+1.253` | `+5.358` | `23.4%` |
| Qwen2.5 32B | `+1.302` | `+3.995` | `32.6%` |
| Qwen 3 4B | `+1.464` | `+3.938` | `37.2%` |
| OLMo 2 7B | `+1.847` | `+5.227` | `35.3%` |
| Mistral 7B | `+2.534` | `+6.437` | `39.4%` |

The Core-5 family interaction range is `+1.253` to `+2.534` logits, with median `+1.464`. Interaction share ranges from `23.4%` to `39.4%`.

The portable-share family range is similarly heterogeneous. Under common-IT readout, `late_it_given_pt_upstream / late_it_given_it_upstream` ranges from `19.5%` to `44.3%`, with median `29.2%`; the Core-5 family-balanced center is `31.1%`. We therefore use `31.1%` as a center-of-mass summary, not as a constant across families.

The Core-small label-swap null is computed from the compatibility-permutation synthesis in Appendix I.2.

On the content/reasoning stress-test support, the late-only PT-upstream term is `-1.176` and the upstream x late interaction is `+1.812` (`[+1.721, +1.901]`).

The Qwen2.5 32B scale-check artifacts include both the larger raw run and the matched-support holdout synthesis; Appendix I.2 lists both audit paths.

---

## Appendix C: Validation Controls

**Claim supported.** The main upstream x late interaction is not made uninformative by broken hybrids, arbitrary selected-token support, or pre-late commitment.

**Primary evidence.** Hybrid interpolation is smooth and positive, random local disagreements retain a reduced but same-sign interaction, pre-divergence future-token scoring is near zero, and pre-late commitment restrictions preserve the interaction.

**What this does not prove.** These controls do not make hybrid states natural deployment trajectories, nor do they estimate completion-level behavior; they show that the main artifact explanations are unlikely to account for the estimand.

**Where to audit.** Full validation artifacts and scripts are listed in Appendix I.2.

| Check | Result | Takeaway |
|---|---:|---|
| Hybrid endpoint interaction | `+1.794` | Constructed cells reproduce the expected endpoint effect. |
| PT-to-IT interpolation slope | `+1.831` | The effect grows smoothly along the residual-state interpolation. |
| Signed-permutation random/observed ratio | `0.31x` | Matched-magnitude random signed patches do not reproduce the effect. |
| Random local disagreement | `+1.005` (`56%` of first divergence) | First divergence is high-signal, not the only same-sign support. |
| Pre-divergence future token pair | `+0.059` (`3%`) | The token pair matters; arbitrary earlier prefixes do not carry the effect. |

### C.1 Hybrid-State Validation

Key checks are summarized over the Core-small support families:

| Check | Result |
|---|---:|
| Endpoint interaction, common-IT | `+1.794` |
| PT-to-IT interpolation slope | `+1.831` |
| Signed-permutation random/observed ratio | `0.31x` |

These checks target the main off-manifold worry. A pathological hybrid artifact would more naturally appear as endpoint mismatch, non-smooth interpolation, or a matched-magnitude random intervention producing similar effects. We do not see that pattern.

### C.2 Selection and Token-Support Controls

The main-text figure and table below report Core-small family-balanced means computed from the per-family random-disagreement summaries.

| Condition | Interaction | Share of first divergence |
|---|---:|---:|
| True first divergence | `+1.794` | `100%` |
| Random local disagreement, source-balanced | `+1.005` | `56%` |
| Random PT-rollout disagreement | `+0.836` | `47%` |
| Random IT-rollout disagreement | `+1.190` | `66%` |
| Pre-divergence prefix, future token pair | `+0.059` | `3%` |

Random local disagreements are later and more content-token-heavy than first divergences, yet their factorial interaction is smaller. The selected-support audit likewise finds that most judged first-divergence events fall in substantive instruction, safety, formatting, response-shaping, or semantic-content categories rather than pure surface formatting. We keep the full category table in the artifact report because it is a support audit, not part of the evidence spine.

### C.3 Pre-Late Commitment Control

The support-run control restricts to events where the IT boundary readout does not yet favor `t_IT`, bins events by IT boundary margin, and fits boundary-margin controls. In all three views the interaction remains positive. This rules out the simplest "the late stack is irrelevant because the boundary already committed" reading without promoting these support-run magnitudes to Core-5 headline estimates.

---

## Appendix D: Depth and Terminal Anatomy

**Claim supported.** Middle windows are relatively more candidate/identity-selective, while late and terminal windows are more margin/readout-sensitive.

**Primary evidence.** Identity-transfer and margin-support tables, terminal-depth retention, and terminal-MLP margin interaction.

**What this does not prove.** The depth windows are not modular circuits, and the boundary between "middle" and "late" is graded.

**Where to audit.** Depth-anatomy artifacts are listed in Appendix I.2.

Main depth-anatomy quantities:

| Readout | Early | Middle | Late / terminal | Interpretation |
|---|---:|---:|---:|---|
| PT host: IT-token identity transfer | - | `25.6%` | `18.8%` | Middle substitutions transfer candidate identity more often. |
| IT host: PT-token identity transfer | - | `28.2%` | `21.5%` | Mirror direction gives the same identity pattern. |
| Pure IT MLP support for `t_IT` | `-0.085` | `+0.136` | `+0.986` | Native IT-token support is late-concentrated. |
| PT-host late MLP margin gain | - | - | `+0.004` | Late MLP updates alone are near zero in PT upstream state. |
| Source decomposition interaction | - | - | `+0.360` | MLP-level readout also shows context gating. |

In the Core-small support set, the final-three stack retains `52%` of the same-prompt full-late interaction; the final block alone retains `23%`. Final-three MLP substitutions transfer IT-token identity `8.8%` of the time, with terminal MLP margin interaction `+0.524` (`[+0.491, +0.558]`). The final layer alone gives terminal MLP margin interaction `+0.146` (`[+0.128, +0.165]`).

---

## Appendix E: Sparse Terminal Feature Bridge

**Claim supported.** Causally ranked terminal crosscoder features and structured boundary-state shifts carry concentrated, partial bridges for the terminal readout interaction.

**Primary evidence.** Top-200 causal features mediate `26-48%` of the terminal interaction in three quality-gated families, matter more under IT-shaped upstream state, partially rescue the weak hybrid, and respond to upstream/preterminal patches. Separately, train-fit PCA components of the descendant-minus-base boundary-state shift rescue the missing margin at terminal boundaries.

**What this does not prove.** This is not full circuit recovery, and it does not prove feature monosemanticity. The boundary-state rescue is not recipe-unique and does not estimate completion-level behavior. OLMo is excluded from feature-level claims because its terminal crosscoder did not pass the reconstruction gate.

**Where to audit.** Crosscoder training, mediation, gating, rescue, structured boundary-state rescue, handoff, autointerp, and structure-readout artifacts are listed in Appendix I.2.

### E.1 Terminal Crosscoder Mediation and Upstream-Conditioning

| Family | Terminal scope | VE / density status | Top causal feature drop | Share of interaction | Matched random | Paper role |
|---|---|---|---:|---:|---:|---|
| Llama | final 3 layers | VE min `0.774`, L0 `64`, alive max `0.096` | `+0.599` `[+0.469, +0.733]` | `48%` | `-0.209` `[-0.255, -0.165]` | clean |
| Mistral | final 3 layers | VE min `0.786`, L0 `64`, alive max `0.089` | `+0.684` `[+0.600, +0.764]` | `26%` | `-0.100` `[-0.159, -0.044]` | clean |
| Qwen | final 2 layers | layer VE `0.957/0.960` and `0.967/0.970` | `+0.324` | `37%` | `-0.033` | clean |

OLMo terminal crosscoder quality did not pass the predeclared reconstruction gate, so OLMo is excluded from feature-level claims.

The mediation curves sweep the number of ablated causally ranked features. The main table reports top-200 because it is fixed across families and far from the full-dictionary reconstruction setting; the curves show saturation rather than a single hand-picked feature count. The top-200 set is a small causally ranked subset of the crosscoder dictionary, not a full reconstruction. Recovering `26-48%` of the exposed terminal interaction from this subset is therefore a concentration result; matched-random features do not reproduce it.

| Family | top-200 share | top-500 share | Saturation read |
|---|---:|---:|---|
| Llama | `48%` | `52%` | modest additional distributed mass |
| Mistral | `26%` | `29%` | modest additional distributed mass |
| Qwen | `37%` | `38%` | mostly saturated by top-200 |

The same feature sets show upstream-conditioned causal importance.

For the same causally ranked terminal features, we compare ablation effects in the `U_IT,L_IT` and `U_PT,L_IT` cells. The primary feature causal gate is:

`[drop when ablating features in U_IT,L_IT] - [drop when ablating features in U_PT,L_IT]`.

Positive values mean the feature set matters more when the IT terminal stack receives IT-shaped upstream state. At top-200 features:

| Metric | Estimate |
|---|---:|
| Absolute causal feature gate, clean-family mean | `+0.703` |
| Causal gate minus matched-random features | `+0.887` |
| Causal gate minus top-active noncausal features | `+1.495` |
| Margin-weighted activation gate minus matched-random features | `+0.520` |

Per-family gates:

| Family | Absolute causal gate | Causal minus matched random |
|---|---:|---:|
| Llama | `+0.922` | `+1.254` |
| Mistral | `+0.816` | `+0.982` |
| Qwen | `+0.370` | `+0.426` |

Raw decoder-weighted activation mass is not uniformly higher under IT-shaped upstream state across families. We therefore use the finite-difference causal gate as the primary upstream-conditioning result and the signed margin-weighted activation gate as supporting evidence.

### E.2 Terminal Feature Rescue and Middle-to-Terminal Handoff

The rescue analysis tests a partial-sufficiency version of the same feature-level story. It runs on the three clean rescue families with quality-gated terminal crosscoders (Llama, Mistral, Qwen). OLMo is excluded because its current terminal crosscoder does not pass the reconstruction-quality gate needed for faithful feature-space rescue edits.

The edit takes the top-200 causal terminal feature activations from the native `U_IT,L_IT` pass and patches them into the `U_PT,L_IT` hybrid, decoded through the IT branch of the paired PT/IT crosscoder. The metric is rescued IT-token margin:

`Y(U_PT,L_IT + rescued features) - Y(U_PT,L_IT)`.

| Rescue metric, Llama/Mistral/Qwen family-balanced | Estimate | 95% CI |
|---|---:|---:|
| Direct top-200 causal feature rescue | `+0.494` | `[+0.451, +0.539]` |
| Direct rescue fraction | `8.1%` | `[5.5%, 10.3%]` |
| Causal minus matched-random rescue | `+0.561` | `[+0.510, +0.613]` |
| Causal minus matched-random rescue fraction | `10.8%` | `[7.6%, 13.7%]` |
| Causal minus same-delta-random rescue | `+0.471` | `[+0.427, +0.517]` |
| Causal minus same-delta-random rescue fraction | `8.3%` | `[5.7%, 10.6%]` |

Per-family direct rescue is Llama `+0.627`, Mistral `+0.755`, and Qwen `+0.101` logits. The alpha-zero no-edit sanity check is exact (`max |rescue_gain| = 0`). This is partial sufficiency rather than circuit recovery: the selected terminal features recover a measurable slice of the missing margin and beat both controls, but most of the `U_IT,L_IT` vs `U_PT,L_IT` gap remains.

The middle-to-terminal handoff analysis tests whether upstream/preterminal computation drives the selected terminal features, rather than merely co-occurring with them. It uses the same three quality-gated feature families and top-200 terminal causal features. In the rescue direction, we start from the weak `U_PT,L_IT` hybrid and replace an upstream MLP window with IT computation before running the IT terminal stack. In the degrade direction, we start from native `U_IT,L_IT` and replace the same window with PT computation. The mediated effect is the part of the margin change that disappears when the selected terminal features are ablated in both the base and perturbed passes.

| Handoff window / direction | Total margin effect | Terminal-feature-mediated part | Mediated fraction |
|---|---:|---:|---:|
| mid-to-preterminal rescue into `U_PT,L_IT` | `+1.714` `[+1.634, +1.791]` | `+0.132` `[+0.118, +0.147]` | `6.5%` |
| mid-to-preterminal degradation of `U_IT,L_IT` | `+3.570` `[+3.427, +3.721]` | `+0.527` `[+0.478, +0.576]` | `10.8%` |
| terminal-entry upper-bound rescue | `+5.147` `[+4.936, +5.351]` | `+0.705` `[+0.643, +0.767]` | `12.5%` |
| terminal-entry upper-bound degradation | `+5.147` `[+4.953, +5.358]` | `+0.705` `[+0.644, +0.762]` | `12.5%` |

The mid-to-preterminal mediated effect beats matched-random features (`+0.188` rescue; `+0.655` degradation), same-delta random directions (`+0.115` rescue; `+0.390` degradation), and top-active noncausal features (`+0.496` rescue; `+0.928` degradation). The terminal-entry rows are upper-bound sanity checks: they patch directly at the boundary into the terminal readout, so they should be larger than nonterminal windows. The late-preterminal-only window is weaker, especially in Qwen's event-permutation null, so the paper-facing claim is about mid-to-preterminal/preterminal handoff, not a late-preterminal-only mechanism.

### E.3 Structured Boundary-State Rescue

The feature-rescue analysis edits a selected sparse terminal feature set. As a complementary check, we ask whether the missing upstream state itself has a structured low-rank form. For five Llama-3.1-8B descendants, we fit PCA components to train-split descendant-minus-base boundary-state shifts and inject held-out projections into the weak base-upstream/descendant-late hybrid. The rank-256 rows are the main structured-rescue test; the full-delta rows are upper-bound sanity checks. Closure fraction is:

`[rescued margin - floor margin] / [native descendant-upstream margin - floor margin]`.

| Boundary | Train-fit PCA rank-256 | Full-delta upper bound | Gaussian full | Random full | Sign-flip full |
|---|---:|---:|---:|---:|---:|
| 29 | `0.634` | `0.929` | `-0.019` | `-0.036` | `-0.816` |
| 31 | `0.707` | `0.966` | `-0.021` | `-0.058` | `-0.788` |

The same-base wrong-descendant controls are nonzero in the artifact report, so we do not interpret these PCA directions as recipe-unique. The defensible claim is narrower: the missing upstream contribution at terminal boundaries is a structured descendant-minus-base residual-state shift, not generic perturbation magnitude.

### E.4 Descriptive Autointerp and Structure-Readout Edit

Across `225` interpreted features from the clean terminal-crosscoder families, mean validation AUROC is `0.886`. We use these labels descriptively, not as causal evidence: the causal claim remains the mediation, upstream-conditioning, and rescue results above. The paper-facing semantic check is the narrower `structure_readout` bucket below, where a predeclared readable subset is edited and tested against controls.

The structure-readout edit tests one readable subset from the taxonomy rather than every label bucket. The predeclared `structure_readout` bucket contains `10` causal features across the three clean crosscoder families, with labels such as paragraph breaks, list openings, answer boundaries, and field separators. This is not an `N=10` statistical generalization claim: the features are the predeclared edit set, while the test is whether the edited terminal readout changes monotonically over prompts and families and beats matched controls. Editing this bucket inside the same terminal crosscoder windows gives a monotone dose response in interaction drop; matched-random and same-delta random controls are much smaller.

| Edit strength `alpha` | Structure bucket | Matched random | Same-delta random |
|---|---:|---:|---:|
| `0.0` | `0.000` | `0.000` | `0.000` |
| `0.5` | `+0.039` | `-0.015` | `+0.001` |
| `1.0` | `+0.078` | `-0.028` | `+0.012` |
| `1.5` | `+0.125` | `-0.041` | `+0.020` |
| `2.0` | `+0.180` | `-0.048` | `+0.039` |

At `alpha=2.0`, per-family structure-bucket interaction drops are Llama `+0.091`, Mistral `+0.339`, and Qwen `+0.110`. Magnitudes are heterogeneous, but all three signs are positive. We use only this selective structure/readout result in the paper-facing feature-label validation. Other bucket edits were run as diagnostics but are not part of the evidence spine because their feature support is smaller or more domain-specific.

---

## Appendix F: Motivating Late-Localization Signatures

**Claim supported.** Late/terminal windows were a motivated target because IT checkpoints retain stronger late-stage distance from their own final readout and learned late MLP substitutions move that signature while matched random projections do not.

**Primary evidence.** Endpoint-matched late KL estimates and late MLP random-control comparisons.

**What this does not prove.** These analyses motivated the late/terminal focus; they are not used to establish the upstream x late interaction and are not interpreted as the mechanism.

**Where to audit.** Late-localization artifacts are listed in Appendix I.2.

Endpoint-matched late KL estimates:

| Metric | Estimate |
|---|---:|
| Raw late `KL(layer || own final)`, IT - PT | `+0.425` `[+0.356, +0.493]` nats |
| Tuned late `KL(layer || own final)`, IT - PT | `+0.762` `[+0.709, +0.814]` nats |
| Remaining adjacent JS, IT - PT | `+0.052` `[+0.048, +0.057]` |
| Future top-1 flips, IT - PT | `+0.203` `[+0.190, +0.215]` |

![Appendix Figure F1: Layerwise stabilization profile. IT checkpoints remain farther from their own final distribution late in the stack.](../results/exp09_cross_model_observational_replication/plots/L2_mean_kl_per_layer_tuned_dense5.png)

The dense-family true late random-control comparison is `+0.327` (`[+0.298, +0.359]`) for the learned late graft versus `+0.003` (`[-0.002, +0.008]`) for matched random residual projections.

![Appendix Figure F2: Late MLP graft/swap localization. Learned late MLP substitutions move the delayed-stabilization metric in both directions, while matched random late projections do not reproduce the effect.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png)

---

## Appendix G: Recipe, Continuation, and Released Stage-Lineage Checks

**Claim supported.** The interaction is not merely generic fine-tune compatibility or a one-token artifact, and fixed-support versions appear coherently along two released dense post-training lineages.

**Primary evidence.** On the same Llama-3.1-8B base, instruction-like descendants show positive interaction on instruction/format supports while OpenMath2 does not. A constrained continuation bridge shows the instruction-like interaction persists beyond the selected first token. On Base->Final support, Tulu-3 and OLMo-2 both show partial SFT presence and near-final DPO/preference-checkpoint interaction.

**What this does not prove.** These are token-factorial recipe, constrained-likelihood, and cumulative checkpoint comparisons, not isolated causal attributions to training algorithms or natural-rollout guarantees for each recipe.

**Where to audit.** Recipe-specificity, constrained-continuation, and stage-sweep artifacts are listed in Appendix I.2.

### G.1 Same-Base Recipe Specificity

The recipe-specificity check compares descendants of the same Llama-3.1-8B base. On instruction/format support, instruction-like descendants are consistently upstream-conditioned, while the OpenMath2 domain-specialized descendant does not reproduce the same interaction.

| Same-base descendant | Support | Interaction | Matched-context late effect | Portable late effect |
|---|---|---:|---:|---:|
| Meta Instruct | instruction/format | `+1.053` | `+2.070` | `+1.018` |
| Tulu SFT | instruction/format | `+0.287` | `+0.895` | `+0.608` |
| Tulu DPO | instruction/format | `+1.131` | `+2.196` | `+1.065` |
| Tulu Final/RLVR | instruction/format | `+1.365` | `+2.407` | `+1.041` |
| OpenMath2 | instruction/format | `-0.358` | `+1.052` | `+1.411` |

The instruction-like mean interaction is `+0.959` `[+0.907, +1.017]`; the matched instruction-like-minus-OpenMath contrast is `+1.335` logits after controlling for prompt category, generated-position bin, and token category. On math-domain support, OpenMath2 still does not show a positive interaction (`-0.154` `[-0.450, +0.128]`), even though both its matched-context and portable late effects are large (`+3.275` and `+3.430`). The instruction-like mean on the same math-domain support is `+1.670` `[+1.540, +1.795]`. Thus the OpenMath control is not merely failing because instruction/format prompts are out of domain; in these supports, its late effect is mostly portable rather than upstream-conditioned. The sign-flip null for the instruction-like orientation is clean (`+0.968` observed vs `+0.108` null 99.9th percentile; `p=5e-5`). We use this as evidence that the readout interaction is recipe-structured, not as a claim that the same token-level interaction directly predicts natural behavior for every descendant.

### G.2 Constrained Continuation Bridge

The constrained continuation bridge extends the Exp47 token-factorial without switching to full free-running benchmarks. For each first-divergence event, we force the descendant-preferred token, construct short native descendant and base continuations, and teacher-force those fixed candidate sequences through the same four hybrid cells. The sequence margin is the log probability of the descendant candidate minus the base candidate; `N=0` recovers the one-token factorial, while `N>0` asks whether the interaction persists over short continuations.

N=0 reproduction is close but not bit-exact because this run re-scores the cells in bf16/runtime conditions: median maximum drift is `0.125`, q99 is `0.375`, and `61/13664` comparisons exceed `0.5`. We therefore use aggregate horizon estimates rather than eventwise exact equality.

| Instruction-like descendants, common-IT readout | `C_N` interaction | Tail-only `C_N` | Events |
|---:|---:|---:|---:|
| `N=0` | `+1.50` `[+1.45, +1.56]` | -- | `5432` |
| `N=1` | `+1.97` `[+1.90, +2.04]` | `+0.43` `[+0.39, +0.46]` | `4801` |
| `N=2` | `+1.74` `[+1.66, +1.81]` | `+0.56` `[+0.51, +0.60]` | `3607` |
| `N=4` | `+2.06` `[+1.97, +2.16]` | `+0.89` `[+0.83, +0.96]` | `3471` |
| `N=8` | `+2.71` `[+2.59, +2.84]` | `+1.54` `[+1.45, +1.64]` (`+0.193/token`) | `3422` |

Controls keep the interpretation local. At `N=8`, the same-forced-descendant-tail control is also positive (`+2.46` `[+2.35, +2.59]`), showing that the effect persists after the forced first token; shuffled descendant tails are much smaller (`+0.87` `[+0.77, +0.97]`), showing that coherent native descendant tails carry substantially more interaction than arbitrary descendant tails. This is constrained likelihood evidence, not a natural-rollout behavior estimate.

OpenMath2 again has a different profile. On math-domain support, the common-IT interaction is near zero at `N=0` and negative at `N=8` (`-4.67` `[-5.52, -3.82]`), while common-PT becomes positive by `N=8` (`+1.81` `[+0.98, +2.61]`). We use this as a readout-sensitive diagnostic for recipe specificity, not as a universal sequence-level taxonomy of fine-tunes.

### G.3 Tulu-3 Fixed-Support Stage Sweep

The primary Tulu analysis fixes the support to Base->Final first-divergence prefixes for the Llama-3.1-8B Base and Tulu-3 final checkpoint, then scores SFT, DPO, and Final on the same `t_Base`/`t_Final` contrast. The checkpoints share architecture; Tulu adds special tokens, so the preflight validates identical raw prompt token IDs and rejects target tokens outside the shared base vocabulary.

| Tulu stage on fixed Base->Final support | Upstream x late interaction | Relative to final contrast | Native top-1 picks `t_Final` |
|---|---:|---:|---:|
| Base | `0` by definition | `0%` | `0.2%` |
| SFT | `+0.419` `[+0.349, +0.491]` | `28.8%` `[25.5%, 31.8%]` | `56.9%` |
| DPO | `+1.216` `[+1.090, +1.341]` | `83.6%` `[81.5%, 85.7%]` | `90.4%` |
| Final/RLVR | `+1.455` `[+1.316, +1.606]` | `100%` | `99.1%` |

Base interaction is zero by definition because Base is the reference checkpoint; the nonzero native top-1 rate reflects rare cases where the Base native readout still selects the final-token label under the fixed token contrast.

The fixed-support label-swap null passes the same orientation test as the main factorial: the observed final interaction is `+1.455`, while the null 99.9th percentile is `+0.296` (`p=5e-5`). Native readout is nearly identical (`+1.470` final interaction). Position `>=3` remains positive for all stages (`+0.172`, `+0.770`, `+0.770`).

Two base-anchored support checks ask whether the Base->Final support is doing the work. On Base->SFT support, the final checkpoint interaction is `+1.322`; on Base->DPO support, it is `+1.436`. Both label-swap nulls pass at `p=5e-5`, and both show the same qualitative pattern: SFT is real but smaller, while the DPO checkpoint reaches most of the final-support interaction.

| Tulu support | Valid events | SFT interaction | DPO interaction | Final interaction |
|---|---:|---:|---:|---:|
| Base->Final | `585/600` | `+0.419` | `+1.216` | `+1.455` |
| Base->SFT | `564/600` | `+0.401` | `+1.146` | `+1.322` |
| Base->DPO | `583/600` | `+0.427` | `+1.241` | `+1.436` |

### G.4 OLMo-2 Fixed-Support Stage Sweep

The primary fixed-support sweep fixes the support to Base->RLVR first-divergence prefixes and scores every intermediate checkpoint against the same `t_Base`/`t_RLVR` contrast. This makes SFT, DPO, and RLVR cumulative estimates comparable on the same local support. The older adjacent-pair analysis is retained only as historical motivation because each adjacent contrast uses its own first-divergence support and token labels; those adjacent estimates are useful local contrasts, but they are not additive attributions to the final Base->RLVR contrast.

| Stage on fixed Base->RLVR support | Upstream x late interaction | Relative to final contrast | Native top-1 picks `t_RLVR` |
|---|---:|---:|---:|
| Base | `0` by definition | `0%` | `0.0%` |
| SFT | `+0.773` `[+0.674, +0.873]` | `40.2%` `[37.6%, 42.8%]` | `61.0%` |
| DPO | `+1.629` `[+1.473, +1.793]` | `84.7%` `[83.3%, 86.0%]` | `93.0%` |
| RLVR/Instruct | `+1.924` `[+1.747, +2.104]` | `100%` | `99.7%` |

Base interaction is zero by definition because Base is the reference checkpoint.

The fixed-support label-swap null passes the same orientation test as the main factorial: the observed RLVR interaction is `+1.924`, while the null 99.9th percentile is `+0.382` (`p=5e-5`). Position `>=3` remains positive for all stages (`+0.283`, `+0.677`, `+0.813`). The result is a local lineage case study: in this released OLMo-2 path, the measured upstream-conditioned interaction is partly present in the SFT checkpoint, largely present in the DPO checkpoint, and strongest in the final RLVR/Instruct checkpoint.

---

## Appendix H: Architecture and MoE Scope

**Claim supported.** The current paper is a dense-family result, with architecture and MoE scope explicitly bounded.

**Primary evidence.** Core-5 covers dense transformer families plus a 32B dense scale check; MoE is artifact-only.

**What this does not prove.** The result may not transfer unchanged to MoE models, expert-routing interventions, or controlled attention-architecture comparisons.

**Where to audit.** Dense/MoE scope and artifact roots are summarized in Appendix I.

The Core-5 set covers dense transformer families only. It includes one Mistral family with an attention variant and one 32B Qwen scale check, but it is not a controlled architecture sweep. MoE generalization remains open: DeepSeek-V2-Lite is artifact-only because expert routing, expert swaps, and sparse activation patterns require additional controls beyond the dense-stack factorial used here. The clean follow-up is to run the same first-divergence factorial across multiple MoE base/instruct pairs with expert-routing controls and to separate attention-pattern effects from post-training recipe effects.

---

## Appendix I: Reproducibility and Artifact Map

**Claim supported.** The paper can be audited from committed summary artifacts, with raw rerun scope and hardware requirements made explicit.

**Primary evidence.** CPU synthesis paths reproduce paper-facing tables/figures; raw intervention reruns are mapped separately.

**What this does not prove.** Full raw GPU reproduction is cheap or necessary for every reviewer.

**Where to audit.** This appendix is the audit map.

### I.1 Reviewer-Facing Reproduction Path

| Reproduction level | What it reproduces | Hardware |
|---|---|---|
| CPU synthesis | Paper tables and figures from committed JSON/CSV artifacts. | CPU |
| Small raw rerun | One 4B/7B family first-divergence factorial and analysis. | 1-8 A100/H100 GPUs, depending on batching |
| Full raw rerun | Core-small dense runs plus Qwen2.5 32B scale check and feature experiments. | Multi-GPU A100/H100 jobs |


### I.2 Full Artifact Map

| Claim | Command/script family | Primary artifact |
|---|---|---|
| Core-5 upstream x late interaction | `scripts/analysis/build_exp23_core5_synthesis.py`; Core-5 figure generated from `exp23_core5_family_effects.csv` | `results/paper_synthesis/exp23_core5/exp23_core5_core_effects.csv`; `results/paper_synthesis/exp23_core5/exp23_core5_family_effects.csv`; `results/paper_synthesis/exp23_core5/exp23_core5_interaction.png` |
| Core first-divergence raw mirrors | raw first-divergence collectors and analysis scripts | `gs://pt-vs-it-results/results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/`; `gs://pt-vs-it-results/results/exp24_32b_external_validity/exp24_qwen25_32b_yanda_all_raw_20260428_101125.tar.zst`; `results/paper_synthesis/exp24_32b_external_validity/exp24_32b_holdout600_summary.json`; `results/paper_synthesis/exp24_32b_external_validity/exp24_32b_holdout600_position_sensitivity.csv` |
| Position sensitivity | same plus `scripts/analysis/analyze_first_divergence_position_sensitivity.py` | `results/paper_synthesis/exp23_core5/exp23_core5_position_sensitivity.csv`; `results/paper_synthesis/exp23_position_sensitivity_table.csv` |
| Label-swap null | `scripts/analysis/analyze_exp23_compatibility_permutation.py` | `results/paper_synthesis/exp23_core_small_compatibility_permutation/` |
| Off-manifold sanity audit | `scripts/analysis/analyze_exp23_offmanifold_sanity.py` | `results/paper_synthesis/exp23_offmanifold_sanity/` |
| Content/reasoning stress test | content/reasoning residual analysis | `results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/analysis/exp23_summary.json` |
| Hybrid-state validation | `scripts/run/run_exp36_offmanifold_validation_runpod.sh`; `scripts/analysis/analyze_exp36_offmanifold_validation.py` | `results/exp36_offmanifold_validation/exp36_offmanifold_dense5_full_a100x8_20260502_233904/analysis/summary.json`; `results/exp36_offmanifold_validation/exp36_offmanifold_dense5_full_a100x8_20260502_233904/analysis/exp36_offmanifold_validation_report.md`; `results/exp36_offmanifold_validation/exp36_offmanifold_dense5_full_a100x8_20260502_233904/analysis/interpolation_dose_response.png`; `results/exp36_offmanifold_validation/exp36_offmanifold_dense5_full_a100x8_20260502_233904/analysis/low_anomaly_robustness.png` |
| Selection baselines and token-support control | `scripts/run/run_exp37_random_prefix_baseline_runpod.sh`; `scripts/analysis/analyze_exp37_random_prefix_baseline.py`; `scripts/analysis/analyze_exp37_token_support_control.py`; `scripts/plot/plot_exp37_selection_baselines_paper.py` | `results/exp37_random_prefix_baseline/exp37_full_dense5_auth_xetfast_h100x8_20260503_002609/analysis/summary.json`; `results/exp37_random_prefix_baseline/exp37_full_dense5_auth_xetfast_h100x8_20260503_002609/analysis/effects.csv`; `results/exp37_random_prefix_baseline/exp37_full_dense5_auth_xetfast_h100x8_20260503_002609/analysis/exp37_matched_prefix_baselines.png`; `results/paper_synthesis/exp37_core_small_selection_baseline/selection_baselines_core_small.png`; `results/exp37_random_prefix_baseline/exp37_full_dense5_auth_xetfast_h100x8_20260503_002609/analysis/token_support_control/summary.json` |
| Selected-support audit | `scripts/analysis/analyze_first_divergence_token_support.py` | `results/first_divergence_token_support/dense5_llm_gpt55_20260503_121500/summary.json`; `results/first_divergence_token_support/dense5_llm_gpt55_20260503_121500/token_support_report.md` |
| Pre-late commitment control | `scripts/analysis/analyze_exp40_prelate_commitment_control.py`; exact collector in `src/poc/exp40_prelate_commitment_control/collect.py` | `results/exp40_prelate_commitment_control/exp40_exp20_layerwise_proxy_20260503_110001/analysis/summary.json`; `results/exp40_prelate_commitment_control/exp40_exp20_layerwise_proxy_20260503_110001/analysis/effects.csv`; `results/exp40_prelate_commitment_control/exp40_exp20_layerwise_proxy_20260503_110001/analysis/exp40_prelate_commitment_report.md`; `results/exp40_prelate_commitment_control/exp40_exp20_layerwise_proxy_20260503_110001/analysis/prelate_commitment_bins.png` |
| Endpoint-matched convergence gap | `scripts/analysis/build_exp22_endpoint_deconfounded_synthesis.py` | `results/exp09_cross_model_observational_replication/data/exp9_summary.json`; `results/exp09_cross_model_observational_replication/data/convergence_gap_values.json`; `results/paper_synthesis/exp22_endpoint_deconfounded_table.csv`; `results/exp09_cross_model_observational_replication/plots/L2_mean_kl_per_layer_tuned_dense5.png` |
| Late MLP random control | late-random-control analysis scripts | `results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/depth_ablation_metrics.json`; `results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_summary.json`; `results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/exp19B_summary_light.json`; `results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png` |
| Identity/margin handoff | `scripts/analysis/build_exp20_exp21_handoff_synthesis.py` | `results/paper_synthesis/exp20_exp21_handoff_table.csv`; `results/paper_synthesis/exp20_exp21_handoff_core_small.png`; `results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early/validation_analysis/summary.json`; `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/summary.json`; `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/effects.csv` |
| Terminal-depth and terminal-MLP audit | `scripts/analysis/analyze_exp31_terminal_depth_factorial.py`; terminal MLP analysis scripts | `results/exp31_terminal_depth_factorial/exp31_terminal_depth_full_a100x4_localdisk_fixedsched_20260502_021238/analysis/terminal_depth_summary.json`; `results/exp31_terminal_depth_factorial/exp31_terminal_depth_full_a100x4_localdisk_fixedsched_20260502_021238/analysis/terminal_depth_effects.csv`; `results/exp32_terminal_mlp_writeout/exp32_terminal_mlp_full_dense5_a100x8_w2_20260502_043950/analysis/exp32_terminal_mlp_summary.json`; `results/exp33_terminal_identity_margin/exp33_terminal_identity_margin_full_dense5_a100x8_overlap_20260502_0509/analysis/exp33_terminal_identity_margin_summary.json` |
| Terminal crosscoder mediation | `scripts/analysis/analyze_exp34_dense5_final_readout_crosscoder.py`; crosscoder hardening analysis | `results/paper_synthesis/exp34_dense5_final_readout_crosscoder/combined_dense5_20260503_0018/exp34_dense5_crosscoder_summary.json`; `results/paper_synthesis/exp34_core_feature_mediation/terminal_crosscoder_core3_mediation.png`; `results/exp38_qwen_olmo_final_layer_crosscoder_hardening/exp38_qwen_olmo_final_summary_20260503/analysis/exp38_qwen_olmo_decision_summary.json`; `results/exp30_final_readout_crosscoder_mediation/exp30_l31_paperfaithful_runpod_20260502_012105_a100x8/selected_d131072_k64/analysis/mediation_curve.png` |
| Terminal feature upstream-conditioning | `src/poc/exp42_terminal_feature_upstream_conditioning/`; `scripts/analysis/analyze_exp42_terminal_feature_upstream_conditioning.py` | `results/exp42_terminal_feature_upstream_conditioning/exp42_full_4fam_h100x8_20260503_155212/analysis/feature_gating_summary.json`; `results/exp42_terminal_feature_upstream_conditioning/exp42_full_4fam_h100x8_20260503_155212/analysis/feature_gating_report.md`; `results/exp42_terminal_feature_upstream_conditioning/exp42_full_4fam_h100x8_20260503_155212/analysis/feature_gating_by_family.png`; `results/exp42_terminal_feature_upstream_conditioning/exp42_full_4fam_h100x8_20260503_155212/analysis/feature_ablation_saturation.png` |
| Terminal feature rescue | `src/poc/exp43_feature_rescue_handoff/`; `scripts/plot/plot_exp43_feature_rescue_handoff.py` | `results/exp43_feature_rescue_handoff/exp43_full_h100x8_clean_20260503_182947/analysis/exp43_report.md`; `results/exp43_feature_rescue_handoff/exp43_full_h100x8_clean_20260503_182947/analysis/primary_family_balanced_effects.csv`; `results/exp43_feature_rescue_handoff/exp43_full_h100x8_clean_20260503_182947/analysis/exp43_family_balanced_rescue_gain.png` |
| Middle-to-terminal feature handoff | `scripts/run/run_exp44_middle_terminal_feature_handoff_runpod.sh`; `src/poc/exp44_middle_terminal_feature_handoff/analyze.py` | `results/exp44_middle_terminal_feature_handoff/exp44_primary_lmq_a100_20260503_combined/analysis/exp44_report.md`; `results/exp44_middle_terminal_feature_handoff/exp44_primary_lmq_a100_20260503_combined/analysis/primary_family_balanced_effects.csv`; `results/exp44_middle_terminal_feature_handoff/exp44_primary_lmq_a100_20260503_combined/analysis/handoff_control_differences.csv`; `results/exp44_middle_terminal_feature_handoff/exp44_primary_lmq_a100_20260503_combined/analysis/exp44_primary_handoff_effects.png` |
| Structured boundary-state rescue | `src/poc/exp48_static_chimera_sequence_validation/structured_rescue.py`; `src/poc/exp48_static_chimera_sequence_validation/analyze.py` | `results/exp48_static_chimera_sequence_validation/exp48_static_chimera_sequence_validation_20260504_1349_a100x16/analysis/structured_rescue_summary.csv`; `results/exp48_static_chimera_sequence_validation/exp48_static_chimera_sequence_validation_20260504_1349_a100x16/analysis/structured_rescue_closure_curves.png`; `results/exp48_static_chimera_sequence_validation/exp48_static_chimera_sequence_validation_20260504_1349_a100x16/analysis/summary.json` |
| Feature autointerp and taxonomy | `src/poc/exp39_causal_feature_interpretation/`; `scripts/analysis/exp39_causal_paper_taxonomy_llm.py` | `results/exp39_causal_feature_interpretation/exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/autointerp/label_validation.json`; `results/exp39_causal_feature_interpretation/exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/autointerp/llm_feature_labels.jsonl`; `results/exp39_causal_feature_interpretation/exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/dashboards/feature_dashboards.jsonl`; `results/exp39_causal_feature_interpretation/exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/analysis/` |
| Structure-readout bucket validation | `src/poc/exp41_causal_feature_bucket_steering/`; structure-readout analysis outputs | `results/exp41_causal_feature_bucket_steering/exp41_terminal_bucket_logit_full_h100x8_20260503_1520/analysis/exp41_logit_replay_summary.json`; `results/exp41_causal_feature_bucket_steering/exp41_terminal_bucket_logit_full_h100x8_20260503_1520/analysis/bucket_effects_by_model.csv`; `results/exp41_causal_feature_bucket_steering/exp41_terminal_bucket_logit_full_h100x8_20260503_1520/bucket_manifest/strict_primary/bucket_features.csv` |
| Same-base recipe specificity | `src/poc/exp47_same_base_recipe_specificity/`; `scripts/analysis/analyze_exp47_same_base_recipe_specificity.py` | `results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24/analysis/summary.json`; `results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24/analysis/portable_coadapted_table.csv`; `results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24/analysis/matched_support_effects.csv`; `results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24/analysis/recipe_domain_two_panel.png` |
| Constrained continuation bridge | `src/poc/exp49_constrained_continuation_bridge/`; Exp49 horizon analysis outputs | `results/exp49_constrained_continuation_bridge/exp49_full_20260504_223652_a100x8/analysis/summary.json`; `results/exp49_constrained_continuation_bridge/exp49_full_20260504_223652_a100x8/analysis/aggregate_effects.csv`; `results/exp49_constrained_continuation_bridge/exp49_full_20260504_223652_a100x8/analysis/sequence_effects.csv`; `results/exp49_constrained_continuation_bridge/exp49_full_20260504_223652_a100x8/analysis/plots/exp49_cumulative_interaction.png`; `results/exp49_constrained_continuation_bridge/exp49_full_20260504_223652_a100x8/analysis/plots/exp49_tail_interaction.png` |
| Tulu fixed-support stage sweep | `src/poc/exp46_tulu_fixed_support_stage_sweep/`; Exp46 analysis outputs | `results/exp46_tulu_fixed_support_stage_sweep/exp46_full_a100x8_localdisk_20260504_103624/analysis/summary.json`; `results/exp46_tulu_fixed_support_stage_sweep/exp46_full_a100x8_localdisk_20260504_103624/analysis/effects.csv`; `results/exp46_tulu_fixed_support_stage_sweep/exp46_full_a100x8_localdisk_20260504_103624/analysis/stage_fraction_ratios.csv`; `results/exp46_tulu_fixed_support_stage_sweep/exp46_full_a100x8_localdisk_20260504_103624/analysis/exp46_stage_decomposition.png`; `results/exp46_tulu_fixed_support_stage_sweep/exp46_full_base_to_S_a100x8_localdisk_20260504_104959/analysis/summary.json`; `results/exp46_tulu_fixed_support_stage_sweep/exp46_full_base_to_D_a100x8_localdisk_20260504_105605/analysis/summary.json` |
| OLMo fixed-support stage sweep | `scripts/analysis/analyze_exp35_olmo_base_anchored_stage_decomposition.py`; `scripts/analysis/build_exp35_stage_ratio_bootstrap.py` | `results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/summary.json`; `results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/effects.csv`; `results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/stage_ratio_bootstrap.csv`; `results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/exp35_stage_decomposition.png` |

All full reruns use bf16 inference and deterministic greedy decoding unless a script states otherwise. The summary audit is CPU-only and reads committed JSON/CSV artifacts. Reproducing raw 4B-8B intervention records requires multiple 80GB A100/H100 jobs; reproducing Qwen2.5 32B additionally requires the multi-GPU run or the committed paper-facing synthesis artifacts.
