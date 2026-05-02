# Post-Training Intensifies a Late Corrective Stage: A Paired-Checkpoint Model-Diffing Study Across Six Dense Language Models

**Anonymous authors** | NeurIPS 2026 Submission

---

## Abstract

Post-training changes what a language model says, but paired pretrained/post-trained checkpoints let us ask *where in the forward pass* that change becomes a next-token logit. We introduce a paired-checkpoint first-divergence factorial: at the first shared-history token where PT and IT prefer different next tokens, we cross PT/IT upstream residual state with PT/IT late stack and measure the IT-vs-PT token margin. Across six representative dense PT/IT pairs, the same IT late stack contributes far more IT-token margin from an IT-shaped upstream state than from a PT-shaped one; the upstream × late interaction is positive in every family (Gemma-removed `+1.71`, Dense-6 `+2.44`). This shows that post-training does not simply add a portable late update. Instead, it changes next-token formation through a middle-to-late handoff: upstream computation exposes IT-like candidates, while late and terminal MLP computation converts IT-shaped state into final margin. Supporting analyses triangulate this interpretation: post-training intensifies delayed stabilization, late residual-opposing MLP geometry, and IT-specific vulnerability to ablating that geometry, while identity/margin tests show middle windows are more candidate-selective and late windows are more readout-sensitive. The result is a local model-diffing account of representative dense PT-to-IT pipelines: late refinement is real, but its post-training effect is upstream-conditioned.

---

## 1. Introduction

When post-training changes the next token, where in the forward pass does that change become a next-token logit? A pretrained checkpoint (PT) and its post-trained descendant (IT) give a rare controlled setting for this question: the architecture and tokenizer are shared, but the forward pass has been altered by instruction tuning, preference optimization, reinforcement-style training, or a mixture of post-training stages. The natural unit of analysis is a *paired-checkpoint model diff*: not "what do late layers do?" — that question is well studied — but "what does post-training change about how late layers do it?"

The tempting one-cell answer is "late layers." Late transformer computation is close to the unembedding, and prior work already makes late-stage refinement plausible: feed-forward layers promote vocabulary-space concepts (Geva et al., 2022b), late layers sharpen or calibrate predictions (Lad et al., 2025; Joshi et al., 2025), and instruction-tuned models show layer-structured task information (Zhao, Ziser, and Cohen, 2024). But "late layers matter" is not enough for a model-diff claim. A late patch can look causal while actually measuring compatibility between a late stack and the upstream residual state it expects, and the late-refinement vocabulary established for individual checkpoints has not been measured as an *intensification* signature of post-training, with cross-family controls, at the natural token where two paired checkpoints first disagree.

We give three convergent measurements that, taken together, show post-training **intensifies a late corrective stage** already nascent in pretrained models. We use *corrective* descriptively, to denote the joint signature of delayed stabilization, intensified residual-opposing late-MLP geometry, and IT-specific causal vulnerability to ablating that geometry; we do not claim a direct mediation chain from geometry to the first-divergence interaction (feature-level mediation is the natural follow-up; §5). The three signatures are: (i) IT checkpoints remain farther from their own final next-token distribution late in the stack, even after endpoint matching; (ii) late IT MLP updates become more residual-opposing than PT updates in 5/5 dense families, and ablating the late residual-opposing component hurts IT next-token prediction roughly an order of magnitude more than PT; and (iii) at natural PT/IT first disagreements, the same IT late stack contributes far more IT-token margin from an IT-shaped upstream state than from a PT-shaped upstream state.

The third signature is the central paired-checkpoint causal test. We define the **first-divergence prefix** as the earliest shared-history prefix where PT and IT prefer different next tokens. At that prefix we cross the upstream residual state (`U_PT` or `U_IT`) with the downstream late stack (`L_PT` or `L_IT`) and score the margin `logit(t_IT) - logit(t_PT)`. If post-training added a portable late-only update, the IT late stack should add similar IT-token margin from either upstream state. It does not: across six dense PT/IT pairs the upstream x late interaction is positive in every family.

The contribution is therefore not a new claim that late computation reads upstream state. It is a paired-checkpoint model-diffing result: post-training changes next-token formation through a middle-to-late handoff. Middle-positioned MLP substitutions are more tied to divergent-token identity, while late and terminal MLPs shape the final margin. The evidence is intervention-scoped: it estimates effects on specified next-token readouts in constructed forward passes, not a complete feature-level circuit.

**Contributions.** The named object of this paper is the **paired-checkpoint first-divergence factorial**: a local counterfactual estimand for how post-training changes next-token formation at the exact token where PT and IT first disagree. The other analyses triangulate why that estimand behaves as it does.

1. **Late corrective stage as an *intensification* signature of post-training.** Lad et al. (2025) document late residual sharpening as a general stage of inference; Joshi et al. (2025) document late confidence calibration. Neither compares paired PT/IT checkpoints at the same natural disagreement token. We show that this late-stage configuration *intensifies* post-training across 5/5 dense families: IT delayed stabilization is positive under endpoint-matched controls, late MLP `δ-cosine` is more residual-opposing in IT than PT in every family, and learned late MLP substitutions localize the delay where matched random projections do not (`+0.327` vs `+0.003` nats).
2. **IT-specific causal vulnerability of the late residual-opposing component.** Removing late residual-opposing MLP components hurts IT own-token next-token prediction roughly an order of magnitude more than PT (`+0.0432` vs `+0.0004` NLL hurt; `+7.37` vs `+0.83` true-logit drop), with norm-preserving and same-magnitude random-removal controls. This causal asymmetry across paired checkpoints, with cross-family controls, has not previously been reported.
3. **First-divergence factorial estimand.** First-divergence factorial diffing crosses upstream residual state with downstream late stack at the natural token where PT and IT first prefer different next tokens. Across six dense PT/IT pairs the upstream × late interaction is positive in every family (Gemma-removed `+1.71`, Dense-6 `+2.44`); it is label-aligned (`p=5×10⁻⁵`), persists at generated positions `≥3`, and dissociates from the simple late-only term, which flips negative on a factual/reasoning stress test while the interaction stays positive (`+1.81`). Middle-positioned MLP substitutions transfer divergent-token identity more often than late (`26%` vs `18%`); late and terminal MLPs dominate margin/readout. To our knowledge no prior paired-checkpoint diffing work measures this estimand.

The first-divergence factorial is the new object; the supporting analyses make its interpretation hard to dismiss as a patching artifact or a generic late-layer fact. Together they show a recurring PT-to-IT model-diff pattern across representative dense instruction-following descendants: post-training changes next-token formation through a middle-to-late handoff, not by adding a portable late-only update. We use **IT** as shorthand for instruction-following post-trained descendants; the recipes are heterogeneous, so the main claim is about paired PT-versus-post-trained model diffs, not one training algorithm.

---

## 2. Setup

### 2.1 Model Sets

The main first-divergence factorial uses six dense PT/IT pairs: Gemma 3 4B, Llama 3.1 8B, Qwen 3 4B, Mistral 7B, OLMo 2 7B, and Qwen2.5 32B. We call this the **Dense-6 core set**. Several supporting analyses were run before the 32B extension and use the five 4B-8B dense families; we call this the **4B-8B Dense-5 support set**. When reporting a conservative first-divergence magnitude, **Gemma-removed Dense-5** means the Dense-6 core set excluding Gemma, so it includes Qwen2.5 32B.

The DeepSeek-V2-Lite MoE pair remains appendix-only because dense MLP grafts and MoE routing interventions are not the same intervention.

### 2.2 First-Divergence Factorial

For each prompt, we generate under PT and IT until the first prefix where their top-1 next tokens differ. Let those tokens be `t_PT` and `t_IT`. The readout is

`Y(U,L) = logit(t_IT) - logit(t_PT)`.

Larger `Y` means the forward pass favors the IT divergent token over the PT divergent token. At a pre-specified late boundary we run four hybrid passes:

| Upstream state | PT late stack `L_PT` | IT late stack `L_IT` |
|---|---:|---:|
| PT upstream `U_PT` | `Y(U_PT,L_PT)` | `Y(U_PT,L_IT)` |
| IT upstream `U_IT` | `Y(U_IT,L_PT)` | `Y(U_IT,L_IT)` |

The primary estimand is the upstream x late interaction:

`[Y(U_IT,L_IT) - Y(U_IT,L_PT)] - [Y(U_PT,L_IT) - Y(U_PT,L_PT)]`.

Equivalently, it asks how much larger the IT-late-stack effect is when the upstream state is IT-shaped rather than PT-shaped. Common-IT and common-PT readouts score all four cells with one fixed final norm, `lm_head`, and real-token mask; native readouts use each host checkpoint's own readout. Unless stated otherwise, the main factorial numbers use common-IT.

All raw-shared first-divergence and residual-state runs force both PT and IT branches to raw text and validate identical raw prompt token IDs before comparing residual states. Position 0 is therefore the first generated token after the full raw prompt, not a chat-template artifact.

### 2.3 Intervals and Scope

First-divergence intervals are 95% percentile bootstraps over prompt clusters within family, then averaged across families. Dense-6 paper-facing intervals combine stored per-family prompt-bootstrap estimates with the 32B family contribution from the paper-facing synthesis. Supporting KL/graft intervals are family-bootstrap or prompt-bootstrap as stated in the relevant section.

All causal language below is readout- and intervention-scoped: replacing an upstream state, late stack, or MLP component changes a specified next-token readout in a constructed forward pass.

---

## 3. Results

The results follow a claim ladder. Section 3.1 establishes a late corrective/readout signature and its IT-specific natural-rollout importance. Section 3.2 gives the primary paired-checkpoint causal test at the first PT/IT divergent token. Section 3.3 decomposes the depth anatomy into identity, margin, and terminal readout components. Section 3.4 uses one released OLMo-2 lineage as a case study of how the same estimand appears during staged post-training.

### 3.1 A Late Corrective Stage, Intensified Post-Training

**Claim.** Prior work already identifies late residual sharpening (Lad et al., 2025) and late confidence calibration (Joshi et al., 2025) in general LLM forward passes, without paired-checkpoint comparison at PT/IT disagreement tokens. We measure three convergent signatures showing this late stage *intensifies* after post-training: IT checkpoints stabilize later toward their final distribution, learned late MLP substitutions localize this delay (matched random late projections do not), late MLP updates become more residual-opposing in 5/5 dense families, and the residual-opposing component is dramatically more important for IT own-token prediction than for PT.

**Delayed stabilization.** Under native free-running decoding, IT models remain farther from their own final next-token distribution than PT models do through much of the stack. This is not only an endpoint artifact. After matching token steps on final entropy, final top-1 confidence, and final top-1/top-2 margin, the late IT-minus-PT `KL(layer || own final)` gap remains positive under raw probes (`+0.425` nats, 95% CI `[+0.356, +0.493]`) and tuned probes (`+0.762`, `[+0.709, +0.814]`). Endpoint-free path checks are also positive: remaining adjacent JS is `+0.052` (`[+0.048, +0.057]`), and future top-1 flips are `+0.203` (`[+0.190, +0.215]`).

![Figure 1: Late corrective/readout signature. IT checkpoints remain farther from their own final distribution late in the stack, motivating matched-prefix late-window tests.](../results/exp09_cross_model_observational_replication/plots/L2_mean_kl_per_layer_tuned_dense5.png)

**Late MLP localization.** Under identical token histories, late MLP substitutions have the largest tested effect on this delayed-stabilization metric. In a PT host, grafting IT MLPs into the late window increases final-20% KL by `+0.338` nats on the dense-family mean, while early and middle windows are near zero. In the mirror direction, replacing late IT MLPs with PT MLPs reduces the IT delay by `-0.509` nats, again the largest tested window effect. A matched random-control follow-up rules out generic late-window fragility: the true learned late graft gives `+0.327` nats (`[+0.298, +0.359]`), while a matched random residual-projection control gives `+0.003` (`[-0.002, +0.008]`).

![Figure 2: Late MLP graft/swap localization. Learned late MLP substitutions move the delayed-stabilization metric in both directions, while matched random late projections do not reproduce the effect.](../results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png)

**Residual-opposing geometry.** The same late stage has a geometric signature. In the 4B-8B Dense-5 support set, the late-window IT-minus-PT shift in MLP `delta_cosine` is negative in every family: Gemma `-0.189`, Llama `-0.029`, Qwen `-0.016`, Mistral `-0.081`, and OLMo `-0.021`. The terminal-layer shifts are also negative in every family. This means post-training makes late MLP updates more opposed to the current residual stream direction; the geometry is also present in PT (not zero) but intensifies after post-training in every family. We use this as a signature of the late corrective stage, not as a claim that one geometric scalar explains all of the effect.

![Figure 3: Delta-cosine profiles. IT late MLP updates are more residual-opposing than PT in the dense families, with heterogeneous magnitude across models.](../results/exp09_cross_model_observational_replication/plots/L1_delta_cosine_6panel.png)

**Natural-rollout importance.** Exp27 tests whether this residual-opposing component matters while each model predicts its own generated continuation. Removing the late residual-opposing component leaves PT own-token NLL essentially unchanged (`+0.0004`, 95% CI `[-0.0016, +0.0027]`) but hurts IT (`+0.0432`, `[+0.0418, +0.0448]`), for an IT-minus-PT difference of `+0.0428` (`[+0.0403, +0.0453]`). The true-token logit drop shows the same asymmetry: PT drops `+0.827` logits and IT drops `+7.370`, for an IT-minus-PT difference of `+6.542` (`[+6.403, +6.684]`). A norm-preserving version preserves the IT-specific NLL hurt (`+0.0336` IT-minus-PT), while same-magnitude random removals do not reproduce the pattern. Thus residual-opposing late MLP geometry is not merely descriptive: it is a causal vulnerability of IT next-token prediction in natural rollouts.

| Exp27 natural-rollout intervention | PT NLL hurt | IT NLL hurt | IT-PT NLL hurt | IT-PT true-logit drop |
|---|---:|---:|---:|---:|
| Remove residual-opposing component | `+0.0004` `[-0.0016, +0.0027]` | `+0.0432` `[+0.0418, +0.0448]` | `+0.0428` `[+0.0403, +0.0453]` | `+6.542` `[+6.403, +6.684]` |
| Norm-preserving removal | `+0.0007` `[-0.0012, +0.0027]` | `+0.0342` `[+0.0329, +0.0356]` | `+0.0336` `[+0.0312, +0.0357]` | `+5.836` `[+5.699, +5.974]` |
| Flip residual-opposing component | `+0.0269` `[+0.0237, +0.0304]` | `+0.1013` `[+0.0983, +0.1043]` | `+0.0744` `[+0.0699, +0.0786]` | `+9.186` `[+9.011, +9.362]` |

This section supports the existence and post-training intensification of a late corrective stage. It does not by itself prove that residual opposition fully mediates the convergence gap or the first-divergence interaction; Section 3.2 supplies the paired-checkpoint causal test.

### 3.2 First-Divergence Factorial: Late Computation Is Upstream-Conditioned

**Claim.** At the first natural PT/IT next-token disagreement, IT late computation is not a portable late-only update. It contributes much more IT-token margin when the upstream residual state is already IT-shaped.

Under common-IT readout, swapping in the IT late stack shifts the IT-vs-PT divergent-token margin by `+0.639` logits (`[+0.570, +0.709]`) from a PT upstream state, but by `+3.076` (`[+2.978, +3.174]`) from an IT upstream state. The Dense-6 upstream x late interaction is therefore `+2.437` logits (`[+2.353, +2.521]`). Because Gemma is the largest family-specific effect, we report the Gemma-removed Dense-5 estimate as the conservative magnitude headline: `+1.709` (`[+1.637, +1.780]`). The common-PT readout cross-check gives the same conclusion: interaction `+2.421` (`[+2.337, +2.506]`).

![Figure 4: First-divergence upstream x late interaction. The same IT late stack has a much larger IT-token margin effect from IT-shaped upstream state than from PT-shaped upstream state.](../results/paper_synthesis/exp23_dense6_core/exp23_dense6_interaction.png)

| Scope/readout | Late effect from PT upstream | Late effect from IT upstream | Upstream x late interaction |
|---|---:|---:|---:|
| Dense-6, common-IT | `+0.639` `[+0.570, +0.709]` | `+3.076` `[+2.978, +3.174]` | `+2.437` `[+2.353, +2.521]` |
| Gemma-removed Dense-5, common-IT | `+0.747` `[+0.673, +0.821]` | `+2.456` `[+2.364, +2.547]` | `+1.709` `[+1.637, +1.780]` |
| Dense-6, common-PT | `+0.662` `[+0.600, +0.724]` | `+3.083` `[+2.986, +3.180]` | `+2.421` `[+2.337, +2.506]` |

The 32B family is part of the Dense-6 core, not a separate pooled claim: Qwen2.5-32B alone gives a positive interaction of `+1.446` (`[+1.321, +1.569]`), with `+0.977` late effect from PT upstream and `+2.423` from IT upstream. The effect is therefore not restricted to 4B-8B models.

**Off-manifold sanity audit.** The factorial cells are hybrid forward passes, so they cannot be proven to be ordinary natural states. We therefore audit the five-family raw-record subset for practical patching artifacts. Diagonal no-op reconstruction is exact over `5,966` checks, common-IT and common-PT readouts give nearly identical interactions (`+2.635` vs `+2.610` logits), and off-diagonal trajectory metrics are never more than `1.074x` the worst diagonal mean. This addresses much of the practical numerical-artifact concern, but the estimand remains a constructed compatibility counterfactual rather than a complete natural-state circuit claim.

**Label and position controls.** A PT/IT label-swap null preserves each prompt's four factorial cell values but randomly swaps PT/IT labels. On the five-family raw-record subset, the observed interaction is far outside the null (`p=5e-5` over 20,000 permutations; null 99.9th percentile `+0.239` logits). The result is also not only a first-token opening effect. Removing immediate position-0 divergences leaves a Dense-6 interaction of `+2.079` (`[+1.963, +2.194]`); restricting to generated position `>=3` gives `+1.434` (`[+1.300, +1.569]`); position `>=5` gives `+1.480` (`[+1.276, +1.684]`). Position changes the regime mix and the magnitude, but the direction remains positive.

**Stress test.** On factual/reasoning prompts, the simple late-only term from PT upstream moves against the IT token (`-1.18` logits, `[-1.26, -1.09]`), while the upstream x late interaction remains positive (`+1.81`, `[+1.72, +1.90]`). This is the key falsifier for a portable late-only summary: late IT computation can be weak or even counterproductive from the wrong upstream state while still being effective in its matched IT context.

### 3.3 Depth Anatomy: Identity, Margin, and Terminal Blocks

**Claim.** The late corrective stage is organized as a middle-to-late handoff. Middle-positioned MLP substitutions are more tied to divergent-token identity; late and terminal MLPs are more tied to final margin/readout.

Exp20 asks whether substituting a depth window transfers the opposite checkpoint's divergent-token identity at the first-divergence prefix. In a PT host, middle-positioned IT MLP substitutions transfer the IT token more often than late substitutions (`26.0%` vs `17.6%`). The mirror direction gives the same pattern: in an IT host, middle PT substitutions transfer the PT token more often than late substitutions (`31.2%` vs `20.8%`). These percentages are well below 50%, so neither window is an independent token selector; the comparison is a relative localization signal.

Exp21 asks what the MLP updates write into the next-token margin. In native IT trajectories, late MLPs provide much stronger support for the IT divergent token than early or middle MLPs (`+0.789` late vs `+0.021` middle and `-0.041` early). But transplanting late IT MLP updates into a PT host has a near-zero fixed-prefix margin effect (`+0.004`, `[-0.001, +0.009]`). This is the MLP-level version of the Section 3.2 result: late MLP write-out is strong in IT context and weak as a portable PT-upstream insertion.

| Readout | Early | Middle | Late / terminal | Interpretation |
|---|---:|---:|---:|---|
| PT host: IT-token identity transfer | - | `26.0%` `[24.5%, 27.7%]` | `17.6%` `[16.2%, 18.9%]` | Middle substitutions transfer candidate identity more often. |
| IT host: PT-token identity transfer | - | `31.2%` `[29.6%, 32.9%]` | `20.8%` `[19.4%, 22.3%]` | Mirror direction gives the same identity pattern. |
| Pure IT MLP support for `t_IT` | `-0.041` `[-0.049, -0.032]` | `+0.021` `[+0.011, +0.032]` | `+0.789` `[+0.754, +0.825]` | Native IT-token support is late-concentrated. |
| PT host late MLP margin gain | - | - | `+0.004` `[-0.001, +0.009]` | Late MLP updates alone are weak in PT upstream state. |
| Source decomposition interaction | - | - | `+0.288` `[+0.277, +0.301]` | MLP-level readout also shows context gating. |

Terminal-depth audits sharpen the "late" side. Moving the factorial boundary to the final three transformer blocks preserves `57%` (`[56%, 58%]`) of the same-prompt full-late Dense-5 interaction; the final block alone preserves `33%` (`[32%, 34%]`). A terminal-only MLP follow-up shows weak identity transfer but strong margin effects: final-three MLP substitutions transfer IT-token identity only `8.4%` of the time, yet their terminal MLP margin interaction is `+1.068` (`[+1.009, +1.127]`); the final layer alone gives `+0.584` (`[+0.541, +0.630]`). Thus terminal layers carry a real readout component, but the full late stack remains stronger and terminal layers are not standalone identity selectors.

![Figure 5: Identity/margin handoff. Middle substitutions transfer divergent-token identity more often, while late and terminal MLPs dominate native IT-token support and margin/readout.](../results/paper_synthesis/exp20_exp21_handoff_synthesis.png)

### 3.4 OLMo-2 Case Study: The Handoff Appears Along the Lineage

OLMo-2 is useful because Ai2 releases a staged post-training path: Base, SFT, DPO, and RLVR/Instruct (Team OLMo et al., 2025; Lambert et al., 2025). We use it as a case study, not as a universal stage decomposition. Exp35 fixes the support to the same `587/600` valid Base->RLVR first-divergence prefixes and the same `t_Base`/`t_RLVR` token contrast, then asks how much of the final Base->RLVR upstream x late interaction is already present at each intermediate stage.

On this fixed support, the handoff grows monotonically across released checkpoints. The SFT checkpoint already expresses about `40.2%` of the final measured interaction (`[37.6%, 42.8%]`), the DPO checkpoint expresses `84.7%` (`[83.3%, 86.0%]`), and the RLVR/Instruct checkpoint gives the final value. The same support also shows monotone native adoption of the final RLVR token and a monotone shift toward more residual-opposing late MLP geometry.

| Stage on fixed Base->RLVR support | Upstream x late interaction | Share of final | Native top-1 picks `t_RLVR` | Late MLP `delta_cosine` |
|---|---:|---:|---:|---:|
| Base | `0` by definition | `0%` | `0.0%` | `+0.064` |
| SFT | `+0.773` `[+0.674, +0.873]` | `40.2%` `[37.6%, 42.8%]` | `61.0%` | `+0.032` |
| DPO | `+1.629` `[+1.473, +1.793]` | `84.7%` `[83.3%, 86.0%]` | `93.0%` | `+0.017` |
| RLVR/Instruct | `+1.924` `[+1.747, +2.104]` | `100%` | `99.7%` | `+0.014` |

![Figure 6: Fixed-support OLMo-2 stage case study. On Base->RLVR first-divergence prefixes, the final upstream x late handoff grows from SFT to DPO to RLVR/Instruct while native top-1 adoption and late residual-opposing geometry move in the same direction.](../results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/exp35_stage_decomposition.png)

The fixed-support label-swap null passes the same orientation test as the main factorial: the observed RLVR interaction is `+1.924`, while the null 99.9th percentile is `+0.382` (`p=5e-5`). Position `>=3` remains positive for all stages (`+0.283`, `+0.677`, `+0.813`). Thus, within this one released lineage, the paired-checkpoint estimand gives a mechanistic timeline: the final post-trained handoff is partly present in the SFT checkpoint, largely present in the DPO checkpoint, and strongest in the final RLVR/Instruct checkpoint. This does not imply that SFT, DPO, and RLVR always contribute these fractions in other model families or prompt regimes.

---

## 4. Related Work

**Late refinement and FFN readout.** A vocabulary for late-stage prediction refinement is well-established. Feed-forward layers promote vocabulary-space concepts and progressively refine predictions (Geva et al., 2022a,b). Layerwise intervention studies describe a late residual-sharpening stage (Lad et al., 2025); calibration analyses find an upper-layer confidence-correction phase with a low-dimensional residual-stream direction (Joshi et al., 2025). The tuned-lens framework operationalizes layerwise prediction refinement (nostalgebraist 2020; Belrose et al., 2023). These works study late-stage computation in individual checkpoints or general LLM forward passes; **none compares paired PT/IT checkpoints on the same first-divergence factorial axis**. Our contribution to this thread is to show that the same late-stage configuration *intensifies* post-training on the IT-vs-PT axis, with cross-family controls, IT-specific causal vulnerability, and a paired-checkpoint counterfactual at the natural disagreement token.

**Post-training model diffs.** Wu et al. (2024) study behavioral shifts from language modeling to instruction following. Du et al. (2025) compare base and post-trained checkpoints mechanistically across knowledge, truthfulness, refusal, and confidence — globally, not at the natural first-divergence prefix and not via a 2×2 upstream × late factorial. Zhao, Ziser, and Cohen (2024) show layer-structured task information in instruction-tuned models. We extend this paired-checkpoint thread by adding a local counterfactual: at the first PT/IT divergent token, does the IT late stack carry the margin by itself, or only with IT-shaped upstream state? This estimand is not in prior paired-checkpoint work to our knowledge.

**Activation patching and feature-level model diffing.** Activation patching requires care: metric choice, intervention direction, and off-manifold hybrids affect interpretation (Heimersheim and Nanda, 2024). We therefore report intervention-scoped readout effects, include common-IT and common-PT readout variants, assert numerical reconstruction at the diagonal cells, and use label-swap and matched-random controls. Cross-model activation patching across base and fine-tuned variants (Prakash et al., 2024) is the closest methodological precedent; their target is entity tracking and ours is the natural PT/IT next-token disagreement. Sparse crosscoders provide a complementary feature-level route for model diffing (Lindsey et al., 2024), while recent work shows that crosscoder sparsity artifacts can misidentify model-specific features (Minder et al., 2025). We treat crosscoder/transcoder feature mediation as the natural next step after the window-level handoff established here.

**Novelty.** Several ingredients have precedents: residual sharpening (Lad), late calibration (Joshi), FFN promotion (Geva), cross-model patching (Prakash), and global PT/IT diffing (Du). The new object is the paired-checkpoint first-divergence factorial estimand: at the natural token where PT and IT first disagree, we test whether the IT late stack carries the IT-token margin portably or only with IT-shaped upstream state. The other analyses triangulate this estimand's interpretation: cross-family delayed-stabilization localization with matched random controls, IT-specific causal vulnerability of the late residual-opposing component (Exp27, with two controls), depth-graded identity/margin decomposition, terminal-depth audit, and the OLMo-2 stage-progression case study. Together they characterize a recurring dense PT-to-IT model-diff pattern: post-training intensifies late corrective/readout computation, but that computation works through an upstream-conditioned handoff rather than a standalone late update.

---

## 5. Limitations and Next Tests

The evidence is window-level. Hybrid forward passes identify effects on specified readouts, not a complete feature-level circuit with named heads, MLP features, and subspaces. The off-manifold audit rules out obvious numerical reconstruction failures and degenerate hybrid trajectories, but it cannot make constructed hybrids fully natural or isolate every latent covariate. Residual-opposing geometry is a signature and causal vulnerability of IT next-token prediction, but Exp27 does not directly mediate the convergence-gap effect or the upstream x late interaction.

First-divergence events are selected disagreement regimes, not random token positions. This is the object of study: the first token where post-training changes the model's next-token preference. Position-stratified results show the interaction remains positive in later shared-history regimes, but position and prompt category change the magnitude.

The core scope is six dense PT/IT pairs. Supporting analyses remain 4B-8B Dense-5 where they were not rerun or pooled with Qwen2.5-32B. The DeepSeek MoE side case is kept out of the main pool because dense MLP window swaps and MoE routing/expert swaps require different controls.

The next decisive experiment is feature-level mediation: train crosscoders or transcoder-style adapters at the late boundary and ask whether sparse IT-specific features mediate the upstream x late interaction, while controlling for known crosscoder sparsity artifacts.

---

## 6. Conclusion

Post-training intensifies a late corrective stage that is already nascent in pretrained transformers. The intensification is measurable on three convergent axes: delayed stabilization (IT layers stay farther from their own final prediction), stronger residual-opposing late MLP geometry (5/5 dense families), and IT-specific causal vulnerability to ablating the residual-opposing component (~10× the PT effect). The central contribution is the paired-checkpoint first-divergence factorial: a per-prompt counterfactual that measures post-training's effect at the exact token where PT and IT first disagree. Across representative dense PT-to-IT pipelines, this estimand shows that the same IT late stack contributes far more IT-token margin from an IT-shaped upstream state than from a PT-shaped one. The depth anatomy is graded: middle-positioned MLP substitutions transfer divergent-token identity more often, while late and terminal MLPs dominate margin and readout. The supporting analyses triangulate the interpretation, but the new insight is the handoff itself: post-training reshapes next-token formation by making late readout depend on upstream state already shaped by the post-trained model.

---

## References

Aghajanyan, A., et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL 2021*.

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. *NeurIPS 2024*.

Belrose, N., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112.

Chuang, Y., et al. (2024). DoLA: Decoding by Contrasting Layers Improves Factuality. *ICLR 2024*.

Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS 2023*.

Deiseroth, B., Meuer, M., Gritsch, N., Eichenberg, C., Schramowski, P., Assenmacher, M., & Kersting, K. (2024). Divergent Token Metrics: Measuring Degradation to Prune Away LLM Components -- and Optimize Quantization. *NAACL 2024*.

Du, H., Li, W., Cai, M., Saraipour, K., Zhang, Z., Lakkaraju, H., Sun, Y., & Zhang, S. (2025). How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence. *COLM 2025*.

Geva, M., Schuster, R., Berant, J., & Levy, O. (2022a). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2022*.

Geva, M., Caciularu, A., Wang, K. R., & Goldberg, Y. (2022b). Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. *EMNLP 2022*.

Heimersheim, S., & Nanda, N. (2024). How to Use and Interpret Activation Patching. arXiv:2404.15255.

Joshi, A., Ahmad, A., & Modi, A. (2025). Calibration Across Layers: Understanding Calibration Evolution in LLMs. *EMNLP 2025*.

Lad, V., Lee, J. H., Gurnee, W., & Tegmark, M. (2025). The Remarkable Robustness of LLMs: Stages of Inference? *NeurIPS 2025*.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., et al. (2025). Tulu 3: Pushing Frontiers in Open Language Model Post-Training. *COLM 2025*.

Lin, B. Y., et al. (2024). The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning. *ICLR 2024*.

Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., & Olah, C. (2024). Sparse Crosscoders for Cross-Layer Features and Model Diffing. *Transformer Circuits Thread*.

Minder, J., Dumas, C., Juang, C., Chughtai, B., & Nanda, N. (2025). Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning. *NeurIPS 2025*.

Panigrahi, A., Saunshi, N., Zhao, H., & Arora, S. (2023). Task-Specific Skill Localization in Fine-tuned Language Models. *ICML 2023*.

Prakash, N., Shaham, T. R., Haklay, T., Belinkov, Y., & Bau, D. (2024). Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking. *ICLR 2024*.

Team OLMo, Walsh, P., Soldaini, L., Groeneveld, D., Lo, K., Arora, S., et al. (2025). 2 OLMo 2 Furious. *COLM 2025*.

Wu, X., Yao, W., Chen, J., Pan, X., Wang, X., Liu, N., & Yu, D. (2024). From Language Modeling to Instruction Following: Understanding the Behavior Shift in LLMs after Instruction Tuning. *NAACL 2024*.

Zhao, Z., Ziser, Y., & Cohen, S. B. (2024). Layer by Layer: Uncovering Where Multi-Task Learning Happens in Instruction-Tuned Large Language Models. *EMNLP 2024*.

---

## Appendix Guide

The appendices are organized by claim rather than experiment number.

| Need to check | Appendix | Contents |
|---|---|---|
| Model scope and checkpoints | A | Dense-6 core set, Dense-5 support set, checkpoint IDs, prompt modes. |
| Late corrective/readout signature | B | Exp9, Exp11/14, Exp19, Exp22, and Exp27 artifacts. |
| First-divergence factorial | C | Exp23/24 Dense-6 synthesis, label-swap null, position sensitivity, stress test. |
| Depth anatomy | D | Exp20/21 identity and write-out, Exp31/32/33 terminal-depth audits. |
| Stage progression | E | Fixed-support OLMo-2 Base/SFT/DPO/RLVR case study. |
| Auxiliary evidence | F | Exp16 JS, Exp15 behavior/human audit, Exp18 chronology, DeepSeek MoE, Exp26, Exp28/30. |
| Reproducibility | G | Artifact map, commands, bootstrap details, hardware notes. |

## Appendix A: Model Scope and Definitions

**Dense-6 core set.** Gemma 3 4B, Llama 3.1 8B, Qwen 3 4B, Mistral 7B, OLMo 2 7B, and Qwen2.5 32B. The first five families are 4B-8B scale; Qwen2.5 32B is included as the sixth dense family in the core first-divergence synthesis.

**4B-8B Dense-5 support set.** Gemma 3 4B, Llama 3.1 8B, Qwen 3 4B, Mistral 7B, and OLMo 2 7B. Supporting identity/margin, residual-opposition, terminal MLP, behavior, and KL analyses use this scope unless explicitly marked Dense-6.

**Gemma-removed Dense-5.** Dense-6 excluding Gemma: Llama 3.1 8B, Qwen 3 4B, Mistral 7B, OLMo 2 7B, and Qwen2.5 32B.

| Family | PT checkpoint | IT checkpoint | Notes |
|---|---|---|---|
| Gemma 3 4B | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` | Hybrid local/global attention. |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` | Dense GQA. |
| Qwen 3 4B | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B` | Dense GQA. |
| Mistral 7B | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | Sliding-window attention. |
| OLMo 2 7B | `allenai/OLMo-2-1124-7B` | `allenai/OLMo-2-1124-7B-Instruct` | Released stage lineage available. |
| Qwen2.5 32B | `Qwen/Qwen2.5-32B` | `Qwen/Qwen2.5-32B-Instruct` | Sixth dense core family. |

## Appendix B: Late Corrective/Readout Signature Artifacts

**Layerwise stabilization.** Primary artifacts:

- `results/exp09_cross_model_observational_replication/data/exp9_summary.json`
- `results/exp09_cross_model_observational_replication/data/convergence_gap_values.json`
- `results/paper_synthesis/exp22_endpoint_deconfounded_table.csv`

Endpoint-matched late KL estimates:

| Metric | Estimate |
|---|---:|
| Raw late `KL(layer || own final)`, IT - PT | `+0.425` `[+0.356, +0.493]` nats |
| Tuned late `KL(layer || own final)`, IT - PT | `+0.762` `[+0.709, +0.814]` nats |
| Remaining adjacent JS, IT - PT | `+0.052` `[+0.048, +0.057]` |
| Future top-1 flips, IT - PT | `+0.203` `[+0.190, +0.215]` |

**Late MLP localization.** Primary artifacts:

- `results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/depth_ablation_metrics.json`
- `results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_summary.json`
- `results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/exp19B_summary_light.json`

The dense-family true late random-control comparison is `+0.327` (`[+0.298, +0.359]`) for the learned late graft versus `+0.003` (`[-0.002, +0.008]`) for matched random residual projections.

**Residual-opposing geometry and Exp27.** Primary artifacts:

- `results/exp09_cross_model_observational_replication/plots/L1_delta_cosine_6panel.png`
- `results/exp27_natural_rollout_residual_opposition_ntp/exp27_full_dense5_combined_20260430_2050/analysis/exp27_summary.json`
- `results/exp27_natural_rollout_residual_opposition_ntp/exp27_full_dense5_combined_20260430_2050/analysis/exp27_effects.csv`

Exp27 is a natural-rollout own-token prediction test. It is not a same-prefix PT/IT factorial and is not used as a mediation test for Section 3.2.

## Appendix C: First-Divergence Factorial Details

Primary Dense-6 artifacts:

- `results/paper_synthesis/exp23_dense6_core/exp23_dense6_core_effects.csv`
- `results/paper_synthesis/exp23_dense6_core/exp23_dense6_family_effects.csv`
- `results/paper_synthesis/exp23_dense6_core/exp23_dense6_position_sensitivity.csv`
- `results/paper_synthesis/exp23_dense6_core/exp23_dense6_interaction.png`

The five-family raw-record label-swap null is computed from:

- `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/compatibility_permutation/`

The CPU-only off-manifold sanity audit is:

- `results/paper_synthesis/exp23_offmanifold_sanity/offmanifold_sanity_report.md`

The content/reasoning stress test is:

- `results/exp23_midlate_interaction_suite/exp23_content_reasoning_residual_20260427_0930_h100x8/analysis/exp23_summary.json`

Qwen2.5 32B artifacts:

- `results/exp24_32b_external_validity/exp24_qwen25_32b_full_eval_v21_20260427_194839/analysis/`
- `results/paper_synthesis/exp24_32b_external_validity/`

## Appendix D: Depth Anatomy Artifacts

Primary identity/margin synthesis:

- `results/paper_synthesis/exp20_exp21_handoff_table.csv`
- `results/paper_synthesis/exp20_exp21_handoff_synthesis.png`
- `results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early/validation_analysis/summary.json`
- `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/summary.json`
- `results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736/analysis/effects.csv`

Terminal-depth and terminal-MLP artifacts:

- `results/exp31_terminal_depth_factorial/exp31_terminal_depth_full_a100x4_localdisk_fixedsched_20260502_021238/analysis/terminal_depth_summary.json`
- `results/exp31_terminal_depth_factorial/exp31_terminal_depth_full_a100x4_localdisk_fixedsched_20260502_021238/analysis/terminal_depth_effects.csv`
- `results/exp32_terminal_mlp_writeout/exp32_terminal_mlp_full_dense5_a100x8_w2_20260502_043950/analysis/exp32_terminal_mlp_summary.json`
- `results/exp33_terminal_identity_margin/exp33_terminal_identity_margin_full_dense5_a100x8_overlap_20260502_0509/analysis/exp33_terminal_identity_margin_summary.json`

Exp32's local terminal MLP write-out proxy has the same sign as the terminal MLP margin interaction but much smaller magnitude (`+0.0867` last-three and `+0.1099` last-one). We keep it as a proxy rather than a load-bearing mediation claim.

## Appendix E: OLMo-2 Stage Progression

Primary artifacts:

- `results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/summary.json`
- `results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/effects.csv`
- `results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/stage_ratio_bootstrap.csv`
- `results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/exp35_stage_decomposition.png`

The primary stage analysis fixes the support to Base->RLVR first-divergence prefixes and scores every intermediate checkpoint against the same `t_Base`/`t_RLVR` contrast. This makes SFT, DPO, and RLVR cumulative estimates comparable on the same local support. The older adjacent-pair Exp25 analysis is retained only as historical motivation because each adjacent contrast uses its own first-divergence support and token labels.

The result is a local lineage case study: in this released OLMo-2 path, the measured handoff is partly present in the SFT checkpoint, largely present in the DPO checkpoint, and strongest in the final RLVR/Instruct checkpoint. It should not be read as a universal additive decomposition of SFT, DPO, and RLVR contributions across model families or prompt distributions.

## Appendix F: Auxiliary Evidence and Omitted Threads

**Exp26 residual-opposition mediation.** Useful but not main-text load-bearing because it is partial and family-heterogeneous. IT-target no-opposition drops the interaction by `+0.258` (`[+0.225, +0.293]`), about `9.8%`; flipping the component drops `+0.481`, about `18.3%`. Artifact: `results/exp26_residual_opposition_mediation/exp26_dense5_full_a100x8_20260429_111420/analysis/`.

**Exp16 same-history JS.** Shows PT/IT output distributions differ under identical histories without relying on free-running endpoint comparisons. It is support for separation, not the main convergence-gap proof. Artifact: `results/exp16_matched_prefix_js_gap/exp16_js_replay_runpod_20260422_075307/js_summary.json`.

**Exp15 behavior and human audit.** Directionally useful but secondary. Behavioral outputs move in the expected direction on aggregate, but unresolved rates and rater heterogeneity prevent family-level behavioral claims. Artifacts: `results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600_t512_dense5/` and `results/exp15_symmetric_behavioral_causality/human_eval/`.

**Exp18 chronology and older Gemma features.** These are explanatory and historical, not inference spine. They help build intuition for candidate flow and late feature redistribution but do not replace the architecture-agnostic Dense-6 factorial.

**Exp28/30 crosscoder pilots.** Interesting future-work evidence only. Single-model Llama crosscoder runs show causal feature sets can mediate part of the interaction, but the results are not cross-family and must be interpreted with recent crosscoder-artifact cautions.

**Rank-1 steering.** We do not use multi-model rank-1 steering as evidence for v26. Cross-model steering was mixed and PC1 explained only a modest share of IT-PT variance in earlier analyses.

## Appendix G: Reproducibility and Artifact Map

| Claim | Command/script family | Primary artifact |
|---|---|---|
| Dense-6 upstream x late interaction | `scripts/analysis/build_exp23_dense6_core_synthesis.py` | `results/paper_synthesis/exp23_dense6_core/` |
| Position sensitivity | same plus `scripts/analysis/analyze_first_divergence_position_sensitivity.py` | `results/paper_synthesis/exp23_dense6_core/exp23_dense6_position_sensitivity.csv` |
| Label-swap null | `scripts/analysis/analyze_exp23_compatibility_permutation.py` | `results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/compatibility_permutation/` |
| Off-manifold sanity audit | `scripts/analysis/analyze_exp23_offmanifold_sanity.py` | `results/paper_synthesis/exp23_offmanifold_sanity/` |
| Endpoint-matched convergence gap | `scripts/analysis/build_exp22_endpoint_deconfounded_synthesis.py` | `results/paper_synthesis/exp22_endpoint_deconfounded_table.csv` |
| Late MLP random control | Exp19 analysis scripts | `results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/` |
| Residual-opposition natural rollout | `scripts/analysis/analyze_exp27_natural_rollout_residual_opposition_ntp.py` | `results/exp27_natural_rollout_residual_opposition_ntp/exp27_full_dense5_combined_20260430_2050/analysis/` |
| Identity/margin handoff | `scripts/analysis/build_exp20_exp21_handoff_synthesis.py` | `results/paper_synthesis/exp20_exp21_handoff_table.csv` |
| Terminal-depth audit | `scripts/analysis/analyze_exp31_terminal_depth_factorial.py` | `results/exp31_terminal_depth_factorial/exp31_terminal_depth_full_a100x4_localdisk_fixedsched_20260502_021238/analysis/` |
| Terminal MLP audit | `scripts/analysis/analyze_exp33_terminal_identity_margin.py` | `results/exp33_terminal_identity_margin/exp33_terminal_identity_margin_full_dense5_a100x8_overlap_20260502_0509/analysis/` |
| OLMo fixed-support stage case study | `scripts/analysis/analyze_exp35_olmo_base_anchored_stage_decomposition.py`; `scripts/analysis/build_exp35_stage_ratio_bootstrap.py` | `results/exp35_olmo_base_anchored_stage_decomposition/exp35_full_olmo_stage_8a100_20260502_2300/analysis/` |

All full reruns use bf16 inference and deterministic greedy decoding unless a script states otherwise. The summary audit is CPU-only and reads committed JSON/CSV artifacts. Reproducing raw 4B-8B intervention records requires multiple 80GB A100/H100 jobs; reproducing Qwen2.5 32B additionally requires the Exp24 multi-GPU run or the committed paper-facing synthesis artifacts.
