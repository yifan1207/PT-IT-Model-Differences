Revised Hypothesis:
NH1: IT's maintained feature activation across generation reflects "output planning," “semantic/strcutcal format enforcement,” and "consistency maintenance (on topic…)”
Format enforcement. Every generated token needs to fit a conversational format — appropriate register, coherent paragraph structure, no truncation. This requires active suppression of tokens that would break the format, which requires features to stay active.
Output planning. The Anthropic Biology paper showed that Claude plans ahead (activating rhyming-word features before writing intermediate words). IT might maintain feature activity because the model is continuously planning the structure of its response, not just predicting the next token.
Consistency maintenance. IT models need to stay on-topic, maintain factual consistency with earlier tokens, and avoid contradicting themselves. This requires the late layers to continuously "check" the generation against the accumulated context — an operation that doesn't exist in PT because PT doesn't have a concept of "response quality."
The Anthropic Biology paper showed that Claude plans ahead — it activates "rhyming word" features before writing the intermediate words of a poem. IT models might maintain high feature activation because they're planning the full response structure, not just generating one token at a time. PT lacks this planning because it was never trained on structured responses. 
Test: check whether IT's late-generation features are "forward-looking" (their logit effects promote tokens that appear later in the generation, not the immediate next token).
NH2: The layer 10-11 transition corresponds to the detokenization/composition boundary, and IT widens this boundary.
What the Layer 11 Dip Tells You
The dip exists in both models but is qualitatively different:
In PT, layers 10-11 are the peak, and the decline afterward is smooth. There's no discontinuity — the model smoothly transitions from peak activity to declining activity. It looks like the model does most of its work in the middle and gradually winds down.
In IT, the dip at layer 11 is a genuine phase transition — a sharp drop of ~90 features (from 160 to 70) followed by an immediate recovery. This looks like an architectural boundary: everything before layer 11 is one computational phase (detokenization + early feature engineering), and everything after is a different phase (semantic processing + output shaping). Instruction tuning appears to have sharpened this boundary — made it more discrete rather than gradual.

NH3: Residual difference between PT and IT is due to alignment behavior
This connects directly to known findings about aligned models. The "Safety Layers in Aligned LLMs" paper (Li et al., ICLR 2025) found that alignment behavior is concentrated in specific later layers. What you're seeing IS the alignment machinery — late layers in IT contain features that actively suppress outputs the model "wants" to produce (based on the residual stream direction from middle layers) and steer toward outputs it "should" produce (based on instruction tuning).
Think about it concretely: the middle layers build up a representation that says "here's what comes next based on the pattern." The late layers in IT then say "but wait — that output needs to be helpful, harmless, formatted correctly, not truncated, not repetitive" and actively push AGAINST the raw prediction to enforce these constraints. PT doesn't have this corrective stage because it was never trained on what a "good" response looks like.
It's a structural property of every forward pass in the IT model due to consistency as generation steps increase.


For papers:
Direction A: Universal Pipeline (IC ≈ OOC ≈ R). The finding that task type doesn't change the computational pathway. This is what the 10 plots were designed for. Clean, testable, connects to Embers of Autoregression.
Primary: "Post-Training Creates a Corrective Computational Stage: How Instruction Tuning Restructures Layer-Wise Computation in LLMs"
Core claim: Instruction tuning doesn't just add behavioral constraints on top of a fixed computational pipeline. It structurally reorganizes the pipeline — concentrating raw prediction/composition in layers 0-20, sharpening the phase boundary at layer ~11, and creating a corrective stage in layers 20+ where the model actively opposes its own residual stream to enforce output quality. This corrective stage is universal (same magnitude and direction regardless of task type or generation step), which explains why IT models appear to process all inputs identically at the macro level.
Ambitious Idea A: The Correction Stage as an "Inner Alignment Tax"
If IT's late layers are always opposing the residual regardless of content, that's wasted computation on inputs that don't need correction. The model spends the same corrective effort on "What is 2+2?" as on a potentially harmful query. This is an alignment tax — the cost of making the model safe/helpful is that every single forward pass pays for correction that's often unnecessary. You could quantify this: how much of the model's total computation (by attribution magnitude) goes to the corrective stage? If it's 30-40% of total MLP contribution, that's a significant efficiency finding. Connects to early-exit literature (AdaInfer) — if you could skip the corrective stage for "safe" inputs, you'd get a major speedup.
IT models, however, are burdened with a "persona." -> the assistant spectrum and maybe this is a tax in itself or similar to this?
Ambitious Idea C: Can You Remove the Corrective Stage and Recover PT Behavior?
If you ablate (zero out) MLP outputs at layers 25-33 in IT, does the model's behavior become more PT-like? Does its generation quality degrade specifically in format/safety while keeping semantic content intact? If you can surgically remove the corrective stage and show it separates from semantic computation, that's a clean mechanistic demonstration of what post-training adds. This would be the strongest version of the paper — not just observing the correction, but demonstrating it's modular and removable.


Tier 1: Must Read and Cite — Directly Related
Paper
Year
Venue
Connection
URIAL (Lin et al.)
2024
ICLR
77.7% of IT tokens unchanged from PT. Your correction stage explains the other 22.3%.
Safety Tax (Huang et al.)
2025
arXiv
Coined "Safety Tax" for reasoning degradation from alignment. Your work provides the mechanism.
Mitigating Alignment Tax of RLHF (Luo et al.)
2024
EMNLP
Weight averaging reduces tax. You explain WHY — it dilutes the corrective stage weights.
Refusal Mediated by Single Direction (Arditi et al.)
2024
NeurIPS
Refusal direction is one component of your broader corrective subspace.
Safety Layers in Aligned LLMs (Li et al.)
2025
ICLR
Safety in specific late layers. You generalize: ALL output correction lives there.
LARF: Layer-Aware Representation Filtering
2025
EMNLP
Identifies safety-sensitive layers via scaling. Your cosine metric is a complementary method.
DoLA: Decoding by Contrasting Layers (Chuang et al.)
2024
ICLR
Works by contrasting early/late layers. Your finding explains WHY it works.
FAST: SAEs for Instruct Models (Li et al.)
2025
arXiv
Base SAEs fail on IT. Critical methodological check for your transcoder validity.
Crosscoder Model Diffing (Lindsey et al.)
2024+2025
Anthropic
Shared/exclusive features between PT and IT. Your dynamics complement their feature analysis.
Stages of Inference (Lad, Gurnee & Tegmark)
2024
arXiv
Four stages for PT. You add/modify the framework for IT: five stages or qualitatively different sharpening.

Tier 2: Important Context
Paper
Year
Connection
SFT Memorizes, RL Generalizes (Chu et al.)
ICML 2025
Disentangle which post-training method creates the correction stage
Embers of Autoregression (McCoy et al.)
PNAS 2023 + 2024
Task-type universality might be a correction-stage artifact, not a base-model property
Kim, Yoo & Oh
ICML 2025
Layer-wise dynamics aligned for certain/uncertain — check if this was tested on IT only
Transformer Dynamics (Guitchounts et al.)
2025
Residual stream as dynamical system; your correction stage creates a repellor
Demystifying Layer Roles
2025
Different tasks need different layers under pruning; connects to where correction matters
Base Models Know How to Reason
ICLR 2026
Reasoning exists in PT; correction stage might suppress as much as it formats
Anthropic Biology (Lindsey et al.)
2025
Circuit tracing shows task-specific circuits; your correction stage might mask these
Temporal SAEs
2024
SAE bias toward syntactic features; relevant limitation of your transcoder measurements

Tier 3: For the Alignment Tax Angle
Paper
Year
Connection
AdaInfer: Not All Layers Necessary
IJCAI 2025
Early exit for efficient inference; your correction stage defines what to skip
Light Alignment via Single Neuron
2026
Lightweight alternatives to full correction; your work shows why they might suffice
Echo Chamber: RL Amplifies Pretraining
2025
Correction might amplify a latent PT signal; check if PT has weak negative cosine
Representation Engineering (Zou et al.)
2023
Control vectors might work by modulating the correction stage
Behavior Shift after Instruction Tuning
NAACL 2024
Systematic PT vs IT representation comparison

Tier 4: Methods You'll Need
Paper
Year
Connection
Logit Lens + Tuned Lens
2020/2023
Project residual before/after correction into vocab space
Sparse Feature Circuits (Marks et al.)
ICLR 2025
Causal feature identification methodology
RouteSAE
EMNLP 2025
Multi-layer feature extraction; could capture cross-layer correction features


More experiments to run:
Stratify by token type at different layers for the layers’s contribution to the type of tokens* important
 Split generated tokens into factual/content, function, punctuation, formatting, role/style, and refusal/safety-like categories. My guess is the PT–IT late-layer gap will be much larger on style/format/control tokens than on core content tokens. That would strongly support the “governance” interpretation. This prediction is motivated by behavior-shift, URIAL, length-control, and chat-tuning crosscoder work.
The token categorization is hard to do cleanly. Instead of trying to classify all tokens, use a simpler split: (a) the first content-bearing token of each response (usually the "answer" token — this is mostly semantic), (b) punctuation and whitespace tokens, (c) discourse markers and filler ("Well," "Let me," "I think"), (d) the EOS/turn tokens. Comparing the corrective stage magnitude across just these four is cleaner than trying to classify everything. You can classify (c) and (d) using a simple token-list lookup rather than needing an LLM judge.

Measure answer emergence vs answer stabilization with Tuned Lens. * interesting
 For each generated token, record:
earliest layer where the final token enters top-k,


earliest layer where it becomes top-1,


earliest layer where the layerwise distribution stays close to the final one.
 If PT reaches emergence early but IT keeps changing after emergence, that is a very clean result. Tuned Lens is the right tool for this.
Do PT↔IT causal patching by depth band. Layers swapping swap the later layers
 Swap only mid layers, then only late layers, between matched PT and IT runs on the same prompt. The key test:
if mid-layer swaps move correctness/content,


and late-layer swaps move style/format/compliance,
 then your story becomes causal, not just descriptive.
Quantify Entropy better
plain entropy only tells you how spread out one distribution is at one place
 but your question is really about how the model’s belief evolves across layers.
So for one generated token, instead of only asking:
“How uncertain is the final prediction?”
you also want to ask:
“When did the final output candidate appear?”
 “Did it gradually strengthen or suddenly pop out?”
 “Were layers pruning bad options, or still exploring?”
 “Did PT and IT differ in trajectory, even if final entropy was similar?”
Top-1 vs top-2 margin for the final output logits
Layer-to-final KL. For each layer lll, compare its decoded distribution to the final distribution.

Cross-layer entropy over candidate tokens
This is not just entropy at one layer.
 It is about how a specific candidate token behaves across layers.
Take one candidate token vvv.
 Look at its probability across layers:

Quantify attraction and repulsion separately. For later layers in IT see which tokens they are pushing for are they already in the TopK or not, do they have similar token cos similarity with topk? Like what are tokens specifically
 For each layer, decompose contribution into:
push toward final correct token,


push away from strongest incorrect alternatives.
 Your negative-direction idea becomes much stronger if you can show IT late layers carry more repulsive mass than PT.
Increase dataset size and also dataset diversity in prompt format
Control prompt format aggressively.
 This is important. Compare:
PT in pure completion format,


PT in chat-like format,


IT in default chat format,


IT with stripped-down completion framing where possible.
New Experiment A: Per-Layer Weight Norm Comparison PT vs IT
Before doing any activation-level analysis, just compare ||W_MLP[layer]||_F for PT vs IT at each layer. If IT's late-layer weights have larger norms, the model has LEARNED to make bigger edits there during post-training. If weight norms are similar but activation patterns differ, the correction emerges from how the model USES existing weights differently. This takes 10 minutes and provides important context.

Next Steps (Revised Priority)
Step 1 (CRITICAL): Reconstruction error check. Compute transcoder MSE per layer for PT vs IT. If IT late-layer error is 2×+ higher, Finding 1 (L0 increase) is partially artifactual. Finding 2 (cosine) is safe regardless since it uses raw residual streams. 1 day.
Step 2: Logit lens before/after correction. At layers 15, 20, 25, 30, 33 — project residual into vocab space, compare predicted tokens. Does the correction stage change WHAT the model predicts, or just the confidence? Does the pre-correction prediction look like PT's output? 1-2 days.
Step 3: Quantify the alignment tax. Sum total |delta_norm| across layers 0-20 (propose) and 20-33 (correct). What fraction of total MLP contribution comes from correction? Compare PT vs IT. 1 day.
Step 4: Ablate the corrective layers. Zero MLP outputs at layers 25-33 in IT, generate on your full prompt set. Does output become PT-like? Does semantic content survive? 1-2 days.
Step 5: Layers 0-20 only IC/OOC/R comparison. Test whether task-type universality holds pre-correction or is a correction-stage artifact. 1 day.
Step 6: Replicate on a second model. Llama 3.1 8B base vs instruct if transcoders available. At minimum, check the raw cosine metric (no transcoder needed) on another model family. 2-3 days.


#
Experiment
Days
Why
0a
Transcoder reconstruction error PT vs IT per layer
0.5
If IT late-layer error is 2×+ higher, L0 finding is partially artifactual. Must check before publishing.
0b
Run IT on raw completion prompts (no chat template)
1
Tests whether corrective stage is weight-driven or format-triggered. Critical confound control.
0c
Per-layer weight norm comparison PT vs IT
0.1
Quick sanity check: did post-training change late-layer weight magnitudes?

Phase 1: Core Mechanistic Evidence
#
Experiment
Days
Why
1a
Answer emergence/stabilization with logit lens PT vs IT
1-2
Track rank and probability of final token across layers. If PT commits early and IT keeps modifying, that's the cleanest corrective stage evidence.
1b
Attraction vs repulsion decomposition per layer
1
Decompose each layer's logit contribution into push-toward-correct vs push-away-from-incorrect. Tests if correction is suppressive.
1c
Layers 0-20 only IC/OOC/R comparison
1
Tests if task-type universality is real or a correction-stage artifact. Redo Jaccard and L0 for pre-correction layers only.

Phase 2: Deepening the Corrective Stage Finding
#
Experiment
Days
Why
2a
Token type stratification
1-2
Split generated tokens by type (content, function, punctuation, discourse markers). Check if correction magnitude differs by token type.
2b
Quantify the alignment tax
0.5
Sum total
2c
Layer-to-final KL divergence trajectory PT vs IT
1
When does each model "commit"? Clean metric that subsumes several of your entropy ideas.
2d
Generation-step resolved corrective stage
1
Per-token cosine heatmap (layers × generation position) for IT. Is correction constant or does it vary across generation?

Phase 3: Causal Validation
#
Experiment
Days
Why
3a
Ablate corrective layers in IT
1
Zero out MLP outputs at layers 25-33, generate. Does output become PT-like? Does content survive while format breaks?
3b
Activation patching: PT residual → IT late layers
1-2
Feed PT's layer-20 residual into IT's layers 20-33. Does the correction stage "fix" PT's raw output into IT format?

Phase 4: Generalization and Scale
#
Experiment
Days
Why
4a
Replicate cosine finding on second model family
2-3
Raw residual cosine (no transcoder needed) on Llama or Qwen PT vs IT. If sign-flip replicates, it's universal.
4b
Increase dataset size and diversity
1-2
More prompts, especially matched pairs across categories. Strengthens all statistical tests.
4c
PT vs IT benchmark accuracy on your prompt set
0.5
Need to know: does IT actually perform better on reasoning, or just format better? Contextualizes all findings.


