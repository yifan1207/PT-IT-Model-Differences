# IT vs PT Feature Contrast — corrective layers 20–33

_Claude Opus 4.6 via OpenRouter. Top 100 English features per variant._



# Comparative Analysis: IT vs PT Corrective-Stage Transcoder Features (Gemma 3 4B, Layers 20–33)

---

## 1. Thematic Groups

### Group A: "Generic Function Words / Discourse Connectives"
**IT:** #3 (L33:F1039 – `The, and, the`), #9 (L33:F361 – `of, the, I, i`), #36 (L33:F11198 – `on, and`), #45 (L33:F594 – `other, natural, most, things`), #80 (L33:F496 – `is, can, implementation, example`)
**PT:** #25 (L33:F12638 – `of, to, if, the`), #31 (L33:F330 – `moreover, then, all, even`), #59 (L32:F8760 – `and, just, the`)

**Shared, but structurally different.** In IT, function-word features dominate the top 10 (rank 3, 9) and collectively carry enormous activation mass. In PT, the equivalent features are mid-ranked (25, 31, 59) with proportionally lower influence relative to the PT total budget. Both are overwhelmingly L32–33.

These features represent the final-layer default/fallback token distribution — correcting logits toward high-frequency tokens when no specialized content word is warranted. IT's higher reliance on these suggests the aligned model more aggressively steers toward safe, generic continuations as a baseline, consistent with RLHF encouraging hedged, fluent outputs.

---

### Group B: "Numeric / Mathematical Token Production"
**IT:** #21 (L33:F140 – `1, all, ALL`), #42 (L33:F815 – `1, 5, balls, ts`), #46 (L33:F494 – `six, Five, Fifth, five`), #48 (L33:F490 – `2, 1, 3`), #87 (L31:F199 – `2, 1`)
**PT:** #83 (L33:F490 – `leg, anthrop, aughter, lanc`) — same feature index, entirely different boosted tokens

**IT-dominant.** The IT model has at least 5 high-ranked features dedicated to numeric token production, spanning both digit tokens and number words. Feature L33:F490 is particularly telling: in IT it fires rarely (5,789 steps) but with extreme mean activation (5,858), functioning as a high-confidence numeric correction spike. In PT, the same feature index boosts unrelated morphological fragments.

These represent the corrective machinery for structured mathematical reasoning — boosting numeric answers at the output layer. RLHF/instruction-tuning on math prompts has carved out dedicated high-precision numeric features that the base model entirely lacks. The concentration in L33 (final layer) and L31 indicates last-moment answer correction.

---

### Group C: "LaTeX / Formal Notation Formatting"
**IT:** #14 (L33:F500 – `t, {, tex`), #35 (L32:F503 – `mathbf, \,`), #50 (L33:F579 – `(=, (,`), #94 (L32:F544 – `{, textit, {"`), #27 (L33:F523 – `-, Write, }., );`)
**PT:** #43 (L33:F1162 – `{, lishes`), #33 (L24:F1148 – `(_), {}`)

**IT-dominant.** IT has a rich constellation of features handling LaTeX delimiters, math-mode commands (`\mathbf`, `\textit`), and structured bracket production. PT has only weak, diffuse bracket features at much lower ranks and earlier layers.

This group implements the formatting conventions expected in instruction-tuned math responses — wrapping answers in `$...$`, producing well-formed LaTeX. These are learned formatting conventions, not mathematical reasoning per se, and their corrective-stage placement (L32–33) suggests they act as final output formatting corrections. This is a clear RLHF artifact.

---

### Group D: "EOS / Sequence Termination / Turn Boundary"
**IT:** #10 (L33:F1443 – `., <start_of_image>`), #58 (L33:F2492 – `<start_of_image>, <eos>, ,`), #85 (L28:F769 – `<eos>, said`)
**PT:** #12 (L31:F2053 – `</h1>, <\/>, </th>, <start_of_image>`), #52 (L28:F14248 – `<start_of_image>, </strong>, </b>`), #74 (L28:F769 – `<eos>, <unused61>, [`)

**Shared, with different character.** Both models have EOS/boundary features, but they serve different purposes. IT's versions (ranks 10, 58) are high-ranked and focused on clean sentence termination and turn-ending — the machinery for knowing when to stop generating. PT's versions (ranks 12, 52) are entangled with HTML tag closing, reflecting the web-scraped pretraining distribution.

IT's EOS features being highly ranked in the corrective stage reflects RLHF training explicit turn-taking discipline. The shared L28:F769 feature exists in both but boosts `<eos>` more cleanly in IT versus being mixed with `<unused61>` and bracket tokens in PT.

---

### Group E: "Multilingual / Cross-lingual Morphological Correction"
**IT:** #12 (L33:F155 – `ihrer, ezek, unseres` — German/Hungarian), #13 (L33:F59 – `wenn, tiap` — German/Malay), #18 (L28:F239 – `үшін, militare` — Kazakh/Italian), #89 (L32:F356 – `usuario, usuarios` — Spanish)
**PT:** #4 (L21:F1333 – `intervalo, Giappone, solução` — Spanish/Italian/Portuguese), #22 (L31:F5300 – `Nella, Dieses, และความ` — Italian/German/Thai), #89 (L24:F8878 – `Dengan, Seperti` — Malay), #73 (L26:F15749 – `russia, october, friday, american` — lowercase English)

**Shared, with dramatic layer shift.** PT's multilingual features cluster in early corrective layers (L21–L26), suggesting the base model handles cross-lingual routing as an early corrective computation. IT's multilingual features have migrated to L28–L33 — the latest corrective layers. PT also has more features in this group and at higher ranks, consistent with the base model devoting more capacity to the multilingual web text distribution.

These features correct the output vocabulary distribution to produce morphologically appropriate tokens in the active language. The layer migration from PT→IT likely reflects RLHF restructuring: alignment compresses multilingual handling later because earlier layers are repurposed for instruction-following.

---

### Group F: "HTML / Markup Structure"
**IT:** #98 (L27:F406 – `<blockquote>, <h1>, <h3>, ---`) — single feature, rank 98
**PT:** #10 (L24:F2930 – `<strong>, Yeah, {{, Yes`), #13 (L28:F3914 – `</strong>, Read, </u>`), #41 (L22:F588 – `<em>, amp, charset`), #42 (L28:F12593 – `[…], </blockquote>, 阅读全文`), #52 (L28:F14248 – `<start_of_image>, </strong>, </b>`)

**PT-dominant.** The base model dedicates substantial corrective capacity to HTML tag production, reflecting its web-crawl pretraining distribution. IT has essentially purged these features — only a single low-ranked HTML feature survives. This is one of the cleanest IT-vs-PT differences in the data.

The base model needs to predict HTML structure because its training data is full of it. RLHF eliminates this because instruction-tuned outputs are never HTML. The corrective capacity freed up by removing HTML features is presumably reallocated to the math, formatting, and safety-adjacent features that appear in IT.

---

### Group G: "Specialized Domain Vocabulary (Medical/Scientific)"
**IT:** #19 (L32:F439 – `coronary, tachycardia`), #66 (L33:F156 – `Cardiac, Podcasts, Modelling`), #44 (L33:F2799 – `sustainability, fibrosis`), #67 (L32:F295 – `aldehyde, synchrotron`)
**PT:** #5 (L30:F60 – `methylation, ccnc, SCPC`), #28 (L30:F523 – `acycline, Quincy`)

**Shared, but IT has more and later.** Both models represent biomedical/scientific terminology, but IT has expanded this capacity and pushed it into L32–33 (latest corrective layers), while PT keeps it at L30. IT rank 19's `coronary, tachycardia` is a clean medical feature with high count (119,024 firings), suggesting it's active across many generation steps as a domain-vocabulary correction.

These features correct the output distribution toward domain-specific terminology when medical/scientific context is detected. IT's expansion here likely reflects instruction-tuning on factual/knowledge QA where precise technical vocabulary matters.

---

### Group H: "Capitalization / Proper Noun Initials"
**IT:** #15 (L33:F83 – `Tex, Mar, Ni, Tem`), #33 (L33:F113 – `Sp, Sil, Sch, Re`), #53 (L33:F621 – `R, Tin, E, Ku`), #70 (L32:F288 – `Más, A, Bre, Grim`)
**PT:** #21 (L30:F74 – `An, ins, As, Th`), #91 (L27:F56 – `THE, die, thé`)

**IT-dominant.** IT has multiple high-ranked features (all L32–33) that boost capitalized token prefixes — the initial fragments of proper nouns and sentence starters. PT has weaker, earlier-layer versions.

These represent corrective capitalization machinery: ensuring the first token of entities and sentences is properly cased. IT's heavy investment here reflects the aligned model's stronger commitment to well-formed text presentation in its outputs.

---

## 2. Most Striking IT vs PT Differences

### IT has / PT lacks:
1. **Numeric answer production** (Group B): Five dedicated features in IT's top 50; effectively absent in PT. This is the most functionally significant difference — IT has built dedicated corrective circuitry for producing numerical answers to math problems.
2. **LaTeX formatting** (Group C): Rich constellation in IT, nearly absent in PT. Pure RLHF artifact from training on math instruction data.
3. **High-confidence sparse firing patterns**: IT features like L33:F490 (mean=5,858, count=5,789) and L33:F374 (mean=5,047, count=5,468) fire very rarely but with enormous magnitude. PT has no features with comparably extreme mean-to-count ratios in the top 100. This suggests RLHF creates "sniper" features — high-precision corrections that fire only when the model is confident about a specific intervention.

### PT has / IT lacks:
1. **HTML/markup structure** (Group F): 5+ features in PT's top 50; single weak feature in IT. Direct reflection of pretraining distribution being overwritten.
2. **Early-layer broad-coverage features**: PT has 10 features from L20–L22 in its top 100; IT has only 1 (L22:F305, rank 71). PT's corrective processing starts earlier and distributes more broadly across layers.
3. **Massive activation volumes**: PT's top feature (L33:F361) has total=646M, 6× larger than IT's top feature (total=111M). PT's top 100 sums are roughly 5–6× larger than IT's. The base model relies more heavily on corrective-layer features overall, while IT has presumably shifted more computation to earlier (non-corrective) layers via alignment.
4. **`<unused>` token features**: PT rank 57 (L33:F305) boosts `<unused1689>, <unused1837>` etc. with very high mean activation (4,213). These are artifacts of the base model's tokenizer/vocabulary that RLHF has suppressed.

---

## 3. Anomalous / Hard-to-Categorize Features

- **IT #8 (L33:F442)**: `he, lik, <unused61>` — Fires only 7,631 times but with the highest mean activation in the IT set (9,225). Extremely sparse, high-magnitude corrective spike. The token mix is incoherent; this may be a "junk" or residual feature, or it could be a safety-related suppression feature whose boosted tokens are misleading (the important computation may be what it *suppresses*, not what it boosts).

- **IT #1 (L33:F394)**: `oper, Rigidbody, osc, ALS` — The single most influential IT feature boosts game-engine (Unity) and oscillation-related tokens. This is surprising as the top feature for an instruction-tuned model. It fires 88,797 times with high mean activation, suggesting it may be a broad "technical code completion" feature that has been co-opted or that remains from pretraining.

- **PT #7 (L32:F240)**: `MeToo, ;');, blkid` — Very high mean activation (4,076) with low count (62,335). The token mix spanning a social movement hashtag, JavaScript syntax, and a Linux utility is deeply incoherent. Likely a polysemantic "grab-bag" feature.

- **IT #5 (L31:F332)** vs **PT #56 (L31:F332)**: Same layer and feature index. IT boosts `가능, setOn, CTAssert` (Korean + code); PT boosts `care, </tr>, i`. The feature has been substantially repurposed by alignment — same circuit location, entirely different function.

- **PT #1 (L33:F361)** vs **IT #9 (L33:F361)**: Same index. PT's top feature (total=646M, boosting `incarn, karm, blackmail, blinking`) dwarfs IT's version (total=61M, boosting `of, the, I, i`). Alignment has dramatically suppressed this feature's activation magnitude and completely changed its output distribution from evocative/narrative vocabulary to generic function words. This single feature may represent one of the largest alignment-induced changes in the entire network.

---

## 4. Layer Distribution Summary

| Layer Range | IT count (top 100) | PT count (top 100) |
|---|---|---|
| L20–L22 | 1 | 8 |
| L23–L25 | 4 | 11 |
| L26–L28 | 7 | 10 |
| L29–L31 | 14 | 15 |
| L32–L33 | 74 | 56 |

IT is dramatically concentrated in L32–33 (74% of top features). PT distributes more evenly, with meaningful early-corrective (L20–25) contributions. RLHF appears to have consolidated corrective processing into the final two layers.

---

## 5. Conclusion

Comparing IT and PT corrective-stage features reveals that RLHF/instruction-tuning performs three major structural transformations on the model's late-layer computation. **First, it creates dedicated high-precision circuitry for task-relevant outputs** — particularly numeric answer tokens and LaTeX formatting — that is entirely absent in the base model, representing genuinely new computational capabilities rather than refinements of existing ones. **Second, it aggressively prunes pretraining-distribution artifacts** (HTML tags, `<unused>` tokens, broad multilingual early-layer routing), freeing corrective capacity for task-aligned features. **Third, it concentrates corrective computation into the final two layers** (L32–33 hold 74% of IT's top features vs. 56% for PT), suggesting alignment makes the model "more decided" by earlier layers and uses the corrective stage primarily for precise output formatting rather than broad distributional correction. The most mechanistically revealing finding is the emergence of ultra-sparse, ultra-high-magnitude "sniper" features in IT (e.g., L33:F490 with mean activation 5,858 across only 5,789 firings) — RLHF appears to train the corrective stage to make rare but extremely confident interventions on specific token choices, a pattern entirely absent from the base model's more diffuse corrective strategy.