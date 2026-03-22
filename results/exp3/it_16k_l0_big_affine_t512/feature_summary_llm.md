# Feature Group Summary — IT corrective layers 20–33

_Claude Opus 4.6 extended thinking via OpenRouter. Top 100 English features._



# Thematic Analysis of Gemma 3 4B IT Corrective Stage Features (L20–L33)

## Group 1: Token/Subword Completion and Morphological Repair
**Features:** #7 (L33:F443), #9 (L33:F361), #15 (L33:F83), #25 (L33:F456), #33 (L33:F113), #53 (L33:F621), #77 (L33:F360), #81 (L32:F424), #92 (L33:F417)

These features boost common subword fragments, single characters, and token-initial syllables (e.g., `Sp`, `Sch`, `Re`, `abil`, `sen`, `ir`, `ap`). They are overwhelmingly concentrated in L33 and L32—the final two layers. This represents last-mile lexical selection: the corrective stage is resolving partially-determined token distributions into specific subword completions. The high activation counts (60K–90K) suggest these fire broadly across generation, functioning as a continuous token-shaping mechanism rather than being content-specific.

## Group 2: Numerical and Mathematical Reasoning
**Features:** #14 (L33:F500), #21 (L33:F140), #35 (L32:F503), #42 (L33:F815), #46 (L33:F494), #48 (L33:F490), #50 (L33:F579), #52 (L32:F315), #87 (L31:F199), #94 (L32:F544)

These features boost digits (`1`, `2`, `3`, `5`), LaTeX formatting tokens (`mathbf`, `textit`, `\,`, `{`), number words (`five`, `six`, `Fifth`), and mathematical delimiters (`$|\`, `(=`). Feature #48 (L33:F490) is notable for extremely high mean activation (5858) with very low count (5789), indicating it fires strongly but selectively during numerical output. Feature #14 (L33:F500) similarly boosts `tex` and `{` with high per-activation intensity. These span L31–L33, consistent with late-stage numerical precision correction. They likely intervene to sharpen digit selection and ensure LaTeX formatting coherence during math problem solving.

## Group 3: Multilingual / Cross-lingual Processing
**Features:** #5 (L31:F332), #11 (L32:F41), #12 (L33:F155), #13 (L33:F59), #18 (L28:F239), #20 (L31:F536), #89 (L32:F356), #97 (L30:F474)

These features boost tokens in diverse non-English scripts and languages: Korean (가능), Bengali (রংপুর), Tamil (ண), Kannada (ಾರದ), Hindi (दिली), German (`ihrer`, `wenn`, `groß`), Hungarian (`ezek`), Turkish (`ınıza`), Spanish (`usuarios`), and various morphological suffixes across Romance/Slavic languages. The layer distribution is notably broader (L28–L33) compared to other groups, suggesting multilingual processing requires earlier intervention. These features likely handle language-identification-dependent token selection—ensuring that once a response language is determined, morphologically appropriate continuations are boosted.

## Group 4: Formatting, Punctuation, and Structural Tokens
**Features:** #3 (L33:F1039), #10 (L33:F1443), #27 (L33:F523), #34 (L33:F1456), #36 (L33:F11198), #57 (L33:F13011), #58 (L33:F2492), #63 (L33:F602), #75 (L33:F374), #85 (L28:F769), #98 (L27:F406)

This group boosts discourse-structural tokens: sentence-ending punctuation (`.`, `.'`, `."`, `).)`, `'.`), continuation connectives (`and`, `the`, `on`), sequence terminators (`<eos>`, `<start_of_image>`), and HTML/Markdown formatting (`<blockquote>`, `<h1>`, `---`). Feature #3 (L33:F1039) is the 3rd most important overall and fires 162K times—nearly every generation step—boosting `The`, `and`, `the`. Feature #98 (L27:F406) at L27 handles structural markup earlier in the corrective stack. This group manages discourse coherence: determining when to continue, when to terminate, and what structural framing to apply.

## Group 5: Specialized Domain Vocabulary (Science/Medical/Technical)
**Features:** #1 (L33:F394), #2 (L33:F244), #17 (L30:F60), #19 (L32:F439), #22 (L33:F1196), #44 (L33:F2799), #56 (L33:F1674), #66 (L33:F156), #67 (L32:F295), #88 (L31:F409)

These features boost highly specialized vocabulary: physics/engineering (`Rigidbody`, `deformation`, `bialgebras`), medical (`coronary`, `tachycardia`, `Cardiac`, `fibrosis`), chemistry (`dichloromethane`, `aldehyde`, `methylation`), biology (`epitopes`, `mtDNA`, `allele`), and mathematics (`nilpotent`, `interpolation`, `reformulation`). Feature #1 is the single most important feature in the entire corrective stack. Many fire with moderate mean but very high count (100K+), suggesting they serve as domain-gating features that suppress or boost technical vocabulary based on context. The spread across L30–L33 indicates domain-specific vocabulary selection begins in mid-corrective layers and sharpens through to the final layer.

## Group 6: Code Generation and Programming Tokens
**Features:** #27 (L33:F523), #51 (L33:F489), #73 (L32:F446), #96 (L32:F134), #30 (L32:F146)

These features boost programming constructs: `import`, `declare`, `Error`, `Redirect`, `Enc`, `);`, `}.`, `Write`, `SuspendLayout`, `currentGame`, `page`, `else`. They cluster in L32–L33. They likely handle code-syntax-specific token selection—ensuring syntactic validity of generated code by boosting structurally appropriate continuations (closing brackets after expressions, keywords after declarations, etc.).

## Group 7: Semantic Content / Explanatory Reasoning
**Features:** #37 (L33:F200), #45 (L33:F594), #55 (L30:F557), #62 (L27:F266), #69 (L24:F100), #79 (L33:F12414), #80 (L33:F496)

These features boost semantically rich common English words used in explanations: `why`, `signals`, `ingredients`, `natural`, `most`, `things`, `but`, `different`, `since`, `raised`, `future`, `recognition`, `management`, `is`, `can`, `implementation`, `example`, `affordability`, `allegations`. Feature #69 at L24 is one of the earliest in this list, suggesting that high-level semantic/conceptual token selection begins early in the corrective stack. The later features (#80 at L33) refine this into specific explanatory vocabulary. This group likely supports the model's ability to produce coherent factual explanations and reasoning chains.

## Anomalous / Hard-to-Categorize Features

- **#8 (L33:F442)**: Boosts `.-`, `he`, `lik`, `<unused61>` with extremely high mean activation (9225) but very low count (7631). This is the spikiest feature in the top 100—it fires rarely but enormously. The `<unused61>` token suggests this may be a residual artifact from tokenizer alignment or a safety-related suppression signal for specific token sequences.

- **#4 (L33:F498)** and **#6 (L31:F376)**: Feature #4 boosts `groß`, `trastornos`, `gimmick`, `alimentation`—a bizarre mix of German, Spanish, and English. Feature #6 boosts `masc`, `IRON`, `inconspicuous`, `Iron`. These have high mean activations and don't fit cleanly into any category. They may represent polysemantic features encoding multiple unrelated contexts that share a latent statistical pattern, or language-switching correction signals.

- **#23 (L28:F300)**: Boosts zero-width characters (`﻿`, `‬`), `prostitutes`, and `redistribute`. The zero-width characters suggest formatting cleanup, but `prostitutes` hints at a safety-adjacent feature that may be involved in sensitive content handling at L28—potentially an early warning/gating signal.

- **#31 (L31:F295)**: Boosts only `` (replacement character) and `feira`. Likely a corruption/encoding-error handler that activates when the model encounters or needs to output malformed Unicode.

- **#75 (L33:F374)**: Boosts `'`, `'`, `similarities`, `super` with the second-highest mean activation (5047) and very low count (5468). Possibly handles quotation/apostrophe disambiguation in very specific syntactic contexts.

## Conclusion

The corrective stage of Gemma 3 4B IT is primarily performing **three tightly coupled functions**: (1) fine-grained lexical/subword selection concentrated in L32–L33, where partially-determined token distributions are sharpened into specific outputs; (2) domain and language gating across L28–L33, ensuring that specialized vocabulary (medical, mathematical, multilingual) is appropriately boosted or suppressed based on conversational context; and (3) structural/discourse management, controlling punctuation, formatting, and sequence termination. The overwhelming concentration of high-importance features in L33 (50+ of the top 100) indicates that the final layer acts as a massive correction/refinement bottleneck. Notably absent from the top features are clear refusal/safety signals—suggesting either that safety behavior is encoded more diffusely, operates through suppression rather than boosting, or is primarily handled in earlier (pre-corrective) layers of this model variant.