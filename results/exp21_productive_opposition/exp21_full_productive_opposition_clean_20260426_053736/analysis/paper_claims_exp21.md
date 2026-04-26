# Exp21 Productive Opposition: Paper-Safe Claims

Primary scope: dense5 pooled native `first_diff`; DeepSeek is reported separately as MoE.

Key dense5 native values:

- Pure IT late `delta_cosine_mlp`: `0.0304`.
- Pure IT late IT-minus-PT margin write-in: `0.7680`.
- Pure IT late pipeline-token-vs-alt margin write-in: `1.2641`.
- Pure IT late negative-parallel contribution to IT-minus-PT margin: `-0.0046`.
- 2x2 late-weight effect on IT-minus-PT margin write-in: `0.1478`.
- 2x2 upstream-context effect on IT-minus-PT margin write-in: `0.4027`.
- 2x2 late interaction on IT-minus-PT margin write-in: `0.2885`.
- PT host, adding IT late: `0.0035`.
- IT host, removing IT late: `0.2920`.
- PT host, adding IT late on top of IT mid: `0.0119`.
- IT host, removing PT mid+late vs PT mid only comparison: `-0.2176`.

Interpretation rule:

- The negative-parallel component is not positive on the pure-IT IT-vs-PT margin proxy, so the paper should not claim that raw negative opposition itself is the main mechanism.
- If the late-weight effect is positive, the paper may say that late IT weights add IT-token readout on this proxy.
- If the upstream effect is also large, the paper must phrase the mechanism as cooperation between earlier IT context and late IT readout, not as a late-module-only story.
- If raw `delta_cosine_mlp` is weaker than token-specific write-in metrics, the paper should treat negative residual opposition as a geometric companion rather than the mechanism itself.

Safe wording:

> Exp21 measures MLP-only finite-difference logit effects at the first PT/IT divergent prefix. In native dense-family runs, late residual opposition is evaluated by whether its negative-parallel component increases IT-vs-PT token margin. This supports a productive-opposition readout only when the negative component helps the IT token relative to the PT token; otherwise, negative delta-cosine remains only a geometric signature.

Do not claim:

> Negative residuals cause assistant behavior.
