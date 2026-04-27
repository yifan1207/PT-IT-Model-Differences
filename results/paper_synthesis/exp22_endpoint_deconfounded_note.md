# Exp22 Endpoint-Matched Convergence Gap

Dense-5 endpoint-control run over 600 prompts per PT/IT branch, raw and tuned probes, token steps after the first five generated tokens. Token steps are coarsened-exact-matched within `model x probe_family` on final-layer entropy, final top-1 confidence, and final top1-top2 margin.

Quality gates pass: `1172058` analyzed token-step/probe rows, maximum malformed branch rate `0.000`, minimum matched retention `0.796`, and maximum post-match endpoint-covariate SMD `0.057`.

Primary endpoint-matched late `KL(layer || own final)` remains higher for IT than PT:

- Raw probe: `0.425` nats, 95% CI `[0.356, 0.493]`.
- Tuned probe: `0.762` nats, 95% CI `[0.709, 0.814]`.

Endpoint-free checks point the same direction after the same endpoint matching:

- Remaining adjacent JS: `0.052`, 95% CI `[0.048, 0.057]`.
- Future top-1 flips: `0.203`, 95% CI `[0.190, 0.215]`.

Paper-use claim: the convergence gap is not explained away by final endpoint entropy, confidence, or top1-top2 margin. We still describe it as endpoint-relative, but now endpoint-matched under the matched-token estimator.
