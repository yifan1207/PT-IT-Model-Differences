# Exp15 Gemma Behavioral Case Study

Selection rule: Gemma is used as an illustrative case study because it has the largest bidirectional late-window convergence intervention effects in the dense-family late-window table.

Primary behavioral readout: compare native Gemma IT (`C_it_chat`) to the same
IT host with a late PT MLP swap (`D_late_ptswap`) under free-running 512-token
generation over the Exp15 600-prompt support.

- Pairwise assistant-register (`G2`) preference for native IT: `0.947`
  with CI `[0.916, 0.973]`.
- Pairwise safety/format (`S2`) preference for native IT: `0.933`
  with CI `[0.867, 0.987]`.
- Paired pointwise assistant-register drop under the late PT swap:
  `0.804` with CI `[0.649, 0.960]`.

This is an illustrative behavioral case study, not a cross-family behavioral
claim. It uses the same matched-prefix intervention family as the convergence
paper's MLP leverage experiments.
