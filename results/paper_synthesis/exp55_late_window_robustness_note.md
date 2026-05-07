# Exp55 Late-Window Robustness

Models analyzed: `gemma3_4b, qwen3_4b, llama31_8b, mistral_7b, olmo2_7b`.

Dense5 model-mean final-20% KL effects:

- `prelate_half`: graft `-0.001`, swap `-0.384`.
- `late_full`: graft `+0.070`, swap `-0.625`.
- `late_front_half`: graft `+0.008`, swap `-0.352`.
- `late_center_half`: graft `+0.022`, swap `-0.352`.
- `late_terminal_half`: graft `+0.050`, swap `-0.443`.
- `terminal_quarter`: graft `+0.033`, swap `-0.347`.

Observed interpretation:

- Late-full final-20% signs are in the expected direction: graft `+0.070 [-0.048, +0.189]`, swap `-0.625 [-1.076, -0.201]`.
- The direct edited-window late-full graft is clearer: `+0.365 [+0.108, +0.629]`.
- The pre-late swap also moves final-20% KL: `-0.384 [-0.601, -0.180]`.
- Use this audit as support for the late window being the strongest tested bidirectional handle, not as evidence for a sharp late-only boundary.
