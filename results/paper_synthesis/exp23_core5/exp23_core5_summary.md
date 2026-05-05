# Exp23 Core-5 Synthesis

Core-5 combines the four smaller dense support families with Qwen2.5-32B on the same holdout support.
CIs use an independent normal approximation to stored per-family prompt-bootstrap intervals.

- Late IT from PT upstream: `+0.759`.
- Late IT from IT upstream: `+2.439`.
- Upstream x late interaction: `+1.680`.

## Position Rows

| Stratum | Core-5 interaction |
|---|---:|
| all positions | `+1.680` `[+1.603, +1.757]` |
| positions >=1 | `+1.096` `[+1.014, +1.179]` |
| positions >=3 | `+0.800` `[+0.716, +0.885]` |
| position >=5 | `+0.829` `[+0.721, +0.938]` |
| position >=10 | `+1.073` `[+0.841, +1.304]` |
