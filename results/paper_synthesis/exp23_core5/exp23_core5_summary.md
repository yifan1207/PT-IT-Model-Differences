# Exp23 Core-5 Synthesis

Core-5 combines the four smaller dense support families with Qwen2.5-32B.
CIs use an independent normal approximation to stored per-family prompt-bootstrap intervals.

- Late IT from PT upstream: `+0.747`.
- Late IT from IT upstream: `+2.456`.
- Upstream x late interaction: `+1.709`.

## Position Rows

| Stratum | Core-5 interaction |
|---|---:|
| all positions | `+1.709` `[+1.637, +1.781]` |
| positions >=1 | `+1.159` `[+1.080, +1.237]` |
| positions >=3 | `+0.834` `[+0.755, +0.914]` |
| position >=5 | `+0.798` `[+0.701, +0.895]` |
| position >=10 | `+1.014` `[+0.813, +1.214]` |
