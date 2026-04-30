# Exp23 Dense-6 Core Synthesis

Dense-6 combines the five 4B-8B Exp23 families with the Qwen2.5-32B Exp24 family.
CIs use an independent normal approximation to the stored per-family prompt-bootstrap intervals.

- Late IT from PT upstream: `+0.639` `[+0.570, +0.709]`.
- Late IT from IT upstream: `+3.076` `[+2.978, +3.174]`.
- Upstream x late interaction: `+2.437` `[+2.353, +2.521]`.
- Gemma-removed Dense-5 interaction: `+1.709` `[+1.637, +1.780]`.
- Common-PT cross-check: PT upstream `+0.662` `[+0.600, +0.724]`; IT upstream `+3.083` `[+2.986, +3.180]`; interaction `+2.421` `[+2.337, +2.506]`.

## Position Rows

| Stratum | Dense-6 interaction | Gemma removed |
|---|---:|---:|
| all positions | `+2.437` `[+2.353, +2.521]` | `+1.709` `[+1.637, +1.781]` |
| positions >=1 | `+2.079` `[+1.963, +2.194]` | `+1.159` `[+1.080, +1.237]` |
| positions >=3 | `+1.434` `[+1.300, +1.569]` | `+0.834` `[+0.755, +0.914]` |
| position >=5 | `+1.480` `[+1.276, +1.684]` | `+0.798` `[+0.701, +0.895]` |
| position >=10 | `+1.393` `[+0.817, +1.969]` | `+1.014` `[+0.813, +1.214]` |
