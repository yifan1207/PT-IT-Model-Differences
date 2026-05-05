# Exp24 Qwen2.5-32B Holdout-600 Subset

This CPU-only synthesis restricts the 32B residual-factorial run to the paper's holdout support: `GOV-CONV/GOV-FORMAT/SAFETY = 300/150/150`. One SAFETY prompt has no valid first-divergence event, so the exact analyzed subset is `599` events: `{'GOV-CONV': 300, 'GOV-FORMAT': 150, 'SAFETY': 149}`.

The full-1400 run remains an audit artifact, but the paper-facing Core-5 synthesis uses this matched support.
