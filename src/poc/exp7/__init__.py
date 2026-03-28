"""Exp7: Tier-0 methodology fixes for NeurIPS 2026 submission.

Experiments:
  0A  Direction calibration sensitivity (bootstrap)
  0B  Matched-token direction validation (force-decode confound control)
  0C  Projection-matched random direction control
  0D  Bootstrap CIs on all main figures
  0E  Token classifier specification + robustness check
  0F  Corrective layer range justification (sensitivity analysis)
  0G  Tuned-lens replication of commitment delay
  0H  Calibration-evaluation split validation
  0I  Intervention formula sensitivity
  0J  Corrective onset threshold sensitivity analysis

All results written to results/exp7/{0A..0J}/
"""
