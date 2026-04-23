"""Exp17: external behavioral-direction replications plus joint synthesis.

This package is for paper-facing replication work that links prior literature on
assistantness / truthfulness / refusal directions to the repo's main
observational and causal results on the convergence gap and late MLP leverage.

Structure:
  - `lu_pipeline.py`: wrapper around Lu et al.'s public Assistant Axis pipeline
  - `du_truthfulness.py`: cross-model truthfulness direction extraction
  - `du_refusal.py`: cross-model refusal-direction candidate extraction
  - `joint_analysis.py`: compares exp17 layer summaries against exp09 peaks
  - `shared.py`: shared model loading, prompt handling, activation collection
"""
