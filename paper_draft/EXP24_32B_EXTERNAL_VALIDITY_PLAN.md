# Exp24 32B External-Validity Replication Plan

## Purpose

Exp24 should test whether the v21 headline mechanism survives scale, not rerun every historical experiment. The current paper's core claim is:

> At first-divergence prefixes, IT late computation is not an autonomous additive module; its effect on the IT-vs-PT margin is much larger when the upstream residual state is already IT-shaped.

The 32B replication should therefore prioritize the residual-state x late-stack factorial from Exp23, then rerun only the supporting decomposition experiments needed to interpret the result.

## Why this is the right next experiment

The latest draft already acknowledges that the direct causal scope is five dense 4B-8B model families. That is a real reviewer vulnerability because the title and abstract talk about instruction-tuned language models broadly. A two-family 32B replication is the cleanest external-validity test for the current story.

The literature makes this valuable but not redundant:

- Late-stage sharpening and calibration are already known phenomena. Lad et al. describe depth-stage behavior including final residual sharpening, and Joshi et al. find a later confidence-correction phase. Exp24 should show that our post-training result is not just generic late sharpening, but late readout conditioned on IT-shaped upstream state.
- Recent instruction/post-training work increasingly argues against a single universal instruction module. Bigoulaeva et al. report nonlinear instruction-vector interactions where later pathways depend on earlier task representations. Rocchetti and Ferrara argue instruction following is coordinated rather than a universal mechanism. This makes v21's "context-gated late readout" framing well aligned with the literature.
- Mid-layer localization papers are still the main conceptual pressure. Zhao et al., Chaudhury, Nepal et al., and related safety/preference-layer work make it important that Exp24 preserves the identity/readout split rather than claiming "late layers contain alignment."

Primary sources checked:

- Qwen2.5 32B base and instruct are an explicit paired family: `Qwen/Qwen2.5-32B` and `Qwen/Qwen2.5-32B-Instruct`. The model cards report 32.5B parameters, 64 layers, hidden size 5120, 40 query heads, and 8 KV heads. Qwen2.5 technical report: https://arxiv.org/abs/2412.15115
- OLMo 2 32B base and instruct are an explicit paired family: `allenai/OLMo-2-0325-32B` and `allenai/OLMo-2-0325-32B-Instruct`. The cards report 64 layers, hidden size 5120, 40 heads, and an Apache-2.0 fully open release. OLMo 2 paper/model card: https://arxiv.org/abs/2501.00656 and https://allenai.org/blog/olmo2-32b
- Relevant mechanism papers: Lad et al. https://arxiv.org/abs/2406.19384, Joshi et al. https://arxiv.org/abs/2511.00280, Bigoulaeva et al. https://arxiv.org/abs/2602.07930, Zhao et al. https://arxiv.org/abs/2410.20008, Chaudhury https://arxiv.org/abs/2510.16167, Nepal et al. https://arxiv.org/abs/2506.22638, Rocchetti and Ferrara https://arxiv.org/abs/2604.06015.

## Model pairs

Primary 32B pairs:

1. `qwen25_32b`
   - PT: `Qwen/Qwen2.5-32B`
   - IT: `Qwen/Qwen2.5-32B-Instruct`
   - Layers: 64
   - Hidden size: 5120
   - Attention: GQA, 40 query heads, 8 KV heads
   - Why: explicit base/instruct pair, dense architecture, strong instruction-tuned open model, likely adapter reuse from the existing Qwen adapter.

2. `olmo2_32b`
   - PT: `allenai/OLMo-2-0325-32B`
   - IT: `allenai/OLMo-2-0325-32B-Instruct`
   - Layers: 64
   - Hidden size: 5120
   - Attention: 40 query heads, 8 KV heads
   - Why: explicit base/instruct/post-training lineage, fully open artifacts, adapter reuse from the existing OLMo2 adapter.

Fallback if one pair is blocked:

- `gemma3_27b`: `google/gemma-3-27b-pt` / `google/gemma-3-27b-it`. This is 27B rather than 32B, but it is a clean PT/IT pair and would test whether the very large Gemma interaction in v21 is a family-specific denominator artifact or a scale-stable signal.

Do not use Qwen3-32B as the default unless a clean base/post-trained pair is verified. The public `Qwen/Qwen3-32B` card is not as clean for this paired-checkpoint protocol as Qwen2.5-32B.

Recommended final selection:

- Run `qwen25_32b` and `olmo2_32b` as the main Exp24 pair.
- Add `gemma3_27b` only as an optional third-family stress test if budget allows.
- Do not replace OLMo2 with Gemma unless access or adapter issues block OLMo2. OLMo2 is more valuable for external validity because Gemma is already in the paper and currently has the most idiosyncratic Exp23 ratio.

The current generalizability gap is scale more than family diversity. The paper already has five dense families at 4B-8B; what it lacks is a large dense-pair check. A same-family size ladder is useful, but it should not come at the cost of losing a genuinely different 32B family.

## Required code changes

1. Add model specs in `src/poc/cross_model/config.py`:

```python
"qwen25_32b": ModelSpec(
    name="qwen25_32b",
    pt_id="Qwen/Qwen2.5-32B",
    it_id="Qwen/Qwen2.5-32B-Instruct",
    n_layers=64,
    d_model=5120,
    n_heads=40,
    n_kv_heads=8,
    global_attn_layers=frozenset(range(64)),
    multi_gpu=True,
),
"olmo2_32b": ModelSpec(
    name="olmo2_32b",
    pt_id="allenai/OLMo-2-0325-32B",
    it_id="allenai/OLMo-2-0325-32B-Instruct",
    n_layers=64,
    d_model=5120,
    n_heads=40,
    n_kv_heads=8,
    global_attn_layers=frozenset(range(64)),
    multi_gpu=True,
),
```

2. Add adapter aliases in `src/poc/cross_model/adapters/__init__.py`:

```python
"qwen25_32b": Qwen3Adapter,
"olmo2_32b": OLMo2Adapter,
```

The Qwen adapter should be renamed later if desired, but the implementation is generic enough: PT gets raw text, IT uses `tokenizer.apply_chat_template(...)`.

3. Add 64-layer windows wherever hard-coded windows exist.

Use the same normalized overlapping-window policy as the existing models:

```python
"qwen25_32b": {"early": (0, 26), "mid": (19, 44), "late": (38, 64)}
"olmo2_32b": {"early": (0, 26), "mid": (19, 44), "late": (38, 64)}
```

Minimum locations:

- `src/poc/exp20_divergence_token_counterfactual/collect.py`
- `src/poc/exp11_matched_prefix_mlp_graft/run.py`
- any modal/run-script validation lists that still use `VALID_MODELS`

4. Do not just append these models to existing run scripts.

The current Exp20/21/22 scripts use `CUDA_VISIBLE_DEVICES="${gpu}"`, which exposes only one GPU. For 32B, the PT and IT checkpoints are loaded in the same process for Exp20/21/23-style paired interventions, so one 80GB GPU is not enough. Exp24 needs a runner that accepts GPU groups, for example:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 uv run python -m ...
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python -m ...
```

For safety, assume one 32B paired worker needs four 80GB GPUs. Run the two families sequentially unless eight GPUs are definitely free.

## Experiment set

### Phase 0: infrastructure smoke

Goal: fail fast on tokenizer templates, adapter hooks, multi-GPU placement, and 64-layer windows.

Run for each model pair:

- Exp23 residual factorial, `raw_shared`, 8-20 prompts, no trajectories if possible.
- Exp20 factorial validation, `raw_shared` and `native`, 8-20 prompts.
- Exp21 MLP write-out, `raw_shared` and `native`, 8-20 prompts, restricted to primary first-diff event.

Pass gates:

- PT and IT both load over the intended GPU group.
- Chat templates work for IT and raw prompts are used for PT.
- First-divergence records exist at a healthy rate.
- No-op boundary patch changes margin by approximately zero.
- 64-layer windows are reflected in configs and outputs.

### Phase 1: primary Exp23 32B replication

This is the main experiment.

Important run-order note: Exp23 consumes first-divergence rows. For brand-new 32B models, run the Exp20 raw-shared collection first, at least enough to materialize `exp20_validation_records.jsonl` for each family. Then run Exp23 residual-state factorial against that Exp20 root.

Run:

- Models: `qwen25_32b`, `olmo2_32b`
- Prompt mode: `raw_shared`
- Dataset: `data/eval_dataset_v2_holdout_0600_1199.jsonl`
- N prompts: 600 per family
- Event kinds: `first_diff`, plus `first_nonformat_diff` if compute permits
- Readout: common IT and common PT raw readouts
- Bootstrap: 2,000 prompt-level resamples within family and pooled 32B

Primary quantities:

- IT late stack effect given PT upstream
- IT late stack effect given IT upstream
- upstream main effect
- late-stack main effect
- interaction
- IT-upstream/PT-upstream ratio, reported as descriptive only

Success criterion:

- The interaction is positive in both 32B families, and the pooled 32B 95% CI excludes zero.

Interpretation:

- If both pass: v21 can say the headline context-gating result replicates at 32B scale in two additional dense PT/IT pairs.
- If one passes and one is weak/null: keep Exp24 as external-validity appendix and discuss family heterogeneity.
- If both fail: do not paper over it. The headline is a 4B-8B dense-family result, and scale changes the mechanism or measurement.

### Phase 2: Exp20 identity/margin decomposition

Run:

- Exp20 factorial validation, `raw_shared` and `native`
- Models: both 32B pairs
- N prompts: 600
- Max new tokens: 64 or 128 depending on memory and runtime

Key readouts:

- `PT + IT mid` vs `PT + IT late` IT-token transfer
- `IT + PT mid` vs `IT + PT late` PT-token transfer
- native IT-host early/mid/late IT-token retention
- native IT-host margin drops

Why:

This tests whether the middle identity / late margin split still holds at 32B. It is the most important interpretive support after Exp23.

### Phase 3: Exp21 MLP write-out decomposition

Run:

- Exp21 productive opposition, `raw_shared` and `native`
- Models: both 32B pairs
- N prompts: 600
- Restrict to the conditions actually used in the paper if runtime is high:
  - `A_pt_raw`
  - `B_mid_raw`
  - `B_late_raw`
  - `B_midlate_raw`
  - `C_it_chat`
  - `D_mid_ptswap`
  - `D_late_ptswap`
  - `D_midlate_ptswap`

Key readouts:

- pure IT write-out support by early/mid/late window
- PT-host late insertion effect
- IT-host late removal effect
- 2x2 source decomposition: late-weight, upstream-context, interaction
- residual-opposition caveat: full-update margin, negative-parallel component, delta-cosine shift

Why:

This guards against an over-broad Exp23 interpretation. If Exp23 replicates but Exp21 does not show late write-out under native IT, then the "late readout" language needs revision.

### Phase 4: convergence-gap bridge, raw-only

Run only the low-probe-dependence version.

Recommended:

- Exp23 Part A MLP KL factorial with `--readout-mode raw`
- 600 prompts per family if runtime permits; otherwise 400 with CIs
- Include mid, late, and mid+late MLP windows

Optional:

- Exp22 endpoint-deconfounded gap with `PROBE_FAMILIES=raw` only.

Do not train 32B tuned lenses for Exp24 unless the 32B raw results are confusing and the paper truly needs the probe comparison. The main causal claims do not require tuned probes, and the v21 reproducibility appendix already says the causal claims are auditable without tuned lenses.

### Phase 5: paper-facing synthesis

Create a new synthesis artifact, not a one-off notebook:

- `scripts/analysis/build_exp24_32b_external_validity_synthesis.py`
- output root: `results/paper_synthesis/exp24_32b_external_validity/`

Required outputs:

- `exp24_32b_summary.csv`
- `exp24_32b_summary.md`
- `exp24_32b_interaction.png`
- `exp24_32b_claims.json`

Minimum table columns:

- model
- n_first_divergence_records
- pt_upstream_late_effect
- pt_upstream_ci_low
- pt_upstream_ci_high
- it_upstream_late_effect
- it_upstream_ci_low
- it_upstream_ci_high
- interaction
- interaction_ci_low
- interaction_ci_high
- ratio_descriptive
- exp20_mid_identity_transfer
- exp20_late_identity_transfer
- exp21_late_writeout_support
- convergence_gap_raw_late_effect

## Compute plan

Assume one paired 32B worker needs four 80GB GPUs because Exp20/21/23 load PT and IT in the same process.

Suggested scheduling:

1. Check `nvidia-smi` and avoid other users' processes.
2. Run Qwen2.5 32B on one group, for example `CUDA_VISIBLE_DEVICES=2,3,4,5`.
3. Run OLMo2 32B on a second group only if four more GPUs are truly free. Otherwise run sequentially.
4. Prefer one worker per family at first. Add sharding only after the smoke confirms multi-GPU memory and resume behavior.

Expected rough order:

```bash
# after code changes and smoke, collect Exp20 raw-shared manifests first
CUDA_VISIBLE_DEVICES=2,3,4,5 bash scripts/run/run_exp24_32b_external_validity.sh \
  --model qwen25_32b --phase exp23-primary --n-prompts 600

CUDA_VISIBLE_DEVICES=2,3,4,5 bash scripts/run/run_exp24_32b_external_validity.sh \
  --model olmo2_32b --phase exp23-primary --n-prompts 600
```

## Rough RunPod cost estimate

Use this only as a planning estimate; actual throughput will depend on host disk, cache locality, exact GPU interconnect, and whether the job keeps both PT and IT resident without offload.

Official RunPod H100 SXM listed price checked on 2026-04-27: from about `$2.99/GPU-hour`. A four-H100 worker is therefore about `$11.96/hour`.

Recommended budget for two 32B families:

| Phase | GPU layout | Wall-clock estimate | Cost estimate |
|---|---:|---:|---:|
| Smoke and debugging | 4xH100 | 2-4 h | `$24-$48` |
| Exp20 raw-shared manifests | 4xH100 sequential per family | 6-12 h total | `$72-$144` |
| Exp23 residual factorial | 4xH100 sequential per family | 8-16 h total | `$96-$191` |
| Exp20 native completion | 4xH100 sequential per family | 6-12 h total | `$72-$144` |
| Exp21 MLP write-out | 4xH100 sequential per family | 8-18 h total | `$96-$215` |
| Raw KL bridge | 4xH100 sequential per family | 6-14 h total | `$72-$167` |
| Analysis/plotting | CPU or one cheap GPU | 1-3 h | `$0-$20` |

Practical total:

- Lean run: about `$430-$550`
- Expected run: about `$600-$850`
- Conservative run with retries/cache misses: about `$1,000-$1,300`

Adding optional `gemma3_27b` as a third family likely adds about `35-50%` more, not a full 50%, because some script/debugging overhead is already paid. Budget another `$250-$450`.

## What to add to the paper

If Exp24 succeeds, do not overload the abstract with all numbers. Add one compact external-validity sentence:

> A two-family 32B replication on Qwen2.5-32B and OLMo-2-32B preserves the positive upstream-state x late-stack interaction, extending the context-gated late-readout result beyond the original 4B-8B dense-family pool.

Then put the detailed table and figure in Appendix F or a new Appendix H. The main text can keep Dense-5 as the primary fully audited package and cite Exp24 as scale replication.

If Exp24 is mixed, keep the current main framing and use Exp24 as a limitation/result:

> At 32B scale the interaction remains positive in one family but weakens in another, suggesting family- or recipe-dependent strength of the middle-to-late handoff.

If Exp24 fails, do not add it as a "minor caveat." It would force a narrower title/abstract and make scale dependence a central discussion point.

## Decision

Run Exp24, but define "all important experiments" narrowly:

1. Primary: Exp23 residual-state x late-stack factorial.
2. Required support: Exp20 first-divergence identity/margin and Exp21 MLP write-out/decomposition.
3. Bridge: raw-only convergence-gap KL factorial/endpoint check.
4. Skip for now: steering, behavioral judge, tuned-lens training, and broad prompt-domain reruns.

This is the highest signal-to-compute plan. It tests the actual v21 claim, handles the biggest reviewer scope concern, and avoids resurrecting the older "universal single late direction" framing that v21 correctly moved away from.
