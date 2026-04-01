# 0G Tuned-Lens Experiment — Handoff for Server Execution

## What This Is

Experiment 0G trains tuned-lens probes (Belrose et al. 2023 exact recipe) for 6 models × 2 variants (PT/IT) = 12 training jobs, then evaluates commitment delay. This is part of Exp7 Tier 0 methodology validation for a NeurIPS submission.

**Status: NO training has completed.** Multiple Modal cloud runs failed due to preemptions, API incompatibilities, and disconnections. Zero checkpoint probes were saved. Everything must run from scratch on the 8×A100 server.

---

## Scientific Standards (DO NOT CUT CORNERS)

The training recipe must match **Belrose et al. 2023 exactly**:

| Parameter | Value | Why |
|-----------|-------|-----|
| Optimizer | SGD + Nesterov | Belrose exact |
| Learning rate | 1.0 | Belrose exact |
| Momentum | 0.9 | Belrose exact |
| Weight decay | 0 | Belrose exact |
| Steps | **250** (not less) | Belrose exact |
| Tokens per step | **262,144** | Belrose exact |
| Total token-activations | 65.5M | 250 × 262,144 |
| LR schedule | Linear decay to 0, no warmup | Belrose exact |
| Chunk length | 2048 tokens | Belrose exact |
| Initialization | Identity weight + zero bias | Belrose exact |
| Loss | KL divergence (sum / batch_size) | Belrose exact |
| Gradient clipping | Norm 1.0 | Belrose exact |
| Training mode | Joint all-layer (all probes per forward pass) | 34× faster, equivalent results |
| `model.requires_grad_(False)` | **Must be called** | Saves ~3 GB on lm_head grads, prevents OOM |

**Do NOT reduce steps, tokens/step, or number of models.** The `micro_batch_size` parameter only controls gradient accumulation chunking — it does not change total tokens per step. Adjusting it for GPU memory is fine.

---

## Code Entry Points

All code is in the repo. No external scripts needed.

### Training + Eval (per model)
```bash
# Train probes for one model+variant:
uv run python -m src.poc.cross_model.tuned_lens \
    --model gemma3_4b --variant pt \
    --device cuda:0

# Eval commitment delay (after training):
uv run python -m src.poc.cross_model.tuned_lens \
    --model gemma3_4b --variant pt \
    --device cuda:0 --eval-only --n-eval-examples 200
```

### Key source files
| File | Purpose |
|------|---------|
| `src/poc/cross_model/tuned_lens.py` | Training (`train_probes`), eval (`eval_commitment`), transfer test (`run_transfer_test`) |
| `src/poc/cross_model/config.py` | `MODEL_REGISTRY`, `get_spec()`, `model_id_for_variant()` |
| `src/poc/cross_model/adapters/__init__.py` | `get_adapter()` — per-model hook paths |
| `src/poc/cross_model/utils.py` | `load_model_and_tokenizer()`, `load_dataset()` |
| `data/eval_dataset_v2.jsonl` | Eval dataset (200 examples) |
| `scripts/plot_tuned_lens_commitment.py` | Generates 0G commitment plots |

---

## The 6 Models

| Name | HF ID (PT) | HF ID (IT) | Layers | d_model | Notes |
|------|-----------|-----------|--------|---------|-------|
| `gemma3_4b` | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` | 34 | 2560 | |
| `llama31_8b` | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` | 32 | 4096 | |
| `qwen3_4b` | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B` | 36 | 2560 | |
| `mistral_7b` | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | 32 | 4096 | |
| `deepseek_v2_lite` | `deepseek-ai/DeepSeek-V2-Lite` | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 27 | 2048 | MoE, `eager_attn=True`, uses `trust_remote_code=True` |
| `olmo2_7b` | `allenai/OLMo-2-1124-7B` | `allenai/OLMo-2-1124-7B-Instruct` | 32 | 4096 | |

All fit on 1×A100 80GB. None need multi-GPU.

---

## Execution Plan (8×A100 server)

### Phase 1: Training (12 jobs, 8 GPUs)
Run 8 jobs in parallel across GPUs, then the remaining 4 when slots free up.

```bash
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
mkdir -p logs/exp7/0G

MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b deepseek_v2_lite olmo2_7b)
VARIANTS=(pt it)
GPU=0

for model in "${MODELS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        CUDA_VISIBLE_DEVICES=$GPU uv run python -m src.poc.cross_model.tuned_lens \
            --model "$model" --variant "$variant" \
            --device "cuda:0" \
            > "logs/exp7/0G/${model}_${variant}.log" 2>&1 &
        
        GPU=$(( (GPU + 1) % 8 ))
        
        # After launching 8, wait for any to finish before continuing
        if [ $GPU -eq 0 ] && [ "$model" != "gemma3_4b" ]; then
            wait -n
        fi
    done
done
wait
echo "All training complete"
```

**Estimated time**: ~4-6h for 4B models, ~8-10h for 7-8B models. Total wall time ~10h with 8 GPUs.

### Phase 2: Eval (12 jobs, 8 GPUs)
Same pattern, with `--eval-only`:
```bash
GPU=0
for model in "${MODELS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        CUDA_VISIBLE_DEVICES=$GPU uv run python -m src.poc.cross_model.tuned_lens \
            --model "$model" --variant "$variant" \
            --device "cuda:0" --eval-only --n-eval-examples 200 \
            > "logs/exp7/0G/${model}_${variant}_eval.log" 2>&1 &
        
        GPU=$(( (GPU + 1) % 8 ))
        if [ $GPU -eq 0 ] && [ "$model" != "gemma3_4b" ]; then
            wait -n
        fi
    done
done
wait
echo "All eval complete"
```

**DeepSeek eval note**: `max_new_tokens` should be 64 (not 512) for deepseek_v2_lite. The code handles this automatically when `spec.is_moe == True` in the eval path — but verify the calling code passes this. If running eval manually, check that generation doesn't hang on deepseek.

### Phase 3: Transfer tests (6 jobs)
```bash
GPU=0
for model in "${MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU uv run python -m src.poc.cross_model.tuned_lens \
        --model "$model" --transfer-test \
        --device "cuda:0" \
        > "logs/exp7/0G/${model}_transfer.log" 2>&1 &
    GPU=$(( (GPU + 1) % 8 ))
done
wait
```

### Phase 4: Push raw results to GCS
```bash
gsutil -m rsync -r results/cross_model/ gs://pt-vs-it-results/cross_model/
```

### Phase 5: Generate plots
```bash
uv run python scripts/plot_tuned_lens_commitment.py
```

### Phase 6: Regenerate CIs and Tier 0 plots
```bash
uv run python -m src.poc.exp7.bootstrap_ci \
    --merged-dir results/exp6/merged_A1_it_v4 \
    --cross-model-dir results/cross_model
cp results/exp7/0D/ci_cross_model.json results/exp7/plots/data/
uv run python scripts/plot_exp7_tier0.py
```

### Phase 7: Push plots to GCS
```bash
gsutil -m cp results/cross_model/plots/0G*.png gs://pt-vs-it-results/plots/0G/
gsutil -m cp results/exp7/plots/0G*.png gs://pt-vs-it-results/plots/0G/
```

---

## Output Paths

| What | Path |
|------|------|
| Trained probes | `results/cross_model/{model}/tuned_lens/{pt,it}/probe_layer_*.pt` |
| Training summary | `results/cross_model/{model}/tuned_lens/{pt,it}/training_summary.json` |
| Checkpoints (intermediate) | `results/cross_model/{model}/tuned_lens/{pt,it}/checkpoint.json` |
| Commitment JSONL | `results/cross_model/{model}/tuned_lens/commitment/tuned_lens_commitment_{pt,it}.jsonl` |
| Eval summary | `results/cross_model/{model}/tuned_lens/commitment/summary_{pt,it}.json` |
| Transfer test | `results/cross_model/{model}/tuned_lens/commitment/transfer_test.json` |
| Plots | `results/cross_model/plots/0G_*.png` and `results/exp7/plots/0G_*.png` |

---

## Checkpoint System (NEW)

`tuned_lens.py` now saves checkpoint probes every 50 training steps:
- At step 50, 100, 150, 200: saves all `probe_layer_*.pt` files + `checkpoint.json`
- At step 250 (end): saves final best probes + `training_summary.json`

**Resume logic**:
- If `training_summary.json` exists → training is complete, skip
- If `probe_layer_*.pt` exist but no `training_summary.json` → retrain from scratch (checkpoints are a safety net for data recovery, not a resume point — optimizer state isn't saved)
- If nothing exists → train normally

This means if a job is killed at step 200, you have step-200 probes on disk. They're usable but not final quality. Restarting the job will retrain from step 0 to get proper step-250 probes.

---

## Errors Encountered During Modal Runs (For Reference)

These are Modal-specific issues. They should NOT occur on the server with local transformers 4.57.3.

1. **`modal.Mount` AttributeError** — Modal 1.4 removed `Mount`; replaced with `image.add_local_dir()`
2. **`concurrency_limit` deprecation** — Modal renamed to `max_containers`
3. **DeepSeek `is_flash_attn_greater_or_equal_2_10` ImportError** — removed in transformers 5.x; DeepSeek's custom `modeling_deepseek.py` imports it at module level
4. **DeepSeek `is_torch_fx_available` ImportError** — same cause, also removed in transformers 5.x
5. **DeepSeek `DynamicCache.from_legacy_cache` AttributeError** — removed in transformers 5.x
6. **Container preemptions** — Modal kills containers under load; ~1 preemption/hour observed. Lost all training progress each time since probes only saved at step 250 (now fixed with checkpoints)
7. **Local client disconnection** — `modal run` without `--detach` kills all containers when local process exits
8. **C4 dataset load timeout** — some containers got HTTP timeouts loading C4; fell back to wikitext (smaller dataset, not ideal)

**Key takeaway**: All DeepSeek issues were caused by `transformers>=5.0.0` on Modal. The server uses transformers 4.57.3 locally — these errors will not occur. If you ever upgrade transformers past 5.0, DeepSeek will need `trust_remote_code=True` compatibility patches.

---

## GPU Usage Notes

- Server has 8×A100 80GB (cuda:0-7)
- Another user (ubuntu) may be running RL training — **check `nvidia-smi` before claiming GPUs**
- Never interfere with other users' processes
- All 6 models fit on 1×A100 80GB; no multi-GPU needed
- `eager_attn=True` only for deepseek_v2_lite (MoE); all others use default SDPA

---

## Verification Checklist

After training completes, verify:
- [ ] All 12 `training_summary.json` files exist
- [ ] Each summary shows 250 steps completed
- [ ] Probe counts match model layers (34, 32, 36, 32, 27, 32)
- [ ] Loss values are reasonable (typically 0.1-0.5 KL at convergence)
- [ ] Commitment JSONL files have 200 records each (12 files total)
- [ ] Transfer test JSON exists for all 6 models
- [ ] Plots generated successfully (4 PNG files)
