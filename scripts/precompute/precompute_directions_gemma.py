#!/usr/bin/env python3
"""Precompute corrective directions v2 — unified pipeline, single inference pass.

All three direction vectors (layers 1-11, 12-19, 20-33) are computed from the
SAME selected records and the SAME token pairs.  A single forward pass hooks all
layers 1-33 simultaneously, so no layer group is trained on a different subset.

Pipeline (orchestrated by run_precompute_v2.sh):

  Phase 1 — gen   (GPU, 8 parallel workers)
    Each worker handles its record slice: generates IT and PT texts (no chat
    template, raw formats["B"]), computes STR and PT_NLL(IT_output).
    Output: work_dir/gen/w{wi}.jsonl

  Phase 2 — score (CPU, no GPU)
    Merges worker gen files.  Runs LLM judge G1 on IT and PT outputs.
    Computes contrast = (G1_IT - G1_PT)/4 + (STR_IT - STR_PT) + norm_NLL.
    Selects top_n high-contrast record IDs.
    Output: work_dir/selected.json, work_dir/gen_merged.jsonl

  Phase 3 — acts  (GPU, 8 parallel workers)
    Each worker re-generates its slice of the SELECTED records with MLP hooks
    active on ALL layers 1-33 simultaneously.  Captures activations at generated
    token positions only (shape[1]==1 with use_cache=True).
    k = min(IT_gen_len, PT_gen_len, MAX_GEN) per record — symmetric for IT/PT.
    Output: work_dir/acts/w{wi}.npz  (per-layer IT/PT sums + token counts)

  Phase 4 — merge (CPU, no GPU)
    Sums all worker act files.  Computes direction for every layer 1-33.
    Saves one npz with all layers — A1/A1_early/A1_mid all reference this file;
    each experiment's ablation_layers config determines which subset to apply.
    Output: OUT_DIR/corrective_directions.npz + meta.json

Activation hook design:
  use_cache=True → prefill is one call [B, T_prompt, D], each generated step
  is [B, 1, D].  Detect shape[1]==1 to collect only generated-token activations.

Usage:
  python scripts/precompute_directions_v2.py --phase gen  --worker-index 0 --n-workers 8 --device cuda:0
  python scripts/precompute_directions_v2.py --phase score --judge-workers 16
  python scripts/precompute_directions_v2.py --phase acts --worker-index 0 --n-workers 8 --device cuda:0
  python scripts/precompute_directions_v2.py --phase merge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.poc.collect import load_dataset_records
from src.poc.exp05_corrective_direction_ablation_cartography.config import Exp5Config
from src.poc.exp05_corrective_direction_ablation_cartography.utils import ensure_dir, save_json
from src.poc.shared.model import load_model

# ── Config ────────────────────────────────────────────────────────────────────
ALL_LAYERS = list(range(1, 34))   # layers 1-33 — hook all simultaneously
D_MODEL    = 2560
MAX_GEN    = 80      # max generated tokens per record
TOP_N      = 600     # high-contrast records to use for direction

WORK_DIR = Path("results/precompute_v2_work")
OUT_DIR  = Path("results/exp5/precompute_v2/precompute")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _structural_token_ratio(text: str) -> float:
    if not text.strip():
        return 0.0
    try:
        from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import classify_generated_tokens_by_word
        words = text.split()
        if not words:
            return 0.0
        cats = classify_generated_tokens_by_word([{"token_str": w + " "} for w in words])
        gov = {"STRUCTURAL", "DISCOURSE", "PUNCTUATION"}
        return sum(1 for c in cats if c in gov) / len(cats)
    except Exception:
        return 0.0


def _get_raw(loaded: Any) -> Any:
    return loaded.model._model


def _generate_batch(model_raw, tokenizer, prompts, device, max_new_tokens=MAX_GEN):
    """Generate from raw prompts, no chat template. Returns list of (text, token_ids)."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask  = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        out = model_raw.generate(
            input_ids, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    results = []
    for i in range(len(prompts)):
        ids = [t for t in out[i, prompt_len:].tolist()
               if t not in (tokenizer.pad_token_id, tokenizer.eos_token_id)]
        results.append((tokenizer.decode(ids, skip_special_tokens=True), ids))
    return results


def _compute_nll(model_raw, tokenizer, prompts, gen_ids_list, device):
    """PT NLL on IT's generated tokens (teacher-forcing). Higher = more surprised."""
    nlls = []
    for prompt, gen_ids in zip(prompts, gen_ids_list):
        if not gen_ids:
            nlls.append(0.0)
            continue
        p_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        g_ids = torch.tensor(gen_ids, dtype=torch.long, device=device).unsqueeze(0)
        full  = torch.cat([p_ids, g_ids], dim=1)
        pl    = p_ids.shape[1]
        with torch.no_grad():
            logits = model_raw(full).logits
        lp    = torch.nn.functional.log_softmax(logits[0, pl-1:-1], dim=-1)
        nll   = -lp[range(len(gen_ids)), gen_ids].float().cpu().mean().item()
        nlls.append(nll)
    return nlls


# ── Phase 1: gen ──────────────────────────────────────────────────────────────

def phase_gen(worker_index: int, n_workers: int, device: str) -> None:
    wdir     = ensure_dir(WORK_DIR / "gen")
    out_path = wdir / f"w{worker_index}.jsonl"
    if out_path.exists():
        print(f"[gen w{worker_index}] already done, skipping.", flush=True)
        return

    records   = load_dataset_records("data/eval_dataset_v2.jsonl", prompt_format="B")
    my_records = records[worker_index::n_workers]
    prompts   = [r["formats"]["B"] for r in my_records]
    print(f"[gen w{worker_index}/{n_workers}] {len(my_records)} records on {device}", flush=True)

    pt_cfg = Exp5Config(experiment="baseline", model_variant="pt", model_id="",
                        run_name=f"pre_gen_pt_w{worker_index}", device=device, skip_transcoders=True)
    it_cfg = Exp5Config(experiment="baseline", model_variant="it", model_id="",
                        run_name=f"pre_gen_it_w{worker_index}", device=device, skip_transcoders=True)
    BS = 8

    # ── PT generation ──────────────────────────────────────────────────────────
    print(f"[gen w{worker_index}] loading PT...", flush=True)
    pt_loaded = load_model(pt_cfg)
    pt_raw    = _get_raw(pt_loaded)
    tokenizer = pt_loaded.tokenizer

    pt_texts, pt_ids_all = [], []
    for i in range(0, len(prompts), BS):
        batch = prompts[i:i+BS]
        for text, ids in _generate_batch(pt_raw, tokenizer, batch, device):
            pt_texts.append(text); pt_ids_all.append(ids)
        if (i // BS + 1) % 10 == 0:
            print(f"  PT gen {min(i+BS, len(prompts))}/{len(prompts)}", flush=True)

    del pt_loaded; torch.cuda.empty_cache()

    # ── IT generation ──────────────────────────────────────────────────────────
    print(f"[gen w{worker_index}] loading IT...", flush=True)
    it_loaded = load_model(it_cfg)
    it_raw    = _get_raw(it_loaded)

    it_texts, it_ids_all = [], []
    for i in range(0, len(prompts), BS):
        batch = prompts[i:i+BS]
        for text, ids in _generate_batch(it_raw, tokenizer, batch, device):
            it_texts.append(text); it_ids_all.append(ids)
        if (i // BS + 1) % 10 == 0:
            print(f"  IT gen {min(i+BS, len(prompts))}/{len(prompts)}", flush=True)

    del it_loaded; torch.cuda.empty_cache()

    # ── PT NLL on IT outputs ───────────────────────────────────────────────────
    print(f"[gen w{worker_index}] loading PT for NLL...", flush=True)
    pt2 = load_model(pt_cfg)
    pt2_raw = _get_raw(pt2)

    nlls = []
    for i in range(0, len(prompts), BS):
        bp = prompts[i:i+BS]
        bi = it_ids_all[i:i+BS]
        nlls.extend(_compute_nll(pt2_raw, tokenizer, bp, bi, device))

    del pt2; torch.cuda.empty_cache()

    # ── Save ───────────────────────────────────────────────────────────────────
    it_strs = [_structural_token_ratio(t) for t in it_texts]
    pt_strs = [_structural_token_ratio(t) for t in pt_texts]

    with open(out_path, "w") as f:
        for j, rec in enumerate(my_records):
            row = {
                "record_id":  rec["id"],
                "category":   rec.get("category", ""),
                "prompt":     prompts[j],
                "it_text":    it_texts[j],
                "pt_text":    pt_texts[j],
                "it_gen_ids": it_ids_all[j],
                "pt_gen_ids": pt_ids_all[j],
                "it_str":     it_strs[j],
                "pt_str":     pt_strs[j],
                "pt_nll":     nlls[j],
            }
            f.write(json.dumps(row) + "\n")

    print(f"[gen w{worker_index}] saved {len(my_records)} records → {out_path}", flush=True)


# ── Phase 2: score ────────────────────────────────────────────────────────────

def _call_judge(client, model: str, prompt: str, retries: int = 4) -> dict:
    import random
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=128, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)
        except Exception as e:
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(2 ** attempt + random.uniform(0, 1))
    return {"error": "max retries"}


def _run_llm_judge_g1(rows: list[dict], judge_workers: int = 16) -> dict[str, dict[str, float]]:
    """Run G1 judge on both IT and PT texts. Returns {record_id: {"it": score, "pt": score}}."""
    from src.poc.exp06_corrective_direction_steering.eval_registry import RUBRICS
    from src.poc.shared.llm_provider import build_openai_client
    rubric = RUBRICS["g1"]

    client_info = build_openai_client("google/gemini-2.5-flash", provider="auto")
    if client_info is None:
        print("  WARNING: no GEMINI_API_KEY or OPENROUTER_API_KEY — skipping LLM judge, using STR+NLL only", flush=True)
        return {}

    client, model, provider = client_info
    print(f"  Using judge provider={provider} model={model}", flush=True)

    tasks = []
    for row in rows:
        q = row["prompt"][:500]
        tasks.append(("it", row["record_id"], rubric.format(question=q, response=row["it_text"][:1200])))
        tasks.append(("pt", row["record_id"], rubric.format(question=q, response=row["pt_text"][:1200])))

    print(f"  Running G1 judge on {len(tasks)} outputs ({len(rows)} records × IT+PT)...", flush=True)
    scores: dict[str, dict[str, float]] = {}

    def score_one(task):
        side, rid, prompt = task
        result = _call_judge(client, model, prompt)
        return side, rid, result.get("g1", -1)

    done = 0
    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        futures = {pool.submit(score_one, t): t for t in tasks}
        for fut in as_completed(futures):
            side, rid, val = fut.result()
            if rid not in scores:
                scores[rid] = {}
            try:
                scores[rid][side] = float(val)
            except (TypeError, ValueError):
                scores[rid][side] = -1.0
            done += 1
            if done % 200 == 0:
                print(f"  G1: {done}/{len(tasks)}", flush=True)

    return scores


def phase_score(top_n: int = TOP_N, judge_workers: int = 16) -> None:
    sel_path = WORK_DIR / "selected.json"
    if sel_path.exists():
        print("[score] selected.json already exists, skipping.", flush=True)
        return

    # Merge all worker gen files
    gen_dir = WORK_DIR / "gen"
    rows = []
    for f in sorted(gen_dir.glob("w*.jsonl")):
        for line in open(f):
            rows.append(json.loads(line))
    print(f"[score] merged {len(rows)} records from {len(list(gen_dir.glob('w*.jsonl')))} workers", flush=True)

    # LLM judge G1 (optional — falls back to STR+NLL if no API key)
    g1_scores = _run_llm_judge_g1(rows, judge_workers=judge_workers)

    # Compute contrast scores
    nlls    = [r["pt_nll"] for r in rows]
    max_nll = max(nlls) if max(nlls) > 0 else 1.0

    scored = []
    for r in rows:
        rid          = r["record_id"]
        str_contrast = r["it_str"] - r["pt_str"]
        norm_nll     = r["pt_nll"] / max_nll
        g1_contrast  = 0.0
        if rid in g1_scores:
            g1_it = g1_scores[rid].get("it", -1)
            g1_pt = g1_scores[rid].get("pt", -1)
            if g1_it >= 0 and g1_pt >= 0:
                g1_contrast = (g1_it - g1_pt) / 4.0
        min_k    = min(len(r["it_gen_ids"]), len(r["pt_gen_ids"]))
        contrast = str_contrast + norm_nll + g1_contrast
        scored.append((contrast, rid, min_k))

    # Filter: must have generated ≥5 tokens from both models; select top_n by contrast
    valid = [(c, rid, k) for c, rid, k in scored if k >= 5]
    valid.sort(reverse=True)
    selected = [rid for _, rid, _ in valid[:top_n]]

    print(f"[score] selected {len(selected)}/{len(rows)} records", flush=True)
    if valid:
        top_scores = [c for c, _, _ in valid[:top_n]]
        print(f"  contrast range: {top_scores[0]:.3f} → {top_scores[-1]:.3f}", flush=True)

    with open(WORK_DIR / "gen_merged.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(sel_path, "w") as f:
        json.dump(selected, f)
    print(f"[score] saved → {sel_path}", flush=True)


# ── Phase 3: acts ─────────────────────────────────────────────────────────────

def phase_acts(worker_index: int, n_workers: int, device: str) -> None:
    out_path = ensure_dir(WORK_DIR / "acts") / f"w{worker_index}.npz"
    if out_path.exists():
        print(f"[acts w{worker_index}] already done, skipping.", flush=True)
        return

    # Load selected IDs and full gen data
    selected_ids = set(json.loads(open(WORK_DIR / "selected.json").read()))
    rows_by_id   = {}
    for line in open(WORK_DIR / "gen_merged.jsonl"):
        r = json.loads(line)
        if r["record_id"] in selected_ids:
            rows_by_id[r["record_id"]] = r

    my_ids  = sorted(rows_by_id.keys())[worker_index::n_workers]
    my_rows = [rows_by_id[rid] for rid in my_ids]
    print(f"[acts w{worker_index}/{n_workers}] {len(my_rows)} records on {device} "
          f"— hooking ALL layers {ALL_LAYERS[0]}-{ALL_LAYERS[-1]}", flush=True)

    pt_cfg = Exp5Config(experiment="baseline", model_variant="pt", model_id="",
                        run_name=f"pre_acts_pt_w{worker_index}", device=device, skip_transcoders=True)
    it_cfg = Exp5Config(experiment="baseline", model_variant="it", model_id="",
                        run_name=f"pre_acts_it_w{worker_index}", device=device, skip_transcoders=True)

    def collect_acts(model_raw, tokenizer, rows_subset, is_it: bool) -> tuple[dict, dict]:
        sums   = {li: torch.zeros(D_MODEL, dtype=torch.float64) for li in ALL_LAYERS}
        counts = {li: 0 for li in ALL_LAYERS}

        for idx, row in enumerate(rows_subset):
            k = min(len(row["it_gen_ids"]), len(row["pt_gen_ids"]), MAX_GEN)
            if k == 0:
                continue

            # One dict per layer — all layers hooked simultaneously
            gen_acts: dict[int, list] = {li: [] for li in ALL_LAYERS}

            def make_hook(li: int):
                def hook(mod, inp, out):
                    # shape[1]==1 → generated step; shape[1]>1 → prefill (skip)
                    if out.shape[1] == 1 and len(gen_acts[li]) < k:
                        gen_acts[li].append(out[0, 0, :].float().cpu())
                return hook

            # Register hooks on ALL layers in one pass
            handles = [
                model_raw.language_model.layers[li].mlp.register_forward_hook(make_hook(li))
                for li in ALL_LAYERS
            ]
            try:
                input_ids = tokenizer.encode(row["prompt"], return_tensors="pt").to(device)
                with torch.no_grad():
                    model_raw.generate(
                        input_ids, max_new_tokens=k, do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
            finally:
                for h in handles:
                    h.remove()

            for li in ALL_LAYERS:
                if gen_acts[li]:
                    stacked     = torch.stack(gen_acts[li])        # [T, d_model]
                    sums[li]   += stacked.sum(dim=0).to(torch.float64)
                    counts[li] += stacked.shape[0]

            if (idx + 1) % 50 == 0:
                label = "IT" if is_it else "PT"
                print(f"  {label} acts {idx+1}/{len(rows_subset)}, "
                      f"layer 20 tokens: {counts[20]}", flush=True)

        return sums, counts

    # IT activations
    print(f"[acts w{worker_index}] loading IT...", flush=True)
    it_loaded = load_model(it_cfg)
    it_sums, it_counts = collect_acts(_get_raw(it_loaded), it_loaded.tokenizer, my_rows, is_it=True)
    del it_loaded; torch.cuda.empty_cache()

    # PT activations
    print(f"[acts w{worker_index}] loading PT...", flush=True)
    pt_loaded = load_model(pt_cfg)
    pt_sums, pt_counts = collect_acts(_get_raw(pt_loaded), pt_loaded.tokenizer, my_rows, is_it=False)
    del pt_loaded; torch.cuda.empty_cache()

    # Per-record token count = min(IT_gen_len, PT_gen_len, MAX_GEN)
    per_record_k = [
        min(len(row["it_gen_ids"]), len(row["pt_gen_ids"]), MAX_GEN)
        for row in my_rows
    ]

    # Save all layers
    payload = {}
    for li in ALL_LAYERS:
        payload[f"it_sum_{li}"]   = it_sums[li].to(torch.float32).numpy()
        payload[f"pt_sum_{li}"]   = pt_sums[li].to(torch.float32).numpy()
        payload[f"it_count_{li}"] = np.array(it_counts[li], dtype=np.int64)
        payload[f"pt_count_{li}"] = np.array(pt_counts[li], dtype=np.int64)
    payload["n_records"]    = np.array(len(my_rows), dtype=np.int64)
    payload["per_record_k"] = np.array(per_record_k, dtype=np.int32)
    np.savez_compressed(out_path, **payload)

    mean_k = sum(per_record_k) / max(len(per_record_k), 1)
    print(f"[acts w{worker_index}] saved → {out_path} "
          f"(mean k={mean_k:.1f} tokens/record, layers 1-33)", flush=True)


# ── Phase 4: merge ────────────────────────────────────────────────────────────

def phase_merge() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    act_files = sorted((WORK_DIR / "acts").glob("w*.npz"))
    print(f"[merge] combining {len(act_files)} worker files (layers 1-33)...", flush=True)

    it_sums    = {li: np.zeros(D_MODEL, dtype=np.float64) for li in ALL_LAYERS}
    pt_sums    = {li: np.zeros(D_MODEL, dtype=np.float64) for li in ALL_LAYERS}
    it_counts  = {li: 0 for li in ALL_LAYERS}
    pt_counts  = {li: 0 for li in ALL_LAYERS}
    total_recs = 0
    all_per_record_k: list[int] = []

    for f in act_files:
        with np.load(f) as d:
            for li in ALL_LAYERS:
                it_sums[li]   += d[f"it_sum_{li}"].astype(np.float64)
                pt_sums[li]   += d[f"pt_sum_{li}"].astype(np.float64)
                it_counts[li] += int(d[f"it_count_{li}"])
                pt_counts[li] += int(d[f"pt_count_{li}"])
            total_recs += int(d["n_records"])
            if "per_record_k" in d:
                all_per_record_k.extend(d["per_record_k"].tolist())

    rep_layer = 20   # representative layer for reporting
    total_tok = it_counts[rep_layer]
    mean_k    = total_tok / max(total_recs, 1)
    min_k     = min(all_per_record_k) if all_per_record_k else 0
    max_k     = max(all_per_record_k) if all_per_record_k else 0

    print(f"\n[merge] ── Sample accounting ──────────────────────────────────", flush=True)
    print(f"  Records selected (high-contrast pairs) : {total_recs}", flush=True)
    print(f"  Tokens per record = min(IT_gen, PT_gen) : min={min_k}  max={max_k}  mean={mean_k:.1f}", flush=True)
    print(f"  Total token-pair observations per layer : {total_tok:,}", flush=True)
    print(f"  ALL layer groups use the SAME {total_recs} records / {total_tok:,} tokens", flush=True)
    print(f"────────────────────────────────────────────────────────────────\n", flush=True)

    payload = {}
    for li in ALL_LAYERS:
        n_it = it_counts[li]; n_pt = pt_counts[li]
        if n_it == 0 or n_pt == 0:
            print(f"  WARNING: layer {li} zero tokens, skipping", flush=True)
            continue
        mean_it = it_sums[li] / n_it
        mean_pt = pt_sums[li] / n_pt
        vec  = mean_it - mean_pt
        norm = float(np.linalg.norm(vec))
        payload[f"layer_{li}"] = (vec / (norm + 1e-12)).astype(np.float32)
        print(f"  layer {li:2d}: raw diff norm={norm:.4f}", flush=True)

    npz_path = OUT_DIR / "corrective_directions.npz"
    np.savez_compressed(npz_path, **payload)

    selected_ids = json.loads(open(WORK_DIR / "selected.json").read())
    save_json(OUT_DIR / "corrective_directions.meta.json", {
        "pt_model_id":     "google/gemma-3-4b-pt",
        "it_model_id":     "google/gemma-3-4b-it",
        "dataset_path":    "data/eval_dataset_v2.jsonl",
        "layers":          ALL_LAYERS,
        "layer_groups": {
            "corrective": list(range(20, 34)),
            "mid":        list(range(12, 20)),
            "early":      list(range(1,  12)),
        },
        "token_positions":  "generated_only (min(IT_gen_len, PT_gen_len, 80) per record)",
        "chat_template":    False,
        "n_total_records":    1400,
        "n_selected_records": total_recs,
        "tokens_per_record":  {"min": min_k, "max": max_k, "mean": round(mean_k, 1)},
        "total_token_pairs_per_layer": total_tok,
        "contrast_signal":  "G1_judge + STR_contrast + norm_PT_NLL(IT_output)",
        "note": ("All layer groups (1-11, 12-19, 20-33) trained on SAME records and tokens. "
                 "A1/A1_early/A1_mid each reference this single file; their ablation_layers "
                 "config determines which layer subset receives the intervention."),
        "total_it_tokens": {str(li): int(it_counts[li]) for li in ALL_LAYERS},
        "total_pt_tokens": {str(li): int(pt_counts[li]) for li in ALL_LAYERS},
        "direction_formula": "normalize(mean_IT_gen_acts - mean_PT_gen_acts) over high-contrast records",
    })
    print(f"[merge] saved → {npz_path}  (layers 1-33, single shared dataset)", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["gen", "score", "acts", "merge"], required=True)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers",    type=int, default=8)
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--top-n",        type=int, default=TOP_N)
    p.add_argument("--judge-workers",type=int, default=16)
    args = p.parse_args()

    match args.phase:
        case "gen":
            phase_gen(args.worker_index, args.n_workers, args.device)
        case "score":
            phase_score(top_n=args.top_n, judge_workers=args.judge_workers)
        case "acts":
            phase_acts(args.worker_index, args.n_workers, args.device)
        case "merge":
            phase_merge()


if __name__ == "__main__":
    main()
