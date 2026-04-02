#!/usr/bin/env python3
"""Precompute corrective directions — multi-model version.

Generalised from precompute_directions_v2.py to work with all 6 model families
in MODEL_REGISTRY.  Architecture-specific layer paths resolved via SteeringAdapter.

Pipeline (same 4-phase design as v2):

  Phase 1 — gen   (GPU, N parallel workers)
    Generate IT and PT texts (no chat template, raw format B).
    Compute STR and PT_NLL(IT_output).
    Output: work_dir/gen/w{wi}.jsonl

  Phase 2 — score (CPU, no GPU)
    Merge worker gen files. Run LLM judge G1. Compute contrast.
    Select top-600 high-contrast records.
    Output: work_dir/selected.json

  Phase 3 — acts  (GPU, N parallel workers)
    Hook ALL layers simultaneously, capture MLP output activations at
    generated token positions (shape[1]==1 with use_cache=True).
    Output: work_dir/acts/w{wi}.npz

  Phase 4 — merge (CPU, no GPU)
    Sum worker act files, compute normalised direction per layer.
    Output: out_dir/corrective_directions.npz + meta.json

Usage:
  python scripts/precompute_directions_multimodel.py --model-name llama31_8b --phase gen  --worker-index 0 --n-workers 2 --device cuda:0
  python scripts/precompute_directions_multimodel.py --model-name llama31_8b --phase score
  python scripts/precompute_directions_multimodel.py --model-name llama31_8b --phase acts --worker-index 0 --n-workers 2 --device cuda:0
  python scripts/precompute_directions_multimodel.py --model-name llama31_8b --phase merge
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.poc.collect import load_dataset_records
from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.exp6.model_adapter import get_steering_adapter

# ── Config (model-independent defaults) ──────────────────────────────────────
TOP_N      = 600     # high-contrast records for direction extraction
DATASET    = "data/eval_dataset_v2.jsonl"


def _work_dir(model_name: str) -> Path:
    return Path(f"results/cross_model/{model_name}/directions_work")


def _out_dir(model_name: str) -> Path:
    return Path(f"results/cross_model/{model_name}/directions")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Helpers ──────────────────────────────────────────────────────────────────

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
        from src.poc.exp3.analysis.word_categories import classify_generated_tokens_by_word
        words = text.split()
        if not words:
            return 0.0
        cats = classify_generated_tokens_by_word([{"token_str": w + " "} for w in words])
        gov = {"STRUCTURAL", "DISCOURSE", "PUNCTUATION"}
        return sum(1 for c in cats if c in gov) / len(cats)
    except Exception:
        return 0.0


def _load_model_raw(model_id: str, device: str):
    """Load raw HF model + tokenizer (no nnsight, no transcoders)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16
    print(f"  Loading {model_id} on {device} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _generate_batch(model_raw, tokenizer, adapter, prompts, device, max_new_tokens):
    """Generate from raw prompts (no chat template). Returns list of (text, token_ids)."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]
    eos_ids = adapter.eos_token_ids(tokenizer)
    with torch.no_grad():
        out = model_raw.generate(
            input_ids, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    results = []
    stop_ids = set(eos_ids) | {tokenizer.pad_token_id}
    for i in range(len(prompts)):
        ids = [t for t in out[i, prompt_len:].tolist() if t not in stop_ids]
        results.append((tokenizer.decode(ids, skip_special_tokens=True), ids))
    return results


def _compute_nll(model_raw, tokenizer, prompts, gen_ids_list, device):
    """PT NLL on IT's generated tokens (teacher-forcing)."""
    nlls = []
    for prompt, gen_ids in zip(prompts, gen_ids_list):
        if not gen_ids:
            nlls.append(0.0)
            continue
        p_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        g_ids = torch.tensor(gen_ids, dtype=torch.long, device=device).unsqueeze(0)
        full = torch.cat([p_ids, g_ids], dim=1)
        pl = p_ids.shape[1]
        with torch.no_grad():
            logits = model_raw(full).logits
        lp = torch.nn.functional.log_softmax(logits[0, pl-1:-1], dim=-1)
        nll = -lp[range(len(gen_ids)), gen_ids].float().cpu().mean().item()
        nlls.append(nll)
    return nlls


# ── Phase 1: gen ─────────────────────────────────────────────────────────────

def phase_gen(model_name: str, worker_index: int, n_workers: int, device: str) -> None:
    spec = get_spec(model_name)
    adapter = get_steering_adapter(model_name)
    max_gen = adapter.max_gen_tokens

    wdir = _ensure_dir(_work_dir(model_name) / "gen")
    out_path = wdir / f"w{worker_index}.jsonl"
    if out_path.exists():
        print(f"[gen w{worker_index}] already done, skipping.", flush=True)
        return

    records = load_dataset_records(DATASET, prompt_format="B")
    my_records = records[worker_index::n_workers]
    prompts = [r["formats"]["B"] for r in my_records]
    print(f"[gen w{worker_index}/{n_workers}] {len(my_records)} records on {device}, "
          f"model={model_name}, max_gen={max_gen}", flush=True)

    BS = 8

    # ── PT generation ─────────────────────────────────────────────────────
    pt_model_id = model_id_for_variant(spec, "pt")
    pt_raw, tokenizer = _load_model_raw(pt_model_id, device)

    pt_texts, pt_ids_all = [], []
    for i in range(0, len(prompts), BS):
        batch = prompts[i:i+BS]
        for text, ids in _generate_batch(pt_raw, tokenizer, adapter, batch, device, max_gen):
            pt_texts.append(text)
            pt_ids_all.append(ids)
        if (i // BS + 1) % 10 == 0:
            print(f"  PT gen {min(i+BS, len(prompts))}/{len(prompts)}", flush=True)
    del pt_raw; torch.cuda.empty_cache()

    # ── IT generation ─────────────────────────────────────────────────────
    it_model_id = model_id_for_variant(spec, "it")
    it_raw, tokenizer = _load_model_raw(it_model_id, device)

    it_texts, it_ids_all = [], []
    for i in range(0, len(prompts), BS):
        batch = prompts[i:i+BS]
        for text, ids in _generate_batch(it_raw, tokenizer, adapter, batch, device, max_gen):
            it_texts.append(text)
            it_ids_all.append(ids)
        if (i // BS + 1) % 10 == 0:
            print(f"  IT gen {min(i+BS, len(prompts))}/{len(prompts)}", flush=True)
    del it_raw; torch.cuda.empty_cache()

    # ── PT NLL on IT outputs ──────────────────────────────────────────────
    pt_raw2, _ = _load_model_raw(pt_model_id, device)
    nlls = []
    for i in range(0, len(prompts), BS):
        bp = prompts[i:i+BS]
        bi = it_ids_all[i:i+BS]
        nlls.extend(_compute_nll(pt_raw2, tokenizer, bp, bi, device))
    del pt_raw2; torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────────
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

    print(f"[gen w{worker_index}] saved {len(my_records)} records -> {out_path}", flush=True)


# ── Phase 2: score ───────────────────────────────────────────────────────────

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
    """Run G1 judge on IT and PT texts."""
    _load_dotenv()
    from openai import OpenAI
    from src.poc.exp6.eval_registry import RUBRICS
    rubric = RUBRICS["g1"]

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  WARNING: no OPENROUTER_API_KEY -- skipping LLM judge, using STR+NLL only", flush=True)
        return {}

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    model = "google/gemini-2.5-flash"

    tasks = []
    for row in rows:
        q = row["prompt"][:500]
        tasks.append(("it", row["record_id"], rubric.format(question=q, response=row["it_text"][:1200])))
        tasks.append(("pt", row["record_id"], rubric.format(question=q, response=row["pt_text"][:1200])))

    print(f"  Running G1 judge on {len(tasks)} outputs ({len(rows)} records x IT+PT)...", flush=True)
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


def phase_score(model_name: str, top_n: int = TOP_N, judge_workers: int = 16) -> None:
    work = _work_dir(model_name)
    sel_path = work / "selected.json"
    if sel_path.exists():
        print("[score] selected.json already exists, skipping.", flush=True)
        return

    gen_dir = work / "gen"
    rows = []
    for f in sorted(gen_dir.glob("w*.jsonl")):
        for line in open(f):
            rows.append(json.loads(line))
    print(f"[score] merged {len(rows)} records from {len(list(gen_dir.glob('w*.jsonl')))} workers", flush=True)

    g1_scores = _run_llm_judge_g1(rows, judge_workers=judge_workers)

    nlls = [r["pt_nll"] for r in rows]
    max_nll = max(nlls) if max(nlls) > 0 else 1.0

    scored = []
    for r in rows:
        rid = r["record_id"]
        str_contrast = r["it_str"] - r["pt_str"]
        norm_nll = r["pt_nll"] / max_nll
        g1_contrast = 0.0
        if rid in g1_scores:
            g1_it = g1_scores[rid].get("it", -1)
            g1_pt = g1_scores[rid].get("pt", -1)
            if g1_it >= 0 and g1_pt >= 0:
                g1_contrast = (g1_it - g1_pt) / 4.0
        min_k = min(len(r["it_gen_ids"]), len(r["pt_gen_ids"]))
        contrast = str_contrast + norm_nll + g1_contrast
        scored.append((contrast, rid, min_k))

    valid = [(c, rid, k) for c, rid, k in scored if k >= 5]
    valid.sort(reverse=True)
    selected = [rid for _, rid, _ in valid[:top_n]]

    print(f"[score] selected {len(selected)}/{len(rows)} records", flush=True)
    if valid:
        top_scores = [c for c, _, _ in valid[:top_n]]
        print(f"  contrast range: {top_scores[0]:.3f} -> {top_scores[-1]:.3f}", flush=True)

    with open(work / "gen_merged.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(sel_path, "w") as f:
        json.dump(selected, f)
    print(f"[score] saved -> {sel_path}", flush=True)


# ── Phase 3: acts ────────────────────────────────────────────────────────────

def phase_acts(model_name: str, worker_index: int, n_workers: int, device: str) -> None:
    spec = get_spec(model_name)
    adapter = get_steering_adapter(model_name)
    all_layers = list(range(1, spec.n_layers))  # layers 1 to n_layers-1
    d_model = spec.d_model
    max_gen = adapter.max_gen_tokens
    work = _work_dir(model_name)

    out_path = _ensure_dir(work / "acts") / f"w{worker_index}.npz"
    if out_path.exists():
        print(f"[acts w{worker_index}] already done, skipping.", flush=True)
        return

    selected_ids = set(json.loads(open(work / "selected.json").read()))
    rows_by_id = {}
    for line in open(work / "gen_merged.jsonl"):
        r = json.loads(line)
        if r["record_id"] in selected_ids:
            rows_by_id[r["record_id"]] = r

    my_ids = sorted(rows_by_id.keys())[worker_index::n_workers]
    my_rows = [rows_by_id[rid] for rid in my_ids]
    print(f"[acts w{worker_index}/{n_workers}] {len(my_rows)} records on {device} "
          f"-- hooking ALL layers {all_layers[0]}-{all_layers[-1]} ({len(all_layers)} total)",
          flush=True)

    def collect_acts(model_raw, tokenizer, rows_subset, is_it: bool):
        sums = {li: torch.zeros(d_model, dtype=torch.float64) for li in all_layers}
        counts = {li: 0 for li in all_layers}

        for idx, row in enumerate(rows_subset):
            k = min(len(row["it_gen_ids"]), len(row["pt_gen_ids"]), max_gen)
            if k == 0:
                continue

            gen_acts: dict[int, list] = {li: [] for li in all_layers}

            def make_hook(li: int):
                def hook(mod, inp, out):
                    if out.shape[1] == 1 and len(gen_acts[li]) < k:
                        gen_acts[li].append(out[0, 0, :].float().cpu())
                return hook

            # Register hooks on ALL layers' MLP modules
            layers_list = adapter.get_layers(model_raw)
            handles = [
                layers_list[li].mlp.register_forward_hook(make_hook(li))
                for li in all_layers
            ]
            try:
                input_ids = tokenizer.encode(row["prompt"], return_tensors="pt").to(device)
                eos_ids = adapter.eos_token_ids(tokenizer)
                with torch.no_grad():
                    model_raw.generate(
                        input_ids, max_new_tokens=k, do_sample=False,
                        eos_token_id=eos_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
            finally:
                for h in handles:
                    h.remove()

            for li in all_layers:
                if gen_acts[li]:
                    stacked = torch.stack(gen_acts[li])
                    sums[li] += stacked.sum(dim=0).to(torch.float64)
                    counts[li] += stacked.shape[0]

            if (idx + 1) % 50 == 0:
                label = "IT" if is_it else "PT"
                rep_layer = all_layers[len(all_layers) // 2]
                print(f"  {label} acts {idx+1}/{len(rows_subset)}, "
                      f"layer {rep_layer} tokens: {counts[rep_layer]}", flush=True)

        return sums, counts

    # IT activations
    it_model_id = model_id_for_variant(spec, "it")
    print(f"[acts w{worker_index}] loading IT ({it_model_id})...", flush=True)
    it_raw, tokenizer = _load_model_raw(it_model_id, device)
    it_sums, it_counts = collect_acts(it_raw, tokenizer, my_rows, is_it=True)
    del it_raw; torch.cuda.empty_cache()

    # PT activations
    pt_model_id = model_id_for_variant(spec, "pt")
    print(f"[acts w{worker_index}] loading PT ({pt_model_id})...", flush=True)
    pt_raw, tokenizer = _load_model_raw(pt_model_id, device)
    pt_sums, pt_counts = collect_acts(pt_raw, tokenizer, my_rows, is_it=False)
    del pt_raw; torch.cuda.empty_cache()

    per_record_k = [
        min(len(row["it_gen_ids"]), len(row["pt_gen_ids"]), max_gen)
        for row in my_rows
    ]

    payload = {}
    for li in all_layers:
        payload[f"it_sum_{li}"] = it_sums[li].to(torch.float32).numpy()
        payload[f"pt_sum_{li}"] = pt_sums[li].to(torch.float32).numpy()
        payload[f"it_count_{li}"] = np.array(it_counts[li], dtype=np.int64)
        payload[f"pt_count_{li}"] = np.array(pt_counts[li], dtype=np.int64)
    payload["n_records"] = np.array(len(my_rows), dtype=np.int64)
    payload["per_record_k"] = np.array(per_record_k, dtype=np.int32)
    np.savez_compressed(out_path, **payload)

    mean_k = sum(per_record_k) / max(len(per_record_k), 1)
    print(f"[acts w{worker_index}] saved -> {out_path} "
          f"(mean k={mean_k:.1f} tokens/record, layers {all_layers[0]}-{all_layers[-1]})",
          flush=True)


# ── Phase 4: merge ───────────────────────────────────────────────────────────

def phase_merge(model_name: str) -> None:
    spec = get_spec(model_name)
    all_layers = list(range(1, spec.n_layers))
    d_model = spec.d_model
    work = _work_dir(model_name)
    out = _ensure_dir(_out_dir(model_name))

    act_files = sorted((work / "acts").glob("w*.npz"))
    print(f"[merge] combining {len(act_files)} worker files (layers 1-{spec.n_layers-1})...", flush=True)

    it_sums = {li: np.zeros(d_model, dtype=np.float64) for li in all_layers}
    pt_sums = {li: np.zeros(d_model, dtype=np.float64) for li in all_layers}
    it_counts = {li: 0 for li in all_layers}
    pt_counts = {li: 0 for li in all_layers}
    total_recs = 0
    all_per_record_k: list[int] = []

    for f in act_files:
        with np.load(f) as d:
            for li in all_layers:
                it_sums[li] += d[f"it_sum_{li}"].astype(np.float64)
                pt_sums[li] += d[f"pt_sum_{li}"].astype(np.float64)
                it_counts[li] += int(d[f"it_count_{li}"])
                pt_counts[li] += int(d[f"pt_count_{li}"])
            total_recs += int(d["n_records"])
            if "per_record_k" in d:
                all_per_record_k.extend(d["per_record_k"].tolist())

    rep_layer = all_layers[len(all_layers) // 2]
    total_tok = it_counts[rep_layer]
    mean_k = total_tok / max(total_recs, 1)
    min_k = min(all_per_record_k) if all_per_record_k else 0
    max_k = max(all_per_record_k) if all_per_record_k else 0

    print(f"\n[merge] -- Sample accounting --", flush=True)
    print(f"  Model:                              {model_name}", flush=True)
    print(f"  Records selected (high-contrast):   {total_recs}", flush=True)
    print(f"  Tokens per record:                  min={min_k}  max={max_k}  mean={mean_k:.1f}", flush=True)
    print(f"  Total token-pair obs per layer:     {total_tok:,}", flush=True)

    payload = {}
    for li in all_layers:
        n_it = it_counts[li]
        n_pt = pt_counts[li]
        if n_it == 0 or n_pt == 0:
            print(f"  WARNING: layer {li} zero tokens, skipping", flush=True)
            continue
        mean_it = it_sums[li] / n_it
        mean_pt = pt_sums[li] / n_pt
        vec = mean_it - mean_pt
        norm = float(np.linalg.norm(vec))
        payload[f"layer_{li}"] = (vec / (norm + 1e-12)).astype(np.float32)
        print(f"  layer {li:2d}: raw diff norm={norm:.4f}", flush=True)

    npz_path = out / "corrective_directions.npz"
    np.savez_compressed(npz_path, **payload)

    corrective_onset = spec.corrective_onset
    meta = {
        "model_name": model_name,
        "pt_model_id": spec.pt_id,
        "it_model_id": spec.it_id,
        "dataset_path": DATASET,
        "layers": all_layers,
        "n_layers": spec.n_layers,
        "d_model": spec.d_model,
        "layer_groups": {
            "corrective": list(range(corrective_onset, spec.n_layers)),
            "mid": list(range(spec.phase_boundary, corrective_onset)),
            "early": list(range(1, spec.phase_boundary)),
        },
        "token_positions": f"generated_only (min(IT_gen_len, PT_gen_len, {get_steering_adapter(model_name).max_gen_tokens}) per record)",
        "chat_template": False,
        "n_total_records": 1400,
        "n_selected_records": total_recs,
        "tokens_per_record": {"min": min_k, "max": max_k, "mean": round(mean_k, 1)},
        "total_token_pairs_per_layer": total_tok,
        "contrast_signal": "G1_judge + STR_contrast + norm_PT_NLL(IT_output)",
        "direction_formula": "normalize(mean_IT_gen_acts - mean_PT_gen_acts) over high-contrast records",
    }

    meta_path = out / "corrective_directions.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[merge] saved -> {npz_path}  ({len(payload)} layers)", flush=True)
    print(f"[merge] metadata -> {meta_path}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Precompute corrective directions (multi-model)")
    p.add_argument("--model-name", required=True, choices=list(MODEL_REGISTRY.keys()),
                   help="Model family from MODEL_REGISTRY")
    p.add_argument("--phase", choices=["gen", "score", "acts", "merge"], required=True)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--judge-workers", type=int, default=16)
    args = p.parse_args()

    match args.phase:
        case "gen":
            phase_gen(args.model_name, args.worker_index, args.n_workers, args.device)
        case "score":
            phase_score(args.model_name, top_n=args.top_n, judge_workers=args.judge_workers)
        case "acts":
            phase_acts(args.model_name, args.worker_index, args.n_workers, args.device)
        case "merge":
            phase_merge(args.model_name)


if __name__ == "__main__":
    main()
