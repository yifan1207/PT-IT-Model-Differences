"""Main orchestrator for Exp6 steering experiments.

Usage (Approach A):
    python src/poc/exp6/run.py \\
        --experiment A1 --variant it --device cuda:0 \\
        --corrective-direction-path results/exp5/precompute_it/precompute/corrective_directions.npz \\
        --worker-index 0 --n-workers 8

Usage (Approach B):
    python src/poc/exp6/run.py \\
        --experiment B1 --variant it --device cuda:0 \\
        --governance-features-path results/exp6/governance_feature_sets.json \\
        --mean-feature-acts-path results/exp6/precompute/mean_feature_acts_it \\
        --feature-set method12_top100 --gamma 5.0 \\
        --worker-index 0 --n-workers 8
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.poc.collect import load_dataset_records
from src.poc.exp6.benchmarks.governance import evaluate_governance_benchmark
from src.poc.exp6.config import Exp6Config
from src.poc.exp6.interventions import build_intervention
from src.poc.exp6.runtime import GeneratedSample6, generate_record_A, generate_record_B, generate_records_A_batch, generate_records_B_batch
from src.poc.exp5.benchmarks.custom import evaluate_custom_benchmark
from src.poc.exp5.utils import ensure_dir, sanitise_json, save_json
from src.poc.shared.model import load_model

_DISK_WARN_GB = 5.0
_DISK_STOP_GB = 2.0
_GOV_BENCHMARKS = {"structural_token_ratio", "turn_structure", "format_compliance"}
_EXP5_BENCHMARKS = {"factual_em": "exp3_factual_em", "reasoning_em": "exp3_reasoning_em",
                     "alignment_behavior": "exp3_alignment_behavior"}


def _free_gb() -> float:
    return shutil.disk_usage(".").free / 1e9


def _check_disk() -> None:
    free = _free_gb()
    if free < _DISK_STOP_GB:
        raise RuntimeError(f"[disk] only {free:.1f} GB free — stopping to prevent data loss.")
    if free < _DISK_WARN_GB:
        print(f"[disk] WARNING: {free:.1f} GB free", flush=True)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitise_json(row), ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Condition generators ──────────────────────────────────────────────────────

def _A1_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A1: Remove corrective direction from IT — α sweep."""
    ALPHA_VALUES = [5.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]
    specs = [("A1_baseline", replace(cfg, method="none", directional_alpha=1.0))]
    for alpha in ALPHA_VALUES:
        name = f"A1_alpha_{alpha:g}"
        specs.append((name, replace(cfg,
            method="directional_remove",
            ablation_layers=list(range(cfg.proposal_boundary, cfg.n_layers)),
            directional_alpha=alpha,
        )))
    return specs  # 14 conditions


def _A2_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """A2: Inject corrective direction into PT — β sweep + controls."""
    BETA_VALUES = [-5.0, -3.0, -2.0, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    CTRL_BETAS = [0.0, 1.0, 3.0]

    specs = [("A2_baseline_pt", replace(cfg, method="none", directional_beta=0.0))]

    # Main β sweep with corrective direction
    for beta in BETA_VALUES:
        name = f"A2_beta_{beta:g}"
        specs.append((name, replace(cfg,
            method="directional_add",
            ablation_layers=list(range(cfg.proposal_boundary, cfg.n_layers)),
            directional_beta=beta,
        )))

    # Control: random unit vector
    for beta in CTRL_BETAS:
        name = f"A2_ctrl_random_b{beta:g}"
        specs.append((name, replace(cfg,
            method="directional_random",
            ablation_layers=list(range(cfg.proposal_boundary, cfg.n_layers)),
            directional_beta=beta,
        )))

    # Control: rotated direction (orthogonal to corrective)
    for beta in CTRL_BETAS:
        name = f"A2_ctrl_rotated_b{beta:g}"
        specs.append((name, replace(cfg,
            method="directional_rotated",
            ablation_layers=list(range(cfg.proposal_boundary, cfg.n_layers)),
            directional_beta=beta,
        )))

    # Control: content-layer direction injected at corrective layers
    for beta in CTRL_BETAS:
        name = f"A2_ctrl_content_b{beta:g}"
        specs.append((name, replace(cfg,
            method="content_direction",
            ablation_layers=list(range(cfg.proposal_boundary, cfg.n_layers)),
            directional_beta=beta,
        )))

    return specs  # 1 + 15 + 9 = 25 conditions


def _B1_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """B1: Feature clamping — γ sweep × feature set sweep."""
    GAMMA_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    # Method 1+2 combined feature sets
    M12_SETS = ["method12_top10", "method12_top50", "method12_top100", "method12_top500", "method12_all"]
    # Method 3: IT-amplified top-100
    M3_SETS = ["method3_it_amplified_top100"]

    specs = [("B1_baseline", replace(cfg, method="none", gamma=1.0))]
    for fset in M12_SETS + M3_SETS:
        for gamma in GAMMA_VALUES:
            name = f"B1_{fset}_g{gamma:g}"
            specs.append((name, replace(cfg, method="feature_clamp", gamma=gamma, feature_set=fset)))
    return specs


def _B2_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """B2: W_dec governance direction injection."""
    BETA_VALUES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    specs = [("B2_baseline", replace(cfg, method="none"))]
    for fset in ["method12_top100", "method3_it_amplified_top100"]:
        for beta in BETA_VALUES:
            # B2a: inject into PT
            specs.append((f"B2a_{fset}_b{beta:g}", replace(cfg,
                model_variant="pt",
                method="wdec_inject",
                directional_beta=beta,
                feature_set=fset,
            )))
        for beta in BETA_VALUES:
            # B2b: subtract from IT
            specs.append((f"B2b_{fset}_b{-beta:g}", replace(cfg,
                model_variant="it",
                method="wdec_inject",
                directional_beta=-beta,
                feature_set=fset,
            )))
    return specs


def _B3_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """B3: Control feature sets."""
    GAMMA_VALUES = [0.0, 1.0, 5.0]
    CTRL_SETS = ["random_100", "method12_content_top100", "method3_it_suppressed_top100"]
    specs = []
    for fset in CTRL_SETS:
        for gamma in GAMMA_VALUES:
            specs.append((f"B3_{fset}_g{gamma:g}", replace(cfg,
                method="feature_clamp", gamma=gamma, feature_set=fset,
            )))
    return specs


def _B4_conditions(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    """B4: Layer specificity — early/mid/late corrective sub-ranges."""
    GAMMA_VALUES = [0.0, 5.0]
    specs = []
    for layer_range in ["early_20_25", "mid_26_29", "late_30_33"]:
        for gamma in GAMMA_VALUES:
            specs.append((f"B4_{layer_range}_g{gamma:g}", replace(cfg,
                method="feature_clamp", gamma=gamma,
                feature_set="method12_top100",
                feature_layer_range=layer_range,
            )))
    return specs


def _condition_specs(cfg: Exp6Config) -> list[tuple[str, Exp6Config]]:
    match cfg.experiment:
        case "A1": return _A1_conditions(cfg)
        case "A2": return _A2_conditions(cfg)
        case "B1": return _B1_conditions(cfg)
        case "B2": return _B2_conditions(cfg)
        case "B3": return _B3_conditions(cfg)
        case "B4": return _B4_conditions(cfg)
        case _: raise ValueError(f"Unknown experiment: {cfg.experiment!r}")


# ── Record selection ──────────────────────────────────────────────────────────

def _select_records(records: list[dict], benchmark: str) -> list[dict]:
    """Filter records to those relevant for the given benchmark."""
    if benchmark in ("factual_em",):
        return [r for r in records if r.get("category") == "CONTENT-FACT"]
    if benchmark in ("reasoning_em",):
        return [r for r in records if r.get("category") == "CONTENT-REASON"]
    if benchmark == "alignment_behavior":
        return [r for r in records if r.get("category") == "SAFETY"]
    if benchmark == "format_compliance":
        return [r for r in records if r.get("category") == "GOV-FORMAT"]
    # structural_token_ratio and turn_structure: use all records
    return records


# ── Scoring dispatch ──────────────────────────────────────────────────────────

def _score_benchmark(
    benchmark: str,
    records: list[dict],
    outputs: list[GeneratedSample6],
) -> Any:
    if benchmark in _GOV_BENCHMARKS:
        return evaluate_governance_benchmark(benchmark, records, outputs)
    # Map exp6 benchmark names to exp5 names
    exp5_name = _EXP5_BENCHMARKS.get(benchmark, benchmark)
    # exp5 scorers expect GeneratedSample objects — adapt GeneratedSample6
    from src.poc.exp5.runtime import GeneratedSample as GS5
    adapted = [GS5(
        record_id=o.record_id, prompt=o.prompt,
        generated_text=o.generated_text, generated_tokens=o.generated_tokens,
        hidden_states={}, logit_lens_entropy=[], top1_token_per_layer=[],
    ) for o in outputs]
    return evaluate_custom_benchmark(exp5_name, records, adapted)


# ── Per-condition runner ──────────────────────────────────────────────────────

def _run_condition_A(
    name: str,
    cfg: Exp6Config,
    records: list[dict],
    loaded: Any,
    scores_jsonl: Path,
    samples_jsonl: Path,
    done_benchmarks: set[str],
) -> list[dict]:
    """Run one A-experiment condition (nnsight trace).

    Benchmarks that use all records (structural_token_ratio, turn_structure) share a
    single generation pass to avoid redundant inference.
    """
    intervention = build_intervention(cfg)
    score_rows: list[dict] = []

    # Pre-generate outputs for all-records benchmarks once (shared across structural benchmarks)
    _ALL_RECORDS_BENCHMARKS = {"structural_token_ratio", "turn_structure"}
    all_records_benchmarks_todo = [b for b in cfg.benchmarks
                                   if b in _ALL_RECORDS_BENCHMARKS and b not in done_benchmarks]
    cached_all_outputs: list[GeneratedSample6] | None = None

    BATCH_SIZE = 8

    if all_records_benchmarks_todo:
        print(f"[exp6] {name}: generating all-records outputs (batch={BATCH_SIZE}, shared for {all_records_benchmarks_todo})", flush=True)
        cached_all_outputs = []
        for batch_start in range(0, len(records), BATCH_SIZE):
            batch = records[batch_start: batch_start + BATCH_SIZE]
            cached_all_outputs.extend(generate_records_A_batch(batch, loaded, cfg, intervention, batch_size=BATCH_SIZE))
            done_so_far = min(batch_start + BATCH_SIZE, len(records))
            if done_so_far % 100 < BATCH_SIZE or done_so_far == len(records):
                _check_disk()
                print(f"[exp6] {name}/all_records: {done_so_far}/{len(records)} done", flush=True)

    for benchmark in cfg.benchmarks:
        if benchmark in done_benchmarks:
            print(f"[exp6] {name}/{benchmark}: already done, skipping", flush=True)
            continue

        if benchmark in _ALL_RECORDS_BENCHMARKS and cached_all_outputs is not None:
            bench_records = records
            outputs = cached_all_outputs
        else:
            bench_records = _select_records(records, benchmark)
            outputs = []
            for batch_start in range(0, len(bench_records), BATCH_SIZE):
                batch = bench_records[batch_start: batch_start + BATCH_SIZE]
                outputs.extend(generate_records_A_batch(batch, loaded, cfg, intervention, batch_size=BATCH_SIZE))
                done_so_far = min(batch_start + BATCH_SIZE, len(bench_records))
                if done_so_far % 100 < BATCH_SIZE or done_so_far == len(bench_records):
                    _check_disk()
                    print(f"[exp6] {name}/{benchmark}: {done_so_far}/{len(bench_records)} done", flush=True)

        result = _score_benchmark(benchmark, bench_records, outputs)
        row = {
            "condition": name,
            "benchmark": result.benchmark,
            "metric": result.metric,
            "value": result.value,
            "n": result.n,
            "experiment": cfg.experiment,
            "method": cfg.method,
            "alpha": cfg.directional_alpha,
            "beta": cfg.directional_beta,
            "layers": cfg.ablation_layers,
        }
        _append_jsonl(scores_jsonl, [row])
        _append_jsonl(samples_jsonl, [{
            "condition": name, "benchmark": benchmark,
            "record_id": o.record_id, "prompt": o.prompt,
            "generated_text": o.generated_text, "category": o.category,
        } for o in outputs])
        score_rows.append(row)
        print(f"[exp6] {name}/{benchmark}: value={result.value:.4f} n={result.n}", flush=True)

    return score_rows


def _get_raw_model(loaded: Any) -> Any:
    """Extract the underlying HuggingFace nn.Module from the nnsight/circuit-tracer wrapper."""
    # nnsight stores the wrapped nn.Module as ._model on both LanguageModel and ReplacementModel.
    # Verify the attribute exists to catch version changes early.
    if not hasattr(loaded.model, "_model"):
        raise AttributeError(
            "loaded.model has no ._model attribute — nnsight API may have changed. "
            "Inspect loaded.model.__dict__ and update _get_raw_model()."
        )
    return loaded.model._model


BATCH_SIZE_B = 8


def _run_condition_B(
    name: str,
    cfg: Exp6Config,
    records: list[dict],
    loaded: Any,
    scores_jsonl: Path,
    samples_jsonl: Path,
    done_benchmarks: set[str],
    hooks_config: dict,
) -> list[dict]:
    """Run one B-experiment condition (batched forward hooks, partial-save for crash resilience)."""
    model_raw = _get_raw_model(loaded)
    tokenizer = loaded.tokenizer
    real_token_mask = loaded.real_token_mask
    score_rows: list[dict] = []

    _ALL_RECORDS_BENCHMARKS = {"structural_token_ratio", "turn_structure"}
    all_records_benchmarks_todo = [b for b in cfg.benchmarks
                                   if b in _ALL_RECORDS_BENCHMARKS and b not in done_benchmarks]
    cached_all_outputs: list[GeneratedSample6] | None = None

    if all_records_benchmarks_todo:
        # Partial-save path — resume across crashes
        partial_path = scores_jsonl.parent / f"_partial_{name}_all_records.jsonl"
        already_done: dict[str, GeneratedSample6] = {}
        if partial_path.exists():
            for line in partial_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    already_done[d["record_id"]] = GeneratedSample6(
                        record_id=d["record_id"], prompt=d["prompt"],
                        generated_text=d["generated_text"],
                        generated_tokens=d.get("generated_tokens", []),
                        category=d.get("category", ""),
                    )
                except Exception:
                    pass

        remaining = [r for r in records if r["id"] not in already_done]
        print(
            f"[exp6] {name}: generating all-records outputs (batch={BATCH_SIZE_B}, "
            f"shared for {all_records_benchmarks_todo}, "
            f"{len(already_done)} resumed, {len(remaining)} to go)",
            flush=True,
        )

        for batch_start in range(0, len(remaining), BATCH_SIZE_B):
            batch = remaining[batch_start: batch_start + BATCH_SIZE_B]
            new_outputs = generate_records_B_batch(
                batch, model_raw, tokenizer, real_token_mask, cfg, hooks_config,
                batch_size=BATCH_SIZE_B,
            )
            for o in new_outputs:
                already_done[o.record_id] = o
            # Append to partial file immediately
            with open(partial_path, "a") as pf:
                for o in new_outputs:
                    pf.write(json.dumps({
                        "record_id": o.record_id, "prompt": o.prompt,
                        "generated_text": o.generated_text,
                        "generated_tokens": o.generated_tokens,
                        "category": o.category,
                    }) + "\n")
            done_so_far = min(batch_start + BATCH_SIZE_B, len(remaining)) + len(records) - len(remaining)
            if done_so_far % 100 < BATCH_SIZE_B or done_so_far >= len(records):
                _check_disk()
                print(f"[exp6] {name}/all_records: {done_so_far}/{len(records)} done", flush=True)

        # Reconstruct in original record order
        cached_all_outputs = [already_done[r["id"]] for r in records if r["id"] in already_done]
        # Clean up partial file once fully done
        if partial_path.exists():
            partial_path.unlink()

    for benchmark in cfg.benchmarks:
        if benchmark in done_benchmarks:
            print(f"[exp6] {name}/{benchmark}: already done, skipping", flush=True)
            continue

        if benchmark in _ALL_RECORDS_BENCHMARKS and cached_all_outputs is not None:
            bench_records = records
            outputs = cached_all_outputs
        else:
            bench_records = _select_records(records, benchmark)
            outputs = generate_records_B_batch(
                bench_records, model_raw, tokenizer, real_token_mask, cfg, hooks_config,
                batch_size=BATCH_SIZE_B,
            )
            print(f"[exp6] {name}/{benchmark}: {len(outputs)}/{len(bench_records)} done", flush=True)

        result = _score_benchmark(benchmark, bench_records, outputs)
        row = {
            "condition": name,
            "benchmark": result.benchmark,
            "metric": result.metric,
            "value": result.value,
            "n": result.n,
            "experiment": cfg.experiment,
            "method": cfg.method,
            "gamma": cfg.gamma,
            "feature_set": cfg.feature_set,
            "feature_layer_range": cfg.feature_layer_range,
        }
        _append_jsonl(scores_jsonl, [row])
        _append_jsonl(samples_jsonl, [{
            "condition": name, "benchmark": benchmark,
            "record_id": o.record_id, "prompt": o.prompt,
            "generated_text": o.generated_text, "category": o.category,
        } for o in outputs])
        score_rows.append(row)
        print(f"[exp6] {name}/{benchmark}: value={result.value:.4f} n={result.n}", flush=True)

    return score_rows


# ── B-experiment hooks config loader ─────────────────────────────────────────

def _load_B_hooks_config(cfg: Exp6Config, loaded: Any) -> dict:
    """Load all feature-steering artifacts into a hooks_config dict."""
    config: dict = {}

    if cfg.method == "feature_clamp":
        # Transcoders: loaded.transcoder_list is list[transcoder] indexed by layer
        config["transcoders"] = loaded.transcoder_list

        # Governance feature sets
        feat_path = Path(cfg.governance_features_path)
        if not feat_path.exists():
            raise FileNotFoundError(f"Governance features not found: {feat_path}")
        with open(feat_path) as f:
            all_sets = json.load(f)
        # Build {layer_idx: [feature_indices]} from the chosen feature set key
        config["governance_features"] = {
            k: all_sets.get(k, {}).get(cfg.feature_set, [])
            for k in all_sets
        }

        # Mean feature activations ā[f]
        mean_acts_dir = Path(cfg.mean_feature_acts_path)
        mean_acts: dict[int, np.ndarray] = {}
        for l_idx in cfg.corrective_layers:
            npy_path = mean_acts_dir / f"layer_{l_idx}.npy"
            if npy_path.exists():
                mean_acts[l_idx] = np.load(str(npy_path))
        config["mean_acts"] = mean_acts

    elif cfg.method == "wdec_inject":
        import torch
        gov_dir_path = Path(cfg.governance_direction_path)
        if not gov_dir_path.exists():
            raise FileNotFoundError(f"Governance direction not found: {gov_dir_path}")
        gov_dirs: dict[int, torch.Tensor] = {}
        with np.load(str(gov_dir_path)) as d:
            for k in d.files:
                if k.startswith("layer_"):
                    l_idx = int(k.split("_", 1)[1])
                    gov_dirs[l_idx] = torch.tensor(d[k], dtype=torch.float32).to(cfg.device)
        config["governance_direction"] = gov_dirs

    return config


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Run exp6 steering experiments.")
    p.add_argument("--experiment", choices=["A1", "A2", "B1", "B2", "B3", "B4"], required=True)
    p.add_argument("--variant", choices=["pt", "it"], default="it")
    p.add_argument("--dataset", default="data/exp6_dataset.jsonl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-gen-tokens", type=int, default=200)
    p.add_argument("--n-eval-examples", type=int, default=1000)
    p.add_argument("--run-name", default="")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)

    # A-experiment paths
    p.add_argument("--corrective-direction-path", default="results/exp5/precompute_it/precompute/corrective_directions.npz")
    p.add_argument("--content-direction-path", default="results/exp6/precompute/content_direction_aggregate.npz")

    # B-experiment paths
    p.add_argument("--governance-features-path", default="results/exp6/governance_feature_sets.json")
    p.add_argument("--mean-feature-acts-path", default="results/exp6/precompute/mean_feature_acts_it")
    p.add_argument("--governance-direction-path", default="results/exp6/precompute/governance_directions.npz")
    p.add_argument("--feature-set", default="method12_top100")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--feature-layer-range",
                   choices=["all", "early_20_25", "mid_26_29", "late_30_33"], default="all")

    args = p.parse_args()

    worker_suffix = f"_w{args.worker_index}" if args.n_workers > 1 else ""
    run_name_base = args.run_name or f"{args.experiment}_{args.variant}"
    run_name = run_name_base + worker_suffix

    is_B = args.experiment.startswith("B")

    cfg = Exp6Config(
        experiment=args.experiment,
        model_variant=args.variant,
        dataset_path=args.dataset,
        device=args.device,
        max_gen_tokens=args.max_gen_tokens,
        n_eval_examples=args.n_eval_examples,
        run_name=run_name,
        corrective_direction_path=args.corrective_direction_path,
        content_direction_path=args.content_direction_path,
        governance_features_path=args.governance_features_path,
        mean_feature_acts_path=args.mean_feature_acts_path,
        governance_direction_path=args.governance_direction_path,
        feature_set=args.feature_set,
        gamma=args.gamma,
        directional_beta=args.beta,
        feature_layer_range=args.feature_layer_range,
        skip_transcoders=(not is_B),
    )

    ensure_dir(cfg.run_dir)
    ensure_dir(cfg.plots_dir)
    save_json(cfg.run_dir / "run_config.json", cfg.to_dict())

    scores_jsonl = cfg.run_dir / "scores.jsonl"
    samples_jsonl = cfg.run_dir / "sample_outputs.jsonl"

    all_scores = _read_jsonl(scores_jsonl)
    done_pairs: set[tuple[str, str]] = {(r["condition"], r["benchmark"]) for r in all_scores}
    if done_pairs:
        print(f"[exp6] resuming — {len(done_pairs)} (condition, benchmark) pairs already done", flush=True)

    _check_disk()
    records = load_dataset_records(cfg.dataset_path, prompt_format=cfg.prompt_format)
    loaded = load_model(cfg)

    condition_specs = _condition_specs(cfg)

    if args.n_workers > 1:
        condition_specs = [
            (name, cond_cfg)
            for i, (name, cond_cfg) in enumerate(condition_specs)
            if i % args.n_workers == args.worker_index
        ]
        print(
            f"[exp6] worker {args.worker_index}/{args.n_workers} — "
            f"{len(condition_specs)} conditions: {[n for n, _ in condition_specs]}",
            flush=True,
        )

    # Load B-experiment hooks config once (same for all B conditions)
    hooks_config: dict = {}
    if is_B:
        hooks_config = _load_B_hooks_config(cfg, loaded)

    for condition_name, condition_cfg in condition_specs:
        # B2 mixes IT and PT conditions — skip any that don't match the loaded model variant.
        if condition_cfg.model_variant != args.variant:
            print(
                f"[exp6] skip {condition_name} "
                f"(variant={condition_cfg.model_variant}, loaded={args.variant})",
                flush=True,
            )
            continue

        done_benchmarks = {b for (c, b) in done_pairs if c == condition_name}
        if done_benchmarks >= set(condition_cfg.benchmarks):
            print(f"[exp6] skip {condition_name} (all done)", flush=True)
            continue

        print(f"[exp6] running {condition_name}", flush=True)

        if is_B:
            # Update hooks_config for condition-specific gamma / feature_set
            condition_hooks = dict(hooks_config)
            if condition_cfg.method == "feature_clamp" and "governance_features" in hooks_config:
                # Re-slice feature sets for this condition
                feat_path = Path(condition_cfg.governance_features_path)
                with open(feat_path) as f:
                    all_sets = json.load(f)
                condition_hooks["governance_features"] = {
                    k: all_sets.get(k, {}).get(condition_cfg.feature_set, [])
                    for k in all_sets
                }
            _run_condition_B(
                condition_name, condition_cfg, records, loaded,
                scores_jsonl, samples_jsonl, done_benchmarks, condition_hooks,
            )
        else:
            _run_condition_A(
                condition_name, condition_cfg, records, loaded,
                scores_jsonl, samples_jsonl, done_benchmarks,
            )

        print(f"[exp6] {condition_name} done", flush=True)

    # Write final summary
    all_scores = _read_jsonl(scores_jsonl)
    save_json(cfg.run_dir / "scores.json", all_scores)
    print(f"[exp6] done → {cfg.run_dir}")


if __name__ == "__main__":
    main()
