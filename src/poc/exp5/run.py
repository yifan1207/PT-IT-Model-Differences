from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.poc.collect import load_dataset_records
from src.poc.exp5.analysis.subspace import summarise_checkpoint_shift
from src.poc.exp5.analysis.summary import write_scores_csv
from src.poc.exp5.benchmarks.custom import evaluate_custom_benchmark
from src.poc.exp5.benchmarks.harness import run_harness_stub
from src.poc.exp5.config import Exp5Config
from src.poc.exp5.plots.dose_response import plot_dose_response
from src.poc.exp5.plots.heatmap import plot_phase_benchmark_heatmap
from src.poc.exp5.plots.subspace import plot_checkpoint_metrics
from src.poc.exp5.runtime import build_intervention, generate_record
from src.poc.exp5.utils import ensure_dir, sanitise_json, save_json, save_jsonl
from src.poc.shared.model import load_model

_DISK_WARN_GB = 5.0   # print warning below this
_DISK_STOP_GB = 2.0   # raise hard stop below this


def _free_gb() -> float:
    return shutil.disk_usage(".").free / 1e9


def _check_disk() -> None:
    free = _free_gb()
    if free < _DISK_STOP_GB:
        raise RuntimeError(
            f"[disk] only {free:.1f} GB free — stopping to prevent data loss. "
            "Free up space before resuming."
        )
    if free < _DISK_WARN_GB:
        print(f"[disk] WARNING: {free:.1f} GB free", flush=True)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append rows to a JSONL file (creates if absent)."""
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


def _done_marker(checkpoints_dir: Path, name: str) -> Path:
    return checkpoints_dir / f"{name}.done"


def _condition_specs(cfg: Exp5Config) -> list[tuple[str, Exp5Config]]:
    if cfg.experiment == "baseline":
        return [("baseline", cfg)]

    if cfg.experiment == "cartography":
        specs = []
        for method in ("mean", "skip"):
            for layer_i in range(cfg.n_layers):
                name = f"{method}_l{layer_i}"
                specs.append((name, replace(cfg, method=method, ablation_layers=[layer_i])))
        return specs

    if cfg.experiment == "phase":
        phases = {
            "content": list(range(0, 12)),
            "format": list(range(12, 20)),
            "corrective": list(range(20, 34)),
        }
        specs = [("baseline", replace(cfg, method="none", ablation_layers=[]))]
        for phase_name, layers in phases.items():
            specs.append((f"{phase_name}_mean", replace(cfg, method="mean", ablation_layers=layers)))
            specs.append((f"{phase_name}_skip", replace(cfg, method="skip", ablation_layers=layers)))
            if phase_name == "corrective":
                specs.append((
                    "corrective_directional",
                    replace(cfg, method="directional", ablation_layers=layers),
                ))
        return specs

    if cfg.experiment == "progressive":
        specs = []
        for start in [33, 32, 31, 30, 28, 26, 24, 20]:
            layers = list(range(start, 34))
            specs.append((f"skip_{start}_33", replace(cfg, method="skip", ablation_layers=layers)))
        for alpha in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
            specs.append((
                f"dir_alpha_{alpha:g}",
                replace(cfg, method="directional", ablation_layers=list(range(20, 34)), directional_alpha=alpha),
            ))
        return specs

    if cfg.experiment == "subspace":
        return [("subspace", cfg)]

    raise ValueError(f"Unknown experiment: {cfg.experiment}")


def _select_records(records: list[dict], benchmark: str, n: int) -> list[dict]:
    """Return a seeded random sample of up to n records for the given benchmark.

    All cited ablation papers (Wang et al. 2023, Li & Janson 2024, Arditi et al.
    2024) draw random samples from their evaluation distributions.  Slicing the
    first N records introduces selection bias when the dataset is ordered by
    category or source.  We use a fixed seed for full reproducibility across
    conditions and runs.
    """
    import random as _random
    if benchmark == "exp3_factual_em":
        subset = [r for r in records if r["split"] == "F"]
    elif benchmark == "exp3_reasoning_em":
        subset = [r for r in records if r["split"] in {"R", "GEN"}]
    elif benchmark in {"exp3_alignment_behavior", "structural_tokens"}:
        subset = [r for r in records if r["split"] == "A"]
    elif benchmark == "exp3_format_adherence":
        # score_format_adherence only handles subcategories 5c/5d/5e; 5a/5b records
        # would be silently dropped from the mean, biasing n and value. Filter here
        # so the scorer receives only the records it can actually score.
        subset = [
            r for r in records
            if r["split"] == "A"
            and r.get("metadata", {}).get("alignment_subcategory") in {"5c", "5d", "5e"}
        ]
    else:
        subset = records
    if len(subset) <= n:
        return subset
    return _random.Random(42).sample(subset, n)


def _run_condition(name: str, cfg: Exp5Config, records: list[dict], loaded) -> tuple[list[dict], list[dict], dict[int, np.ndarray]]:
    intervention = build_intervention(cfg)
    score_rows: list[dict] = []
    sample_rows: list[dict] = []
    checkpoint_store: dict[int, list[np.ndarray]] = {li: [] for li in cfg.checkpoint_layers}
    records_done = 0

    for benchmark in cfg.benchmarks:
        benchmark_records = _select_records(records, benchmark, cfg.n_eval_examples)
        outputs = []
        for rec in benchmark_records:
            outputs.append(generate_record(rec, loaded, cfg, intervention))
            records_done += 1
            if records_done % 200 == 0:
                _check_disk()
                print(f"[exp5] {name}: {records_done} records done, {_free_gb():.1f} GB free", flush=True)
        for out in outputs:
            sample_rows.append({
                "condition": name,
                "record_id": out.record_id,
                "prompt": out.prompt,
                "generated_text": out.generated_text,
            })
            for li, arr in out.hidden_states.items():
                if arr.size > 0:
                    checkpoint_store[li].append(arr[-1])
        if (cfg.eval_backend in {"custom", "hybrid"} and benchmark.startswith("exp3_")) or benchmark == "structural_tokens":
            result = evaluate_custom_benchmark(benchmark, benchmark_records, outputs)
            score_rows.append({
                "condition": name,
                "benchmark": result.benchmark,
                "metric": result.metric,
                "value": result.value,
                "n": result.n,
                "method": cfg.method,
                "layers": cfg.ablation_layers,
                "alpha": cfg.directional_alpha,
            })

    if cfg.use_lm_eval and cfg.eval_backend in {"harness", "hybrid"}:
        for hs in run_harness_stub(cfg=cfg):
            score_rows.append({
                "condition": name,
                "benchmark": hs.benchmark,
                "metric": hs.metric,
                "value": hs.value,
                "n": 0,
                "method": cfg.method,
                "layers": cfg.ablation_layers,
                "alpha": cfg.directional_alpha,
            })

    stacked = {
        li: np.stack(values, axis=0) if values else np.zeros((0, cfg.d_model), dtype=np.float32)
        for li, values in checkpoint_store.items()
    }
    return score_rows, sample_rows, stacked


def main() -> None:
    p = argparse.ArgumentParser(description="Run exp5 three-phase ablation experiments.")
    p.add_argument("--experiment", choices=["baseline", "cartography", "phase", "progressive", "subspace"], default="baseline")
    p.add_argument("--variant", choices=["pt", "it"], default="it")
    p.add_argument("--dataset", default="data/exp3_dataset.jsonl")
    p.add_argument("--prompt-format", choices=["A", "B"], default="B")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--method", choices=["none", "mean", "skip", "directional", "resample"], default="none")
    p.add_argument("--layers", default="")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-gen-tokens", type=int, default=200)
    p.add_argument("--n-eval-examples", type=int, default=500)
    p.add_argument("--run-name", default="")
    p.add_argument("--mean-acts-path", default="")
    p.add_argument("--corrective-direction-path", default="")
    p.add_argument("--resample-bank-path", default="")
    p.add_argument("--use-lm-eval", action="store_true")
    args = p.parse_args()

    if args.use_lm_eval:
        p.error(
            "--use-lm-eval is not yet implemented: the lm-eval harness stub always "
            "raises NotImplementedError. Remove --use-lm-eval to run the custom benchmarks."
        )

    layers = []
    if args.layers:
        for part in args.layers.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                layers.extend(range(int(lo), int(hi) + 1))
            elif part:
                layers.append(int(part))

    cfg = Exp5Config(
        experiment=args.experiment,
        model_variant=args.variant,
        dataset_path=args.dataset,
        prompt_format=args.prompt_format,
        apply_chat_template=args.chat_template,
        method=args.method,
        ablation_layers=layers,
        directional_alpha=args.alpha,
        max_gen_tokens=args.max_gen_tokens,
        n_eval_examples=args.n_eval_examples,
        run_name=args.run_name,
        mean_acts_path=args.mean_acts_path,
        corrective_direction_path=args.corrective_direction_path,
        resample_bank_path=args.resample_bank_path,
        use_lm_eval=args.use_lm_eval,
    )

    ensure_dir(cfg.run_dir)
    ensure_dir(cfg.plots_dir)
    ensure_dir(cfg.checkpoints_dir)
    save_json(cfg.run_dir / "run_config.json", cfg.to_dict())

    scores_jsonl = cfg.run_dir / "scores.jsonl"
    samples_jsonl = cfg.run_dir / "sample_outputs.jsonl"

    # Resume: load any scores already written from a previous interrupted run.
    all_scores: list[dict] = _read_jsonl(scores_jsonl)
    done_conditions = {r["condition"] for r in all_scores}
    if done_conditions:
        print(f"[exp5] resuming — already done: {sorted(done_conditions)}", flush=True)

    _check_disk()
    print(f"[exp5] disk free: {_free_gb():.1f} GB", flush=True)

    records = load_dataset_records(cfg.dataset_path, prompt_format=cfg.prompt_format)
    loaded = load_model(cfg)
    condition_specs = _condition_specs(cfg)

    baseline_checkpoints: dict[int, np.ndarray] | None = None
    subspace_rows: dict[str, dict[int, dict[str, float]]] = {}

    # Reference pass for subspace analysis — skip if checkpoint already on disk.
    needs_reference = cfg.save_hidden_states and cfg.experiment != "baseline"
    if needs_reference:
        ref_npz = cfg.checkpoints_dir / "baseline_ref_hidden_states.npz"
        if ref_npz.exists():
            print("[exp5] loading baseline_ref checkpoints from disk", flush=True)
            with np.load(ref_npz) as d:
                baseline_checkpoints = {
                    int(k.split("_", 1)[1]): d[k] for k in d.files
                }
        else:
            baseline_cfg = replace(cfg, method="none", ablation_layers=[], directional_alpha=1.0)
            _, _, baseline_checkpoints = _run_condition("baseline_ref", baseline_cfg, records, loaded)
            np.savez_compressed(ref_npz, **{
                f"layer_{li}": arr for li, arr in baseline_checkpoints.items()
            })

    for condition_name, condition_cfg in condition_specs:
        if condition_name in done_conditions:
            print(f"[exp5] skip {condition_name} (already done)", flush=True)
            # Reload its checkpoints so subspace analysis still works.
            npz = condition_cfg.checkpoints_dir / f"{condition_name}_hidden_states.npz"
            if npz.exists() and baseline_checkpoints is not None:
                with np.load(npz) as d:
                    checkpoints = {int(k.split("_", 1)[1]): d[k] for k in d.files}
                subspace_rows[condition_name] = summarise_checkpoint_shift(baseline_checkpoints, checkpoints)
            continue

        print(f"[exp5] running condition: {condition_name}", flush=True)
        score_rows, sample_rows, checkpoints = _run_condition(condition_name, condition_cfg, records, loaded)

        # Persist immediately — before touching any aggregates.
        np.savez_compressed(condition_cfg.checkpoints_dir / f"{condition_name}_hidden_states.npz", **{
            f"layer_{li}": arr for li, arr in checkpoints.items()
        })
        _append_jsonl(scores_jsonl, score_rows)
        _append_jsonl(samples_jsonl, sample_rows)

        all_scores.extend(score_rows)
        if baseline_checkpoints is None:
            baseline_checkpoints = checkpoints
        if baseline_checkpoints is not None:
            subspace_rows[condition_name] = summarise_checkpoint_shift(baseline_checkpoints, checkpoints)

        print(f"[exp5] {condition_name} done — {_free_gb():.1f} GB free", flush=True)

    # Re-read the full JSONL so scores.json / scores.csv reflect all runs including resumed ones.
    all_scores = _read_jsonl(scores_jsonl)
    save_json(cfg.run_dir / "scores.json", all_scores)
    write_scores_csv(cfg.run_dir / "scores.csv", all_scores)
    save_json(cfg.run_dir / "subspace_metrics.json", subspace_rows)

    plot_phase_benchmark_heatmap(all_scores, cfg.plots_dir / "phase_benchmark_heatmap.png")
    if cfg.experiment == "progressive":
        progressive = []
        directional = []
        for row in all_scores:
            cond = row["condition"]
            if cond.startswith("skip_"):
                progressive.append({**row, "x_value": len(row["layers"])})
            elif cond.startswith("dir_alpha_"):
                directional.append({**row, "x_value": row["alpha"]})
        plot_dose_response(progressive, "x_value", cfg.plots_dir / "progressive_skip.png", "Exp5 — Progressive Skip")
        plot_dose_response(directional, "x_value", cfg.plots_dir / "directional_alpha.png", "Exp5 — Directional Alpha Sweep")
    if baseline_checkpoints is not None and subspace_rows:
        # Plot ALL conditions on the same axes so the full
        # (condition × checkpoint × metric) table is visible — per Exp D design.
        plot_checkpoint_metrics(subspace_rows, cfg.plots_dir / "checkpoint_metrics.png")

    print(f"Saved exp5 outputs → {cfg.run_dir}")


if __name__ == "__main__":
    main()
