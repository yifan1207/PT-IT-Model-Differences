"""Modal runners for cross-model L8 residual collection + rank corroboration.

Usage:
    modal run src/poc/cross_model/modal_l8.py --mode smoke --model gemma3_4b --run-name l8_smoke_gemma
    modal deploy src/poc/cross_model/modal_l8.py
    modal run src/poc/cross_model/modal_l8.py --mode balanced-10gpu-full --run-name l8_eval_v1
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

from src.poc.cross_model.config import MODEL_REGISTRY


VALID_MODELS = list(MODEL_REGISTRY)
DEFAULT_DATASET = "eval_v2"
BALANCED_10GPU_SHARDS_L8 = {
    "gemma3_4b": 1,
    "llama31_8b": 2,
    "qwen3_4b": 1,
    "mistral_7b": 2,
    "olmo2_7b": 2,
    "deepseek_v2_lite": 2,
}


def _repo_root_for_local_paths() -> Path:
    here = Path(__file__).resolve()
    if len(here.parents) >= 4:
        return here.parents[3]
    return Path.cwd()


DATASET_PATHS = {
    "eval_v2": "/root/data/eval_dataset_v2.jsonl",
}
LOCAL_DATASET_PATHS = {
    "eval_v2": str((_repo_root_for_local_paths() / "data" / "eval_dataset_v2.jsonl")),
}


app = modal.App("cross-model-l8")
results_vol = modal.Volume.from_name("l8-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("0g-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.7.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==4.57.3",
        "accelerate",
        "numpy",
        "scipy",
        "scikit-dimension>=0.3.4",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "einops",
        "tqdm",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("data/eval_dataset_v2.jsonl", remote_path="/root/data/eval_dataset_v2.jsonl")
)

VOLUME_MOUNTS = {
    "/root/results": results_vol,
    "/root/.cache/huggingface": hf_cache_vol,
}


def _setup() -> None:
    os.chdir("/root")
    os.environ["PYTHONPATH"] = "/root"
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    env["HF_HOME"] = env.get("HF_HOME", "/root/.cache/huggingface")
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _commit_volumes() -> None:
    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[volume] commit warn: {exc}", flush=True)


def _run_subprocess(cmd: list[str]) -> None:
    print("[exec]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=_base_env())


def _dataset_path(dataset: str) -> str:
    try:
        remote_path = DATASET_PATHS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset '{dataset}'. Use one of {sorted(DATASET_PATHS)}.") from exc
    if Path(remote_path).exists():
        return remote_path
    return LOCAL_DATASET_PATHS[dataset]


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path) as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _variant_dir(run_name: str, model: str, variant: str) -> str:
    return f"/root/results/cross_model_runs/{run_name}/{model}/{variant}"


def _run_collect_l8(
    *,
    run_name: str,
    model: str,
    variant: str,
    dataset_path: str,
    n_prompts: int,
    shard_index: int,
    num_shards: int,
) -> None:
    cmd = [
        "python",
        "-m",
        "src.poc.cross_model.collect_L8",
        "--model",
        model,
        "--variant",
        variant,
        "--dataset",
        dataset_path,
        "--n-eval-examples",
        str(n_prompts),
        "--device",
        "cuda:0",
        "--worker-index",
        str(shard_index),
        "--n-workers",
        str(num_shards),
        "--out-dir",
        _variant_dir(run_name, model, variant),
    ]
    _run_subprocess(cmd)


def _run_merge_and_rank(*, run_name: str, model: str, variant: str, num_shards: int) -> dict[str, Any]:
    out_dir = _variant_dir(run_name, model, variant)
    merged_npz = Path(out_dir) / "L8_residuals.npz"
    id_profile_path = Path(out_dir) / "L8_id_profile.json"
    rank_metrics_path = Path(out_dir) / "L8_rank_metrics.json"
    if not (merged_npz.exists() and id_profile_path.exists()):
        _run_subprocess([
            "python",
            "-m",
            "src.poc.cross_model.collect_L8",
            "--model",
            model,
            "--variant",
            variant,
            "--merge-only",
            "--n-workers",
            str(num_shards),
            "--out-dir",
            out_dir,
        ])
    if not rank_metrics_path.exists():
        _run_subprocess([
            "python",
            "-m",
            "src.poc.cross_model.l8_rank_metrics",
            "--residuals-npz",
            str(merged_npz),
            "--out-json",
            str(rank_metrics_path),
            "--max-prompts",
            "1400",
        ])
    id_profile = json.loads(id_profile_path.read_text())
    rank_metrics = json.loads(rank_metrics_path.read_text())
    return {
        "model": model,
        "variant": variant,
        "n_prompts": id_profile["n_prompts"],
        "id_last": id_profile["intrinsic_dim"][-1],
        "id_late_mean": float(sum(id_profile["intrinsic_dim"][int(len(id_profile["intrinsic_dim"]) * 0.8):]) / max(1, len(id_profile["intrinsic_dim"][int(len(id_profile["intrinsic_dim"]) * 0.8):]))),
        "pr_last": rank_metrics["summary"]["participation_ratio_last"],
        "pr_late_mean": rank_metrics["summary"]["participation_ratio_late_mean"],
        "erank_last": rank_metrics["summary"]["effective_rank_last"],
        "erank_late_mean": rank_metrics["summary"]["effective_rank_late_mean"],
        "pc1_last": rank_metrics["summary"]["pc1_ratio_last"],
        "pc1_late_mean": rank_metrics["summary"]["pc1_ratio_late_mean"],
    }


def _write_cross_model_summary(run_name: str, summaries: list[dict[str, Any]]) -> str:
    out_dir = Path("/root/results/cross_model_runs") / "_l8_rank_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(summaries, key=lambda row: (row["model"], row["variant"]))
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = row["model"]
        by_model.setdefault(model, {})[row["variant"]] = row
    deltas = []
    for model, pair in by_model.items():
        if "pt" not in pair or "it" not in pair:
            continue
        pt = pair["pt"]
        it = pair["it"]
        deltas.append(
            {
                "model": model,
                "delta_id_last": it["id_last"] - pt["id_last"],
                "delta_id_late_mean": it["id_late_mean"] - pt["id_late_mean"],
                "delta_pr_last": it["pr_last"] - pt["pr_last"],
                "delta_pr_late_mean": it["pr_late_mean"] - pt["pr_late_mean"],
                "delta_erank_last": it["erank_last"] - pt["erank_last"],
                "delta_erank_late_mean": it["erank_late_mean"] - pt["erank_late_mean"],
                "delta_pc1_last": it["pc1_last"] - pt["pc1_last"],
                "delta_pc1_late_mean": it["pc1_late_mean"] - pt["pc1_late_mean"],
            }
        )
    payload = {"run_name": run_name, "rows": rows, "deltas": deltas}
    out_path = out_dir / f"{run_name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    _commit_volumes()
    return str(out_path)


@app.function(
    gpu="B200",
    timeout=21600,
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=131072,
    retries=modal.Retries(max_retries=4, initial_delay=30.0, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def full_model_shard_run(
    model: str,
    run_name: str,
    shard_index: int,
    num_shards: int = 2,
    n_prompts: int = 1400,
    dataset: str = DEFAULT_DATASET,
) -> str:
    _setup()
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model for L8 sharded run: {model}")
    dataset_path = _dataset_path(dataset)
    print(
        f"=== L8 shard run model={model} shard={shard_index}/{num_shards} "
        f"n_prompts={n_prompts} dataset={dataset} ===",
        flush=True,
    )

    for variant in ("pt", "it"):
        worker_path = Path(_variant_dir(run_name, model, variant)) / f"L8_residuals_w{shard_index}.npz"
        if worker_path.exists():
            print(f"[skip] existing worker residuals for {model} {variant} shard={shard_index}", flush=True)
            continue
        _run_collect_l8(
            run_name=run_name,
            model=model,
            variant=variant,
            dataset_path=dataset_path,
            n_prompts=n_prompts,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        _commit_volumes()

    return (
        f"L8 shard complete: model={model}, run_name={run_name}, shard={shard_index}/{num_shards}, "
        f"variants=pt,it, n_prompts={n_prompts}"
    )


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=32768,
    timeout=21600,
)
def merge_model_results(
    run_name: str,
    *,
    model: str,
    num_shards: int,
) -> str:
    _setup()
    summaries = []
    for variant in ("pt", "it"):
        summaries.append(_run_merge_and_rank(run_name=run_name, model=model, variant=variant, num_shards=num_shards))
    summary_path = _write_cross_model_summary(f"{run_name}_{model}", summaries)
    return f"L8 merge complete: model={model}, summary={summary_path}"


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=4096,
    timeout=21600,
)
def orchestrate_parallel_merge_all(run_prefix: str) -> str:
    print("=" * 60)
    print(f"L8 PARALLEL MERGE ORCHESTRATOR — run_prefix={run_prefix}")
    print("=" * 60)

    calls: list[tuple[str, Any]] = []
    for model, num_shards in BALANCED_10GPU_SHARDS_L8.items():
        run_name = f"{run_prefix}_{model}"
        call = merge_model_results.spawn(run_name, model=model, num_shards=num_shards)
        calls.append((model, call))
        print(
            f"spawned merge model={model} shards={num_shards} "
            f"call_id={getattr(call, 'object_id', call)}",
            flush=True,
        )

    failed_models: list[str] = []
    for model, call in calls:
        print(f"[wait-merge] model={model}", flush=True)
        try:
            print(call.get(), flush=True)
        except Exception as exc:
            failed_models.append(model)
            print(f"[ERROR] merge failed for {model}: {exc}", flush=True)

    global_summary = write_global_summary.remote(run_prefix)
    print(global_summary, flush=True)
    if failed_models:
        return f"PARTIAL MERGE — failed models: {failed_models}"
    return "ALL L8 MERGE/RANK COMPLETE"


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=32768,
    timeout=21600,
)
def write_global_summary(run_name: str) -> str:
    _setup()
    summaries: list[dict[str, Any]] = []
    for model in VALID_MODELS:
        for variant in ("pt", "it"):
            model_run_name = f"{run_name}_{model}"
            out_path = Path(_variant_dir(model_run_name, model, variant)) / "L8_rank_metrics.json"
            id_path = Path(_variant_dir(model_run_name, model, variant)) / "L8_id_profile.json"
            if not out_path.exists() or not id_path.exists():
                continue
            rank_metrics = json.loads(out_path.read_text())
            id_profile = json.loads(id_path.read_text())
            summaries.append(
                {
                    "model": model,
                    "variant": variant,
                    "n_prompts": id_profile["n_prompts"],
                    "id_last": id_profile["intrinsic_dim"][-1],
                    "id_late_mean": float(sum(id_profile["intrinsic_dim"][int(len(id_profile["intrinsic_dim"]) * 0.8):]) / max(1, len(id_profile["intrinsic_dim"][int(len(id_profile["intrinsic_dim"]) * 0.8):]))),
                    "pr_last": rank_metrics["summary"]["participation_ratio_last"],
                    "pr_late_mean": rank_metrics["summary"]["participation_ratio_late_mean"],
                    "erank_last": rank_metrics["summary"]["effective_rank_last"],
                    "erank_late_mean": rank_metrics["summary"]["effective_rank_late_mean"],
                    "pc1_last": rank_metrics["summary"]["pc1_ratio_last"],
                    "pc1_late_mean": rank_metrics["summary"]["pc1_ratio_late_mean"],
                }
            )
    out = _write_cross_model_summary(run_name, summaries)
    return f"L8 global summary written: {out}"


@app.function(
    gpu="B200",
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=65536,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def smoke_run(
    model: str = "gemma3_4b",
    run_name: str = "l8_smoke",
) -> str:
    _setup()
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model: {model}")
    dataset_path = _dataset_path(DEFAULT_DATASET)
    for variant in ("pt", "it"):
        variant_dir = Path(_variant_dir(run_name, model, variant))
        for fname in ("L8_residuals_w0.npz", "L8_residuals.npz", "L8_id_profile.json", "L8_rank_metrics.json"):
            path = variant_dir / fname
            if path.exists():
                path.unlink()
        _run_collect_l8(
            run_name=run_name,
            model=model,
            variant=variant,
            dataset_path=dataset_path,
            n_prompts=2,
            shard_index=0,
            num_shards=1,
        )
    _commit_volumes()
    merge_model_results.remote(run_name, model=model, num_shards=1)
    return f"L8 smoke launched for {model}"


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=4096,
    timeout=86400,
)
def orchestrate_balanced_10gpu_full(
    run_prefix: str,
    dataset: str = DEFAULT_DATASET,
    n_prompts: int = 1400,
) -> str:
    print("=" * 60)
    print(
        f"L8 BALANCED 10GPU ORCHESTRATOR — run_prefix={run_prefix}, dataset={dataset}, "
        f"shard_map={BALANCED_10GPU_SHARDS_L8}, n_prompts={n_prompts}"
    )
    print("=" * 60)

    per_model_calls: dict[str, list[tuple[int, Any]]] = {}
    for model in BALANCED_10GPU_SHARDS_L8:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_10GPU_SHARDS_L8[model]
        calls: list[tuple[int, Any]] = []
        for shard_index in range(num_shards):
            call = full_model_shard_run.spawn(
                model,
                run_name,
                shard_index,
                num_shards,
                n_prompts,
                dataset,
            )
            calls.append((shard_index, call))
            print(
                f"spawned model={model} shard={shard_index}/{num_shards} "
                f"call_id={getattr(call, 'object_id', call)}",
                flush=True,
            )
        per_model_calls[model] = calls

    failed_models: list[str] = []
    for model in BALANCED_10GPU_SHARDS_L8:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_10GPU_SHARDS_L8[model]
        calls = per_model_calls[model]
        model_ok = True
        for shard_index, call in calls:
            print(f"[wait] model={model} shard={shard_index}/{num_shards}", flush=True)
            try:
                result = call.get()
                print(result, flush=True)
            except Exception as exc:
                print(f"[ERROR] model={model} shard={shard_index}/{num_shards} failed: {exc}", flush=True)
                model_ok = False
        if model_ok:
            merge_result = merge_model_results.remote(
                run_name,
                model=model,
                num_shards=num_shards,
            )
            print(merge_result, flush=True)
        else:
            failed_models.append(model)
            print(f"[SKIP] merge skipped for {model} due to shard failures", flush=True)

    global_summary = write_global_summary.remote(run_prefix)
    print(global_summary, flush=True)

    if failed_models:
        return f"PARTIAL — failed models: {failed_models}. Re-run to resume."
    return "ALL L8 MODELS COMPLETE"


def _launch_balanced_10gpu_full(
    run_prefix: str,
    *,
    dataset: str,
    n_prompts: int,
) -> None:
    print("Triggering L8 remote orchestrator on deployed app...", flush=True)
    call = orchestrate_balanced_10gpu_full.spawn(run_prefix, dataset, n_prompts)
    call_id = getattr(call, "object_id", call)
    print(f"L8 orchestrator spawned: call_id={call_id}", flush=True)
    print("Waiting for result (safe to Ctrl-C — orchestrator continues on Modal)...", flush=True)
    try:
        result = call.get()
        print(result, flush=True)
    except KeyboardInterrupt:
        print(f"\nLocal client interrupted — orchestrator {call_id} continues on Modal.", flush=True)
    except Exception as exc:
        print(f"Local client error: {exc}", flush=True)
        print(f"L8 orchestrator {call_id} may still be running on Modal.", flush=True)


def _launch_parallel_merge_all(run_prefix: str) -> None:
    print("Triggering L8 parallel merge orchestrator on deployed app...", flush=True)
    call = orchestrate_parallel_merge_all.spawn(run_prefix)
    call_id = getattr(call, "object_id", call)
    print(f"L8 parallel merge orchestrator spawned: call_id={call_id}", flush=True)
    print("Waiting for result (safe to Ctrl-C — orchestrator continues on Modal)...", flush=True)
    try:
        result = call.get()
        print(result, flush=True)
    except KeyboardInterrupt:
        print(f"\nLocal client interrupted — orchestrator {call_id} continues on Modal.", flush=True)
    except Exception as exc:
        print(f"Local client error: {exc}", flush=True)
        print(f"L8 parallel merge orchestrator {call_id} may still be running on Modal.", flush=True)


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    run_name: str = "l8_eval_v1",
    model: str = "gemma3_4b",
    dataset: str = DEFAULT_DATASET,
    n_prompts: int = 1400,
) -> None:
    if mode == "smoke":
        print(smoke_run.remote(model, run_name))
        return
    if mode == "balanced-10gpu-full":
        _launch_balanced_10gpu_full(
            run_name,
            dataset=dataset,
            n_prompts=n_prompts,
        )
        return
    if mode == "parallel-merge-all":
        _launch_parallel_merge_all(run_name)
        return
    raise ValueError("Unknown mode. Use 'smoke', 'balanced-10gpu-full', or 'parallel-merge-all'.")
