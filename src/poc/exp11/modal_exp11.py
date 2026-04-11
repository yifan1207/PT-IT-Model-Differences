"""Modal runners for Exp11 smoke, batching validation, and full B200 launches.

Usage examples:
    modal run src/poc/exp11/modal_exp11.py --mode smoke --run-name gemma3_4b_smoke_v2
    modal run src/poc/exp11/modal_exp11.py --mode preflight --model gemma3_4b --run-name gemma3_4b_preflight_v1
    modal run src/poc/exp11/modal_exp11.py --mode full --model gemma3_4b --run-name gemma3_4b_full_v1
    modal run src/poc/exp11/modal_exp11.py --mode full --run-name exp11_full_20260409
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.utils import load_dataset
from src.poc.exp11.run import _apply_prompt_shard, _sample_prompts


VALID_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
SMOKE_MODEL = "gemma3_4b"
DEFAULT_DATASET = "exp3"
DATASET_PATHS = {
    "eval_v2": "/root/data/eval_dataset_v2.jsonl",
    "exp3": "/root/data/exp3_dataset.jsonl",
}
BALANCED_8GPU_SHARDS = {
    "llama31_8b": 2,
    "mistral_7b": 2,
    "olmo2_7b": 2,
    "gemma3_4b": 1,
    "qwen3_4b": 1,
}
FULL_BATCH_HINTS = {
    "gemma3_4b": 128,
    "qwen3_4b": 128,
    "llama31_8b": 64,
    "mistral_7b": 64,
    "olmo2_7b": 64,
}
_HF_MODELS_TO_BAKE = sorted(
    {
        model_id_for_variant(get_spec(model_name), variant)
        for model_name in VALID_MODELS
        for variant in ("pt", "it")
    }
)

app = modal.App("exp11-mlp-graft")
results_vol = modal.Volume.from_name("exp11-results", create_if_missing=True)

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
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "einops",
        "tqdm",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .run_commands(
        *[f"huggingface-cli download {model_id}" for model_id in _HF_MODELS_TO_BAKE],
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("data/eval_dataset_v2.jsonl", remote_path="/root/data/eval_dataset_v2.jsonl")
    .add_local_file("data/exp3_dataset.jsonl", remote_path="/root/data/exp3_dataset.jsonl")
)

VOLUME_MOUNTS = {"/root/results": results_vol}


def _setup() -> None:
    os.chdir("/root")
    os.environ["PYTHONPATH"] = "/root"
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")


def _base_env(seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    env["HF_HOME"] = env.get("HF_HOME", "/root/.cache/huggingface")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONHASHSEED"] = str(seed)
    # Enables deterministic cuBLAS kernels for subprocessed Python runs.
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _commit_volumes() -> None:
    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[volume] commit warn: {exc}")


def _run_subprocess(cmd: list[str], *, seed: int) -> None:
    print("[exec]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=_base_env(seed))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _compare_numbers(a: float, b: float, *, atol: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= atol + rtol * abs(b)


def _compare_values(a: Any, b: Any, *, path: str, atol: float, rtol: float) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise AssertionError(f"{path}: key mismatch {set(a) ^ set(b)}")
        for key in sorted(a):
            _compare_values(a[key], b[key], path=f"{path}.{key}", atol=atol, rtol=rtol)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(f"{path}: length mismatch {len(a)} != {len(b)}")
        for idx, (left, right) in enumerate(zip(a, b, strict=True)):
            _compare_values(left, right, path=f"{path}[{idx}]", atol=atol, rtol=rtol)
        return
    if isinstance(a, bool) and isinstance(b, bool):
        if a != b:
            raise AssertionError(f"{path}: bool mismatch {a} != {b}")
        return
    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            raise AssertionError(f"{path}: int mismatch {a} != {b}")
        return
    if isinstance(a, float) and isinstance(b, float):
        if not _compare_numbers(a, b, atol=atol, rtol=rtol):
            raise AssertionError(f"{path}: float mismatch {a} != {b}")
        return
    if a != b:
        raise AssertionError(f"{path}: value mismatch {a!r} != {b!r}")


def _compare_generated_outputs(ref_dir: Path, test_dir: Path) -> None:
    ref_rows = sorted(
        _load_jsonl(ref_dir / "generated_texts.jsonl"),
        key=lambda row: (row["prompt_id"], row["pipeline"]),
    )
    test_rows = sorted(
        _load_jsonl(test_dir / "generated_texts.jsonl"),
        key=lambda row: (row["prompt_id"], row["pipeline"]),
    )
    _compare_values(ref_rows, test_rows, path="generated_texts", atol=0.0, rtol=0.0)


def _compare_step_metrics(ref_dir: Path, test_dir: Path) -> None:
    ref_rows = sorted(
        _load_jsonl(ref_dir / "step_metrics.jsonl"),
        key=lambda row: (row["prompt_id"], row["pipeline"], row["step"]),
    )
    test_rows = sorted(
        _load_jsonl(test_dir / "step_metrics.jsonl"),
        key=lambda row: (row["prompt_id"], row["pipeline"], row["step"]),
    )
    _compare_values(ref_rows, test_rows, path="step_metrics", atol=1e-4, rtol=1e-4)


def _run_exp11(
    *,
    model: str,
    run_dir: str,
    dataset_path: str,
    n_prompts: int,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    shard_index: int = 0,
    num_shards: int = 1,
    limit_prompts: int | None = None,
    resume: bool = False,
) -> None:
    cmd = [
        "python",
        "-m",
        "src.poc.exp11.run",
        "--model",
        model,
        "--dataset",
        dataset_path,
        "--n-prompts",
        str(n_prompts),
        "--seed",
        str(seed),
        "--max-new-tokens",
        str(max_new_tokens),
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda:0",
        "--out-dir",
        run_dir,
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
    ]
    if limit_prompts is not None:
        cmd.extend(["--limit-prompts", str(limit_prompts)])
    if resume:
        cmd.append("--resume")
    _run_subprocess(cmd, seed=seed)


def _analyze_exp11(run_dir: str, *, seed: int) -> None:
    _run_subprocess(
        ["python", "-m", "src.poc.exp11.analyze", "--run-dir", run_dir],
        seed=seed,
    )


def _dataset_path(dataset: str) -> str:
    try:
        return DATASET_PATHS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset '{dataset}'. Use one of {sorted(DATASET_PATHS)}.") from exc


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_shard_prompts(dataset_path: str, *, n_prompts: int, seed: int, shard_index: int, num_shards: int) -> int:
    dataset = load_dataset(dataset_path)
    prompts = _sample_prompts(dataset, n_prompts, seed, None)
    prompts = _apply_prompt_shard(prompts, shard_index, num_shards)
    return len(prompts)


def _merge_jsonl_unique(
    shard_paths: list[Path],
    out_path: Path,
    *,
    key_fn,
    sort_key,
) -> int:
    merged: dict[Any, dict[str, Any]] = {}
    for shard_path in shard_paths:
        if not shard_path.exists():
            continue
        for row in _load_jsonl(shard_path):
            key = key_fn(row)
            if key in merged:
                raise ValueError(f"Duplicate merged row for key={key} from {shard_path}")
            merged[key] = row
    rows = sorted(merged.values(), key=sort_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def _validate_batched_equivalence(
    *,
    model: str,
    requested_batch_size: int,
    run_prefix: str,
    seed: int,
) -> int:
    if requested_batch_size <= 1:
        return 1

    ref_dir = f"/root/results/exp11/{run_prefix}_equiv_bs1"
    test_batch_size = min(requested_batch_size, 8)
    test_dir = f"/root/results/exp11/{run_prefix}_equiv_bs{test_batch_size}"
    print(
        f"[preflight] validating {model}: batch_size=1 vs batch_size={test_batch_size} "
        f"on matched prompts before full launch",
        flush=True,
    )
    _run_exp11(
        model=model,
        run_dir=ref_dir,
        dataset_path=_dataset_path("eval_v2"),
        n_prompts=8,
        max_new_tokens=64,
        batch_size=1,
        seed=seed,
    )
    _run_exp11(
        model=model,
        run_dir=test_dir,
        dataset_path=_dataset_path("eval_v2"),
        n_prompts=8,
        max_new_tokens=64,
        batch_size=test_batch_size,
        seed=seed,
    )
    _compare_generated_outputs(Path(ref_dir), Path(test_dir))
    _compare_step_metrics(Path(ref_dir), Path(test_dir))
    print(f"[preflight] exact generated outputs and step metrics matched for {model}", flush=True)
    return requested_batch_size


def _results_hint(run_name: str) -> str:
    return f"modal volume get exp11-results exp11/{run_name} results/exp11/{run_name}"


@app.function(
    gpu="H100",
    timeout=7200,
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=65536,
    retries=modal.Retries(max_retries=2, initial_delay=15.0, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def smoke_test(run_name: str = "gemma3_4b_smoke") -> str:
    _setup()
    run_dir = f"/root/results/exp11/{run_name}"
    _run_exp11(
        model=SMOKE_MODEL,
        run_dir=run_dir,
        dataset_path=_dataset_path(DEFAULT_DATASET),
        n_prompts=5,
        max_new_tokens=8,
        batch_size=8,
        seed=0,
    )
    _analyze_exp11(run_dir, seed=0)
    _commit_volumes()
    return (
        f"exp11 smoke complete: model={SMOKE_MODEL}, batch_size=8, run_name={run_name}\n"
        f"Download with: {_results_hint(run_name)}"
    )


@app.function(
    gpu="H100",
    timeout=14400,
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=65536,
    retries=modal.Retries(max_retries=1, initial_delay=15.0, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def preflight_model_run(
    model: str,
    run_name: str,
    batch_size: int | None = None,
    seed: int = 0,
) -> str:
    _setup()
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model for exp11 preflight: {model}")
    requested_batch_size = batch_size or FULL_BATCH_HINTS[model]
    effective_batch_size = _validate_batched_equivalence(
        model=model,
        requested_batch_size=requested_batch_size,
        run_prefix=run_name,
        seed=seed,
    )
    _commit_volumes()
    return (
        f"exp11 preflight complete: model={model}, run_name={run_name}, "
        f"requested_batch_size={requested_batch_size}, effective_batch_size={effective_batch_size}\n"
        f"Download with: {_results_hint(run_name)}"
    )


@app.function(
    gpu="B200",
    timeout=86400,
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=131072,
    retries=modal.Retries(max_retries=1, initial_delay=30.0, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def full_model_run(
    model: str,
    run_name: str,
    n_prompts: int = 200,
    max_new_tokens: int = 512,
    batch_size: int | None = None,
    seed: int = 0,
    preflight: bool = True,
    dataset: str = DEFAULT_DATASET,
) -> str:
    _setup()
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model for exp11 full run: {model}")

    requested_batch_size = batch_size or FULL_BATCH_HINTS[model]
    effective_batch_size = requested_batch_size
    if preflight:
        effective_batch_size = _validate_batched_equivalence(
            model=model,
            requested_batch_size=requested_batch_size,
            run_prefix=f"{run_name}_preflight",
            seed=seed,
        )

    run_dir = f"/root/results/exp11/{run_name}"
    _run_exp11(
        model=model,
        run_dir=run_dir,
        dataset_path=_dataset_path(dataset),
        n_prompts=n_prompts,
        max_new_tokens=max_new_tokens,
        batch_size=effective_batch_size,
        seed=seed,
    )
    _analyze_exp11(run_dir, seed=seed)
    _commit_volumes()
    return (
        f"exp11 full run complete: model={model}, run_name={run_name}, "
        f"dataset={dataset}, requested_batch_size={requested_batch_size}, effective_batch_size={effective_batch_size}, "
        f"n_prompts={n_prompts}, max_new_tokens={max_new_tokens}\n"
        f"Download with: {_results_hint(run_name)}"
    )


@app.function(
    gpu="B200",
    timeout=86400,
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=131072,
    retries=modal.Retries(max_retries=1, initial_delay=30.0, backoff_coefficient=2.0),
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def full_model_shard_run(
    model: str,
    run_name: str,
    shard_index: int,
    num_shards: int = 3,
    n_prompts: int = 200,
    max_new_tokens: int = 512,
    prompts_per_chunk: int = 16,
    seed: int = 0,
    dataset: str = DEFAULT_DATASET,
) -> str:
    _setup()
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model for exp11 sharded run: {model}")
    dataset_path = _dataset_path(dataset)
    run_dir = f"/root/results/exp11/{run_name}__shard{shard_index}of{num_shards}"
    shard_total = _count_shard_prompts(
        dataset_path,
        n_prompts=n_prompts,
        seed=seed,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    prompts_path = Path(run_dir) / "prompts.jsonl"
    done_path = Path(run_dir) / "prompt_summaries.jsonl"
    while True:
        done_count = _count_jsonl_rows(done_path)
        if done_count >= shard_total:
            break
        _run_exp11(
            model=model,
            run_dir=run_dir,
            dataset_path=dataset_path,
            n_prompts=n_prompts,
            max_new_tokens=max_new_tokens,
            batch_size=1,
            seed=seed,
            shard_index=shard_index,
            num_shards=num_shards,
            limit_prompts=prompts_per_chunk,
            resume=True,
        )
        if not prompts_path.exists():
            raise RuntimeError(f"Shard run failed to materialize prompts.jsonl: {prompts_path}")
        new_done_count = _count_jsonl_rows(done_path)
        _commit_volumes()
        if new_done_count <= done_count:
            raise RuntimeError(
                f"Shard {shard_index}/{num_shards} made no progress for model={model}: "
                f"done_count stayed at {done_count}"
            )
    _commit_volumes()
    return (
        f"exp11 shard complete: model={model}, dataset={dataset}, run_name={run_name}, "
        f"shard={shard_index}/{num_shards}, prompts={shard_total}\n"
        f"Download with: {_results_hint(f'{run_name}__shard{shard_index}of{num_shards}')}"
    )


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=16384,
    timeout=7200,
)
def merge_model_shards(
    run_name: str,
    *,
    model: str,
    dataset: str,
    num_shards: int = 3,
    seed: int = 0,
) -> str:
    _setup()
    base_dir = Path("/root/results/exp11")
    shard_dirs = [base_dir / f"{run_name}__shard{idx}of{num_shards}" for idx in range(num_shards)]
    merged_dir = base_dir / run_name
    merged_dir.mkdir(parents=True, exist_ok=True)
    shard_configs = [json.loads((shard_dir / "config.json").read_text()) for shard_dir in shard_dirs if (shard_dir / "config.json").exists()]

    prompt_count = _merge_jsonl_unique(
        [shard_dir / "prompts.jsonl" for shard_dir in shard_dirs],
        merged_dir / "prompts.jsonl",
        key_fn=lambda row: row["id"],
        sort_key=lambda row: (row.get("category", ""), row["id"]),
    )
    summary_count = _merge_jsonl_unique(
        [shard_dir / "prompt_summaries.jsonl" for shard_dir in shard_dirs],
        merged_dir / "prompt_summaries.jsonl",
        key_fn=lambda row: row["prompt_id"],
        sort_key=lambda row: row["prompt_id"],
    )
    _merge_jsonl_unique(
        [shard_dir / "generated_texts.jsonl" for shard_dir in shard_dirs],
        merged_dir / "generated_texts.jsonl",
        key_fn=lambda row: (row["prompt_id"], row["pipeline"]),
        sort_key=lambda row: (row["prompt_id"], row["pipeline"]),
    )
    _merge_jsonl_unique(
        [shard_dir / "step_metrics.jsonl" for shard_dir in shard_dirs],
        merged_dir / "step_metrics.jsonl",
        key_fn=lambda row: (row["prompt_id"], row["pipeline"], row["step"]),
        sort_key=lambda row: (row["prompt_id"], row["pipeline"], row["step"]),
    )

    merged_config = dict(shard_configs[0]) if shard_configs else {}
    merged_config.update(
        {
            "model": model,
            "dataset": _dataset_path(dataset),
            "seed": seed,
            "num_shards": num_shards,
            "shard_run_dirs": [str(path.name) for path in shard_dirs],
            "scheduler_mode": "single_prompt_data_parallel",
            "batch_size_pipeline_a": 1,
            "batch_size_pipeline_b": 1,
            "n_prompts_sampled_after_sharding": prompt_count,
            "n_prompts_completed": summary_count,
        }
    )
    (merged_dir / "config.json").write_text(json.dumps(merged_config, indent=2))
    _analyze_exp11(str(merged_dir), seed=seed)
    _commit_volumes()
    return (
        f"exp11 shard merge complete: model={model}, dataset={dataset}, run_name={run_name}, "
        f"completed_prompts={summary_count}\n"
        f"Download with: {_results_hint(run_name)}"
    )


def _run_smoke(run_name: str) -> None:
    print("=" * 60)
    print(f"EXP11 MODAL SMOKE — model={SMOKE_MODEL}, run_name={run_name}")
    print("=" * 60)
    result = smoke_test.remote(run_name=run_name)
    print(result)


def _run_preflight(model: str, run_name: str, *, batch_size: int | None) -> None:
    print("=" * 60)
    print(
        f"EXP11 PREFLIGHT — model={model}, run_name={run_name}, "
        f"batch_size={batch_size or FULL_BATCH_HINTS[model]}"
    )
    print("=" * 60)
    result = preflight_model_run.remote(model, run_name, batch_size, 0)
    print(result)


def _launch_one_full(model: str, run_name: str, *, batch_size: int | None, preflight: bool, dataset: str) -> None:
    print("=" * 60)
    print(
        f"EXP11 FULL LAUNCH — model={model}, run_name={run_name}, "
        f"batch_size={batch_size or FULL_BATCH_HINTS[model]}, preflight={preflight}"
    )
    print("=" * 60)
    result = full_model_run.remote(
        model,
        run_name,
        200,
        512,
        batch_size,
        0,
        preflight,
        dataset,
    )
    print(result)


def _launch_all_full(run_prefix: str, *, batch_size: int | None, preflight: bool, dataset: str) -> None:
    print("=" * 60)
    batch_desc = batch_size if batch_size is not None else "per-model-default"
    print(
        f"EXP11 FULL LAUNCH — all models, run_prefix={run_prefix}, "
        f"batch_size={batch_desc}, preflight={preflight}"
    )
    print("=" * 60)
    calls = []
    for model in VALID_MODELS:
        run_name = f"{run_prefix}_{model}"
        requested_batch_size = batch_size or FULL_BATCH_HINTS[model]
        call = full_model_run.spawn(model, run_name, 200, 512, requested_batch_size, 0, preflight, dataset)
        calls.append((model, run_name, requested_batch_size, call))
    for model, run_name, requested_batch_size, call in calls:
        object_id = getattr(call, "object_id", None)
        print(
            f"spawned model={model} run_name={run_name} requested_batch_size={requested_batch_size} "
            f"call_id={object_id or call}"
        )
        print(f"results: {_results_hint(run_name)}")
        print(f"dashboard: {call.get_dashboard_url()}")
    print("waiting for spawned full-model runs to finish...", flush=True)
    for model, run_name, _requested_batch_size, call in calls:
        print(f"[wait] model={model} run_name={run_name}", flush=True)
        try:
            result = call.get()
            print(result)
        except Exception as exc:
            print(
                f"[error] model={model} run_name={run_name} "
                f"dashboard={call.get_dashboard_url()} error={exc!r}",
                flush=True,
            )


def _launch_one_sharded_full(
    model: str,
    run_name: str,
    *,
    dataset: str,
    num_shards: int,
    prompts_per_chunk: int,
) -> None:
    print("=" * 60)
    print(
        f"EXP11 SHARDED FULL LAUNCH — model={model}, run_name={run_name}, "
        f"dataset={dataset}, num_shards={num_shards}, prompts_per_chunk={prompts_per_chunk}"
    )
    print("=" * 60)
    calls = []
    for shard_index in range(num_shards):
        call = full_model_shard_run.spawn(
            model,
            run_name,
            shard_index,
            num_shards,
            200,
            512,
            prompts_per_chunk,
            0,
            dataset,
        )
        calls.append((shard_index, call))
        print(
            f"spawned shard={shard_index}/{num_shards} call_id={getattr(call, 'object_id', call)} "
            f"dashboard={call.get_dashboard_url()}",
            flush=True,
        )
    for shard_index, call in calls:
        print(f"[wait] shard={shard_index}/{num_shards}", flush=True)
        result = call.get()
        print(result)
    merge_result = merge_model_shards.remote(
        run_name,
        model=model,
        dataset=dataset,
        num_shards=num_shards,
        seed=0,
    )
    print(merge_result)


def _launch_all_sharded_full(
    run_prefix: str,
    *,
    dataset: str,
    num_shards: int,
    prompts_per_chunk: int,
) -> None:
    print("=" * 60)
    print(
        f"EXP11 SHARDED FULL LAUNCH — all models sequentially, run_prefix={run_prefix}, "
        f"dataset={dataset}, num_shards={num_shards}, prompts_per_chunk={prompts_per_chunk}"
    )
    print("=" * 60)
    for model in VALID_MODELS:
        _launch_one_sharded_full(
            model,
            f"{run_prefix}_{model}",
            dataset=dataset,
            num_shards=num_shards,
            prompts_per_chunk=prompts_per_chunk,
        )


def _launch_balanced_8gpu_full(
    run_prefix: str,
    *,
    dataset: str,
    prompts_per_chunk: int,
) -> None:
    print("=" * 60)
    print(
        f"EXP11 BALANCED 8-GPU LAUNCH — all models, run_prefix={run_prefix}, "
        f"dataset={dataset}, shard_map={BALANCED_8GPU_SHARDS}, prompts_per_chunk={prompts_per_chunk}"
    )
    print("=" * 60)

    per_model_calls: dict[str, list[tuple[int, Any]]] = {}
    for model in VALID_MODELS:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_8GPU_SHARDS[model]
        calls: list[tuple[int, Any]] = []
        for shard_index in range(num_shards):
            call = full_model_shard_run.spawn(
                model,
                run_name,
                shard_index,
                num_shards,
                200,
                512,
                prompts_per_chunk,
                0,
                dataset,
            )
            calls.append((shard_index, call))
            print(
                f"spawned model={model} shard={shard_index}/{num_shards} "
                f"call_id={getattr(call, 'object_id', call)} dashboard={call.get_dashboard_url()}",
                flush=True,
            )
        per_model_calls[model] = calls

    for model in VALID_MODELS:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_8GPU_SHARDS[model]
        calls = per_model_calls[model]
        for shard_index, call in calls:
            print(f"[wait] model={model} shard={shard_index}/{num_shards}", flush=True)
            result = call.get()
            print(result)
        merge_result = merge_model_shards.remote(
            run_name,
            model=model,
            dataset=dataset,
            num_shards=num_shards,
            seed=0,
        )
        print(merge_result)


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    run_name: str = "gemma3_4b_smoke",
    model: str = "",
    batch_size: int = 0,
    preflight: bool = True,
    dataset: str = DEFAULT_DATASET,
    num_shards: int = 3,
    prompts_per_chunk: int = 16,
) -> None:
    if mode == "smoke":
        _run_smoke(run_name)
        return
    if mode == "preflight":
        if not model:
            raise ValueError("`--model` is required for preflight mode.")
        _run_preflight(model, run_name, batch_size=batch_size or None)
        return
    if mode == "full":
        if model:
            _launch_one_full(model, run_name, batch_size=batch_size or None, preflight=preflight, dataset=dataset)
        else:
            _launch_all_full(run_name, batch_size=batch_size or None, preflight=preflight, dataset=dataset)
        return
    if mode == "sharded-full":
        if model:
            _launch_one_sharded_full(
                model,
                run_name,
                dataset=dataset,
                num_shards=num_shards,
                prompts_per_chunk=prompts_per_chunk,
            )
        else:
            _launch_all_sharded_full(
                run_name,
                dataset=dataset,
                num_shards=num_shards,
                prompts_per_chunk=prompts_per_chunk,
            )
        return
    if mode == "balanced-8gpu-full":
        if model:
            raise ValueError("`--model` is not supported for balanced-8gpu-full mode.")
        _launch_balanced_8gpu_full(
            run_name,
            dataset=dataset,
            prompts_per_chunk=prompts_per_chunk,
        )
        return
    print("Unknown mode. Use 'smoke', 'preflight', 'full', 'sharded-full', or 'balanced-8gpu-full'.")
