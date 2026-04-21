"""Modal runners for Exp12 free-running A/B/C raw-output collection.

Usage examples:
    modal run src/poc/exp12_free_running_abc_graft/modal_exp12.py --mode smoke --model gemma3_4b --run-name exp12_smoke_gemma
    modal run src/poc/exp12_free_running_abc_graft/modal_exp12.py --mode balanced-10gpu-full --run-name exp12_eval_v1
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

from src.poc.cross_model.utils import load_dataset
from src.poc.exp11_matched_prefix_mlp_graft.modal_exp11 import (
    _apply_prompt_shard,
    _random_subsample,
    _sample_prompts,
)


VALID_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b", "deepseek_v2_lite"]
DEFAULT_DATASET = "eval_v2"
BALANCED_10GPU_SHARDS_EXP12 = {
    "gemma3_4b": 1,
    "llama31_8b": 2,
    "qwen3_4b": 1,
    "mistral_7b": 2,
    "olmo2_7b": 2,
    "deepseek_v2_lite": 2,
}
ABC_BATCH_HINTS = {
    "gemma3_4b": 96,
    "qwen3_4b": 96,
    "llama31_8b": 64,
    "mistral_7b": 64,
    "olmo2_7b": 64,
    "deepseek_v2_lite": 64,
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


app = modal.App("exp12-abc-raw")
results_vol = modal.Volume.from_name("exp12-results", create_if_missing=True)
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


def _base_env(seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    env["HF_HOME"] = env.get("HF_HOME", "/root/.cache/huggingface")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _commit_volumes() -> None:
    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[volume] commit warn: {exc}", flush=True)


def _run_subprocess(cmd: list[str], *, seed: int) -> None:
    print("[exec]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=_base_env(seed))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def _count_condition_rows(path: Path) -> int:
    if not path.exists():
        return 0
    seen: set[tuple[str, str]] = set()
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            seen.add((row["condition"], row["record_id"]))
    return len(seen)


def _count_shard_prompts(
    dataset_path: str,
    *,
    n_prompts: int,
    seed: int,
    shard_index: int,
    num_shards: int,
    prompt_seed: int | None = None,
) -> int:
    dataset = load_dataset(dataset_path)
    if prompt_seed is not None:
        prompts = _random_subsample(dataset, n_prompts, prompt_seed)
    else:
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
    with open(out_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


def _run_exp12(
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
    prompt_seed: int | None = None,
) -> None:
    cmd = [
        "python",
        "-m",
        "src.poc.exp12_free_running_abc_graft.run",
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
    if prompt_seed is not None:
        cmd.extend(["--prompt-seed", str(prompt_seed)])
    _run_subprocess(cmd, seed=seed)


@app.function(
    gpu="B200",
    timeout=86400,
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
    max_new_tokens: int = 512,
    prompts_per_chunk: int = 1400,
    seed: int = 0,
    dataset: str = DEFAULT_DATASET,
    batch_size: int = 64,
    prompt_seed: int | None = None,
) -> str:
    _setup()
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model for exp12 sharded run: {model}")
    dataset_path = _dataset_path(dataset)
    run_dir = f"/root/results/exp12_free_running_abc_graft/{run_name}__shard{shard_index}of{num_shards}"
    shard_total = _count_shard_prompts(
        dataset_path,
        n_prompts=n_prompts,
        seed=seed,
        shard_index=shard_index,
        num_shards=num_shards,
        prompt_seed=prompt_seed,
    )
    done_path = Path(run_dir) / "sample_outputs.jsonl"
    target_rows = shard_total * 3
    while True:
        done_count = _count_condition_rows(done_path)
        if done_count >= target_rows:
            break
        _run_exp12(
            model=model,
            run_dir=run_dir,
            dataset_path=dataset_path,
            n_prompts=n_prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            seed=seed,
            shard_index=shard_index,
            num_shards=num_shards,
            limit_prompts=prompts_per_chunk,
            resume=True,
            prompt_seed=prompt_seed,
        )
        new_done_count = _count_condition_rows(done_path)
        _commit_volumes()
        if new_done_count <= done_count:
            raise RuntimeError(
                f"Shard {shard_index}/{num_shards} made no progress for model={model}: "
                f"done_count stayed at {done_count}"
            )
    _commit_volumes()
    return (
        f"exp12 shard complete: model={model}, dataset={dataset}, run_name={run_name}, "
        f"shard={shard_index}/{num_shards}, prompts={shard_total}, rows={target_rows}"
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
    num_shards: int = 2,
    seed: int = 0,
) -> str:
    _setup()
    base_dir = Path("/root/results/exp12_free_running_abc_graft")
    shard_dirs = [base_dir / f"{run_name}__shard{idx}of{num_shards}" for idx in range(num_shards)]
    merged_dir = base_dir / run_name
    merged_dir.mkdir(parents=True, exist_ok=True)
    shard_configs = [
        json.loads((shard_dir / "config.json").read_text())
        for shard_dir in shard_dirs
        if (shard_dir / "config.json").exists()
    ]

    prompt_count = _merge_jsonl_unique(
        [shard_dir / "prompts.jsonl" for shard_dir in shard_dirs],
        merged_dir / "prompts.jsonl",
        key_fn=lambda row: row["id"],
        sort_key=lambda row: (row.get("category", ""), row["id"]),
    )
    sample_count = _merge_jsonl_unique(
        [shard_dir / "sample_outputs.jsonl" for shard_dir in shard_dirs],
        merged_dir / "sample_outputs.jsonl",
        key_fn=lambda row: (row["condition"], row["record_id"]),
        sort_key=lambda row: (row["condition"], row["record_id"]),
    )
    merged_config = dict(shard_configs[0]) if shard_configs else {}
    merged_config.update(
        {
            "model": model,
            "dataset": _dataset_path(dataset),
            "seed": seed,
            "num_shards": num_shards,
            "shard_run_dirs": [str(path.name) for path in shard_dirs],
            "n_prompts_after_sharding": prompt_count,
            "n_rows_completed": sample_count,
        }
    )
    (merged_dir / "config.json").write_text(json.dumps(merged_config, indent=2))
    summary = {
        "model": model,
        "dataset": dataset,
        "n_prompts": prompt_count,
        "n_sample_rows": sample_count,
        "conditions": sorted({row["condition"] for row in _load_jsonl(merged_dir / "sample_outputs.jsonl")}),
    }
    (merged_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _commit_volumes()
    return (
        f"exp12 shard merge complete: model={model}, dataset={dataset}, run_name={run_name}, "
        f"prompts={prompt_count}, rows={sample_count}"
    )


@app.function(
    gpu="B200",
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def smoke_run(
    model: str = "gemma3_4b",
    run_name: str = "exp12_smoke",
) -> str:
    _setup()
    dataset_path = _dataset_path(DEFAULT_DATASET)
    run_dir = f"/root/results/exp12_free_running_abc_graft/{run_name}"
    _run_exp12(
        model=model,
        run_dir=run_dir,
        dataset_path=dataset_path,
        n_prompts=2,
        max_new_tokens=64,
        batch_size=min(8, ABC_BATCH_HINTS[model]),
        seed=0,
        shard_index=0,
        num_shards=1,
        limit_prompts=2,
        resume=False,
        prompt_seed=0,
    )
    _commit_volumes()
    return f"exp12 smoke complete: model={model}, run_name={run_name}"


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
    prompt_seed: int = 0,
) -> str:
    print("=" * 60)
    print(
        f"EXP12 BALANCED 10GPU ORCHESTRATOR — run_prefix={run_prefix}, dataset={dataset}, "
        f"shard_map={BALANCED_10GPU_SHARDS_EXP12}, n_prompts={n_prompts}, prompt_seed={prompt_seed}"
    )
    print("=" * 60)

    per_model_calls: dict[str, list[tuple[int, Any]]] = {}
    for model in BALANCED_10GPU_SHARDS_EXP12:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_10GPU_SHARDS_EXP12[model]
        model_batch_size = ABC_BATCH_HINTS[model]
        calls: list[tuple[int, Any]] = []
        for shard_index in range(num_shards):
            call = full_model_shard_run.spawn(
                model,
                run_name,
                shard_index,
                num_shards,
                n_prompts,
                512,
                n_prompts,
                0,
                dataset,
                model_batch_size,
                prompt_seed,
            )
            calls.append((shard_index, call))
            print(
                f"spawned model={model} shard={shard_index}/{num_shards} batch_size={model_batch_size} "
                f"call_id={getattr(call, 'object_id', call)}",
                flush=True,
            )
        per_model_calls[model] = calls

    failed_models: list[str] = []
    for model in BALANCED_10GPU_SHARDS_EXP12:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_10GPU_SHARDS_EXP12[model]
        calls = per_model_calls[model]
        model_ok = True
        for shard_index, call in calls:
            print(f"[wait] model={model} shard={shard_index}/{num_shards}", flush=True)
            try:
                result = call.get()
                print(result)
            except Exception as exc:
                print(f"[ERROR] model={model} shard={shard_index}/{num_shards} failed: {exc}", flush=True)
                model_ok = False
        if model_ok:
            merge_result = merge_model_shards.remote(
                run_name,
                model=model,
                dataset=dataset,
                num_shards=num_shards,
                seed=0,
            )
            print(merge_result)
        else:
            failed_models.append(model)
            print(f"[SKIP] merge skipped for {model} due to shard failures", flush=True)

    if failed_models:
        return f"PARTIAL — failed models: {failed_models}. Re-run to resume."
    return "ALL EXP12 MODELS COMPLETE"


def _launch_balanced_10gpu_full(
    run_prefix: str,
    *,
    dataset: str,
    n_prompts: int,
    prompt_seed: int,
) -> None:
    print("Triggering exp12 remote orchestrator on deployed app...", flush=True)
    call = orchestrate_balanced_10gpu_full.spawn(run_prefix, dataset, n_prompts, prompt_seed)
    call_id = getattr(call, "object_id", call)
    print(f"EXP12 orchestrator spawned: call_id={call_id}", flush=True)
    print("Waiting for result (safe to Ctrl-C — orchestrator continues on Modal)...", flush=True)
    try:
        result = call.get()
        print(result)
    except KeyboardInterrupt:
        print(f"\nLocal client interrupted — orchestrator {call_id} continues on Modal.", flush=True)
    except Exception as exc:
        print(f"Local client error: {exc}", flush=True)
        print(f"EXP12 orchestrator {call_id} may still be running on Modal.", flush=True)


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    run_name: str = "exp12_eval_v1",
    model: str = "gemma3_4b",
    dataset: str = DEFAULT_DATASET,
    n_prompts: int = 1400,
    prompt_seed: int = 0,
) -> None:
    if mode == "smoke":
        print(smoke_run.remote(model, run_name))
        return
    if mode == "balanced-10gpu-full":
        _launch_balanced_10gpu_full(
            run_name,
            dataset=dataset,
            n_prompts=n_prompts,
            prompt_seed=prompt_seed,
        )
        return
    raise ValueError("Unknown mode. Use 'smoke' or 'balanced-10gpu-full'.")
