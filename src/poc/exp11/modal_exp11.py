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
import tempfile
from pathlib import Path
from typing import Any

import modal
import torch

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.tuned_lens import _load_probes
from src.poc.cross_model.utils import load_dataset
from src.poc.exp11.run import _apply_prompt_shard, _random_subsample, _sample_prompts


VALID_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b", "deepseek_v2_lite"]
# v10 models (legacy, without deepseek). Retained for the 5-model 10-GPU orchestrator
# so existing v10 code paths don't suddenly pull deepseek.
VALID_MODELS_V10 = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
SMOKE_MODEL = "gemma3_4b"
DEFAULT_DATASET = "exp3"
DEFAULT_TUNED_RUN_NAME = "exp11_exp3_all2936_tunedlens_v1"


def _repo_root_for_local_paths() -> Path:
    here = Path(__file__).resolve()
    if len(here.parents) >= 4:
        return here.parents[3]
    return Path.cwd()


DATASET_PATHS = {
    "eval_v2": "/root/data/eval_dataset_v2.jsonl",
    "exp3": "/root/data/exp3_dataset.jsonl",
}
LOCAL_DATASET_PATHS = {
    "eval_v2": str((_repo_root_for_local_paths() / "data" / "eval_dataset_v2.jsonl")),
    "exp3": str((_repo_root_for_local_paths() / "data" / "exp3_dataset.jsonl")),
}
BALANCED_8GPU_SHARDS = {
    "llama31_8b": 2,
    "mistral_7b": 2,
    "olmo2_7b": 2,
    "gemma3_4b": 1,
    "qwen3_4b": 1,
}
BALANCED_10GPU_SHARDS = {model: 2 for model in VALID_MODELS_V10}
# v11.1 teacher-forced layout: 6 models across 10 GPUs.
# 2 GPUs per large model (data-parallel across prompts), 1 GPU per small model.
BALANCED_10GPU_SHARDS_V11 = {
    "gemma3_4b": 2,
    "llama31_8b": 2,
    "mistral_7b": 2,
    "olmo2_7b": 2,
    "qwen3_4b": 1,
    "deepseek_v2_lite": 1,
}
FULL_BATCH_HINTS = {
    "gemma3_4b": 128,
    "qwen3_4b": 128,
    "llama31_8b": 64,
    "mistral_7b": 64,
    "olmo2_7b": 64,
    "deepseek_v2_lite": 64,
}
# Tuned-lens runs load 2 models + probes + baseline cache; B200 192GB
TUNED_BATCH_HINTS = {
    "gemma3_4b": 64,
    "qwen3_4b": 64,
    "llama31_8b": 48,
    "mistral_7b": 48,
    "olmo2_7b": 48,
    "deepseek_v2_lite": 48,
}
# v11.1: 5 teacher-forced pipelines per prompt (C, A', B, A'_tmpl, B2). C's residuals are
# cached as teacher baseline for all teacher-forced branches. Smaller batches to
# stay under B200 192 GB.
TUNED_BATCH_HINTS_V11 = {
    "gemma3_4b": 64,
    "qwen3_4b": 64,
    "llama31_8b": 48,
    "mistral_7b": 48,
    "olmo2_7b": 48,
    "deepseek_v2_lite": 48,
}
TUNED_LENS_GCS_ROOT = "gs://pt-vs-it-results/tuned_lens_probes_v3"
TUNED_LENS_VOLUME_NAME = "exp11-tuned-lens-probes-v3"
_HF_MODELS_TO_BAKE = sorted(
    {
        model_id_for_variant(get_spec(model_name), variant)
        for model_name in VALID_MODELS
        for variant in ("pt", "it")
    }
)

app = modal.App("exp11-mlp-graft")
results_vol = modal.Volume.from_name("exp11-results", create_if_missing=True)
probes_vol = modal.Volume.from_name(TUNED_LENS_VOLUME_NAME, create_if_missing=True)
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
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "einops",
        "tqdm",
        "matplotlib",
        "google-cloud-storage",
        "google-auth",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("data/eval_dataset_v2.jsonl", remote_path="/root/data/eval_dataset_v2.jsonl")
    .add_local_file("data/exp3_dataset.jsonl", remote_path="/root/data/exp3_dataset.jsonl")
)

VOLUME_MOUNTS = {
    "/root/results": results_vol,
    "/root/tuned_lens_probes": probes_vol,
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
    # Enables deterministic cuBLAS kernels for subprocessed Python runs.
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def _materialize_gcp_adc_from_env() -> Path:
    payload = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not payload:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON is not set")
    creds_path = Path(tempfile.gettempdir()) / "exp11_gcp_adc.json"
    creds_path.write_text(payload)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
    return creds_path


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


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
    chunk_size: int | None = None,
    resume: bool = False,
    readout_mode: str = "raw",
    tuned_lens_dir: str | None = None,
    teacher_forced: bool = False,
    prompt_seed: int | None = None,
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
        "--readout-mode",
        readout_mode,
    ]
    if limit_prompts is not None:
        cmd.extend(["--limit-prompts", str(limit_prompts)])
    if chunk_size is not None:
        cmd.extend(["--chunk-size", str(chunk_size)])
    if resume:
        cmd.append("--resume")
    if tuned_lens_dir is not None:
        cmd.extend(["--tuned-lens-dir", tuned_lens_dir])
    if teacher_forced:
        cmd.append("--teacher-forced")
    if prompt_seed is not None:
        cmd.extend(["--prompt-seed", str(prompt_seed)])
    _run_subprocess(cmd, seed=seed)


def _analyze_exp11(run_dir: str, *, seed: int) -> None:
    _run_subprocess(
        ["python", "-m", "src.poc.exp11.analyze", "--run-dir", run_dir],
        seed=seed,
    )


def _dataset_path(dataset: str) -> str:
    try:
        remote_path = DATASET_PATHS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset '{dataset}'. Use one of {sorted(DATASET_PATHS)}.") from exc
    if Path(remote_path).exists():
        return remote_path
    return LOCAL_DATASET_PATHS[dataset]


def _dataset_prompt_count(dataset: str) -> int:
    return len(load_dataset(_dataset_path(dataset)))


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


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


def _verify_probe_mount(model: str) -> None:
    spec = get_spec(model)
    for variant, expected in (("pt", spec.n_layers), ("it", spec.n_layers)):
        probe_dir = Path(f"/root/tuned_lens_probes/{model}/{variant}")
        count = len(list(probe_dir.glob("probe_layer_*.pt")))
        if count != expected:
            raise RuntimeError(
                f"Probe mount incomplete for {model}/{variant}: expected {expected} probes, found {count} at {probe_dir}"
            )


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


def _concat_jsonl_files(shard_paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for shard_path in shard_paths:
            if not shard_path.exists():
                continue
            with open(shard_path) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)


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


def _local_probe_cache_root() -> Path:
    return Path.home() / ".cache" / "exp11_tuned_lens_probes_v3"


def _download_probe_file(remote: str, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()
    stale_gstmp = final_path.parent / f"{final_path.name}_.gstmp"
    if stale_gstmp.exists():
        stale_gstmp.unlink()
    gcloud_cmd = ["gcloud", "storage", "cp", remote, str(tmp_path)]
    result = subprocess.run(
        gcloud_cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        gsutil_cmd = [
            "gsutil",
            "-o",
            "GSUtil:parallel_process_count=1",
            "cp",
            remote,
            str(tmp_path),
        ]
        subprocess.run(gsutil_cmd, check=True)
    if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        raise RuntimeError(f"Probe download failed for {remote}: no completed file at {tmp_path}")
    os.replace(tmp_path, final_path)


def _validate_local_probe_dir(model: str, variant: str, probe_dir: Path) -> None:
    spec = get_spec(model)
    expected = spec.n_layers
    probes = _load_probes(probe_dir, d_model=spec.d_model, device=torch.device("cpu"))
    expected_keys = list(range(expected))
    actual_keys = sorted(probes)
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"Probe validation failed for {model}/{variant}: "
            f"expected keys {expected_keys[:5]}...{expected_keys[-1] if expected_keys else 'none'}, "
            f"got {actual_keys[:5]}...{actual_keys[-1] if actual_keys else 'none'}"
        )


def _download_probe_file_via_gcs_client(client: Any, bucket_name: str, blob_name: str, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(tmp_path))
    if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        raise RuntimeError(f"GCS download failed for gs://{bucket_name}/{blob_name}: no completed file at {tmp_path}")
    os.replace(tmp_path, final_path)


def _stage_tuned_lens_probes(models: list[str] | None = None) -> None:
    models = models or list(VALID_MODELS)
    cache_root = _local_probe_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "create", TUNED_LENS_VOLUME_NAME],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for model in models:
        spec = get_spec(model)
        for variant in ("pt", "it"):
            local_dir = cache_root / model / variant
            local_dir.mkdir(parents=True, exist_ok=True)
            expected = spec.n_layers
            stage_marker = local_dir / ".modal_uploaded"
            print(f"[stage] ensuring probes for {model}/{variant}", flush=True)
            for layer_idx in range(expected):
                final_path = local_dir / f"probe_layer_{layer_idx}.pt"
                if final_path.exists() and final_path.stat().st_size > 0:
                    continue
                remote = f"{TUNED_LENS_GCS_ROOT}/{model}/{variant}/probe_layer_{layer_idx}.pt"
                print(f"[stage] download {model}/{variant} layer={layer_idx}", flush=True)
                _download_probe_file(remote, final_path)
            final_files = list(local_dir.glob("probe_layer_*.pt"))
            if len(final_files) != expected:
                raise RuntimeError(
                    f"Local staged probes incomplete for {model}/{variant}: expected {expected}, found {len(final_files)}"
                )
            print(f"[stage] validate {model}/{variant}", flush=True)
            _validate_local_probe_dir(model, variant, local_dir)
            if not stage_marker.exists():
                print(f"[stage] upload {model}/{variant} to modal volume", flush=True)
                volume = modal.Volume.from_name(TUNED_LENS_VOLUME_NAME, create_if_missing=True)
                with volume.batch_upload(force=True) as batch:
                    for path in sorted(final_files):
                        batch.put_file(str(path), f"/{model}/{variant}/{path.name}")
                stage_marker.write_text("uploaded\n")
                print(f"[stage] uploaded {model}/{variant}", flush=True)


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=32768,
    timeout=28800,
    secrets=[modal.Secret.from_name("gcp-adc-exp11")],
)
def stage_tuned_lens_probes_remote(models: list[str] | None = None) -> str:
    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    _setup()
    _materialize_gcp_adc_from_env()
    creds_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    client = storage.Client(
        project=None,
        credentials=Credentials.from_authorized_user_info(creds_info),
    )
    models = models or list(VALID_MODELS)
    bucket_name = TUNED_LENS_GCS_ROOT.removeprefix("gs://").split("/", 1)[0]
    root_prefix = TUNED_LENS_GCS_ROOT.removeprefix(f"gs://{bucket_name}/")
    staged: list[str] = []
    for model in models:
        spec = get_spec(model)
        for variant in ("pt", "it"):
            local_dir = Path("/root/tuned_lens_probes") / model / variant
            local_dir.mkdir(parents=True, exist_ok=True)
            expected = spec.n_layers
            stage_marker = local_dir / ".modal_uploaded"
            print(f"[remote-stage] ensuring probes for {model}/{variant}", flush=True)
            for layer_idx in range(expected):
                final_path = local_dir / f"probe_layer_{layer_idx}.pt"
                if final_path.exists() and final_path.stat().st_size > 0:
                    continue
                blob_name = f"{root_prefix}/{model}/{variant}/probe_layer_{layer_idx}.pt"
                print(f"[remote-stage] download {model}/{variant} layer={layer_idx}", flush=True)
                _download_probe_file_via_gcs_client(client, bucket_name, blob_name, final_path)
            print(f"[remote-stage] validate {model}/{variant}", flush=True)
            _validate_local_probe_dir(model, variant, local_dir)
            stage_marker.write_text("uploaded\n")
            staged.append(f"{model}/{variant}")
            print(f"[remote-stage] ready {model}/{variant}", flush=True)
    try:
        probes_vol.commit()
    except Exception as exc:
        print(f"[remote-stage] volume commit warn: {exc}", flush=True)
    return "staged " + ", ".join(staged)


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
    readout_mode: str = "raw",
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
    tuned_lens_dir = "/root/tuned_lens_probes" if readout_mode != "raw" else None
    if tuned_lens_dir is not None:
        _verify_probe_mount(model)
    _run_exp11(
        model=model,
        run_dir=run_dir,
        dataset_path=_dataset_path(dataset),
        n_prompts=n_prompts,
        max_new_tokens=max_new_tokens,
        batch_size=effective_batch_size,
        seed=seed,
        readout_mode=readout_mode,
        tuned_lens_dir=tuned_lens_dir,
    )
    _analyze_exp11(run_dir, seed=seed)
    _commit_volumes()
    return (
        f"exp11 full run complete: model={model}, run_name={run_name}, "
        f"dataset={dataset}, readout_mode={readout_mode}, requested_batch_size={requested_batch_size}, "
        f"effective_batch_size={effective_batch_size}, "
        f"n_prompts={n_prompts}, max_new_tokens={max_new_tokens}\n"
        f"Download with: {_results_hint(run_name)}"
    )


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
    num_shards: int = 3,
    n_prompts: int = 200,
    max_new_tokens: int = 512,
    prompts_per_chunk: int = 512,
    seed: int = 0,
    dataset: str = DEFAULT_DATASET,
    readout_mode: str = "raw",
    chunk_size: int = 64,
    batch_size: int = 32,
    teacher_forced: bool = False,
    prompt_seed: int | None = None,
) -> str:
    _setup()
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model for exp11 sharded run: {model}")
    dataset_path = _dataset_path(dataset)
    tuned_lens_dir = "/root/tuned_lens_probes" if readout_mode != "raw" else None
    if tuned_lens_dir is not None:
        _verify_probe_mount(model)
    run_dir = f"/root/results/exp11/{run_name}__shard{shard_index}of{num_shards}"
    shard_total = _count_shard_prompts(
        dataset_path,
        n_prompts=n_prompts,
        seed=seed,
        shard_index=shard_index,
        num_shards=num_shards,
        prompt_seed=prompt_seed,
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
            batch_size=batch_size,
            seed=seed,
            shard_index=shard_index,
            num_shards=num_shards,
            limit_prompts=prompts_per_chunk,
            chunk_size=chunk_size,
            resume=True,
            readout_mode=readout_mode,
            tuned_lens_dir=tuned_lens_dir,
            teacher_forced=teacher_forced,
            prompt_seed=prompt_seed,
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
        f"exp11 shard complete: model={model}, dataset={dataset}, readout_mode={readout_mode}, run_name={run_name}, "
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
    _concat_jsonl_files(
        [shard_dir / "step_metrics.jsonl" for shard_dir in shard_dirs],
        merged_dir / "step_metrics.jsonl",
    )
    secondary_payloads = [
        _read_json(shard_dir / "secondary_trajectory_stats.json")
        for shard_dir in shard_dirs
        if (shard_dir / "secondary_trajectory_stats.json").exists()
    ]
    if secondary_payloads:
        merged_secondary = {
            "readout_name": secondary_payloads[0].get("readout_name"),
            "metrics": {},
        }
        readout_name_by_pipeline: dict[str, str] = {}
        for payload in secondary_payloads:
            readout_name_by_pipeline.update(payload.get("readout_name_by_pipeline", {}))
            for pipeline, pipeline_metrics in payload.get("metrics", {}).items():
                dst_pipeline = merged_secondary["metrics"].setdefault(pipeline, {})
                for metric_name, metric_stats in pipeline_metrics.items():
                    dst_metric = dst_pipeline.setdefault(
                        metric_name,
                        {
                            "sum": [0.0] * len(metric_stats.get("sum", [])),
                            "count": [0.0] * len(metric_stats.get("count", [])),
                        },
                    )
                    for idx, value in enumerate(metric_stats.get("sum", [])):
                        dst_metric["sum"][idx] += value
                    for idx, value in enumerate(metric_stats.get("count", [])):
                        dst_metric["count"][idx] += value
        if readout_name_by_pipeline:
            merged_secondary["readout_name_by_pipeline"] = readout_name_by_pipeline
        (merged_dir / "secondary_trajectory_stats.json").write_text(json.dumps(merged_secondary, indent=2))

    merged_config = dict(shard_configs[0]) if shard_configs else {}
    merged_config.update(
        {
            "model": model,
            "dataset": _dataset_path(dataset),
            "seed": seed,
            "num_shards": num_shards,
            "shard_run_dirs": [str(path.name) for path in shard_dirs],
            "scheduler_mode": "single_prompt_data_parallel",
            "batch_size_pipeline_a": shard_configs[0].get("batch_size_pipeline_a") if shard_configs else None,
            "batch_size_pipeline_b": shard_configs[0].get("batch_size_pipeline_b") if shard_configs else None,
            "batch_size_by_pipeline": shard_configs[0].get("batch_size_by_pipeline") if shard_configs else None,
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


def _run_tuned_smoke(run_name: str, *, model: str) -> None:
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model for tuned smoke: {model}")
    print(stage_tuned_lens_probes_remote.remote([model]))
    result = full_model_shard_run.remote(
        model,
        run_name,
        0,
        1,
        4,
        16,
        4,
        0,
        DEFAULT_DATASET,
        "both",
    )
    print(result)
    merge_result = merge_model_shards.remote(
        run_name,
        model=model,
        dataset=DEFAULT_DATASET,
        num_shards=1,
        seed=0,
    )
    print(merge_result)


def _run_preflight(model: str, run_name: str, *, batch_size: int | None) -> None:
    print("=" * 60)
    print(
        f"EXP11 PREFLIGHT — model={model}, run_name={run_name}, "
        f"batch_size={batch_size or FULL_BATCH_HINTS[model]}"
    )
    print("=" * 60)
    result = preflight_model_run.remote(model, run_name, batch_size, 0)
    print(result)


def _launch_one_full(
    model: str,
    run_name: str,
    *,
    batch_size: int | None,
    preflight: bool,
    dataset: str,
    n_prompts: int,
    readout_mode: str = "raw",
) -> None:
    print("=" * 60)
    print(
        f"EXP11 FULL LAUNCH — model={model}, run_name={run_name}, "
        f"batch_size={batch_size or FULL_BATCH_HINTS[model]}, preflight={preflight}"
    )
    print("=" * 60)
    result = full_model_run.remote(
        model,
        run_name,
        n_prompts,
        512,
        batch_size,
        0,
        preflight,
        dataset,
        readout_mode,
    )
    print(result)


def _launch_all_full(
    run_prefix: str,
    *,
    batch_size: int | None,
    preflight: bool,
    dataset: str,
    n_prompts: int,
    readout_mode: str = "raw",
) -> None:
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
        call = full_model_run.spawn(
            model,
            run_name,
            n_prompts,
            512,
            requested_batch_size,
            0,
            preflight,
            dataset,
            readout_mode,
        )
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
    n_prompts: int,
    readout_mode: str = "raw",
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
            n_prompts,
            512,
            prompts_per_chunk,
            0,
            dataset,
            readout_mode,
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
    n_prompts: int,
    readout_mode: str = "raw",
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
            n_prompts=n_prompts,
            readout_mode=readout_mode,
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
                "raw",
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


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=4096,
    timeout=86400,
    secrets=[modal.Secret.from_name("gcp-adc-exp11")],
)
def orchestrate_balanced_10gpu_tuned_full(
    run_prefix: str,
    dataset: str,
    prompts_per_chunk: int,
) -> str:
    """Remote orchestrator — runs entirely on Modal, no local client dependency."""
    print("=" * 60)
    print(
        f"EXP11 REMOTE ORCHESTRATOR — run_prefix={run_prefix}, "
        f"dataset={dataset}, shard_map={BALANCED_10GPU_SHARDS}, prompts_per_chunk={prompts_per_chunk}"
    )
    print("=" * 60)
    print(stage_tuned_lens_probes_remote.remote())
    n_prompts = _dataset_prompt_count(dataset)

    per_model_calls: dict[str, list[tuple[int, Any]]] = {}
    for model in VALID_MODELS:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_10GPU_SHARDS[model]
        model_batch_size = TUNED_BATCH_HINTS[model]
        calls: list[tuple[int, Any]] = []
        for shard_index in range(num_shards):
            call = full_model_shard_run.spawn(
                model,
                run_name,
                shard_index,
                num_shards,
                n_prompts,
                512,
                prompts_per_chunk,
                0,
                dataset,
                "both",
                64,  # chunk_size
                model_batch_size,
            )
            calls.append((shard_index, call))
            print(
                f"spawned model={model} shard={shard_index}/{num_shards} "
                f"batch_size={model_batch_size} "
                f"call_id={getattr(call, 'object_id', call)}",
                flush=True,
            )
        per_model_calls[model] = calls

    failed_models: list[str] = []
    for model in VALID_MODELS:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_10GPU_SHARDS[model]
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
    return "ALL MODELS COMPLETE"


@app.function(
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=4096,
    timeout=86400,
    secrets=[modal.Secret.from_name("gcp-adc-exp11")],
)
def orchestrate_balanced_10gpu_v11_teacherforced(
    run_prefix: str,
    dataset: str = DEFAULT_DATASET,
    n_prompts: int = 400,
    prompt_seed: int = 0,
    chunk_size: int = 64,
) -> str:
    """v11.1 remote orchestrator: 6 models × 5 teacher-forced pipelines (C, A', B, A'_tmpl, B2) × 400 prompts on 10 GPUs.

    Pipelines A', B, A'_tmpl, and B2 are teacher-forced to pipeline C's (true IT
    free-run) token sequence, so cross-pipeline KL / residual-cosine share token
    history and measure graft / backbone effect without context drift.
    """
    print("=" * 60)
    print(
        f"EXP11 V11 TEACHER-FORCED ORCHESTRATOR — run_prefix={run_prefix}, "
        f"dataset={dataset}, shard_map={BALANCED_10GPU_SHARDS_V11}, "
        f"n_prompts={n_prompts}, prompt_seed={prompt_seed}, chunk_size={chunk_size}"
    )
    print("=" * 60)
    print(stage_tuned_lens_probes_remote.remote(list(BALANCED_10GPU_SHARDS_V11.keys())))

    per_model_calls: dict[str, list[tuple[int, Any]]] = {}
    for model in BALANCED_10GPU_SHARDS_V11:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_10GPU_SHARDS_V11[model]
        model_batch_size = TUNED_BATCH_HINTS_V11[model]
        calls: list[tuple[int, Any]] = []
        for shard_index in range(num_shards):
            call = full_model_shard_run.spawn(
                model,
                run_name,
                shard_index,
                num_shards,
                n_prompts,
                512,  # max_new_tokens
                n_prompts,  # prompts_per_chunk (1 pass)
                0,  # seed
                dataset,
                "both",  # readout_mode
                chunk_size,
                model_batch_size,
                True,  # teacher_forced
                prompt_seed,
            )
            calls.append((shard_index, call))
            print(
                f"spawned model={model} shard={shard_index}/{num_shards} "
                f"batch_size={model_batch_size} teacher_forced=True "
                f"call_id={getattr(call, 'object_id', call)}",
                flush=True,
            )
        per_model_calls[model] = calls

    failed_models: list[str] = []
    for model in BALANCED_10GPU_SHARDS_V11:
        run_name = f"{run_prefix}_{model}"
        num_shards = BALANCED_10GPU_SHARDS_V11[model]
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
    return "ALL V11 MODELS COMPLETE"


def _launch_balanced_10gpu_v11_teacherforced(
    run_prefix: str,
    *,
    dataset: str,
    n_prompts: int,
    prompt_seed: int,
    chunk_size: int,
) -> None:
    """Local launcher for the v11 remote orchestrator."""
    print("Triggering v11 remote orchestrator on deployed app...", flush=True)
    print(
        "IMPORTANT: First deploy with `modal deploy src/poc/exp11/modal_exp11.py`, "
        "then trigger with `modal run`.",
        flush=True,
    )
    call = orchestrate_balanced_10gpu_v11_teacherforced.spawn(
        run_prefix, dataset, n_prompts, prompt_seed, chunk_size,
    )
    call_id = getattr(call, "object_id", call)
    print(f"V11 orchestrator spawned: call_id={call_id}", flush=True)
    print("Waiting for result (safe to Ctrl-C — orchestrator continues on Modal)...", flush=True)
    try:
        result = call.get()
        print(result)
    except KeyboardInterrupt:
        print(f"\nLocal client interrupted — orchestrator {call_id} continues on Modal.", flush=True)
    except Exception as exc:
        print(f"Local client error: {exc}", flush=True)
        print(f"V11 orchestrator {call_id} may still be running on Modal.", flush=True)


def _launch_balanced_10gpu_tuned_full(
    run_prefix: str,
    *,
    dataset: str,
    prompts_per_chunk: int,
) -> None:
    """Local launcher — triggers the remote orchestrator on the deployed app."""
    print("Triggering remote orchestrator on deployed app...", flush=True)
    print(
        "IMPORTANT: First deploy with `modal deploy src/poc/exp11/modal_exp11.py`, "
        "then trigger with `modal run`.",
        flush=True,
    )
    call = orchestrate_balanced_10gpu_tuned_full.spawn(
        run_prefix, dataset, prompts_per_chunk,
    )
    call_id = getattr(call, "object_id", call)
    print(f"Orchestrator spawned: call_id={call_id}", flush=True)
    print("Waiting for result (safe to Ctrl-C — orchestrator continues on Modal)...", flush=True)
    try:
        result = call.get()
        print(result)
    except KeyboardInterrupt:
        print(f"\nLocal client interrupted — orchestrator {call_id} continues on Modal.", flush=True)
    except Exception as exc:
        print(f"Local client error: {exc}", flush=True)
        print(f"Orchestrator {call_id} may still be running on Modal.", flush=True)


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    run_name: str = DEFAULT_TUNED_RUN_NAME,
    model: str = "",
    batch_size: int = 0,
    preflight: bool = True,
    dataset: str = DEFAULT_DATASET,
    num_shards: int = 3,
    prompts_per_chunk: int = 512,
    n_prompts: int = 200,
    prompt_seed: int = 0,
    chunk_size: int = 64,
) -> None:
    if mode == "smoke":
        _run_smoke(run_name)
        return
    if mode == "tuned-smoke":
        smoke_model = model or SMOKE_MODEL
        _run_tuned_smoke(run_name, model=smoke_model)
        return
    if mode == "preflight":
        if not model:
            raise ValueError("`--model` is required for preflight mode.")
        _run_preflight(model, run_name, batch_size=batch_size or None)
        return
    if mode == "full":
        if model:
            _launch_one_full(
                model,
                run_name,
                batch_size=batch_size or None,
                preflight=preflight,
                dataset=dataset,
                n_prompts=_dataset_prompt_count(dataset),
            )
        else:
            _launch_all_full(
                run_name,
                batch_size=batch_size or None,
                preflight=preflight,
                dataset=dataset,
                n_prompts=_dataset_prompt_count(dataset),
            )
        return
    if mode == "sharded-full":
        if model:
            _launch_one_sharded_full(
                model,
                run_name,
                dataset=dataset,
                num_shards=num_shards,
                prompts_per_chunk=prompts_per_chunk,
                n_prompts=_dataset_prompt_count(dataset),
            )
        else:
            _launch_all_sharded_full(
                run_name,
                dataset=dataset,
                num_shards=num_shards,
                prompts_per_chunk=prompts_per_chunk,
                n_prompts=_dataset_prompt_count(dataset),
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
    if mode == "stage-tuned-lens-probes":
        _stage_tuned_lens_probes([model] if model else None)
        return
    if mode == "balanced-10gpu-tuned-full":
        if model:
            raise ValueError("`--model` is not supported for balanced-10gpu-tuned-full mode.")
        _launch_balanced_10gpu_tuned_full(
            run_name,
            dataset=dataset,
            prompts_per_chunk=prompts_per_chunk,
        )
        return
    if mode == "balanced-10gpu-v11-teacherforced":
        if model:
            raise ValueError("`--model` is not supported for balanced-10gpu-v11-teacherforced mode.")
        _launch_balanced_10gpu_v11_teacherforced(
            run_name,
            dataset=dataset,
            n_prompts=n_prompts,
            prompt_seed=prompt_seed,
            chunk_size=chunk_size,
        )
        return
    print(
        "Unknown mode. Use 'smoke', 'tuned-smoke', 'preflight', 'full', 'sharded-full', "
        "'balanced-8gpu-full', 'stage-tuned-lens-probes', 'balanced-10gpu-tuned-full', "
        "or 'balanced-10gpu-v11-teacherforced'."
    )
