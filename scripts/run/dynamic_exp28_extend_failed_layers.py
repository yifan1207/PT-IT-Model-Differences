#!/usr/bin/env python3
"""Dynamic Exp28 crosscoder extension scheduler.

This helper is intentionally conservative: it only launches a layer on a GPU
whose memory is essentially free, and it decides whether to extend again from
the layer's saved config metrics after each training job completes.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Job:
    layer: int
    target_steps: int
    gpu: int
    process: subprocess.Popen
    log_path: Path


def env_str(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"missing required env var {name}")
    return value


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def layer_config(run_root: Path, layer: int) -> dict:
    path = run_root / "dictionaries" / f"layer_{layer}" / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def checkpoint_step(run_root: Path, layer: int) -> int:
    path = run_root / "dictionaries" / f"layer_{layer}" / "checkpoint_latest.pt"
    if not path.exists():
        return 0
    # Avoid importing torch in the scheduler. The checkpoint step is only used
    # for logging/target decisions, and config.json is authoritative on finish.
    stat = path.stat()
    return 1 if stat.st_size > 0 else 0


def config_steps(data: dict) -> int:
    return int(data.get("steps", data.get("extra", {}).get("steps", 0)) or 0)


def metrics_pass(data: dict, min_ve: float) -> bool:
    metrics = data.get("metrics", {})
    ve_pt = float(metrics.get("heldout_variance_explained_pt", -1))
    ve_it = float(metrics.get("heldout_variance_explained_it", -1))
    return ve_pt >= min_ve and ve_it >= min_ve


def metric_summary(data: dict) -> str:
    metrics = data.get("metrics", {})
    ve_pt = float(metrics.get("heldout_variance_explained_pt", -1))
    ve_it = float(metrics.get("heldout_variance_explained_it", -1))
    return f"ve_pt={ve_pt:.4f} ve_it={ve_it:.4f}"


def free_gpus(mem_threshold_mb: int) -> list[int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free: list[int] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        idx_raw, mem_raw, *_ = line.split(",")
        idx = int(idx_raw.strip())
        mem = int(mem_raw.strip())
        if mem < mem_threshold_mb:
            free.append(idx)
    return free


def desired_target(run_root: Path, layer: int, min_ve: float, base_steps: int, step_increment: int, max_steps: int) -> int | None:
    data = layer_config(run_root, layer)
    if data and metrics_pass(data, min_ve):
        print(f"[exp28-dyn] layer={layer} PASS {metric_summary(data)}", flush=True)
        return None
    steps = config_steps(data)
    if steps < base_steps:
        return base_steps
    if steps < max_steps:
        return min(max_steps, steps + step_increment)
    print(f"[exp28-dyn] layer={layer} STILL_LOW at max_steps={max_steps} {metric_summary(data)}", flush=True)
    return None


def launch_job(
    *,
    root: Path,
    run_root: Path,
    run_name: str,
    model: str,
    py_runner: str,
    layer: int,
    target_steps: int,
    gpu: int,
    dict_size: int,
    k: int,
    batch_tokens: int,
    checkpoint_every: int,
) -> Job:
    log_path = run_root / "logs" / f"dynamic_extend_layer_{layer}_to_{target_steps}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        *shlex.split(py_runner),
        "-m",
        "src.poc.exp28_late_mlp_crosscoder_mediation",
        "train",
        "--run-root",
        str(run_root),
        "--layers",
        str(layer),
        "--dict-size",
        str(dict_size),
        "--k",
        str(k),
        "--steps",
        str(target_steps),
        "--batch-tokens",
        str(batch_tokens),
        "--checkpoint-every",
        str(checkpoint_every),
        "--device",
        "cuda:0",
        "--resume",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["RUN_NAME"] = run_name
    env["MODEL"] = model
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}"
    log_f = log_path.open("ab")
    print(
        f"[exp28-dyn] launch layer={layer} target_steps={target_steps} gpu={gpu} log={log_path}",
        flush=True,
    )
    process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return Job(layer=layer, target_steps=target_steps, gpu=gpu, process=process, log_path=log_path)


def main() -> None:
    root = Path(env_str("ROOT", str(Path(__file__).resolve().parents[2])))
    run_root = Path(env_str("RUN_ROOT"))
    run_name = env_str("RUN_NAME", run_root.name)
    model = env_str("MODEL", "llama31_8b")
    py_runner = env_str("PY_RUNNER", "uv run python")
    layers = [int(x) for x in env_str("LAYERS", "19 20 21 22 23 24 25").split()]
    min_ve = float(env_str("MIN_VE", "0.80"))
    base_steps = env_int("BASE_STEPS", 8000)
    step_increment = env_int("STEP_INCREMENT", 4000)
    max_steps = env_int("MAX_STEPS", 12000)
    dict_size = env_int("DICT_SIZE", 65536)
    k = env_int("K", 512)
    batch_tokens = env_int("BATCH_TOKENS", 8192)
    checkpoint_every = env_int("CHECKPOINT_EVERY", 1000)
    mem_threshold_mb = env_int("GPU_FREE_MEM_MB", 1000)
    poll_seconds = env_int("POLL_SECONDS", 30)

    pending = set(layers)
    active: dict[int, Job] = {}
    print(f"[exp28-dyn] start layers={layers} base={base_steps} max={max_steps}", flush=True)

    while pending or active:
        for layer, job in list(active.items()):
            ret = job.process.poll()
            if ret is None:
                continue
            del active[layer]
            print(f"[exp28-dyn] finished layer={layer} target={job.target_steps} ret={ret}", flush=True)
            if ret != 0:
                raise SystemExit(f"layer={layer} failed; see {job.log_path}")
            pending.add(layer)

        busy_gpus = {job.gpu for job in active.values()}
        available = [gpu for gpu in free_gpus(mem_threshold_mb) if gpu not in busy_gpus]
        launched = False
        for layer in sorted(list(pending)):
            if not available:
                break
            target = desired_target(run_root, layer, min_ve, base_steps, step_increment, max_steps)
            pending.remove(layer)
            if target is None:
                continue
            if checkpoint_step(run_root, layer):
                print(f"[exp28-dyn] layer={layer} has checkpoint; resuming", flush=True)
            gpu = available.pop(0)
            active[layer] = launch_job(
                root=root,
                run_root=run_root,
                run_name=run_name,
                model=model,
                py_runner=py_runner,
                layer=layer,
                target_steps=target,
                gpu=gpu,
                dict_size=dict_size,
                k=k,
                batch_tokens=batch_tokens,
                checkpoint_every=checkpoint_every,
            )
            launched = True

        if not launched:
            status = " ".join(
                f"L{job.layer}->s{job.target_steps}@gpu{job.gpu}" for job in active.values()
            )
            print(
                f"[exp28-dyn] wait pending={sorted(pending)} active=[{status}] free={available} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
                flush=True,
            )
        time.sleep(poll_seconds)

    print("[exp28-dyn] complete", flush=True)


if __name__ == "__main__":
    main()
