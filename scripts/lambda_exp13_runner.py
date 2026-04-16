#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from urllib import request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = Path("/Users/Yifan/Research/.env")
DEFAULT_SSH_KEY = Path("/Users/Yifan/Research/.ssh/id_ed25519")
DEFAULT_SSH_PUB = Path("/Users/Yifan/Research/.ssh/id_ed25519.pub")
API_ROOT = "https://cloud.lambda.ai/api/v1"
SSH_USER_CANDIDATES = ["ubuntu", "root", "lambda"]
DEFAULT_BUDGET = 500.0


FULL_INSTANCE_CANDIDATES = [
    "gpu_8x_a100_80gb_sxm4",
    "gpu_8x_h100_sxm5",
    "gpu_8x_b200_sxm6",
]
SMOKE_INSTANCE_CANDIDATES = [
    "gpu_1x_h100_pcie",
    "gpu_1x_gh200",
    "gpu_1x_b200_sxm6",
]


def _load_env(env_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key, value)
    return env


def _api_request(env: dict[str, str], method: str, path: str, payload: dict | None = None) -> dict:
    token = env.get("LAMBDA_API_KEY")
    if not token:
        raise RuntimeError("LAMBDA_API_KEY is missing from env")
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.7.1",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(f"{API_ROOT}{path}", data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_instance_types(env: dict[str, str]) -> dict:
    return _api_request(env, "GET", "/instance-types")["data"]


def _pick_instance_type(env: dict[str, str], *, mode: str) -> tuple[str, str, float, str]:
    data = _get_instance_types(env)
    candidates = SMOKE_INSTANCE_CANDIDATES if mode == "smoke" else FULL_INSTANCE_CANDIDATES
    for name in candidates:
        payload = data.get(name)
        if not payload:
            continue
        regions = payload.get("regions_with_capacity_available", [])
        if not regions:
            continue
        region = regions[0]["name"]
        price_dollars = payload["instance_type"]["price_cents_per_hour"] / 100.0
        desc = payload["instance_type"]["description"]
        return name, region, price_dollars, desc
    raise RuntimeError(f"No viable Lambda instance type with capacity for mode={mode}: {candidates}")


def _estimate_hours(mode: str, n_prompts: int) -> float:
    if mode == "smoke":
        return 1.5
    return 9.0 if n_prompts >= 600 else 6.5


def _projected_cost(price_per_hour: float, hours: float) -> float:
    return price_per_hour * hours * 1.15


def _launch_instance(env: dict[str, str], *, mode: str, name: str, instance_type: str, region: str) -> str:
    ssh_pub = DEFAULT_SSH_PUB.read_text().strip()
    user_data = f"""#cloud-config
runcmd:
  - mkdir -p /home/ubuntu/.ssh || true
  - printf '%s\\n' '{ssh_pub}' >> /home/ubuntu/.ssh/authorized_keys || true
  - chown -R ubuntu:ubuntu /home/ubuntu/.ssh || true
  - chmod 700 /home/ubuntu/.ssh || true
  - chmod 600 /home/ubuntu/.ssh/authorized_keys || true
  - mkdir -p /root/.ssh || true
  - printf '%s\\n' '{ssh_pub}' >> /root/.ssh/authorized_keys || true
  - chmod 700 /root/.ssh || true
  - chmod 600 /root/.ssh/authorized_keys || true
"""
    payload = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": ["yifan"],
        "name": name,
        "user_data": user_data,
    }
    resp = _api_request(env, "POST", "/instance-operations/launch", payload)
    instance_ids = resp["data"]["instance_ids"]
    if len(instance_ids) != 1:
        raise RuntimeError(f"Unexpected launch response: {resp}")
    return instance_ids[0]


def _get_instance(env: dict[str, str], instance_id: str) -> dict:
    return _api_request(env, "GET", f"/instances/{instance_id}")["data"]


def _wait_for_active(env: dict[str, str], instance_id: str, *, timeout_s: int = 1800) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        data = _get_instance(env, instance_id)
        status = data.get("status")
        ip = data.get("ip")
        print(f"[lambda] poll instance={instance_id} status={status} ip={ip}", flush=True)
        if status == "active" and ip:
            return data
        time.sleep(15)
    raise TimeoutError(f"Timed out waiting for instance {instance_id} to become active")


def _terminate_instance(env: dict[str, str], instance_id: str) -> None:
    _api_request(env, "POST", "/instance-operations/terminate", {"instance_ids": [instance_id]})


def _ssh_base(ip: str, ssh_key: Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(ssh_key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]


def _scp_base(ip: str, ssh_key: Path) -> list[str]:
    return [
        "scp",
        "-i",
        str(ssh_key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]


def _rsync_ssh_arg(ssh_key: Path) -> str:
    return (
        f"ssh -i {shlex.quote(str(ssh_key))} "
        "-o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null"
    )


def _remote_home(remote_user: str) -> Path:
    return Path("/root") if remote_user == "root" else Path(f"/home/{remote_user}")


def _remote_repo(remote_user: str) -> Path:
    return _remote_home(remote_user) / "structral-semantic-features"


def _remote_probe_dir(remote_user: str) -> Path:
    return _remote_home(remote_user) / ".cache" / "exp11_tuned_lens_probes_v3"


def _run(cmd: list[str]) -> None:
    print("[exec]", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _ensure_local_probes(mode: str) -> None:
    models = ["gemma3_4b"] if mode == "smoke" else [
        "gemma3_4b",
        "llama31_8b",
        "qwen3_4b",
        "mistral_7b",
        "olmo2_7b",
        "deepseek_v2_lite",
    ]
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/sync_exp11_tuned_lens_probes.py",
            "--models",
            *models,
            "--variants",
            "pt",
            "it",
        ]
    )


def _wait_for_ssh(ip: str, ssh_key: Path, *, timeout_s: int = 900) -> str:
    start = time.time()
    while time.time() - start < timeout_s:
        for user in SSH_USER_CANDIDATES:
            cmd = _ssh_base(ip, ssh_key) + [
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                f"{user}@{ip}",
                "echo ready",
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return user
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for SSH access to {ip}")


def _bootstrap_remote(ip: str, ssh_key: Path, *, remote_user: str) -> None:
    bootstrap = """
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y rsync curl git build-essential
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
mkdir -p ~/.cache
"""
    _run(_ssh_base(ip, ssh_key) + [f"{remote_user}@{ip}", bootstrap])


def _sync_repo(ip: str, ssh_key: Path, *, remote_user: str) -> None:
    excludes = [
        ".git",
        ".venv",
        "results",
        "logs",
        "cache",
    ]
    cmd = [
        "rsync",
        "-az",
        "--delete",
        "-e",
        _rsync_ssh_arg(ssh_key),
    ]
    for item in excludes:
        cmd.extend(["--exclude", item])
    cmd.extend([f"{ROOT}/", f"{remote_user}@{ip}:{_remote_repo(remote_user)}/"])
    _run(cmd)


def _sync_remote_env(ip: str, ssh_key: Path, env: dict[str, str], *, remote_user: str) -> None:
    hf_token = env.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN missing from env; needed on remote for gated model downloads")
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write(f"export HF_TOKEN={json.dumps(hf_token)}\n")
        tmp.write(f"export HUGGINGFACE_HUB_TOKEN={json.dumps(hf_token)}\n")
        tmp_path = Path(tmp.name)
    try:
        _run(_scp_base(ip, ssh_key) + [str(tmp_path), f"{remote_user}@{ip}:{_remote_repo(remote_user)}/.remote_env.sh"])
    finally:
        tmp_path.unlink(missing_ok=True)


def _sync_probes(ip: str, ssh_key: Path, *, mode: str, remote_user: str) -> None:
    local_probe_root = Path.home() / ".cache" / "exp11_tuned_lens_probes_v3"
    models = ["gemma3_4b"] if mode == "smoke" else [
        "gemma3_4b",
        "llama31_8b",
        "qwen3_4b",
        "mistral_7b",
        "olmo2_7b",
        "deepseek_v2_lite",
    ]
    _run(_ssh_base(ip, ssh_key) + [f"{remote_user}@{ip}", f"mkdir -p {shlex.quote(str(_remote_probe_dir(remote_user)))}"])
    for model in models:
        src = local_probe_root / model
        if not src.exists():
            raise FileNotFoundError(f"Missing local probe dir: {src}")
        _run(
            [
                "rsync",
                "-az",
                "-e",
                _rsync_ssh_arg(ssh_key),
                f"{src}/",
                f"{remote_user}@{ip}:{_remote_probe_dir(remote_user)}/{model}/",
            ]
        )


def _run_remote(ip: str, ssh_key: Path, *, mode: str, run_name: str, n_prompts: int, remote_user: str) -> None:
    remote_repo = _remote_repo(remote_user)
    remote_probe_dir = _remote_probe_dir(remote_user)
    if mode == "smoke":
        remote_cmd = f"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd {shlex.quote(str(remote_repo))}
source .remote_env.sh
uv sync
bash scripts/run_exp13_exp14_local.sh --mode smoke --model gemma3_4b --smoke-prompts 8 --run-name {shlex.quote(run_name)} --run-root results/exp13/{shlex.quote(run_name)} --smoke-gpu 0 --tuned-lens-dir {shlex.quote(str(remote_probe_dir))}
"""
    else:
        remote_cmd = f"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd {shlex.quote(str(remote_repo))}
source .remote_env.sh
uv sync
bash scripts/run_exp13_exp14_local.sh --mode full --n-prompts {n_prompts} --run-name {shlex.quote(run_name)} --run-root results/exp13/{shlex.quote(run_name)} --tuned-lens-dir {shlex.quote(str(remote_probe_dir))}
"""
    _run(_ssh_base(ip, ssh_key) + [f"{remote_user}@{ip}", remote_cmd])


def _sync_results_back(ip: str, ssh_key: Path, run_name: str, *, remote_user: str) -> None:
    local_dir = ROOT / "results" / "exp13" / run_name
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-az",
            "-e",
            _rsync_ssh_arg(ssh_key),
            f"{remote_user}@{ip}:{_remote_repo(remote_user)}/results/exp13/{run_name}/",
            f"{local_dir}/",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Lambda smoke/full runs for exp13 full + exp14.")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
    parser.add_argument("--run-name", default=f"exp13_exp14_lambda_{int(time.time())}")
    parser.add_argument("--n-prompts", type=int, default=600)
    parser.add_argument("--soft-budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print instance choice and projected cost only; do not launch.")
    args = parser.parse_args()

    env = _load_env(args.env_file)
    effective_prompts = args.n_prompts

    if args.mode == "full":
        try:
            instance_type, region, hourly, description = _pick_instance_type(env, mode="full")
        except RuntimeError:
            instance_type, region, hourly, description = _pick_instance_type(env, mode="smoke")
            raise RuntimeError(
                "No viable 8x high-memory Lambda node currently has capacity. "
                "The smoke-capable types are available, but the full run should wait for better capacity."
            ) from None
        projected = _projected_cost(hourly, _estimate_hours("full", effective_prompts))
        if projected > args.soft_budget and effective_prompts == 600:
            effective_prompts = 400
            projected = _projected_cost(hourly, _estimate_hours("full", effective_prompts))
        print(
            json.dumps(
                {
                    "instance_type": instance_type,
                    "region": region,
                    "description": description,
                    "hourly_price_dollars": hourly,
                    "effective_prompts": effective_prompts,
                    "projected_cost_with_safety": projected,
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        instance_type, region, hourly, description = _pick_instance_type(env, mode="smoke")
        projected = _projected_cost(hourly, _estimate_hours("smoke", 8))
        print(
            json.dumps(
                {
                    "instance_type": instance_type,
                    "region": region,
                    "description": description,
                    "hourly_price_dollars": hourly,
                    "projected_cost_with_safety": projected,
                },
                indent=2,
            ),
            flush=True,
        )

    if args.dry_run:
        print("[lambda] dry-run only; not launching an instance", flush=True)
        return

    instance_id = None
    try:
        _ensure_local_probes(args.mode)
        instance_id = _launch_instance(
            env,
            mode=args.mode,
            name=args.run_name,
            instance_type=instance_type,
            region=region,
        )
        data = _wait_for_active(env, instance_id)
        ip = data["ip"]
        print(f"[lambda] active ip={ip}", flush=True)
        remote_user = _wait_for_ssh(ip, args.ssh_key)
        print(f"[lambda] ssh ready as user={remote_user}", flush=True)
        _bootstrap_remote(ip, args.ssh_key, remote_user=remote_user)
        _sync_repo(ip, args.ssh_key, remote_user=remote_user)
        _sync_remote_env(ip, args.ssh_key, env, remote_user=remote_user)
        _sync_probes(ip, args.ssh_key, mode=args.mode, remote_user=remote_user)
        _run_remote(ip, args.ssh_key, mode=args.mode, run_name=args.run_name, n_prompts=effective_prompts, remote_user=remote_user)
        _sync_results_back(ip, args.ssh_key, args.run_name, remote_user=remote_user)
        print(f"[lambda] results synced to results/exp13/{args.run_name}", flush=True)
    finally:
        if instance_id and not args.keep_instance:
            try:
                _terminate_instance(env, instance_id)
                print(f"[lambda] terminated instance {instance_id}", flush=True)
            except Exception as exc:
                print(f"[lambda] terminate warn: {exc}", flush=True)


if __name__ == "__main__":
    main()
