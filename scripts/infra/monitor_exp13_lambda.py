#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results" / "exp13"
DEFAULT_MONITOR_DIR = RESULTS_ROOT / "exp13exp14_lambda_monitor"
DEFAULT_SMOKE_RUN = "exp13exp14_lambda_smoke_20260415"
DEFAULT_FULL_RUN = "exp13exp14_lambda_full_20260415"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(text.rstrip() + "\n")


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def _summary_exists(run_name: str) -> bool:
    return (RESULTS_ROOT / run_name / "exp13_full_summary.json").exists()


def _latest_status(smoke_run: str, full_run: str) -> dict[str, Any]:
    return {
        "smoke_summary_exists": _summary_exists(smoke_run),
        "full_summary_exists": _summary_exists(full_run),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Idempotent Lambda monitor/launcher for exp13 full + exp14.")
    parser.add_argument("--monitor-dir", type=Path, default=DEFAULT_MONITOR_DIR)
    parser.add_argument("--smoke-run", default=DEFAULT_SMOKE_RUN)
    parser.add_argument("--full-run", default=DEFAULT_FULL_RUN)
    parser.add_argument("--n-prompts", type=int, default=600)
    args = parser.parse_args()

    status_path = args.monitor_dir / "status.json"
    log_path = args.monitor_dir / "monitor.log"
    status = _read_json(status_path)
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    current = _latest_status(args.smoke_run, args.full_run)

    event: dict[str, Any] = {"checked_at": now, **current}

    if current["full_summary_exists"]:
        event["state"] = "complete"
        status.update(event)
        _write_json(status_path, status)
        _append_log(log_path, f"[{now}] full run already complete")
        print(json.dumps(event, indent=2))
        return

    if not current["smoke_summary_exists"]:
        cmd = ["uv", "run", "python", "scripts/lambda_exp13_runner.py", "--mode", "smoke", "--run-name", args.smoke_run]
        code, output = _run(cmd)
        event["action"] = "launch_smoke"
        event["returncode"] = code
        event["state"] = "smoke_complete" if code == 0 else "smoke_waiting_or_failed"
        event["output_tail"] = output[-4000:]
        _append_log(log_path, f"[{now}] launch_smoke rc={code}\n{output}")
        status.update(event)
        _write_json(status_path, status)
        print(json.dumps(event, indent=2))
        return

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/lambda_exp13_runner.py",
        "--mode",
        "full",
        "--run-name",
        args.full_run,
        "--n-prompts",
        str(args.n_prompts),
    ]
    code, output = _run(cmd)
    event["action"] = "launch_full"
    event["returncode"] = code
    event["state"] = "full_complete" if code == 0 else "full_waiting_or_failed"
    event["output_tail"] = output[-4000:]
    _append_log(log_path, f"[{now}] launch_full rc={code}\n{output}")
    status.update(event)
    _write_json(status_path, status)
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
