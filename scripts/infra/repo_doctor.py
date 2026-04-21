#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOTS = [ROOT / "src", ROOT / "scripts"]
HELP_COMMANDS = [
    ["bash", "scripts/run/run_exp13_exp14_local.sh", "--help"],
    [sys.executable, "-m", "src.poc.exp14_symmetric_matched_prefix_causality", "--help"],
    [sys.executable, "-m", "src.poc.exp15_symmetric_behavioral_causality", "--help"],
    [sys.executable, "scripts/analysis/analyze_exp13a_lite.py", "--help"],
]
PYTEST_TARGETS = [
    "src/poc/exp06_corrective_direction_steering/tests",
]


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in CODE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return sorted(files)


def _shell_files() -> list[Path]:
    scripts_root = ROOT / "scripts"
    return sorted(p for p in scripts_root.rglob("*.sh") if p.is_file())


def _symlink_targets() -> list[Path]:
    scripts_root = ROOT / "scripts"
    return sorted(p for p in scripts_root.rglob("*") if p.is_symlink())


def _run(cmd: list[str]) -> None:
    print(f"[doctor] exec: {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def check_symlinks() -> tuple[int, str]:
    symlinks = _symlink_targets()
    broken = [path for path in symlinks if not path.exists()]
    if broken:
        lines = "\n".join(str(path.relative_to(ROOT)) for path in broken[:50])
        raise RuntimeError(f"broken symlinks detected:\n{lines}")
    return len(symlinks), "ok"


def check_py_compile() -> tuple[int, str]:
    files = _python_files()
    for path in files:
        py_compile.compile(str(path), doraise=True)
    return len(files), "ok"


def check_shell_parse() -> tuple[int, str]:
    files = _shell_files()
    for path in files:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"bash -n failed for {path.relative_to(ROOT)}:\n{result.stdout}")
    return len(files), "ok"


def check_help_smoke() -> tuple[int, str]:
    for cmd in HELP_COMMANDS:
        _run(cmd)
    return len(HELP_COMMANDS), "ok"


def check_pytest() -> tuple[int, str]:
    _run([sys.executable, "-m", "pytest", *PYTEST_TARGETS, "-q"])
    return len(PYTEST_TARGETS), "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight repo health checks for canonical entrypoints.")
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="Also run the small checked-in pytest target set.",
    )
    parser.add_argument(
        "--skip-help-smoke",
        action="store_true",
        help="Skip entrypoint --help smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks: list[tuple[str, callable]] = [
        ("symlinks", check_symlinks),
        ("py_compile", check_py_compile),
        ("shell_parse", check_shell_parse),
    ]
    if not args.skip_help_smoke:
        checks.append(("help_smoke", check_help_smoke))
    if args.pytest:
        checks.append(("pytest", check_pytest))

    print(f"[doctor] root: {ROOT}")
    for name, fn in checks:
        count, status = fn()
        print(f"[doctor] {name}: {status} ({count})")
    print("[doctor] all checks passed")


if __name__ == "__main__":
    main()
