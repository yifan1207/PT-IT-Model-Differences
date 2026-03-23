#!/usr/bin/env python3
"""Upload large exp5 raw data files to GCP and delete local copies.

Uploads sample_outputs.jsonl and hidden-state npz files to GCS, then removes
the local files. Plots, scores, and merged results are kept locally.

Usage:
    python scripts/gcp_upload_and_cleanup.py \
        --bucket gs://your-bucket/exp5 \
        --run-dirs results/exp5/phase_it_none_t200 results/exp5/progressive_it_w* \
        --dry-run   # preview without deleting

    python scripts/gcp_upload_and_cleanup.py \
        --bucket gs://your-bucket/exp5 \
        --run-dirs results/exp5/phase_it_none_t200
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Files that are large raw data — safe to remove after upload.
# Scores, plots, and merged outputs are intentionally excluded.
_UPLOAD_PATTERNS = [
    "sample_outputs.jsonl",          # generated text per record (~20-100 MB each)
    "checkpoints/*.npz",             # hidden-state arrays per condition (~50-500 MB each)
]


def _gcs_upload(local: Path, bucket_prefix: str, dry_run: bool) -> bool:
    dest = f"{bucket_prefix.rstrip('/')}/{local}"
    if dry_run:
        print(f"  [dry-run] would upload {local} → {dest}")
        return True
    print(f"  uploading {local} → {dest} ...", flush=True)
    result = subprocess.run(
        ["gcloud", "storage", "cp", str(local), dest],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR uploading {local}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True,
                   help="GCS destination prefix, e.g. gs://my-bucket/exp5")
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="Run directories to upload from, e.g. results/exp5/phase_it_none_t200")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be uploaded/deleted without doing it")
    args = p.parse_args()

    uploaded: list[Path] = []
    failed: list[Path] = []

    for run_dir_str in args.run_dirs:
        # Support globs in run-dirs
        import glob
        dirs = [Path(d) for d in glob.glob(run_dir_str)] or [Path(run_dir_str)]
        for run_dir in dirs:
            if not run_dir.exists():
                print(f"[gcp] skipping {run_dir} (does not exist)")
                continue
            print(f"\n[gcp] processing {run_dir}")
            for pattern in _UPLOAD_PATTERNS:
                for local in sorted(run_dir.glob(pattern)):
                    if not local.is_file():
                        continue
                    size_mb = local.stat().st_size / 1e6
                    print(f"  found {local} ({size_mb:.1f} MB)")
                    ok = _gcs_upload(local, args.bucket, args.dry_run)
                    if ok:
                        uploaded.append(local)
                    else:
                        failed.append(local)

    print(f"\n[gcp] uploaded: {len(uploaded)} files, failed: {len(failed)}")
    if failed:
        print("[gcp] FAILED files — NOT deleting anything:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)

    if not uploaded:
        print("[gcp] nothing to upload")
        return

    # Delete local copies only after all uploads succeeded
    print("\n[gcp] deleting local copies ...")
    for local in uploaded:
        if args.dry_run:
            print(f"  [dry-run] would delete {local}")
        else:
            local.unlink()
            print(f"  deleted {local}")

    print("[gcp] done")


if __name__ == "__main__":
    main()
