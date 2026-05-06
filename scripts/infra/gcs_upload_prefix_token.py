#!/usr/bin/env python
"""Upload a local directory to a GCS prefix using an access token from GCS_TOKEN."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests


def _upload_one(bucket: str, prefix: str, root: Path, path: Path) -> tuple[str, int]:
    rel = path.relative_to(root).as_posix()
    name = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
    url = (
        f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
        f"?uploadType=media&name={quote(name, safe='')}"
    )
    headers = {
        "Authorization": f"Bearer {os.environ['GCS_TOKEN']}",
        "Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }
    size = path.stat().st_size
    for attempt in range(6):
        with path.open("rb") as handle:
            response = requests.post(url, headers=headers, data=handle, timeout=600)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return rel, size
        time.sleep(2**attempt)
    response.raise_for_status()
    return rel, size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bucket")
    parser.add_argument("prefix")
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    files = [p for p in args.root.rglob("*") if p.is_file() and not p.is_symlink()]
    total = sum(path.stat().st_size for path in files)
    print(
        f"[gcs-upload] {args.root} -> gs://{args.bucket}/{args.prefix.rstrip('/')}: "
        f"{len(files)} files, {total / 1024**3:.2f} GiB",
        flush=True,
    )
    done = 0
    bytes_done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [
            pool.submit(_upload_one, args.bucket, args.prefix, args.root, path)
            for path in files
        ]
        for future in futures.as_completed(pending):
            rel, size = future.result()
            done += 1
            bytes_done += size
            if done % 20 == 0 or done == len(files):
                print(
                    f"[gcs-upload] {done}/{len(files)} files, "
                    f"{bytes_done / 1024**3:.2f} GiB ({rel})",
                    flush=True,
                )
    print("[gcs-upload] complete", flush=True)


if __name__ == "__main__":
    main()
