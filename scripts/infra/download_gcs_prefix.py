#!/usr/bin/env python
"""Download a GCS prefix using Application Default Credentials.

This is a small fallback for RunPod images that do not have a configured
``gsutil`` binary but do have ADC copied into the environment.
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse


def parse_gs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected gs://bucket/prefix URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--project", default=os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--include-regex", default="")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    try:
        from google.cloud import storage
    except Exception as exc:  # pragma: no cover - dependency bootstrap path
        raise SystemExit(
            "google-cloud-storage is required for download_gcs_prefix.py; "
            "install it in the active environment first"
        ) from exc

    bucket_name, prefix = parse_gs_uri(args.uri.rstrip("/") + "/")
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=args.project or "pt-it-model-differences")
    bucket = client.bucket(bucket_name)
    include = re.compile(args.include_regex) if args.include_regex else None
    jobs = []
    n_skipped = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix) :]
        if not rel:
            continue
        if include is not None and include.search(rel) is None:
            continue
        out_path = dest / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        size = int(blob.size or 0)
        if out_path.exists() and size > 0 and out_path.stat().st_size == size:
            n_skipped += 1
            continue
        jobs.append((blob.name, out_path, size))

    def fetch(job: tuple[str, Path, int]) -> int:
        blob_name, out_path, size = job
        print(f"[gcs-download] {blob_name} -> {out_path}", flush=True)
        bucket.blob(blob_name).download_to_filename(out_path)
        return size

    n_files = 0
    n_bytes = 0
    workers = max(1, int(args.max_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            n_bytes += int(future.result())
            n_files += 1
    print(
        f"[gcs-download] complete uri={args.uri} files={n_files} skipped={n_skipped} bytes={n_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
