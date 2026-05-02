#!/usr/bin/env python3
"""Small GCS directory sync helper for short-lived GPU workers.

This intentionally uses Application Default Credentials directly through the
Python client rather than shelling out to gsutil. Some RunPod images install a
PyPI gsutil build that ignores copied ADC credentials and falls back to
anonymous requests.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
from pathlib import Path
from typing import Iterable

import google.auth
from google.cloud import storage


def parse_gs_url(url: str) -> tuple[str, str]:
    if not url.startswith("gs://"):
        raise ValueError(f"expected gs:// URL, got {url!r}")
    rest = url.removeprefix("gs://")
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def make_client(credentials_file: str | None) -> storage.Client:
    if credentials_file:
        creds, project = google.auth.load_credentials_from_file(credentials_file)
    else:
        creds, project = google.auth.default()
    return storage.Client(credentials=creds, project=project)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            yield path


def download_prefix(
    client: storage.Client,
    gs_url: str,
    dest: Path,
    *,
    workers: int,
    skip_existing: bool,
) -> None:
    bucket_name, prefix = parse_gs_url(gs_url)
    bucket = client.bucket(bucket_name)
    dest.mkdir(parents=True, exist_ok=True)
    blobs = list(client.list_blobs(bucket_name, prefix=prefix + "/" if prefix else None))
    blobs = [blob for blob in blobs if not blob.name.endswith("/")]
    print(f"[gcs-sync] download {len(blobs)} objects from {gs_url} -> {dest}", flush=True)

    def one(blob: storage.Blob) -> tuple[str, int, str]:
        rel = blob.name[len(prefix) :].lstrip("/") if prefix else blob.name
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and out.exists() and out.stat().st_size == blob.size:
            return rel, blob.size or 0, "skip"
        tmp = out.with_suffix(out.suffix + ".tmp")
        blob.download_to_filename(tmp)
        os.replace(tmp, out)
        return rel, blob.size or 0, "download"

    done = 0
    bytes_done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for rel, size, action in ex.map(one, blobs):
            done += 1
            bytes_done += size
            if done == 1 or done % 100 == 0 or done == len(blobs):
                print(
                    f"[gcs-sync] {action} {done}/{len(blobs)} "
                    f"bytes={bytes_done / (1024 ** 3):.2f}GiB last={rel}",
                    flush=True,
                )


def upload_prefix(
    client: storage.Client,
    src: Path,
    gs_url: str,
    *,
    workers: int,
    skip_existing: bool,
) -> None:
    bucket_name, prefix = parse_gs_url(gs_url)
    bucket = client.bucket(bucket_name)
    files = list(iter_files(src))
    print(f"[gcs-sync] upload {len(files)} files from {src} -> {gs_url}", flush=True)

    def one(path: Path) -> tuple[str, int, str]:
        rel = path.relative_to(src).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        blob = bucket.blob(blob_name)
        size = path.stat().st_size
        if skip_existing and blob.exists() and blob.size == size:
            return rel, size, "skip"
        blob.upload_from_filename(path)
        return rel, size, "upload"

    done = 0
    bytes_done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for rel, size, action in ex.map(one, files):
            done += 1
            bytes_done += size
            if done == 1 or done % 100 == 0 or done == len(files):
                print(
                    f"[gcs-sync] {action} {done}/{len(files)} "
                    f"bytes={bytes_done / (1024 ** 3):.2f}GiB last={rel}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["download", "upload"])
    parser.add_argument("src")
    parser.add_argument("dest")
    parser.add_argument("--credentials-file", default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    client = make_client(args.credentials_file)
    skip_existing = not args.no_skip_existing
    if args.mode == "download":
        download_prefix(client, args.src, Path(args.dest), workers=args.workers, skip_existing=skip_existing)
    else:
        upload_prefix(client, Path(args.src), args.dest, workers=args.workers, skip_existing=skip_existing)


if __name__ == "__main__":
    main()
