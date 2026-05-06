#!/usr/bin/env python
"""Fetch a GCS prefix using an access token from GCS_TOKEN.

This is intentionally small and dependency-light for ephemeral RunPod setup.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests


def _request(url: str, *, params: dict[str, str] | None = None, stream: bool = False) -> requests.Response:
    token = os.environ["GCS_TOKEN"]
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(6):
        response = requests.get(url, headers=headers, params=params, stream=stream, timeout=120)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        time.sleep(2**attempt)
    response.raise_for_status()
    return response


def _list_objects(bucket: str, prefix: str) -> list[dict[str, str]]:
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    page_token: str | None = None
    objects: list[dict[str, str]] = []
    while True:
        params = {"prefix": prefix, "fields": "items(name,size),nextPageToken"}
        if page_token:
            params["pageToken"] = page_token
        payload = _request(url, params=params).json()
        objects.extend(obj for obj in payload.get("items", []) if not obj["name"].endswith("/"))
        page_token = payload.get("nextPageToken")
        if not page_token:
            return objects


def _download_one(bucket: str, prefix: str, out_dir: Path, obj: dict[str, str]) -> tuple[str, int]:
    name = obj["name"]
    expected_size = int(obj.get("size", 0))
    rel = name[len(prefix) :]
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected_size:
        return rel, expected_size

    tmp = dest.with_name(dest.name + ".tmp")
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(name, safe='')}?alt=media"
    response = _request(url, stream=True)
    written = 0
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                written += len(chunk)
    if expected_size and written != expected_size:
        raise RuntimeError(f"size mismatch for {name}: got {written}, expected {expected_size}")
    tmp.replace(dest)
    return rel, written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bucket")
    parser.add_argument("prefix")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    prefix = args.prefix.rstrip("/") + "/"
    objects = _list_objects(args.bucket, prefix)
    total = sum(int(obj.get("size", 0)) for obj in objects)
    print(f"[gcs-fetch] {prefix}: {len(objects)} objects, {total / 1024**3:.2f} GiB", flush=True)

    done = 0
    bytes_done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(_download_one, args.bucket, prefix, args.out_dir, obj) for obj in objects]
        for future in futures.as_completed(pending):
            rel, size = future.result()
            done += 1
            bytes_done += size
            if done % 5 == 0 or done == len(objects):
                print(
                    f"[gcs-fetch] {prefix}: {done}/{len(objects)} files, "
                    f"{bytes_done / 1024**3:.2f} GiB ({rel})",
                    flush=True,
                )
    print(f"[gcs-fetch] {prefix}: complete", flush=True)


if __name__ == "__main__":
    main()
