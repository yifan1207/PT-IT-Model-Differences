#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.poc.exp45_behavioral_bridge.judge_prompts import parse_jsonish
from src.poc.shared.llm_provider import build_openai_client


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("request_id")) for row in _iter_jsonl(path)}


def _call(client: Any, model: str, prompt: str, retries: int = 4) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=180,
                messages=[{"role": "user", "content": prompt}],
            )
            return parse_jsonish(resp.choices[0].message.content or "")
        except Exception as exc:
            if attempt == retries - 1:
                return {"error": str(exc)}
            time.sleep(2**attempt + random.random())
    return {"error": "unreachable"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge-model", default="google/gemini-2.5-flash")
    parser.add_argument("--provider", default="auto", choices=["auto", "gemini", "openrouter", "openai"])
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    client_info = build_openai_client(args.judge_model, provider=args.provider)
    if client_info is None:
        raise SystemExit("No GEMINI_API_KEY or OPENROUTER_API_KEY available; judge requests were generated but not scored.")
    client, model, provider = client_info
    rows = list(_iter_jsonl(args.requests))
    if args.limit:
        rows = rows[: int(args.limit)]
    done = _done_ids(args.out)
    rows = [row for row in rows if str(row.get("request_id")) not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[exp45-judge] provider={provider} model={model} remaining={len(rows)}", flush=True)

    def score(row: dict[str, Any]) -> dict[str, Any]:
        result = _call(client, model, row["prompt"])
        return {k: v for k, v in row.items() if k != "prompt"} | {"judge_provider": provider, "judge_model": model, "result": result}

    with args.out.open("a", encoding="utf-8") as handle:
        done_n = 0
        with ThreadPoolExecutor(max_workers=max(1, int(args.parallelism))) as pool:
            futures = {pool.submit(score, row): row for row in rows}
            for fut in as_completed(futures):
                handle.write(json.dumps(fut.result(), separators=(",", ":")) + "\n")
                handle.flush()
                done_n += 1
                if done_n == 1 or done_n % 100 == 0 or done_n == len(rows):
                    print(f"[exp45-judge] {done_n}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
