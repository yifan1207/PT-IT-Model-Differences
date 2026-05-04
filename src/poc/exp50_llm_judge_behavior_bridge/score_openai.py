from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from src.poc.shared.llm_provider import load_dotenv_near_repo


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _done_ids(path: Path, *, retry_errors: bool = False) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in _iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        result = row.get("result") or {}
        if retry_errors and isinstance(result, dict) and "error" in result:
            continue
        if request_id:
            done.add(request_id)
    return done


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return None


def _metadata_for_output(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"messages", "response_format"}}


def _request_body(row: dict[str, Any], *, model: str, max_output_tokens: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": row["messages"],
        "response_format": row["response_format"],
        "temperature": 0,
        "max_completion_tokens": max_output_tokens,
    }
    reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "minimal").strip()
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return body


def _chat_completion_with_fallback(
    client: OpenAI,
    *,
    body: dict[str, Any],
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**body)
            choice = response.choices[0]
            message = choice.message
            content = message.content
            if not content:
                return (
                    {
                        "error": "empty_content",
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "refusal": getattr(message, "refusal", None),
                    },
                    _usage_dict(response.usage),
                )
            return json.loads(content), _usage_dict(response.usage)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "max_completion_tokens" in message and "max_tokens" not in body:
                body = dict(body)
                body["max_tokens"] = body.pop("max_completion_tokens", 256)
            elif "reasoning_effort" in message and "reasoning_effort" in body:
                body = dict(body)
                body.pop("reasoning_effort", None)
            elif "temperature" in message.lower() and "temperature" in body:
                body = dict(body)
                body.pop("temperature", None)
            elif attempt == retries - 1:
                break
            time.sleep(min(30.0, 2**attempt + random.random()))
    return {"error": str(last_error) if last_error else "unknown_error"}, None


def score_sync(
    *,
    requests_path: Path,
    out_path: Path,
    model: str,
    limit: int,
    parallelism: int,
    retry_errors: bool,
    max_output_tokens: int,
    retries: int,
) -> None:
    load_dotenv_near_repo()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it in the environment; do not pass it on the CLI.")
    client = OpenAI()
    rows = list(_iter_jsonl(requests_path))
    if limit:
        rows = rows[:limit]
    done = _done_ids(out_path, retry_errors=retry_errors)
    rows = [row for row in rows if str(row.get("request_id") or "") not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"mode": "sync", "model": model, "remaining": len(rows), "out": str(out_path)}, sort_keys=True))

    def score_one(row: dict[str, Any]) -> dict[str, Any]:
        body = _request_body(row, model=model, max_output_tokens=max_output_tokens)
        result, usage = _chat_completion_with_fallback(client, body=body, retries=retries)
        out = _metadata_for_output(row)
        out["judge_provider"] = "openai"
        out["judge_model"] = model
        out["result"] = result
        if usage is not None:
            out["usage"] = usage
        return out

    with _open_text(out_path, "a") as handle:
        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, int(parallelism))) as pool:
            futures = {pool.submit(score_one, row): row for row in rows}
            for future in as_completed(futures):
                handle.write(json.dumps(future.result(), ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                completed += 1
                if completed == 1 or completed % 100 == 0 or completed == len(rows):
                    print(json.dumps({"completed": completed, "remaining": len(rows) - completed}, sort_keys=True))


def prepare_batch_input(
    *,
    requests_path: Path,
    out_path: Path,
    model: str,
    max_output_tokens: int,
    limit: int,
) -> None:
    rows = list(_iter_jsonl(requests_path))
    if limit:
        rows = rows[:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            body = _request_body(row, model=model, max_output_tokens=max_output_tokens)
            # Batch API accepts the same body shape as the target non-streaming endpoint.
            payload = {
                "custom_id": str(row["request_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "requests": str(requests_path),
        "batch_input": str(out_path),
        "model": model,
        "n_requests": len(rows),
        "max_output_tokens": max_output_tokens,
    }
    (out_path.parent / "batch_prepare_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, **metadata}, indent=2, sort_keys=True))


def submit_batch(*, batch_input: Path, out_metadata: Path) -> None:
    load_dotenv_near_repo()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it in the environment; do not pass it on the CLI.")
    client = OpenAI()
    with batch_input.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"experiment": "exp50_llm_judge_behavior_bridge"},
    )
    payload = {
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }
    out_metadata.parent.mkdir(parents=True, exist_ok=True)
    out_metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _file_content_text(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "read"):
        data = content.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)
    if hasattr(content, "text"):
        return str(content.text)
    return str(content)


def retrieve_batch(
    *,
    requests_path: Path,
    batch_id: str,
    out_path: Path,
    metadata_out: Path,
) -> None:
    load_dotenv_near_repo()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it in the environment; do not pass it on the CLI.")
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    payload = batch.model_dump() if hasattr(batch, "model_dump") else dict(batch)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    output_file_id = payload.get("output_file_id")
    if not output_file_id:
        print(json.dumps({"ok": False, "status": payload.get("status"), "reason": "no output_file_id"}, indent=2))
        return

    request_by_id = {str(row["request_id"]): row for row in _iter_jsonl(requests_path)}
    text = _file_content_text(client, str(output_file_id))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with _open_text(out_path, "w") as handle:
        for line in text.splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            request_id = str(raw.get("custom_id") or "")
            base = _metadata_for_output(request_by_id.get(request_id, {"request_id": request_id}))
            base["judge_provider"] = "openai"
            response = raw.get("response") or {}
            error = raw.get("error")
            body = response.get("body") if isinstance(response, dict) else None
            try:
                content = ((body or {}).get("choices") or [{}])[0].get("message", {}).get("content") or "{}"
                result = json.loads(content)
            except Exception as exc:
                result = {"error": f"batch_parse_error: {exc}"}
            if error:
                result = {"error": error}
            base["result"] = result
            if isinstance(body, dict) and body.get("usage") is not None:
                base["usage"] = body.get("usage")
            handle.write(json.dumps(base, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    print(json.dumps({"ok": True, "written": written, "out": str(out_path)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Exp50 judge requests with OpenAI Structured Outputs.")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("judge_responses.jsonl.gz"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.2"))
    parser.add_argument("--mode", choices=["sync", "batch-prepare", "batch-submit", "batch-retrieve"], default="sync")
    parser.add_argument("--parallelism", type=int, default=int(os.environ.get("OPENAI_PARALLELISM", "8")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=800)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--batch-input", type=Path, default=None)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--batch-metadata", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "sync":
        score_sync(
            requests_path=args.requests,
            out_path=args.out,
            model=args.model,
            limit=args.limit,
            parallelism=args.parallelism,
            retry_errors=args.retry_errors,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
        )
    elif args.mode == "batch-prepare":
        prepare_batch_input(
            requests_path=args.requests,
            out_path=args.batch_input or args.out.with_suffix(".batch.jsonl"),
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            limit=args.limit,
        )
    elif args.mode == "batch-submit":
        if args.batch_input is None:
            raise SystemExit("--batch-input is required for batch-submit")
        submit_batch(batch_input=args.batch_input, out_metadata=args.batch_metadata or args.batch_input.with_suffix(".batch_job.json"))
    elif args.mode == "batch-retrieve":
        if not args.batch_id:
            raise SystemExit("--batch-id is required for batch-retrieve")
        retrieve_batch(
            requests_path=args.requests,
            batch_id=args.batch_id,
            out_path=args.out,
            metadata_out=args.batch_metadata or args.out.with_suffix(".batch_status.json"),
        )


if __name__ == "__main__":
    main()
