"""
Segment content boundary in generated tokens using LLM-as-judge (Option 2).

Post-hoc analysis — runs AFTER inference, on saved results JSONL.

For each generation, reconstructs the full text and asks an LLM to identify
the token index where discourse preamble ends and actual answer content begins.
The boundary is saved as 'content_start_idx' in each result record.

This replaces the string-lookup CONTENT classification in token_types.py with
a more accurate, context-aware boundary. token_types.classify_generated_tokens()
respects 'content_start_idx' if present (see apply_segmentation()).

Usage
-----
  python -m src.poc.exp3.analysis.segment_content \\
      --input  results/exp3_it_results.jsonl \\
      --output results/exp3_it_results_segmented.jsonl \\
      --model  meta-llama/llama-3.3-70b-instruct   # any OpenRouter model ID

The script is resumable: records that already have 'content_start_idx' are
skipped unless --force is passed.

Environment
-----------
  OPENROUTER_API_KEY   Required. OpenRouter API key.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a linguistics annotation assistant. Your task is to identify where a \
language model's discourse preamble ends and the actual answer content begins.

Discourse preamble includes:
- Filler / acknowledgement phrases: "Well,", "Sure!", "Of course,", \
"Certainly!", "Great question!", "I'd be happy to", "Let me", "Absolutely!"
- Discourse markers: "First,", "So,", "Now,", "To answer your question,"
- Formatting tokens at the very start: "**", "##", newlines, "- "
- Any token that does not directly carry answer content

Actual answer content is the FIRST token that is semantically part of the \
answer — a noun, verb, number, name, entity, or content word that directly \
addresses the prompt. If the very first token is already content, reply with 0.

Reply with ONLY a single integer (the 0-based index). No explanation."""

USER_TEMPLATE = """\
Prompt given to the model:
{prompt}

Generated tokens (0-based index: token string):
{token_list}

Reply with ONLY the integer index of the first content token."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_token_list(generated_tokens: list[dict]) -> str:
    lines = []
    for i, t in enumerate(generated_tokens):
        lines.append(f"  {i}: {repr(t['token_str'])}")
        if i >= 39:  # cap at 40 tokens — preamble is never this long
            lines.append(f"  ... ({len(generated_tokens) - 40} more tokens)")
            break
    return "\n".join(lines)


def _call_llm(
    client,
    model: str,
    prompt: str,
    generated_tokens: list[dict],
    max_retries: int = 3,
) -> tuple[int, str]:
    """Call OpenRouter LLM, return (content_start_idx, raw_response)."""
    token_list = _format_token_list(generated_tokens)
    user_msg = USER_TEMPLATE.format(
        prompt=prompt[:400],
        token_list=token_list,
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=8,
                temperature=0,
                extra_headers={
                    "HTTP-Referer": "https://github.com/yifan/structural-semantic-features",
                    "X-Title": "exp3-segment-content",
                },
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\d+", raw)
            if m:
                idx = int(m.group())
                idx = max(0, min(idx, len(generated_tokens) - 1))
                return idx, raw
            # No integer found — default to 0
            return 0, f"PARSE_FAIL: {raw}"

        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    [WARN] attempt {attempt+1} failed ({exc}), retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"    [ERROR] all retries exhausted: {exc}")
                return 0, f"ERROR: {exc}"

    return 0, "ERROR: unreachable"


# ---------------------------------------------------------------------------
# Public helper consumed by token_types.py / plots
# ---------------------------------------------------------------------------

def apply_segmentation(result: dict) -> list[str]:
    """Return token type list honouring 'content_start_idx' if present.

    Delegates to token_types.classify_generated_tokens() for everything except
    the CONTENT label, which is forced to whichever index the LLM identified.

    If 'content_start_idx' is absent the function falls back to the standard
    string-lookup behaviour.
    """
    from src.poc.exp3.analysis.token_types import classify_generated_tokens

    types = classify_generated_tokens(result.get("generated_tokens", []))

    if "content_start_idx" not in result:
        return types

    llm_idx = result["content_start_idx"]
    tokens  = result.get("generated_tokens", [])

    # Clear any CONTENT label the string-lookup assigned, then set the LLM one.
    for i in range(len(types)):
        if types[i] == "CONTENT":
            types[i] = "OTHER"
    if 0 <= llm_idx < len(tokens):
        types[llm_idx] = "CONTENT"

    return types


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge content boundary segmentation (post-hoc)."
    )
    parser.add_argument("--input",  required=True, help="Input results JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL with content_start_idx")
    parser.add_argument(
        "--model",
        default="meta-llama/llama-3.3-70b-instruct",
        help="OpenRouter model ID (default: meta-llama/llama-3.3-70b-instruct)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even for records that already have content_start_idx",
    )
    parser.add_argument(
        "--delay", type=float, default=0.2,
        help="Seconds to sleep between API calls (default: 0.2)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set.")

    # Import openai here so the module is usable without it installed.
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package not installed. Run: pip install openai")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Load records
    records: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {args.input}")

    already_done = sum(1 for r in records if "content_start_idx" in r)
    if already_done and not args.force:
        print(f"  {already_done} already have content_start_idx — skipping (use --force to redo)")

    to_process = [
        r for r in records
        if args.force or "content_start_idx" not in r
    ]
    print(f"  Will segment {len(to_process)} records with model={args.model}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(to_process)
    for i, rec in enumerate(to_process):
        prompt    = rec.get("prompt", "")
        gen_toks  = rec.get("generated_tokens", [])

        if not gen_toks:
            rec["content_start_idx"] = 0
            rec["segmentation_raw"]  = "EMPTY"
            continue

        idx, raw = _call_llm(client, args.model, prompt, gen_toks)
        rec["content_start_idx"] = idx
        rec["segmentation_raw"]  = raw

        # Progress
        tok_preview = "".join(t["token_str"] for t in gen_toks[:idx+3])
        print(
            f"  [{i+1}/{n}] prompt_id={rec.get('prompt_id','?')!r:20s}  "
            f"content_start={idx}  preview={repr(tok_preview[:40])}"
        )

        if args.delay > 0 and i < n - 1:
            time.sleep(args.delay)

    # Write output — all records (updated + untouched)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records → {output_path}")
    done_count = sum(1 for r in records if "content_start_idx" in r)
    print(f"  {done_count}/{len(records)} have content_start_idx")


if __name__ == "__main__":
    main()
