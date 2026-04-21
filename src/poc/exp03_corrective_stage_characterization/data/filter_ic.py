"""
Filter IC (In-Context) items in the dataset by running the PT model and
keeping only questions it answers correctly.

This step is optional but recommended: it ensures every IC question in your
analysis was actually answerable by the base model (i.e., the knowledge was
present in its weights), making the PT/IT comparison cleaner.

Usage
-----
  python -m src.poc.exp03_corrective_stage_characterization.data.filter_ic \\
      --input  data/exp3_dataset.jsonl \\
      --output data/exp3_dataset_filtered.jsonl \\
      --model  google/gemma-3-4b-pt \\
      --gpu    0 \\
      --target-ic  250    # keep up to this many verified IC items (in dataset order)

The script generates at most max_new_tokens=64 tokens per IC prompt (Format B),
then passes the generation to a local string-match judge.  If the model's
output contains any of the canonical answers (case-insensitive, after light
normalisation), the item is marked verified_pt=True; otherwise False.

Non-IC items are passed through unchanged.

String-match judge
------------------
We use a lenient judge:
  1. Normalise both prediction and answer (lowercase, strip punctuation, collapse
     whitespace).
  2. Check if *any* of (answer, answer_aliases) appears as a substring of the
     normalised prediction.
For factual QA this catches paraphrased answers, units, abbreviated forms, etc.
The exp3 LLM-as-judge can later re-evaluate borderline cases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalisation / match
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalise(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _string_match(prediction: str, answer: str, aliases: list[str]) -> bool:
    pred_norm = _normalise(prediction)
    for candidate in [answer] + aliases:
        if _normalise(candidate) in pred_norm:
            return True
    return False


# ---------------------------------------------------------------------------
# Inference helper (loads model lazily)
# ---------------------------------------------------------------------------

def _run_model_on_ic(
    records: list[dict],
    model_name: str,
    gpu: int,
    max_new_tokens: int,
    batch_size: int,
) -> list[dict]:
    """Run PT model on IC records, fill in verified_pt field."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = f"cuda:{gpu}"
    print(f"Loading {model_name} on {device} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    n = len(records)
    for batch_start in range(0, n, batch_size):
        batch = records[batch_start : batch_start + batch_size]
        prompts = [r["formats"]["B"] for r in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated portion (strip prompt tokens)
        for rec, inp_ids, gen_ids in zip(
            batch, inputs["input_ids"], out
        ):
            new_tokens = gen_ids[inp_ids.shape[0]:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
            rec["verified_pt"] = _string_match(
                pred, rec["answer"], rec.get("answer_aliases", [])
            )
            rec["metadata"]["pt_prediction"] = pred

        done = min(batch_start + batch_size, n)
        print(f"  [{done}/{n}] verified so far …", flush=True)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter IC items using PT model string-match verification."
    )
    parser.add_argument("--input",  required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--model", default="google/gemma-3-4b-pt",
        help="HuggingFace model ID for PT model (default: google/gemma-3-4b-pt)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument(
        "--target-ic", type=int, default=250,
        help="Number of verified IC items to keep (default: 250)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max generation tokens per IC prompt (default: 64)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Inference batch size (default: 16)",
    )
    args = parser.parse_args()

    # Load dataset
    records: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    ic_records    = [r for r in records if r["split"] == "IC"]
    non_ic        = [r for r in records if r["split"] != "IC"]

    print(f"Total records  : {len(records)}")
    print(f"IC records     : {len(ic_records)}")
    print(f"Non-IC records : {len(non_ic)}")

    # Run model on IC
    ic_verified = _run_model_on_ic(
        ic_records,
        model_name=args.model,
        gpu=args.gpu,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    passed = [r for r in ic_verified if r["verified_pt"] is True]
    failed = [r for r in ic_verified if r["verified_pt"] is False]
    print(f"\nIC verification: {len(passed)} passed / {len(failed)} failed "
          f"({100*len(passed)/len(ic_records):.1f}% pass rate)")

    # Keep the first target_ic verified items in their original dataset order.
    # (No confidence score is computed; filtering is binary pass/fail.)
    kept_ic = passed[: args.target_ic]
    if len(kept_ic) < args.target_ic:
        print(f"[WARN] Only {len(kept_ic)} IC items verified correct "
              f"(target was {args.target_ic}). Consider increasing --ic-oversample "
              f"in build_dataset.py.", file=sys.stderr)

    all_out = non_ic + kept_ic

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_out)} records → {output_path}")
    split_counts: dict[str, int] = {}
    for r in all_out:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    for k, v in sorted(split_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
