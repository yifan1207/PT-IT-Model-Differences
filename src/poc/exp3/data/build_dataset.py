"""
Build the Exp3 evaluation dataset (~2000 prompts).

Output: data/exp3_dataset.jsonl  (one JSON object per line)

Categories
----------
  F    Factual Knowledge (~500)
         TriviaQA (Joshi et al. 2017)               ~200
         NQ-Open  (Kwiatkowski et al. 2019)          ~175
         WebQuestions (Berant et al. 2013)           ~125
         Control: model knows the answer from PT.  Corrective stage baseline.

  R    Reasoning (~500)
         GSM8K (Cobbe et al. 2021)                   ~150
         ARC-Challenge (Clark et al. 2018)           ~130
         StrategyQA (Geva et al. 2021)               ~100
         CommonsenseQA (Talmor et al. 2019)          ~120
         Tests whether corrective stage activates MORE for reasoning vs factual.
         Hypothesis: it should NOT if corrective stage is about alignment, not content.

  OOD  Out-of-Distribution (~500)
         Novel fictional entities                    ~200
         Post-training unknowable                    ~150
         Counterfactual / indeterminate              ~150
         Model has no confident answer — tests suppression/hedging behavior.

  A    Alignment-Sensitive (~500) — directly tests corrective stage function
         4a  Harmful requests (AdvBench + hand-written)          ~100
         4b  Borderline/ambiguous (XSTest + hand-written)        ~100
         4c  Format-constrained (IFEval + hand-written)          ~100
         4d  Conversational/persona (multi-turn + roleplay)      ~100
         4e  Raw completion (Wikipedia truncations + hand-written)~100

Two prompt formats per record:

  A: Raw completion — question rephrased as an incomplete statement (natural for PT)
  B: Q&A format    — "Question: …\\nAnswer:" (PRIMARY, same for PT and IT)
  [Category A 4e exception: both formats are the raw continuation text]

Tags
----
Each record has:
  split         : "F" | "R" | "OOD" | "A"
  question_type : "factual" | "yes_no" | "numerical" | "multi_choice" |
                  "code" | "completion" | "coreference" |
                  "fictional" | "unknowable" | "counterfactual" |
                  "harmful" | "borderline" | "format" | "continuation" |
                  "conversation" | "roleplay"
  domain        : subject area
  metadata.alignment_subcategory : "4a"–"4e" for A-split records
  metadata.expected_behavior     : "refuse" | "comply" | "comply_safely"
  metadata.ooc_type              : "novel_entity" | "post_training" | "counterfactual"

Usage
-----
  python -m src.poc.exp3.data.build_dataset [--output PATH] [--seed N]

  --output   path for output JSONL (default: data/exp3_dataset.jsonl)
  --seed     random seed (default: 42)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Prompt formats
# ---------------------------------------------------------------------------

def _fmt_a(question: str) -> str:
    """Format A: raw completion — rewrite as incomplete statement."""
    q = question.strip().rstrip("?").rstrip(".")
    rewrites = [
        (r"^What (?:is|are|was|were) (?:the )?(.+)$",   r"\1 is"),
        (r"^Who (?:is|was|were) (.+)$",                  r"\1 was"),
        (r"^How many (.+)$",                              r"The number of \1 is"),
        (r"^How (?:does|do|did) (.+)$",                   r"\1"),
        (r"^When (?:did|was|were|is) (.+)$",              r"\1 was"),
        (r"^Where (?:is|was|did|are) (.+)$",              r"\1 is located in"),
        (r"^Which (.+)$",                                 r"The \1 is"),
        (r"^Why (?:is|was|did|does|do) (.+)$",            r"\1 because"),
        (r"^In what (?:year|century|decade) (.+)$",       r"\1 in the year"),
        (r"^Does? (.+)$",                                 r"\1 is"),
        (r"^Did (.+)$",                                   r"\1"),
        (r"^Is (.+)$",                                    r"\1 is"),
        (r"^Can (.+)$",                                   r"\1 can"),
    ]
    for pattern, replacement in rewrites:
        if re.match(pattern, q, re.IGNORECASE):
            return re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    return f"{q}: "


def _fmt_b(question: str) -> str:
    """Format B: Q&A — primary format, identical for PT and IT."""
    return f"Question: {question}\nAnswer:"


def _apply_formats(question: str) -> dict[str, str]:
    return {"A": _fmt_a(question), "B": _fmt_b(question)}


def _make_id(split: str, source: str, idx: int) -> str:
    return f"{split.lower()}_{source}_{idx:04d}"


def _rec(split: str, source: str, question: str, answer: str,
         aliases: list[str], question_type: str, domain: str,
         ooc_type: str | None = None, metadata: dict | None = None) -> dict:
    return {
        "split": split,
        "source": source,
        "question": question,
        "answer": answer,
        "answer_aliases": aliases,
        "question_type": question_type,
        "domain": domain,
        "ooc_type": ooc_type,
        "verified_pt": None,
        "verified_it": None,
        "formats": _apply_formats(question),
        "metadata": metadata or {},
    }


def _rec_a(subcategory: str, source: str, question: str,
           question_type: str, domain: str,
           expected_behavior: str,
           raw_continuation: bool = False) -> dict:
    """Build an A-split record with alignment metadata."""
    r = _rec("A", source, question, answer="", aliases=[],
             question_type=question_type, domain=domain,
             metadata={"alignment_subcategory": subcategory,
                       "expected_behavior": expected_behavior})
    if raw_continuation:
        r["formats"] = {"A": question, "B": question}
    return r


# ---------------------------------------------------------------------------
# F loaders (Factual Knowledge)
# ---------------------------------------------------------------------------

def _load_trivia_qa(n: int, rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
    indices = rng.choice(len(ds), size=min(n * 2, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        q = item["question"].strip()
        a_obj = item["answer"]
        answer = a_obj["value"].strip()
        aliases = [v.strip() for v in a_obj.get("aliases", []) if v.strip()]
        results.append(_rec("F", "triviaqa", q, answer, aliases,
                            "factual", "trivia",
                            metadata={"trivia_qa_id": item.get("question_id", "")}))
        if len(results) == n:
            break
    return results


def _load_nq_open(n: int, rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("nq_open", split="train")
    indices = rng.choice(len(ds), size=min(n * 2, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        q = item["question"].strip()
        if not q.endswith("?"):
            q += "?"
        answers = item["answer"]
        if not answers:
            continue
        answer = answers[0].strip()
        aliases = [a.strip() for a in answers[1:] if a.strip()]
        results.append(_rec("F", "nq_open", q, answer, aliases, "factual", "trivia"))
        if len(results) == n:
            break
    return results


def _load_web_questions(n: int, rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("web_questions", split="train")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        q = item["question"].strip()
        answers = item["answers"]
        if not answers:
            continue
        answer = answers[0].strip()
        aliases = [a.strip() for a in answers[1:] if a.strip()]
        results.append(_rec("F", "web_questions", q, answer, aliases,
                            "factual", "general"))
    return results


# ---------------------------------------------------------------------------
# R loaders (Reasoning)
# ---------------------------------------------------------------------------

def _load_gsm8k(n: int, rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        q = item["question"].strip()
        raw = item["answer"]
        m = re.search(r"####\s*(.+)$", raw)
        numeric = m.group(1).strip() if m else raw.strip()
        results.append(_rec("R", "gsm8k", q, numeric, [], "numerical", "math",
                            metadata={"full_solution": raw}))
    return results


def _load_arc_challenge(n: int, rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        q = item["question"].strip()
        labels = item["choices"]["label"]
        texts  = item["choices"]["text"]
        key    = item["answerKey"].strip()
        choices_str = "  ".join(f"({l}) {t}" for l, t in zip(labels, texts))
        full_q = f"{q}\n{choices_str}"
        answer = key
        for l, t in zip(labels, texts):
            if l == key:
                answer = f"({key}) {t}"
                break
        results.append(_rec("R", "arc_challenge", full_q, answer, [key],
                            "multi_choice", "science",
                            metadata={"raw_question": q,
                                      "choices": dict(zip(labels, texts))}))
    return results


def _load_strategy_qa(n: int, rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    try:
        ds_train = load_dataset("ChilleD/StrategyQA", split="train")
        ds_test  = load_dataset("ChilleD/StrategyQA", split="test")
        all_items = list(ds_train) + list(ds_test)
    except Exception as e:
        print(f"[WARN] StrategyQA unavailable ({e}); skipping.", file=sys.stderr)
        return []
    indices = rng.choice(len(all_items), size=min(n, len(all_items)), replace=False)
    results = []
    for i in indices:
        item = all_items[int(i)]
        q = item["question"].strip()
        raw = item["answer"]
        answer = "Yes" if raw else "No"
        aliases = ["yes" if raw else "no", "true" if raw else "false"]
        results.append(_rec("R", "strategy_qa", q, answer, aliases,
                            "yes_no", "reasoning",
                            metadata={"facts": item.get("facts", [])}))
    return results


def _load_commonsense_qa(n: int, rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("commonsense_qa", split="train")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        q = item["question"].strip()
        labels = item["choices"]["label"]
        texts  = item["choices"]["text"]
        key    = item["answerKey"].strip()
        choices_str = "  ".join(f"({l}) {t}" for l, t in zip(labels, texts))
        full_q = f"{q}\n{choices_str}"
        answer = key
        for l, t in zip(labels, texts):
            if l == key:
                answer = f"({key}) {t}"
                break
        results.append(_rec("R", "commonsense_qa", full_q, answer, [key],
                            "multi_choice", "commonsense",
                            metadata={"raw_question": q,
                                      "concept": item.get("question_concept", "")}))
    return results


# ---------------------------------------------------------------------------
# OOD loader
# ---------------------------------------------------------------------------

def _load_ooc() -> list[dict]:
    from src.poc.exp3.data.ooc_prompts import all_ooc_prompts
    _TYPE_TO_QTYPE = {
        "novel_entity":   "fictional",
        "post_training":  "unknowable",
        "counterfactual": "counterfactual",
    }
    results = []
    for item in all_ooc_prompts():
        q = item["question"]
        ooc_type = item["ooc_type"]
        results.append(_rec("OOD", "custom", q, item["answer"], [],
                            _TYPE_TO_QTYPE[ooc_type], "ooc",
                            ooc_type=ooc_type))
    return results


# ---------------------------------------------------------------------------
# A loaders (Alignment-Sensitive)
# ---------------------------------------------------------------------------

def _load_advbench_csv() -> list[str]:
    """Fetch AdvBench harmful_behaviors.csv from the public llm-attacks GitHub repo.

    The original AdvBench (Zou et al. 2023) is available at:
    https://github.com/llm-attacks/llm-attacks — the CSV is public domain.
    Returns list of 'goal' strings (the harmful behavior requests).
    """
    import urllib.request, csv, io
    url = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
           "main/data/advbench/harmful_behaviors.csv")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        goals = [row["goal"].strip() for row in reader if row.get("goal", "").strip()]
        if goals:
            print(f"  Loaded {len(goals)} AdvBench goals from GitHub CSV.")
        return goals
    except Exception as e:
        print(f"  [WARN] AdvBench GitHub CSV unavailable ({e}).", file=sys.stderr)
        return []


def _load_a4a_harmful(n: int, rng: np.random.Generator) -> list[dict]:
    """4a: Direct harmful requests.
    Priority: (1) AdvBench GitHub CSV, (2) AdvBench on HuggingFace,
    (3) hand-written fallback list.
    """
    from src.poc.exp3.data.safety_prompts import A4A_HARMFUL

    # Try GitHub CSV first (public, no auth required)
    github_goals = _load_advbench_csv()
    if github_goals:
        records = [_rec_a("4a", "advbench", q, "harmful", "safety", "refuse")
                   for q in github_goals]
        # Sample down to n if more available
        if len(records) > n:
            idxs = rng.choice(len(records), size=n, replace=False)
            records = [records[int(i)] for i in idxs]
        return records

    # Try HuggingFace mirrors
    hf_records: list[dict] = []
    try:
        from datasets import load_dataset
        for hf_path, split_name, col in [
            ("walledai/AdvBench",    "train", "goal"),
            ("llm-attacks/advbench", "train", "goal"),
        ]:
            try:
                ds = load_dataset(hf_path, split=split_name)
                col_use = col if col in ds.column_names else ds.column_names[0]
                for item in ds:
                    hf_records.append(
                        _rec_a("4a", "advbench", str(item[col_use]).strip(),
                               "harmful", "safety", "refuse"))
                break
            except Exception:
                continue
    except Exception:
        pass

    if hf_records:
        if len(hf_records) > n:
            idxs = rng.choice(len(hf_records), size=n, replace=False)
            hf_records = [hf_records[int(i)] for i in idxs]
        return hf_records

    # Fall back to hand-written list
    print("  [INFO] Using hand-written A4A prompts (AdvBench not accessible).",
          file=sys.stderr)
    records = [_rec_a("4a", "malicious_instruct", q, "harmful", "safety", "refuse")
               for q in A4A_HARMFUL]
    if len(records) > n:
        idxs = rng.choice(len(records), size=n, replace=False)
        records = [records[int(i)] for i in idxs]
    return records


def _load_a4b_borderline(n: int, rng: np.random.Generator) -> list[dict]:
    """4b: Borderline/ambiguous — XSTest (HF) + hand-written.

    XSTest (Röttger et al. 2023, Paul/xstest) has 250 safe prompts
    (label=="safe") that look harmful but should be answered, spanning:
    safe_targets, safe_contexts, homonyms, figurative_language,
    historical_events, definitions, privacy_public, etc.
    We use the 'label' field (not 'type') to select all 250.
    """
    from src.poc.exp3.data.safety_prompts import A4B_BORDERLINE
    hf_records: list[dict] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("Paul/xstest", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        # Use label field: "safe" = prompts that SHOULD be answered
        safe_items = [x for x in ds if str(x.get("label", "")).lower() == "safe"]
        if not safe_items:
            safe_items = list(ds)   # fallback: use all
        for item in safe_items:
            hf_records.append(
                _rec_a("4b", "xstest", str(item[col]).strip(),
                       "borderline", "safety", "comply",
                       ))
        print(f"  Loaded {len(hf_records)} XSTest safe prompts.")
    except Exception as e:
        print(f"[WARN] XSTest unavailable ({e}); using hand-written 4b only.",
              file=sys.stderr)

    hand_written = [_rec_a("4b", "custom", q, "borderline", "safety", "comply")
                    for q in A4B_BORDERLINE]
    combined = hf_records + hand_written
    if len(combined) > n:
        idxs = rng.choice(len(combined), size=n, replace=False)
        combined = [combined[int(i)] for i in idxs]
    return combined


def _load_a4c_format(n: int, rng: np.random.Generator) -> list[dict]:
    """4c: Format-constrained — IFEval (HF) + hand-written."""
    from src.poc.exp3.data.safety_prompts import A4C_FORMAT
    hf_records: list[dict] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("google/IFEval", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        for item in ds:
            q = str(item[col]).strip()
            r = _rec_a("4c", "ifeval", q, "format", "instruction_following", "comply")
            r["formats"]["B"] = q   # IFEval prompt already contains constraints
            hf_records.append(r)
    except Exception as e:
        print(f"[WARN] IFEval unavailable ({e}); using hand-written 4c only.",
              file=sys.stderr)

    hand_written = [_rec_a("4c", "custom", q, "format", "instruction_following", "comply")
                    for q in A4C_FORMAT]
    combined = hf_records + hand_written
    if len(combined) > n:
        idxs = rng.choice(len(combined), size=n, replace=False)
        combined = [combined[int(i)] for i in idxs]
    return combined


def _load_a4d_conversational() -> list[dict]:
    """4d: Conversational/persona — multi-turn + roleplay, all hand-written."""
    from src.poc.exp3.data.safety_prompts import A4D_MULTI_TURN, A4D_ROLEPLAY
    results = []
    for prompt, expected in A4D_MULTI_TURN:
        results.append(_rec_a("4d", "custom", prompt, "conversation",
                              "general", expected))
    for prompt, expected in A4D_ROLEPLAY:
        results.append(_rec_a("4d", "custom", prompt, "roleplay",
                              "general", expected))
    return results


def _load_a4e_raw_completion(n: int, rng: np.random.Generator) -> list[dict]:
    """4e: Raw completion — Wikipedia truncations + hand-written.
    Format B = raw text (no Q&A wrapper). Tests corrective stage with no alignment target.
    """
    from src.poc.exp3.data.safety_prompts import A4E_EXTRA_CONTINUATIONS
    candidates: list[str] = []

    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True)
        for item in ds:
            text = item.get("text", "").strip()
            sentences = text.split(". ")
            if sentences and len(sentences[0]) >= 60:
                first = sentences[0].strip()
                cut = max(20, int(len(first) * 0.65))
                candidates.append(first[:cut])
            if len(candidates) >= (n - len(A4E_EXTRA_CONTINUATIONS)) * 4:
                break
    except Exception as e:
        print(f"[WARN] Wikipedia unavailable ({e}); using hand-written 4e only.",
              file=sys.stderr)

    candidates.extend(A4E_EXTRA_CONTINUATIONS)

    if len(candidates) > n:
        idxs = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idxs]

    return [_rec_a("4e", "custom" if t in A4E_EXTRA_CONTINUATIONS else "wikipedia",
                   t, "continuation", "general", "comply", raw_continuation=True)
            for t in candidates]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

TARGETS = {
    # F: Factual Knowledge (~500)
    "F_triviaqa":      200,
    "F_nq_open":       175,
    "F_webquestions":  125,
    # R: Reasoning (~500)
    "R_gsm8k":         150,
    "R_arc":           130,
    "R_strategy":      100,
    "R_commonsenseqa": 120,
    # OOD: Out-of-Distribution — from ooc_prompts.py (all loaded)
    # A: Alignment-Sensitive, 100 per subcategory
    "A_4a":            100,
    "A_4b":            100,
    "A_4c":            100,
    # A_4d: all hand-written (~100 from multi-turn + roleplay combined)
    # A_4e: 100 total (Wikipedia + hand-written)
    "A_4e":            100,
}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_dataset(output_path: str | Path, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading F — TriviaQA …")
    f_tqa  = _load_trivia_qa(TARGETS["F_triviaqa"], rng)

    print("Loading F — NQ-Open …")
    f_nq   = _load_nq_open(TARGETS["F_nq_open"], rng)

    print("Loading F — WebQuestions …")
    f_wq   = _load_web_questions(TARGETS["F_webquestions"], rng)

    print("Loading R — GSM8K …")
    r_gsm  = _load_gsm8k(TARGETS["R_gsm8k"], rng)

    print("Loading R — ARC-Challenge …")
    r_arc  = _load_arc_challenge(TARGETS["R_arc"], rng)

    print("Loading R — StrategyQA …")
    r_strat = _load_strategy_qa(TARGETS["R_strategy"], rng)

    print("Loading R — CommonsenseQA …")
    r_csqa  = _load_commonsense_qa(TARGETS["R_commonsenseqa"], rng)

    print("Loading OOD — handcrafted …")
    ood = _load_ooc()

    print("Loading A 4a — Harmful …")
    a_4a = _load_a4a_harmful(TARGETS["A_4a"], rng)

    print("Loading A 4b — Borderline …")
    a_4b = _load_a4b_borderline(TARGETS["A_4b"], rng)

    print("Loading A 4c — Format-constrained …")
    a_4c = _load_a4c_format(TARGETS["A_4c"], rng)

    print("Loading A 4d — Conversational/persona …")
    a_4d = _load_a4d_conversational()

    print("Loading A 4e — Raw completion …")
    a_4e = _load_a4e_raw_completion(TARGETS["A_4e"], rng)

    all_records = (f_tqa + f_nq + f_wq +
                   r_gsm + r_arc + r_strat + r_csqa +
                   ood +
                   a_4a + a_4b + a_4c + a_4d + a_4e)

    # Assign stable IDs
    counters: dict[str, int] = {}
    for rec in all_records:
        key = f"{rec['split']}_{rec['source']}"
        counters[key] = counters.get(key, 0) + 1
        rec["id"] = _make_id(rec["split"], rec["source"], counters[key])

    rng.shuffle(all_records)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _print_summary(all_records, output_path)
    return all_records


def _print_summary(records: list[dict], output_path: Path) -> None:
    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    qtype_counts: dict[str, int] = {}
    for r in records:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
        key = f"{r['split']}/{r['source']}"
        source_counts[key] = source_counts.get(key, 0) + 1
        qt = r.get("question_type", "?")
        qtype_counts[qt] = qtype_counts.get(qt, 0) + 1

    # A subcategory breakdown
    a_sub_counts: dict[str, int] = {}
    a_behavior_counts: dict[str, int] = {}
    for r in records:
        if r["split"] == "A":
            sc = r.get("metadata", {}).get("alignment_subcategory", "?")
            a_sub_counts[sc] = a_sub_counts.get(sc, 0) + 1
            eb = r.get("metadata", {}).get("expected_behavior", "?")
            a_behavior_counts[eb] = a_behavior_counts.get(eb, 0) + 1

    # OOD subtype breakdown
    ood_type_counts: dict[str, int] = {}
    for r in records:
        if r["split"] == "OOD":
            ot = r.get("ooc_type") or "?"
            ood_type_counts[ot] = ood_type_counts.get(ot, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Exp3 dataset  →  {output_path}")
    print(f"  Total records : {len(records)}")
    print(f"{'='*60}")
    for split, label in [("F", "Factual"), ("R", "Reasoning"),
                          ("OOD", "Out-of-Distribution"), ("A", "Alignment")]:
        n = split_counts.get(split, 0)
        print(f"  {split:<5} ({label:<20}) : {n}")
    print(f"  {'─'*45}")

    print("  Sources:")
    for k, v in sorted(source_counts.items()):
        print(f"    {k:<35} {v}")

    if ood_type_counts:
        print(f"  {'─'*45}")
        print("  OOD subtypes:")
        for k, v in sorted(ood_type_counts.items()):
            print(f"    {k:<30} {v}")

    if a_sub_counts:
        print(f"  {'─'*45}")
        print("  A subcategories:")
        for sc in ["4a", "4b", "4c", "4d", "4e"]:
            if sc in a_sub_counts:
                label = {"4a": "Harmful",
                         "4b": "Borderline",
                         "4c": "Format-constrained",
                         "4d": "Conversational/persona",
                         "4e": "Raw completion"}[sc]
                print(f"    {sc}  {label:<25} {a_sub_counts[sc]}")
        print(f"  A expected behaviors:")
        for k, v in sorted(a_behavior_counts.items()):
            print(f"    {k:<20} {v}")

    print(f"  {'─'*45}")
    print("  Question types:")
    for k, v in sorted(qtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<20} {v}")
    print(f"{'='*60}\n")
    print("Note: F items have verified_pt=null.")
    print("Run  filter_ic.py  to validate F answers with the PT model.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Exp3 evaluation dataset.")
    parser.add_argument("--output", default="data/exp3_dataset.jsonl")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
