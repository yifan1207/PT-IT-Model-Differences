"""
Build the Exp3 evaluation dataset (~2935 prompts).

Output: data/exp3_dataset.jsonl  (one JSON object per line)

Categories
----------
  F    Factual Knowledge (~625)
         TriviaQA (Joshi et al. 2017)               ~225
         NQ-Open  (Kwiatkowski et al. 2019)          ~200
         WebQuestions (Berant et al. 2013)           ~200
         Control: model knows the answer from PT.  Corrective stage baseline.

  R    Reasoning (~700)
         GSM8K (Cobbe et al. 2021)                   ~150
         ARC-Challenge (Clark et al. 2018)           ~130
         StrategyQA (Geva et al. 2021)               ~100
         CommonsenseQA (Talmor et al. 2019)          ~125
         BoolQ (Clark et al. 2019)                   ~110
         WinoGrande (Sakaguchi et al. 2021)          ~85
         Tests whether corrective stage activates MORE for reasoning vs factual.

  OOD  Out-of-Distribution (~485)
         Novel fictional entities                    ~200
         Post-training unknowable                    ~150
         Counterfactual / indeterminate              ~135
         Model has no confident answer — tests suppression/hedging behavior.

  GEN  General/Diverse (~625)
         MMLU-Pro (Wang et al. 2024)                 ~325
         HumanEval (Chen et al. 2021)                ~100
         MBPP (Austin et al. 2021)                    ~75
         HellaSwag (Zellers et al. 2019)             ~125
         Diverse tasks: multi-domain MCQA, code gen, commonsense NLI.

  A    Alignment-Sensitive (~500) — directly tests corrective stage function
         5a  Harmful requests (AdvBench + hand-written)          ~100
         5b  Borderline/ambiguous (XSTest + hand-written)         ~80
         5c  Format-constrained (IFEval + hand-written)          ~120
         5d  Conversational/persona (multi-turn + roleplay)      ~100
         5e  Raw completion (Wikipedia truncations + hand-written)~100

Two prompt formats per record:

  A: Raw completion — question rephrased as an incomplete statement (natural for PT)
  B: Q&A format    — "Question: …\\nAnswer:" (PRIMARY, same for PT and IT)
  [Category A 5e exception: both formats are the raw continuation text]

Tags
----
Each record has:
  split         : "F" | "R" | "OOD" | "GEN" | "A"
  question_type : "factual" | "yes_no" | "numerical" | "multi_choice" |
                  "code" | "completion" | "coreference" |
                  "fictional" | "unknowable" | "counterfactual" |
                  "harmful" | "borderline" | "format" | "continuation" |
                  "conversation" | "roleplay"
  domain        : subject area
  metadata.alignment_subcategory : "5a"–"5e" for A-split records
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


def _load_boolq(n: int, rng: np.random.Generator) -> list[dict]:
    """BoolQ — yes/no reading comprehension questions."""
    from datasets import load_dataset
    try:
        ds = load_dataset("google/boolq", split="train")
    except Exception:
        ds = load_dataset("boolq", split="train")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        q = item["question"].strip()
        if not q.endswith("?"):
            q += "?"
        answer = "Yes" if item["answer"] else "No"
        aliases = ["yes" if item["answer"] else "no"]
        passage = item.get("passage", "")
        results.append(_rec("R", "boolq", q, answer, aliases,
                            "yes_no", "reading_comprehension",
                            metadata={"passage": passage[:300] if passage else ""}))
    return results


def _load_winogrande(n: int, rng: np.random.Generator) -> list[dict]:
    """WinoGrande — commonsense pronoun resolution."""
    from datasets import load_dataset
    try:
        ds = load_dataset("winogrande", "winogrande_xl", split="train",
                          trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] WinoGrande unavailable ({e}); skipping.", file=sys.stderr)
        return []
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        sentence = item["sentence"].strip()
        opt1 = item["option1"].strip()
        opt2 = item["option2"].strip()
        answer_key = item["answer"].strip()  # "1" or "2"
        answer = opt1 if answer_key == "1" else opt2
        q = f"{sentence}\n(1) {opt1}  (2) {opt2}"
        results.append(_rec("R", "winogrande", q, answer, [],
                            "coreference", "commonsense",
                            metadata={"option1": opt1, "option2": opt2}))
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
# GEN loaders (General / Diverse)
# ---------------------------------------------------------------------------

def _load_mmlu_pro(n: int, rng: np.random.Generator) -> list[dict]:
    """MMLU-Pro — multi-domain 10-choice MCQA.

    Loads 25 items from each of 13 subjects = 325 total.
    """
    from datasets import load_dataset
    SUBJECTS = [
        "math", "physics", "chemistry", "biology", "computer science",
        "engineering", "law", "economics", "psychology", "history",
        "philosophy", "health", "business",
    ]
    per_subject = max(1, n // len(SUBJECTS))
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    except Exception as e:
        print(f"[WARN] MMLU-Pro unavailable ({e}); skipping.", file=sys.stderr)
        return []

    by_subject: dict[str, list] = defaultdict(list)
    for item in ds:
        subj = item.get("category", "general").lower()
        for s in SUBJECTS:
            if s in subj:
                by_subject[s].append(item)
                break
        else:
            by_subject["general"].append(item)

    results = []
    for subj in SUBJECTS:
        items = by_subject[subj]
        if not items:
            continue
        idxs = rng.choice(len(items), size=min(per_subject, len(items)), replace=False)
        for idx in idxs:
            item = items[int(idx)]
            q = item["question"].strip()
            options = item.get("options", [])
            choices_str = "  ".join(f"({chr(65+j)}) {opt}"
                                    for j, opt in enumerate(options))
            full_q = f"{q}\n{choices_str}" if choices_str else q
            answer_idx = item.get("answer_index", 0)
            answer = (f"({chr(65+answer_idx)}) {options[answer_idx]}"
                      if options and answer_idx < len(options)
                      else item.get("answer", ""))
            results.append(_rec("GEN", "mmlu_pro", full_q, answer,
                                [chr(65 + answer_idx)],
                                "multi_choice", subj,
                                metadata={"subject": subj}))
    return results


def _load_humaneval(n: int, rng: np.random.Generator) -> list[dict]:
    """HumanEval — Python function synthesis from docstrings."""
    from datasets import load_dataset
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        try:
            ds = load_dataset("openai/openai-human-eval", split="test")
        except Exception as e:
            print(f"[WARN] HumanEval unavailable ({e}); skipping.", file=sys.stderr)
            return []

    all_items = list(ds)
    indices = rng.choice(len(all_items), size=min(n, len(all_items)), replace=False)
    results = []
    for i in indices:
        item = all_items[int(i)]
        prompt = item["prompt"].strip()
        entry = item.get("entry_point", "")
        q = f"Complete the following Python function:\n\n{prompt}"
        results.append(_rec("GEN", "humaneval", q, "", [],
                            "code", "programming",
                            metadata={"task_id": item.get("task_id", ""),
                                      "entry_point": entry}))
    return results


def _load_mbpp(n: int, rng: np.random.Generator) -> list[dict]:
    """MBPP — Mostly Basic Python Programming problems."""
    from datasets import load_dataset
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    except Exception:
        try:
            ds = load_dataset("mbpp", split="train")
        except Exception as e:
            print(f"[WARN] MBPP unavailable ({e}); skipping.", file=sys.stderr)
            return []

    all_items = list(ds)
    indices = rng.choice(len(all_items), size=min(n, len(all_items)), replace=False)
    results = []
    for i in indices:
        item = all_items[int(i)]
        text = item.get("text", item.get("prompt", "")).strip()
        q = f"Write a Python function to {text}"
        test_list = item.get("test_list", [])
        results.append(_rec("GEN", "mbpp", q, "", [],
                            "code", "programming",
                            metadata={"task_id": item.get("task_id", ""),
                                      "test_list": test_list[:3]}))
    return results


def _load_hellaswag(n: int, rng: np.random.Generator) -> list[dict]:
    """HellaSwag — commonsense NLI, sentence completion."""
    from datasets import load_dataset
    try:
        ds = load_dataset("Rowan/hellaswag", split="train")
    except Exception as e:
        print(f"[WARN] HellaSwag unavailable ({e}); skipping.", file=sys.stderr)
        return []

    all_items = list(ds)
    indices = rng.choice(len(all_items), size=min(n, len(all_items)), replace=False)
    results = []
    for i in indices:
        item = all_items[int(i)]
        ctx = (item.get("ctx") or "").strip()
        if not ctx:
            ctx_a = (item.get("ctx_a") or "").strip()
            ctx_b = (item.get("ctx_b") or "").strip()
            ctx = f"{ctx_a} {ctx_b}".strip()
        endings = item.get("endings", [])
        label = int(item.get("label", 0))
        choices_str = "  ".join(f"({chr(65+j)}) {e}" for j, e in enumerate(endings))
        q = f"{ctx}\n{choices_str}" if choices_str else ctx
        answer = (f"({chr(65+label)}) {endings[label]}"
                  if endings and label < len(endings) else "")
        results.append(_rec("GEN", "hellaswag", q, answer,
                            [chr(65 + label)],
                            "multi_choice", "commonsense",
                            metadata={"activity_label": item.get("activity_label", "")}))
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


def _load_a5a_harmful(n: int, rng: np.random.Generator) -> list[dict]:
    """5a: Direct harmful requests.
    Priority: (1) AdvBench GitHub CSV, (2) AdvBench on HuggingFace,
    (3) hand-written fallback list.
    """
    from src.poc.exp3.data.safety_prompts import A4A_HARMFUL

    # Try GitHub CSV first (public, no auth required)
    github_goals = _load_advbench_csv()
    if github_goals:
        records = [_rec_a("5a", "advbench", q, "harmful", "safety", "refuse")
                   for q in github_goals]
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
                        _rec_a("5a", "advbench", str(item[col_use]).strip(),
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
    print("  [INFO] Using hand-written A5A prompts (AdvBench not accessible).",
          file=sys.stderr)
    records = [_rec_a("5a", "malicious_instruct", q, "harmful", "safety", "refuse")
               for q in A4A_HARMFUL]
    if len(records) > n:
        idxs = rng.choice(len(records), size=n, replace=False)
        records = [records[int(i)] for i in idxs]
    return records


def _load_a5b_borderline(n: int, rng: np.random.Generator) -> list[dict]:
    """5b: Borderline/ambiguous — XSTest (HF) + hand-written.

    XSTest (Röttger et al. 2023, Paul/xstest) has 250 safe prompts
    (label=="safe") that look harmful but should be answered.
    We use the 'label' field to select all 250.
    """
    from src.poc.exp3.data.safety_prompts import A4B_BORDERLINE
    hf_records: list[dict] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("Paul/xstest", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        safe_items = [x for x in ds if str(x.get("label", "")).lower() == "safe"]
        if not safe_items:
            safe_items = list(ds)
        for item in safe_items:
            hf_records.append(
                _rec_a("5b", "xstest", str(item[col]).strip(),
                       "borderline", "safety", "comply"))
        print(f"  Loaded {len(hf_records)} XSTest safe prompts.")
    except Exception as e:
        print(f"[WARN] XSTest unavailable ({e}); using hand-written 5b only.",
              file=sys.stderr)

    hand_written = [_rec_a("5b", "custom", q, "borderline", "safety", "comply")
                    for q in A4B_BORDERLINE]
    combined = hf_records + hand_written
    if len(combined) > n:
        idxs = rng.choice(len(combined), size=n, replace=False)
        combined = [combined[int(i)] for i in idxs]
    return combined


def _load_a5c_format(n: int, rng: np.random.Generator) -> list[dict]:
    """5c: Format-constrained — IFEval (HF) + hand-written."""
    from src.poc.exp3.data.safety_prompts import A4C_FORMAT
    hf_records: list[dict] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("google/IFEval", split="train")
        col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        for item in ds:
            q = str(item[col]).strip()
            r = _rec_a("5c", "ifeval", q, "format", "instruction_following", "comply")
            r["formats"]["B"] = q
            hf_records.append(r)
    except Exception as e:
        print(f"[WARN] IFEval unavailable ({e}); using hand-written 5c only.",
              file=sys.stderr)

    hand_written = [_rec_a("5c", "custom", q, "format", "instruction_following", "comply")
                    for q in A4C_FORMAT]
    combined = hf_records + hand_written
    if len(combined) > n:
        idxs = rng.choice(len(combined), size=n, replace=False)
        combined = [combined[int(i)] for i in idxs]
    return combined


def _load_a5d_conversational() -> list[dict]:
    """5d: Conversational/persona — multi-turn + roleplay, all hand-written."""
    from src.poc.exp3.data.safety_prompts import A4D_MULTI_TURN, A4D_ROLEPLAY
    results = []
    for prompt, expected in A4D_MULTI_TURN:
        results.append(_rec_a("5d", "custom", prompt, "conversation",
                              "general", expected))
    for prompt, expected in A4D_ROLEPLAY:
        results.append(_rec_a("5d", "custom", prompt, "roleplay",
                              "general", expected))
    return results


def _load_a5e_raw_completion(n: int, rng: np.random.Generator) -> list[dict]:
    """5e: Raw completion — Wikipedia truncations + hand-written.
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
        print(f"[WARN] Wikipedia unavailable ({e}); using hand-written 5e only.",
              file=sys.stderr)

    candidates.extend(A4E_EXTRA_CONTINUATIONS)

    if len(candidates) > n:
        idxs = rng.choice(len(candidates), size=n, replace=False)
        candidates = [candidates[int(i)] for i in idxs]

    return [_rec_a("5e", "custom" if t in A4E_EXTRA_CONTINUATIONS else "wikipedia",
                   t, "continuation", "general", "comply", raw_continuation=True)
            for t in candidates]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

TARGETS = {
    # F: Factual Knowledge (~625)
    "F_triviaqa":      225,
    "F_nq_open":       200,
    "F_webquestions":  200,
    # R: Reasoning (~700)
    "R_gsm8k":         150,
    "R_arc":           130,
    "R_strategy":      100,
    "R_commonsenseqa": 125,
    "R_boolq":         110,
    "R_winogrande":     85,
    # OOD: Out-of-Distribution — from ooc_prompts.py (all loaded)
    # GEN: General/Diverse (~625)
    "GEN_mmlu_pro":    325,
    "GEN_humaneval":   100,
    "GEN_mbpp":         75,
    "GEN_hellaswag":   125,
    # A: Alignment-Sensitive
    "A_5a":            100,
    "A_5b":             80,
    "A_5c":            120,
    # A_5d: all hand-written (~100 from multi-turn + roleplay combined)
    # A_5e: 100 total (Wikipedia + hand-written)
    "A_5e":            100,
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

    print("Loading R — BoolQ …")
    r_boolq = _load_boolq(TARGETS["R_boolq"], rng)

    print("Loading R — WinoGrande …")
    r_wino  = _load_winogrande(TARGETS["R_winogrande"], rng)

    print("Loading OOD — handcrafted …")
    ood = _load_ooc()

    print("Loading GEN — MMLU-Pro …")
    g_mmlu  = _load_mmlu_pro(TARGETS["GEN_mmlu_pro"], rng)

    print("Loading GEN — HumanEval …")
    g_heval = _load_humaneval(TARGETS["GEN_humaneval"], rng)

    print("Loading GEN — MBPP …")
    g_mbpp  = _load_mbpp(TARGETS["GEN_mbpp"], rng)

    print("Loading GEN — HellaSwag …")
    g_hella = _load_hellaswag(TARGETS["GEN_hellaswag"], rng)

    print("Loading A 5a — Harmful …")
    a_5a = _load_a5a_harmful(TARGETS["A_5a"], rng)

    print("Loading A 5b — Borderline …")
    a_5b = _load_a5b_borderline(TARGETS["A_5b"], rng)

    print("Loading A 5c — Format-constrained …")
    a_5c = _load_a5c_format(TARGETS["A_5c"], rng)

    print("Loading A 5d — Conversational/persona …")
    a_5d = _load_a5d_conversational()

    print("Loading A 5e — Raw completion …")
    a_5e = _load_a5e_raw_completion(TARGETS["A_5e"], rng)

    all_records = (f_tqa + f_nq + f_wq +
                   r_gsm + r_arc + r_strat + r_csqa + r_boolq + r_wino +
                   ood +
                   g_mmlu + g_heval + g_mbpp + g_hella +
                   a_5a + a_5b + a_5c + a_5d + a_5e)

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
    for split, label in [("F",   "Factual"),
                          ("R",   "Reasoning"),
                          ("OOD", "Out-of-Distribution"),
                          ("GEN", "General/Diverse"),
                          ("A",   "Alignment")]:
        n = split_counts.get(split, 0)
        print(f"  {split:<5} ({label:<20}) : {n}")
    print(f"  {'─'*45}")

    print("  Sources:")
    for k, v in sorted(source_counts.items()):
        print(f"    {k:<40} {v}")

    if ood_type_counts:
        print(f"  {'─'*45}")
        print("  OOD subtypes:")
        for k, v in sorted(ood_type_counts.items()):
            print(f"    {k:<30} {v}")

    if a_sub_counts:
        print(f"  {'─'*45}")
        print("  A subcategories:")
        labels_map = {
            "5a": "Harmful",
            "5b": "Borderline",
            "5c": "Format-constrained",
            "5d": "Conversational/persona",
            "5e": "Raw completion",
        }
        for sc in ["5a", "5b", "5c", "5d", "5e"]:
            if sc in a_sub_counts:
                print(f"    {sc}  {labels_map[sc]:<25} {a_sub_counts[sc]}")
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
