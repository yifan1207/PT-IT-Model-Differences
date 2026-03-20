"""
Build the Exp3 evaluation dataset (~2500 questions).

Output: data/exp3_dataset.jsonl  (one JSON object per line)

Splits
------
  IC   (In-Context / factual recall)  — ~625 questions
         TriviaQA (Joshi et al. 2017)               ~225
         NQ-Open  (Kwiatkowski et al. 2019)          ~200
         WebQuestions (Berant et al. 2013)           ~200

  R    (Reasoning)                    — ~700 questions
         GSM8K (Cobbe et al. 2021)                   ~150
         ARC-Challenge (Clark et al. 2018)           ~130
         StrategyQA (Geva et al. 2021)               ~100
         CommonsenseQA (Talmor et al. 2019)          ~125
         BoolQ (Clark et al. 2019)                   ~110
         WinoGrande (Sakaguchi et al. 2021)           ~85

  OOC  (Out-of-Context)               — all handcrafted prompts (~485)
         Novel fictional entities                    ~199
         Post-training unknowable                    ~149
         Counterfactual / indeterminate              ~137

  GEN  (General / diverse — mixes recall and reasoning across domains)
                                       — ~625 questions
         MMLU-Pro (Wang et al. 2024)  25/subj × 14 subjs   ~350
         HumanEval (Chen et al. 2021)                       ~100
         MBPP (Austin et al. 2021)                           ~75
         HellaSwag (Zellers et al. 2019)                    ~100

Two prompt formats per record:

  A: Raw completion — question rephrased as an incomplete statement (natural for PT)
  B: Q&A format    — "Question: …\\nAnswer:" (PRIMARY, same for PT and IT)

Tags
----
Each record has:
  question_type : "factual" | "yes_no" | "numerical" | "multi_choice" |
                  "code" | "completion" | "coreference" |
                  "fictional" | "unknowable" | "counterfactual"
  domain        : subject area (from dataset metadata where available)

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


# ---------------------------------------------------------------------------
# IC loaders
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
        results.append(_rec("IC", "triviaqa", q, answer, aliases,
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
        results.append(_rec("IC", "nq_open", q, answer, aliases, "factual", "trivia"))
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
        answers = item["answers"]  # list of strings
        if not answers:
            continue
        answer = answers[0].strip()
        aliases = [a.strip() for a in answers[1:] if a.strip()]
        results.append(_rec("IC", "web_questions", q, answer, aliases, "factual", "general"))
    return results


# ---------------------------------------------------------------------------
# R loaders
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
                            metadata={"raw_question": q, "choices": dict(zip(labels, texts))}))
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
                            metadata={"raw_question": q, "concept": item.get("question_concept", "")}))
    return results


def _load_boolq(n: int, rng: np.random.Generator) -> list[dict]:
    """BoolQ without passage — tests whether model recalls the fact from pretraining."""
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="train")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        q = item["question"].strip()
        if not q.endswith("?"):
            q += "?"
        # Capitalise first letter
        q = q[0].upper() + q[1:]
        raw = item["answer"]
        answer = "Yes" if raw else "No"
        aliases = ["yes" if raw else "no", "true" if raw else "false"]
        results.append(_rec("R", "boolq", q, answer, aliases, "yes_no", "general"))
    return results


def _load_winogrande(n: int, rng: np.random.Generator) -> list[dict]:
    """WinoGrande: pronoun/coreference fill-in-the-blank reasoning."""
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        sentence = item["sentence"].strip()
        opt1, opt2 = item["option1"].strip(), item["option2"].strip()
        answer_num = str(item["answer"]).strip()  # "1" or "2"
        # Format as a multiple-choice question
        full_q = f"Fill in the blank: {sentence}\n(A) {opt1}  (B) {opt2}"
        answer_letter = "A" if answer_num == "1" else "B"
        answer_text   = opt1 if answer_num == "1" else opt2
        answer = f"({answer_letter}) {answer_text}"
        results.append(_rec("R", "winogrande", full_q, answer, [answer_letter, answer_text],
                            "coreference", "reasoning",
                            metadata={"raw_sentence": sentence}))
    return results


# ---------------------------------------------------------------------------
# GEN loaders
# ---------------------------------------------------------------------------

_MMLU_PRO_ALL_SUBJECTS = [
    "math", "physics", "chemistry", "biology",
    "computer science", "economics", "engineering",
    "health", "history", "law", "math",
    "philosophy", "psychology", "other",
]
# Deduplicated, in dataset order:
MMLU_PRO_SUBJECTS = [
    "math", "physics", "chemistry", "biology",
    "computer science", "economics", "engineering",
    "health", "history", "law",
    "philosophy", "psychology", "other",
]


def _load_mmlu_pro(n_per_subject: int, subjects: list[str],
                   rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    _LETTERS = list("ABCDEFGHIJ")
    ds_full = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    all_categories = set(ds_full["category"])

    # Pre-group by category
    by_cat: dict[str, list] = defaultdict(list)
    for item in ds_full:
        by_cat[item["category"]].append(item)

    results: list[dict] = []
    for subj in subjects:
        matched = [c for c in all_categories if c.lower() == subj.lower()]
        if not matched:
            matched = [c for c in all_categories if subj.lower() in c.lower()]
        if not matched:
            print(f"[WARN] MMLU-Pro subject '{subj}' not found; skipping.", file=sys.stderr)
            continue
        cat_items: list = []
        for m in matched:
            cat_items.extend(by_cat[m])
        n_take = min(n_per_subject, len(cat_items))
        idxs = rng.choice(len(cat_items), size=n_take, replace=False)
        for i in idxs:
            item = cat_items[int(i)]
            q = item["question"].strip()
            options = item["options"]
            answer_letter = item.get("answer", "").strip()
            answer_idx    = item.get("answer_index")
            choices_str = "  ".join(f"({_LETTERS[j]}) {opt}"
                                    for j, opt in enumerate(options)
                                    if j < len(_LETTERS))
            full_q = f"{q}\n{choices_str}"
            if answer_idx is not None and answer_idx < len(options):
                answer = f"({_LETTERS[answer_idx]}) {options[answer_idx]}"
                aliases = [answer_letter, str(answer_idx)]
            elif answer_letter in _LETTERS:
                idx = _LETTERS.index(answer_letter)
                answer = f"({answer_letter}) {options[idx]}" if idx < len(options) else answer_letter
                aliases = [answer_letter]
            else:
                answer = answer_letter or "Unknown"
                aliases = []
            results.append(_rec("GEN", "mmlu_pro", full_q, answer, aliases,
                                "multi_choice", item["category"],
                                metadata={"subject": item["category"], "raw_question": q}))
    return results


def _load_humaneval(n: int, rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        prompt = item["prompt"].strip()
        canonical = item["canonical_solution"].strip()
        results.append(_rec("GEN", "humaneval", prompt, canonical, [],
                            "code", "code",
                            metadata={"task_id": item.get("task_id", "")}))
    return results


def _load_mbpp(n: int, rng: np.random.Generator) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        item = ds[int(i)]
        q = item["prompt"].strip()
        code = item["code"].strip()
        results.append(_rec("GEN", "mbpp", q, code, [],
                            "code", "code",
                            metadata={"task_id": item.get("task_id", "")}))
    return results


def _load_hellaswag(n: int, rng: np.random.Generator) -> list[dict]:
    """
    HellaSwag: given a sentence context, choose the most plausible continuation.
    Question is framed as a 4-way multiple choice.
    Format A uses the raw context (natural completion prompt for PT).
    """
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="train")
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    _LETTERS = ["A", "B", "C", "D"]
    results = []
    for i in indices:
        item = ds[int(i)]
        ctx = item["ctx"].strip()
        endings = item["endings"]   # list of 4 strings
        label   = int(item["label"])
        choices_str = "  ".join(f"({_LETTERS[j]}) {e.strip()}"
                                for j, e in enumerate(endings))
        # Format B question
        full_q = f"What is the most plausible continuation of the following story?\n{ctx}\n{choices_str}"
        answer_letter = _LETTERS[label]
        answer = f"({answer_letter}) {endings[label].strip()}"
        # Format A naturally uses the context as the completion prompt
        rec = _rec("GEN", "hellaswag", full_q, answer, [answer_letter],
                   "completion", "commonsense",
                   metadata={"ctx": ctx, "activity": item.get("activity_label", "")})
        # Override Format A to use bare context (natural completion)
        rec["formats"]["A"] = ctx
        results.append(rec)
        if len(results) == n:
            break
    return results


# ---------------------------------------------------------------------------
# OOC loader
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
        results.append(_rec("OOC", "custom", q, item["answer"], [],
                            _TYPE_TO_QTYPE[ooc_type], "ooc",
                            ooc_type=ooc_type))
    return results


# ---------------------------------------------------------------------------
# Coding fallback prompts (if HumanEval/MBPP unavailable)
# ---------------------------------------------------------------------------

_CODING_FALLBACK: list[tuple[str, str]] = [
    ("Write a Python function `is_prime(n)` that returns True if n is a prime number.",
     "Return False for n<=1. Check divisors up to sqrt(n). Return True if none found."),
    ("Write a Python function `fibonacci(n)` returning the n-th Fibonacci number iteratively.",
     "Use two variables a, b = 0, 1 and iterate n times."),
    ("Write a Python function `binary_search(arr, target)` returning the index or -1.",
     "lo, hi, mid = 0, len-1, (lo+hi)//2. Compare and halve. O(log n)."),
    ("Write a Python function `flatten(nested)` that recursively flattens nested lists.",
     "Recurse on list elements; otherwise append to output list."),
    ("Write a Python function `lru_cache_class(capacity)` implementing an LRU cache.",
     "Use OrderedDict. move_to_end on get. popitem(last=False) on overflow."),
]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

TARGETS = {
    "IC_triviaqa":       225,
    "IC_nq_open":        200,
    "IC_webquestions":   200,
    "R_gsm8k":           150,
    "R_arc":             130,
    "R_strategy":        100,
    "R_commonsenseqa":   125,
    "R_boolq":           110,
    "R_winogrande":       85,
    "GEN_mmlu_per_subj":  25,   # × 13 subjects ≈ 325
    "GEN_humaneval":     100,
    "GEN_mbpp":           75,
    "GEN_hellaswag":     100,
}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_dataset(output_path: str | Path, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading IC — TriviaQA …")
    ic_tqa  = _load_trivia_qa(TARGETS["IC_triviaqa"], rng)

    print("Loading IC — NQ-Open …")
    ic_nq   = _load_nq_open(TARGETS["IC_nq_open"], rng)

    print("Loading IC — WebQuestions …")
    ic_wq   = _load_web_questions(TARGETS["IC_webquestions"], rng)

    print("Loading R — GSM8K …")
    r_gsm   = _load_gsm8k(TARGETS["R_gsm8k"], rng)

    print("Loading R — ARC-Challenge …")
    r_arc   = _load_arc_challenge(TARGETS["R_arc"], rng)

    print("Loading R — StrategyQA …")
    r_strat = _load_strategy_qa(TARGETS["R_strategy"], rng)

    print("Loading R — CommonsenseQA …")
    r_csqa  = _load_commonsense_qa(TARGETS["R_commonsenseqa"], rng)

    print("Loading R — BoolQ …")
    r_bool  = _load_boolq(TARGETS["R_boolq"], rng)

    print("Loading R — WinoGrande …")
    r_wino  = _load_winogrande(TARGETS["R_winogrande"], rng)

    print("Loading GEN — MMLU-Pro …")
    gen_mmlu = _load_mmlu_pro(TARGETS["GEN_mmlu_per_subj"], MMLU_PRO_SUBJECTS, rng)

    print("Loading GEN — HumanEval …")
    gen_he   = _load_humaneval(TARGETS["GEN_humaneval"], rng)

    print("Loading GEN — MBPP …")
    gen_mbpp = _load_mbpp(TARGETS["GEN_mbpp"], rng)

    print("Loading GEN — HellaSwag …")
    gen_hs   = _load_hellaswag(TARGETS["GEN_hellaswag"], rng)

    print("Loading OOC — handcrafted …")
    ooc = _load_ooc()

    all_records = (ic_tqa + ic_nq + ic_wq +
                   r_gsm + r_arc + r_strat + r_csqa + r_bool + r_wino +
                   gen_mmlu + gen_he + gen_mbpp + gen_hs +
                   ooc)

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

    print(f"\n{'='*60}")
    print(f"  Exp3 dataset  →  {output_path}")
    print(f"  Total records : {len(records)}")
    print(f"{'='*60}")
    for split in ["IC", "R", "OOC", "GEN"]:
        print(f"  {split:<5} : {split_counts.get(split, 0)}")
    print(f"  {'─'*45}")
    for k, v in sorted(source_counts.items()):
        print(f"    {k:<35} {v}")
    print(f"  {'─'*45}")
    print("  Question types:")
    for k, v in sorted(qtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<20} {v}")
    print(f"{'='*60}\n")
    print("Note: IC items have verified_pt=null.")
    print("Run  filter_ic.py  to validate IC answers with the PT model.")


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
