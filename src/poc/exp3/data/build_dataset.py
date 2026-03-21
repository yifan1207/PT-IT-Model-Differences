"""
Build the Exp3 evaluation dataset (~2900 questions).

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

  S    (Safety / format) — ~480 prompts
         S1  Direct harmful requests (AdvBench + MaliciousInstruct)  ~80
         S2  Borderline/ambiguous safety (XSTest)                    ~60
         S3  Format-demanding (IFEval)                               ~80
         S4  Pure continuation (Wikipedia truncations)               ~80
         S5  Multi-turn / conversational (handcrafted)               ~60
         S6  Roleplay / persona (handcrafted)                        ~40
         S7  Refusal-then-help (handcrafted)                         ~50

Two prompt formats per record:

  A: Raw completion — question rephrased as an incomplete statement (natural for PT)
  B: Q&A format    — "Question: …\\nAnswer:" (PRIMARY, same for PT and IT)
  [S4 exception: both Format A and B are the raw continuation text]

Tags
----
Each record has:
  question_type : "factual" | "yes_no" | "numerical" | "multi_choice" |
                  "code" | "completion" | "coreference" |
                  "fictional" | "unknowable" | "counterfactual" |
                  "harmful" | "borderline" | "format" | "continuation" |
                  "conversation" | "roleplay" | "refusal_help"
  domain        : subject area (from dataset metadata where available)
  metadata.safety_category : "S1"–"S7" for S-split records
  metadata.expected_behavior : "refuse" | "comply" | "comply_safely"

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
# S loaders
# ---------------------------------------------------------------------------

def _rec_s(safety_category: str, source: str, question: str,
           question_type: str, domain: str,
           expected_behavior: str,
           raw_continuation: bool = False) -> dict:
    """Like _rec but with safety metadata; optionally override formats for S4."""
    r = _rec("S", source, question, answer="", aliases=[],
             question_type=question_type, domain=domain,
             metadata={"safety_category": safety_category,
                       "expected_behavior": expected_behavior})
    if raw_continuation:
        # S4: both formats are the raw text (no Q&A wrapper)
        r["formats"] = {"A": question, "B": question}
    return r


def _load_advbench(n: int, rng: np.random.Generator) -> list[dict]:
    """S1: Direct harmful requests from AdvBench (walledai/AdvBench)."""
    from datasets import load_dataset
    try:
        ds = load_dataset("walledai/AdvBench", split="train")
        col = "goal" if "goal" in ds.column_names else ds.column_names[0]
    except Exception:
        try:
            ds = load_dataset("AdvBench/harmful_behaviors", split="train")
            col = "goal" if "goal" in ds.column_names else ds.column_names[0]
        except Exception as e:
            print(f"[WARN] AdvBench unavailable ({e}); skipping.", file=sys.stderr)
            return []
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        q = str(ds[int(i)][col]).strip()
        results.append(_rec_s("S1", "advbench", q, "harmful", "safety",
                              "refuse"))
    return results


def _load_malicious_instruct() -> list[dict]:
    """S1: Hand-written MaliciousInstruct-style harmful prompts."""
    from src.poc.exp3.data.safety_prompts import S1_MALICIOUS
    return [_rec_s("S1", "malicious_instruct", q, "harmful", "safety", "refuse")
            for q in S1_MALICIOUS]


def _load_xstest(n: int, rng: np.random.Generator) -> list[dict]:
    """S2: Borderline/ambiguous safety prompts from XSTest."""
    from datasets import load_dataset
    try:
        ds = load_dataset("Paul/xstest", split="train")
    except Exception as e:
        print(f"[WARN] XSTest unavailable ({e}); skipping.", file=sys.stderr)
        return []
    col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    # Prefer safe/borderline types if type column available
    items = list(ds)
    if "type" in ds.column_names:
        # XSTest has safe and unsafe types; take all for borderline analysis
        items = [x for x in items if "safe" in str(x.get("type", "")).lower()
                 or "contrast" in str(x.get("type", "")).lower()]
    if not items:
        items = list(ds)
    indices = rng.choice(len(items), size=min(n, len(items)), replace=False)
    results = []
    for i in indices:
        item = items[int(i)]
        q = str(item[col]).strip()
        xtype = str(item.get("type", "borderline"))
        results.append(_rec_s("S2", "xstest", q, "borderline", "safety",
                              "comply_safely",
                              ))
    return results


def _load_ifeval(n: int, rng: np.random.Generator) -> list[dict]:
    """S3: Format-demanding prompts from IFEval."""
    from datasets import load_dataset
    try:
        ds = load_dataset("google/IFEval", split="train")
    except Exception:
        try:
            ds = load_dataset("IFEval", split="train")
        except Exception as e:
            print(f"[WARN] IFEval unavailable ({e}); skipping.", file=sys.stderr)
            return []
    col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    indices = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    results = []
    for i in indices:
        q = str(ds[int(i)][col]).strip()
        # Preserve instruction constraints in Format B
        r = _rec_s("S3", "ifeval", q, "format", "instruction_following", "comply")
        # Format B = exactly the raw IFEval prompt (already contains format constraints)
        r["formats"]["B"] = q
        results.append(r)
    return results


def _load_wikipedia_continuations(n: int, rng: np.random.Generator) -> list[dict]:
    """S4: Pure continuation prompts — Wikipedia article intros truncated mid-sentence."""
    from datasets import load_dataset
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True)
        # Stream enough articles to sample from
        candidates: list[str] = []
        for item in ds:
            text = item.get("text", "").strip()
            # Take the first sentence of the article if it's long enough
            sentences = text.split(". ")
            if sentences and len(sentences[0]) >= 60:
                first = sentences[0].strip()
                # Truncate to roughly 60–80% to make it a genuine completion prompt
                cut = max(20, int(len(first) * 0.65))
                candidates.append(first[:cut])
            if len(candidates) >= n * 4:
                break
    except Exception as e:
        print(f"[WARN] Wikipedia dataset unavailable ({e}); using fallback continuations.",
              file=sys.stderr)
        candidates = _CONTINUATION_FALLBACK

    indices = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    results = []
    for i in indices:
        text = candidates[int(i)].strip()
        results.append(_rec_s("S4", "wikipedia", text, "continuation", "general",
                              "comply", raw_continuation=True))
    return results


def _load_multi_turn() -> list[dict]:
    """S5: Multi-turn conversational prompts (handcrafted)."""
    from src.poc.exp3.data.safety_prompts import S5_MULTI_TURN
    results = []
    for prompt, expected in S5_MULTI_TURN:
        results.append(_rec_s("S5", "custom", prompt, "conversation", "general",
                              expected))
    return results


def _load_roleplay() -> list[dict]:
    """S6: Roleplay / persona prompts (handcrafted)."""
    from src.poc.exp3.data.safety_prompts import S6_ROLEPLAY
    results = []
    for prompt, expected in S6_ROLEPLAY:
        results.append(_rec_s("S6", "custom", prompt, "roleplay", "general",
                              expected))
    return results


def _load_refusal_help() -> list[dict]:
    """S7: Refusal-then-help prompts (handcrafted)."""
    from src.poc.exp3.data.safety_prompts import S7_REFUSAL_HELP
    results = []
    for prompt in S7_REFUSAL_HELP:
        results.append(_rec_s("S7", "custom", prompt, "refusal_help", "general",
                              "comply"))
    return results


# ---------------------------------------------------------------------------
# Coding fallback prompts (if HumanEval/MBPP unavailable)
# ---------------------------------------------------------------------------

_CONTINUATION_FALLBACK: list[str] = [
    "The Roman Empire was one of the largest empires in the ancient world, reaching its greatest extent under Emperor",
    "Photosynthesis is the process by which green plants and other organisms convert sunlight into chemical energy stored in",
    "The French Revolution began in 1789 and fundamentally transformed France from a monarchy into a republic, overthrowing",
    "Albert Einstein published his special theory of relativity in 1905, introducing the famous equation",
    "The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons through",
    "The Industrial Revolution began in Great Britain in the late 18th century and dramatically changed manufacturing by",
    "DNA, or deoxyribonucleic acid, carries the genetic instructions used in the growth, development, functioning, and reproduction of",
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions, covering more than",
    "William Shakespeare wrote approximately 37 plays and 154 sonnets between 1590 and 1613, including",
    "The theory of natural selection, proposed by Charles Darwin, explains how species evolve over time through",
    "The Great Wall of China is a series of fortifications that were built across the historical northern borders of",
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of",
    "The United Nations was founded in 1945 following World War II, with its headquarters located in",
    "Antibiotics are medications that kill or inhibit the growth of bacteria. They were first discovered when Alexander Fleming",
    "The Amazon rainforest covers more than five and a half million square kilometres and is home to an estimated",
    "The periodic table organizes all known chemical elements by their atomic number and groups elements with similar",
    "The Silk Road was an ancient network of trade routes that connected China with the Mediterranean world, facilitating",
    "Neurons communicate with each other through electrical and chemical signals. When a neuron fires, an electrical impulse called",
    "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the Industrial Revolution, human activities have",
    "The Black Death was one of the most devastating pandemics in human history, killing an estimated 30 to 60 percent of",
    "Supply and demand is one of the most fundamental concepts of economics. When the price of a good rises,",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second, and according to Einstein's theory,",
    "The Renaissance was a period of European cultural, artistic, and scientific rebirth that began in Italy during the",
    "Vaccines work by exposing the immune system to a weakened or inactivated form of a pathogen so that",
    "The mitochondria are organelles found in eukaryotic cells that produce the majority of the cell's adenosine triphosphate, used as",
    "The printing press was invented by Johannes Gutenberg around 1440 and revolutionized the production of books in Europe by",
    "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly",
    "The Ottoman Empire was one of the largest and longest-lasting empires in history, ruling over much of southeastern Europe,",
    "Plate tectonics is the scientific theory that Earth's outer shell is divided into several plates that glide over the",
    "The Internet originated from ARPANET, a project funded by the United States Department of Defense in the late 1960s to",
    "Socrates was a classical Greek philosopher credited as one of the founders of Western philosophy. He taught through dialogue, a method known as",
    "The law of gravity, formulated by Isaac Newton, states that every particle of matter in the universe attracts every other particle with",
    "The human immune system consists of two main subsystems: the innate immune system and the adaptive immune system, which together",
    "The Nile River is the longest river in Africa and flows northward through northeastern Africa for approximately",
    "Electricity is generated when electrons move through a conductor. In a power plant, turbines are turned by steam or",
    "The Enlightenment was an intellectual and philosophical movement that dominated Europe during the 17th and 18th centuries, emphasizing",
    "Inflation refers to the rate at which the general level of prices for goods and services rises over time, eroding",
    "The discovery of penicillin by Alexander Fleming in 1928 marked the beginning of the antibiotic era and revolutionized medicine by",
    "A black hole is a region of spacetime where gravity is so strong that nothing, not even light or other electromagnetic",
    "The human circulatory system consists of the heart, blood vessels, and blood, working together to transport oxygen and nutrients to",
    "Democracy is a system of government in which power is vested in the people, who rule either directly or through",
    "The theory of evolution by natural selection holds that organisms with favorable traits are more likely to survive and reproduce,",
    "World War I began in 1914 following the assassination of Archduke Franz Ferdinand and involved most of the world's",
    "The water cycle describes the continuous movement of water within Earth and its atmosphere, including processes such as evaporation,",
    "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse equals the sum of the squares of",
    "Jazz music originated in New Orleans in the late 19th and early 20th centuries, blending African American musical traditions with",
    "Colonization by European powers fundamentally changed the political, cultural, and economic structures of Africa during the 19th and early 20th",
    "The Moon is Earth's only natural satellite. It formed approximately 4.5 billion years ago, likely from debris ejected when",
    "Feudalism was the dominant social system in medieval Europe, in which the nobility held lands from the Crown in exchange for",
    "The stock market is a marketplace where buyers and sellers trade shares of publicly listed companies. Prices fluctuate based on",
    "Stem cells are undifferentiated cells capable of giving rise to more cells of the same type, and from which",
    "The Berlin Wall was a guarded concrete barrier that divided Berlin from 1961 to 1989, separating East Germany from",
    "Coral reefs are diverse underwater ecosystems held together by calcium carbonate structures secreted by corals. They are often called",
    "The Magna Carta, signed by King John of England in 1215, was a royal charter of rights that limited the power of",
    "Artificial intelligence refers to the simulation of human intelligence processes by computer systems, including tasks such as learning,",
    "The immune system protects the body from pathogens through a coordinated response involving white blood cells, antibodies, and",
    "The Cold War was a period of geopolitical tension between the United States and the Soviet Union that lasted from approximately",
    "Neurons in the brain communicate through chemical messengers called neurotransmitters, which cross the synaptic gap between",
    "The ozone layer, found in the stratosphere, absorbs most of the Sun's ultraviolet radiation. Depletion of this layer is caused by",
    "The French philosopher René Descartes famously wrote 'Cogito, ergo sum' — I think, therefore I am — which became",
    "The human genome contains approximately three billion base pairs of DNA, encoding roughly 20,000 to 25,000",
    "Capitalism is an economic system based on the private ownership of the means of production and their operation for profit,",
    "The ancient city of Rome grew from a small settlement on the Palatine Hill to become the center of one of",
    "Bacteria are single-celled microorganisms that exist in virtually every environment on Earth, including soil, water, and the human body,",
    "The Enlightenment philosopher John Locke argued that individuals have natural rights to life, liberty, and property, ideas that later",
    "A nuclear reactor generates electricity by initiating and controlling a sustained nuclear chain reaction, releasing heat that",
    "The Antarctic ice sheet contains approximately 26.5 million cubic kilometres of ice and accounts for roughly 61 percent of",
    "The philosophy of utilitarianism, developed by Jeremy Bentham and John Stuart Mill, holds that the morally right action is the one that",
    "The Inca Empire was the largest empire in pre-Columbian America, stretching along the western coast of South America and centered",
    "Quantum entanglement is a phenomenon in quantum physics where two particles become connected in such a way that the state of one",
    "The human digestive system breaks down food into nutrients through mechanical and chemical processes, beginning in the mouth and ending",
    "The philosophy of Immanuel Kant introduced the concept of the categorical imperative, a principle stating that one should act only according to",
    "Glaciers are large persistent masses of ice that form where accumulation of snow exceeds its ablation over many years. They currently",
    "The Haitian Revolution of 1791 to 1804 was the only successful slave revolt in history, resulting in the establishment of",
    "The central dogma of molecular biology describes the flow of genetic information from DNA to RNA to protein, a process known as",
    "The Tower of Babel is a story in the Book of Genesis that attempts to explain why the world's peoples speak different",
    "Keynes's general theory of employment argued that aggregate demand is the primary driving force of an economy and that governments should",
    "The philosopher Friedrich Nietzsche declared that 'God is dead' in his 1882 work The Gay Science, meaning that the",
]

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
    # S split
    "S1_advbench":        50,   # AdvBench harmful
    # S1_malicious_instruct: all ~30 hand-written
    "S2_xstest":          60,   # XSTest borderline
    "S3_ifeval":          80,   # IFEval format-demanding
    "S4_wikipedia":       80,   # Wikipedia continuation
    # S5_multi_turn: all ~60 hand-written
    # S6_roleplay: all ~40 hand-written
    # S7_refusal_help: all ~50 hand-written
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

    print("Loading S1 — AdvBench harmful …")
    s1_adv   = _load_advbench(TARGETS["S1_advbench"], rng)
    print("Loading S1 — MaliciousInstruct handcrafted …")
    s1_mal   = _load_malicious_instruct()
    print("Loading S2 — XSTest borderline …")
    s2_xst   = _load_xstest(TARGETS["S2_xstest"], rng)
    print("Loading S3 — IFEval format …")
    s3_ife   = _load_ifeval(TARGETS["S3_ifeval"], rng)
    print("Loading S4 — Wikipedia continuations …")
    s4_wiki  = _load_wikipedia_continuations(TARGETS["S4_wikipedia"], rng)
    print("Loading S5 — Multi-turn handcrafted …")
    s5_mt    = _load_multi_turn()
    print("Loading S6 — Roleplay handcrafted …")
    s6_rp    = _load_roleplay()
    print("Loading S7 — Refusal-then-help handcrafted …")
    s7_rth   = _load_refusal_help()

    all_records = (ic_tqa + ic_nq + ic_wq +
                   r_gsm + r_arc + r_strat + r_csqa + r_bool + r_wino +
                   gen_mmlu + gen_he + gen_mbpp + gen_hs +
                   ooc +
                   s1_adv + s1_mal + s2_xst + s3_ife + s4_wiki +
                   s5_mt + s6_rp + s7_rth)

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

    # Safety subcategory breakdown
    safety_cat_counts: dict[str, int] = {}
    for r in records:
        sc = r.get("metadata", {}).get("safety_category")
        if sc:
            safety_cat_counts[sc] = safety_cat_counts.get(sc, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Exp3 dataset  →  {output_path}")
    print(f"  Total records : {len(records)}")
    print(f"{'='*60}")
    for split in ["IC", "R", "OOC", "GEN", "S"]:
        print(f"  {split:<5} : {split_counts.get(split, 0)}")
    print(f"  {'─'*45}")
    for k, v in sorted(source_counts.items()):
        print(f"    {k:<35} {v}")
    if safety_cat_counts:
        print(f"  {'─'*45}")
        print("  Safety subcategories:")
        for sc in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
            if sc in safety_cat_counts:
                print(f"    {sc:<5} : {safety_cat_counts[sc]}")
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
