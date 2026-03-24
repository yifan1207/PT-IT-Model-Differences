#!/usr/bin/env python3
"""Build the 1000-prompt Exp6 governance dataset.

Categories:
  GOV-FORMAT    (200): IFEval format-instruction prompts (downloaded from HuggingFace)
  GOV-CONV      (200): Custom conversational Q&A
  GOV-REGISTER  (100): Custom register-sensitivity prompts
  CONTENT-FACT  (200): MMLU + TriviaQA factual questions
  CONTENT-REASON(100): GSM8K math word problems
  SAFETY        (100): AdvBench-style harmful (50) + XSTest-style safe (50) from exp3
  BASELINE-EASY (100): Trivially easy questions (one-word answers, greetings, arithmetic)

Output: data/exp6_dataset.jsonl (1000 JSONL records)

Record schema:
{
  "id": "exp6_{CATEGORY}_{idx:04d}",
  "category": "GOV-FORMAT|GOV-CONV|...",
  "formats": {"A": str, "B": str},
  "expected_answer": str | null,
  "expected_behavior": "comply|refuse|comply_safely|null",
  "format_instruction": str | null,
  "expected_format": str | null,
  "source": str,
  "metadata": {}
}
"""
from __future__ import annotations

import json
import random
from pathlib import Path


# ── Inline custom prompt lists ────────────────────────────────────────────────

_GOV_CONV_PROMPTS = [
    ("What is the capital of France?", "Paris", None),
    ("What is photosynthesis?", None, None),
    ("Compare Python and JavaScript as programming languages.", None, None),
    ("What should I cook for dinner tonight?", None, None),
    ("Explain how the internet works.", None, None),
    ("What are the main causes of climate change?", None, None),
    ("How do I start learning to play guitar?", None, None),
    ("What is the difference between machine learning and AI?", None, None),
    ("What is the best way to study for exams?", None, None),
    ("How does a car engine work?", None, None),
    ("What is inflation and how does it affect me?", None, None),
    ("What are the benefits of regular exercise?", None, None),
    ("How do I write a good cover letter?", None, None),
    ("What is quantum computing?", None, None),
    ("Explain the water cycle.", None, None),
    ("What is the stock market and how does it work?", None, None),
    ("How do vaccines work?", None, None),
    ("What are the main differences between cats and dogs as pets?", None, None),
    ("How does GPS work?", None, None),
    ("What is blockchain technology?", None, None),
    ("What is the difference between a virus and a bacterium?", None, None),
    ("How do I start a small business?", None, None),
    ("What is the greenhouse effect?", None, None),
    ("Explain relativity in simple terms.", None, None),
    ("What are the health benefits of meditation?", None, None),
    ("How does a computer processor work?", None, None),
    ("What is the difference between democracy and republic?", None, None),
    ("How do I improve my writing skills?", None, None),
    ("What causes earthquakes?", None, None),
    ("Explain the human digestive system.", None, None),
    ("What is the difference between RAM and storage?", None, None),
    ("How do I invest my savings?", None, None),
    ("What is machine learning?", None, None),
    ("How do solar panels generate electricity?", None, None),
    ("What is the difference between weather and climate?", None, None),
    ("How does the human brain work?", None, None),
    ("What are the symptoms of depression?", None, None),
    ("How do I learn a new language?", None, None),
    ("What is the difference between a meteor and a meteorite?", None, None),
    ("Explain supply and demand.", None, None),
    ("What is the difference between HTML and CSS?", None, None),
    ("How does the stock market affect the economy?", None, None),
    ("What causes the seasons?", None, None),
    ("How do antibiotics work?", None, None),
    ("What is the difference between a psychologist and a psychiatrist?", None, None),
    ("How does WiFi work?", None, None),
    ("What is the difference between speed and velocity?", None, None),
    ("How do I maintain a healthy diet?", None, None),
    ("What is nuclear energy?", None, None),
    ("How does the electoral college work?", None, None),
    ("What is the difference between debit and credit cards?", None, None),
    ("How do plants grow?", None, None),
    ("What is the difference between astronomy and astrology?", None, None),
    ("How does the immune system work?", None, None),
    ("What is the difference between socialism and capitalism?", None, None),
    ("How do I give a good presentation?", None, None),
    ("What causes thunder and lightning?", None, None),
    ("How does a battery work?", None, None),
    ("What is the difference between a lake and a pond?", None, None),
    ("How do I build good habits?", None, None),
    ("What is genetic engineering?", None, None),
    ("How does the Federal Reserve work?", None, None),
    ("What is the difference between a simile and a metaphor?", None, None),
    ("How do airplanes fly?", None, None),
    ("What is the difference between introvert and extrovert?", None, None),
    ("How do I negotiate a salary?", None, None),
    ("What is CRISPR?", None, None),
    ("How does meditation reduce stress?", None, None),
    ("What is the difference between organic and conventional food?", None, None),
    ("How do I read a scientific paper?", None, None),
    ("What is the Big Bang theory?", None, None),
    ("How does social media affect mental health?", None, None),
    ("What is the difference between a CV and a resume?", None, None),
    ("How do I become a better listener?", None, None),
    ("What is artificial intelligence?", None, None),
    ("How does compound interest work?", None, None),
    ("What is the difference between a hypothesis and a theory?", None, None),
    ("How do I reduce my carbon footprint?", None, None),
    ("What causes allergies?", None, None),
    ("How does encryption work?", None, None),
    ("What is the difference between a skill and a talent?", None, None),
    ("How do I deal with stress at work?", None, None),
    ("What is the difference between a podcast and a radio show?", None, None),
    ("How does the human eye work?", None, None),
    ("What is nanotechnology?", None, None),
    ("How do I build an emergency fund?", None, None),
    ("What is the difference between a patent and a trademark?", None, None),
    ("How does peer pressure affect decision making?", None, None),
    ("What is the difference between a planet and a dwarf planet?", None, None),
    ("How do I start journaling?", None, None),
    ("What is cryptocurrency?", None, None),
    ("How does the nervous system work?", None, None),
    ("What is the difference between coaching and mentoring?", None, None),
    ("How do I become more creative?", None, None),
    ("What causes ocean tides?", None, None),
    ("How does 5G differ from 4G?", None, None),
    ("What is the difference between a democracy and an autocracy?", None, None),
    ("How do I improve my memory?", None, None),
    ("What are the stages of grief?", None, None),
    ("How does a microwave oven work?", None, None),
    ("What is the difference between a cult and a religion?", None, None),
    ("How do I manage my time better?", None, None),
    ("What is photovoltaic energy?", None, None),
    ("How does evolution work?", None, None),
    ("What is the difference between a novel and a novella?", None, None),
    ("How do I handle conflict at work?", None, None),
    ("What is dark matter?", None, None),
    ("How does the legal system work?", None, None),
    ("What is the difference between empathy and sympathy?", None, None),
    ("How do I write a professional email?", None, None),
    ("What are black holes?", None, None),
    ("How does the respiratory system work?", None, None),
    ("What is the difference between a fixed and variable mortgage?", None, None),
    ("How do I become more confident?", None, None),
    ("What is augmented reality?", None, None),
    ("How does the cardiovascular system work?", None, None),
    ("What is the difference between a coup and a revolution?", None, None),
    ("How do I cook rice perfectly?", None, None),
    ("What are stem cells?", None, None),
    ("How does Wi-Fi encryption work?", None, None),
    ("What is the difference between a memoir and an autobiography?", None, None),
    ("How do I ask for a promotion?", None, None),
    ("What is consciousness?", None, None),
    ("How does the kidney filter blood?", None, None),
    ("What is the difference between a hurricane and a typhoon?", None, None),
    ("How do I set SMART goals?", None, None),
    ("What is virtual reality?", None, None),
    ("How does inflation affect savings?", None, None),
    ("What is the difference between a symphony and a concerto?", None, None),
    ("How do I apologize effectively?", None, None),
    ("What is the difference between a comet and an asteroid?", None, None),
    ("How does a nuclear reactor work?", None, None),
    ("What is the placebo effect?", None, None),
    ("How do I make a good first impression?", None, None),
    ("What is the difference between RAM and CPU?", None, None),
    ("How does the lymphatic system work?", None, None),
    ("What is existentialism?", None, None),
    ("How do I become a better public speaker?", None, None),
    ("What is the difference between a fable and a parable?", None, None),
    ("How does sleep affect learning?", None, None),
    ("What is the internet of things?", None, None),
    ("How do I negotiate a car price?", None, None),
    ("What are the main branches of philosophy?", None, None),
    ("How does the endocrine system work?", None, None),
    ("What is the difference between a virus and malware?", None, None),
    ("How do I build a morning routine?", None, None),
    ("What is the difference between a theorem and a law in science?", None, None),
    ("How does osmosis work?", None, None),
    ("What is corporate social responsibility?", None, None),
    ("How do I file my taxes?", None, None),
    ("What is the difference between a myth and a legend?", None, None),
    ("How does chemotherapy work?", None, None),
    ("What is the difference between weather and climate change?", None, None),
    ("How do I start a podcast?", None, None),
    ("What is the Dunning-Kruger effect?", None, None),
    ("How does the stock market work?", None, None),
    ("What is the difference between a republic and a monarchy?", None, None),
    ("How do I get better at chess?", None, None),
    ("What is machine translation?", None, None),
    ("How does a refrigerator work?", None, None),
    ("What is the difference between a psychopath and a sociopath?", None, None),
    ("How do I improve my emotional intelligence?", None, None),
    ("What is the Fermi paradox?", None, None),
    ("How does carbon dating work?", None, None),
    ("What is the difference between a strategy and a tactic?", None, None),
    ("How do I write a business plan?", None, None),
    ("What is the butterfly effect?", None, None),
    ("How does the moon affect tides?", None, None),
    ("What is the difference between a leader and a manager?", None, None),
    ("How do I develop a growth mindset?", None, None),
    ("What is quantum entanglement?", None, None),
    ("How does photosynthesis work?", None, None),
    ("What is the difference between empathy and compassion?", None, None),
    ("How do I get better sleep?", None, None),
    ("What is the Drake equation?", None, None),
    ("How does CRISPR gene editing work?", None, None),
    ("What is cognitive behavioral therapy?", None, None),
    ("How do I communicate better with my team?", None, None),
    ("What is the difference between a star and a planet?", None, None),
    ("How does microwave heating work differently from convection?", None, None),
    ("What is the prisoner's dilemma?", None, None),
][:200]


_GOV_REGISTER_PROMPTS = [
    "Explain machine learning",
    "What are the benefits of exercise",
    "Describe the water cycle",
    "Explain inflation",
    "What is democracy",
    "Describe how vaccines work",
    "Explain the concept of gravity",
    "What is the human digestive system",
    "Explain climate change",
    "Describe how the internet works",
    "What is the role of the liver",
    "Explain how banks work",
    "Describe the process of photosynthesis",
    "What is artificial intelligence",
    "Explain evolution by natural selection",
    "Describe how computers store data",
    "What is the immune system",
    "Explain supply and demand",
    "Describe the water treatment process",
    "What is cognitive dissonance",
    "Explain recursion in programming",
    "Describe the formation of stars",
    "What is the role of mitochondria",
    "Explain opportunity cost in economics",
    "Describe the process of digestion",
    "What is the purpose of a central bank",
    "Explain how DNA replication works",
    "Describe the layers of the atmosphere",
    "What is the role of sleep in memory consolidation",
    "Explain the prisoner dilemma",
    "Describe plate tectonics",
    "What is the role of insulin in the body",
    "Explain how encryption protects data",
    "Describe the nitrogen cycle",
    "What is neuroplasticity",
    "Explain how search engines rank pages",
    "Describe the lifecycle of a star",
    "What is the role of the amygdala",
    "Explain how HTTPS works",
    "Describe the carbon cycle",
    "What is confirmation bias",
    "Explain how machine translation works",
    "Describe the structure of DNA",
    "What is the role of dopamine",
    "Explain how recommendation algorithms work",
    "Describe the human respiratory system",
    "What is the placebo effect",
    "Explain how neural networks learn",
    "Describe the phases of the moon",
    "What is the Coriolis effect",
    "Explain how mRNA vaccines work",
    "Describe the endocrine system",
    "What is cognitive load",
    "Explain how GPS calculates position",
    "Describe the formation of mountains",
    "What is the role of the prefrontal cortex",
    "Explain how transformers work in AI",
    "Describe the process of fermentation",
    "What is sunk cost fallacy",
    "Explain how antibiotics work",
    "Describe the structure of a cell",
    "What is the role of cortisol in stress",
    "Explain how the TCP/IP protocol works",
    "Describe the carbon sequestration process",
    "What is working memory",
    "Explain how a compiler works",
    "Describe the human circulatory system",
    "What is the availability heuristic",
    "Explain how light interacts with matter",
    "Describe the endosymbiotic theory",
    "What is the role of the hippocampus",
    "Explain how nuclear fission works",
    "Describe the water table and aquifer system",
    "What is loss aversion",
    "Explain how mRNA is transcribed from DNA",
    "Describe the layers of the earth",
    "What is the role of the vagus nerve",
    "Explain how cache memory works",
    "Describe the life cycle of a cell",
    "What is the tragedy of the commons",
    "Explain how deep learning differs from traditional programming",
    "Describe the ecosystem food web",
    "What is habituation in neuroscience",
    "Explain how a hash function works",
    "Describe the structure of the atom",
    "What is the role of serotonin",
    "Explain how the CRISPR mechanism identifies target DNA",
    "Describe the human lymphatic system",
    "What is motivated reasoning",
    "Explain how a relational database stores data",
    "Describe the formation of tornadoes",
    "What is the basal ganglia",
    "Explain how the compiler handles memory allocation",
    "Describe the immune response to infection",
    "What is survivorship bias",
    "Explain how quantum superposition works",
    "Describe the human nervous system hierarchy",
    "What is the role of telomeres",
    "Explain how electrical circuits work",
][:100]


_BASELINE_EASY_PROMPTS = [
    ("What is 2 + 2?", "4"),
    ("Hello!", "Hello"),
    ("What color is the sky?", "blue"),
    ("What is 5 times 5?", "25"),
    ("What is the opposite of hot?", "cold"),
    ("What language do people speak in France?", "French"),
    ("What is 10 minus 3?", "7"),
    ("What is the first letter of the alphabet?", "A"),
    ("What day comes after Monday?", "Tuesday"),
    ("How many days are in a week?", "7"),
    ("What is 3 + 7?", "10"),
    ("What planet do we live on?", "Earth"),
    ("What is the color of grass?", "green"),
    ("What is the largest ocean?", "Pacific"),
    ("What is 100 divided by 10?", "10"),
    ("What is the opposite of up?", "down"),
    ("How many months are in a year?", "12"),
    ("What is 8 times 8?", "64"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What is the color of the sun?", "yellow"),
    ("What is 15 + 15?", "30"),
    ("What is the capital of the United States?", "Washington"),
    ("What is the square root of 4?", "2"),
    ("How many sides does a triangle have?", "3"),
    ("What comes after the number 9?", "10"),
    ("What is 7 times 3?", "21"),
    ("What season comes after winter?", "spring"),
    ("How many hours in a day?", "24"),
    ("What is 50 divided by 5?", "10"),
    ("What is the color of an apple?", "red"),
    ("What is 6 + 9?", "15"),
    ("How many fingers do humans typically have?", "10"),
    ("What is the opposite of black?", "white"),
    ("What is 4 times 4?", "16"),
    ("How many cents in a dollar?", "100"),
    ("What is the letter after Z?", None),
    ("What is 20 + 20?", "40"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is the opposite of big?", "small"),
    ("How many legs does a dog have?", "4"),
    ("What is 9 minus 4?", "5"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What color do you get by mixing blue and yellow?", "green"),
    ("How many wheels does a car typically have?", "4"),
    ("What is the first number after 0?", "1"),
    ("What is 12 times 12?", "144"),
    ("What shape is a football?", "oval"),
    ("How many seconds in a minute?", "60"),
    ("What is 1000 minus 1?", "999"),
    ("What is the color of a banana?", "yellow"),
    ("What is the opposite of left?", "right"),
    ("How many letters in the English alphabet?", "26"),
    ("What is 3 squared?", "9"),
    ("What is the nearest star to Earth?", "Sun"),
    ("How many sides does a square have?", "4"),
    ("What is 7 + 8?", "15"),
    ("What is the opposite of day?", "night"),
    ("How many wheels does a bicycle have?", "2"),
    ("What is 11 times 11?", "121"),
    ("What is the color of the ocean?", "blue"),
    ("How many minutes in an hour?", "60"),
    ("What is 25 + 25?", "50"),
    ("What comes before the number 1?", "0"),
    ("How many sides does a pentagon have?", "5"),
    ("What is 2 times 2?", "4"),
    ("What is the color of snow?", "white"),
    ("How many legs does a spider have?", "8"),
    ("What is 100 + 100?", "200"),
    ("What is the opposite of open?", "closed"),
    ("How many days in a leap year?", "366"),
    ("What is 6 times 7?", "42"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What is the opposite of fast?", "slow"),
    ("How many months have 30 days?", "4"),
    ("What is 16 divided by 4?", "4"),
    ("What is the color of a stop sign?", "red"),
    ("How many weeks in a year?", "52"),
    ("What is 9 times 9?", "81"),
    ("What is the largest continent?", "Asia"),
    ("What is the opposite of ancient?", "modern"),
    ("How many vowels are in the English alphabet?", "5"),
    ("What is 500 divided by 10?", "50"),
    ("What shape has no corners?", "circle"),
    ("What is 13 + 13?", "26"),
    ("What is the color of a lime?", "green"),
    ("How many sides does a hexagon have?", "6"),
    ("What is 3 times 8?", "24"),
    ("What is the opposite of empty?", "full"),
    ("How many states are in the United States?", "50"),
    ("What is 144 divided by 12?", "12"),
    ("What is the color of a strawberry?", "red"),
    ("What is 2 cubed?", "8"),
    ("What is the opposite of north?", "south"),
    ("How many sides does an octagon have?", "8"),
    ("What is the first planet from the sun?", "Mercury"),
    ("What is 200 + 300?", "500"),
    ("What is the color of the moon?", "gray"),
    ("How many ounces in a pound?", "16"),
    ("What is 5 + 5 + 5?", "15"),
][:100]


# ── HuggingFace dataset loaders ───────────────────────────────────────────────

def _load_ifeval(n: int = 200) -> list[dict]:
    """Load IFEval prompts and extract format instructions."""
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("google/IFEval", split="train", trust_remote_code=True)
    records = []
    seen = set()
    for row in ds:
        prompt = row["prompt"]
        if prompt in seen or len(records) >= n:
            continue
        seen.add(prompt)
        # Extract first instruction type for expected_format
        instruction_ids = row.get("instruction_id_list", [])
        fmt = "other"
        for iid in instruction_ids:
            if "bullet" in iid:
                fmt = "bullet_list"; break
            if "numbered" in iid or "number" in iid:
                fmt = "numbered_list"; break
            if "json" in iid:
                fmt = "json"; break
            if "markdown" in iid:
                fmt = "markdown"; break
        records.append({
            "question": prompt,
            "format_instruction": prompt,
            "expected_format": fmt,
            "source": "IFEval",
        })
    return records[:n]


def _load_mmlu(n: int = 150) -> list[dict]:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    rng = random.Random(42)
    rows = list(ds)
    rng.shuffle(rows)
    records = []
    for row in rows[:n]:
        choices = row["choices"]
        letters = "ABCD"
        q = row["question"] + "\n" + "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
        ans = letters[row["answer"]]
        records.append({
            "question": q, "expected_answer": ans,
            "answer_aliases": [ans.lower(), f"({ans})", f"({ans.lower()})"],
            "source": "MMLU",
        })
    return records


def _load_triviaqa(n: int = 50) -> list[dict]:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("trivia_qa", "rc", split="validation", trust_remote_code=True)
    rng = random.Random(42)
    rows = list(ds)
    rng.shuffle(rows)
    records = []
    for row in rows[:n]:
        answer = row["answer"]["value"]
        aliases = row["answer"].get("aliases", [])[:5]
        records.append({
            "question": row["question"], "expected_answer": answer,
            "answer_aliases": aliases, "source": "TriviaQA",
        })
    return records


def _load_gsm8k(n: int = 100) -> list[dict]:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    rng = random.Random(42)
    rows = list(ds)
    rng.shuffle(rows)
    records = []
    for row in rows[:n]:
        # Extract final answer (after "####")
        answer_raw = row["answer"]
        if "####" in answer_raw:
            answer = answer_raw.split("####")[-1].strip()
        else:
            answer = answer_raw.strip()
        records.append({
            "question": row["question"], "expected_answer": answer,
            "answer_aliases": [answer], "source": "GSM8K",
        })
    return records


def _load_safety_from_exp3(n_harmful: int = 50, n_safe: int = 50) -> list[dict]:
    """Load safety records from exp3_dataset.jsonl (split="A")."""
    exp3_path = Path("data/exp3_dataset.jsonl")
    all_records = []
    with open(exp3_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("split") == "A":
                all_records.append(rec)

    rng = random.Random(42)
    rng.shuffle(all_records)

    harmful = [r for r in all_records if r.get("metadata", {}).get("expected_behavior") == "refuse"]
    safe = [r for r in all_records if r.get("metadata", {}).get("expected_behavior") in ("comply", "comply_safely")]

    records = []
    for r in (harmful[:n_harmful] + safe[:n_safe]):
        records.append({
            "question": r["formats"]["B"],
            "expected_answer": None,
            "expected_behavior": r.get("metadata", {}).get("expected_behavior", "comply"),
            "source": r.get("source", "exp3"),
            "original_id": r.get("id", ""),
        })
    return records


# ── Record formatters ─────────────────────────────────────────────────────────

def _make_record(idx: int, category: str, data: dict) -> dict:
    q = data.get("question", "")
    q_b = data.get("format_instruction", q)

    # Format A: natural (no format enforcement)
    format_a = q

    # Format B: structured prompt (confound-controlled)
    if category == "GOV-FORMAT":
        format_b = q_b
    elif category == "GOV-CONV":
        format_b = f"Question: {q}\nAnswer:"
    elif category == "GOV-REGISTER":
        format_b = f"Explain: {q}\nResponse:"
    elif category == "CONTENT-FACT":
        format_b = f"Question: {q}\nAnswer:"
    elif category == "CONTENT-REASON":
        format_b = f"Question: {q}\nAnswer:"
    elif category == "SAFETY":
        format_b = q_b  # use original format from exp3
    elif category == "BASELINE-EASY":
        format_b = f"Question: {q}\nAnswer:"
    else:
        format_b = f"Question: {q}\nAnswer:"

    return {
        "id": f"exp6_{category}_{idx:04d}",
        "category": category,
        "formats": {"A": format_a, "B": format_b},
        "expected_answer": data.get("expected_answer"),
        "answer_aliases": data.get("answer_aliases", []),
        "expected_behavior": data.get("expected_behavior", "comply"),
        "format_instruction": data.get("format_instruction"),
        "expected_format": data.get("expected_format"),
        "source": data.get("source", "custom"),
        "metadata": {
            "original_id": data.get("original_id", ""),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Build exp6 governance dataset.")
    p.add_argument("--output", default="data/exp6_dataset.jsonl")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip HuggingFace downloads; use only inline/exp3 data")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_records = []
    idx = 0

    # ── GOV-FORMAT (200): IFEval ──────────────────────────────────────────
    print("Building GOV-FORMAT (IFEval)...", flush=True)
    if args.skip_download:
        ifeval_data = [{"question": f"Write your response as a numbered list. {q}", "expected_format": "numbered_list", "source": "custom_fallback"}
                       for q, _, _ in _GOV_CONV_PROMPTS[:200]]
    else:
        ifeval_data = _load_ifeval(200)
    for d in ifeval_data:
        all_records.append(_make_record(idx, "GOV-FORMAT", d))
        idx += 1
    print(f"  {len(ifeval_data)} GOV-FORMAT records", flush=True)

    # ── GOV-CONV (200): Custom ────────────────────────────────────────────
    print("Building GOV-CONV (custom)...", flush=True)
    for q, ans, fmt in _GOV_CONV_PROMPTS:
        all_records.append(_make_record(idx, "GOV-CONV", {
            "question": q, "expected_answer": ans, "source": "custom",
        }))
        idx += 1
    print(f"  {len(_GOV_CONV_PROMPTS)} GOV-CONV records", flush=True)

    # ── GOV-REGISTER (100): Custom ────────────────────────────────────────
    print("Building GOV-REGISTER (custom)...", flush=True)
    for topic in _GOV_REGISTER_PROMPTS:
        all_records.append(_make_record(idx, "GOV-REGISTER", {
            "question": topic, "source": "custom",
        }))
        idx += 1
    print(f"  {len(_GOV_REGISTER_PROMPTS)} GOV-REGISTER records", flush=True)

    # ── CONTENT-FACT (200): MMLU + TriviaQA ──────────────────────────────
    print("Building CONTENT-FACT (MMLU + TriviaQA)...", flush=True)
    if args.skip_download:
        fact_data = [{"question": q, "expected_answer": a, "source": "inline"}
                     for q, a in [("What is the capital of France?", "Paris"),
                                   ("What year did WWI end?", "1918")] * 100][:200]
    else:
        fact_data = _load_mmlu(150) + _load_triviaqa(50)
    random.Random(42).shuffle(fact_data)
    for d in fact_data[:200]:
        all_records.append(_make_record(idx, "CONTENT-FACT", d))
        idx += 1
    print(f"  {min(len(fact_data), 200)} CONTENT-FACT records", flush=True)

    # ── CONTENT-REASON (100): GSM8K ───────────────────────────────────────
    print("Building CONTENT-REASON (GSM8K)...", flush=True)
    if args.skip_download:
        reason_data = [{"question": "What is 15 * 7?", "expected_answer": "105", "source": "inline"}] * 100
    else:
        reason_data = _load_gsm8k(100)
    for d in reason_data[:100]:
        all_records.append(_make_record(idx, "CONTENT-REASON", d))
        idx += 1
    print(f"  {min(len(reason_data), 100)} CONTENT-REASON records", flush=True)

    # ── SAFETY (100): exp3 split A ────────────────────────────────────────
    print("Building SAFETY (exp3 split A)...", flush=True)
    safety_data = _load_safety_from_exp3(50, 50)
    for d in safety_data[:100]:
        all_records.append(_make_record(idx, "SAFETY", d))
        idx += 1
    print(f"  {len(safety_data[:100])} SAFETY records", flush=True)

    # ── BASELINE-EASY (100): Inline ───────────────────────────────────────
    print("Building BASELINE-EASY (inline)...", flush=True)
    for q, ans in _BASELINE_EASY_PROMPTS:
        all_records.append(_make_record(idx, "BASELINE-EASY", {
            "question": q, "expected_answer": ans, "source": "custom",
        }))
        idx += 1
    print(f"  {len(_BASELINE_EASY_PROMPTS)} BASELINE-EASY records", flush=True)

    # ── Write output ──────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_records)} records → {out_path}")

    # Category summary
    from collections import Counter
    cats = Counter(r["category"] for r in all_records)
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
