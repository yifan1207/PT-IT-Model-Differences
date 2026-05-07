#!/usr/bin/env python3
"""Build the anonymous NeurIPS PDF and compact reviewer supplement.

The script intentionally builds from ``paper_draft/PAPER_DRAFT_v25.md`` rather
than the older hand-generated LaTeX directory. It creates two artifacts:

* ``paper_draft/submission/neurips2026_main.pdf``
* ``paper_draft/submission/neurips2026_supplement.zip``
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from pypdf import PdfReader


REPO = Path(__file__).resolve().parents[2]
PAPER_MD = REPO / "paper_draft/PAPER_DRAFT_v25.md"
SUBMISSION_DIR = REPO / "paper_draft/submission"
BUILD_DIR = SUBMISSION_DIR / "build"
SUPP_STAGE = SUBMISSION_DIR / "supplement_staging"
PDF_OUT = SUBMISSION_DIR / "neurips2026_main.pdf"
SUPP_ZIP = SUBMISSION_DIR / "neurips2026_supplement.zip"
NEURIPS_ZIP_DEFAULT = Path.home() / "Downloads/Formatting_Instructions_For_NeurIPS_2026.zip"

TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

LEAK_PATTERNS = [
    "Yifan",
    "/Users/",
    "/mnt/storage/",
    "/workspace/structral-semantic-features",
    "/home/",
    "github.com/yifan",
    "pt-vs-it-results",
    "gs://",
    "RunPod",
    "runpod",
]

SECRET_REGEXES = [
    re.compile(r"(?i)(openai|hf|wandb)[_-]?api[_-]?key\s*=\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)\b(hf|ghp|sk)-[A-Za-z0-9_\-]{20,}\b"),
]

SUPPLEMENT_JSON_KEY_EXCLUDE_SUBSTRINGS = (
    "part_a_mlp_kl",
    "convergence_gap",
    "endpoint_matched",
    "late_localization",
)

SUPPLEMENT_JSON_KEYS_EXCLUDE = {
    "requests",
    "responses",
    "raw_requests",
    "raw_responses",
}

EXP50_RAW_RUN_PREFIX = (
    "results/exp50_llm_judge_behavior_bridge/"
    "exp50_openai_judge_requests_20260504_233946/analysis_gpt52_sync"
)
EXP50_SUPPLEMENT_PREFIX = "results/exp50_llm_judge_behavior_bridge/analysis_summary"

STATIC_SUPPLEMENT_FILES = [
    "scripts/reproduce/check_paper_claims.py",
    "scripts/reproduce/reproduce_minimal.sh",
    "scripts/reproduce/check_minimal_shard.py",
    "scripts/plot/plot_first_divergence_schematic_examples.py",
]

STATIC_SUPPLEMENT_DIRS = [
    "src/poc/exp42_terminal_feature_upstream_conditioning",
    "src/poc/exp43_feature_rescue_handoff",
    "src/poc/exp44_middle_terminal_feature_handoff",
    "src/poc/exp45_behavioral_bridge",
    "src/poc/exp46_tulu_fixed_support_stage_sweep",
    "src/poc/exp48_static_chimera_sequence_validation",
    "src/poc/exp49_constrained_continuation_bridge",
    "src/poc/exp50_llm_judge_behavior_bridge",
    "src/poc/exp51_native_history_crosspatch",
    "src/poc/exp52_forced_token_consequence_bridge",
    "src/poc/exp53_controlled_domain_finetunes",
]

SUPPLEMENT_RESULT_GLOBS = [
    "results/paper_synthesis/**",
    "results/exp42_terminal_feature_upstream_conditioning/exp42_full_4fam_h100x8_20260503_155212/analysis/*.json",
    "results/exp42_terminal_feature_upstream_conditioning/exp42_full_4fam_h100x8_20260503_155212/analysis/*.csv",
    "results/exp42_terminal_feature_upstream_conditioning/exp42_full_4fam_h100x8_20260503_155212/analysis/*.md",
    "results/exp42_terminal_feature_upstream_conditioning/exp42_full_4fam_h100x8_20260503_155212/analysis/*.png",
    "results/exp43_feature_rescue_handoff/exp43_full_h100x8_clean_20260503_182947/analysis/*.json",
    "results/exp43_feature_rescue_handoff/exp43_full_h100x8_clean_20260503_182947/analysis/*.csv",
    "results/exp43_feature_rescue_handoff/exp43_full_h100x8_clean_20260503_182947/analysis/*.md",
    "results/exp43_feature_rescue_handoff/exp43_full_h100x8_clean_20260503_182947/analysis/*.png",
    "results/exp44_middle_terminal_feature_handoff/exp44_primary_lmq_a100_20260503_combined/analysis/*.json",
    "results/exp44_middle_terminal_feature_handoff/exp44_primary_lmq_a100_20260503_combined/analysis/*.csv",
    "results/exp44_middle_terminal_feature_handoff/exp44_primary_lmq_a100_20260503_combined/analysis/*.md",
    "results/exp44_middle_terminal_feature_handoff/exp44_primary_lmq_a100_20260503_combined/analysis/*.png",
    "results/exp46_tulu_fixed_support_stage_sweep/exp46_full_a100x8_localdisk_20260504_103624/analysis/*.json",
    "results/exp46_tulu_fixed_support_stage_sweep/exp46_full_a100x8_localdisk_20260504_103624/analysis/*.csv",
    "results/exp46_tulu_fixed_support_stage_sweep/exp46_full_a100x8_localdisk_20260504_103624/analysis/*.png",
    "results/exp46_tulu_fixed_support_stage_sweep/exp46_full_base_to_S_a100x8_localdisk_20260504_104959/analysis/effects.csv",
    "results/exp46_tulu_fixed_support_stage_sweep/exp46_full_base_to_D_a100x8_localdisk_20260504_105605/analysis/effects.csv",
    "results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24/analysis/*.json",
    "results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24/analysis/*.csv",
    "results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24/analysis/*.md",
    "results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24/analysis/*.png",
    "results/exp48_static_chimera_sequence_validation/exp48_static_chimera_sequence_validation_20260504_1349_a100x16/analysis/*.json",
    "results/exp48_static_chimera_sequence_validation/exp48_static_chimera_sequence_validation_20260504_1349_a100x16/analysis/*.csv",
    "results/exp48_static_chimera_sequence_validation/exp48_static_chimera_sequence_validation_20260504_1349_a100x16/analysis/*.md",
    "results/exp48_static_chimera_sequence_validation/exp48_static_chimera_sequence_validation_20260504_1349_a100x16/analysis/*.png",
    "results/exp49_constrained_continuation_bridge/exp49_full_20260504_223652_a100x8/analysis/summary.json",
    "results/exp49_constrained_continuation_bridge/exp49_full_20260504_223652_a100x8/analysis/aggregate_effects.csv",
    "results/exp49_constrained_continuation_bridge/exp49_full_20260504_223652_a100x8/analysis/paper_claims_exp49.md",
    "results/exp49_constrained_continuation_bridge/exp49_full_20260504_223652_a100x8/analysis/plots/*.png",
    "results/exp50_llm_judge_behavior_bridge/exp50_openai_judge_requests_20260504_233946/analysis_gpt52_sync/judge_summary.json",
    "results/exp50_llm_judge_behavior_bridge/exp50_openai_judge_requests_20260504_233946/analysis_gpt52_sync/behavioral_interactions.csv",
    "results/exp50_llm_judge_behavior_bridge/exp50_openai_judge_requests_20260504_233946/analysis_gpt52_sync/pairwise_winrates.csv",
    "results/exp50_llm_judge_behavior_bridge/exp50_openai_judge_requests_20260504_233946/analysis_gpt52_sync/order_bias_audit.csv",
    "results/exp50_llm_judge_behavior_bridge/exp50_openai_judge_requests_20260504_233946/analysis_gpt52_sync/paper_claims_exp50.md",
    "results/exp50_llm_judge_behavior_bridge/exp50_openai_judge_requests_20260504_233946/analysis_gpt52_sync/*.png",
    "results/exp51_native_history_crosspatch/exp51_full_rpro6000x4_20260505_235900/analysis/*.json",
    "results/exp51_native_history_crosspatch/exp51_full_rpro6000x4_20260505_235900/analysis/*.csv",
    "results/exp51_native_history_crosspatch/exp51_full_rpro6000x4_20260505_235900/analysis/*.md",
    "results/exp51_native_history_crosspatch/exp51_full_rpro6000x4_20260505_235900/analysis/*.png",
    "results/exp52_forced_token_consequence_bridge/exp52_full_combined_20260506_0127_a100x4/analysis/*.json",
    "results/exp52_forced_token_consequence_bridge/exp52_full_combined_20260506_0127_a100x4/analysis/*.csv",
    "results/exp52_forced_token_consequence_bridge/exp52_full_combined_20260506_0127_a100x4/analysis/*.md",
    "results/exp52_forced_token_consequence_bridge/exp52_full_combined_20260506_0127_a100x4/analysis/plots/*.png",
    "results/exp53_controlled_domain_finetunes/exp53_full_h200x2_20260506_0234/analysis/*.json",
    "results/exp53_controlled_domain_finetunes/exp53_full_h200x2_20260506_0234/analysis/*.csv",
    "results/exp53_controlled_domain_finetunes/exp53_full_h200x2_20260506_0234/analysis/*.md",
    "results/exp53_controlled_domain_finetunes/exp53_full_h200x2_20260506_0234/analysis/*.png",
]

SUPPLEMENT_EXCLUDE_PATTERNS = (
    r"exp0?9",
    r"exp1[149]",
    r"exp2[2]",
    r"exp55",
    r"conv" r"ergence[_-]gap",
    r"late[_-]localization",
    r"L2_mean" r"_kl",
    r"part_a_mlp_kl",
    r"exp24_32b_summary\.csv",
)


def run(cmd: list[str], *, cwd: Path = REPO, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=True,
    )


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def sanitize_path(relpath: str) -> str:
    relpath = relpath.replace(EXP50_RAW_RUN_PREFIX, EXP50_SUPPLEMENT_PREFIX)
    relpath = relpath.replace("runpod", "remotejob")
    relpath = relpath.replace("RunPod", "RemoteJob")
    relpath = relpath.replace("pt-vs-it-results", "anonymous-artifact-store")
    relpath = relpath.replace("yifan1207", "anonymous")
    return relpath


def sanitize_text(text: str) -> str:
    replacements = {
        EXP50_RAW_RUN_PREFIX: EXP50_SUPPLEMENT_PREFIX,
        "results/exp50_llm_judge_behavior_bridge/exp50_openai_judge_requests_20260504_233946": "results/exp50_llm_judge_behavior_bridge/analysis_summary",
        "exp50_openai_judge_requests": "exp50_judge_batch",
        "llm_judge_requests.jsonl": "llm_judge_inputs.jsonl",
        "judge_requests.jsonl": "judge_inputs.jsonl",
        "judge_responses_gpt52_sync.jsonl.gz": "judge_outputs_gpt52_sync.jsonl.gz",
        "judge_responses.jsonl.gz": "judge_outputs.jsonl.gz",
        "/Users/Yifan/Research/structral-semantic-features/": "",
        "/Users/Yifan/Research/structral-semantic-features": ".",
        "/Users/Yifan/": "./",
        "/mnt/storage/": "anonymous-remote-storage/",
        "/workspace/structral-semantic-features/": "",
        "/workspace/structral-semantic-features": ".",
        "Yifan": "Anonymous",
        "github.com/yifan1207/PT-IT-Model-Differences": "anonymous-code-archive.invalid/reviewer-artifacts",
        "github.com/yifan1207": "anonymous-code-archive.invalid",
        "pt-vs-it-results": "anonymous-artifact-store",
        "gs://": "anonymous-object-store://",
        "RunPod": "RemoteJob",
        "runpod": "remotejob",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"(?i)(openai|hf|wandb)[_-]?api[_-]?key\\s*=\\s*['\\\"][^'\\\"]+['\\\"]", r"\\1_API_KEY=<redacted>", text)
    text = re.sub(r"(?i)(hf|ghp|sk)-[A-Za-z0-9_\\-]{20,}", "<redacted-token>", text)
    return text


def strip_nonpaper_json_keys(obj: object) -> object:
    if isinstance(obj, dict):
        clean: dict[str, object] = {}
        for key, value in obj.items():
            lowered = key.lower()
            if lowered in SUPPLEMENT_JSON_KEYS_EXCLUDE:
                continue
            if any(substr in lowered for substr in SUPPLEMENT_JSON_KEY_EXCLUDE_SUBSTRINGS):
                continue
            clean[key] = strip_nonpaper_json_keys(value)
        return clean
    if isinstance(obj, list):
        return [strip_nonpaper_json_keys(value) for value in obj]
    return obj


def sanitize_json_text(text: str) -> str:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(strip_nonpaper_json_keys(obj), indent=2, sort_keys=True) + "\n"


def strip_submission_only_comments(text: str) -> str:
    return re.sub(
        r"\n?<!-- Detailed per-file path table retained.*?-->\n?",
        "\n",
        text,
        flags=re.DOTALL,
    )


def tex_safe_unicode(text: str) -> str:
    """Keep generated TeX compatible with the NeurIPS pdfLaTeX-style template."""
    replacements = {
        "–": "--",
        "—": "---",
        "−": "-",
        "×": r"$\times$",
        "≥": r"$\geq$",
        "≤": r"$\leq$",
        "’": "'",
        "‘": "'",
        "“": "``",
        "”": "''",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def copy_sanitized(src: Path, dest_root: Path, dest_rel: str | None = None) -> None:
    if not src.exists() or src.is_dir():
        return
    if "__pycache__" in src.parts or src.suffix in {".pyc", ".pyo"}:
        return
    source_rel = rel(src)
    if any(re.search(pattern, source_rel) for pattern in SUPPLEMENT_EXCLUDE_PATTERNS):
        return
    dest_rel = sanitize_path(dest_rel or source_rel)
    dest = dest_root / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_SUFFIXES:
        try:
            text = src.read_text()
        except UnicodeDecodeError:
            shutil.copy2(src, dest)
        else:
            text = strip_submission_only_comments(sanitize_text(text))
            if src.suffix.lower() == ".json":
                text = sanitize_json_text(text)
            dest.write_text(text)
    else:
        shutil.copy2(src, dest)


def copy_tree_sanitized(src_dir: Path, dest_root: Path) -> None:
    if not src_dir.exists():
        return
    for path in src_dir.rglob("*"):
        if path.is_file():
            copy_sanitized(path, dest_root)


def extract_template(neurips_zip: Path, build_dir: Path) -> None:
    with zipfile.ZipFile(neurips_zip) as zf:
        official_style = zf.read("neurips_2026.sty")
        (build_dir / "neurips_2026.sty").write_bytes(official_style)
        if (build_dir / "neurips_2026.sty").read_bytes() != official_style:
            raise RuntimeError("Generated neurips_2026.sty differs from the official template zip")
        (build_dir / "checklist.tex").write_bytes(zf.read("checklist.tex"))


def split_markdown(md: str) -> tuple[str, str, str]:
    lines = md.splitlines()
    title = lines[0].removeprefix("# ").strip()
    abstract_idx = lines.index("## Abstract")
    after_abs = abstract_idx + 1
    while after_abs < len(lines) and not lines[after_abs].strip():
        after_abs += 1
    end_abs = next(i for i in range(after_abs, len(lines)) if lines[i].strip() == "---")
    abstract = " ".join(line.strip() for line in lines[after_abs:end_abs] if line.strip())
    body = "\n".join(lines[end_abs + 1 :]).strip() + "\n"
    return tex_safe_unicode(title), tex_safe_unicode(abstract), body


def strip_heading_number(text: str) -> str:
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text)
    text = re.sub(r"^[A-I]\.\d+\s+", "", text)
    text = re.sub(r"^Appendix\s+[A-I]:\s+", "", text)
    return text


def preprocess_body_markdown(body: str, figures_dir: Path) -> str:
    out_lines: list[str] = []
    appendix_started = False
    for line in body.splitlines():
        if line.strip() == "---":
            continue
        if line.startswith("## References"):
            out_lines.extend(["", "\\clearpage", "\\section*{References}", ""])
            continue
        if line.startswith("## Appendix Roadmap"):
            out_lines.extend(["", "\\clearpage", "\\section*{Appendix Roadmap}", ""])
            continue
        if re.match(r"^## Appendix [A-I]:", line):
            if not appendix_started:
                out_lines.extend(["", "\\appendix", ""])
                appendix_started = True
            text = strip_heading_number(line[3:].strip())
            out_lines.append(f"## {text}")
            continue
        if re.match(r"^#{2,4} ", line):
            hashes, text = line.split(" ", 1)
            text = strip_heading_number(text.strip())
            out_lines.append(f"{hashes} {text}")
            continue
        out_lines.append(line)

    processed = strip_submission_only_comments("\n".join(out_lines) + "\n")
    return tex_safe_unicode(sanitize_text(rewrite_image_paths(processed, figures_dir)))


def rewrite_image_paths(md: str, figures_dir: Path) -> str:
    pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    seen: dict[str, str] = {}

    def caption_tex(text: str) -> str:
        text = re.sub(r"^\s*Figure\s+\d+\s*:\s*", "", text)
        text = tex_safe_unicode(sanitize_text(text))
        text = text.replace("\\", r"\textbackslash{}")
        for src, dst in {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }.items():
            text = text.replace(src, dst)
        text = text.replace("~", r"\textasciitilde{}")
        text = text.replace("^", r"\textasciicircum{}")
        return text

    def figure_label(target: str) -> str:
        stem = Path(target).stem
        stem = re.sub(r"_[0-9a-f]{8}$", "", stem)
        stem = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
        return f"fig:{stem}"

    def repl(match: re.Match[str]) -> str:
        alt, raw_path = match.group(1), match.group(2)
        source = (PAPER_MD.parent / raw_path).resolve()
        if source.name == "first_divergence_schematic_examples.png":
            pdf_source = source.with_suffix(".pdf")
            if pdf_source.exists():
                source = pdf_source
        if not source.exists():
            raise FileNotFoundError(source)
        if str(source) not in seen:
            digest = hashlib.sha1(str(source).encode()).hexdigest()[:8]
            target = f"figures/{source.stem}_{digest}{source.suffix}"
            shutil.copy2(source, figures_dir / Path(target).name)
            seen[str(source)] = target
        target = seen[str(source)]
        if "first_divergence_schematic_examples" in target:
            opts = r"width=\linewidth,height=0.39\textheight,keepaspectratio"
        elif "selection_baselines" in target:
            opts = r"width=0.94\linewidth,height=0.24\textheight,keepaspectratio"
        elif "exp20_exp21_handoff" in target or "terminal_crosscoder" in target:
            opts = r"width=0.94\linewidth,height=0.25\textheight,keepaspectratio"
        else:
            opts = r"width=0.96\linewidth,height=0.27\textheight,keepaspectratio"
        caption = caption_tex(alt) if alt.strip() else "Paper figure."
        return "\n".join(
            [
                r"\begin{figure}[t]",
                r"\centering",
                rf"\includegraphics[{opts}]{{{target}}}",
                rf"\caption{{{caption}}}",
                rf"\label{{{figure_label(target)}}}",
                r"\end{figure}",
            ]
        )

    return pattern.sub(repl, md)


def convert_body_to_latex(md_path: Path, tex_path: Path) -> None:
    run(
        [
            "pandoc",
            str(md_path),
            "--from",
            "markdown+pipe_tables+tex_math_dollars+raw_tex+backtick_code_blocks+fenced_code_blocks",
            "--to",
            "latex",
            "--standalone=false",
            "--wrap=none",
            "--top-level-division=section",
            "--shift-heading-level-by=-1",
            "-o",
            str(tex_path),
        ],
        cwd=BUILD_DIR,
    )
    tex = tex_path.read_text()
    tex = tex.replace("\\begin{quote}", "\\begin{quote}\\small")
    math_replacements = {
        r"\texttt{t\_PT}": r"$t_{\mathrm{PT}}$",
        r"\texttt{t\_IT}": r"$t_{\mathrm{IT}}$",
        r"\texttt{t\_Base}": r"$t_{\mathrm{Base}}$",
        r"\texttt{t\_Final}": r"$t_{\mathrm{Final}}$",
        r"\texttt{t\_RLVR}": r"$t_{\mathrm{RLVR}}$",
        r"\texttt{U\_PT}": r"$U_{\mathrm{PT}}$",
        r"\texttt{U\_IT}": r"$U_{\mathrm{IT}}$",
        r"\texttt{L\_PT}": r"$L_{\mathrm{PT}}$",
        r"\texttt{L\_IT}": r"$L_{\mathrm{IT}}$",
        r"\texttt{U\_PT,L\_PT}": r"$U_{\mathrm{PT}},L_{\mathrm{PT}}$",
        r"\texttt{U\_PT,L\_IT}": r"$U_{\mathrm{PT}},L_{\mathrm{IT}}$",
        r"\texttt{U\_IT,L\_PT}": r"$U_{\mathrm{IT}},L_{\mathrm{PT}}$",
        r"\texttt{U\_IT,L\_IT}": r"$U_{\mathrm{IT}},L_{\mathrm{IT}}$",
        r"\texttt{Y(U,L)}": r"$Y(U,L)$",
        r"\texttt{Y(U\_PT,L\_PT)}": r"$Y(U_{\mathrm{PT}},L_{\mathrm{PT}})$",
        r"\texttt{Y(U\_PT,L\_IT)}": r"$Y(U_{\mathrm{PT}},L_{\mathrm{IT}})$",
        r"\texttt{Y(U\_IT,L\_PT)}": r"$Y(U_{\mathrm{IT}},L_{\mathrm{PT}})$",
        r"\texttt{Y(U\_IT,L\_IT)}": r"$Y(U_{\mathrm{IT}},L_{\mathrm{IT}})$",
    }
    for src, dst in math_replacements.items():
        tex = tex.replace(src, dst)
    tex = re.sub(
        r"\\pandocbounded\{\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}\}",
        r"\\includegraphics[width=\\linewidth,keepaspectratio]{\1}",
        tex,
    )
    tex_path.write_text(tex)


def filled_checklist(build_dir: Path) -> None:
    src = REPO / "paper_draft/neurips_v25_build/checklist_v25_filled.tex"
    text = src.read_text()
    text = sanitize_text(text)
    (build_dir / "checklist_filled.tex").write_text(text)


def write_main_tex(title: str, abstract: str, build_dir: Path) -> None:
    abstract = abstract.replace("\n", " ")
    abstract = abstract.replace("%", r"\%")
    title = title.replace("%", r"\%")
    main = rf"""
\documentclass{{article}}
\usepackage[main]{{neurips_2026}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage{{booktabs}}
\usepackage{{amsfonts}}
\usepackage{{amsmath}}
\usepackage{{nicefrac}}
\usepackage{{microtype}}
\usepackage{{xcolor}}
\usepackage{{graphicx}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{calc}}
\usepackage{{textcomp}}
\usepackage{{upquote}}
\usepackage{{enumitem}}
\usepackage{{etoolbox}}
\setlist{{nosep,leftmargin=*}}
\setlength{{\LTleft}}{{0pt}}
\setlength{{\LTright}}{{0pt}}
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\newcounter{{none}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\title{{{title}}}
\author{{Anonymous authors\\NeurIPS 2026 Submission}}

\begin{{document}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}

% START_MAIN_TEXT
\input{{body.tex}}

\clearpage
% START_CHECKLIST
\input{{checklist_filled.tex}}
\end{{document}}
"""
    (build_dir / "main.tex").write_text(main.strip() + "\n")


def build_pdf(neurips_zip: Path) -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    (BUILD_DIR / "figures").mkdir(parents=True)
    extract_template(neurips_zip, BUILD_DIR)

    run(["python", "scripts/plot/plot_first_divergence_schematic_examples.py"])
    title, abstract, body = split_markdown(PAPER_MD.read_text())
    processed = preprocess_body_markdown(body, BUILD_DIR / "figures")
    (BUILD_DIR / "body.md").write_text(processed)
    convert_body_to_latex(BUILD_DIR / "body.md", BUILD_DIR / "body.tex")
    filled_checklist(BUILD_DIR)
    write_main_tex(title, abstract, BUILD_DIR)
    run(["tectonic", "main.tex", "--keep-logs", "--keep-intermediates"], cwd=BUILD_DIR)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD_DIR / "main.pdf", PDF_OUT)


def pdf_text_by_page(pdf: Path) -> list[str]:
    reader = PdfReader(str(pdf))
    return [page.extract_text() or "" for page in reader.pages]


def first_page_containing(pages: list[str], needle: str) -> int | None:
    for idx, text in enumerate(pages, start=1):
        if needle in text:
            return idx
    return None


def validate_pdf() -> dict[str, object]:
    pages = pdf_text_by_page(PDF_OUT)
    refs_page = first_page_containing(pages, "References")
    checklist_page = first_page_containing(pages, "NeurIPS Paper Checklist")
    appendix_page = first_page_containing(pages, "The main text is written around stable claim names")
    if refs_page is None:
        raise RuntimeError("Could not find References in generated PDF")
    if checklist_page is None:
        raise RuntimeError("Could not find NeurIPS Paper Checklist in generated PDF")
    main_pages = refs_page - 1
    size = PDF_OUT.stat().st_size
    if main_pages > 9:
        raise RuntimeError(f"Main text is {main_pages} pages; NeurIPS limit is 9")
    if size >= 50 * 1024 * 1024:
        raise RuntimeError(f"PDF is {size / (1024 * 1024):.1f}MB; limit is 50MB")
    if appendix_page is not None and not (refs_page < appendix_page < checklist_page):
        raise RuntimeError("Expected order is references, appendices, then checklist")
    return {
        "pdf": rel(PDF_OUT),
        "bytes": size,
        "total_pages": len(pages),
        "main_pages_before_references": main_pages,
        "references_page": refs_page,
        "appendix_roadmap_page": appendix_page,
        "checklist_page": checklist_page,
    }


def render_previews(validation: dict[str, object]) -> list[str]:
    preview_dir = SUBMISSION_DIR / "previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True)
    doc = fitz.open(PDF_OUT)
    candidates = {
        "first_page": 1,
        "figure1_page": first_page_containing(pdf_text_by_page(PDF_OUT), "Figure 1"),
        "references_page": validation.get("references_page"),
        "appendix_roadmap_page": validation.get("appendix_roadmap_page"),
        "checklist_page": validation.get("checklist_page"),
    }
    written: list[str] = []
    for name, page_num in candidates.items():
        if not page_num:
            continue
        page = doc[int(page_num) - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
        out = preview_dir / f"{name}_p{page_num}.png"
        pix.save(out)
        written.append(rel(out))
    return written


def collect_claim_dependency_paths() -> set[str]:
    opened: set[str] = set()
    import gzip

    orig_path_open = Path.open
    orig_gzip_open = gzip.open

    def record(path: object) -> None:
        try:
            p = Path(path)
        except TypeError:
            return
        if p.is_absolute():
            try:
                p = p.relative_to(REPO)
            except ValueError:
                return
        s = p.as_posix()
        if s.startswith("results/") or s.startswith("data/"):
            opened.add(s)

    def logged_path_open(self: Path, *args, **kwargs):
        record(self)
        return orig_path_open(self, *args, **kwargs)

    def logged_gzip_open(filename, *args, **kwargs):
        record(filename)
        return orig_gzip_open(filename, *args, **kwargs)

    Path.open = logged_path_open  # type: ignore[assignment]
    gzip.open = logged_gzip_open  # type: ignore[assignment]
    try:
        old_argv = sys.argv[:]
        sys.argv = [str(REPO / "scripts/reproduce/check_paper_claims.py")]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                runpy.run_path(str(REPO / "scripts/reproduce/check_paper_claims.py"), run_name="__main__")
            except SystemExit as exc:
                code = exc.code if exc.code is not None else 0
                if code not in (0, "0"):
                    raise
    finally:
        sys.argv = old_argv
        Path.open = orig_path_open  # type: ignore[assignment]
        gzip.open = orig_gzip_open  # type: ignore[assignment]
    return opened


def write_supplement_readme(stage: Path, validation: dict[str, object]) -> None:
    readme = f"""# Anonymous NeurIPS 2026 Reviewer Supplement

This archive contains an anonymized, compact reproduction bundle for the paper
PDF submitted alongside it. Written appendices are intentionally not duplicated
here; they are part of the main PDF.

## Contents

- `results/`: compact JSON/CSV summaries and paper-facing plots.
- `scripts/`: CPU claim checker plus analysis/plot scripts.
- `src/poc/`: selected experiment packages for the feature/recipe/continuation analyses.
- `MANIFEST.sha256`: SHA-256 hash manifest for all files in this archive,
  compatible with `sha256sum -c MANIFEST.sha256`.

## Reviewer-Facing Reproduction

```bash
python scripts/reproduce/check_paper_claims.py
```

This CPU-only check verifies paper-facing numeric claims against the included
summary artifacts. Full raw intervention reruns require multi-GPU A100/H100
hardware and are documented in the paper appendix.

Optional shard-level smoke checks are exposed through
`bash scripts/reproduce/reproduce_minimal.sh`. The raw shard itself is not
bundled in this compact supplement; the script exits cleanly with instructions
unless `SHARD_DIR` points to an unpacked anonymous audit shard.

## Built PDF Summary

- Main pages before references: `{validation["main_pages_before_references"]}`
- Total pages including appendices/checklist: `{validation["total_pages"]}`
- Checklist page: `{validation["checklist_page"]}`
"""
    (stage / "README.md").write_text(readme)


def stage_supplement(validation: dict[str, object]) -> None:
    if SUPP_STAGE.exists():
        shutil.rmtree(SUPP_STAGE)
    SUPP_STAGE.mkdir(parents=True)

    for relpath in STATIC_SUPPLEMENT_FILES:
        copy_sanitized(REPO / relpath, SUPP_STAGE)
    for relpath in STATIC_SUPPLEMENT_DIRS:
        copy_tree_sanitized(REPO / relpath, SUPP_STAGE)
    for glob in SUPPLEMENT_RESULT_GLOBS:
        for path in REPO.glob(glob):
            if path.is_file():
                copy_sanitized(path, SUPP_STAGE)

    for dep in sorted(collect_claim_dependency_paths()):
        copy_sanitized(REPO / dep, SUPP_STAGE)

    write_supplement_readme(SUPP_STAGE, validation)
    (SUPP_STAGE / "artifact_map.json").write_text(
        json.dumps(
            {
                "note": "MANIFEST.sha256 contains per-file hashes for this archive.",
                "claim_groups": {
                    "core5_first_divergence": [
                        "results/paper_synthesis/exp23_core5",
                        "results/paper_synthesis/exp24_32b_external_validity",
                    ],
                    "validation_controls": [
                        "results/exp36_offmanifold_validation",
                        "results/exp37_random_prefix_baseline",
                        "results/exp40_prelate_commitment_control",
                    ],
                    "depth_terminal_anatomy": [
                        "results/paper_synthesis/exp20_exp21_handoff_table.csv",
                        "results/exp31_terminal_depth_factorial",
                        "results/exp32_terminal_mlp_writeout",
                        "results/exp33_terminal_identity_margin",
                    ],
                    "sparse_terminal_features": [
                        "results/exp34_dense5_final_readout_crosscoder",
                        "results/exp39_causal_feature_interpretation",
                        "results/exp41_causal_feature_bucket_steering",
                        "results/exp42_terminal_feature_upstream_conditioning",
                        "results/exp43_feature_rescue_handoff",
                        "results/exp44_middle_terminal_feature_handoff",
                    ],
                    "structured_boundary_state": [
                        "results/exp48_static_chimera_sequence_validation",
                    ],
                    "recipe_stage_behavior": [
                        "results/exp35_olmo_base_anchored_stage_decomposition",
                        "results/exp46_tulu_fixed_support_stage_sweep",
                        "results/exp47_same_base_recipe_specificity",
                        "results/exp49_constrained_continuation_bridge",
                        "results/exp50_llm_judge_behavior_bridge",
                        "results/exp52_forced_token_consequence_bridge",
                        "results/exp53_controlled_domain_finetunes",
                    ],
                    "cpu_checks": [
                        "scripts/reproduce/check_paper_claims.py",
                        "scripts/reproduce/reproduce_minimal.sh",
                        "scripts/reproduce/check_minimal_shard.py",
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    write_manifest(SUPP_STAGE)
    scan_for_leaks(SUPP_STAGE)

    result = run(["python", "scripts/reproduce/check_paper_claims.py"], cwd=SUPP_STAGE, capture=True)
    (SUBMISSION_DIR / "supplement_check_output.md").write_text(result.stdout or "")

    if SUPP_ZIP.exists():
        SUPP_ZIP.unlink()
    with zipfile.ZipFile(SUPP_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(SUPP_STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(SUPP_STAGE).as_posix())
    if SUPP_ZIP.stat().st_size >= 100 * 1024 * 1024:
        raise RuntimeError(f"Supplement ZIP is {SUPP_ZIP.stat().st_size / (1024 * 1024):.1f}MB; limit is 100MB")


def write_manifest(stage: Path) -> None:
    rows: list[tuple[str, str]] = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((digest, path.relative_to(stage).as_posix()))
    with (stage / "MANIFEST.sha256").open("w") as f:
        for digest, relpath in rows:
            f.write(f"{digest}  {relpath}\n")


def scan_for_leaks(root: Path) -> None:
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for pattern in LEAK_PATTERNS:
            if pattern in text:
                hits.append(f"{path.relative_to(root)}: {pattern}")
        for pattern in SECRET_REGEXES:
            if pattern.search(text):
                hits.append(f"{path.relative_to(root)}: secret-like token")
    if hits:
        sample = "\n".join(hits[:40])
        raise RuntimeError(f"Anonymity scan failed:\n{sample}")


def scan_generated_sources() -> None:
    scan_for_leaks(BUILD_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurips-template-zip", type=Path, default=NEURIPS_ZIP_DEFAULT)
    parser.add_argument("--skip-supplement", action="store_true")
    args = parser.parse_args()

    if not args.neurips_template_zip.exists():
        raise FileNotFoundError(args.neurips_template_zip)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    build_pdf(args.neurips_template_zip)
    scan_generated_sources()
    validation = validate_pdf()
    previews = render_previews(validation)
    validation["preview_pngs"] = previews
    (SUBMISSION_DIR / "build_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))

    if not args.skip_supplement:
        stage_supplement(validation)
        validation["supplement_zip"] = rel(SUPP_ZIP)
        validation["supplement_zip_bytes"] = SUPP_ZIP.stat().st_size
        (SUBMISSION_DIR / "build_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))

    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
