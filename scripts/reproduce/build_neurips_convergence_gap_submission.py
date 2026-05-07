#!/usr/bin/env python3
"""Build the convergence-gap second-paper PDF and compact supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import fitz
from pypdf import PdfReader


REPO = Path(__file__).resolve().parents[2]
PAPER_MD = REPO / "paper_draft/PAPER_DRAFT_convergence_gap_v3.md"
SUBMISSION_DIR = REPO / "paper_draft/submission_convergence_gap"
BUILD_DIR = SUBMISSION_DIR / "build"
SUPP_STAGE = SUBMISSION_DIR / "supplement_staging"
PDF_OUT = SUBMISSION_DIR / "neurips2026_main.pdf"
SUPP_ZIP = SUBMISSION_DIR / "neurips2026_supplement.zip"
NEURIPS_ZIP_DEFAULT = Path.home() / "Downloads/Formatting_Instructions_For_NeurIPS_2026.zip"

TEXT_SUFFIXES = {".bib", ".csv", ".json", ".jsonl", ".md", ".py", ".sh", ".tex", ".txt", ".yaml", ".yml"}
LEAK_PATTERNS = ["Yifan", "/Users/", "github.com/yifan", "pt-vs-it-results", "gs://", "RunPod", "runpod"]

STATIC_FILES = [
    "paper_draft/PAPER_DRAFT_convergence_gap_v3.md",
    "scripts/reproduce/check_convergence_gap_claims.py",
    "scripts/reproduce/reproduce_convergence_gap_minimal.sh",
    "scripts/analysis/build_convergence_gap_reporting_tables.py",
    "scripts/analysis/build_exp22_endpoint_deconfounded_synthesis.py",
    "scripts/analysis/build_exp22_template_regime_audit.py",
    "scripts/analysis/analyze_exp22_fixed_history_template_audit.py",
    "scripts/analysis/build_exp22_fixed_history_template_audit.py",
    "scripts/analysis/analyze_exp55_late_window_robustness.py",
    "scripts/run/run_exp22_fixed_history_template_audit_runpod.sh",
    "scripts/run/run_exp55_late_window_robustness_runpod.sh",
]

RESULT_FILES = [
    "results/exp09_cross_model_observational_replication/data/convergence_gap_values.json",
    "results/exp09_cross_model_observational_replication/data/exp9_summary.json",
    "results/exp09_cross_model_observational_replication/plots/L2_mean_kl_per_layer_tuned.png",
    "results/exp09_cross_model_observational_replication/plots/L2_commitment_tuned_top1.png",
    "results/exp09_cross_model_observational_replication/plots/L1_delta_cosine_6panel.png",
    "results/paper_synthesis/convergence_gap_reporting_tables.md",
    "results/paper_synthesis/convergence_gap_reporting_tables.json",
    "results/paper_synthesis/convergence_gap_reporting_tables.csv",
    "results/paper_synthesis/exp22_endpoint_deconfounded_table.csv",
    "results/paper_synthesis/exp22_endpoint_deconfounded_summary.png",
    "results/paper_synthesis/exp22_template_raw_public600_audit.json",
    "results/paper_synthesis/exp22_template_raw_public600_effects.csv",
    "results/paper_synthesis/exp22_template_raw_public600_lengths.csv",
    "results/paper_synthesis/exp22_template_raw_public600_note.md",
    "results/paper_synthesis/exp22_fixed_history_template_audit.json",
    "results/paper_synthesis/exp22_fixed_history_template_audit_effects.csv",
    "results/paper_synthesis/exp22_fixed_history_template_audit_support.csv",
    "results/paper_synthesis/exp22_fixed_history_template_audit_note.md",
    "results/paper_synthesis/exp22_fixed_history_template_audit.png",
    "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit.json",
    "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit_effects.csv",
    "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit_support.csv",
    "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit_note.md",
    "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit.png",
    "results/paper_synthesis/exp55_late_window_robustness.json",
    "results/paper_synthesis/exp55_late_window_robustness_effects.csv",
    "results/paper_synthesis/exp55_late_window_robustness_note.md",
    "results/paper_synthesis/exp55_late_window_robustness.png",
    "results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/depth_ablation_metrics.json",
    "results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_600rand_v11_depthablation_full/depth_ablation_paper_main.png",
    "results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_summary.json",
    "results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416/exp13_full_causal_main.png",
    "results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/exp19B_summary_light.json",
    "results/exp19_late_mlp_specificity_controls/exp19B_core120_h100x8_20260424_050421_analysis/exp19B_final20_kl_true_vs_random.png",
]


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


def sanitize_text(text: str) -> str:
    replacements = {
        "/Users/Yifan/Research/structral-semantic-features/": "",
        "/Users/Yifan/Research/structral-semantic-features": ".",
        "/Users/Yifan/": "./",
        "Yifan": "Anonymous",
        "pt-vs-it-results": "anonymous-artifact-store",
        "gs://": "anonymous-object-store://",
        "RunPod": "RemoteJob",
        "runpod": "remotejob",
        "Yanda": "InternalServer",
        "yanda": "internal-server",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"(?i)(openai|hf|wandb)[_-]?api[_-]?key\s*=\s*['\"][^'\"]+['\"]", r"\1_API_KEY=<redacted>", text)
    text = re.sub(r"(?i)\b(hf|ghp|sk)-[A-Za-z0-9_\-]{20,}\b", "<redacted-token>", text)
    return text


def tex_safe_unicode(text: str) -> str:
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
    if not src.exists() or src.is_dir() or "__pycache__" in src.parts or src.suffix in {".pyc", ".pyo"}:
        return
    dest = dest_root / (dest_rel or rel(src))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_SUFFIXES:
        try:
            dest.write_text(sanitize_text(src.read_text()))
        except UnicodeDecodeError:
            shutil.copy2(src, dest)
    else:
        shutil.copy2(src, dest)


def extract_template(neurips_zip: Path) -> None:
    with zipfile.ZipFile(neurips_zip) as zf:
        for name in ["neurips_2026.sty", "checklist.tex"]:
            (BUILD_DIR / name).write_bytes(zf.read(name))


def split_markdown(md: str) -> tuple[str, str, str]:
    lines = md.splitlines()
    title = lines[0].removeprefix("# ").strip()
    abstract_idx = lines.index("## Abstract")
    start = abstract_idx + 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "---")
    abstract = " ".join(line.strip() for line in lines[start:end] if line.strip())
    body = "\n".join(lines[end + 1 :]).strip() + "\n"
    return tex_safe_unicode(title), tex_safe_unicode(abstract), body


def refresh_reporting_tables() -> None:
    run(["python", "scripts/analysis/build_convergence_gap_reporting_tables.py"])


def apply_reporting_tables(md: str) -> str:
    path = REPO / "results/paper_synthesis/convergence_gap_reporting_tables.json"
    data = json.loads(path.read_text())
    tables = data["tables"]
    pattern = re.compile(
        r"<!-- REPORT_TABLE: ([A-Za-z0-9_]+) -->.*?<!-- /REPORT_TABLE -->",
        re.DOTALL,
    )

    used: set[str] = set()

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in tables:
            raise KeyError(f"unknown generated reporting table {key!r}")
        used.add(key)
        table = tables[key]["markdown"].rstrip()
        return f"<!-- REPORT_TABLE: {key} -->\n{table}\n<!-- /REPORT_TABLE -->"

    out = pattern.sub(repl, md)
    missing = set(tables) - used
    if missing:
        raise RuntimeError(f"generated reporting tables not used in paper: {sorted(missing)}")
    return out


def strip_heading_number(text: str) -> str:
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text)
    text = re.sub(r"^Appendix\s+[A-Z]:\s+", "", text)
    return text


def rewrite_image_paths(md: str, figures_dir: Path) -> str:
    pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    seen: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        raw_path = match.group(2)
        source = (PAPER_MD.parent / raw_path).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if str(source) not in seen:
            digest = hashlib.sha1(str(source).encode()).hexdigest()[:8]
            target = f"figures/{source.stem}_{digest}{source.suffix}"
            shutil.copy2(source, figures_dir / Path(target).name)
            seen[str(source)] = target
        target = seen[str(source)]
        height = "0.24\\textheight"
        if "L2_mean_kl_per_layer" in target:
            height = "0.30\\textheight"
        if "dose_response" in target:
            height = "0.22\\textheight"
        return "\n".join(
            [
                r"\begin{center}",
                rf"\includegraphics[width=0.94\linewidth,height={height},keepaspectratio]{{{target}}}",
                r"\end{center}",
            ]
        )

    return pattern.sub(repl, md)


def preprocess_body(body: str) -> str:
    out: list[str] = []
    appendix_started = False
    for line in body.splitlines():
        if line.strip() == "---":
            continue
        if line.startswith("## References"):
            out.extend(["", r"\clearpage", r"\section*{References}", ""])
            continue
        if line.startswith("## Appendix Roadmap"):
            out.extend(["", r"\clearpage", r"\section*{Appendix Roadmap}", ""])
            continue
        if re.match(r"^## Appendix [A-Z]:", line):
            if not appendix_started:
                out.extend(["", r"\appendix", ""])
                appendix_started = True
            out.append("## " + strip_heading_number(line[3:].strip()))
            continue
        if re.match(r"^#{2,4} ", line):
            hashes, text = line.split(" ", 1)
            out.append(f"{hashes} {strip_heading_number(text.strip())}")
            continue
        out.append(line)
    text = "\n".join(out) + "\n"
    text = rewrite_image_paths(text, BUILD_DIR / "figures")
    return tex_safe_unicode(sanitize_text(text))


def convert_body_to_latex() -> None:
    run(
        [
            "pandoc",
            "body.md",
            "--from",
            "markdown+pipe_tables+tex_math_dollars+raw_tex+backtick_code_blocks+fenced_code_blocks",
            "--to",
            "latex",
            "--standalone=false",
            "--wrap=none",
            "--top-level-division=section",
            "--shift-heading-level-by=-1",
            "-o",
            "body.tex",
        ],
        cwd=BUILD_DIR,
    )
    tex = (BUILD_DIR / "body.tex").read_text()
    tex = tex.replace(r"\begin{quote}", r"\begin{quote}\small")
    (BUILD_DIR / "body.tex").write_text(tex)


def write_checklist() -> None:
    text = r"""
\section*{NeurIPS Paper Checklist}

\begin{enumerate}
\item {\bf Claims}
\item[] Answer: \answerYes{}
\item[] Justification: The abstract and introduction state the convergence-gap target, endpoint controls, and matched-prefix intervention scope.

\item {\bf Limitations}
\item[] Answer: \answerYes{}
\item[] Justification: The discussion and appendices cover endpoint/probe dependence, constructed interventions, model-family scope, and MoE scope.

\item {\bf Theory assumptions and proofs}
\item[] Answer: \answerNA{}
\item[] Justification: The paper is empirical and presents no formal theorem.

\item {\bf Experimental result reproducibility}
\item[] Answer: \answerYes{}
\item[] Justification: The appendix and supplement include a CPU claim checker over committed JSON/CSV summaries and identify the raw multi-GPU rerun families.

\item {\bf Open access to data and code}
\item[] Answer: \answerYes{}
\item[] Justification: The intended anonymous supplement includes analysis code, summary artifacts, figures, and a manifest.

\item {\bf Experimental setting/details}
\item[] Answer: \answerYes{}
\item[] Justification: The setup describes model pairs, convergence metrics, endpoint controls, and matched-prefix interventions.

\item {\bf Experiment statistical significance}
\item[] Answer: \answerYes{}
\item[] Justification: Main numeric claims report bootstrap confidence intervals or matched random controls where applicable.

\item {\bf Experiments compute resources}
\item[] Answer: \answerYes{}
\item[] Justification: The artifact map distinguishes CPU summary checks from optional multi-GPU raw reruns.

\item {\bf Code of ethics}
\item[] Answer: \answerYes{}
\item[] Justification: The work analyzes released language-model checkpoints and synthetic/evaluation prompts.

\item {\bf Broader impacts}
\item[] Answer: \answerYes{}
\item[] Justification: The work is a mechanistic diagnostic for model diffing; it does not release a new deployed model.

\item {\bf Safeguards}
\item[] Answer: \answerNA{}
\item[] Justification: No new high-risk model or dataset is released.

\item {\bf Licenses for existing assets}
\item[] Answer: \answerYes{}
\item[] Justification: Final artifacts should preserve model and dataset license metadata.

\item {\bf New assets}
\item[] Answer: \answerNA{}
\item[] Justification: The paper releases derived summaries and code, not a new dataset or model.

\item {\bf Crowdsourcing and research with human subjects}
\item[] Answer: \answerNA{}
\item[] Justification: No human-subject data collection is used for the main claims.

\item {\bf Institutional review board approvals or equivalent}
\item[] Answer: \answerNA{}
\item[] Justification: No human-subject study was conducted.

\item {\bf Declaration of LLM usage}
\item[] Answer: \answerYes{}
\item[] Justification: LLMs are used in auxiliary judge-style scoring; the core convergence-gap and intervention estimands are deterministic.
\end{enumerate}
"""
    (BUILD_DIR / "checklist_filled.tex").write_text(text.strip() + "\n")


def write_main_tex(title: str, abstract: str) -> None:
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
\usepackage{{caption}}
\usepackage{{textcomp}}
\usepackage{{upquote}}
\usepackage{{enumitem}}
\usepackage{{etoolbox}}
\setlist{{nosep,leftmargin=*}}
\setlength{{\LTleft}}{{0pt}}
\setlength{{\LTright}}{{0pt}}
\newcounter{{none}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\AtBeginEnvironment{{longtable}}{{\scriptsize}}
\title{{{title}}}
\author{{Anonymous authors\\NeurIPS 2026 Submission}}

\begin{{document}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}

\input{{body.tex}}

\clearpage
\input{{checklist_filled.tex}}
\end{{document}}
"""
    (BUILD_DIR / "main.tex").write_text(main.strip() + "\n")


def build_pdf(neurips_zip: Path) -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    (BUILD_DIR / "figures").mkdir(parents=True)
    extract_template(neurips_zip)
    title, abstract, body = split_markdown(apply_reporting_tables(PAPER_MD.read_text()))
    (BUILD_DIR / "body.md").write_text(preprocess_body(body))
    convert_body_to_latex()
    write_checklist()
    write_main_tex(title, abstract)
    run(["tectonic", "main.tex", "--keep-logs", "--keep-intermediates"], cwd=BUILD_DIR)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILD_DIR / "main.pdf", PDF_OUT)


def pdf_pages(pdf: Path) -> list[str]:
    reader = PdfReader(str(pdf))
    return [page.extract_text() or "" for page in reader.pages]


def first_page_containing(pages: list[str], needle: str) -> int | None:
    for idx, text in enumerate(pages, start=1):
        if needle in text:
            return idx
    return None


def validate_pdf() -> dict[str, object]:
    pages = pdf_pages(PDF_OUT)
    refs_page = first_page_containing(pages, "References")
    checklist_page = first_page_containing(pages, "NeurIPS Paper Checklist")
    appendix_page = first_page_containing(pages, "Appendix Roadmap")
    if refs_page is None or checklist_page is None:
        raise RuntimeError("Generated PDF is missing references or checklist")
    main_pages = refs_page - 1
    size = PDF_OUT.stat().st_size
    if main_pages > 9:
        raise RuntimeError(f"Main text is {main_pages} pages; NeurIPS limit is 9")
    if size >= 50 * 1024 * 1024:
        raise RuntimeError(f"PDF is {size / (1024 * 1024):.1f}MB; limit is 50MB")
    return {
        "pdf": rel(PDF_OUT),
        "bytes": size,
        "total_pages": len(pages),
        "main_pages_before_references": main_pages,
        "references_page": refs_page,
        "appendix_roadmap_page": appendix_page,
        "checklist_page": checklist_page,
    }


def render_previews(validation: dict[str, object]) -> None:
    preview_dir = SUBMISSION_DIR / "previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True)
    pages = pdf_pages(PDF_OUT)
    doc = fitz.open(PDF_OUT)
    candidates = {
        "first_page": 1,
        "figure1_page": first_page_containing(pages, "Figure 1"),
        "references_page": validation.get("references_page"),
        "appendix_roadmap_page": validation.get("appendix_roadmap_page"),
        "checklist_page": validation.get("checklist_page"),
    }
    written = []
    for name, page_num in candidates.items():
        if not page_num:
            continue
        page = doc[int(page_num) - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
        out = preview_dir / f"{name}_p{page_num}.png"
        pix.save(out)
        written.append(rel(out))
    validation["preview_pngs"] = written


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
    if hits:
        raise RuntimeError("Anonymity scan failed:\n" + "\n".join(hits[:40]))


def write_manifest(stage: Path) -> None:
    with (stage / "MANIFEST.sha256").open("w") as f:
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.sha256":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                f.write(f"{digest}  {path.relative_to(stage).as_posix()}  {path.stat().st_size}\n")


def stage_supplement(validation: dict[str, object]) -> None:
    if SUPP_STAGE.exists():
        shutil.rmtree(SUPP_STAGE)
    SUPP_STAGE.mkdir(parents=True)

    for path in [BUILD_DIR / "main.tex", BUILD_DIR / "body.tex", BUILD_DIR / "body.md", BUILD_DIR / "checklist_filled.tex", BUILD_DIR / "neurips_2026.sty"]:
        copy_sanitized(path, SUPP_STAGE, f"paper/{path.name}")
    for figure in (BUILD_DIR / "figures").glob("*"):
        copy_sanitized(figure, SUPP_STAGE, f"paper/figures/{figure.name}")
    copy_sanitized(PDF_OUT, SUPP_STAGE, "paper/neurips2026_main.pdf")

    for relpath in STATIC_FILES + RESULT_FILES:
        copy_sanitized(REPO / relpath, SUPP_STAGE)

    readme = f"""# Anonymous NeurIPS 2026 Reviewer Supplement: Convergence Gap

This archive contains a compact reproduction bundle for the convergence-gap
second paper. Written appendices are part of the main PDF and are not duplicated
as a separate prose document.

## Contents

- `paper/`: generated LaTeX source, NeurIPS style file, filled checklist, figure assets, and PDF.
- `paper_draft/PAPER_DRAFT_convergence_gap_v3.md`: source Markdown.
- `results/`: compact JSON/CSV summaries and paper-facing figures.
- `scripts/reproduce/check_convergence_gap_claims.py`: CPU-only claim checker.
- `MANIFEST.sha256`: SHA-256 hash manifest.

## Reviewer-Facing Reproduction

```bash
python scripts/reproduce/check_convergence_gap_claims.py
```

The CPU claim checker and reporting-table check are self-contained within this
archive and use only the bundled JSON/CSV summaries. The included
`scripts/analysis/` and `scripts/run/` entrypoints document how the summaries
were produced, but they are non-minimal audit scripts: full raw intervention
reruns require the full repository environment, model access, and multi-GPU
hardware.

## Built PDF Summary

- PDF path inside this archive: `paper/neurips2026_main.pdf`
- Main pages before references: `{validation["main_pages_before_references"]}`
- Total pages including appendices/checklist: `{validation["total_pages"]}`
- Checklist page: `{validation["checklist_page"]}`
"""
    (SUPP_STAGE / "README.md").write_text(readme)
    (SUPP_STAGE / "artifact_map.json").write_text(
        json.dumps(
            {
                "pdf": "paper/neurips2026_main.pdf",
                "supplement_root": ".",
                "claim_groups": {
                    "convergence_gap": [
                        "results/exp09_cross_model_observational_replication",
                        "results/paper_synthesis/convergence_gap_reporting_tables.json",
                        "results/paper_synthesis/exp22_endpoint_deconfounded_table.csv",
                        "results/paper_synthesis/exp22_fixed_history_template_audit.json",
                        "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit.json",
                    ],
                    "matched_prefix_localization": [
                        "results/exp11_matched_prefix_mlp_graft",
                        "results/exp14_symmetric_matched_prefix_causality",
                        "results/exp19_late_mlp_specificity_controls",
                    ],
                    "cpu_checks": [
                        "scripts/reproduce/check_convergence_gap_claims.py",
                        "scripts/reproduce/reproduce_convergence_gap_minimal.sh",
                    ],
                },
                "optional_full_repo_scripts": "Bundled scripts/analysis and scripts/run entrypoints document provenance but are not the minimal reproduction path.",
                "notes": {
                    "minimal_reproduction": "Run the CPU claim checker from this supplement root; it uses bundled summaries only.",
                    "optional_reruns": "Analysis and remote multi-GPU scripts are included for audit provenance and require the full repository environment, model access, and multi-GPU hardware.",
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    write_manifest(SUPP_STAGE)
    scan_for_leaks(SUPP_STAGE)
    result = run(["python", "scripts/reproduce/check_convergence_gap_claims.py"], cwd=SUPP_STAGE, capture=True)
    (SUBMISSION_DIR / "supplement_check_output.md").write_text(result.stdout or "")

    if SUPP_ZIP.exists():
        SUPP_ZIP.unlink()
    with zipfile.ZipFile(SUPP_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(SUPP_STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(SUPP_STAGE).as_posix())
    if SUPP_ZIP.stat().st_size >= 100 * 1024 * 1024:
        raise RuntimeError(f"Supplement ZIP is {SUPP_ZIP.stat().st_size / (1024 * 1024):.1f}MB; limit is 100MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurips-template-zip", type=Path, default=NEURIPS_ZIP_DEFAULT)
    parser.add_argument("--skip-supplement", action="store_true")
    args = parser.parse_args()
    if not args.neurips_template_zip.exists():
        raise FileNotFoundError(args.neurips_template_zip)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    refresh_reporting_tables()
    build_pdf(args.neurips_template_zip)
    scan_for_leaks(BUILD_DIR)
    validation = validate_pdf()
    render_previews(validation)
    if not args.skip_supplement:
        stage_supplement(validation)
        validation["supplement_zip"] = rel(SUPP_ZIP)
        validation["supplement_zip_bytes"] = SUPP_ZIP.stat().st_size
    (SUBMISSION_DIR / "build_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
