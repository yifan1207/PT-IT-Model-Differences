#!/usr/bin/env python3
"""Build non-anonymous NeurIPS-style arXiv PDFs and source bundles.

This intentionally reuses the paper-specific NeurIPS build preprocessors so
the arXiv PDFs keep the same typography/layout as the submission PDFs. The
wrapper switches the official style to ``preprint`` mode, restores real author
metadata, and puts public code/artifact links on the first page.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


REPO = Path(__file__).resolve().parents[2]
NEURIPS_ZIP_DEFAULT = Path.home() / "Downloads/Formatting_Instructions_For_NeurIPS_2026.zip"
NEURIPS_STYLE_FALLBACK = REPO / "paper_draft/submission/build/neurips_2026.sty"
ARXIV_ROOT = REPO / "paper_draft/arxiv"


@dataclass(frozen=True)
class PaperSpec:
    key: str
    build_module: Path
    paper_md: Path
    out_dir: Path
    github: str
    release: str
    gcs: str
    support_zip_name: str
    quick_reproduction: str
    convergence: bool = False


PAPERS = [
    PaperSpec(
        key="first_divergence_crosspatching",
        build_module=REPO / "scripts/reproduce/build_neurips_submission.py",
        paper_md=REPO / "paper_draft/PAPER_DRAFT_v25.md",
        out_dir=ARXIV_ROOT / "first_divergence_crosspatching",
        github="https://github.com/yifan1207/first-divergence-crosspatching",
        release="https://github.com/yifan1207/first-divergence-crosspatching/releases/tag/paper-artifacts-v1",
        gcs="gs://pt-vs-it-results/papers/first_divergence_crosspatching/",
        support_zip_name="first-divergence-crosspatching-supporting-material.zip",
        quick_reproduction="python scripts/reproduce/check_paper_claims.py",
    ),
    PaperSpec(
        key="convergence_gap",
        build_module=REPO / "scripts/reproduce/build_neurips_convergence_gap_submission.py",
        paper_md=REPO / "paper_draft/PAPER_DRAFT_convergence_gap_v3.md",
        out_dir=ARXIV_ROOT / "convergence_gap",
        github="https://github.com/yifan1207/convergence-gap-instruction-tuning",
        release="https://github.com/yifan1207/convergence-gap-instruction-tuning/releases/tag/paper-artifacts-v1",
        gcs="gs://pt-vs-it-results/papers/convergence_gap/",
        support_zip_name="convergence-gap-instruction-tuning-supporting-material.zip",
        quick_reproduction="python scripts/reproduce/check_convergence_gap_claims.py",
        convergence=True,
    ),
]


def run(cmd: list[str], *, cwd: Path = REPO) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_template(module, neurips_zip: Path | None, build_dir: Path) -> None:
    if neurips_zip is None:
        if not NEURIPS_STYLE_FALLBACK.exists():
            raise FileNotFoundError(NEURIPS_STYLE_FALLBACK)
        shutil.copy2(NEURIPS_STYLE_FALLBACK, build_dir / "neurips_2026.sty")
        return
    sig = inspect.signature(module.extract_template)
    if len(sig.parameters) == 2:
        module.extract_template(neurips_zip, build_dir)
    else:
        module.extract_template(neurips_zip)


def tex_escape(text: str) -> str:
    return text.replace("%", r"\%").replace("&", r"\&")


def write_arxiv_main_tex(spec: PaperSpec, title: str, abstract: str, build_dir: Path) -> None:
    abstract = abstract.replace("\n", " ")
    title = tex_escape(title)
    abstract = tex_escape(abstract)
    longtable_extra = r"\AtBeginEnvironment{longtable}{\scriptsize}" if spec.convergence else ""
    visible_github = spec.github.removeprefix("https://")
    main = rf"""
\documentclass{{article}}
\usepackage[preprint]{{neurips_2026}}
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
\usepackage{{float}}
\usepackage{{textcomp}}
\usepackage{{upquote}}
\usepackage{{enumitem}}
\usepackage{{etoolbox}}
\setlist{{nosep,leftmargin=*}}
\setlength{{\LTleft}}{{\fill}}
\setlength{{\LTright}}{{\fill}}
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\newcounter{{none}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
{longtable_extra}
\title{{{title}}}
\author{{Yifan Zhou\\
\normalfont\small University of California, Los Angeles\\
\normalfont\small \texttt{{yifanz1207@gmail.com}}\\
\normalfont\small Code: \href{{{spec.github}}}{{\texttt{{{visible_github}}}}}\\
\normalfont\small Artifact release: \href{{{spec.release}}}{{paper-artifacts-v1}}\\
\normalfont\small Raw mirror: \url{{{spec.gcs}}}}}

\begin{{document}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}

\input{{body.tex}}
\end{{document}}
"""
    (build_dir / "main.tex").write_text(main.strip() + "\n")


def build_pdf(spec: PaperSpec, neurips_zip: Path | None) -> tuple[str, str]:
    module = load_module(spec.build_module)
    build_dir = spec.out_dir / "build_neurips"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    (build_dir / "figures").mkdir(parents=True)

    module.BUILD_DIR = build_dir
    module.PAPER_MD = spec.paper_md
    module.sanitize_text = lambda text: text
    extract_template(module, neurips_zip, build_dir)

    if spec.convergence:
        if hasattr(module, "refresh_reporting_tables"):
            module.refresh_reporting_tables()
        md = module.apply_reporting_tables(spec.paper_md.read_text())
        title, abstract, body = module.split_markdown(md)
        (build_dir / "body.md").write_text(module.preprocess_body(body))
        module.convert_body_to_latex()
    else:
        run(["python", "scripts/plot/plot_first_divergence_schematic_examples.py"])
        title, abstract, body = module.split_markdown(spec.paper_md.read_text())
        processed = module.preprocess_body_markdown(body, build_dir / "figures")
        (build_dir / "body.md").write_text(processed)
        module.convert_body_to_latex(build_dir / "body.md", build_dir / "body.tex")

    write_arxiv_main_tex(spec, title, abstract, build_dir)
    run(["tectonic", "main.tex", "--keep-logs", "--keep-intermediates"], cwd=build_dir)
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build_dir / "main.pdf", spec.out_dir / "paper.pdf")
    return title, abstract


def copy_source_tree(build_dir: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ["main.tex", "body.tex", "neurips_2026.sty"]:
        shutil.copy2(build_dir / name, dest / name)
    if (build_dir / "figures").exists():
        shutil.copytree(build_dir / "figures", dest / "figures")


def write_metadata(spec: PaperSpec, title: str) -> None:
    readme = f"""# arXiv artifacts: {title}

Author: Yifan Zhou

- Paper PDF: `paper.pdf`
- Source bundle: `source.zip`
- Supporting material: `{spec.support_zip_name}`
- GitHub repository: {spec.github}
- GitHub release bundle: {spec.release}
- Large artifact mirror: `{spec.gcs}`

The supporting-material zip contains the compact code/data artifact bundle and
can be checked with the quick reproduction command documented in its README.
"""
    (spec.out_dir / "ARXIV_README.md").write_text(readme)
    (spec.out_dir / "ARTIFACT_LINKS.json").write_text(
        json.dumps(
            {
                "paper_title": title,
                "author": "Yifan Zhou",
                "github_repository": spec.github,
                "github_release": spec.release,
                "gcs_artifact_prefix": spec.gcs,
                "main_pdf": "paper.pdf",
                "source": "source.zip",
                "supporting_material": spec.support_zip_name,
                "quick_reproduction": spec.quick_reproduction,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def public_paper_markdown(spec: PaperSpec) -> str:
    text = spec.paper_md.read_text()
    public_header = (
        f"**Yifan Zhou** | University of California, Los Angeles | "
        f"`yifanz1207@gmail.com` | "
        f"[Code]({spec.github}) | "
        f"[Artifact release]({spec.release}) | "
        f"Raw mirror: `{spec.gcs}`"
    )
    lines = text.splitlines()
    for idx, line in enumerate(lines[:8]):
        if "Anonymous authors" in line or "NeurIPS 2026 Submission" in line:
            lines[idx] = public_header
            break
    else:
        if lines and lines[0].startswith("# "):
            lines.insert(1, "")
            lines.insert(2, public_header)
    return "\n".join(lines).rstrip() + "\n"


def zip_dir(source: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source).as_posix())


def refresh_zips(spec: PaperSpec, title: str) -> None:
    build_dir = spec.out_dir / "build_neurips"
    write_metadata(spec, title)

    source_stage = spec.out_dir / "source_stage"
    copy_source_tree(build_dir, source_stage)
    paper_dir = source_stage / "paper"
    paper_dir.mkdir()
    (paper_dir / "PAPER_DRAFT.md").write_text(public_paper_markdown(spec))
    shutil.copy2(spec.out_dir / "ARXIV_README.md", source_stage / "ARXIV_README.md")
    shutil.copy2(spec.out_dir / "ARTIFACT_LINKS.json", source_stage / "ARTIFACT_LINKS.json")
    zip_dir(source_stage, spec.out_dir / "source.zip")

    support = spec.out_dir / "supporting_material"
    if support.exists():
        (support / "paper").mkdir(exist_ok=True)
        shutil.copy2(spec.out_dir / "paper.pdf", support / "paper/paper.pdf")
        (support / "paper/PAPER_DRAFT.md").write_text(public_paper_markdown(spec))
        shutil.copy2(spec.out_dir / "ARXIV_README.md", support / "ARXIV_README.md")
        shutil.copy2(spec.out_dir / "ARTIFACT_LINKS.json", support / "ARTIFACT_LINKS.json")
        copy_source_tree(build_dir, support / "source")
        zip_dir(support, spec.out_dir / spec.support_zip_name)


def validate_pdf(spec: PaperSpec) -> None:
    pdf = spec.out_dir / "paper.pdf"
    reader = PdfReader(str(pdf))
    first = reader.pages[0].extract_text() or ""
    required = ["Yifan Zhou", "github.com/yifan1207", "Preprint"]
    forbidden = ["Anonymous authors", "NeurIPS 2026 Submission", "Paper Checklist", "Submitted to"]
    missing = [item for item in required if item not in first]
    bad = [item for item in forbidden if item in first]
    if missing or bad:
        raise RuntimeError(f"{pdf} validation failed; missing={missing}, forbidden={bad}")
    print(
        json.dumps(
            {
                "pdf": str(pdf),
                "pages": len(reader.pages),
                "bytes": pdf.stat().st_size,
                "source_zip_bytes": (spec.out_dir / "source.zip").stat().st_size,
                "support_zip_bytes": (spec.out_dir / spec.support_zip_name).stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    neurips_zip = NEURIPS_ZIP_DEFAULT if NEURIPS_ZIP_DEFAULT.exists() else None
    if neurips_zip is None:
        print(f"Template zip not found; using extracted style fallback: {NEURIPS_STYLE_FALLBACK}")
    for spec in PAPERS:
        print(f"== building {spec.key} ==")
        title, _abstract = build_pdf(spec, neurips_zip)
        refresh_zips(spec, title)
        validate_pdf(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
