# arXiv v24 draft bundle

This folder contains a LaTeX draft generated from `../PAPER_DRAFT_v24.md`.

- `main.tex`: standalone LaTeX source for the current v24 paper text.
- `figures/`: copied PNG figure assets referenced by `main.tex`; the first-divergence schematic is rendered directly in TikZ.
- `figure_manifest.md`: source-to-bundle mapping for every copied figure.
- `data/`: compact CSV/JSON/Markdown artifacts for the headline tables and support analyses.
- `data_manifest.md`: source-to-bundle mapping for every copied data artifact.

Build command, once a TeX toolchain is available:

```bash
cd paper_draft/arxiv_v24
latexmk -pdf main.tex
```

Exp27 natural-rollout residual-opposition results are not included yet; the current text only notes it as an in-progress follow-up.
