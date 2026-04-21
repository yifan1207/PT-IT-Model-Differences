#!/usr/bin/env python3
"""Comprehensive dose-response plots for Exp6 A1/A2 experiments.

Governance metrics (IT >> PT):
  - coherent_assistant_rate  (GOV-CONV records): IT ~99%, PT ~12%
  - structural_token_ratio   (all records):       IT ~4%,  PT ~2%
  - format_compliance        (GOV-FORMAT records): IT ~17%, PT ~13%

Content metrics (baseline check):
  - mmlu_accuracy / exp3_factual_em
  - reasoning_em / exp3_reasoning_em

Safety (from LLM judge if available):
  - safety (0–3 → normalized 0–1)
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.poc.exp06_corrective_direction_steering.pca_analysis import (
    PCAArtifactsUnavailableError,
    compute_and_plot_pca_analysis,
)


# ── Coherent assistant rate (computed on-the-fly from sample_outputs.jsonl) ──

_QA_CONT_RE = re.compile(r'\n\s*(Question|Q)\s*:', re.IGNORECASE)
_ANS_CONT_RE = re.compile(r'\n\s*Answer\s*:', re.IGNORECASE)


def _coherent(text: str) -> float:
    if not text or len(text.split()) < 3:
        return 0.0
    words = text.split()
    if len(words) >= 8 and max(words.count(w) for w in set(words)) / len(words) > 0.30:
        return 0.0
    ns = text.replace(" ", "").replace("\n", "")
    if len(ns) > 15 and len(set(ns)) < 6:
        return 0.0
    if _QA_CONT_RE.search(text) or _ANS_CONT_RE.search(text):
        return 0.0
    return 1.0


def _load_coherent_from_samples(
    samples_path: Path,
    category_filter: str | None = "GOV-CONV",
) -> dict[str, float]:
    """Return {condition: mean_coherent_rate} from sample_outputs.jsonl."""
    if not samples_path.exists():
        return {}
    by_cond: dict[str, list[float]] = defaultdict(list)
    seen: set[tuple] = set()
    with open(samples_path) as f:
        for line in f:
            r = json.loads(line.strip())
            if category_filter and r.get("category") != category_filter:
                continue
            key = (r.get("condition", ""), r.get("record_id", ""))
            if key in seen:
                continue
            seen.add(key)
            by_cond[r["condition"]].append(_coherent(r.get("generated_text", "")))
    return {c: sum(v) / len(v) for c, v in by_cond.items() if v}


# ── Load CSV scores ───────────────────────────────────────────────────────────

def _load_scores(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    if not path.exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out[row["condition"]][row["benchmark"]] = float(row["value"])
            except (ValueError, TypeError):
                pass
    return out


# ── Load LLM judge scores ─────────────────────────────────────────────────────

def _load_judge(path: Path) -> dict[str, dict[str, float]]:
    """Returns {condition: {task: mean_score_0_1}} — legacy 0-3 format."""
    if not path.exists():
        return {}
    by_ct: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("score", -1) < 0:
                continue
            by_ct[r["condition"]][r["task"]].append(float(r["score"]) / 3.0)
    return {
        cond: {task: sum(v) / len(v) for task, v in tasks.items()}
        for cond, tasks in by_ct.items()
    }


def _load_judge_v2(path: Path) -> dict[str, dict[str, float]]:
    """Returns {condition: {task: normalized_0_1}} — v2 G1/G2/S1/S2 format.

    G1: mean/5  (governance quality, higher=better)
    G2: mean/5  (register quality, higher=better)
    S1: mean/3  (safety rate: REFUSE=3, COMPLY=0, INCOHERENT=1; higher=safer)
    S2: mean/1  (false-refusal rate; lower=better)
    """
    if not path.exists():
        return {}
    max_scores = {"g1": 5.0, "g2": 5.0, "s1": 3.0, "s2": 1.0}
    by_ct: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = r.get("score", -1)
            task = r.get("task", "")
            if score < 0 or task not in max_scores:
                continue
            norm = float(score) / max_scores[task]
            by_ct[r["condition"]][task].append(norm)
    return {
        cond: {task: sum(v) / len(v) for task, v in tasks.items()}
        for cond, tasks in by_ct.items()
    }


# ── Parse α / β from condition name ──────────────────────────────────────────

def _alpha(cond: str) -> float | None:
    if not cond.startswith("A1_alpha_"):
        return None
    try:
        return float(cond[len("A1_alpha_"):])
    except ValueError:
        return None


def _beta(cond: str) -> float | None:
    if not cond.startswith("A2_beta_"):
        return None
    try:
        return float(cond[len("A2_beta_"):])
    except ValueError:
        return None


# ── Shared plot helpers ───────────────────────────────────────────────────────

_COLORS = {
    "coherent_assistant_rate": "#1f77b4",   # blue  — legacy governance
    "structural_token_ratio":   "#ff7f0e",  # orange
    "format_compliance":        "#2ca02c",  # green
    "format_compliance_v2":     "#2ca02c",  # green (v2)
    "mmlu_accuracy":            "#d62728",  # red
    "mmlu_forced_choice":       "#d62728",  # red (v2)
    "exp3_factual_em":          "#d62728",
    "exp3_reasoning_em":        "#9467bd",  # purple
    "reasoning_em":             "#9467bd",
    "exp3_alignment_behavior":  "#8c564b",  # brown
    "alignment_behavior":       "#8c564b",
    "safety":                   "#7f7f7f",  # gray (legacy)
    "g1":                       "#1f77b4",  # blue — primary governance quality (v2)
    "g2":                       "#17becf",  # teal — register quality (v2)
    "s1":                       "#7f7f7f",  # gray — safety refuse rate (v2)
    "s2":                       "#bcbd22",  # yellow-green — false-refusal (v2, lower=better)
}

_LABELS = {
    "coherent_assistant_rate": "coherent_assistant (deprecated)",
    "structural_token_ratio":   "structural_token_ratio",
    "format_compliance":        "format_compliance (legacy)",
    "format_compliance_v2":     "format_compliance_v2 (IFEval criteria)",
    "mmlu_accuracy":            "mmlu_accuracy (legacy)",
    "mmlu_forced_choice":       "mmlu_forced_choice",
    "exp3_factual_em":          "factual_em",
    "exp3_reasoning_em":        "reasoning_em",
    "reasoning_em":             "reasoning_em",
    "exp3_alignment_behavior":  "alignment_behavior",
    "alignment_behavior":       "alignment_behavior",
    "safety":                   "safety (judge v1, 0-3→1)",
    "g1":                       "G1 governance quality (LLM judge, /5)",
    "g2":                       "G2 register quality (LLM judge, /5)",
    "s1":                       "S1 safety refuse rate (LLM judge)",
    "s2":                       "S2 false-refusal rate (lower=better)",
}


def _plot_panel(ax, xs, series: dict[str, list[float | None]], baselines: dict[str, tuple],
                title: str, xlabel: str, baseline_x: float) -> None:
    """Draw one panel: lines + baseline reference lines with explicit value annotations."""
    for bench, ys in series.items():
        color = _COLORS.get(bench, "#333333")
        label = _LABELS.get(bench, bench)
        clean = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not clean:
            continue
        cx, cy = zip(*clean)
        ax.plot(cx, cy, marker="o", markersize=4, color=color, label=label, linewidth=1.5)
        # PT/IT baseline reference lines with explicit value annotations
        if bench in baselines:
            pt_val, it_val = baselines[bench]
            x_right = max(xs) + (max(xs) - min(xs)) * 0.01 if len(xs) > 1 else baseline_x
            if pt_val is not None:
                ax.axhline(pt_val, color=color, linestyle=":", linewidth=1.8, alpha=0.8,
                           label=f"{label} PT={pt_val:.2f}")
                ax.annotate(f"PT {pt_val:.2f}", xy=(x_right, pt_val),
                            fontsize=6, color=color, alpha=0.9, va="center",
                            ha="left", clip_on=True, fontweight="bold")
            if it_val is not None:
                ax.axhline(it_val, color=color, linestyle="-.", linewidth=1.6, alpha=0.7,
                           label=f"{label} IT={it_val:.2f}")
                ax.annotate(f"IT {it_val:.2f}", xy=(x_right, it_val),
                            fontsize=6, color=color, alpha=0.8, va="center",
                            ha="left", clip_on=True)
    ax.axvline(baseline_x, color="gray", linestyle="--", alpha=0.35, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Score (0-1)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.25)


def _add_alpha_emphasis(axes, xs) -> None:
    """Shade α=0 and α=-1 columns to highlight 'returns to PT quality' regime."""
    x_min, x_max = min(xs), max(xs)
    half = (x_max - x_min) / len(xs) * 0.4  # half-width of each band
    for ax in axes:
        ylim = ax.get_ylim()
        y_top = ylim[1] * 0.98
        # α=0: full direction removal
        if x_min <= 0 <= x_max:
            ax.axvspan(0 - half, 0 + half, alpha=0.10, color="#e74c3c", zorder=0)
            ax.axvline(0, color="#e74c3c", linestyle="--", linewidth=1.4, alpha=0.55, zorder=1)
            ax.text(0, y_top, "α=0\n(ablated)", ha="center", va="top",
                    fontsize=7, color="#c0392b", fontweight="bold")
        # α=-1: over-removal, empirically returns to PT level
        if x_min <= -1 <= x_max:
            ax.axvspan(-1 - half, -1 + half, alpha=0.10, color="#8e44ad", zorder=0)
            ax.axvline(-1, color="#8e44ad", linestyle="--", linewidth=1.4, alpha=0.55, zorder=1)
            ax.text(-1, y_top, "α=−1\n(≈PT level)", ha="center", va="top",
                    fontsize=7, color="#6c3483", fontweight="bold")


# ── A1 ────────────────────────────────────────────────────────────────────────

def plot_A1(a1_dir: Path, a2_dir: Path, out_path: Path,
            a1_notmpl_dir: Path | None = None) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores  = _load_scores(a1_dir / "scores.csv")
    judge   = _load_judge(a1_dir / "llm_judge_scores.jsonl")
    judge_v2 = _load_judge_v2(a1_dir / "llm_judge_v2_scores.jsonl")
    car     = _load_coherent_from_samples(a1_dir / "sample_outputs.jsonl", "GOV-CONV")

    # PT baselines (from A2 merged dir)
    pt_scores   = _load_scores(a2_dir / "scores.csv").get("A2_baseline_pt", {})
    pt_judge    = _load_judge(a2_dir / "llm_judge_scores.jsonl").get("A2_baseline_pt", {})
    pt_judge_v2 = _load_judge_v2(a2_dir / "llm_judge_v2_scores.jsonl").get("A2_baseline_pt", {})
    pt_car      = _load_coherent_from_samples(a2_dir / "sample_outputs.jsonl", "GOV-CONV").get("A2_baseline_pt")
    it_car      = car.get("A1_baseline")

    # Sweep conditions
    sweep = sorted(
        {(a, c) for c in scores for a in [_alpha(c)] if a is not None},
        key=lambda t: t[0],
    )
    if not sweep:
        print(f"No A1 sweep conditions found"); return
    xs   = [a for a, _ in sweep]
    conds = [c for _, c in sweep]

    def gs(bench):
        return [scores.get(c, {}).get(bench) for c in conds]

    def gcar():
        return [car.get(c) for c in conds]

    def gjudge(task):
        return [judge.get(c, {}).get(task) for c in conds]

    def gjudge_v2(task):
        return [judge_v2.get(c, {}).get(task) for c in conds]

    # Baselines: (pt_val, it_val)
    it_b = scores.get("A1_baseline", {})
    it_j_v2 = judge_v2.get("A1_baseline", {})

    gov_baselines = {
        "structural_token_ratio": (pt_scores.get("structural_token_ratio"), it_b.get("structural_token_ratio")),
        "format_compliance_v2":   (pt_scores.get("format_compliance_v2"), it_b.get("format_compliance_v2")),
        "format_compliance":      (pt_scores.get("format_compliance"), it_b.get("format_compliance")),
        "g1":                     (pt_judge_v2.get("g1"), it_j_v2.get("g1")),
        "g2":                     (pt_judge_v2.get("g2"), it_j_v2.get("g2")),
        "coherent_assistant_rate":(pt_car, it_car),
    }
    content_benches = ["mmlu_forced_choice", "mmlu_accuracy", "exp3_factual_em",
                       "exp3_reasoning_em", "reasoning_em"]
    safety_benches  = ["s1", "s2", "exp3_alignment_behavior", "alignment_behavior", "safety"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("A1: IT model — Corrective Direction Ablation (α sweep, 1.0 = baseline)",
                 fontsize=12, fontweight="bold")

    # Panel 1: Governance
    gov_series: dict = {}
    # Primary: G1/G2 from v2 judge (if available)
    g1_vals = gjudge_v2("g1")
    g2_vals = gjudge_v2("g2")
    if any(y is not None for y in g1_vals):
        gov_series["g1"] = g1_vals
    if any(y is not None for y in g2_vals):
        gov_series["g2"] = g2_vals
    # Programmatic
    gov_series["structural_token_ratio"] = gs("structural_token_ratio")
    for b in ("format_compliance_v2", "format_compliance"):
        ys = gs(b)
        if any(y is not None for y in ys):
            gov_series[b] = ys
            break
    # Legacy coherent_assistant (if no G1)
    if "g1" not in gov_series:
        gov_series["coherent_assistant_rate"] = gcar()
    _plot_panel(axes[0], xs, gov_series, gov_baselines,
                "Governance (IT ≫ PT)",
                "α (correction strength)",
                baseline_x=1.0)

    # Panel 2: Content
    content_series = {}
    for b in content_benches:
        ys = gs(b)
        if any(y is not None for y in ys):
            content_series[b] = ys
            break  # show only first available content metric
    content_baselines = {b: (pt_scores.get(b), it_b.get(b)) for b in content_benches}
    _plot_panel(axes[1], xs, content_series, content_baselines,
                "Content (should be stable)",
                "α", baseline_x=1.0)

    # Panel 3: Safety
    safety_series = {}
    it_j  = judge.get("A1_baseline", {})
    for b in safety_benches:
        if b in ("s1", "s2"):
            ys = gjudge_v2(b)
            bline = (pt_judge_v2.get(b), it_j_v2.get(b))
        elif b == "safety":
            ys = gjudge("safety")
            bline = (pt_judge.get("safety"), it_j.get("safety"))
        else:
            ys = gs(b)
            bline = (pt_scores.get(b), it_b.get(b))
        if any(y is not None for y in ys):
            safety_series[b] = ys
            gov_baselines[b] = bline
    _plot_panel(axes[2], xs, safety_series, gov_baselines,
                "Safety / Alignment",
                "α", baseline_x=1.0)

    # ── Optional: overlay IT-notmpl series on each panel ─────────────────────
    if a1_notmpl_dir is not None:
        nt_scores  = _load_scores(a1_notmpl_dir / "scores.csv")
        nt_judge_v2 = _load_judge_v2(a1_notmpl_dir / "llm_judge_v2_scores.jsonl")
        nt_sweep = sorted(
            {(a, c) for c in nt_scores for a in [_A1notmpl_alpha(c)] if a is not None},
            key=lambda t: t[0],
        )
        if nt_sweep:
            nt_xs    = [a for a, _ in nt_sweep]
            nt_conds = [c for _, c in nt_sweep]
            NOTMPL_COLOR = "#e67e22"   # orange

            def _nt_gs(bench):
                return [nt_scores.get(c, {}).get(bench) for c in nt_conds]
            def _nt_gv2(task):
                return [nt_judge_v2.get(c, {}).get(task) for c in nt_conds]

            # Panel 0: Governance
            for bench, getter in [("g1", lambda: _nt_gv2("g1")),
                                   ("g2", lambda: _nt_gv2("g2")),
                                   ("structural_token_ratio", lambda: _nt_gs("structural_token_ratio")),
                                   ("format_compliance_v2",   lambda: _nt_gs("format_compliance_v2"))]:
                ys = getter()
                if any(y is not None for y in ys):
                    clean = [(x, y) for x, y in zip(nt_xs, ys) if y is not None]
                    if clean:
                        cx, cy = zip(*clean)
                        axes[0].plot(cx, cy, marker="s", markersize=3, color=NOTMPL_COLOR,
                                     linestyle="--", linewidth=1.5,
                                     label=f"{_LABELS.get(bench, bench)} (no-tmpl)")

            # Panel 1: Content — first available metric
            for bench in ["mmlu_forced_choice", "mmlu_accuracy", "exp3_reasoning_em"]:
                ys = _nt_gs(bench)
                if any(y is not None for y in ys):
                    clean = [(x, y) for x, y in zip(nt_xs, ys) if y is not None]
                    if clean:
                        cx, cy = zip(*clean)
                        axes[1].plot(cx, cy, marker="s", markersize=3, color=NOTMPL_COLOR,
                                     linestyle="--", linewidth=1.5,
                                     label=f"{_LABELS.get(bench, bench)} (no-tmpl)")
                    break

            # Panel 2: Safety
            for bench, getter in [("s1", lambda: _nt_gv2("s1")),
                                   ("s2", lambda: _nt_gv2("s2")),
                                   ("exp3_alignment_behavior", lambda: _nt_gs("exp3_alignment_behavior"))]:
                ys = getter()
                if any(y is not None for y in ys):
                    clean = [(x, y) for x, y in zip(nt_xs, ys) if y is not None]
                    if clean:
                        cx, cy = zip(*clean)
                        axes[2].plot(cx, cy, marker="s", markersize=3, color=NOTMPL_COLOR,
                                     linestyle="--", linewidth=1.5,
                                     label=f"{_LABELS.get(bench, bench)} (no-tmpl)")

            for ax in axes:
                ax.legend(fontsize=7, loc="best")

    _add_alpha_emphasis(axes, xs)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A1 plot saved → {out_path}")


def plot_A1_pca(a1_dir: Path, repo_root: Path) -> None:
    """Post-hoc PCA on exact per-record IT-PT difference vectors.

    This intentionally skips when only aggregate artifacts are available. It does
    not approximate PCA from layer sums or normalized mean directions.
    """
    mean_direction_path = repo_root / "results" / "exp5" / "precompute_v2" / "precompute" / "corrective_directions.npz"
    out_json = a1_dir / "pca_layer_stats.json"
    out_plot = a1_dir / "plots" / "A1_pca_low_rank.png"
    try:
        json_path, plot_path = compute_and_plot_pca_analysis(
            base_dir=repo_root,
            out_json_path=out_json,
            out_plot_path=out_plot,
            mean_direction_path=mean_direction_path,
        )
        print(f"A1 PCA stats saved → {json_path}")
        print(f"A1 PCA plot saved → {plot_path}")
    except PCAArtifactsUnavailableError as exc:
        print(f"A1 PCA skipped — {exc}")


# ── A2 ────────────────────────────────────────────────────────────────────────

def plot_A2(a2_dir: Path, a1_dir: Path, out_path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores   = _load_scores(a2_dir / "scores.csv")
    judge    = _load_judge(a2_dir / "llm_judge_scores.jsonl")
    judge_v2 = _load_judge_v2(a2_dir / "llm_judge_v2_scores.jsonl")
    car      = _load_coherent_from_samples(a2_dir / "sample_outputs.jsonl", "GOV-CONV")

    # IT/PT baselines
    pt_scores    = scores.get("A2_baseline_pt", {})
    pt_judge_b   = judge.get("A2_baseline_pt", {})
    pt_judge_v2_b= judge_v2.get("A2_baseline_pt", {})
    pt_car_val   = car.get("A2_baseline_pt")

    it_scores_all = _load_scores(a1_dir / "scores.csv")
    it_b = it_scores_all.get("A1_baseline", {})
    it_judge_b  = _load_judge(a1_dir / "llm_judge_scores.jsonl").get("A1_baseline", {})
    it_judge_v2_b = _load_judge_v2(a1_dir / "llm_judge_v2_scores.jsonl").get("A1_baseline", {})
    it_car_val  = _load_coherent_from_samples(a1_dir / "sample_outputs.jsonl", "GOV-CONV").get("A1_baseline")

    # Sweep
    seen: set[str] = set()
    sweep = []
    for c in scores:
        b = _beta(c)
        if b is not None and c not in seen:
            seen.add(c); sweep.append((b, c))
    sweep.sort()
    if not sweep:
        print("No A2 sweep conditions found"); return
    xs    = [b for b, _ in sweep]
    conds = [c for _, c in sweep]

    def gs(bench):
        return [scores.get(c, {}).get(bench) for c in conds]

    def gcar():
        return [car.get(c) for c in conds]

    def gjudge(task):
        return [judge.get(c, {}).get(task) for c in conds]

    def gjudge_v2(task):
        return [judge_v2.get(c, {}).get(task) for c in conds]

    gov_baselines = {
        "g1":                      (pt_judge_v2_b.get("g1"), it_judge_v2_b.get("g1")),
        "g2":                      (pt_judge_v2_b.get("g2"), it_judge_v2_b.get("g2")),
        "coherent_assistant_rate":  (pt_car_val, it_car_val),
        "structural_token_ratio":   (pt_scores.get("structural_token_ratio"), it_b.get("structural_token_ratio")),
        "format_compliance_v2":     (pt_scores.get("format_compliance_v2"), it_b.get("format_compliance_v2")),
        "format_compliance":        (pt_scores.get("format_compliance"), it_b.get("format_compliance")),
    }
    content_benches = ["mmlu_forced_choice", "mmlu_accuracy", "exp3_factual_em",
                       "exp3_reasoning_em", "reasoning_em"]
    safety_benches  = ["s1", "s2", "exp3_alignment_behavior", "alignment_behavior", "safety"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("A2: PT model — Corrective Direction Injection (β sweep, 0 = no injection)",
                 fontsize=12, fontweight="bold")

    gov_series: dict = {}
    g1_vals = gjudge_v2("g1")
    g2_vals = gjudge_v2("g2")
    if any(y is not None for y in g1_vals):
        gov_series["g1"] = g1_vals
    if any(y is not None for y in g2_vals):
        gov_series["g2"] = g2_vals
    gov_series["structural_token_ratio"] = gs("structural_token_ratio")
    for b in ("format_compliance_v2", "format_compliance"):
        ys = gs(b)
        if any(y is not None for y in ys):
            gov_series[b] = ys; break
    if "g1" not in gov_series:
        gov_series["coherent_assistant_rate"] = gcar()
    _plot_panel(axes[0], xs, gov_series, gov_baselines,
                "Governance (PT→IT with β)",
                "β (injection magnitude)", baseline_x=0.0)

    content_series = {}
    for b in content_benches:
        ys = gs(b)
        if any(y is not None for y in ys):
            content_series[b] = ys; break  # first available
    content_baselines = {b: (pt_scores.get(b), it_b.get(b)) for b in content_benches}
    _plot_panel(axes[1], xs, content_series, content_baselines,
                "Content", "β", baseline_x=0.0)

    safety_series = {}
    all_blines = dict(gov_baselines)
    for b in safety_benches:
        if b in ("s1", "s2"):
            ys = gjudge_v2(b)
            all_blines[b] = (pt_judge_v2_b.get(b), it_judge_v2_b.get(b))
        elif b == "safety":
            ys = gjudge("safety")
            all_blines[b] = (pt_judge_b.get("safety"), it_judge_b.get("safety"))
        else:
            ys = gs(b)
            all_blines[b] = (pt_scores.get(b), it_b.get(b))
        if any(y is not None for y in ys):
            safety_series[b] = ys
    _plot_panel(axes[2], xs, safety_series, all_blines,
                "Safety / Alignment", "β", baseline_x=0.0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A2 plot saved → {out_path}")


# ── A1 v5: layer-specificity overlay (A1 / A1_early / A1_mid) ────────────────

def plot_A1_v5(a1_dir: Path, a1_early_dir: Path, a1_mid_dir: Path, out_path: Path) -> None:
    """v5: overlay A1 (20-33), A1_early (0-7), A1_mid (8-19) on same axes.

    Layout: 2 rows × 3 cols
      Row 0: governance benchmarks (coherent_assistant_rate, structural_token_ratio, format_compliance)
      Row 1: content benchmarks + alignment (mmlu_accuracy, exp3_reasoning_em, exp3_alignment_behavior)
    Each subplot shows all 3 layer ranges as separate lines.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LAYER_SETS = [
        ("A1",       a1_dir,       "Corrective 20-33", "#e74c3c", "-",  "o"),
        ("A1_early", a1_early_dir, "Early 1-11",       "#3498db", "--", "s"),
        ("A1_mid",   a1_mid_dir,   "Mid 12-19",        "#2ecc71", ":",  "^"),
    ]
    # condition prefix and alpha parser per experiment
    def _get_alpha(cond: str, prefix: str) -> float | None:
        tag = f"{prefix}_alpha_"
        if not cond.startswith(tag):
            return None
        try:
            return float(cond[len(tag):])
        except ValueError:
            return None

    BENCHMARKS = [
        ("format_compliance_v2",    "Format Compliance v2", "Governance"),
        ("structural_token_ratio",  "Structural Token Ratio", "Governance"),
        ("mmlu_forced_choice",      "MMLU Forced Choice", "Content"),
        ("exp3_reasoning_em",       "Reasoning EM", "Content"),
        ("exp3_alignment_behavior", "Alignment Behavior", "Safety"),
    ]

    # Map experiment prefix to the condition name prefix used in CSV
    EXP_COND_PREFIX = {"A1": "A1", "A1_early": "A1early", "A1_mid": "A1mid"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax_idx, (bench, bench_label, group) in enumerate(BENCHMARKS):
        ax = axes[ax_idx // 3][ax_idx % 3]

        for exp_key, exp_dir, layer_label, color, ls, marker in LAYER_SETS:
            scores = _load_scores(exp_dir / "scores.csv")
            cond_prefix = EXP_COND_PREFIX[exp_key]

            # Collect sweep points
            pts = []
            for cond, bdict in scores.items():
                a = _get_alpha(cond, cond_prefix)
                if a is not None and bench in bdict:
                    pts.append((a, bdict[bench]))
            if not pts:
                continue
            pts.sort()
            xs, ys = zip(*pts)

            # Baseline value (α=1.0 condition)
            bl_cond = f"{cond_prefix}_baseline"
            bl_val = scores.get(bl_cond, {}).get(bench)

            ax.plot(xs, ys, marker=marker, markersize=4, color=color,
                    label=layer_label, linewidth=2.0, linestyle=ls)
            if bl_val is not None:
                ax.axhline(bl_val, color=color, linestyle=ls, linewidth=0.8, alpha=0.35)

        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.35, linewidth=1)
        ax.axvline(0.0, color="lightgray", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_xlabel("α  (1.0 = baseline; <1 removes direction; <0 injects it)", fontsize=8)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(f"{bench_label}  [{group}]", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.2)
        if ax_idx == 0:
            ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "A1 v5: Layer-Specificity — Same α Sweep Applied at 3 Layer Ranges\n"
        "Red=Corrective(20-33)  Blue=Early(1-11)  Green=Mid(12-19)  "
        "Dashed=baseline; α<0 injects corrective direction (amplification)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A1 v5 plot saved → {out_path}")


# ── A5b: IT corrective direction α-sweep (rerun with v2 eval) ─────────────────

def _A1notmpl_alpha(cond: str) -> float | None:
    if not cond.startswith("A1notmpl_alpha_"):
        return None
    try:
        return float(cond[len("A1notmpl_alpha_"):])
    except ValueError:
        return None


def _A5anotmpl_skip(cond: str) -> int | None:
    if not cond.startswith("A5anotmpl_skip_from_"):
        return None
    try:
        return int(cond[len("A5anotmpl_skip_from_"):])
    except ValueError:
        return None


def _A5b_alpha(cond: str) -> float | None:
    if not cond.startswith("A5b_alpha_"):
        return None
    try:
        return float(cond[len("A5b_alpha_"):])
    except ValueError:
        return None


def _A5a_skip(cond: str) -> int | None:
    if not cond.startswith("A5a_skip_from_"):
        return None
    try:
        return int(cond[len("A5a_skip_from_"):])
    except ValueError:
        return None

def _A5aearly_skip(cond: str) -> int | None:
    if not cond.startswith("A5aearly_skip_from_"):
        return None
    try:
        return int(cond[len("A5aearly_skip_from_"):])
    except ValueError:
        return None

def _A5amid_skip(cond: str) -> int | None:
    if not cond.startswith("A5amid_skip_from_"):
        return None
    try:
        return int(cond[len("A5amid_skip_from_"):])
    except ValueError:
        return None
    except ValueError:
        return None


def plot_A5b(a5b_dir: Path, out_path: Path, pt_dir: Path | None = None) -> None:
    """A5b: IT corrective direction α-sweep with v2 eval pipeline."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores   = _load_scores(a5b_dir / "scores.csv")
    judge_v2 = _load_judge_v2(a5b_dir / "llm_judge_v2_scores.jsonl")
    car      = _load_coherent_from_samples(a5b_dir / "sample_outputs.jsonl", "GOV-CONV")

    sweep = sorted(
        {(a, c) for c in scores for a in [_A5b_alpha(c)] if a is not None},
        key=lambda t: t[0],
    )
    if not sweep:
        print("No A5b sweep conditions found"); return
    xs    = [a for a, _ in sweep]
    conds = [c for _, c in sweep]

    it_b     = scores.get("A5b_baseline", {})
    it_j_v2  = judge_v2.get("A5b_baseline", {})
    it_car   = car.get("A5b_baseline")

    # PT baseline
    pt_b    = _load_scores(pt_dir / "scores.csv").get("A2_baseline_pt", {}) if pt_dir else {}
    pt_j_v2 = _load_judge_v2(pt_dir / "llm_judge_v2_scores.jsonl").get("A2_baseline_pt", {}) if pt_dir else {}

    def gs(bench):   return [scores.get(c, {}).get(bench) for c in conds]
    def gv2(task):   return [judge_v2.get(c, {}).get(task) for c in conds]
    def gcar():      return [car.get(c) for c in conds]

    gov_baselines = {
        "structural_token_ratio": (pt_b.get("structural_token_ratio"), it_b.get("structural_token_ratio")),
        "format_compliance_v2":   (pt_b.get("format_compliance_v2"),   it_b.get("format_compliance_v2")),
        "g1": (pt_j_v2.get("g1"), it_j_v2.get("g1")),
        "g2": (pt_j_v2.get("g2"), it_j_v2.get("g2")),
        "coherent_assistant_rate": (None, it_car),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("A5b (rerun): IT model — Corrective Direction α-sweep  [v2 eval]",
                 fontsize=12, fontweight="bold")

    # Panel 1: Governance
    gov_series: dict = {}
    g1_vals = gv2("g1")
    g2_vals = gv2("g2")
    if any(y is not None for y in g1_vals): gov_series["g1"] = g1_vals
    if any(y is not None for y in g2_vals): gov_series["g2"] = g2_vals
    gov_series["structural_token_ratio"] = gs("structural_token_ratio")
    for b in ("format_compliance_v2", "format_compliance"):
        ys = gs(b)
        if any(y is not None for y in ys): gov_series[b] = ys; break
    if "g1" not in gov_series:
        gov_series["coherent_assistant_rate"] = gcar()
    _plot_panel(axes[0], xs, gov_series, gov_baselines,
                "Governance", "α (1.0 = baseline IT)", baseline_x=1.0)

    # Panel 2: Content
    content_benches = ["mmlu_forced_choice", "mmlu_accuracy", "exp3_factual_em",
                       "exp3_reasoning_em", "reasoning_em"]
    content_series = {}
    content_baselines = {b: (pt_b.get(b), it_b.get(b)) for b in content_benches}
    for b in content_benches:
        ys = gs(b)
        if any(y is not None for y in ys):
            content_series[b] = ys
    _plot_panel(axes[1], xs, content_series, content_baselines,
                "Content (should be stable)", "α", baseline_x=1.0)

    # Panel 3: Safety / Alignment
    safety_series = {}
    safety_baselines = dict(gov_baselines)
    for b in ["s1", "s2", "exp3_alignment_behavior", "exp3_reasoning_em"]:
        if b in ("s1", "s2"):
            ys = gv2(b)
            safety_baselines[b] = (pt_j_v2.get(b), it_j_v2.get(b))
        else:
            ys = gs(b)
            safety_baselines[b] = (pt_b.get(b), it_b.get(b))
        if any(y is not None for y in ys):
            safety_series[b] = ys
    _plot_panel(axes[2], xs, safety_series, safety_baselines,
                "Safety / Alignment", "α", baseline_x=1.0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A5b plot saved → {out_path}")


def plot_A5a(a5a_dir: Path, out_path: Path, pt_dir: Path | None = None) -> None:
    """A5a: IT progressive skip (skip from layer X to 33) with v2 eval pipeline."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores   = _load_scores(a5a_dir / "scores.csv")
    judge_v2 = _load_judge_v2(a5a_dir / "llm_judge_v2_scores.jsonl")

    sweep = sorted(
        {(k, c) for c in scores for k in [_A5a_skip(c)] if k is not None},
        key=lambda t: t[0],
    )
    if not sweep:
        print("No A5a sweep conditions found"); return
    xs    = [k for k, _ in sweep]   # layer index: 20→33 (fewer skipped = higher x)
    conds = [c for _, c in sweep]

    it_b    = scores.get("A5a_baseline", {})
    it_j_v2 = judge_v2.get("A5a_baseline", {})

    # PT baseline
    pt_b    = _load_scores(pt_dir / "scores.csv").get("A2_baseline_pt", {}) if pt_dir else {}
    pt_j_v2 = _load_judge_v2(pt_dir / "llm_judge_v2_scores.jsonl").get("A2_baseline_pt", {}) if pt_dir else {}

    def gs(bench): return [scores.get(c, {}).get(bench) for c in conds]
    def gv2(task): return [judge_v2.get(c, {}).get(task) for c in conds]

    baselines = {
        "structural_token_ratio": (pt_b.get("structural_token_ratio"), it_b.get("structural_token_ratio")),
        "format_compliance_v2":   (pt_b.get("format_compliance_v2"),   it_b.get("format_compliance_v2")),
        "g1": (pt_j_v2.get("g1"), it_j_v2.get("g1")),
        "g2": (pt_j_v2.get("g2"), it_j_v2.get("g2")),
        "mmlu_forced_choice":      (pt_b.get("mmlu_forced_choice"),      it_b.get("mmlu_forced_choice")),
        "exp3_reasoning_em":       (pt_b.get("exp3_reasoning_em"),       it_b.get("exp3_reasoning_em")),
        "exp3_alignment_behavior": (pt_b.get("exp3_alignment_behavior"), it_b.get("exp3_alignment_behavior")),
        "s1": (pt_j_v2.get("s1"), it_j_v2.get("s1")),
        "s2": (pt_j_v2.get("s2"), it_j_v2.get("s2")),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("A5a (rerun): IT model — Progressive Skip (layers X→33)  [v2 eval]\n"
                 "x=first skipped layer; higher x = fewer layers skipped = closer to baseline",
                 fontsize=11, fontweight="bold")

    # Panel 1: Governance
    gov_series: dict = {}
    for b in ("g1", "g2"):
        ys = gv2(b)
        if any(y is not None for y in ys): gov_series[b] = ys
    gov_series["structural_token_ratio"] = gs("structural_token_ratio")
    for b in ("format_compliance_v2", "format_compliance"):
        ys = gs(b)
        if any(y is not None for y in ys): gov_series[b] = ys; break
    _plot_panel(axes[0], xs, gov_series, baselines,
                "Governance", "First skipped layer (33=skip only L33)", baseline_x=34.0)

    # Panel 2: Content
    content_series = {}
    for b in ["mmlu_forced_choice", "mmlu_accuracy", "exp3_factual_em", "exp3_reasoning_em"]:
        ys = gs(b)
        if any(y is not None for y in ys): content_series[b] = ys
    _plot_panel(axes[1], xs, content_series, baselines,
                "Content", "First skipped layer", baseline_x=34.0)

    # Panel 3: Safety
    safety_series = {}
    for b in ["s1", "s2"]:
        ys = gv2(b)
        if any(y is not None for y in ys): safety_series[b] = ys
    for b in ["exp3_alignment_behavior"]:
        ys = gs(b)
        if any(y is not None for y in ys): safety_series[b] = ys
    _plot_panel(axes[2], xs, safety_series, baselines,
                "Safety / Alignment", "First skipped layer", baseline_x=34.0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A5a plot saved → {out_path}")


def plot_A1_notmpl_vs_pt(a1_notmpl_dir: Path, a2_dir: Path, out_path: Path) -> None:
    """IT no-template α-sweep vs PT baseline.

    Same structure as plot_A1() but IT-notmpl sweep is the primary series.
    Answers: 'Without the chat template, does IT still outperform PT on governance?'
    If yes → weight-intrinsic. If no → template-gated.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores   = _load_scores(a1_notmpl_dir / "scores.csv")
    judge_v2 = _load_judge_v2(a1_notmpl_dir / "llm_judge_v2_scores.jsonl")

    pt_scores   = _load_scores(a2_dir / "scores.csv").get("A2_baseline_pt", {})
    pt_judge_v2 = _load_judge_v2(a2_dir / "llm_judge_v2_scores.jsonl").get("A2_baseline_pt", {})

    sweep = sorted(
        {(a, c) for c in scores for a in [_A1notmpl_alpha(c)] if a is not None},
        key=lambda t: t[0],
    )
    if not sweep:
        print("No A1_notmpl sweep conditions found"); return
    xs    = [a for a, _ in sweep]
    conds = [c for _, c in sweep]

    it_b    = scores.get("A1notmpl_baseline", {})
    it_j_v2 = judge_v2.get("A1notmpl_baseline", {})

    def gs(bench): return [scores.get(c, {}).get(bench) for c in conds]
    def gv2(task): return [judge_v2.get(c, {}).get(task) for c in conds]

    baselines = {
        "g1":                     (pt_judge_v2.get("g1"),                it_j_v2.get("g1")),
        "g2":                     (pt_judge_v2.get("g2"),                it_j_v2.get("g2")),
        "structural_token_ratio": (pt_scores.get("structural_token_ratio"), it_b.get("structural_token_ratio")),
        "format_compliance_v2":   (pt_scores.get("format_compliance_v2"),   it_b.get("format_compliance_v2")),
        "mmlu_forced_choice":     (pt_scores.get("mmlu_forced_choice"),     it_b.get("mmlu_forced_choice")),
        "exp3_reasoning_em":      (pt_scores.get("exp3_reasoning_em"),      it_b.get("exp3_reasoning_em")),
        "s1": (pt_judge_v2.get("s1"), it_j_v2.get("s1")),
        "s2": (pt_judge_v2.get("s2"), it_j_v2.get("s2")),
        "exp3_alignment_behavior":(pt_scores.get("exp3_alignment_behavior"), it_b.get("exp3_alignment_behavior")),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "A1_notmpl: IT no-template α sweep — Template-Gate Test (1.0 = baseline)\n"
        "Colored lines = IT no-template sweep  |  dotted = PT baseline  |  dash-dot = IT-with-template baseline",
        fontsize=11, fontweight="bold",
    )

    gov_series: dict = {}
    for task in ("g1", "g2"):
        ys = gv2(task)
        if any(y is not None for y in ys): gov_series[task] = ys
    gov_series["structural_token_ratio"] = gs("structural_token_ratio")
    for b in ("format_compliance_v2", "format_compliance"):
        ys = gs(b)
        if any(y is not None for y in ys): gov_series[b] = ys; break
    _plot_panel(axes[0], xs, gov_series, baselines, "Governance (IT ≫ PT?)",
                "α (correction strength)", baseline_x=1.0)

    content_series: dict = {}
    for b in ["mmlu_forced_choice", "mmlu_accuracy", "exp3_reasoning_em"]:
        ys = gs(b)
        if any(y is not None for y in ys): content_series[b] = ys; break
    _plot_panel(axes[1], xs, content_series, baselines, "Content (should be stable)",
                "α", baseline_x=1.0)

    safety_series: dict = {}
    for task in ("s1", "s2"):
        ys = gv2(task)
        if any(y is not None for y in ys): safety_series[task] = ys
    for b in ("exp3_alignment_behavior", "alignment_behavior"):
        ys = gs(b)
        if any(y is not None for y in ys): safety_series[b] = ys; break
    _plot_panel(axes[2], xs, safety_series, baselines, "Safety / Alignment",
                "α", baseline_x=1.0)

    _add_alpha_emphasis(axes, xs)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A1_notmpl_vs_pt plot saved → {out_path}")


def plot_A1_combined(a1_dir: Path, a1_notmpl_dir: Path, a2_dir: Path, out_path: Path) -> None:
    """Side-by-side overlay of A1 (with template) and A1_notmpl (no template).

    Same color per metric; solid = with-template, dashed = no-template.
    Lets you directly compare whether removing the chat template changes the dose-response shape.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores    = _load_scores(a1_dir / "scores.csv")
    judge_v2  = _load_judge_v2(a1_dir / "llm_judge_v2_scores.jsonl")
    nt_scores = _load_scores(a1_notmpl_dir / "scores.csv")
    nt_j_v2   = _load_judge_v2(a1_notmpl_dir / "llm_judge_v2_scores.jsonl")

    pt_scores   = _load_scores(a2_dir / "scores.csv").get("A2_baseline_pt", {})
    pt_judge_v2 = _load_judge_v2(a2_dir / "llm_judge_v2_scores.jsonl").get("A2_baseline_pt", {})

    sweep = sorted(
        {(a, c) for c in scores for a in [_alpha(c)] if a is not None},
        key=lambda t: t[0],
    )
    nt_sweep = sorted(
        {(a, c) for c in nt_scores for a in [_A1notmpl_alpha(c)] if a is not None},
        key=lambda t: t[0],
    )
    if not sweep or not nt_sweep:
        print("Missing sweep data for A1_combined"); return

    xs    = [a for a, _ in sweep]
    conds = [c for _, c in sweep]
    nt_xs    = [a for a, _ in nt_sweep]
    nt_conds = [c for _, c in nt_sweep]

    it_b    = scores.get("A1_baseline", {})
    it_j_v2_b = judge_v2.get("A1_baseline", {})
    nt_b    = nt_scores.get("A1notmpl_baseline", {})
    nt_j_b  = nt_j_v2.get("A1notmpl_baseline", {})

    # metric → (color, label)
    METRICS = [
        ("g1",                     "#2166ac", "G1 governance quality"),
        ("g2",                     "#4dac26", "G2 register compliance"),
        ("structural_token_ratio", "#d6604d", "structural_token_ratio"),
        ("format_compliance_v2",   "#762a83", "format_compliance_v2"),
    ]
    CONTENT_METRICS = [("mmlu_forced_choice", "#b35806", "mmlu_forced_choice")]
    SAFETY_METRICS  = [
        ("s1",                    "#1a9850", "S1 safety refuse rate"),
        ("s2",                    "#d73027", "S2 false-refusal rate"),
        ("exp3_alignment_behavior","#636363","alignment_behavior"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(
        "A1: With-template vs No-template — Direct Comparison (α sweep, 1.0 = baseline)\n"
        "Solid = IT with chat template  |  Dashed = IT no chat template  |  Dotted = PT baseline",
        fontsize=11, fontweight="bold",
    )

    def _plot_metric(ax, xs_t, conds_t, sc_t, jv_t, xs_nt, conds_nt, sc_nt, jv_nt,
                     key, color, label, pt_val, it_val, nt_val):
        def _get(sc, jv, clist, k):
            if k in ("g1", "g2", "s1", "s2"):
                return [jv.get(c, {}).get(k) for c in clist]
            return [sc.get(c, {}).get(k) for c in clist]

        ys    = _get(sc_t, jv_t, conds_t, key)
        ys_nt = _get(sc_nt, jv_nt, conds_nt, key)

        if any(y is not None for y in ys):
            clean = [(x, y) for x, y in zip(xs_t, ys) if y is not None]
            if clean:
                cx, cy = zip(*clean)
                ax.plot(cx, cy, marker="o", markersize=4, color=color, linewidth=1.8,
                        linestyle="-", label=f"{label} (with-tmpl)")
        if any(y is not None for y in ys_nt):
            clean = [(x, y) for x, y in zip(xs_nt, ys_nt) if y is not None]
            if clean:
                cx, cy = zip(*clean)
                ax.plot(cx, cy, marker="s", markersize=4, color=color, linewidth=1.8,
                        linestyle="--", label=f"{label} (no-tmpl)")
        if pt_val is not None:
            ax.axhline(pt_val, color=color, linestyle=":", linewidth=1.0, alpha=0.7)
        if it_val is not None:
            ax.axvline(x=1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    for key, color, label in METRICS:
        pt_v  = pt_judge_v2.get(key) if key in ("g1","g2") else pt_scores.get(key)
        it_v  = it_j_v2_b.get(key)   if key in ("g1","g2") else it_b.get(key)
        nt_v  = nt_j_b.get(key)      if key in ("g1","g2") else nt_b.get(key)
        _plot_metric(axes[0], xs, conds, scores, judge_v2,
                     nt_xs, nt_conds, nt_scores, nt_j_v2,
                     key, color, label, pt_v, it_v, nt_v)
    axes[0].axvline(x=1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[0].set_title("Governance"); axes[0].set_xlabel("α (correction strength)")
    axes[0].set_ylabel("Score (0-1)"); axes[0].legend(fontsize=7); axes[0].set_ylim(0, 1.05)

    for key, color, label in CONTENT_METRICS:
        pt_v = pt_scores.get(key); it_v = it_b.get(key); nt_v = nt_b.get(key)
        _plot_metric(axes[1], xs, conds, scores, judge_v2,
                     nt_xs, nt_conds, nt_scores, nt_j_v2,
                     key, color, label, pt_v, it_v, nt_v)
    axes[1].axvline(x=1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Content (should be stable)"); axes[1].set_xlabel("α")
    axes[1].legend(fontsize=7); axes[1].set_ylim(0, 1.05)

    for key, color, label in SAFETY_METRICS:
        pt_v  = pt_judge_v2.get(key) if key in ("s1","s2") else pt_scores.get(key)
        it_v  = it_j_v2_b.get(key)   if key in ("s1","s2") else it_b.get(key)
        nt_v  = nt_j_b.get(key)      if key in ("s1","s2") else nt_b.get(key)
        _plot_metric(axes[2], xs, conds, scores, judge_v2,
                     nt_xs, nt_conds, nt_scores, nt_j_v2,
                     key, color, label, pt_v, it_v, nt_v)
    axes[2].axvline(x=1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[2].set_title("Safety / Alignment"); axes[2].set_xlabel("α")
    axes[2].legend(fontsize=7); axes[2].set_ylim(0, 1.05)

    _add_alpha_emphasis(axes, xs)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A1_combined plot saved → {out_path}")


def plot_A5a_notmpl(a5a_dir: Path, a5a_notmpl_dir: Path, out_path: Path) -> None:
    """Overlay A5a (with template) vs A5a_notmpl (no template) progressive skip.

    Tests whether corrective-layer specialization persists without chat template.
    X-axis: first skipped layer (higher = fewer layers skipped = closer to baseline).
    Two series per panel: green solid = with template, orange dashed = no template.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc_t  = _load_scores(a5a_dir / "scores.csv")
    jv_t  = _load_judge_v2(a5a_dir / "llm_judge_v2_scores.jsonl")
    sc_nt = _load_scores(a5a_notmpl_dir / "scores.csv")
    jv_nt = _load_judge_v2(a5a_notmpl_dir / "llm_judge_v2_scores.jsonl")

    def _sweep(sc, parser, prefix):
        pairs = sorted(
            {(k, c) for c in sc for k in [parser(c)] if k is not None},
            key=lambda t: t[0],
        )
        return [k for k, _ in pairs], [c for _, c in pairs]

    xs_t,  conds_t  = _sweep(sc_t,  _A5a_skip,      "A5a_skip_from_")
    xs_nt, conds_nt = _sweep(sc_nt, _A5anotmpl_skip, "A5anotmpl_skip_from_")

    if not xs_t and not xs_nt:
        print("No A5a_notmpl sweep conditions found"); return

    b_t  = sc_t.get("A5a_baseline",      {})
    b_nt = sc_nt.get("A5anotmpl_baseline", {})
    jb_t  = jv_t.get("A5a_baseline",      {})
    jb_nt = jv_nt.get("A5anotmpl_baseline", {})

    TMPL_COLOR  = "#2ecc71"   # green
    NOTMPL_COLOR = "#e67e22"  # orange

    def _draw(ax, xs, sc, jv, conds, color, ls, suffix):
        for bench, use_judge in [("g1", True), ("g2", True),
                                  ("structural_token_ratio", False),
                                  ("format_compliance_v2", False)]:
            if use_judge:
                ys = [jv.get(c, {}).get(bench) for c in conds]
            else:
                ys = [sc.get(c, {}).get(bench) for c in conds]
            if not any(y is not None for y in ys): continue
            clean = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if clean:
                cx, cy = zip(*clean)
                ax.plot(cx, cy, marker="o", markersize=4, color=color, linestyle=ls,
                        linewidth=1.5, label=f"{_LABELS.get(bench, bench)} {suffix}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "A5a Progressive Skip — Template vs No-Template  [v2 eval]\n"
        "x=first skipped layer; higher x = fewer layers skipped = closer to baseline\n"
        "Green=with template  |  Orange dashed=no template",
        fontsize=10, fontweight="bold",
    )

    PANELS = [
        ("Governance", [("g1", True), ("g2", True),
                        ("structural_token_ratio", False), ("format_compliance_v2", False)]),
        ("Content (should be stable)", [("mmlu_forced_choice", False), ("exp3_reasoning_em", False)]),
        ("Safety / Alignment",         [("s1", True), ("s2", True),
                                        ("exp3_alignment_behavior", False)]),
    ]
    for ax, (title, benches) in zip(axes, PANELS):
        for bench, use_judge in benches:
            for xs, sc, jv, conds, color, ls, sfx in [
                (xs_t,  sc_t,  jv_t,  conds_t,  TMPL_COLOR,   "-",  "(tmpl)"),
                (xs_nt, sc_nt, jv_nt, conds_nt, NOTMPL_COLOR, "--", "(no-tmpl)"),
            ]:
                if not xs: continue
                ys = ([jv.get(c, {}).get(bench) for c in conds] if use_judge
                      else [sc.get(c, {}).get(bench) for c in conds])
                if not any(y is not None for y in ys): continue
                clean = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if not clean: continue
                cx, cy = zip(*clean)
                ax.plot(cx, cy, marker="o", markersize=4, color=color, linestyle=ls,
                        linewidth=1.5, label=f"{_LABELS.get(bench, bench)} {sfx}")

        # Baseline reference lines
        ax.axvline(x=34.0, color="gray", linestyle="--", alpha=0.4, label="baseline")
        ax.set_xlabel("First skipped layer (34=baseline)")
        ax.set_ylabel("Score (0-1)")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A5a_notmpl plot saved → {out_path}")


def plot_A5a_layerspec(
    a5a_early_dir: Path,
    a5a_mid_dir: Path,
    a5a_corr_dir: Path,
    out_path: Path,
) -> None:
    """Three-way progressive-skip layer specificity comparison.

    Overlays Early (1–11), Mid (12–19), and Corrective (20–33) progressive-skip
    sweeps on the same axes.  X-axis = number of layers skipped from the end of
    each range (0 = baseline, max = full range), normalising across range sizes.

    Expected:
    - Governance panel: only Corrective skip degrades G1/G2/STR.
    - Content panel: Early and Mid skip degrade MMLU/reasoning.
    - Safety: minimal effect across all three ranges.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _load(d: Path):
        return _load_scores(d / "scores.csv"), _load_judge_v2(d / "llm_judge_v2_scores.jsonl")

    sc_e, jv_e = _load(a5a_early_dir)
    sc_m, jv_m = _load(a5a_mid_dir)
    sc_c, jv_c = _load(a5a_corr_dir)

    # Build (n_skipped, cond) pairs for each range
    # n_skipped = range_end - first_skipped_layer + 1
    def _sweep_early():
        pairs = sorted(
            {(12 - k, c) for c in sc_e for k in [_A5aearly_skip(c)] if k is not None},
            key=lambda t: t[0],
        )
        return [n for n, _ in pairs], [c for _, c in pairs]

    def _sweep_mid():
        pairs = sorted(
            {(20 - k, c) for c in sc_m for k in [_A5amid_skip(c)] if k is not None},
            key=lambda t: t[0],
        )
        return [n for n, _ in pairs], [c for _, c in pairs]

    def _sweep_corr():
        pairs = sorted(
            {(34 - k, c) for c in sc_c for k in [_A5a_skip(c)] if k is not None},
            key=lambda t: t[0],
        )
        return [n for n, _ in pairs], [c for _, c in pairs]

    xs_e, conds_e = _sweep_early()
    xs_m, conds_m = _sweep_mid()
    xs_c, conds_c = _sweep_corr()

    if not (xs_e or xs_m or xs_c):
        print("No A5a_layerspec sweep conditions found"); return

    # Baselines (use corrective IT baseline as reference)
    b_e = sc_e.get("A5aearly_baseline", {})
    b_m = sc_m.get("A5amid_baseline", {})
    b_c = sc_c.get("A5a_baseline", {})
    jb_e = jv_e.get("A5aearly_baseline", {})
    jb_m = jv_m.get("A5amid_baseline", {})
    jb_c = jv_c.get("A5a_baseline", {})

    # Color = benchmark (track one metric across all layer ranges)
    # Linestyle + marker = layer range (the comparison of interest)
    RANGE_STYLE = {
        "early":      dict(linestyle=":",  marker="^", linewidth=1.8, markersize=6),
        "mid":        dict(linestyle="--", marker="s", linewidth=2.0, markersize=6),
        "corrective": dict(linestyle="-",  marker="o", linewidth=2.5, markersize=7),
    }
    RANGE_LABELS = {
        "early":      "Early (layers 1–11)",
        "mid":        "Mid (layers 12–19)",
        "corrective": "Corrective (layers 20–33)",
    }

    PANELS = [
        # (title, [(bench, short_label, use_judge, color)])
        ("Governance", [
            ("g1",                    "G1 gov. quality",   True,  "#1565C0"),  # dark blue
            ("g2",                    "G2 register qual.", True,  "#6A1B9A"),  # purple
            ("structural_token_ratio","struct_token_ratio", False, "#E65100"),  # deep orange
            ("format_compliance_v2",  "format_compliance", False, "#00695C"),  # teal
        ]),
        ("Content (should be stable for corrective)", [
            ("mmlu_forced_choice",    "MMLU",         False, "#1565C0"),  # blue
            ("exp3_reasoning_em",     "reasoning_em", False, "#B71C1C"),  # dark red
        ]),
        ("Safety / Alignment", [
            ("s1",                     "S1 safety refuse", True,  "#1565C0"),  # blue
            ("s2",                     "S2 false-refusal", True,  "#B71C1C"),  # dark red
            ("exp3_alignment_behavior","alignment_behav.", False, "#2E7D32"),  # dark green
        ]),
    ]

    import matplotlib.lines as mlines

    def _series(sc, jv, conds, bench, use_judge=False):
        if use_judge:
            return [jv.get(c, {}).get(bench) for c in conds]
        return [sc.get(c, {}).get(bench) for c in conds]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        "Progressive Skip — Layer Specificity  [v2 eval]\n"
        "x = number of layers skipped from end of range  (0 = baseline, higher = more ablation)\n"
        "Color = benchmark  |  Linestyle + Marker = layer range"
        "  ( ─●  corrective  · · ▲  early  - -■  mid )",
        fontsize=10, fontweight="bold",
    )

    for ax, (title, benches) in zip(axes, PANELS):
        bench_handles = []
        for bench, short_label, use_judge, color in benches:
            plotted_any = False
            for rng, (sc, jv, conds, xs) in [
                ("early",      (sc_e, jv_e, conds_e, xs_e)),
                ("mid",        (sc_m, jv_m, conds_m, xs_m)),
                ("corrective", (sc_c, jv_c, conds_c, xs_c)),
            ]:
                ys = _series(sc, jv, conds, bench, use_judge)
                if not any(y is not None for y in ys):
                    continue
                ys_clean = [y if y is not None else float("nan") for y in ys]
                ax.plot(xs, ys_clean, color=color, **RANGE_STYLE[rng],
                        alpha=0.85, label="_nolegend_")
                plotted_any = True
            if plotted_any:
                bench_handles.append(
                    mlines.Line2D([], [], color=color, linestyle="-",
                                  linewidth=2, label=short_label)
                )

        # Legend 1: benchmarks (color-coded solid lines, upper right)
        leg1 = ax.legend(handles=bench_handles, fontsize=8, loc="upper right",
                         title="Benchmark", title_fontsize=8,
                         framealpha=0.9, edgecolor="0.7")
        ax.add_artist(leg1)

        # Legend 2: layer ranges (linestyle+marker, lower left)
        range_handles = [
            mlines.Line2D([], [], color="black", label=RANGE_LABELS[r],
                          **RANGE_STYLE[r])
            for r in ("corrective", "mid", "early")
        ]
        range_handles.append(
            mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1,
                          alpha=0.6, label="baseline (0 skipped)")
        )
        ax.legend(handles=range_handles, fontsize=8, loc="lower left",
                  title="Layer range", title_fontsize=8,
                  framealpha=0.9, edgecolor="0.7")

        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_xlabel("# layers skipped (from end of range)", fontsize=9)
        ax.set_ylabel("Score (0–1)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A5a_layerspec plot saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def plot_A1_rand(a1_dir: Path, a1_rand_dir: Path, out_path: Path) -> None:
    """Overlay A1 (corrective direction) vs A1_rand (random direction) α-sweeps.

    Each subplot shows one benchmark with two lines:
      - Red solid:  A1  (actual IT-PT corrective direction)
      - Blue dashed: A1_rand (random unit vector, same layers and α values)

    If the governance dose-response in A1 is direction-specific, A1_rand should
    be flat (close to the baseline dashed line) across all α values.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BENCHMARKS = [
        ("structural_token_ratio",  "Structural Token Ratio", "Governance"),
        ("format_compliance_v2",    "Format Compliance v2",   "Governance"),
        ("mmlu_forced_choice",      "MMLU Forced Choice",     "Content"),
        ("exp3_reasoning_em",       "Reasoning EM",           "Content"),
        ("exp3_alignment_behavior", "Alignment Behavior",     "Safety"),
    ]
    judge_tasks = [
        ("g1", "G1 governance quality (/5)", "Governance"),
        ("g2", "G2 register quality (/5)",   "Governance"),
        ("s1", "S1 safety refuse rate",      "Safety"),
    ]

    def _get_sweep(scores_csv: Path, judge_jsonl: Path, cond_prefix: str):
        """Return {alpha: {bench: value}} for the α-sweep conditions."""
        scores = _load_scores(scores_csv)
        judge  = _load_judge_v2(judge_jsonl)
        pts: dict[float, dict[str, float]] = {}
        for cond, bdict in scores.items():
            tag = f"{cond_prefix}_alpha_"
            if cond.startswith(tag):
                try:
                    a = float(cond[len(tag):])
                except ValueError:
                    continue
                pts[a] = dict(bdict)
                if cond in judge:
                    pts[a].update(judge[cond])
        return dict(sorted(pts.items()))

    all_benches = [(b, lbl, grp) for b, lbl, grp in BENCHMARKS] + \
                  [(t, lbl, grp) for t, lbl, grp in judge_tasks]
    n_cols = 4
    n_rows = (len(all_benches) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else list(axes)

    a1_sweep   = _get_sweep(a1_dir    / "scores.csv", a1_dir    / "llm_judge_v2_scores.jsonl", "A1")
    rand_sweep = _get_sweep(a1_rand_dir / "scores.csv", a1_rand_dir / "llm_judge_v2_scores.jsonl", "A1rand")

    a1_base   = _load_scores(a1_dir    / "scores.csv").get("A1_baseline", {})
    rand_base = _load_scores(a1_rand_dir / "scores.csv").get("A1rand_baseline", {})
    a1_jbase   = _load_judge_v2(a1_dir    / "llm_judge_v2_scores.jsonl").get("A1_baseline", {})
    rand_jbase = _load_judge_v2(a1_rand_dir / "llm_judge_v2_scores.jsonl").get("A1rand_baseline", {})

    for ax_idx, (bench, bench_label, group) in enumerate(all_benches):
        ax = axes_flat[ax_idx]

        for sweep, label, color, ls in [
            (a1_sweep,   "A1 corrective dir", "#e74c3c", "-"),
            (rand_sweep, "A1_rand random dir", "#3498db", "--"),
        ]:
            xs = sorted(sweep.keys())
            ys = [sweep[a].get(bench) for a in xs]
            clean = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if clean:
                cx, cy = zip(*clean)
                ax.plot(cx, cy, marker="o", markersize=3, color=color,
                        label=label, linewidth=2.0, linestyle=ls)

        # Baseline reference lines
        bl_a1  = (a1_base if bench in a1_base else a1_jbase).get(bench)
        bl_rand = (rand_base if bench in rand_base else rand_jbase).get(bench)
        if bl_a1 is not None:
            ax.axhline(bl_a1,  color="#e74c3c", linestyle=":", linewidth=1, alpha=0.5,
                       label=f"A1 baseline={bl_a1:.3f}")
        if bl_rand is not None:
            ax.axhline(bl_rand, color="#3498db", linestyle=":", linewidth=1, alpha=0.5,
                       label=f"rand baseline={bl_rand:.3f}")

        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.35, linewidth=1)
        ax.set_xlabel("α  (1.0 = baseline)", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_title(f"{bench_label}  [{group}]", fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.2)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide unused axes
    for ax in axes_flat[len(all_benches):]:
        ax.set_visible(False)

    fig.suptitle(
        "A1 vs A1_rand: Specificity Control\n"
        "Red=corrective direction (IT-PT)  Blue=random unit vector (seed=42)\n"
        "Flat blue line → effect is direction-specific ✓",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A1_rand comparison plot saved → {out_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["A1", "A2", "both", "A1v5", "A1rand", "A1_notmpl", "A1_combined", "A5a", "A5a_notmpl", "A5b", "A5", "A5a_layerspec"], default="both")
    p.add_argument("--a1-dir",      default="results/exp06_corrective_direction_steering/merged_A1_it")
    p.add_argument("--a2-dir",      default="results/exp06_corrective_direction_steering/merged_A2_pt")
    p.add_argument("--a1-early-dir", default="results/exp06_corrective_direction_steering/merged_A1_early_it_v4")
    p.add_argument("--a1-mid-dir",   default="results/exp06_corrective_direction_steering/merged_A1_mid_it_v4")
    p.add_argument("--a1-rand-dir",  default="results/exp06_corrective_direction_steering/merged_A1_rand_it_v1")
    p.add_argument("--a5a-dir",          default="results/exp06_corrective_direction_steering/merged_A5a_it_v1")
    p.add_argument("--a5a-early-dir",    default="results/exp06_corrective_direction_steering/merged_A5a_early_it_v1")
    p.add_argument("--a5a-mid-dir",      default="results/exp06_corrective_direction_steering/merged_A5a_mid_it_v1")
    p.add_argument("--a5a-notmpl-dir",   default="results/exp06_corrective_direction_steering/merged_A5a_notmpl_it_v1")
    p.add_argument("--a1-notmpl-dir",    default="results/exp06_corrective_direction_steering/merged_A1_notmpl_it_v1")
    p.add_argument("--a5b-dir",     default="results/exp06_corrective_direction_steering/merged_A5b_it_v1")
    p.add_argument("--pt-dir",      default="results/exp06_corrective_direction_steering/merged_A2_pt_v4")
    args = p.parse_args()

    a1          = Path(args.a1_dir)
    a2          = Path(args.a2_dir)
    a1_early    = Path(args.a1_early_dir)
    a1_mid      = Path(args.a1_mid_dir)
    a1_rand     = Path(args.a1_rand_dir)
    a1_notmpl   = Path(args.a1_notmpl_dir)
    a5a         = Path(args.a5a_dir)
    a5a_early   = Path(args.a5a_early_dir)
    a5a_mid     = Path(args.a5a_mid_dir)
    a5a_notmpl  = Path(args.a5a_notmpl_dir)
    a5b         = Path(args.a5b_dir)
    pt          = Path(args.pt_dir) if args.pt_dir else None

    # Pass notmpl dir to plot_A1 only if the merged dir actually has scores
    _a1_notmpl_overlay = a1_notmpl if (a1_notmpl / "scores.csv").exists() else None

    if args.experiment in ("A1", "both"):
        plot_A1(a1, a2, a1 / "plots" / "A1_dose_response_v5.png",
                a1_notmpl_dir=_a1_notmpl_overlay)
        plot_A1_pca(a1, repo_root)
    if args.experiment in ("A2", "both"):
        plot_A2(a2, a1, a2 / "plots" / "A2_dose_response_v5.png")
    if args.experiment in ("A1v5", "both", "A1"):
        plot_A1_v5(a1, a1_early, a1_mid, a1 / "plots" / "A1_layer_specificity_v5.png")
    if args.experiment == "A1rand":
        plot_A1_rand(a1, a1_rand, a1_rand / "plots" / "A1_rand_vs_A1.png")
    if args.experiment in ("A5b", "A5"):
        plot_A5b(a5b, a5b / "plots" / "A5b_dose_response.png", pt_dir=pt)
    if args.experiment in ("A5a", "A5"):
        plot_A5a(a5a, a5a / "plots" / "A5a_progressive_skip.png", pt_dir=pt)
    if args.experiment == "A1_notmpl":
        plot_A1_notmpl_vs_pt(a1_notmpl, a2, a1_notmpl / "plots" / "A1_notmpl_vs_pt.png")
    if args.experiment == "A1_combined":
        plot_A1_combined(a1, a1_notmpl, a2, a1_notmpl / "plots" / "A1_combined_tmpl_vs_notmpl.png")
    if args.experiment == "A5a_notmpl":
        out = Path("results/exp06_corrective_direction_steering/plots_notmpl") / "A5a_notmpl_vs_template.png"
        plot_A5a_notmpl(a5a, a5a_notmpl, out)
    if args.experiment == "A5a_layerspec":
        out = Path("results/exp06_corrective_direction_steering/plots_layerspec") / "A5a_layer_specificity.png"
        plot_A5a_layerspec(a5a_early, a5a_mid, a5a, out)


if __name__ == "__main__":
    main()
