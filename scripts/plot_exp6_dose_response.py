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
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    """Returns {condition: {task: mean_score_0_1}}."""
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
    "coherent_assistant_rate": "#1f77b4",   # blue  — primary governance
    "structural_token_ratio":   "#ff7f0e",  # orange
    "format_compliance":        "#2ca02c",  # green
    "mmlu_accuracy":            "#d62728",  # red
    "exp3_factual_em":          "#d62728",
    "exp3_reasoning_em":        "#9467bd",  # purple
    "reasoning_em":             "#9467bd",
    "exp3_alignment_behavior":  "#8c564b",  # brown
    "alignment_behavior":       "#8c564b",
    "safety":                   "#7f7f7f",  # gray
}

_LABELS = {
    "coherent_assistant_rate": "coherent_assistant (GOV-CONV)",
    "structural_token_ratio":   "structural_token_ratio",
    "format_compliance":        "format_compliance (IFEval)",
    "mmlu_accuracy":            "mmlu_accuracy",
    "exp3_factual_em":          "factual_em",
    "exp3_reasoning_em":        "reasoning_em",
    "reasoning_em":             "reasoning_em",
    "exp3_alignment_behavior":  "alignment_behavior",
    "alignment_behavior":       "alignment_behavior",
    "safety":                   "safety (judge, 0-3→1)",
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
                ax.axhline(pt_val, color=color, linestyle=":", linewidth=1.2, alpha=0.6,
                           label=f"{label} PT={pt_val:.2f}")
                ax.annotate(f"PT {pt_val:.2f}", xy=(x_right, pt_val),
                            fontsize=6, color=color, alpha=0.8, va="center",
                            ha="left", clip_on=True)
            if it_val is not None:
                ax.axhline(it_val, color=color, linestyle="-.", linewidth=1.2, alpha=0.6,
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


# ── A1 ────────────────────────────────────────────────────────────────────────

def plot_A1(a1_dir: Path, a2_dir: Path, out_path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = _load_scores(a1_dir / "scores.csv")
    judge  = _load_judge(a1_dir / "llm_judge_scores.jsonl")
    car    = _load_coherent_from_samples(a1_dir / "sample_outputs.jsonl", "GOV-CONV")

    # PT baselines (from A2 merged dir)
    pt_scores = _load_scores(a2_dir / "scores.csv").get("A2_baseline_pt", {})
    pt_judge  = _load_judge(a2_dir / "llm_judge_scores.jsonl").get("A2_baseline_pt", {})
    pt_car    = _load_coherent_from_samples(a2_dir / "sample_outputs.jsonl", "GOV-CONV").get("A2_baseline_pt")
    it_car    = car.get("A1_baseline")

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

    # Baselines: (pt_val, it_val)
    it_b = scores.get("A1_baseline", {})

    gov_baselines = {
        "coherent_assistant_rate": (pt_car, it_car),
        "structural_token_ratio":  (pt_scores.get("structural_token_ratio"), it_b.get("structural_token_ratio")),
        "format_compliance":       (pt_scores.get("format_compliance"), it_b.get("format_compliance")),
    }
    content_benches = ["mmlu_accuracy", "exp3_factual_em", "exp3_reasoning_em", "reasoning_em"]
    safety_benches  = ["exp3_alignment_behavior", "alignment_behavior", "safety"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("A1: IT model — Corrective Direction Ablation (α sweep, 1.0 = baseline)",
                 fontsize=12, fontweight="bold")

    # Panel 1: Governance
    gov_series = {
        "coherent_assistant_rate": gcar(),
        "structural_token_ratio":  gs("structural_token_ratio"),
        "format_compliance":       gs("format_compliance"),
    }
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
    content_baselines = {
        b: (pt_scores.get(b), it_b.get(b)) for b in content_benches
    }
    _plot_panel(axes[1], xs, content_series, content_baselines,
                "Content (should be stable)",
                "α", baseline_x=1.0)

    # Panel 3: Safety
    safety_series = {}
    for b in safety_benches:
        if b == "safety":
            ys = gjudge("safety")
        else:
            ys = gs(b)
        if any(y is not None for y in ys):
            safety_series[b] = ys
    safety_baselines = {
        b: (pt_judge.get("safety") if b == "safety" else pt_scores.get(b),
            judge.get("A1_baseline", {}).get("safety") if b == "safety" else it_b.get(b))
        for b in safety_benches
    }
    _plot_panel(axes[2], xs, safety_series, safety_baselines,
                "Safety / Alignment",
                "α", baseline_x=1.0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A1 plot saved → {out_path}")


# ── A2 ────────────────────────────────────────────────────────────────────────

def plot_A2(a2_dir: Path, a1_dir: Path, out_path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = _load_scores(a2_dir / "scores.csv")
    judge  = _load_judge(a2_dir / "llm_judge_scores.jsonl")
    car    = _load_coherent_from_samples(a2_dir / "sample_outputs.jsonl", "GOV-CONV")

    # IT/PT baselines
    pt_scores = scores.get("A2_baseline_pt", {})
    pt_judge_b = judge.get("A2_baseline_pt", {})
    pt_car_val = car.get("A2_baseline_pt")

    it_scores_all = _load_scores(a1_dir / "scores.csv")
    it_b = it_scores_all.get("A1_baseline", {})
    it_judge_b = _load_judge(a1_dir / "llm_judge_scores.jsonl").get("A1_baseline", {})
    it_car_val = _load_coherent_from_samples(a1_dir / "sample_outputs.jsonl", "GOV-CONV").get("A1_baseline")

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

    gov_baselines = {
        "coherent_assistant_rate": (pt_car_val, it_car_val),
        "structural_token_ratio":  (pt_scores.get("structural_token_ratio"), it_b.get("structural_token_ratio")),
        "format_compliance":       (pt_scores.get("format_compliance"), it_b.get("format_compliance")),
    }
    content_benches = ["mmlu_accuracy", "exp3_factual_em", "exp3_reasoning_em", "reasoning_em"]
    safety_benches  = ["exp3_alignment_behavior", "alignment_behavior", "safety"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("A2: PT model — Corrective Direction Injection (β sweep, 0 = no injection)",
                 fontsize=12, fontweight="bold")

    gov_series = {
        "coherent_assistant_rate": gcar(),
        "structural_token_ratio":  gs("structural_token_ratio"),
        "format_compliance":       gs("format_compliance"),
    }
    _plot_panel(axes[0], xs, gov_series, gov_baselines,
                "Governance (PT→IT with β)",
                "β (injection magnitude)", baseline_x=0.0)

    content_series = {}
    for b in content_benches:
        ys = gs(b)
        if any(y is not None for y in ys):
            content_series[b] = ys
    content_baselines = {b: (pt_scores.get(b), it_b.get(b)) for b in content_benches}
    _plot_panel(axes[1], xs, content_series, content_baselines,
                "Content", "β", baseline_x=0.0)

    safety_series = {}
    for b in safety_benches:
        ys = gjudge("safety") if b == "safety" else gs(b)
        if any(y is not None for y in ys):
            safety_series[b] = ys
    safety_baselines = {
        b: (pt_judge_b.get("safety") if b == "safety" else pt_scores.get(b),
            it_judge_b.get("safety") if b == "safety" else it_b.get(b))
        for b in safety_benches
    }
    _plot_panel(axes[2], xs, safety_series, safety_baselines,
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
        ("A1_early", a1_early_dir, "Early 0-7",        "#3498db", "--", "s"),
        ("A1_mid",   a1_mid_dir,   "Mid 8-19",         "#2ecc71", ":",  "^"),
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
        ("coherent_assistant_rate", "Coherent Asst Rate", "Governance"),
        ("structural_token_ratio",  "Structural Token Ratio", "Governance"),
        ("format_compliance",       "Format Compliance", "Governance"),
        ("mmlu_accuracy",           "MMLU Accuracy", "Content"),
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
        "Red=Corrective(20-33)  Blue=Early(0-7)  Green=Mid(8-19)  "
        "Dashed=baseline; α<0 injects corrective direction (amplification)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"A1 v5 plot saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["A1", "A2", "both", "A1v5"], default="both")
    p.add_argument("--a1-dir", default="results/exp6/merged_A1_it")
    p.add_argument("--a2-dir", default="results/exp6/merged_A2_pt")
    p.add_argument("--a1-early-dir", default="results/exp6/merged_A1_early_it")
    p.add_argument("--a1-mid-dir", default="results/exp6/merged_A1_mid_it")
    args = p.parse_args()

    a1 = Path(args.a1_dir)
    a2 = Path(args.a2_dir)
    a1_early = Path(args.a1_early_dir)
    a1_mid = Path(args.a1_mid_dir)

    if args.experiment in ("A1", "both"):
        plot_A1(a1, a2, a1 / "plots" / "A1_dose_response_v5.png")
    if args.experiment in ("A2", "both"):
        plot_A2(a2, a1, a2 / "plots" / "A2_dose_response_v5.png")
    if args.experiment in ("A1v5", "both", "A1"):
        plot_A1_v5(a1, a1_early, a1_mid, a1 / "plots" / "A1_layer_specificity_v5.png")


if __name__ == "__main__":
    main()
