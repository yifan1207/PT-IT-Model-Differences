"""Cross-model overview plot for exp11 MLP graft runs.

Reads per-model merged run directories (each containing summary.json,
trajectory_summary.json, secondary_trajectory_summary.json,
prompt_summaries.jsonl) and emits:

  - overview_metrics.json      (per-model aggregate numbers)
  - overview_panel.png         (4-panel cross-model bar chart)
  - overview_trajectories.png  (3-panel cross-model per-layer line chart)
  - {model}_panel.png          (per-model trajectory panel copied from run dir)

Usage:
    uv run python scripts/plot_exp11_overview.py \\
        --run-root results/exp11/data/exp11_exp3_all2936_tunedlens_v10 \\
        --out-dir  results/exp11/plots/exp11_exp3_all2936_tunedlens_v10
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MODEL_DISPLAY = {
    "gemma3_4b": "Gemma3 4B",
    "qwen3_4b": "Qwen3 4B",
    "llama31_8b": "Llama 3.1 8B",
    "mistral_7b": "Mistral 7B",
    "olmo2_7b": "OLMo2 7B",
    "deepseek_v2_lite": "DeepSeek V2 Lite",
}
# Keep a stable display order
MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _summary_key(pipeline_name: str) -> str:
    return f"pipeline_{pipeline_name.lower()}"


def _iter_prompt_summaries(path: Path):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _collect_trajectories(run_dir: Path) -> dict:
    """Load per-layer trajectory curves needed for overview_trajectories.png."""
    traj = _read_json(run_dir / "trajectory_summary.json")
    config = _read_json(run_dir / "config.json")
    a = traj.get("A", {})
    b = traj.get("B", {})
    c = traj.get("C", {})
    a_prime = traj.get("A_prime", {})
    a_prime_tmpl = traj.get("A_prime_tmpl", {})
    b2 = traj.get("B2", {})

    def _arr(d: dict, key: str) -> list[float]:
        return [float(x) if x is not None else float("nan") for x in (d.get(key) or [])]

    delta_a = _arr(a, "delta_cosine")
    delta_b = _arr(b, "delta_cosine")
    residual_cos = _arr(b, "residual_cosine")
    cross_kl = _arr(b, "cross_kl")
    delta_a_prime = _arr(a_prime, "delta_cosine")
    residual_cos_a_prime = _arr(a_prime, "residual_cosine")
    cross_kl_a_prime = _arr(a_prime, "cross_kl")
    delta_a_prime_tmpl = _arr(a_prime_tmpl, "delta_cosine")
    residual_cos_a_prime_tmpl = _arr(a_prime_tmpl, "residual_cosine")
    cross_kl_a_prime_tmpl = _arr(a_prime_tmpl, "cross_kl")
    delta_b2 = _arr(b2, "delta_cosine")
    residual_cos_b2 = _arr(b2, "residual_cosine")
    cross_kl_b2 = _arr(b2, "cross_kl")
    n_layers = max(
        len(delta_a), len(delta_b), len(residual_cos), len(cross_kl),
        len(delta_a_prime), len(residual_cos_a_prime), len(cross_kl_a_prime),
        len(delta_a_prime_tmpl), len(residual_cos_a_prime_tmpl), len(cross_kl_a_prime_tmpl),
        len(delta_b2), len(residual_cos_b2), len(cross_kl_b2),
    )
    return {
        "n_layers": n_layers,
        "onset_layer": config.get("onset_layer"),
        "delta_cosine_a": delta_a,
        "delta_cosine_b": delta_b,
        "delta_cosine_a_prime": delta_a_prime,
        "delta_cosine_a_prime_tmpl": delta_a_prime_tmpl,
        "delta_cosine_b2": delta_b2,
        "residual_cosine": residual_cos,
        "cross_kl": cross_kl,
        "residual_cosine_a_prime": residual_cos_a_prime,
        "cross_kl_a_prime": cross_kl_a_prime,
        "residual_cosine_a_prime_tmpl": residual_cos_a_prime_tmpl,
        "cross_kl_a_prime_tmpl": cross_kl_a_prime_tmpl,
        "residual_cosine_b2": residual_cos_b2,
        "cross_kl_b2": cross_kl_b2,
        "teacher_forced": bool(config.get("teacher_forced")),
    }


def _collect_model(run_dir: Path) -> dict:
    summary = _read_json(run_dir / "summary.json").get("overall", {})
    config = _read_json(run_dir / "config.json")
    teacher_forced = bool(config.get("teacher_forced"))

    kl_key = "mean_commitment_layer_kl_0.1"
    configured_pipelines = config.get("pipelines")
    if configured_pipelines:
        pipeline_keys = [_summary_key(name) for name in configured_pipelines]
    else:
        pipeline_keys = ["pipeline_a", "pipeline_b"]
        if teacher_forced:
            pipeline_keys += ["pipeline_c", "pipeline_a_prime"]
    bucket: dict[str, dict[str, list]] = {
        k: {
            "length": [], "struct": [], "commit_kl": [],
            "paragraph": [], "header": [], "bullet": [], "first_token": [],
        }
        for k in pipeline_keys
    }
    div_steps: list[int] = []
    div_a_vs_c: list[int] = []
    div_b_vs_c: list[int] = []
    div_a_prime_vs_c: list[int] = []
    div_a_prime_tmpl_vs_c: list[int] = []
    div_b2_vs_c: list[int] = []
    first_token_matches = {k: 0 for k in pipeline_keys if k != "pipeline_c"}
    first_token_totals = {k: 0 for k in pipeline_keys if k != "pipeline_c"}

    for row in _iter_prompt_summaries(run_dir / "prompt_summaries.jsonl"):
        for pk in pipeline_keys:
            p = row.get(pk)
            if not p:
                continue
            bucket[pk]["length"].append(p.get("generated_text_length"))
            bucket[pk]["struct"].append(p.get("structural_token_ratio_tier1_proxy"))
            if kl_key in p:
                bucket[pk]["commit_kl"].append(p[kl_key])
            bucket[pk]["paragraph"].append(p.get("paragraph_count"))
            bucket[pk]["header"].append(p.get("header_count"))
            bucket[pk]["bullet"].append(p.get("bullet_count"))
            bucket[pk]["first_token"].append(p.get("first_token_str"))
        if row.get("divergence_step") is not None:
            div_steps.append(row["divergence_step"])
        if row.get("divergence_step_a_vs_c") is not None:
            div_a_vs_c.append(row["divergence_step_a_vs_c"])
        if row.get("divergence_step_b_vs_c") is not None:
            div_b_vs_c.append(row["divergence_step_b_vs_c"])
        if row.get("divergence_step_a_prime_vs_c") is not None:
            div_a_prime_vs_c.append(row["divergence_step_a_prime_vs_c"])
        if row.get("divergence_step_a_prime_tmpl_vs_c") is not None:
            div_a_prime_tmpl_vs_c.append(row["divergence_step_a_prime_tmpl_vs_c"])
        if row.get("divergence_step_b2_vs_c") is not None:
            div_b2_vs_c.append(row["divergence_step_b2_vs_c"])
        if teacher_forced and row.get("pipeline_c"):
            c_first = row["pipeline_c"].get("first_token_str")
            for pk in first_token_matches:
                if row.get(pk):
                    first_token_totals[pk] += 1
                    if row[pk].get("first_token_str") == c_first:
                        first_token_matches[pk] += 1

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    result = {
        "model": run_dir.name,
        "n_prompts": summary.get("n_prompts", 0),
        "n_layers": config.get("n_layers"),
        "teacher_forced": teacher_forced,
        "mean_divergence_step": summary.get("mean_divergence_step"),
    }
    for pk in pipeline_keys:
        result[f"{pk}_mean_length"] = _mean(bucket[pk]["length"])
        result[f"{pk}_mean_structural_ratio_tier1_proxy"] = _mean(bucket[pk]["struct"])
        result[f"{pk}_mean_commitment_layer_kl_0.1"] = _mean(bucket[pk]["commit_kl"])
        result[f"{pk}_mean_paragraph_count"] = _mean(bucket[pk]["paragraph"])
        result[f"{pk}_mean_header_count"] = _mean(bucket[pk]["header"])
        result[f"{pk}_mean_bullet_count"] = _mean(bucket[pk]["bullet"])
    if teacher_forced:
        result["mean_divergence_step_a_vs_c"] = _mean(div_a_vs_c)
        result["mean_divergence_step_b_vs_c"] = _mean(div_b_vs_c)
        result["mean_divergence_step_a_prime_vs_c"] = _mean(div_a_prime_vs_c)
        result["mean_divergence_step_a_prime_tmpl_vs_c"] = _mean(div_a_prime_tmpl_vs_c)
        result["mean_divergence_step_b2_vs_c"] = _mean(div_b2_vs_c)
        for pk in first_token_matches:
            result[f"{pk}_first_token_match_fraction_vs_c"] = (
                (first_token_matches[pk] / first_token_totals[pk])
                if first_token_totals[pk] else None
            )
    return result


def _plot_overview(metrics: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    teacher_forced_any = any(m.get("teacher_forced") for m in metrics)
    if teacher_forced_any:
        labels = [MODEL_DISPLAY.get(m["model"], m["model"]) for m in metrics]
        raw_base_div = [m.get("mean_divergence_step_a_prime_vs_c") or 0 for m in metrics]
        tmpl_base_div = [m.get("mean_divergence_step_a_prime_tmpl_vs_c") or 0 for m in metrics]
        raw_match_gain = [
            (m.get("pipeline_b_first_token_match_fraction_vs_c") or 0)
            - (m.get("pipeline_a_prime_first_token_match_fraction_vs_c") or 0)
            for m in metrics
        ]
        tmpl_match_gain = [
            (m.get("pipeline_b2_first_token_match_fraction_vs_c") or 0)
            - (m.get("pipeline_a_prime_tmpl_first_token_match_fraction_vs_c") or 0)
            for m in metrics
        ]
        raw_commit_shift = [
            (m.get("pipeline_b_mean_commitment_layer_kl_0.1") or 0)
            - (m.get("pipeline_a_prime_mean_commitment_layer_kl_0.1") or 0)
            for m in metrics
        ]
        tmpl_commit_shift = [
            (m.get("pipeline_b2_mean_commitment_layer_kl_0.1") or 0)
            - (m.get("pipeline_a_prime_tmpl_mean_commitment_layer_kl_0.1") or 0)
            for m in metrics
        ]

        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        ax_div, ax_match, ax_commit_raw, ax_commit_tmpl = axes

        x = range(len(labels))
        width = 0.36
        ax_div.bar([i - width / 2 for i in x], raw_base_div, width, color="#9467bd", label="Raw control: A' vs C")
        ax_div.bar([i + width / 2 for i in x], tmpl_base_div, width, color="#c5b0d5", label="Template control: A'_tmpl vs C")
        ax_div.set_xticks(list(x))
        ax_div.set_xticklabels(labels, rotation=20, ha="right")
        ax_div.set_title("Control divergence from C")
        ax_div.set_ylabel("First divergence step of free argmax vs C")
        ax_div.legend(frameon=False, fontsize=8)

        ax_match.bar([i - width / 2 for i in x], raw_match_gain, width, color="#d62728", label="Raw branch: B - A'")
        ax_match.bar([i + width / 2 for i in x], tmpl_match_gain, width, color="#ff7f0e", label="Template branch: B2 - A'_tmpl")
        ax_match.set_xticks(list(x))
        ax_match.set_xticklabels(labels, rotation=20, ha="right")
        ax_match.set_title("First-token match with C gain")
        ax_match.set_ylabel("Δ match fraction vs C")
        ax_match.axhline(0, color="#888", linewidth=0.8)
        ax_match.legend(frameon=False, fontsize=8)

        ax_commit_raw.bar(labels, raw_commit_shift, color="#2ca02c")
        ax_commit_raw.set_title("Commitment shift\nRaw branch: B - A'")
        ax_commit_raw.set_ylabel("Δ mean commitment layer")
        ax_commit_raw.axhline(0, color="#888", linewidth=0.8)

        ax_commit_tmpl.bar(labels, tmpl_commit_shift, color="#bcbd22")
        ax_commit_tmpl.set_title("Commitment shift\nTemplate branch: B2 - A'_tmpl")
        ax_commit_tmpl.set_ylabel("Δ mean commitment layer")
        ax_commit_tmpl.axhline(0, color="#888", linewidth=0.8)

        for ax in axes:
            for label in ax.get_xticklabels():
                label.set_rotation(20)
                label.set_ha("right")
            ax.grid(axis="y", alpha=0.2)

        fig.suptitle(
            "exp11 v11.1 teacher-forced graft overview\n"
            "Raw branch isolates pure late-MLP effect; template branch tests prompt-scaffolding sensitivity",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    labels = [MODEL_DISPLAY.get(m["model"], m["model"]) for m in metrics]
    div = [m.get("mean_divergence_step") or 0 for m in metrics]
    struct_shift = [
        (m.get("pipeline_b_mean_structural_ratio_tier1_proxy") or 0)
        - (m.get("pipeline_a_mean_structural_ratio_tier1_proxy") or 0)
        for m in metrics
    ]
    length_shift = [
        (m.get("pipeline_b_mean_length") or 0) - (m.get("pipeline_a_mean_length") or 0)
        for m in metrics
    ]
    commit_shift = [
        (m.get("pipeline_b_mean_commitment_layer_kl_0.1") or 0)
        - (m.get("pipeline_a_mean_commitment_layer_kl_0.1") or 0)
        for m in metrics
    ]
    # legacy shape preserved; v11.1 governance plot is generated separately in _plot_governance

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ax_div, ax_struct, ax_length, ax_commit = axes

    ax_div.bar(labels, div, color="#2c2c2c")
    ax_div.set_title("Mean divergence step")
    ax_div.set_ylabel("Step index of first A/B token mismatch")

    ax_struct.bar(labels, struct_shift, color="#d62728")
    ax_struct.set_title("Structural ratio shift (B − A)")
    ax_struct.set_ylabel("Δ mean structural token ratio (tier1 proxy)")
    ax_struct.axhline(0, color="#888", linewidth=0.8)

    ax_length.bar(labels, length_shift, color="#1f77b4")
    ax_length.set_title("Generated length shift (B − A)")
    ax_length.set_ylabel("Δ mean generated token count")
    ax_length.axhline(0, color="#888", linewidth=0.8)

    ax_commit.bar(labels, commit_shift, color="#2ca02c")
    ax_commit.set_title("Commitment-layer shift (B − A)\nKL threshold = 0.1")
    ax_commit.set_ylabel("Δ mean commitment layer")
    ax_commit.axhline(0, color="#888", linewidth=0.8)

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "exp11 MLP graft — PT vs PT+IT graft @ onset layer\n"
        "B = pipeline with IT MLP grafted on top of PT backbone",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectories(trajectories: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "gemma3_4b": "#1f77b4",
        "qwen3_4b": "#d62728",
        "llama31_8b": "#2ca02c",
        "mistral_7b": "#9467bd",
        "olmo2_7b": "#ff7f0e",
        "deepseek_v2_lite": "#17becf",
    }

    teacher_forced_any = any(entry.get("teacher_forced") for entry in trajectories)
    n_panels = 5 if teacher_forced_any else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    ax_delta, ax_resid, ax_kl = axes[0], axes[1], axes[2]
    ax_b_vs_aprime = axes[3] if teacher_forced_any else None
    ax_template = axes[4] if teacher_forced_any else None

    for entry in trajectories:
        model = entry["model"]
        n_layers = entry["n_layers"]
        if n_layers == 0:
            continue
        depths = [i / max(n_layers - 1, 1) for i in range(n_layers)]
        label = MODEL_DISPLAY.get(model, model)
        color = colors.get(model, "#444444")

        delta_a = entry["delta_cosine_a"]
        delta_b = entry["delta_cosine_b"]
        delta_a_prime = entry.get("delta_cosine_a_prime") or []
        delta_a_prime_tmpl = entry.get("delta_cosine_a_prime_tmpl") or []
        delta_b2 = entry.get("delta_cosine_b2") or []
        if teacher_forced_any:
            if delta_a_prime and delta_b and len(delta_a_prime) == len(delta_b):
                diff = [b - a for a, b in zip(delta_a_prime, delta_b)]
                ax_delta.plot(depths[: len(diff)], diff, label=label, color=color, linewidth=2)
            if delta_a_prime_tmpl and delta_b2 and len(delta_a_prime_tmpl) == len(delta_b2):
                diff = [b - a for a, b in zip(delta_a_prime_tmpl, delta_b2)]
                ax_delta.plot(depths[: len(diff)], diff, color=color, linewidth=2, linestyle="--")
        elif delta_a and delta_b and len(delta_a) == len(delta_b):
            diff = [b - a for a, b in zip(delta_a, delta_b)]
            ax_delta.plot(depths[: len(diff)], diff, label=label, color=color, linewidth=2)

        if entry["residual_cosine"]:
            ax_resid.plot(
                depths[: len(entry["residual_cosine"])],
                entry["residual_cosine"],
                label=label,
                color=color,
                linewidth=2,
            )
        if teacher_forced_any and entry.get("residual_cosine_b2"):
            ax_resid.plot(
                depths[: len(entry["residual_cosine_b2"])],
                entry["residual_cosine_b2"],
                color=color,
                linewidth=2,
                linestyle="--",
            )

        if entry["cross_kl"]:
            ax_kl.plot(
                depths[: len(entry["cross_kl"])],
                entry["cross_kl"],
                label=label,
                color=color,
                linewidth=2,
            )
        if teacher_forced_any and entry.get("cross_kl_b2"):
            ax_kl.plot(
                depths[: len(entry["cross_kl_b2"])],
                entry["cross_kl_b2"],
                color=color,
                linewidth=2,
                linestyle="--",
            )

        # v11 headline: residual_cosine(B vs C) − residual_cosine(A' vs C) —
        # pure graft effect isolated from backbone drift.
        if (
            ax_b_vs_aprime is not None
            and entry.get("residual_cosine")
            and entry.get("residual_cosine_a_prime")
            and len(entry["residual_cosine"]) == len(entry["residual_cosine_a_prime"])
        ):
            diff = [b - a for a, b in zip(entry["residual_cosine_a_prime"], entry["residual_cosine"])]
            ax_b_vs_aprime.plot(depths[: len(diff)], diff, label=label, color=color, linewidth=2)
        if (
            ax_b_vs_aprime is not None
            and entry.get("residual_cosine_b2")
            and entry.get("residual_cosine_a_prime_tmpl")
            and len(entry["residual_cosine_b2"]) == len(entry["residual_cosine_a_prime_tmpl"])
        ):
            diff = [b - a for a, b in zip(entry["residual_cosine_a_prime_tmpl"], entry["residual_cosine_b2"])]
            ax_b_vs_aprime.plot(
                depths[: len(diff)], diff, color=color, linewidth=2, linestyle="--",
            )

        if (
            ax_template is not None
            and delta_b
            and delta_b2
            and len(delta_b) == len(delta_b2)
        ):
            diff = [b2 - b1 for b1, b2 in zip(delta_b, delta_b2)]
            ax_template.plot(depths[: len(diff)], diff, label=label, color=color, linewidth=2)

        onset = entry.get("onset_layer")
        if onset is not None and n_layers > 1:
            onset_depth = onset / (n_layers - 1)
            for ax in axes:
                ax.axvline(onset_depth, color=color, linestyle=":", alpha=0.35, linewidth=1)

    ax_delta.set_title(
        "Δ delta_cosine\nsolid = raw branch (B − A'), dashed = template branch (B2 − A'_tmpl)"
        if teacher_forced_any
        else "Δ delta_cosine (B − A)\ncos(MLP out, pre-MLP residual)"
    )
    ax_delta.set_xlabel("Normalized depth (layer / n_layers)")
    ax_delta.set_ylabel("Cosine difference (B − A')" if teacher_forced_any else "Cosine difference (B − A)")
    ax_delta.axhline(0, color="#888", linewidth=0.8)

    resid_title = (
        "Residual cosine to C\nsolid = raw graft B, dashed = template graft B2"
        if teacher_forced_any
        else "Residual cosine (B vs A)"
    )
    ax_resid.set_title(f"{resid_title}\ncos(B_resid, baseline_resid) per layer")
    ax_resid.set_xlabel("Normalized depth (layer / n_layers)")
    ax_resid.set_ylabel("Cosine similarity")

    kl_title = (
        "Cross-pipeline KL to C\nsolid = raw graft B, dashed = template graft B2"
        if teacher_forced_any
        else "Cross-pipeline KL\nKL(B ‖ A) per layer"
    )
    ax_kl.set_title(kl_title)
    ax_kl.set_xlabel("Normalized depth (layer / n_layers)")
    ax_kl.set_ylabel("KL divergence (nats)")

    if ax_b_vs_aprime is not None:
        ax_b_vs_aprime.set_title(
            "Pure graft effect toward C\nsolid = raw [B - A'], dashed = template [B2 - A'_tmpl]"
        )
        ax_b_vs_aprime.set_xlabel("Normalized depth (layer / n_layers)")
        ax_b_vs_aprime.set_ylabel("Cosine diff (positive = graft pulls toward C)")
        ax_b_vs_aprime.axhline(0, color="#888", linewidth=0.8)

    if ax_template is not None:
        ax_template.set_title("Template ablation on graft\nB2 - B delta_cosine")
        ax_template.set_xlabel("Normalized depth (layer / n_layers)")
        ax_template.set_ylabel("Cosine diff (positive = template strengthens graft)")
        ax_template.axhline(0, color="#888", linewidth=0.8)

    for ax in axes:
        ax.grid(alpha=0.2)

    handles, labels = ax_delta.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False)

    subtitle = (
        "Teacher-forced: B and A' share C's token history → cross-pipeline metrics are pure graft/backbone effect.\n"
        "Dotted verticals mark each model's onset layer."
        if teacher_forced_any
        else "Dotted verticals mark each model's onset layer; B = PT+IT-MLP graft, A = PT baseline"
    )
    fig.suptitle(
        f"exp11 MLP graft — per-layer trajectories across models\n{subtitle}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.9])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_governance(metrics: list[dict], out_path: Path) -> None:
    """Grouped bar chart of whole-response structure counts across teacher-forced pipelines."""
    import matplotlib.pyplot as plt

    teacher_forced = any(m.get("teacher_forced") for m in metrics)
    if not teacher_forced:
        return

    labels = [MODEL_DISPLAY.get(m["model"], m["model"]) for m in metrics]
    key_order = [
        ("pipeline_a", "A (PT)"),
        ("pipeline_c", "C (IT teacher)"),
        ("pipeline_a_prime", "A' raw"),
        ("pipeline_b", "B raw graft"),
        ("pipeline_a_prime_tmpl", "A'_tmpl"),
        ("pipeline_b2", "B2 tmpl graft"),
    ]
    pipeline_keys = [(k, label) for k, label in key_order if any(m.get(f"{k}_mean_paragraph_count") is not None for m in metrics)]
    pipeline_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#c5b0d5", "#ff7f0e"][: len(pipeline_keys)]

    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    ax_para, ax_head, ax_bullet = axes

    def _bars(ax, field: str, title: str, ylabel: str) -> None:
        x = np.arange(len(labels))
        width = 0.18
        center = (len(pipeline_keys) - 1) / 2
        for i, ((pk, pname), color) in enumerate(zip(pipeline_keys, pipeline_colors)):
            values = [m.get(f"{pk}_mean_{field}") or 0.0 for m in metrics]
            ax.bar(x + (i - center) * width, values, width, label=pname, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)

    _bars(ax_para, "paragraph_count", "Mean paragraph count", "\\n\\n occurrences")
    _bars(ax_head, "header_count", "Mean header count", "^# lines")
    _bars(ax_bullet, "bullet_count", "Mean bullet count", "^[-*] or ^N. lines")

    handles, labels_h = ax_para.get_legend_handles_labels()
    fig.legend(handles, labels_h, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(
        "exp11 v11.1 governance — whole-response structure counts across pipelines\n"
        "solid branch = raw prompt, suffix _tmpl = IT chat-template prompt",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True,
                        help="Directory containing per-model run subdirs")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write overview_metrics.json and plots")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure out which model dirs exist. Accept both bare model-name subdirs
    # (e.g. gemma3_4b) and full run-name subdirs (exp11_..._gemma3_4b).
    subdirs = {p.name: p for p in run_root.iterdir() if p.is_dir()}
    if not subdirs:
        raise SystemExit(f"No subdirectories found in {run_root}")

    metrics: list[dict] = []
    trajectories: list[dict] = []
    for model in MODEL_ORDER:
        path = None
        if model in subdirs:
            path = subdirs[model]
        else:
            for name, p in subdirs.items():
                if name.endswith(f"_{model}") or name.endswith(model):
                    path = p
                    break
        if path is None:
            print(f"[skip] no directory for {model}")
            continue
        entry = _collect_model(path)
        entry["model"] = model  # normalize display key
        metrics.append(entry)

        traj_entry = _collect_trajectories(path)
        traj_entry["model"] = model
        trajectories.append(traj_entry)

        # Copy per-model panel.png if present
        panel_src = path / "plots" / "panel.png"
        if panel_src.exists():
            shutil.copy2(panel_src, out_dir / f"{model}_panel.png")

    (out_dir / "overview_metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot_overview(metrics, out_dir / "overview_panel.png")
    _plot_trajectories(trajectories, out_dir / "overview_trajectories.png")
    teacher_forced = any(m.get("teacher_forced") for m in metrics)
    if teacher_forced:
        _plot_governance(metrics, out_dir / "overview_governance.png")
    print(f"Wrote {len(metrics)} model entries to {out_dir / 'overview_metrics.json'}")
    print(f"Wrote overview panel to {out_dir / 'overview_panel.png'}")
    print(f"Wrote overview trajectories to {out_dir / 'overview_trajectories.png'}")
    if teacher_forced:
        print(f"Wrote overview governance to {out_dir / 'overview_governance.png'}")


if __name__ == "__main__":
    main()
