#!/usr/bin/env python3
"""Exp8: Phase 0 multi-model steering plots.

Generates all cross-model comparison figures for the paper.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODELS = ['gemma3_4b', 'llama31_8b', 'qwen3_4b', 'mistral_7b', 'deepseek_v2_lite', 'olmo2_7b']
MODEL_LABELS = {
    'gemma3_4b': 'Gemma-3 4B',
    'llama31_8b': 'LLaMA-3.1 8B',
    'qwen3_4b': 'Qwen-3 4B',
    'mistral_7b': 'Mistral 7B',
    'deepseek_v2_lite': 'DeepSeek-V2-Lite',
    'olmo2_7b': 'OLMo-2 7B',
}
COLORS = {
    'gemma3_4b': '#E41A1C',
    'llama31_8b': '#377EB8',
    'qwen3_4b': '#4DAF4A',
    'mistral_7b': '#984EA3',
    'deepseek_v2_lite': '#FF7F00',
    'olmo2_7b': '#A65628',
}

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent

ALPHAS = [-5, -3, -2, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]
BENCH_SHORT = {
    'structural_token_ratio': 'STR',
    'format_compliance_v2': 'FC',
    'mmlu_forced_choice': 'MMLU',
    'exp3_reasoning_em': 'RE',
    'exp3_alignment_behavior': 'AB',
}
BENCH_FULL = {
    'structural_token_ratio': 'Structural Token Ratio',
    'format_compliance_v2': 'Format Compliance',
    'mmlu_forced_choice': 'MMLU (Forced Choice)',
    'exp3_reasoning_em': 'Reasoning (Exact Match)',
    'exp3_alignment_behavior': 'Alignment Behavior',
}
GOV_BENCHMARKS = ['structural_token_ratio', 'format_compliance_v2']
CONTENT_BENCHMARKS = ['mmlu_forced_choice', 'exp3_reasoning_em', 'exp3_alignment_behavior']


def load_scores():
    """Load all model scores into {model: {(condition, benchmark): value}}."""
    all_scores = {}
    for m in MODELS:
        path = DATA_DIR / f"{m}_scores.jsonl"
        rows = [json.loads(l) for l in open(path)]
        all_scores[m] = {(r['condition'], r['benchmark']): r for r in rows}
    return all_scores


def load_judge_scores():
    """Load judge scores into {model: {(task, condition): [scores]}}."""
    all_judge = {}
    for m in MODELS:
        path = DATA_DIR / f"{m}_judge_scores.jsonl"
        rows = [json.loads(l) for l in open(path)]
        tc = defaultdict(list)
        for r in rows:
            s = r.get('score', -1)
            if s >= 0:  # exclude parse failures
                tc[(r['task'], r['condition'])].append(s)
        all_judge[m] = tc
    return all_judge


def plot_cross_model_dose_response(all_scores):
    """Fig 1: Cross-model dose-response — governance and content metrics.

    Layout: 2 governance on top row + 3 content on bottom row = 5 panels.
    Top row has 2 panels centered in a 3-column grid (left + center, right hidden).
    """
    all_benchmarks = GOV_BENCHMARKS + CONTENT_BENCHMARKS  # 5 total
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: 2 governance benchmarks in columns 0, 1
    for col, bench in enumerate(GOV_BENCHMARKS):
        ax = axes[0, col]
        for m in MODELS:
            vals = []
            for a in ALPHAS:
                r = all_scores[m].get((f'A1_alpha_{a}', bench))
                vals.append(r['value'] if r else None)
            ax.plot(ALPHAS, vals, marker='o', markersize=4, label=MODEL_LABELS[m],
                    color=COLORS[m], linewidth=1.5)
        ax.set_title(BENCH_FULL[bench], fontsize=13, fontweight='bold')
        ax.set_xlabel('α (correction strength)')
        ax.set_ylabel('Score')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.4, label='α=1 (baseline)')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, loc='best')

    # Hide empty top-right cell
    axes[0, 2].set_visible(False)

    # Bottom row: 3 content benchmarks
    for col, bench in enumerate(CONTENT_BENCHMARKS):
        ax = axes[1, col]
        for m in MODELS:
            vals = []
            for a in ALPHAS:
                r = all_scores[m].get((f'A1_alpha_{a}', bench))
                vals.append(r['value'] if r else None)
            ax.plot(ALPHAS, vals, marker='o', markersize=4, label=MODEL_LABELS[m],
                    color=COLORS[m], linewidth=1.5)
        ax.set_title(BENCH_FULL[bench], fontsize=13, fontweight='bold')
        ax.set_xlabel('α (correction strength)')
        ax.set_ylabel('Score')
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.4)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Cross-Model Dose-Response: Corrective Direction Steering (Phase 0)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_dose_response_cross_model.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_selectivity_comparison(all_scores):
    """Fig 2: Corrective vs random direction — selectivity per model (6 panels)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        benchmarks = list(BENCH_SHORT.keys())
        x = np.arange(len(benchmarks))
        width = 0.25

        bl_vals = [all_scores[m].get(('A1_baseline', b), {}).get('value', 0) for b in benchmarks]
        a3_vals = [all_scores[m].get(('A1_alpha_3', b), {}).get('value', 0) for b in benchmarks]
        rb3_vals = [all_scores[m].get(('A1_ctrl_random_b3', b), {}).get('value', 0) for b in benchmarks]

        ax.bar(x - width, bl_vals, width, label='Baseline (α=1)', color='#2196F3', alpha=0.8)
        ax.bar(x, a3_vals, width, label='Corrective α=3', color='#FF9800', alpha=0.8)
        ax.bar(x + width, rb3_vals, width, label='Random β=3', color='#F44336', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([BENCH_SHORT[b] for b in benchmarks], fontsize=9)
        ax.set_title(MODEL_LABELS[m], fontsize=12, fontweight='bold', color=COLORS[m])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis='y')
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Selectivity: Corrective Direction vs Random Perturbation (magnitude=3)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_selectivity_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_corrective_vs_random_dose(all_scores):
    """Corrective direction vs random control dose-response for ALL 6 models.

    Each panel: one model, STR on y-axis, α on x-axis.
    Shows corrective direction line + random control band (±1σ from 3 seeds).
    Replaces the old 4-panel direction_validation_corrective_vs_random.png.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Only positive alphas for clean visualization
    pos_alphas = [a for a in ALPHAS if a >= 0]

    for idx, m in enumerate(MODELS):
        ax = axes[idx]

        # Corrective direction STR
        corr_vals = []
        for a in pos_alphas:
            r = all_scores[m].get((f'A1_alpha_{a}', 'structural_token_ratio'))
            corr_vals.append(r['value'] if r else None)

        # Baseline (α=1)
        bl = all_scores[m].get(('A1_alpha_1', 'structural_token_ratio'))
        bl_val = bl['value'] if bl else 0

        # Random controls at b0, b1, b3
        random_strs = {}
        for ctrl in ['A1_ctrl_random_b0', 'A1_ctrl_random_b1', 'A1_ctrl_random_b3']:
            r = all_scores[m].get((ctrl, 'structural_token_ratio'))
            if r:
                beta = int(ctrl.split('_b')[-1])
                random_strs[beta] = r['value']

        # Plot corrective direction
        ax.plot(pos_alphas, corr_vals, 'o-', color='#E41A1C', linewidth=2,
                markersize=5, label='Corrective direction', zorder=3)

        # Plot random controls as horizontal band
        if random_strs:
            rand_vals = list(random_strs.values())
            rand_mean = np.mean(rand_vals)
            rand_std = np.std(rand_vals) if len(rand_vals) > 1 else 0
            ax.axhspan(rand_mean - rand_std, rand_mean + rand_std,
                       color='lightblue', alpha=0.4, label=f'Random ctrl (±1σ, n={len(rand_vals)})')
            ax.axhline(rand_mean, color='cornflowerblue', linestyle=':', alpha=0.6)

        # Baseline reference
        ax.axhline(bl_val, color='gray', linestyle='--', alpha=0.5,
                   label=f'Baseline ({bl_val:.3f})')

        ax.set_title(MODEL_LABELS[m], fontsize=12, fontweight='bold', color=COLORS[m])
        ax.set_xlabel('α (correction strength)')
        ax.set_ylabel('Structural Token Ratio')
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, loc='best')

    fig.suptitle('Direction Specificity: Corrective Direction vs Random Control (STR)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_corrective_vs_random_dose.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_normalized_dose_response(all_scores):
    """Fig 3: Normalized dose-response (each metric normalized to baseline=1.0)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Governance panel
    ax = axes[0]
    for m in MODELS:
        bl_str = all_scores[m].get(('A1_baseline', 'format_compliance_v2'), {}).get('value', 0.001)
        if bl_str < 0.01:
            bl_str = 0.01  # avoid div by zero
        vals = []
        for a in ALPHAS:
            r = all_scores[m].get((f'A1_alpha_{a}', 'format_compliance_v2'))
            vals.append(r['value'] / bl_str if r else None)
        ax.plot(ALPHAS, vals, marker='o', markersize=4, label=MODEL_LABELS[m],
                color=COLORS[m], linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('α')
    ax.set_ylabel('FC / FC_baseline')
    ax.set_title('Governance (Format Compliance)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Content panel
    ax = axes[1]
    for m in MODELS:
        bl_mmlu = all_scores[m].get(('A1_baseline', 'mmlu_forced_choice'), {}).get('value', 0.001)
        if bl_mmlu < 0.01:
            bl_mmlu = 0.01
        vals = []
        for a in ALPHAS:
            r = all_scores[m].get((f'A1_alpha_{a}', 'mmlu_forced_choice'))
            vals.append(r['value'] / bl_mmlu if r else None)
        ax.plot(ALPHAS, vals, marker='o', markersize=4, label=MODEL_LABELS[m],
                color=COLORS[m], linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('α')
    ax.set_ylabel('MMLU / MMLU_baseline')
    ax.set_title('Content (MMLU Forced Choice)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Normalized Dose-Response: Structure-Content Dissociation',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_normalized_dose_response.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_judge_heatmap(all_judge):
    """Fig 4: LLM judge g1 scores as heatmap (model × condition)."""
    conditions = ['A1_baseline'] + [f'A1_alpha_{a}' for a in ALPHAS] + \
                 ['A1_ctrl_random_b0', 'A1_ctrl_random_b1', 'A1_ctrl_random_b3']
    cond_labels = ['bl'] + [str(a) for a in ALPHAS] + ['rb0', 'rb1', 'rb3']

    matrix = np.zeros((len(MODELS), len(conditions)))
    for i, m in enumerate(MODELS):
        for j, c in enumerate(conditions):
            scores = all_judge[m].get(('g1', c), [])
            matrix[i, j] = np.mean(scores) if scores else 0

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=1, vmax=5)
    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)

    # Annotate cells
    for i in range(len(MODELS)):
        for j in range(len(conditions)):
            ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if matrix[i,j] < 2.5 else 'black')

    plt.colorbar(im, ax=ax, label='G1 Judge Score (1-5)')
    ax.set_title('LLM Judge G1 (Governance Quality) — All Models × Conditions',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_judge_heatmap.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_dissociation_summary(all_scores):
    """Fig 5: Summary dissociation — governance effect vs content preservation per model."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for m in MODELS:
        # Governance effect: |FC(α=0) - FC(baseline)| / FC(baseline)
        bl_fc = all_scores[m].get(('A1_baseline', 'format_compliance_v2'), {}).get('value', 0.001)
        a0_fc = all_scores[m].get(('A1_alpha_0', 'format_compliance_v2'), {}).get('value', 0)
        gov_effect = abs(a0_fc - bl_fc) / max(bl_fc, 0.01)

        # Content preservation: MMLU(α=0) / MMLU(baseline)
        bl_mmlu = all_scores[m].get(('A1_baseline', 'mmlu_forced_choice'), {}).get('value', 0.001)
        a0_mmlu = all_scores[m].get(('A1_alpha_0', 'mmlu_forced_choice'), {}).get('value', 0)
        content_pres = a0_mmlu / max(bl_mmlu, 0.01)

        ax.scatter(gov_effect, content_pres, s=150, color=COLORS[m],
                   label=MODEL_LABELS[m], zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(MODEL_LABELS[m], (gov_effect, content_pres),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.set_xlabel('Governance Effect at α=0\n|FC(α=0) - FC(baseline)| / FC(baseline)', fontsize=11)
    ax.set_ylabel('Content Preservation at α=0\nMMLU(α=0) / MMLU(baseline)', fontsize=11)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    ax.set_title('Structure-Content Dissociation Across Model Families', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_dissociation_summary.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_random_control_bars(all_scores):
    """Fig 6: Random controls — effect size comparison across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # STR panel
    ax = axes[0]
    x = np.arange(len(MODELS))
    width = 0.2
    for i, ctrl in enumerate(['A1_ctrl_random_b0', 'A1_ctrl_random_b1', 'A1_ctrl_random_b3']):
        vals = []
        for m in MODELS:
            bl = all_scores[m].get(('A1_baseline', 'structural_token_ratio'), {}).get('value', 0)
            cv = all_scores[m].get((ctrl, 'structural_token_ratio'), {}).get('value', 0)
            vals.append(abs(cv - bl))
        label = ctrl.replace('A1_ctrl_random_', 'β=')
        ax.bar(x + i * width, vals, width, label=f'Random {label}', alpha=0.8)

    # Add corrective α=1 effect
    corr_vals = []
    for m in MODELS:
        bl = all_scores[m].get(('A1_baseline', 'structural_token_ratio'), {}).get('value', 0)
        a1 = all_scores[m].get(('A1_alpha_1', 'structural_token_ratio'), {}).get('value', 0)
        corr_vals.append(abs(a1 - bl))
    ax.bar(x + 3 * width, corr_vals, width, label='Corrective α=1', color='#2196F3', alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('|Effect - Baseline| (STR)')
    ax.set_title('STR Effect Size', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')

    # MMLU panel
    ax = axes[1]
    for i, ctrl in enumerate(['A1_ctrl_random_b0', 'A1_ctrl_random_b1', 'A1_ctrl_random_b3']):
        vals = []
        for m in MODELS:
            bl = all_scores[m].get(('A1_baseline', 'mmlu_forced_choice'), {}).get('value', 0)
            cv = all_scores[m].get((ctrl, 'mmlu_forced_choice'), {}).get('value', 0)
            vals.append(abs(cv - bl))
        label = ctrl.replace('A1_ctrl_random_', 'β=')
        ax.bar(x + i * width, vals, width, label=f'Random {label}', alpha=0.8)

    corr_vals = []
    for m in MODELS:
        bl = all_scores[m].get(('A1_baseline', 'mmlu_forced_choice'), {}).get('value', 0)
        a1 = all_scores[m].get(('A1_alpha_1', 'mmlu_forced_choice'), {}).get('value', 0)
        corr_vals.append(abs(a1 - bl))
    ax.bar(x + 3 * width, corr_vals, width, label='Corrective α=1', color='#2196F3', alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('|Effect - Baseline| (MMLU)')
    ax.set_title('MMLU Effect Size', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Random Direction Controls vs Corrective Direction',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_random_control_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def compute_effect_sizes(all_scores):
    """§5.5: Cohen's d at α=0 (full removal) vs α=1 (baseline) for each model × benchmark.

    Uses pooled SD across all α conditions for each model/benchmark as the denominator,
    since we have single aggregate scores per condition (not per-item distributions).
    We approximate variability from the dose-response curve spread.
    """
    from scipy import stats as scipy_stats

    benchmarks = list(BENCH_SHORT.keys())
    results = {}

    for m in MODELS:
        results[m] = {}
        for b in benchmarks:
            # Baseline (α=1) and treatment (α=0)
            bl_row = all_scores[m].get(('A1_alpha_1', b))
            tx_row = all_scores[m].get(('A1_alpha_0', b))

            if not bl_row or not tx_row:
                results[m][b] = None
                continue

            bl_val = bl_row['value']
            tx_val = tx_row['value']
            bl_n = bl_row.get('n', 100)
            tx_n = tx_row.get('n', 100)

            # Compute SE from score data if available, else use binomial approximation
            bl_se = bl_row.get('se', np.sqrt(bl_val * (1 - bl_val) / bl_n) if 0 < bl_val < 1 else 0.01)
            tx_se = tx_row.get('se', np.sqrt(tx_val * (1 - tx_val) / tx_n) if 0 < tx_val < 1 else 0.01)

            # Pooled SD
            bl_sd = bl_se * np.sqrt(bl_n)
            tx_sd = tx_se * np.sqrt(tx_n)
            pooled_sd = np.sqrt(((bl_n - 1) * bl_sd**2 + (tx_n - 1) * tx_sd**2) / (bl_n + tx_n - 2))

            if pooled_sd > 0:
                d = (tx_val - bl_val) / pooled_sd
            else:
                d = 0.0

            results[m][b] = {
                'cohens_d': float(d),
                'baseline': float(bl_val),
                'treatment': float(tx_val),
                'diff': float(tx_val - bl_val),
                'pooled_sd': float(pooled_sd),
                'bl_n': int(bl_n),
                'tx_n': int(tx_n),
            }

    # Save
    out_path = DATA_DIR / "phase0_effect_sizes.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Effect sizes → {out_path.name}")

    return results


def compute_spearman_monotonicity(all_scores):
    """§5.6: Spearman ρ between α and each metric, per model."""
    from scipy.stats import spearmanr

    benchmarks = list(BENCH_SHORT.keys())
    results = {}

    for m in MODELS:
        results[m] = {}
        for b in benchmarks:
            alphas_used = []
            vals = []
            for a in ALPHAS:
                r = all_scores[m].get((f'A1_alpha_{a}', b))
                if r:
                    alphas_used.append(a)
                    vals.append(r['value'])

            if len(alphas_used) >= 4:
                rho, p = spearmanr(alphas_used, vals)
                results[m][b] = {
                    'spearman_rho': float(rho),
                    'p_value': float(p),
                    'n_points': len(alphas_used),
                }
            else:
                results[m][b] = None

    out_path = DATA_DIR / "phase0_monotonicity.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Monotonicity → {out_path.name}")

    return results


def plot_effect_size_table(effect_sizes):
    """§5.5 plot: Cohen's d heatmap across models × benchmarks."""
    benchmarks = list(BENCH_SHORT.keys())

    # Build matrix
    d_matrix = np.zeros((len(MODELS), len(benchmarks)))
    for i, m in enumerate(MODELS):
        for j, b in enumerate(benchmarks):
            r = effect_sizes[m].get(b)
            d_matrix[i, j] = r['cohens_d'] if r else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(d_matrix, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels([BENCH_SHORT[b] for b in benchmarks], fontsize=11)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)

    # Annotate cells
    for i in range(len(MODELS)):
        for j in range(len(benchmarks)):
            val = d_matrix[i, j]
            color = 'white' if abs(val) > 1.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10, color=color)

    plt.colorbar(im, ax=ax, label="Cohen's d (α=0 vs α=1)")
    ax.set_title("Effect Sizes: Full Corrective Direction Removal (α=0 vs baseline)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_effect_sizes.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_monotonicity_table(monotonicity):
    """§5.6 plot: Spearman ρ heatmap across models × benchmarks."""
    benchmarks = list(BENCH_SHORT.keys())

    rho_matrix = np.zeros((len(MODELS), len(benchmarks)))
    for i, m in enumerate(MODELS):
        for j, b in enumerate(benchmarks):
            r = monotonicity[m].get(b)
            rho_matrix[i, j] = r['spearman_rho'] if r else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(rho_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels([BENCH_SHORT[b] for b in benchmarks], fontsize=11)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)

    for i in range(len(MODELS)):
        for j in range(len(benchmarks)):
            r = monotonicity[MODELS[i]].get(benchmarks[j])
            if r:
                sig = '*' if r['p_value'] < 0.05 else ''
                ax.text(j, i, f'{r["spearman_rho"]:.2f}{sig}', ha='center', va='center',
                        fontsize=10, color='white' if abs(r['spearman_rho']) > 0.6 else 'black')

    plt.colorbar(im, ax=ax, label="Spearman ρ (α vs metric)")
    ax.set_title("Dose-Response Monotonicity: Spearman ρ (* = p < 0.05)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_spearman_monotonicity.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_direction_stability(stability_data):
    """§5.4: Per-layer split-half cosine for each model's corrective direction."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        if m not in stability_data:
            ax.set_title(f"{MODEL_LABELS[m]} — N/A")
            continue

        data = stability_data[m]
        per_layer = data["split_half_per_layer"]
        corrective_onset = data["corrective_onset"]

        layers = sorted([int(k) for k in per_layer.keys()])
        cosines = [per_layer[str(li)]["mean_cosine"] for li in layers]

        colors_layer = ['#e74c3c' if li >= corrective_onset else '#95a5a6' for li in layers]
        ax.bar(layers, cosines, color=colors_layer, alpha=0.7, width=0.8)
        ax.axhline(y=0.9, color='black', linestyle='--', alpha=0.5, label='Stability threshold (0.9)')
        ax.axvline(x=corrective_onset - 0.5, color='blue', linestyle=':', alpha=0.5, label='Corrective onset')

        s = data["corrective_summary"]
        ax.set_title(f"{MODEL_LABELS[m]} (mean={s['split_half_mean']:.3f})",
                     fontsize=11, fontweight='bold', color=COLORS[m])
        ax.set_xlabel('Layer')
        ax.set_ylabel('Split-half cosine')
        ax.set_ylim(0.8, 1.01)
        ax.grid(True, alpha=0.2, axis='y')
        if idx == 0:
            ax.legend(fontsize=7, loc='lower left')

    fig.suptitle('Direction Stability: Split-Half Worker Cosine Similarity per Layer',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_direction_stability.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_bootstrap_stability(boot_data):
    """§5.3: Bootstrap direction stability — pairwise cosine distribution per model."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        if m not in boot_data:
            ax.set_title(f"{MODEL_LABELS[m]} — N/A")
            continue

        d = boot_data[m]
        summary = d.get("summary", d)  # handle both flat and nested formats
        mean_pw = summary.get("mean_pairwise", summary.get("mean_pairwise_cosine", 0))
        min_pw = summary.get("min_pairwise", summary.get("min_pairwise_cosine", 0))
        mean_bvf = summary.get("mean_bootstrap_vs_full", summary.get("mean_boot_vs_full_cosine", 0))
        stable = summary.get("stable", False)

        # Summary bar chart: mean pairwise, min pairwise, mean boot-vs-full
        labels = ['Mean\npairwise', 'Min\npairwise', 'Mean\nboot-vs-full']
        vals = [mean_pw, min_pw, mean_bvf]
        colors_bar = [COLORS[m]] * 3
        ax.bar(labels, vals, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0.95, color='black', linestyle='--', alpha=0.5, label='Stability threshold (0.95)')
        ax.set_ylim(0, 1.05)

        status = "STABLE" if stable else "UNSTABLE"
        ax.set_title(f"{MODEL_LABELS[m]} ({status})",
                     fontsize=11, fontweight='bold', color=COLORS[m])
        ax.set_ylabel('Cosine Similarity')
        ax.grid(True, alpha=0.2, axis='y')
        if idx == 0:
            ax.legend(fontsize=7)

        # Annotate values
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Bootstrap Direction Stability (100 resamples × 200 records)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_bootstrap_stability.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_pca_spectrum(pca_data):
    """§5.10: PCA variance explained spectrum per model.

    Each panel: one model, bar chart of top-7 PC variance ratios per corrective layer,
    with mean across layers shown as line overlay.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    N_PCS = 7  # show top 7 components

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        if m not in pca_data:
            ax.set_title(f"{MODEL_LABELS[m]} — N/A")
            continue

        d = pca_data[m]
        per_layer = d["per_layer_pca"]
        corrective_layers = d.get("corrective_layers", [])

        # Collect variance explained across corrective layers
        all_varexp = []
        for li_str, layer_d in per_layer.items():
            if int(li_str) in corrective_layers:
                ve = layer_d["variance_explained"][:N_PCS]
                # Pad if fewer than N_PCS
                ve = ve + [0.0] * (N_PCS - len(ve))
                all_varexp.append(ve)

        if not all_varexp:
            # Fallback: use all layers
            for li_str, layer_d in per_layer.items():
                ve = layer_d["variance_explained"][:N_PCS]
                ve = ve + [0.0] * (N_PCS - len(ve))
                all_varexp.append(ve)

        all_varexp = np.array(all_varexp)  # (n_layers, N_PCS)
        mean_ve = all_varexp.mean(axis=0)
        std_ve = all_varexp.std(axis=0)

        x = np.arange(1, N_PCS + 1)
        ax.bar(x, mean_ve, color=COLORS[m], alpha=0.7, yerr=std_ve, capsize=3,
               error_kw={'linewidth': 1, 'alpha': 0.5})
        ax.axhline(y=0.6, color='black', linestyle='--', alpha=0.4, label='Rank-1 threshold (60%)')

        summary = d["summary"]
        ax.set_title(f"{MODEL_LABELS[m]} (PC1={summary['mean_pc1_variance_ratio']:.1%})",
                     fontsize=11, fontweight='bold', color=COLORS[m])
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained')
        ax.set_xticks(x)
        ax.set_ylim(0, 0.8)
        ax.grid(True, alpha=0.2, axis='y')
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle('PCA Spectrum of IT−PT MLP Differences at Corrective Layers',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_pca_spectrum.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_id_profiles(id_data):
    """§5.9: TwoNN intrinsic dimensionality profiles at α=1.0, 0.0, -1.0.

    Each panel: one model, 3 lines (one per α) showing ID across layers.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    alpha_colors = {'1.0': '#2196F3', '0.0': '#E41A1C', '-1.0': '#4CAF50'}
    alpha_labels = {'1.0': 'α=1.0 (baseline)', '0.0': 'α=0.0 (removal)', '-1.0': 'α=−1.0 (reversal)'}

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        if m not in id_data:
            ax.set_title(f"{MODEL_LABELS[m]} — N/A")
            continue

        profiles = id_data[m]["profiles"]
        for alpha_str in ['1.0', '0.0', '-1.0']:
            if alpha_str not in profiles:
                continue
            ids = profiles[alpha_str]["id_twonn"]
            layers = list(range(len(ids)))
            ax.plot(layers, ids, marker='.', markersize=3, linewidth=1.5,
                    color=alpha_colors[alpha_str], label=alpha_labels[alpha_str])

        ax.set_title(MODEL_LABELS[m], fontsize=11, fontweight='bold', color=COLORS[m])
        ax.set_xlabel('Layer')
        ax.set_ylabel('Intrinsic Dimensionality (TwoNN)')
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Intrinsic Dimensionality Under Corrective Direction Steering',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_id_profiles.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def plot_commitment_vs_alpha(commitment_data):
    """§5.8: Mean commitment layer vs α for each model.

    Single panel with all 6 models overlaid, α on x-axis, mean commitment layer on y-axis.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for m in MODELS:
        if m not in commitment_data:
            continue

        cond_data = commitment_data[m]
        alphas = []
        means = []
        stds = []
        for cond_name, cd in cond_data.items():
            a = cd.get("alpha")
            if a is not None:
                alphas.append(a)
                means.append(cd["mean_commitment"])
                stds.append(cd.get("std_commitment", 0))

        # Sort by alpha
        order = np.argsort(alphas)
        alphas = [alphas[i] for i in order]
        means = [means[i] for i in order]
        stds = [stds[i] for i in order]

        ax.plot(alphas, means, marker='o', markersize=4, linewidth=1.5,
                color=COLORS[m], label=MODEL_LABELS[m])
        ax.fill_between(alphas,
                        [m_ - s for m_, s in zip(means, stds)],
                        [m_ + s for m_, s in zip(means, stds)],
                        color=COLORS[m], alpha=0.1)

    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.4, label='α=1 (baseline)')
    ax.set_xlabel('α (correction strength)', fontsize=12)
    ax.set_ylabel('Mean Commitment Layer', fontsize=12)
    ax.set_title('Logit-Lens Commitment Layer vs Steering Strength',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "exp8_commitment_vs_alpha.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def main():
    print("Generating Exp8 Phase 0 plots...")
    all_scores = load_scores()
    all_judge = load_judge_scores()

    plot_cross_model_dose_response(all_scores)
    plot_selectivity_comparison(all_scores)
    plot_corrective_vs_random_dose(all_scores)
    plot_normalized_dose_response(all_scores)
    plot_judge_heatmap(all_judge)
    plot_dissociation_summary(all_scores)
    plot_random_control_bars(all_scores)

    # §5.5 and §5.6: Effect sizes and monotonicity
    effect_sizes = compute_effect_sizes(all_scores)
    monotonicity = compute_spearman_monotonicity(all_scores)
    plot_effect_size_table(effect_sizes)
    plot_monotonicity_table(monotonicity)

    # §5.4: Direction stability
    stability_path = DATA_DIR / "phase0_direction_stability.json"
    if stability_path.exists():
        plot_direction_stability(json.loads(stability_path.read_text()))
    else:
        print(f"  [skip] direction stability — run src.poc.exp8.direction_stability first")

    # §5.10: PCA spectrum
    pca_path = DATA_DIR / "phase0_pca_scree.json"
    if pca_path.exists():
        plot_pca_spectrum(json.loads(pca_path.read_text()))
    else:
        print(f"  [skip] PCA spectrum — no data")

    # §5.9: ID profiles
    id_path = DATA_DIR / "phase0_id_steering.json"
    if id_path.exists():
        plot_id_profiles(json.loads(id_path.read_text()))
    else:
        print(f"  [skip] ID profiles — no data")

    # §5.8: Commitment vs alpha
    cm_path = DATA_DIR / "phase0_commitment.json"
    if cm_path.exists():
        plot_commitment_vs_alpha(json.loads(cm_path.read_text()))
    else:
        print(f"  [skip] commitment vs alpha — no data")

    # §5.3: Bootstrap stability
    boot_path = DATA_DIR / "phase0_bootstrap_stability.json"
    if boot_path.exists():
        plot_bootstrap_stability(json.loads(boot_path.read_text()))
    else:
        print(f"  [skip] bootstrap stability — no data")

    print(f"\nAll plots → {PLOTS_DIR}")


if __name__ == "__main__":
    main()
