"""
Canonical split/category color palette shared across all experiment plots.

Supports both old split names (IC/R/OOC/GEN) and new names (F/R/OOD/A)
as well as subcategory names (4a–4e, in_context, out_of_context, …).

Usage:
    from src.poc.shared.plot_colors import split_color, SPLIT_COLORS

    color = split_color("F")        # new name
    color = split_color("IC")       # old name → same color
    color = split_color("4a")       # alignment subcategory
"""
from __future__ import annotations

# Primary split palette
# New names: F, R, OOD, A
# Old names: IC, R, OOC, GEN  (preserved for backward compat with existing results)
SPLIT_COLORS: dict[str, str] = {
    # ── New taxonomy ───────────────────────────────────────
    "F":    "#2196F3",   # Factual          — blue
    "R":    "#4CAF50",   # Reasoning        — green
    "OOD":  "#FF9800",   # Out-of-Distrib   — orange
    "A":    "#E91E63",   # Alignment        — pink/red

    # ── Old taxonomy (backward compat) ─────────────────────
    "IC":              "#2196F3",   # In-Context → same as F
    "in_context":      "#2196F3",
    "OOC":             "#FF9800",   # Out-of-Context → same as OOD
    "out_of_context":  "#FF9800",
    "GEN":             "#9C27B0",   # General → purple (retired split)
    "generation":      "#9C27B0",
    # R is the same in both taxonomies
    "rule":            "#4CAF50",
    "reasoning":       "#4CAF50",

    # ── Alignment subcategories ─────────────────────────────
    "4a":  "#D32F2F",   # Harmful        — dark red
    "4b":  "#FF7043",   # Borderline     — deep orange
    "4c":  "#FBC02D",   # Format         — amber
    "4d":  "#7B1FA2",   # Conversational — purple
    "4e":  "#78909C",   # Raw completion — grey-blue

    # ── Question types ──────────────────────────────────────
    "factual":       "#2196F3",
    "numerical":     "#4CAF50",
    "yes_no":        "#8BC34A",
    "multi_choice":  "#00BCD4",
    "code":          "#607D8B",
    "fictional":     "#FF9800",
    "unknowable":    "#FF5722",
    "counterfactual":"#FF9800",
    "harmful":       "#D32F2F",
    "borderline":    "#FF7043",
    "format":        "#FBC02D",
    "continuation":  "#78909C",
    "conversation":  "#7B1FA2",
    "roleplay":      "#AB47BC",
    "refusal_help":  "#CE93D8",
    "completion":    "#78909C",
    "coreference":   "#00ACC1",
}

_DEFAULT_COLOR = "#888888"


def split_color(category: str) -> str:
    """Return the canonical hex color for a split or category name."""
    return SPLIT_COLORS.get(category, _DEFAULT_COLOR)


# Ordered list of splits for consistent legend ordering
SPLIT_ORDER = ["F", "R", "OOD", "A", "IC", "OOC", "GEN"]

# Line styles for PT vs IT overlays
PT_LINESTYLE = "--"
IT_LINESTYLE = "-"
PT_ALPHA     = 0.6
IT_ALPHA     = 1.0
