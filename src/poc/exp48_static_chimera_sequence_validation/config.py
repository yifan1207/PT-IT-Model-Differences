"""Configuration for Exp48 static chimera sequence validation.

The experiment is intentionally Llama-3.1 same-base only: the point is to test
whether static late-stack swaps and low-rank upstream rescue behave like the
same-base recipe-specificity effects measured in Exp47.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS = (
    "llama31_meta_instruct",
    "llama31_tulu3_sft",
    "llama31_tulu3_dpo",
    "llama31_tulu3_final",
    "llama31_openmath2",
)

INSTRUCTION_LIKE = {
    "llama31_meta_instruct",
    "llama31_tulu3_sft",
    "llama31_tulu3_dpo",
    "llama31_tulu3_final",
    "llama31_hermes3",
}

OPENMATH = "llama31_openmath2"

BOUNDARIES = (16, 19, 24, 29, 31)
PRIMARY_BOUNDARY = 19
COMPONENT_BOUNDARIES = (19, 29, 31)
CONTROL_BOUNDARIES = (19, 29, 31)

STATIC_CELLS = ("BB", "BF", "FB", "FF")
COMPONENT_VARIANTS = ("blocks_plus_head", "blocks_only", "head_only")
DECOMPOSITION_VARIANTS = ("mlp_only", "attn_only")
CONTROL_VARIANTS = ("wrong_descendant", "permuted_blocks", "interpolated_late")
INTERPOLATION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

RESCUE_KS = ("1", "4", "16", "64", "256", "full")
RESCUE_ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5)
RESCUE_CONTROLS = (
    "paired_pca",
    "mean_delta",
    "random_full",
    "random_delta_span",
    "shuffled_delta",
    "sign_flip",
    "gaussian",
    "wrong_descendant",
)

DATASET = Path("data/eval_dataset_v2.jsonl")
DEFAULT_EXP47_ROOT = Path("results/exp47_same_base_recipe_specificity")


@dataclass(frozen=True)
class SequenceJob:
    model: str
    boundary: int
    scenario: str
    component: str
    cell: str
    alpha: float | None = None
    wrong_model: str | None = None

    @property
    def stem(self) -> str:
        parts = [
            self.model,
            f"b{self.boundary}",
            self.scenario,
            self.component,
            self.cell,
        ]
        if self.alpha is not None:
            parts.append(f"a{self.alpha:g}".replace(".", "p"))
        if self.wrong_model:
            parts.append(f"wrong_{self.wrong_model}")
        return "__".join(parts)


def wrong_descendant_for(model: str, models: tuple[str, ...] = DEFAULT_MODELS) -> str:
    """Deterministic same-base wrong-descendant control."""
    if model not in models:
        return models[0]
    return models[(models.index(model) + 1) % len(models)]


def prompt_split(prompt_id: str, *, train_fraction: float = 0.7, salt: str = "exp48") -> str:
    """Stable held-out split used by both sequence and rescue phases."""
    import hashlib

    digest = hashlib.sha256(f"{salt}:{prompt_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    return "train" if value < train_fraction else "heldout"

