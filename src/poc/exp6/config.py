from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


SteeringMethod = Literal[
    "none",
    "directional_remove",   # A1: h' = h - (1-α) * proj(h, v̂)  — remove corrective dir from IT
    "directional_add",      # A2: h' = h + β * d * ‖h‖           — inject direction into PT
    "directional_random",   # Control: inject a random unit vector
    "directional_rotated",  # Control: inject a 90°-rotated corrective direction
    "content_direction",    # Control: inject the content-layer IT-PT direction
    "feature_clamp",        # B1: clamp governance features at γ × mean_activation
    "wdec_inject",          # B2: inject W_dec-projected governance direction
]

FeatureLayerRange = Literal["all", "early_20_25", "mid_26_29", "late_30_33"]


@dataclass
class Exp6Config:
    """Configuration for Experiment 6 (steering experiments).

    Two approaches:
      A — Direction-level steering (nnsight trace, no transcoders needed).
          A1: remove corrective direction from IT  (directional_remove, α sweep)
          A2: add corrective direction to PT        (directional_add, β sweep)
          Controls: random/rotated/content-layer directions
      B — Feature-level steering (plain forward hooks, transcoders required).
          B1: clamp governance features at γ × mean_activation
          B2: project governance features → residual space via W_dec, inject
          B3: control features (random / content / IT-suppressed)
          B4: layer-range specificity
    """

    # ── Experiment identity ────────────────────────────────────────────────
    experiment: str = "A1"          # "A1" | "A2" | "B1" | "B2" | "B3" | "B4"
    model_variant: str = "it"       # "it" | "pt"
    model_id: str = ""              # auto-derived from variant

    # ── Dataset & prompt ──────────────────────────────────────────────────
    dataset_path: str = "data/exp6_dataset.jsonl"
    prompt_format: str = "B"
    apply_chat_template: bool = True   # IT gets chat template; PT does not
    max_gen_tokens: int = 200
    n_eval_examples: int = 1000       # use all 1000 exp6 prompts by default
    batch_size: int = 1

    # ── Output ────────────────────────────────────────────────────────────
    output_base: str = "results/exp6"
    run_name: str = ""              # auto-derived if empty

    # ── Model architecture ────────────────────────────────────────────────
    n_layers: int = 34
    d_model: int = 2560
    proposal_boundary: int = 20
    backend: str = "nnsight"
    device: str = "cuda"
    dtype_str: str = "bfloat16"

    # ── Direction-steering parameters (Approach A) ────────────────────────
    method: SteeringMethod = "none"
    ablation_layers: list[int] = field(default_factory=list)
    directional_alpha: float = 1.0   # for directional_remove: α (1.0 = baseline, 0.0 = full remove)
    directional_beta: float = 0.0    # for directional_add: β injection magnitude

    # Precomputed direction paths (same format as Exp5 corrective_directions.npz)
    corrective_direction_path: str = ""   # IT-PT diff at corrective layers (layers 20-33)
    content_direction_path: str = ""      # IT-PT diff at content layers (layers 0-11), for control

    # Random direction seeds (one per layer, for reproducibility)
    random_direction_seed: int = 42

    # ── Feature-steering parameters (Approach B) ──────────────────────────
    skip_transcoders: bool = True    # False for B experiments (adds ~11 GB)
    transcoder_release: str = ""     # auto-derived from variant
    transcoder_variant: str = "width_16k_l0_big_affine"

    gamma: float = 1.0               # feature clamp magnitude: γ × mean_activation
    feature_set: str = "method12_top100"  # key in governance_feature_sets.json
    feature_layer_range: FeatureLayerRange = "all"

    governance_features_path: str = ""   # results/exp6/governance_feature_sets.json
    mean_feature_acts_path: str = ""     # results/exp6/precompute/mean_feature_acts_it/
    governance_direction_path: str = ""  # results/exp6/precompute/governance_directions.npz

    # ── Benchmarks ────────────────────────────────────────────────────────
    benchmarks: list[str] = field(default_factory=lambda: [
        "coherent_assistant_rate",  # primary governance: IT ~90% vs PT ~10%, deterministic
        "structural_token_ratio",
        # turn_structure removed: confounded by response length and web training artifacts
        "format_compliance",
        "mmlu_accuracy",   # replaces factual_em: standard multiple-choice, no LLM judge
        "reasoning_em",
        "alignment_behavior",
    ])

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = f"google/gemma-3-4b-{self.model_variant}"
        if not self.transcoder_release:
            self.transcoder_release = f"google/gemma-scope-2-4b-{self.model_variant}"
        if not self.run_name:
            self.run_name = self._derive_run_name()
        # IT model gets chat template by default; PT never does.
        if self.model_variant == "pt":
            object.__setattr__(self, "apply_chat_template", False)

    def _derive_run_name(self) -> str:
        method_tag = self.method
        if self.method == "directional_remove":
            method_tag = f"dir_remove_a{self.directional_alpha:g}"
        elif self.method == "directional_add":
            method_tag = f"dir_add_b{self.directional_beta:g}"
        elif self.method == "feature_clamp":
            method_tag = f"feat_clamp_g{self.gamma:g}_{self.feature_set}"
        elif self.method == "wdec_inject":
            method_tag = f"wdec_b{self.directional_beta:g}_{self.feature_set}"
        layer_tag = f"_lr{self.feature_layer_range}" if self.feature_layer_range != "all" else ""
        return f"{self.experiment}_{self.model_variant}_{method_tag}{layer_tag}_t{self.max_gen_tokens}"

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Alias expected by shared/model.py."""
        return self.model_id

    @property
    def transcoder_set(self) -> str:
        """Alias expected by shared/model.py."""
        return self.transcoder_release

    @property
    def run_dir(self) -> Path:
        return Path(self.output_base) / self.run_name

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    @property
    def corrective_layers(self) -> list[int]:
        match self.feature_layer_range:
            case "early_20_25":
                return list(range(20, 26))
            case "mid_26_29":
                return list(range(26, 30))
            case "late_30_33":
                return list(range(30, 34))
            case _:
                return list(range(self.proposal_boundary, self.n_layers))

    @property
    def proposal_layers(self) -> list[int]:
        return list(range(0, self.proposal_boundary))

    def to_dict(self) -> dict:
        return asdict(self)
