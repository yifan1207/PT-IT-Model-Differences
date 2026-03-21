"""
Configuration for Experiment 4: Phase Transition Characterisation.

Core claim: IT sharpens the pre-existing representational phase transition at
layer ~11 (the L0 dip) through attention-mediated reorganisation, without
modifying the gate's own MLP weights.

This experiment tests three predictions from the hypothesis document:
  P1 — Feature Population Shift  (E0b: adjacent-layer Jaccard across dip)
  P2 — Attention Divergence at the Dip (E0a: attention entropy + divergence)
  P5 — Feature Category Transition   (E1a: Neuronpedia feature label analysis)
  P6 — ID Geometry Confirmation       (E0c: TwoNN intrinsic dimension profile)

Data collection: SINGLE FORWARD PASS per prompt (no generation loop).
We analyse how the model processes and represents the prompt itself, not
what it generates.  For generation-level analysis, use exp3 results.

Quantities collected
--------------------
  collect_residuals : bool  (default True)
      Saves last-token residual vector at every layer [d_model].
      Stacked across prompts to [n_prompts, n_layers, d_model] in .npz.
      Used by analysis/intrinsic_dim.py for TwoNN estimation.

  collect_attention : bool  (default True)
      Saves per-layer, per-head attention entropy across all layers.
      Also saves dense attention weight matrices at attn_save_layers for
      full PT-vs-IT divergence analysis.
      Requires attn_implementation="eager" at model load (set in model.py).
      Entropy saved in main results JSON; dense weights in attn_weights.npz.

  collect_features : bool   (default True)
      Records active transcoder feature sets at every layer (last-token
      position) for adjacent-layer Jaccard analysis.
      Saved to exp4_features.npz (same format as exp3).

Dip-layer settings
------------------
  dip_layer   : int  (default 11)
      The single gate layer where the L0 discontinuity appears.

  attn_save_layers : list[int]  (default 6–14)
      Layers at which to save full attention weight matrices.
      Layers outside this list still contribute to entropy statistics
      (entropy is cheap to compute on the fly), but dense matrices are
      only saved here to control memory.

  analysis_range : tuple[int, int]  (default (8, 15))
      Half-open range [start, stop) of layers for Jaccard analysis.
      Covers layers 8–14 inclusive by default.
"""
from dataclasses import dataclass, field


@dataclass
class Exp4Config:
    # ── model variant ─────────────────────────────────────────────────
    model_variant: str = "pt"   # "pt" or "it"

    # Derived in __post_init__ — do not set directly.
    model_name:     str = field(init=False)
    transcoder_set: str = field(init=False)

    # ── transcoder / backend ──────────────────────────────────────────
    transcoder_variant: str = "width_16k_l0_big_affine"
    backend:   str = "nnsight"
    dtype_str: str = "bfloat16"
    device:    str = "cuda"

    # ── collection flags ──────────────────────────────────────────────
    collect_residuals: bool = True
    # Saves last-token residual vector per layer per prompt.

    collect_attention: bool = True
    # Saves attention entropy per head per layer per prompt.
    # Also saves dense attention matrices at attn_save_layers.
    # Requires model loaded with attn_implementation="eager".

    collect_features: bool = True
    # Saves active transcoder feature indices per layer per prompt.

    # ── dip layer settings ────────────────────────────────────────────
    dip_layer: int = 11
    # Layer index where the L0 discontinuity appears (hypothesis: ~11).

    attn_save_layers: list = field(default_factory=lambda: list(range(6, 15)))
    # Layers [6..14] for dense attention weight saving.

    analysis_range: tuple = (8, 15)
    # [start, stop) for Jaccard analysis.  Default covers layers 8–14.

    # ── output paths (derived in __post_init__) ───────────────────────
    output_path:    str = field(init=False)
    residuals_path: str = field(init=False)
    features_path:  str = field(init=False)
    attn_path:      str = field(init=False)
    plot_path:      str = field(init=False)

    def __post_init__(self) -> None:
        if self.model_variant not in ("pt", "it"):
            raise ValueError(
                f"model_variant must be 'pt' or 'it', got {self.model_variant!r}"
            )
        self.model_name     = f"google/gemma-3-4b-{self.model_variant}"
        self.transcoder_set = f"google/gemma-scope-2-4b-{self.model_variant}"
        rd = self.run_dir
        self.output_path    = f"{rd}/exp4_results.json"
        self.residuals_path = f"{rd}/exp4_residuals.npz"
        self.features_path  = f"{rd}/exp4_features.npz"
        self.attn_path      = f"{rd}/exp4_attn.npz"
        self.plot_path      = f"{rd}/plots"

    @property
    def run_dir(self) -> str:
        variant = self.transcoder_variant.replace("width_", "")
        return f"results/exp4/{self.model_variant}_{variant}"

    @property
    def is_instruction_tuned(self) -> bool:
        return self.model_variant == "it"

    @property
    def pre_dip_layer(self) -> int:
        """Layer just before the dip."""
        return self.dip_layer - 1

    @property
    def post_dip_layer(self) -> int:
        """Layer just after the dip."""
        return self.dip_layer + 1
