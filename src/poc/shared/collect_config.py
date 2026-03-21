"""
CollectionConfig — unified configuration for all structural-semantic experiments.

Supersedes Exp3Config and Exp4Config.  The existing per-experiment configs remain
for backward compatibility, but new runs should use CollectionConfig directly via
`uv run python -m src.poc.collect`.

Architecture note
-----------------
Hook paths are specified per model family via ModelHooks so that switching to a
different model (e.g. Gemma 2 2B, Gemma 3 12B) requires only a different ModelHooks
instance, not changes to the collection logic.

Output directory
----------------
results/{run_name}/
    results.jsonl       — one JSON record per line (streaming, resumable)
    features.npz        — generation-mode active features {record_id: [n_steps, n_layers]}
    enc_features.npz    — encode-mode active features    {record_id: [n_layers]}
    enc_residuals.npz   — encode-mode dense residuals    {record_id: [n_layers, d_model]}
    run_config.json     — full config snapshot for reproducibility
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Model-family hook descriptors
# ---------------------------------------------------------------------------

@dataclass
class ModelHooks:
    """Layer-relative nnsight hook paths.

    All paths are relative to ``model.{layers_root}[i]``.
    Override for different model families.
    """
    # Residual stream after full layer (MLP + self-attn + residual adds)
    layer_residual: str = "output[0]"

    # Pre-MLP LayerNorm output — input to the MLP (and to the transcoder)
    mlp_input: str = "pre_feedforward_layernorm.output"

    # MLP module output (before residual add); needed for transcoder MSE
    mlp_output: str = "mlp.output"

    # Self-attention weights; index [1] of output tuple when output_attentions=True
    attn_weights: str = "self_attn.output[1]"

    # Path from model root to the layer list: model.{layers_root}[i]
    layers_root: str = "language_model.layers"

    # Final RMSNorm before lm_head: model.{final_norm}
    final_norm: str = "language_model.norm"

    # LM head: model.{lm_head}
    lm_head: str = "lm_head"

    # Attention implementation required for weight capture ("eager" or "sdpa")
    attn_implementation: str = "eager"


# Pre-defined hook sets for known model families
GEMMA3_4B_HOOKS  = ModelHooks()                           # default — Gemma 3 4B (PT and IT)
GEMMA2_2B_HOOKS  = ModelHooks()                           # same layout, verify before use
GEMMA2_9B_HOOKS  = ModelHooks()                           # same layout, verify before use


def hooks_for_model(model_id: str) -> ModelHooks:
    """Return the appropriate ModelHooks for a known model_id.

    Falls back to the Gemma 3 4B hooks for unknown models — check hook paths
    manually before running on an unfamiliar architecture.
    """
    mid = model_id.lower()
    if "gemma-3" in mid or "gemma3" in mid:
        return GEMMA3_4B_HOOKS
    if "gemma-2" in mid or "gemma2" in mid:
        return GEMMA2_2B_HOOKS
    # Unknown — return defaults and log a warning at runtime
    return ModelHooks()


# ---------------------------------------------------------------------------
# Transcoder registry
# ---------------------------------------------------------------------------

# Maps (model_id_fragment, variant) to the HuggingFace transcoder release.
# Add entries here when new transcoder releases become available.
_TRANSCODER_REGISTRY: dict[tuple[str, str], str] = {
    ("gemma-3-4b", "pt"): "google/gemma-scope-2-4b-pt",
    ("gemma-3-4b", "it"): "google/gemma-scope-2-4b-it",
    ("gemma-2-2b", "pt"): "google/gemma-scope-2b-pt",
    ("gemma-2-2b", "it"): "google/gemma-scope-2b-it",
}


def transcoder_release_for(model_id: str, variant: str) -> str:
    """Look up the default transcoder release for a model.  Returns '' if unknown."""
    mid = model_id.lower()
    for (frag, v), release in _TRANSCODER_REGISTRY.items():
        if frag in mid and v == variant:
            return release
    return ""


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class CollectionConfig:
    """Unified configuration for structural-semantic feature collection.

    Parameters are grouped by concern.  Derived fields (output_dir, run_dir,
    model_name, transcoder_set, corrective_layers, …) are computed in
    __post_init__ and should not be set directly.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    run_name: str = ""
    """Human-readable name for this run.  Auto-derived from model/variant/flags
    if left empty.  Used as the output subdirectory name."""

    # ── Model ─────────────────────────────────────────────────────────────────
    model_variant: str = "it"
    """'pt' (pre-trained base) or 'it' (instruction-tuned).  Controls which
    HuggingFace model ID and transcoder release to use when the explicit
    model_id / transcoder_release fields are left as defaults."""

    model_id: str = ""
    """Explicit HuggingFace model ID.  Auto-derived from model_variant if empty.
    Set explicitly to use a non-standard checkpoint (e.g. fine-tuned variant)."""

    transcoder_release: str = ""
    """HuggingFace transcoder release.  Auto-derived from model_id + variant."""

    transcoder_variant: str = "width_16k_l0_big_affine"
    """Which transcoder width/variant to load from the release."""

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_path: str = "data/exp3_dataset.jsonl"
    """Path to the JSONL dataset (one record per line, format from build_dataset.py)."""

    splits: list[str] | None = None
    """Which splits to collect.  None = all splits in the dataset.
    Example: ['F', 'R'] to collect only Factual and Reasoning prompts."""

    prompt_format: str = "B"
    """Which format field to use from each record's 'formats' dict.
    'B' = 'Question: …\\nAnswer:' (default, same for PT and IT, confound-controlled).
    'A' = raw completion format (natural for PT)."""

    apply_chat_template: bool = False
    """If True, IT prompts are wrapped with the model's chat template.
    Default False — both PT and IT receive the raw prompt for confound-controlled
    comparison.  Set True for IT-native behaviour analysis."""

    # ── Collection modes ──────────────────────────────────────────────────────
    mode: set[str] = field(default_factory=lambda: {"generate"})
    """Which collection modes to run.  Options:
      'generate' — autoregressive generation; collects per-step per-layer metrics.
                   Required for exp3-style corrective stage analysis.
      'encode'   — single forward pass on the prompt; collects last-token metrics.
                   Required for exp4-style phase-transition and TwoNN analysis.
    Both modes can be run in one pass: mode={'generate', 'encode'}."""

    # ── Generation settings ────────────────────────────────────────────────────
    max_gen_tokens: int = 200
    """Maximum number of tokens to generate per prompt (generation mode)."""

    ablation_layers: list[int] = field(default_factory=list)
    """Layer indices whose MLP output is zeroed during generation.
    Empty list = no ablation (normal collection).
    Example: list(range(20, 34)) for full corrective-stage ablation."""

    # ── Metric flags ──────────────────────────────────────────────────────────
    collect_emergence: bool = True
    """Collect next_token_rank, next_token_prob, kl_to_final (generation mode).
    Cost: one O(V) sort per layer per step (~10% overhead)."""

    collect_attribution: bool = True
    """Collect logit_delta_contrib: per-layer logit difference for generated token.
    Cost: negligible (one scalar subtract per layer per step)."""

    collect_transcoder_mse: bool = False
    """Collect ||transcoder(x) − mlp(x)||² per layer per step (generation mode).
    Requires an additional nnsight hook for mlp.output.  Disabled by default
    until hook paths are verified for each model family."""

    collect_attention: bool = True
    """Collect attention entropy per head per layer (encode mode).
    Requires attn_implementation='eager' and adds ~20% overhead to encode pass."""

    collect_residuals: bool = False
    """Save full residual vectors [n_layers, d_model] per record (encode mode).
    Large disk usage (~700 KB per record at bfloat16).  Required for TwoNN
    intrinsic-dimension analysis.  Disabled by default."""

    # ── Architecture ──────────────────────────────────────────────────────────
    n_layers: int = 34
    """Number of transformer layers."""

    d_model: int = 2560
    """Residual stream dimensionality."""

    n_heads: int = 8
    """Number of attention heads per layer."""

    proposal_boundary: int = 20
    """Layer index dividing proposal stage [0, boundary) from corrective stage
    [boundary, n_layers).  Used in plots and attribution analyses."""

    hooks: ModelHooks = field(default_factory=ModelHooks)
    """Model-family hook paths.  Auto-set from model_id in __post_init__."""

    # ── Output ────────────────────────────────────────────────────────────────
    output_base: str = "results"
    """Base directory for all results.  Full path = {output_base}/{run_name}."""

    checkpoint_every: int = 100
    """Save NPZ checkpoint every N records (0 = only at end)."""

    # ── Hardware ──────────────────────────────────────────────────────────────
    n_gpus: int = 1
    """Number of GPUs to use.  > 1 distributes records across GPUs via
    multiprocessing (each worker loads its own model copy)."""

    device: str = "cuda"
    dtype_str: str = "bfloat16"

    # ── Derived fields (set in __post_init__, do not pass to constructor) ──────
    model_name: str = field(init=False)
    transcoder_set: str = field(init=False)

    def __post_init__(self) -> None:
        if self.model_variant not in ("pt", "it"):
            raise ValueError(
                f"model_variant must be 'pt' or 'it', got {self.model_variant!r}"
            )
        if isinstance(self.mode, (list, tuple)):
            self.mode = set(self.mode)
        valid_modes = {"generate", "encode"}
        unknown = self.mode - valid_modes
        if unknown:
            raise ValueError(f"Unknown mode(s): {unknown}. Valid: {valid_modes}")

        # Derive model_id and transcoder_release from variant if not explicit
        if not self.model_id:
            self.model_id = f"google/gemma-3-4b-{self.model_variant}"
        if not self.transcoder_release:
            self.transcoder_release = transcoder_release_for(
                self.model_id, self.model_variant
            )
            if not self.transcoder_release:
                self.transcoder_release = (
                    f"google/gemma-scope-2-4b-{self.model_variant}"
                )

        # Set derived fields for backward-compat with code that reads Exp3Config fields
        self.model_name     = self.model_id
        self.transcoder_set = self.transcoder_release

        # Hook paths from model family
        self.hooks = hooks_for_model(self.model_id)

        # Auto-derive run_name
        if not self.run_name:
            variant = self.transcoder_variant.replace("width_", "")
            suffix = "_chat" if self.apply_chat_template else ""
            mode_tag = ""
            if self.mode == {"encode"}:
                mode_tag = "_enc"
            elif self.mode == {"generate", "encode"}:
                mode_tag = "_full"
            abl_tag = f"_abl{'_'.join(str(l) for l in self.ablation_layers)}" \
                      if self.ablation_layers else ""
            self.run_name = (
                f"{self.model_variant}_{variant}_t{self.max_gen_tokens}"
                f"{suffix}{mode_tag}{abl_tag}"
            )

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def output_dir(self) -> str:
        return f"{self.output_base}/{self.run_name}"

    @property
    def corrective_layers(self) -> range:
        return range(self.proposal_boundary, self.n_layers)

    @property
    def proposal_layers(self) -> range:
        return range(0, self.proposal_boundary)

    @property
    def is_instruction_tuned(self) -> bool:
        return self.model_variant == "it"

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for run_config.json)."""
        import dataclasses
        d = dataclasses.asdict(self)
        d["mode"] = sorted(self.mode)  # sets aren't JSON-serializable
        return d

    @classmethod
    def from_exp3_config(cls, exp3_cfg: object) -> "CollectionConfig":
        """Convert an Exp3Config to a CollectionConfig for backward compatibility."""
        return cls(
            model_variant=exp3_cfg.model_variant,
            model_id=exp3_cfg.model_name,
            transcoder_release=exp3_cfg.transcoder_set,
            transcoder_variant=exp3_cfg.transcoder_variant,
            mode={"generate"},
            max_gen_tokens=exp3_cfg.max_gen_tokens,
            apply_chat_template=getattr(exp3_cfg, "apply_chat_template", False),
            ablation_layers=getattr(exp3_cfg, "ablation_layers", []),
            collect_emergence=getattr(exp3_cfg, "collect_emergence", True),
            collect_attribution=getattr(exp3_cfg, "collect_attribution", True),
            collect_transcoder_mse=getattr(exp3_cfg, "collect_transcoder_mse", False),
            n_gpus=getattr(exp3_cfg, "n_gpus", 1),
            device=exp3_cfg.device,
            dtype_str=exp3_cfg.dtype_str,
        )
