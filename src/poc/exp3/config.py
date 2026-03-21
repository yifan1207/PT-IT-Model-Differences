"""
Configuration for Experiment 3: The Corrective Computational Stage.

Core claim: Instruction tuning creates a corrective computational stage in
late layers (~20–33) that actively opposes the raw PT prediction to enforce
format, helpfulness, and safety.

Collection modes
----------------
Exp3 extends exp2's collection with additional per-layer quantities derived
from the same nnsight trace.  Most new quantities require NO additional hooks
— they are computed post-trace from the residuals already captured in exp2.

New per-step, per-layer quantities:
  next_token_rank[step][layer]      rank of the generated token in the logit-lens
                                    distribution at layer i (1 = already top-1)
  next_token_prob[step][layer]      probability of that token under logit-lens at layer i
  logit_delta_contrib[step][layer]  logit_lens[i][token] - logit_lens[i-1][token]
                                    (how much layer i pushed the prediction toward
                                    or away from the token we actually generated)
  kl_to_final[step][layer]          KL( softmax(logit_lens_i) ∥ softmax(logit_lens_33) )
                                    (how far layer i's belief is from the final prediction)
  transcoder_mse[step][layer]       ||transcoder(x_i) - mlp(x_i)||² / d_model
                                    REQUIRES new nnsight hook — see collect.py note.

Collection flags (set on Exp3Config) control which quantities are computed.
Disabled flags skip that computation but do not affect other quantities.

Intervention modes
------------------
  ablation_layers: list of layer indices whose MLP output is zeroed during
                   generation.  Implemented in interventions/ablation.py.
  raw_completion:  if True, IT model is run WITHOUT the chat template (treats
                   prompts as plain completion, same as PT).  Tests whether the
                   corrective stage is weight-driven or format-triggered.

Stage boundary
--------------
  proposal_end_layer: int  (default 20)
    Layers 0 .. proposal_end_layer-1  = "proposal stage"  (raw prediction)
    Layers proposal_end_layer .. 33   = "corrective stage" (governance / alignment)
    Used by alignment_tax.py and by all analyses that split by stage.
"""
from dataclasses import dataclass, field


@dataclass
class Exp3Config:
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

    # ── generation ────────────────────────────────────────────────────
    max_gen_tokens: int = 512

    # ── parallelism ───────────────────────────────────────────────────
    n_gpus: int = 8

    # ── collection flags ──────────────────────────────────────────────
    # Core exp2 quantities are always collected.
    # These flags enable additional exp3 quantities.
    collect_emergence: bool = True
    # Enables: next_token_rank, next_token_prob, kl_to_final
    # Cost: one extra O(V) sort per layer per step — ~10% slower.

    collect_attribution: bool = True
    # Enables: logit_delta_contrib[step][layer]
    # Cost: one extra scalar lookup per layer per step — negligible.

    collect_transcoder_mse: bool = False
    # Enables: transcoder_mse[step][layer]
    # REQUIRES a new nnsight hook for mlp.output — see collect.py for details.
    # Disabled by default until hook path is verified on the live model.

    # ── intervention mode ─────────────────────────────────────────────
    apply_chat_template: bool = False
    # Primary analysis (exp3 default): False.
    # Both PT and IT receive the prompt string as-is (Format B: "Question: ...\nAnswer:").
    # This is the confound-controlled comparison — same input format, different weights.
    #
    # Set True for the IT-native secondary analysis, where IT gets the Gemma chat
    # template wrapper.  Use --chat-template on the CLI.

    raw_completion: bool = False
    # Kept for exp 0b: IT model receives prompts WITHOUT chat template, same as PT.
    # apply_chat_template=False already achieves this for the primary analysis.
    # raw_completion=True is preserved for backwards compatibility with exp2 prompts.

    ablation_layers: list = field(default_factory=list)
    # Layer indices whose MLP output is set to zero during generation.
    # e.g. [25, 26, 27, 28, 29, 30, 31, 32, 33] for corrective-stage ablation.
    # Implemented in interventions/ablation.py, not in standard collect.py.

    # ── stage boundary ────────────────────────────────────────────────
    proposal_end_layer: int = 20
    # Layers [0, proposal_end_layer)  = proposal stage
    # Layers [proposal_end_layer, 34) = corrective stage

    # ── output paths (derived in __post_init__) ───────────────────────
    output_path: str = field(init=False)
    plot_path:   str = field(init=False)
    feature_importance_path: str = field(init=False)

    def __post_init__(self) -> None:
        if self.model_variant not in ("pt", "it"):
            raise ValueError(
                f"model_variant must be 'pt' or 'it', got {self.model_variant!r}"
            )
        self.model_name     = f"google/gemma-3-4b-{self.model_variant}"
        self.transcoder_set = f"google/gemma-scope-2-4b-{self.model_variant}"
        rd = self.run_dir
        self.output_path = f"{rd}/exp3_results.json"
        self.plot_path   = f"{rd}/plots"
        self.feature_importance_path = f"{rd}/feature_importance_summary.npz"

    @property
    def run_dir(self) -> str:
        variant = self.transcoder_variant.replace("width_", "")
        # Encode chat_template in path so secondary analysis runs never overwrite
        # the confound-controlled (default) results.
        if self.apply_chat_template:
            suffix = "_chat"
        elif self.raw_completion:
            suffix = "_raw"
        else:
            suffix = ""
        return f"results/exp3/{self.model_variant}_{variant}_t{self.max_gen_tokens}{suffix}"

    @property
    def is_instruction_tuned(self) -> bool:
        return self.model_variant == "it"

    @property
    def corrective_layers(self) -> range:
        """Range of layer indices considered part of the corrective stage."""
        return range(self.proposal_end_layer, 34)

    @property
    def proposal_layers(self) -> range:
        """Range of layer indices considered part of the proposal stage."""
        return range(0, self.proposal_end_layer)
