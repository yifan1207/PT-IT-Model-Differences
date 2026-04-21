"""
Configuration for Experiment 2: IC vs OOC vs R mechanistic comparison.

Hypothesis under test:
  There is no mechanistically special "reasoning mode" in a pretrained LM.
  Feature activation patterns, residual stream dynamics, and layer-wise
  computation profiles are statistically indistinguishable between IC, OOC,
  and R task categories.

Unlike Exp1 (single next-token attribution), Exp2:
  - Generates up to max_gen_tokens tokens per prompt (autoregressive)
  - Captures per-layer, per-token residual stream and MLP statistics
  - Captures feature L0 (active feature count) per layer via transcoders
  - Computes logit-lens entropy at each layer

Model variants
--------------
  "pt"  google/gemma-3-4b-pt   Pretrained base model. Raw next-token completion.
                                Prompts are plain text; no chat template applied.
                                Stop token: <eos> only.
                                Transcoders: google/gemma-scope-2-4b-pt

  "it"  google/gemma-3-4b-it   Instruction-tuned (RLHF) model. Responds to
                                conversational turns. Prompts are wrapped in the
                                Gemma chat template before tokenisation.
                                Stop tokens: <eos> AND <end_of_turn>.
                                Transcoders: google/gemma-scope-2-4b-it
                                NOTE: circuit-tracer zeros out the first 4 token
                                positions (<bos><start_of_turn>user\\n) for -it
                                models to avoid artefacts at those positions.
"""
from dataclasses import dataclass, field


@dataclass
class Exp2Config:
    # ── model variant ────────────────────────────────────────────────
    # "pt" = pretrained base  |  "it" = instruction-tuned
    # model_name and transcoder_set are derived from this in __post_init__.
    model_variant: str = "pt"

    # Derived in __post_init__ — do not set directly.
    model_name:     str = field(init=False)
    transcoder_set: str = field(init=False)

    # ── transcoder / backend ─────────────────────────────────────────
    transcoder_variant: str = "width_16k_l0_big_affine"
    backend:   str = "nnsight"
    dtype_str: str = "bfloat16"   # "float32" | "bfloat16"
    device:    str = "cuda"

    # ── generation ───────────────────────────────────────────────────
    # Safety ceiling only — generation stops at EOS / end_of_turn first.
    max_gen_tokens: int = 512

    # ── parallelism ──────────────────────────────────────────────────
    # Each GPU gets its own worker process with a full model copy.
    n_gpus: int = 8

    # ── data collection flags ────────────────────────────────────────
    collect_attribution: bool = False

    # When False, IT model receives raw tokenized prompts (no chat template).
    # Ablation control: tests whether δ-cosine patterns are weight-intrinsic
    # or activated by the chat template wrapping.
    apply_chat_template: bool = True

    # ── output paths (derived in __post_init__) ──────────────────────
    output_path: str = field(init=False)
    plot_path:   str = field(init=False)

    def __post_init__(self) -> None:
        if self.model_variant not in ("pt", "it"):
            raise ValueError(f"model_variant must be 'pt' or 'it', got {self.model_variant!r}")
        self.model_name    = f"google/gemma-3-4b-{self.model_variant}"
        self.transcoder_set = f"google/gemma-scope-2-4b-{self.model_variant}"
        rd = self.run_dir
        self.output_path = f"{rd}/exp2_results.json"
        self.plot_path   = f"{rd}/plots"

    @property
    def run_dir(self) -> str:
        variant = self.transcoder_variant.replace("width_", "")
        suffix = "_notmpl" if not self.apply_chat_template else ""
        return f"results/exp02_ic_ooc_reasoning_mechanistic_comparison/{self.model_variant}_{variant}_t{self.max_gen_tokens}{suffix}"

    @property
    def is_instruction_tuned(self) -> bool:
        return self.model_variant == "it"
