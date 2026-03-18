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
"""
from dataclasses import dataclass, field


@dataclass
class Exp2Config:
    # ── model / transcoder ───────────────────────────────────────────
    model_name: str = "google/gemma-3-4b-pt"
    transcoder_set: str = "google/gemma-scope-2-4b-pt"
    transcoder_variant: str = "width_16k_l0_big_affine"
    backend: str = "nnsight"
    dtype_str: str = "bfloat16"   # "float32" | "bfloat16"
    device: str = "cuda"

    # ── generation ───────────────────────────────────────────────────
    # Safety ceiling only — generation stops at EOS first.
    # 512 is generous enough that no IC/OOC/R prompt should hit it
    # before the model generates EOS naturally.
    max_gen_tokens: int = 512     # max new tokens per prompt (EOS stops earlier)
    # Greedy decoding (argmax) for deterministic, comparable outputs

    # ── parallelism ──────────────────────────────────────────────────
    # Number of GPUs to use. Each GPU gets its own worker process with a
    # full copy of the model. Set to 1 to disable multi-GPU.
    n_gpus: int = 8

    # ── data collection flags ────────────────────────────────────────
    collect_attribution: bool = False  # run circuit-tracer for plot 10 (slow)

    # ── output paths (derived in __post_init__) ──────────────────────
    output_path: str = field(init=False)
    plot_path: str = field(init=False)   # directory (exp2 plots are per-category)

    def __post_init__(self) -> None:
        rd = self.run_dir
        self.output_path = f"{rd}/exp2_results.json"
        self.plot_path = f"{rd}/plots"

    @property
    def run_dir(self) -> str:
        variant = self.transcoder_variant.replace("width_", "")
        return f"results/exp2/{variant}_t{self.max_gen_tokens}"
