from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


AblationMethod = Literal["none", "mean", "skip", "directional", "resample"]
ExperimentKind = Literal["baseline", "cartography", "phase", "progressive", "subspace", "single_layer_phase"]
EvalBackend = Literal["custom", "harness", "hybrid"]


@dataclass
class Exp5Config:
    """Configuration for Experiment 5 ablation runs.

    Exp5 intentionally sits above the shared model loader and dataset format.
    It owns intervention choice, benchmark surface, and artifact layout.
    """

    experiment: ExperimentKind = "baseline"
    model_variant: str = "it"
    model_id: str = ""
    transcoder_release: str = ""
    transcoder_variant: str = "width_16k_l0_big_affine"
    dataset_path: str = "data/exp3_dataset.jsonl"
    output_base: str = "results/exp05_corrective_direction_ablation_cartography"
    run_name: str = ""

    prompt_format: str = "B"
    apply_chat_template: bool = False
    max_gen_tokens: int = 200
    n_eval_examples: int = 500
    batch_size: int = 1

    method: AblationMethod = "none"
    ablation_layers: list[int] = field(default_factory=list)
    directional_alpha: float = 1.0

    # Model loading — shared/model.py accesses these three fields by name.
    # backend selects the nnsight/transformerlens backend in circuit_tracer.
    backend: str = "nnsight"

    mean_acts_path: str = ""
    corrective_direction_path: str = ""
    resample_bank_path: str = ""

    benchmarks: list[str] = field(default_factory=lambda: [
        "exp3_factual_em",
        "exp3_reasoning_em",
        "exp3_alignment_behavior",
        "exp3_format_adherence",
        "structural_tokens",
    ])
    eval_backend: EvalBackend = "hybrid"
    use_lm_eval: bool = False

    save_hidden_states: bool = True
    checkpoint_layers: list[int] = field(default_factory=lambda: [0, 11, 20, 33])
    n_prompts_for_subspace: int = 500

    skip_transcoders: bool = True  # exp5 never uses transcoder features — skip the ~11 GB load

    device: str = "cuda"
    dtype_str: str = "bfloat16"

    n_layers: int = 34
    d_model: int = 2560
    proposal_boundary: int = 20

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = f"google/gemma-3-4b-{self.model_variant}"
        if not self.transcoder_release:
            self.transcoder_release = f"google/gemma-scope-2-4b-{self.model_variant}"
        if not self.run_name:
            method_tag = self.method
            layers_tag = ""
            if self.ablation_layers:
                layers_tag = "_l" + "-".join(str(i) for i in self.ablation_layers[:4])
                if len(self.ablation_layers) > 4:
                    layers_tag += f"_n{len(self.ablation_layers)}"
            alpha_tag = ""
            if self.method == "directional":
                alpha_tag = f"_a{self.directional_alpha:g}"
            chat_tag = "_chat" if self.apply_chat_template else ""
            self.run_name = (
                f"{self.experiment}_{self.model_variant}_{method_tag}"
                f"{layers_tag}{alpha_tag}{chat_tag}_t{self.max_gen_tokens}"
            )

    # Properties that shared/model.py's load_model() accesses by name.
    # CollectionConfig exposes these as derived fields; Exp5Config does it via properties.
    @property
    def model_name(self) -> str:
        return self.model_id

    @property
    def transcoder_set(self) -> str:
        return self.transcoder_release

    @property
    def run_dir(self) -> Path:
        return Path(self.output_base) / self.run_name

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def corrective_layers(self) -> range:
        return range(self.proposal_boundary, self.n_layers)

    @property
    def proposal_layers(self) -> range:
        return range(0, self.proposal_boundary)

    def to_dict(self) -> dict:
        return asdict(self)
