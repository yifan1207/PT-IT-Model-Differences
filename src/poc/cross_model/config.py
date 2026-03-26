"""
Cross-model replication study: model registry and config.

Six model families to test whether the corrective computational stage is
a universal property of instruction tuning or a Gemma-specific artifact.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

BASE_RESULTS = Path("results/cross_model")
DATASET_PATH = Path("data/eval_dataset_v2.jsonl")


@dataclass
class ModelSpec:
    name: str                         # short name, e.g. "gemma3_4b"
    pt_id: str                        # HuggingFace model ID for base/pretrained variant
    it_id: str                        # HuggingFace model ID for instruct/chat variant
    n_layers: int
    d_model: int
    n_heads: int                      # number of query heads
    n_kv_heads: int                   # number of key-value heads (GQA)
    global_attn_layers: frozenset[int]  # layers with full global attention
    is_moe: bool = False
    is_sliding_window: bool = False
    sliding_window_size: int | None = None

    @property
    def result_dir(self) -> Path:
        return BASE_RESULTS / self.name

    @property
    def corrective_onset(self) -> int:
        """Predicted corrective onset layer (~60% depth)."""
        return round(self.n_layers * 0.60)

    @property
    def phase_boundary(self) -> int:
        """Predicted phase boundary (~33% depth, ID dip)."""
        return round(self.n_layers * 0.33)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gemma3_4b": ModelSpec(
        name="gemma3_4b",
        pt_id="google/gemma-3-4b-pt",
        it_id="google/gemma-3-4b-it",
        n_layers=34,
        d_model=2560,
        n_heads=8,
        n_kv_heads=4,
        # Global attention at layers where layer_idx % 6 == 5: 5, 11, 17, 23, 29
        global_attn_layers=frozenset({5, 11, 17, 23, 29}),
    ),
    "llama31_8b": ModelSpec(
        name="llama31_8b",
        pt_id="meta-llama/Llama-3.1-8B",
        it_id="meta-llama/Llama-3.1-8B-Instruct",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        global_attn_layers=frozenset(range(32)),
    ),
    "qwen3_4b": ModelSpec(
        name="qwen3_4b",
        pt_id="Qwen/Qwen3-4B-Base",
        it_id="Qwen/Qwen3-4B",
        n_layers=36,
        d_model=2560,
        n_heads=32,
        n_kv_heads=8,
        global_attn_layers=frozenset(range(36)),
    ),
    "mistral_7b": ModelSpec(
        name="mistral_7b",
        pt_id="mistralai/Mistral-7B-v0.3",
        it_id="mistralai/Mistral-7B-Instruct-v0.3",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        # All layers use sliding window attention
        global_attn_layers=frozenset(range(32)),
        is_sliding_window=True,
        sliding_window_size=4096,
    ),
    "deepseek_v2_lite": ModelSpec(
        name="deepseek_v2_lite",
        pt_id="deepseek-ai/DeepSeek-V2-Lite",
        it_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        n_layers=27,
        d_model=2048,
        n_heads=16,
        n_kv_heads=2,
        global_attn_layers=frozenset(range(27)),
        is_moe=True,
    ),
    "olmo2_7b": ModelSpec(
        name="olmo2_7b",
        pt_id="allenai/OLMo-2-0325-7B",
        it_id="allenai/OLMo-2-0325-7B-Instruct",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        global_attn_layers=frozenset(range(32)),
    ),
}


def get_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name]


def model_id_for_variant(spec: ModelSpec, variant: str) -> str:
    if variant == "pt":
        return spec.pt_id
    elif variant == "it":
        return spec.it_id
    raise ValueError(f"variant must be 'pt' or 'it', got {variant!r}")
