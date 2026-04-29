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

OLMO2_7B_PT_SHA = "7df9a82518afdecae4e8c026b27adccc8c1f0032"
OLMO2_7B_SFT_SHA = "1de02c0175118a9de5854aec80a1f970e701e928"
OLMO2_7B_DPO_SHA = "e34ea60adff2e575f4fe7569eaffd1b28509b6fd"
OLMO2_7B_RLVR_SHA = "470b1fba1ae01581f270116362ee4aa1b97f4c84"


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
    multi_gpu: bool = False   # True → load with device_map="sequential" across all GPUs
    pt_revision: str | None = None    # immutable HF commit SHA used for reproducible reruns
    it_revision: str | None = None    # immutable HF commit SHA used for reproducible reruns

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
        pt_revision="cc012e0a6d0787b4adcc0fa2c4da74402494554d",
        it_revision="093f9f388b31de276ce2de164bdc2081324b9767",
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
        pt_revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        it_revision="0e9e39f249a16976918f6564b8830bc894c89659",
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
        pt_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        it_revision="1cfa9a7208912126459214e8b04321603b3df60c",
    ),
    "qwen25_32b": ModelSpec(
        name="qwen25_32b",
        pt_id="Qwen/Qwen2.5-32B",
        it_id="Qwen/Qwen2.5-32B-Instruct",
        n_layers=64,
        d_model=5120,
        n_heads=40,
        n_kv_heads=8,
        global_attn_layers=frozenset(range(64)),
        multi_gpu=True,
        pt_revision="1818d35814b8319459f4bd55ed1ac8709630f003",
        it_revision="5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    ),
    "mistral_7b": ModelSpec(
        name="mistral_7b",
        pt_id="mistralai/Mistral-7B-v0.3",
        it_id="mistralai/Mistral-7B-Instruct-v0.3",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        # v0.3 configs set sliding_window=None; unlike v0.1, SWA is disabled.
        global_attn_layers=frozenset(range(32)),
        is_sliding_window=False,
        sliding_window_size=None,
        pt_revision="caa1feb0e54d415e2df31207e5f4e273e33509b1",
        it_revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
    ),
    "deepseek_v2_lite": ModelSpec(
        name="deepseek_v2_lite",
        pt_id="deepseek-ai/DeepSeek-V2-Lite",
        it_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
        n_layers=27,
        d_model=2048,
        n_heads=16,
        # MLA: num_key_value_heads=16 in config.json (same as query heads).
        # KV compression is done via kv_lora_rank=512 latent space, not GQA.
        n_kv_heads=16,
        global_attn_layers=frozenset(range(27)),
        is_moe=True,
        # multi_gpu=False (default): fits on 1×80GB GPU at max_new_tokens=64.
        # At 512 tokens the KV cache + activations push it over; 64 is sufficient
        # to observe commitment behaviour (median commit at layer ~15 for MoE).
        pt_revision="604d5664dddd88a0433dbae533b7fe9472482de0",
        it_revision="85864749cd611b4353ce1decdb286193298f64c7",
    ),
    "olmo2_7b": ModelSpec(
        name="olmo2_7b",
        # OLMo 2 7B was released 2024-11 (1124).  There is no 0325-7B variant.
        pt_id="allenai/OLMo-2-1124-7B",
        it_id="allenai/OLMo-2-1124-7B-Instruct",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        # OLMo 2 7B uses standard MHA (n_kv_heads == n_heads), not GQA.
        n_kv_heads=32,
        global_attn_layers=frozenset(range(32)),
        pt_revision=OLMO2_7B_PT_SHA,
        it_revision=OLMO2_7B_RLVR_SHA,
    ),
    "olmo2_7b_pt_sft": ModelSpec(
        name="olmo2_7b_pt_sft",
        pt_id="allenai/OLMo-2-1124-7B",
        it_id="allenai/OLMo-2-1124-7B-SFT",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        n_kv_heads=32,
        global_attn_layers=frozenset(range(32)),
        pt_revision=OLMO2_7B_PT_SHA,
        it_revision=OLMO2_7B_SFT_SHA,
    ),
    "olmo2_7b_sft_dpo": ModelSpec(
        name="olmo2_7b_sft_dpo",
        pt_id="allenai/OLMo-2-1124-7B-SFT",
        it_id="allenai/OLMo-2-1124-7B-DPO",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        n_kv_heads=32,
        global_attn_layers=frozenset(range(32)),
        pt_revision=OLMO2_7B_SFT_SHA,
        it_revision=OLMO2_7B_DPO_SHA,
    ),
    "olmo2_7b_dpo_rlvr": ModelSpec(
        name="olmo2_7b_dpo_rlvr",
        pt_id="allenai/OLMo-2-1124-7B-DPO",
        it_id="allenai/OLMo-2-1124-7B-Instruct",
        n_layers=32,
        d_model=4096,
        n_heads=32,
        n_kv_heads=32,
        global_attn_layers=frozenset(range(32)),
        pt_revision=OLMO2_7B_DPO_SHA,
        it_revision=OLMO2_7B_RLVR_SHA,
    ),
    "olmo2_32b": ModelSpec(
        name="olmo2_32b",
        pt_id="allenai/OLMo-2-0325-32B",
        it_id="allenai/OLMo-2-0325-32B-Instruct",
        n_layers=64,
        d_model=5120,
        n_heads=40,
        n_kv_heads=8,
        global_attn_layers=frozenset(range(64)),
        multi_gpu=True,
        pt_revision="cc9d3cf9c7230b86ee6b84607b37db1c01e3f1ed",
        it_revision="b96024342a77a69aa0dda815c3454a671f477463",
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


def model_revision_for_variant(spec: ModelSpec, variant: str) -> str | None:
    """Return the immutable Hugging Face revision configured for this variant."""
    if variant == "pt":
        return spec.pt_revision
    elif variant == "it":
        return spec.it_revision
    raise ValueError(f"variant must be 'pt' or 'it', got {variant!r}")


def revision_for_model_id(model_id: str) -> str | None:
    """Infer the configured immutable revision for a registered HF model ID.

    Most experiment runners pass only a repo ID into the shared loader. Keeping
    this lookup central makes those runners snapshot-pinned without changing
    every call site. Unknown repos intentionally fall back to the caller's
    explicit `revision=` argument or the Transformers default.
    """
    for spec in MODEL_REGISTRY.values():
        if model_id == spec.pt_id:
            return spec.pt_revision
        if model_id == spec.it_id:
            return spec.it_revision
    return None
