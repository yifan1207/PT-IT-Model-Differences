"""Model adapters for cross-model replication study."""
from .base import ModelAdapter
from .gemma3 import Gemma3Adapter
from .llama31 import Llama31Adapter
from .qwen3 import Qwen3Adapter
from .mistral import MistralAdapter
from .deepseek_v2 import DeepseekV2Adapter
from .olmo2 import OLMo2Adapter

_ADAPTER_MAP: dict[str, type[ModelAdapter]] = {
    "gemma3_4b":        Gemma3Adapter,
    "llama31_8b":       Llama31Adapter,
    "qwen3_4b":         Qwen3Adapter,
    "qwen25_32b":       Qwen3Adapter,
    "mistral_7b":       MistralAdapter,
    "deepseek_v2_lite": DeepseekV2Adapter,
    "olmo2_7b":         OLMo2Adapter,
    "olmo2_32b":        OLMo2Adapter,
}


def get_adapter(model_name: str) -> ModelAdapter:
    if model_name not in _ADAPTER_MAP:
        raise ValueError(f"No adapter for {model_name!r}. Available: {list(_ADAPTER_MAP)}")
    return _ADAPTER_MAP[model_name]()
