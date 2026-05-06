"""Exp51 native-history residual-state x late-stack cross-patching."""

DEFAULT_MODELS = ("qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b")
DEFAULT_HORIZONS = (0, 1, 2, 4, 8, 16, 32)
PRIMARY_HORIZONS = (4, 8, 16)
DEFAULT_READOUTS = ("common_it", "common_pt")
