"""
Architecture constants for Gemma 3 4B PT + Gemma Scope 2 transcoders.
Centralised here so exp1 and exp2 stay in sync.
"""

N_LAYERS: int = 34
D_MODEL: int = 2560
VOCAB_SIZE: int = 262_144

# Hook names used by circuit-tracer (TransformerLens convention).
# These correspond to nnsight module paths in the ReplacementModel.
FEATURE_INPUT_HOOK: str = "mlp.hook_in"    # pre_feedforward_layernorm output
FEATURE_OUTPUT_HOOK: str = "hook_mlp_out"  # post_feedforward_layernorm output

# Exp2 uses Gemma3ForConditionalGeneration via nnsight, where decoder layers live
# under `language_model.layers`. Keep this as documentation only unless a caller
# proves it against the live mapping at runtime.
LAYER_PATH: str = "language_model.layers"
