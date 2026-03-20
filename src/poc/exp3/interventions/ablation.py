"""
Corrective layer ablation (Experiment 3a).

Generates text with IT model while zeroing out the MLP contribution of
specified layers during each forward pass.

Hypothesis: ablating layers 25–33 should:
  - Preserve semantic content (middle layers already determined "what" to say)
  - Degrade format, helpfulness, safety (corrective stage removed)
  - Make IT output more similar to PT output

nnsight intervention pattern
-----------------------------
To zero the MLP contribution at layer i, we need to hook the MLP output
BEFORE it is added to the residual.  In Gemma3DecoderLayer the computation is:

    h1 = h_in + attention(...)
    mlp_out = mlp(pre_feedforward_layernorm(h1))      ← zero this
    h2 = h1 + mlp_out                                 ← layer output

The cleanest hook is to set mlp_out to zero AFTER computation but BEFORE
the residual add.  In nnsight 0.6 this is done with an assignment inside
the trace block:

    with loaded.model.trace(current_ids):
        for i in ablation_layers:
            # TODO: verify exact submodule path on live model.
            # The mlp output tensor can be zeroed with an in-place operation.
            # loaded.model.language_model.layers[i].mlp.output[:] = 0
            pass

IMPORTANT: verify the submodule name before running.  Use:
    list(loaded.model.language_model.layers[0].named_children())
to confirm that 'mlp' is the correct name and that its output is a plain
tensor (not a tuple).

If the output is a tuple (hidden_states, ...), zero only output[0].
"""
from src.poc.exp3.config import Exp3Config
from src.poc.exp3.collect import collect_prompt

def collect_prompt_ablated(
    prompt_id: str,
    category: str,
    prompt: str,
    loaded,
    cfg: Exp3Config,
) -> dict:
    """Generate with MLP outputs zeroed at cfg.ablation_layers.

    Returns a result dict in the same format as collect.py's collect_prompt,
    with an additional field:
        ablation_layers: list[int]  which layers were ablated
    """
    if not cfg.ablation_layers:
        raise ValueError("ablation_layers must be set in cfg to use ablation collection.")

    # Ablation is now natively supported in collect.py via cfg.ablation_layers.
    result = collect_prompt(prompt_id, category, prompt, loaded, cfg)
    result["ablation_layers"] = cfg.ablation_layers
    return result
