"""
Activation patching: PT residual → IT late layers (Experiment 3b).

Patches the IT model's residual stream at a specified layer with the PT model's
residual from the same prompt, then continues IT generation.

Hypothesis: feeding PT's mid-layer representation into IT's corrective stage
should produce output that is semantically PT-like (same content) but formatted
like IT (style, helpfulness).  This would causally demonstrate that:
  - IT's corrective stage operates on the residual stream direction
  - The corrective stage is independent of the specific content

Two-model coordination pattern
--------------------------------
Step 1: Run PT forward pass on prompt, capture h_{patch_layer} (residual after layer patch_layer).
Step 2: Run IT forward pass on same prompt; at layer patch_layer, replace IT's residual with PT's.
Step 3: Continue IT generation from the patched state.

nnsight pattern (conceptual — verify on live model):

    # Step 1: PT
    with pt_model.trace(prompt_ids):
        pt_residual_save = nnsave(pt_model.language_model.layers[patch_layer].output[0])
        nnsave(pt_residual_save_list := [pt_residual_save])
    pt_h = pt_residual_save[0, -1, :]   # [d_model]

    # Step 2: IT with patch
    with it_model.trace(prompt_ids):
        # At layer patch_layer, override the output before it propagates.
        # This requires an assignment in the trace — nnsight 0.6 syntax TBD.
        # it_model.language_model.layers[patch_layer].output[0][:, -1, :] = pt_h
        pass

IMPORTANT: nnsight 0.6 assignment syntax inside a trace block must be
verified before implementation.  The intervention mechanism differs from
plain saving.  Consult nnsight docs / circuit-tracer source for examples.
"""


import torch
from nnsight import save as nnsave

def collect_patched_generation(
    prompt_id: str,
    category: str,
    prompt: str,
    pt_loaded,
    it_loaded,
    cfg,
    patch_layer: int = 20,
) -> dict:
    """Generate with IT model using PT's residual at patch_layer."""
    tokenizer = it_loaded.tokenizer
    device = it_loaded.W_U.device
    
    # PT runs on raw prompt, IT runs with chat template for style
    pt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    user_message = f"Complete the following sentence: {prompt}"
    it_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        return_tensors="pt", add_generation_prompt=True
    ).to(device)

    eos_token_id = tokenizer.eos_token_id
    stop_token_ids = {eos_token_id}
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end_of_turn_id is not None and end_of_turn_id != tokenizer.unk_token_id:
        stop_token_ids.add(end_of_turn_id)

    generated_tokens = []

    for step in range(cfg.max_gen_tokens):
        # Step 1: PT forward pass to get h_patch
        with pt_loaded.model.trace(pt_ids):
            pt_h_save = nnsave(pt_loaded.model.language_model.layers[patch_layer].output[0][:, -1, :])
            
        pt_h = pt_h_save  # shape: [d_model]; indexing [:, -1, :] on [1, T, d_model] gives [d_model]
        
        # Step 2: IT forward pass with patch
        with it_loaded.model.trace(it_ids):
            # Patch the residual
            it_loaded.model.language_model.layers[patch_layer].output[0][:, -1, :] = pt_h
            logits_save = nnsave(it_loaded.model.lm_head.output)
            
        logits = logits_save[0, -1, :].float()
        
        real_mask = it_loaded.real_token_mask
        masked_logits = logits.clone()
        masked_logits[~real_mask] = float("-inf")
        next_token_id = int(masked_logits.argmax().item())
        
        next_token_str = tokenizer.decode([next_token_id])
        generated_tokens.append({"token_id": next_token_id, "token_str": next_token_str})
        
        # Update inputs for next step
        pt_ids = torch.cat([pt_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        it_ids = torch.cat([it_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        
        if next_token_id in stop_token_ids:
            break

    return {
        "prompt_id": prompt_id,
        "category": category,
        "prompt": prompt,
        "generated_tokens": generated_tokens,
        "patch_layer": patch_layer,
        "patch_source": "pt"
    }
