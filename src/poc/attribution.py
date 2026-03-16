from collections import defaultdict
from dataclasses import dataclass, field

import torch

from circuit_tracer import attribute


@dataclass
class FeatureRecord:
    feature_idx: int
    layer: int
    position: int
    prompt_id: str           # e.g. "A1", "B3" — for grouping in plots

    # Raw quantities from the model
    activation: float        # how strongly the feature fired on this token position

    # Logit-lens: project W_dec[f] through the unembedding matrix → vocab logit vector
    logit_target: float      # W_dec[f] · W_U[:, target]  (signed; can be negative)
    logit_norm: float        # ||W_dec[f] @ W_U||₂  — L2 norm of the full logit vector
    logit_entropy: float     # H(softmax(W_dec[f] @ W_U))  — broad vs narrow in prob space

    # What does this feature most promote?
    top1_token_id: int       # argmax of W_dec[f] @ W_U  (most promoted token ID)
    top1_token_str: str      # decoded string of top1_token_id  (for interpretability)
    correct_token_rank: int  # rank of target token in W_dec[f] @ W_U  (1 = top prediction)
                             # low rank → feature "knows" the correct answer

    # Plot axes
    specificity: float       # logit_target / logit_norm  (~0 = broad, ~±1 = focused on target)
    attribution: float       # |activation × logit_target|  — direct effect on target token

    # Broadness: how many tokens does this feature's contribution concentrate on?
    # c(f) = activation × (W_dec[f] @ W_U)  [shape: vocab_size]
    c_total_mass: float      # Σ|c(f)| — unnormalized total contribution magnitude
    n50: int                 # min tokens for 50% of Σ|c(f)| mass  (sharp core)
    n90: int                 # min tokens for 90% of Σ|c(f)| mass  (broad reach)

    # Mechanism: does the feature work by promoting or suppressing?
    promote_ratio: float     # Σmax(c(f),0) / Σ|c(f)|  — 1.0 = pure promotion, 0.0 = pure suppression

    # Graph structure: how connected is this feature in the attribution graph?
    incoming_edge_count: int # number of other features with non-negligible edges into this one
                             # high = compositional (depends on upstream); low = lookup/direct

    # Top-50 token contributions for heatmap (Plot 5), sorted by |c_value|
    top50_contributions: list = field(default_factory=list)
    # Each entry: {"token_id": int, "token_str": str, "c_value": float}


def run_attribution(prompt: str, correct_token_id: int, prompt_id: str,
                    loaded, cfg) -> tuple[object, list[FeatureRecord]]:
    """Run circuit-tracer and extract one FeatureRecord per selected feature."""
    graph = attribute(
        prompt=prompt,
        model=loaded.model,
        max_n_logits=cfg.max_n_logits,
        desired_logit_prob=cfg.desired_logit_prob,
        batch_size=cfg.batch_size,
        max_feature_nodes=cfg.max_feature_nodes,
        verbose=True,
    )
    tokenizer = loaded.tokenizer
    records = _extract_records(graph, correct_token_id, prompt_id, loaded, tokenizer)
    return graph, records


def _extract_records(graph, correct_token_id: int, prompt_id: str,
                     loaded, tokenizer) -> list[FeatureRecord]:
    """Extract all per-feature metrics, batching W_dec @ W_U by layer for GPU efficiency.

    Two-phase:
      Phase 1: group features by layer, do one W_dec @ W_U matmul per layer,
               compute all metrics from the contribution vector c(f) = activation × logit_vec.
      Phase 2: assemble FeatureRecord list in original selection order.
    """
    n_features = len(graph.selected_features)

    # Phase 1: group by layer, batch compute
    layer_to_sel_indices: dict[int, list[int]] = defaultdict(list)
    for sel_idx in range(n_features):
        layer = int(graph.active_features[graph.selected_features[sel_idx]][0])
        layer_to_sel_indices[layer].append(sel_idx)

    data: dict[int, dict] = {}  # sel_idx → computed metrics dict

    for layer, sel_indices in layer_to_sel_indices.items():
        feat_idxs = [
            int(graph.active_features[graph.selected_features[i]][2])
            for i in sel_indices
        ]
        transcoder = loaded.model.transcoders[layer]
        # float() cast for precision; both W_dec and W_U are on the model device
        batch_logit_vecs = transcoder.W_dec[feat_idxs].float() @ loaded.W_U  # [n, vocab_size]

        for j, sel_idx in enumerate(sel_indices):
            active_idx = int(graph.selected_features[sel_idx])
            activation = graph.activation_values[active_idx].item()
            logit_vec = batch_logit_vecs[j]  # [vocab_size], on model device

            data[sel_idx] = _compute_feature_metrics(
                activation, logit_vec, correct_token_id, tokenizer
            )

    # Phase 2: build FeatureRecord list
    # incoming_edge_count: number of active features with non-negligible edges into this one
    # adj rows=target, cols=source; first n_active cols correspond to active_features
    adj = graph.adjacency_matrix  # [n_nodes, n_nodes]
    n_active = graph.active_features.shape[0]
    records = []
    for sel_idx in range(n_features):
        active_idx = int(graph.selected_features[sel_idx])
        layer, pos, feat_idx = graph.active_features[active_idx].tolist()
        incoming = int((adj[active_idx, :n_active].abs() > 1e-4).sum().item())
        d = data[sel_idx]
        records.append(FeatureRecord(
            feature_idx=int(feat_idx),
            layer=int(layer),
            position=int(pos),
            prompt_id=prompt_id,
            incoming_edge_count=incoming,
            **d,
        ))

    return records


def _compute_feature_metrics(activation: float, logit_vec: torch.Tensor,
                              correct_token_id: int, tokenizer) -> dict:
    """Compute all scalar metrics and top-50 contributions from one feature's logit vector."""
    # c(f) = activation × logit_vec  — the actual contribution to each token's logit
    c_vec = activation * logit_vec
    abs_c = c_vec.abs()
    total_mass = abs_c.sum().item()

    # Logit-lens metrics
    logit_target = logit_vec[correct_token_id].item()
    logit_norm = logit_vec.norm().item()
    p = torch.softmax(logit_vec, dim=-1)
    logit_entropy = float(-(p * torch.log(p.clamp(min=1e-10))).sum())

    # What token does this feature most promote? (interpretability)
    top1_token_id = int(logit_vec.argmax().item())
    top1_token_str = _safe_decode(tokenizer, top1_token_id)

    # Rank of the correct token in the logit distribution (1 = top prediction)
    correct_token_rank = int((logit_vec > logit_vec[correct_token_id]).sum().item()) + 1

    # Nₓ: fewest tokens whose |c(f)| values sum to ≥x% of Σ|c(f)|
    # Formula: (cumsum < threshold).sum() + 1 counts the index where cumsum
    # first reaches the threshold, i.e. the minimum number of tokens needed.
    if total_mass > 0:
        sorted_abs = abs_c.sort(descending=True).values
        cumsum = sorted_abs.cumsum(0)
        n50 = int((cumsum < 0.5 * total_mass).sum().item()) + 1
        n90 = int((cumsum < 0.9 * total_mass).sum().item()) + 1
    else:
        # Zero total mass (all-zero W_dec row) — feature has no distributional effect.
        # Shouldn't occur with trained transcoders; mark as -1 to distinguish from valid n90.
        n50 = -1
        n90 = -1

    # promote_ratio: fraction of total |contribution| that is positive (promotion vs suppression)
    pos_mass = c_vec.clamp(min=0).sum().item()
    promote_ratio = pos_mass / (total_mass + 1e-10)

    # Top-50 tokens by absolute contribution — used by heatmap (Plot 5).
    # Batch decode all token IDs at once (50x faster than individual decode calls).
    top_k = min(50, int(abs_c.numel()))
    _, top50_idxs = abs_c.topk(top_k)
    top50_ids = top50_idxs.tolist()
    top50_strs = _batch_decode(tokenizer, top50_ids)
    top50_contributions = [
        {
            "token_id": int(idx),
            "token_str": top50_strs[i],
            "c_value": float(c_vec[idx]),
        }
        for i, idx in enumerate(top50_ids)
    ]

    return dict(
        activation=activation,
        logit_target=logit_target,
        logit_norm=logit_norm,
        logit_entropy=logit_entropy,
        top1_token_id=top1_token_id,
        top1_token_str=top1_token_str,
        correct_token_rank=correct_token_rank,
        specificity=logit_target / (logit_norm + 1e-10),
        attribution=abs(activation * logit_target),
        c_total_mass=total_mass,
        n50=n50,
        n90=n90,
        promote_ratio=promote_ratio,
        top50_contributions=top50_contributions,
    )


def _batch_decode(tokenizer, token_ids: list[int]) -> list[str]:
    """Decode a list of token IDs to strings, handling errors per token."""
    try:
        # batch_decode expects list of lists; skip_special_tokens=False preserves BOS/EOS for display
        return tokenizer.batch_decode([[tid] for tid in token_ids])
    except Exception:
        return [_safe_decode(tokenizer, tid) for tid in token_ids]


def _safe_decode(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return f"<tok_{token_id}>"
