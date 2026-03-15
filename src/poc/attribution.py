from collections import defaultdict
from dataclasses import dataclass

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

    # Logit-lens quantities (W_dec[f] @ W_U projected onto vocabulary)
    logit_target: float      # W_dec[f] · W_unembed[target]  (signed; can be negative)
    logit_norm: float        # ||W_dec[f] @ W_U||  (L2 norm of full vocab logit vector)
    logit_entropy: float     # H(softmax(W_dec[f] @ W_U))  — broad vs narrow feature

    # The two regression axes (derived)
    specificity: float       # logit_target / logit_norm  (x-axis; ~0 = broad, ~1 = focused)
    attribution: float       # |activation × logit_target|  (y-axis; direct effect formula)


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
    records = _extract_records(graph, correct_token_id, prompt_id, loaded)
    return graph, records


def _extract_records(graph, correct_token_id: int, prompt_id: str,
                     loaded) -> list[FeatureRecord]:
    """Extract per-feature metrics, batching W_dec @ W_U by layer for GPU efficiency."""
    n_features = len(graph.selected_features)

    # Group selected-feature indices by layer so we do one matmul per layer
    # instead of one matmul per feature
    layer_to_sel_indices: dict[int, list[int]] = defaultdict(list)
    for sel_idx in range(n_features):
        active_idx = graph.selected_features[sel_idx]
        layer = int(graph.active_features[active_idx][0])
        layer_to_sel_indices[layer].append(sel_idx)

    # Batch compute logit vectors per layer: W_dec[layer][feat_idxs] @ W_U
    logit_vecs: dict[int, torch.Tensor] = {}  # sel_idx → [vocab_size]
    for layer, sel_indices in layer_to_sel_indices.items():
        feat_idxs = [
            int(graph.active_features[graph.selected_features[i]][2])
            for i in sel_indices
        ]
        transcoder = loaded.model.transcoders[layer]
        # W_dec[feat_idxs]: [n, d_model] — float() cast for precision; stays on model device
        batch_logit_vecs = transcoder.W_dec[feat_idxs].float() @ loaded.W_U  # [n, vocab_size]
        for j, sel_idx in enumerate(sel_indices):
            logit_vecs[sel_idx] = batch_logit_vecs[j]

    records = []
    for sel_idx in range(n_features):
        active_idx = graph.selected_features[sel_idx]
        layer, pos, feat_idx = graph.active_features[active_idx].tolist()
        activation = graph.activation_values[active_idx].item()

        logit_vec = logit_vecs[sel_idx]
        logit_target = logit_vec[correct_token_id].item()
        logit_norm = logit_vec.norm().item()

        # Compute softmax once and reuse for entropy
        p = torch.softmax(logit_vec, dim=-1)
        logit_entropy = float(-(p * torch.log(p + 1e-10)).sum())

        records.append(FeatureRecord(
            feature_idx=int(feat_idx),
            layer=int(layer),
            position=int(pos),
            prompt_id=prompt_id,
            activation=activation,
            logit_target=logit_target,
            logit_norm=logit_norm,
            logit_entropy=logit_entropy,
            specificity=logit_target / (logit_norm + 1e-10),
            attribution=abs(activation * logit_target),
        ))

    return records
