import torch
from dataclasses import dataclass

from circuit_tracer import attribute


@dataclass
class FeatureRecord:
    feature_idx: int
    layer: int
    position: int
    prompt_id: str           # e.g. "A1", "B3" — set by caller for grouping in plots

    # Raw quantities from the model
    activation: float        # how strongly the feature fired on this token position

    # Logit-lens quantities (W_dec[f] @ W_U projected onto vocabulary)
    logit_target: float      # W_dec[f] · W_unembed[target]  (signed; can be negative)
    logit_norm: float        # ||W_dec[f] @ W_U||  (L2 norm of full vocab logit vector)
    logit_entropy: float     # H(softmax(W_dec[f] @ W_U))  — kept for reference

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
    """Extract (specificity, attribution) for every selected feature in the graph."""
    records = []

    for i in range(len(graph.selected_features)):
        active_idx = graph.selected_features[i]
        layer, pos, feat_idx = graph.active_features[active_idx].tolist()
        activation = graph.activation_values[active_idx].item()

        # Logit lens: project decoder direction through unembedding matrix
        transcoder = loaded.model.transcoders[int(layer)]
        w_dec_row = transcoder.W_dec[int(feat_idx)].float().to(loaded.W_U.device)
        logit_vec = w_dec_row @ loaded.W_U      # [vocab_size]

        logit_target = logit_vec[correct_token_id].item()
        logit_norm = logit_vec.norm().item()

        specificity = logit_target / (logit_norm + 1e-10)
        attribution = abs(activation * logit_target)

        records.append(FeatureRecord(
            feature_idx=int(feat_idx),
            layer=int(layer),
            position=int(pos),
            prompt_id=prompt_id,
            activation=activation,
            logit_target=logit_target,
            logit_norm=logit_norm,
            logit_entropy=_entropy(logit_vec),
            specificity=specificity,
            attribution=attribution,
        ))

    return records


def _entropy(logit_vec: torch.Tensor) -> float:
    p = torch.softmax(logit_vec, dim=-1)
    return float(-(p * (p + 1e-10).log()).sum())
