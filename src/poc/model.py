import torch
from dataclasses import dataclass
from circuit_tracer import ReplacementModel


@dataclass
class LoadedModel:
    model: ReplacementModel
    W_U: torch.Tensor  # [d_model, vocab_size] unembedding matrix, float32


def load_model(cfg) -> LoadedModel:
    dtype = torch.float32 if cfg.dtype_str == "float32" else torch.bfloat16
    model = ReplacementModel.from_pretrained(
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        backend="transformerlens",
        device=torch.device(cfg.device),
        dtype=dtype,
        lazy_encoder=False,
        lazy_decoder=False,  # load W_dec upfront — we need it for logit lens on every feature
    )
    model.eval()
    # model.model is the HookedTransformer; W_U shape: [d_model, vocab_size]
    # Keep on same device as model so logit-lens matmuls stay on GPU
    W_U = model.model.W_U.detach().float().to(torch.device(cfg.device))
    return LoadedModel(model=model, W_U=W_U)


def get_token_id(loaded: LoadedModel, token_str: str) -> int:
    """Get single token id. Raises AssertionError if token_str is multi-token."""
    ids = loaded.model.model.tokenizer.encode(token_str, add_special_tokens=False)
    assert len(ids) == 1, f"'{token_str}' tokenizes to {len(ids)} tokens {ids} — fix in config"
    return ids[0]


def get_digit_token_ids(loaded: LoadedModel) -> list[int]:
    """Token ids for digits 0-9 with and without a leading space."""
    ids: set[int] = set()
    for digit in range(10):
        for s in [str(digit), f" {digit}"]:
            toks = loaded.model.model.tokenizer.encode(s, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
    return list(ids)
