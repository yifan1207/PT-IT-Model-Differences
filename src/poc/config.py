from dataclasses import dataclass, field


@dataclass
class PocConfig:
    model_name: str = "gemma-2-2b"
    transcoder_set: str = "gemma"  # resolves to mwhanna/gemma-scope-transcoders
    device: str = "cpu"            # override: "mps" (Apple Silicon), "cuda" (NVIDIA)
    dtype_str: str = "float32"     # "bfloat16" on GPU

    # circuit-tracer attribution settings
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    batch_size: int = 64           # lower to 32 for CPU memory safety
    max_feature_nodes: int = 200

    output_path: str = "results/poc_results.json"
    plot_path: str = "results/specificity_vs_attribution.png"

    # 20 prompts: (prompt_text, target_token_str)
    # Multi-digit answers use the first token (first digit). Leading space = separate token.
    # GROUP A — Arithmetic
    # GROUP B — Factual
    # GROUP C — ICL pattern completion
    # GROUP D — Harder/novel (some may fail / be skipped if multi-token)
    prompts: list[tuple[str, str]] = field(default_factory=lambda: [
        # A — Arithmetic
        ("3 x 2 is",                          " 6"),   # A1 trivial
        ("8 x 7 is",                          " 5"),   # A2 first digit of 56
        ("13 x 4 is",                         " 5"),   # A3 first digit of 52
        ("9 - 3 is",                          " 6"),   # A4
        ("7 + 8 is",                          " 1"),   # A5 first digit of 15
        # B — Factual
        ("The capital of France is",          " Paris"),
        ("The capital of Peru is",            " Lima"),
        ("H2O is called",                     " water"),
        ("Opposite of hot is",                " cold"),
        ("Color of grass is",                 " green"),
        # C — ICL pattern completion
        ("2+3=5, 4+1=5, 7+8=",               " 1"),   # C1 first digit of 15
        ("cat:animal, rose:",                 " plant"),
        ("AB->BA, CD->DC, EF->",              "FE"),   # C3 no leading space after ->
        ("1:one, 2:two, 5:",                  " five"),
        ("hot->cold, big->small, fast->",     " slow"),
        # D — Harder / novel
        ("2=4, 3=9, 5=",                      " 2"),   # D1 first digit of 25 (squaring)
        ("Fibonacci after 5,8 is",            " 1"),   # D2 first digit of 13
        ("Vowels in hello is",                " 2"),   # D3
        ("Protons in helium is",              " 2"),   # D4
        ("Next prime after 7 is",             " 1"),   # D5 first digit of 11
    ])
