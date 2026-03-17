from dataclasses import dataclass, field


@dataclass
class PocConfig:
    # ------------------------------------------------------------------ model
    model_name: str = "google/gemma-3-4b-pt"          # base (pretrained) model — sentence completion
    transcoder_set: str = "google/gemma-scope-2-4b-pt" # Gemma Scope 2 transcoders, all layers
    backend: str = "nnsight"                            # nnsight required for Gemma 3

    device: str = "cpu"        # override: "mps" (Apple Silicon), "cuda" (NVIDIA)
    dtype_str: str = "float32" # "bfloat16" on GPU

    # ---------------------------------------------------------------- transcoder variant
    # google/gemma-scope-2-4b-pt has 18 variants per layer:
    #   width  : 16k | 65k | 262k     (feature dictionary size; 262k = 262,144 features)
    #   l0     : small (~10-20 active) | medium (~30-60 active) | big (~60-150 active)
    #   affine : plain | _affine (adds W_skip affine skip connection)
    # Using 65k + medium + affine — 4× richer than 16k, fits on single H100, captures skip connection
    transcoder_variant: str = "width_65k_l0_medium_affine"

    # -------------------------------------------------------- circuit-tracer
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    batch_size: int = 512       # H100: 512; MPS: 64; CPU: 32
    max_feature_nodes: int = 200

    # ------------------------------------------------------------ output paths
    output_path: str = "results/poc_results.json"
    plot_path: str = "results/specificity_vs_attribution.png"

    # ---------------------------------------------------------------- prompts
    # Two groups: "memorization" (direct retrieval) and "reasoning" (computation/composition).
    # All prompts are raw sentence-completion format for the base language model.
    # NO chat template, NO instruction format — the model continues the text.
    #
    # Convention: prompts do NOT have a trailing space; target tokens have a leading space.
    # This matches how modern tokenizers encode word boundaries (e.g., " Paris", " 6").
    # Multi-digit answers use only the first digit token (e.g., 15 → " 1", 56 → " 5").
    # Run `uv run python -m src.poc.inference_test` to verify tokenization before the full run.
    prompts: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: {

        # ==================================================================
        # MEMORIZATION
        # Direct retrieval. Answer type is naturally constrained by the prompt.
        # ==================================================================
        "memorization": [
            # --- capitals ---
            ("The capital of France is",       " Paris"),
            ("The capital of Japan is",        " Tokyo"),
            ("The capital of Italy is",        " Rome"),
            ("The capital of Egypt is",        " Cairo"),
            ("The capital of Peru is",         " Lima"),

            # --- antonyms ---
            ("The opposite of hot is",         " cold"),
            ("The opposite of big is",         " small"),
            ("The opposite of fast is",        " slow"),
            ("The opposite of dark is",        " light"),
            ("The opposite of old is",         " young"),

            # --- colors ---
            ("The color of grass is",          " green"),
            ("The color of the sky is",        " blue"),
            ("The color of snow is",           " white"),
            ("The color of coal is",           " black"),
            ("The color of a banana is",       " yellow"),

            # --- facts ---
            ("The largest planet is",          " Jupiter"),
            ("The Earth orbits the",           " Sun"),

            # --- memorized arithmetic (single-digit results) ---
            ("3 x 2 is ",                       "6"),
            ("4 + 5 is ",                       "9"),
            ("9 - 3 is ",                       "6"),
            ("8 - 1 is ",                       "7"),
            ("2 x 4 is ",                       "8"),
            ("7 - 4 is ",                       "3"),
            ("3 + 4 is ",                       "7"),
            ("6 / 2 is ",                       "3"),
            ("5 - 2 is ",                       "3"),
            ("2 + 6 is ",                       "8"),

            # --- languages ---
            ("People in France speak",         " French"),
            ("People in Germany speak",        " German"),
            ("People in Japan speak",          " Japanese"),
            ("People in Brazil speak",         " Portuguese"),
            ("People in Czech Republic speak", " Czech"),
        ],

        # ==================================================================
        # REASONING
        # Requires computation, composition, or multi-step inference.
        # ==================================================================
        "reasoning": [

            # --- OOCR: compose two facts → single answer ---
            ("The language spoken in Tokyo is",       " Japanese"),
            ("The language spoken in Marseille is",      " French"),
            ("The language spoken in Sao Paulo is",      " Portuguese"),
            ("The continent where the Eiffel Tower is located is",   " Europe"),
            ("The continent where the Pyramids are located is",      " Africa"),
            ("The color of the sky on Mars is",                      " red"),
            ("The season that comes right after winter is",          " spring"),
            ("The season that comes right after summer is",          " autumn"),
            ("The ocean between Europe and America is the",          " Atlantic"),
            ("The currency used in the land of sushi is the",        " yen"),

            # --- ICL novel operations (answer type from pattern) ---
            ("f(1)=3, f(2)=5, f(0)= ",                               "1"),
            ("f(1)=2, f(2)=4, f(3)=6, f(4)= ",                       "8"),
            ("f(0)=1, f(1)=3, f(2)=5, f(3)= ",                       "7"),
            ("3#1=4, 2#5=7, 4#4= ",                                   "8"),
            ("3#1=2, 5#2=3, 7#3= ",                                   "4"),
            ("aa:2, bbb:3, cccc: ",                                   "4"),

            # --- multi-step reasoning (natural answer type constraint) ---
            ("The square root of 9 is",                              " 3"),
            ("Half of a dozen is",                                   " 6"),
            ("A baker's dozen minus a dozen is",                     " 1"),
            ("The number of vowels in the word hello is",            " 2"),
            ("The next prime after 7 is",                            " 1"),  # first digit of 11

            # --- analogies (structural relational mapping) ---
            ("Hot is to cold as day is to",                          " night"),
            ("Paris is to France as Rome is to",                     " Italy"),
            ("Fast is to slow as dark is to",                        " light"),
            ("Sister is to brother as aunt is to",                   " uncle"),
            ("Dog is to bark as cat is to",                          " me"),   # first token of "meow"

            # --- syllogisms and logical deduction ---
            ("All fish live in water. A salmon is a fish. A salmon lives in",  " water"),
            ("If penguins are birds and all birds have wings, penguins have",   " wings"),
            ("Every prime greater than 2 is odd. 7 is prime and greater than 2. 7 is",  " odd"),

            # --- arithmetic embedded in language ---
            ("Two squared equals",                                   " 4"),
            ("The square root of 4 is",                              " 2"),
            ("Half of 8 is",                                         " 4"),
            ("A dozen minus 9 equals",                               " 3"),

            # --- numeric sequences ---
            ("9, 7, 5, 3,",                                         " 1"),
            ("1+1=2, 2+2=4, 3+3=",                                  " 6"),

            # --- word structure ---
            ("The word 'was' spelled backwards is",                  " saw"),
            ("The opposite of the word 'false' is",                  " true"),
        ],
    })
