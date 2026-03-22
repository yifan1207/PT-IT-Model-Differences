import json
from pathlib import Path

from src.poc.exp3.plots.plot_e3_10_mind_change import make_plot


it_path = Path("results/exp3/it_16k_l0_big_affine_t512/exp3_results.json")
pt_path = Path("results/exp3/pt_16k_l0_big_affine_t512/exp3_results.json")

print("loading it", flush=True)
with open(it_path, encoding="utf-8") as f:
    it_results = json.load(f)
print("loading pt", flush=True)
with open(pt_path, encoding="utf-8") as f:
    pt_results = json.load(f)

print("loaded", len(it_results), len(pt_results), flush=True)
print("rendering", flush=True)
make_plot(it_results, str(it_path.parent / "plots"), pt_results=pt_results)
print("done", flush=True)
