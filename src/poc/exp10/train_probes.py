"""
Exp10 — Phase 2 + 2.5: Ridge-Regression Probes + Direction Comparison.

For each layer ℓ, fits a ridge regression:
    Δc ≈ w_ℓ · Δh_ℓ

The normalised weight vector d_commit[ℓ] = normalize(w_ℓ) is the
commitment-change direction: the axis in activation space that best
predicts whether commitment shifted between IT and PT.

Phase 2.5 compares d_commit to the existing mean IT-PT direction d_mean
and to the KL gradient direction d_grad.  If cosine > 0.95 at all
corrective layers → go/no-go = "redundant" (d_commit ≈ d_mean).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.poc.cross_model.config import get_spec
from src.poc.exp10.collect_paired import LayerAccumulator

log = logging.getLogger(__name__)

# ── Ridge regression ───────────────────────────────────────────────────────────

RIDGE_LAMBDA = 1.0  # Regularisation strength


def _ridge_solve(XtX: torch.Tensor, Xty: torch.Tensor, Xsum: torch.Tensor,
                 ysum: float, n: int, lam: float = RIDGE_LAMBDA
                 ) -> tuple[torch.Tensor, float]:
    """Closed-form centred ridge regression.

    Returns (w, intercept) where w is [d_model].
    """
    d = XtX.shape[0]
    # Centre: XtX_c = XtX - (1/n) Xsum ⊗ Xsum
    x_mean = Xsum / n
    y_mean = ysum / n
    XtX_c = XtX - n * x_mean.unsqueeze(1) * x_mean.unsqueeze(0)
    Xty_c = Xty - n * x_mean * y_mean

    # Solve (XtX_c + λI) w = Xty_c
    A = XtX_c + lam * torch.eye(d, dtype=XtX.dtype)
    w = torch.linalg.solve(A, Xty_c)

    intercept = y_mean - w @ x_mean
    return w.float(), float(intercept)


def _compute_r2(delta_h: torch.Tensor, delta_c: torch.Tensor,
                w: torch.Tensor, intercept: float) -> float:
    """Compute R² on held-out data.

    Args:
        delta_h: [n, d_model] float32
        delta_c: [n] float32
        w: [d_model] float32
        intercept: scalar
    """
    y_pred = delta_h @ w + intercept
    ss_res = ((delta_c - y_pred) ** 2).sum().item()
    ss_tot = ((delta_c - delta_c.mean()) ** 2).sum().item()
    if ss_tot < 1e-12:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


# ── PCA rank diagnostic ───────────────────────────────────────────────────────

def _pca_explained_variance(delta_h: torch.Tensor, delta_c: torch.Tensor,
                            n_components: int = 10) -> list[float]:
    """PCA of commitment-weighted Δh.

    Returns explained variance ratios for top n_components PCs.
    """
    # Weight each Δh by its Δc
    weighted = delta_h * delta_c.unsqueeze(1)  # [n, d]
    # Centre
    weighted = weighted - weighted.mean(0, keepdim=True)

    # SVD (truncated if possible)
    n, d = weighted.shape
    k = min(n_components, min(n, d))
    if n == 0 or d == 0:
        return [0.0] * n_components

    try:
        # Use randomized SVD for efficiency
        U, S, V = torch.svd_lowrank(weighted.float(), q=k)
    except Exception:
        # Fallback to full SVD
        U, S, V = torch.linalg.svd(weighted.float(), full_matrices=False)
        S = S[:k]

    var = S ** 2
    total_var = var.sum().item()
    if total_var < 1e-12:
        return [0.0] * n_components

    explained = (var / total_var).tolist()
    # Pad if needed
    while len(explained) < n_components:
        explained.append(0.0)
    return explained[:n_components]


# ── Main training function ─────────────────────────────────────────────────────

def train_probes(
    model_name: str,
    paired_data_dir: str | Path,
    mean_dir_path: str | Path,
    output_dir: str | Path,
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
    test_fraction: float = 0.2,
    redundancy_cosine_threshold: float = 0.95,
):
    """Train per-layer ridge probes and compare to existing directions.

    Args:
        model_name: Key in MODEL_REGISTRY.
        paired_data_dir: Phase 1 output directory.
        mean_dir_path: Path to corrective_directions.npz (existing d_mean).
        output_dir: Where to save probes and analysis.
        ridge_lambda: Ridge regression regularisation.
        test_fraction: Fraction of PCA subsample used for R² evaluation.
        redundancy_cosine_threshold: Cosine above which directions are "redundant".
    """
    paired_data_dir = Path(paired_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = get_spec(model_name)
    n_layers = spec.n_layers
    d_model = spec.d_model
    corrective_onset = spec.corrective_onset

    # ── Load existing mean directions ─────────────────────────────────────────
    mean_dirs: dict[int, torch.Tensor] = {}
    mean_dir_path = Path(mean_dir_path)
    if mean_dir_path.exists():
        data = np.load(mean_dir_path)
        for k in data.files:
            if k.startswith("layer_"):
                li = int(k.split("_", 1)[1])
                mean_dirs[li] = torch.tensor(data[k], dtype=torch.float32)
        log.info("Loaded mean directions for %d layers from %s", len(mean_dirs), mean_dir_path)

    # ── Load KL gradient directions ───────────────────────────────────────────
    grad_dirs: dict[int, torch.Tensor] = {}
    grad_path = paired_data_dir / "kl_gradients.npz"
    if grad_path.exists():
        data = np.load(grad_path)
        for k in data.files:
            if k.startswith("layer_"):
                li = int(k.split("_", 1)[1])
                grad_dirs[li] = torch.tensor(data[k], dtype=torch.float32)
        log.info("Loaded KL gradient directions for %d layers", len(grad_dirs))

    # ── Load PCA subsample (for R² evaluation and rank diagnostic) ────────────
    pca_dir = paired_data_dir / "pca_subsample"
    pca_dc_path = pca_dir / "delta_c.npy"
    delta_c_all = None
    if pca_dc_path.exists():
        delta_c_all = torch.from_numpy(np.load(pca_dc_path)).float()
        log.info("PCA subsample: %d tokens", delta_c_all.shape[0])

    # Train/test split indices (by token position in subsample)
    n_pca = delta_c_all.shape[0] if delta_c_all is not None else 0
    n_test = int(n_pca * test_fraction)
    n_train = n_pca - n_test

    # ── Per-layer probe training ──────────────────────────────────────────────
    acc_dir = paired_data_dir / "accumulators"
    per_layer_results = []
    commitment_dir_arrays = {}

    for li in range(n_layers):
        # Load accumulator
        acc = LayerAccumulator.load(acc_dir, li, d_model)
        if acc.n < 10:
            log.warning("Layer %d: only %d tokens, skipping", li, acc.n)
            per_layer_results.append(_empty_layer_result(li))
            commitment_dir_arrays[f"layer_{li}"] = np.zeros(d_model, dtype=np.float32)
            continue

        # Ridge solve
        w, intercept = _ridge_solve(
            acc.XtX, acc.Xty, acc.Xsum, acc.ysum, acc.n, lam=ridge_lambda,
        )

        # Commitment direction: normalised weight
        w_norm = w.norm().item()
        if w_norm > 1e-12:
            d_commit = w / w_norm
        else:
            d_commit = torch.zeros(d_model)
        commitment_dir_arrays[f"layer_{li}"] = d_commit.numpy()

        # R² on held-out PCA subsample
        r2_test = 0.0
        if delta_c_all is not None and n_test > 0:
            dh_path = pca_dir / f"delta_h_layer_{li}.npy"
            if dh_path.exists():
                dh_subsample = torch.from_numpy(np.load(dh_path)).float()
                # Use last `n_test` tokens as test
                dh_test = dh_subsample[-n_test:]
                dc_test = delta_c_all[-n_test:]
                if dh_test.shape[0] == dc_test.shape[0] and dh_test.shape[0] > 0:
                    r2_test = _compute_r2(dh_test, dc_test, w, intercept)

        # Cosine with mean direction
        cos_mean = 0.0
        if li in mean_dirs:
            d_mean = mean_dirs[li]
            d_mean = d_mean / (d_mean.norm() + 1e-12)
            cos_mean = (d_commit @ d_mean).item()

        # Cosine with KL gradient
        cos_grad = 0.0
        if li in grad_dirs:
            d_grad = grad_dirs[li]
            d_grad = d_grad / (d_grad.norm() + 1e-12)
            cos_grad = (d_commit @ d_grad).item()

        # PCA rank diagnostic
        pca_explained = [0.0] * 10
        if delta_c_all is not None:
            dh_path = pca_dir / f"delta_h_layer_{li}.npy"
            if dh_path.exists():
                dh_subsample = torch.from_numpy(np.load(dh_path)).float()
                # Use train portion for PCA
                dh_train_pca = dh_subsample[:n_train]
                dc_train_pca = delta_c_all[:n_train]
                if dh_train_pca.shape[0] == dc_train_pca.shape[0] and dh_train_pca.shape[0] > 10:
                    pca_explained = _pca_explained_variance(dh_train_pca, dc_train_pca)

        # Δh norm mean (from accumulator)
        dh_norm_mean = 0.0
        if acc.n > 0:
            # ‖Δh‖² ≈ tr(XtX) / n  → ‖Δh‖ ≈ sqrt(tr(XtX) / n)
            trace_xtx = acc.XtX.diagonal().sum().item()
            dh_norm_mean = (trace_xtx / acc.n) ** 0.5

        result = {
            "layer": li,
            "r2_test": r2_test,
            "probe_magnitude": w_norm,
            "cosine_with_mean_dir": cos_mean,
            "cosine_with_kl_gradient": cos_grad,
            "pca_explained_var": pca_explained,
            "pca_explained_var_pc1": pca_explained[0],
            "delta_h_norm_mean": dh_norm_mean,
            "n_tokens": acc.n,
        }
        per_layer_results.append(result)

        if li % 5 == 0:
            log.info(
                "Layer %2d: R²=%.4f, ‖w‖=%.4f, cos(d_mean)=%.3f, cos(d_grad)=%.3f, PC1=%.3f",
                li, r2_test, w_norm, cos_mean, cos_grad, pca_explained[0],
            )

    # ── Go/No-Go decision ─────────────────────────────────────────────────────
    corrective_layers = list(range(corrective_onset, n_layers))
    corrective_cosines = [
        r["cosine_with_mean_dir"] for r in per_layer_results
        if r["layer"] in corrective_layers and r["cosine_with_mean_dir"] != 0.0
    ]
    if corrective_cosines:
        mean_cos = np.mean(np.abs(corrective_cosines))
        all_above_threshold = all(
            abs(c) > redundancy_cosine_threshold for c in corrective_cosines
        )
    else:
        mean_cos = 0.0
        all_above_threshold = False

    if all_above_threshold:
        go_nogo = "redundant"
        log.info("GO/NO-GO: REDUNDANT — d_commit ≈ d_mean at corrective layers (mean |cos|=%.3f)", mean_cos)
    else:
        go_nogo = "proceed"
        log.info("GO/NO-GO: PROCEED — d_commit differs from d_mean (mean |cos|=%.3f)", mean_cos)

    # ── Save outputs ──────────────────────────────────────────────────────────

    # Commitment directions NPZ (compatible with load_directions_from_npz)
    np.savez(output_dir / "commitment_directions.npz", **commitment_dir_arrays)
    log.info("Saved commitment directions → %s", output_dir / "commitment_directions.npz")

    # KL gradient directions (copy for convenience)
    if grad_path.exists():
        import shutil
        shutil.copy2(grad_path, output_dir / "kl_gradient_directions.npz")

    # Probe summary
    summary = {
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "corrective_onset": corrective_onset,
        "ridge_lambda": ridge_lambda,
        "go_nogo": go_nogo,
        "mean_corrective_cosine_abs": float(mean_cos),
        "per_layer": per_layer_results,
    }
    with open(output_dir / "probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved probe summary → %s", output_dir / "probe_summary.json")

    return summary


def _empty_layer_result(layer: int) -> dict:
    return {
        "layer": layer,
        "r2_test": 0.0,
        "probe_magnitude": 0.0,
        "cosine_with_mean_dir": 0.0,
        "cosine_with_kl_gradient": 0.0,
        "pca_explained_var": [0.0] * 10,
        "pca_explained_var_pc1": 0.0,
        "delta_h_norm_mean": 0.0,
        "n_tokens": 0,
    }


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Exp10 Phase 2+2.5: Ridge probes")
    parser.add_argument("--model", required=True)
    parser.add_argument("--paired-data-dir", required=True)
    parser.add_argument("--mean-dir-path", required=True,
                        help="Path to corrective_directions.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ridge-lambda", type=float, default=RIDGE_LAMBDA)
    args = parser.parse_args()

    train_probes(
        model_name=args.model,
        paired_data_dir=args.paired_data_dir,
        mean_dir_path=args.mean_dir_path,
        output_dir=args.output_dir,
        ridge_lambda=args.ridge_lambda,
    )
