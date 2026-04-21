from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.poc.exp05_corrective_direction_ablation_cartography.utils import ensure_dir, save_json


class PCAArtifactsUnavailableError(RuntimeError):
    """Raised when exact per-record PCA inputs are unavailable."""


@dataclass
class LayerPCAStats:
    layer: int
    n_records: int
    pc1_explained_variance: float
    cumulative_explained_variance: list[float]
    n_components_50: int
    n_components_80: int
    n_components_90: int
    mean_direction_pc1_abs_cosine: float | None
    participation_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "n_records": self.n_records,
            "pc1_explained_variance": self.pc1_explained_variance,
            "cumulative_explained_variance": self.cumulative_explained_variance,
            "n_components_50": self.n_components_50,
            "n_components_80": self.n_components_80,
            "n_components_90": self.n_components_90,
            "mean_direction_pc1_abs_cosine": self.mean_direction_pc1_abs_cosine,
            "participation_ratio": self.participation_ratio,
        }


def _is_record_diff_npz(path: Path) -> bool:
    if not path.exists() or path.suffix != ".npz":
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
        if "record_ids" not in keys:
            return False
        return any(
            key.startswith("layer_") or key.startswith("it_layer_") or key.startswith("pt_layer_")
            for key in keys
        )
    except Exception:
        return False


def _aggregate_only_npz(path: Path) -> bool:
    if not path.exists() or path.suffix != ".npz":
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
        return any(key.startswith("it_sum_") for key in keys) and any(key.startswith("pt_sum_") for key in keys)
    except Exception:
        return False


def _iter_local_candidates(base_dir: Path) -> list[Path]:
    candidates = [
        base_dir / "results" / "precompute_v2_work" / "record_diff_vectors.npz",
        base_dir / "results" / "precompute_v2_work" / "pca" / "record_diff_vectors.npz",
        base_dir / "results" / "exp5" / "precompute_v2" / "precompute" / "record_diff_vectors.npz",
    ]
    candidates.extend(sorted((base_dir / "results" / "precompute_v2_work").rglob("*record*diff*.npz")))
    candidates.extend(sorted((base_dir / "results" / "precompute_v2_work").rglob("*pca*.npz")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _load_record_diff_npz(path: Path, layers: list[int]) -> tuple[list[str], dict[int, np.ndarray]]:
    with np.load(path, allow_pickle=True) as data:
        record_ids = [str(x) for x in data["record_ids"].tolist()]
        out: dict[int, np.ndarray] = {}
        for layer in layers:
            layer_key = f"layer_{layer}"
            it_key = f"it_layer_{layer}"
            pt_key = f"pt_layer_{layer}"
            if layer_key in data.files:
                arr = np.asarray(data[layer_key], dtype=np.float64)
            elif it_key in data.files and pt_key in data.files:
                arr = np.asarray(data[it_key], dtype=np.float64) - np.asarray(data[pt_key], dtype=np.float64)
            else:
                continue
            if arr.ndim != 2:
                raise PCAArtifactsUnavailableError(
                    f"PCA artifact {path} contains {layer_key} with ndim={arr.ndim}; expected [n_records, d_model]."
                )
            if arr.shape[0] != len(record_ids):
                raise PCAArtifactsUnavailableError(
                    f"PCA artifact {path} has {arr.shape[0]} rows for layer {layer} but {len(record_ids)} record_ids."
                )
            out[layer] = arr
    if not out:
        raise PCAArtifactsUnavailableError(f"PCA artifact {path} does not contain any requested layers: {layers}")
    return record_ids, out


def _load_record_diff_dir(path: Path, layers: list[int]) -> tuple[list[str], dict[int, np.ndarray]]:
    npz_files = sorted(path.glob("*.npz"))
    if not npz_files:
        raise PCAArtifactsUnavailableError(f"No .npz shards found in {path}")
    all_ids: list[str] = []
    shards: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for npz_path in npz_files:
        ids, per_layer = _load_record_diff_npz(npz_path, layers)
        all_ids.extend(ids)
        for layer, arr in per_layer.items():
            shards[layer].append(arr)
    out = {
        layer: np.concatenate(parts, axis=0)
        for layer, parts in shards.items()
        if parts
    }
    if not out:
        raise PCAArtifactsUnavailableError(f"No usable PCA shards found in {path}")
    return all_ids, out


def _gsutil_ls(prefix: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["gsutil", "ls", prefix],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _try_fetch_gcs_record_diff_artifact(base_dir: Path, bucket: str = "pt-vs-it-results") -> Path | None:
    cache_dir = ensure_dir(base_dir / "results" / "precompute_v2_work" / "pca_cache")
    exact_candidates = [
        f"gs://{bucket}/results/precompute_v2_work/record_diff_vectors.npz",
        f"gs://{bucket}/results/precompute_v2_work/pca/record_diff_vectors.npz",
        f"gs://{bucket}/results/exp5/precompute_v2/precompute/record_diff_vectors.npz",
    ]
    for remote in exact_candidates:
        try:
            proc = subprocess.run(
                ["gsutil", "cp", remote, str(cache_dir / Path(remote).name)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except FileNotFoundError:
            return None
        local = cache_dir / Path(remote).name
        if proc.returncode == 0 and _is_record_diff_npz(local):
            return local

    broad_matches: list[str] = []
    for prefix in (
        f"gs://{bucket}/results/precompute_v2_work/**",
        f"gs://{bucket}/results/exp5/precompute_v2/precompute/**",
    ):
        broad_matches.extend(_gsutil_ls(prefix))

    for remote in broad_matches:
        name = Path(remote).name.lower()
        if not remote.endswith(".npz"):
            continue
        if "record" not in name and "diff" not in name and "pca" not in name:
            continue
        local = cache_dir / Path(remote).name
        try:
            proc = subprocess.run(
                ["gsutil", "cp", remote, str(local)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except FileNotFoundError:
            return None
        if proc.returncode == 0 and _is_record_diff_npz(local):
            return local
    return None


def discover_record_diff_artifact(base_dir: str | Path, bucket: str = "pt-vs-it-results") -> Path:
    root = Path(base_dir)
    for candidate in _iter_local_candidates(root):
        if _is_record_diff_npz(candidate):
            return candidate
    aggregate_candidates = [
        root / "results" / "precompute_v2_work" / "acts" / "w0.npz",
        root / "results" / "exp5" / "precompute_v2" / "precompute" / "corrective_directions.npz",
    ]
    remote = _try_fetch_gcs_record_diff_artifact(root, bucket=bucket)
    if remote is not None:
        return remote
    if any(_aggregate_only_npz(p) for p in aggregate_candidates if p.exists()):
        raise PCAArtifactsUnavailableError(
            "Exact PCA requires per-record IT–PT difference vectors, but the available v2/v4 artifacts "
            "only contain aggregated layer sums/counts or normalized mean directions."
        )
    raise PCAArtifactsUnavailableError("No PCA-ready per-record artifact found locally or in GCS.")


def load_record_diffs(base_dir: str | Path, layers: list[int], bucket: str = "pt-vs-it-results") -> tuple[Path, list[str], dict[int, np.ndarray]]:
    artifact = discover_record_diff_artifact(base_dir, bucket=bucket)
    if artifact.is_dir():
        record_ids, per_layer = _load_record_diff_dir(artifact, layers)
    else:
        record_ids, per_layer = _load_record_diff_npz(artifact, layers)
    return artifact, record_ids, per_layer


def _n_components_for_threshold(cumulative: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def _abs_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return abs(float(np.dot(a, b) / denom))


def load_mean_directions(path: str | Path) -> dict[int, np.ndarray]:
    src = Path(path)
    if not src.exists():
        return {}
    with np.load(src) as data:
        out = {}
        for key in data.files:
            if key.startswith("layer_"):
                out[int(key.split("_", 1)[1])] = np.asarray(data[key], dtype=np.float64)
        return out


def compute_layerwise_pca_stats(
    per_layer: dict[int, np.ndarray],
    mean_directions: dict[int, np.ndarray] | None = None,
) -> dict[int, LayerPCAStats]:
    stats: dict[int, LayerPCAStats] = {}
    mean_directions = mean_directions or {}
    for layer, arr in sorted(per_layer.items()):
        if arr.ndim != 2 or arr.shape[0] < 2:
            raise PCAArtifactsUnavailableError(
                f"Layer {layer} needs at least 2 per-record vectors for PCA; got shape {arr.shape}."
            )
        centered = arr - arr.mean(axis=0, keepdims=True)
        _, singular_vals, vt = np.linalg.svd(centered, full_matrices=False)
        variances = (singular_vals ** 2) / max(arr.shape[0] - 1, 1)
        total = float(variances.sum())
        if total <= 0:
            explained = np.zeros_like(variances)
        else:
            explained = variances / total
        cumulative = np.cumsum(explained)
        pc1 = vt[0]
        mean_dir = mean_directions.get(layer)
        abs_cos = _abs_cosine(pc1, mean_dir) if mean_dir is not None else None
        participation_ratio = float((variances.sum() ** 2) / max((variances ** 2).sum(), 1e-12))
        stats[layer] = LayerPCAStats(
            layer=layer,
            n_records=int(arr.shape[0]),
            pc1_explained_variance=float(explained[0]) if explained.size else 0.0,
            cumulative_explained_variance=[float(x) for x in cumulative[: min(10, cumulative.size)]],
            n_components_50=_n_components_for_threshold(cumulative, 0.50),
            n_components_80=_n_components_for_threshold(cumulative, 0.80),
            n_components_90=_n_components_for_threshold(cumulative, 0.90),
            mean_direction_pc1_abs_cosine=abs_cos,
            participation_ratio=participation_ratio,
        )
    return stats


def save_pca_stats_json(
    out_path: str | Path,
    artifact_path: Path,
    record_ids: list[str],
    stats: dict[int, LayerPCAStats],
) -> Path:
    payload = {
        "artifact_path": str(artifact_path),
        "n_records": len(record_ids),
        "layers": {str(layer): stat.to_dict() for layer, stat in stats.items()},
    }
    save_json(out_path, payload)
    return Path(out_path)


def plot_layerwise_pca_stats(
    stats: dict[int, LayerPCAStats],
    out_path: str | Path,
    title: str = "Exp6 A1 — PCA of IT-PT Difference Vectors",
) -> Path:
    layers = sorted(stats)
    pc1 = [stats[layer].pc1_explained_variance for layer in layers]
    n80 = [stats[layer].n_components_80 for layer in layers]
    cos = [stats[layer].mean_direction_pc1_abs_cosine for layer in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    def _shade_corrective(ax):
        ax.axvspan(19.5, 33.5, color="#fce5cd", alpha=0.35, zorder=0)

    _shade_corrective(ax_a)
    ax_a.plot(layers, pc1, marker="o", color="#d62728", linewidth=2.0)
    ax_a.set_title("Panel A — PC1 explained variance by layer", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("Explained variance ratio")
    ax_a.set_ylim(0.0, 1.0)
    ax_a.grid(True, alpha=0.2)

    _shade_corrective(ax_b)
    ax_b.plot(layers, n80, marker="o", color="#1f77b4", linewidth=2.0)
    ax_b.set_title("Panel B — PCs needed to reach 80% variance", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Layer")
    ax_b.set_ylabel("# PCs to 80%")
    ax_b.grid(True, alpha=0.2)

    _shade_corrective(ax_c)
    clean_cos = [np.nan if c is None else c for c in cos]
    ax_c.plot(layers, clean_cos, marker="o", color="#2ca02c", linewidth=2.0)
    ax_c.set_title("Panel C — |cos(PC1, mean IT-PT direction)|", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Layer")
    ax_c.set_ylabel("Absolute cosine")
    ax_c.set_ylim(0.0, 1.0)
    ax_c.grid(True, alpha=0.2)

    corrective_layers = [layer for layer in layers if 20 <= layer <= 33]
    if corrective_layers:
        curves = []
        for layer in corrective_layers:
            curve = stats[layer].cumulative_explained_variance
            curves.append(curve)
            ax_d.plot(
                range(1, len(curve) + 1),
                curve,
                color="#7f7f7f",
                alpha=0.35,
                linewidth=1.0,
            )
        max_len = max(len(curve) for curve in curves)
        padded = np.full((len(curves), max_len), np.nan, dtype=np.float64)
        for idx, curve in enumerate(curves):
            padded[idx, : len(curve)] = curve
        mean_curve = np.nanmean(padded, axis=0)
        ax_d.plot(range(1, max_len + 1), mean_curve, color="#d62728", linewidth=2.5, label="Mean corrective layer")
        ax_d.set_title("Panel D — Corrective-layer cumulative variance", fontsize=11, fontweight="bold")
        ax_d.set_xlabel("Principal component")
        ax_d.set_ylabel("Cumulative explained variance")
        ax_d.set_ylim(0.0, 1.0)
        ax_d.grid(True, alpha=0.2)
        ax_d.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"{title}\nPer-record token-mean IT-PT difference vectors; corrective layers shaded",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def compute_and_plot_pca_analysis(
    base_dir: str | Path,
    out_json_path: str | Path,
    out_plot_path: str | Path,
    mean_direction_path: str | Path,
    layers: list[int] | None = None,
    bucket: str = "pt-vs-it-results",
) -> tuple[Path, Path]:
    layers = layers or list(range(1, 34))
    artifact, record_ids, per_layer = load_record_diffs(base_dir, layers, bucket=bucket)
    mean_dirs = load_mean_directions(mean_direction_path)
    stats = compute_layerwise_pca_stats(per_layer, mean_dirs)
    json_path = save_pca_stats_json(out_json_path, artifact, record_ids, stats)
    plot_path = plot_layerwise_pca_stats(stats, out_plot_path)
    return json_path, plot_path
