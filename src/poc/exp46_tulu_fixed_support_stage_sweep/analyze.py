"""Analyze Exp46 fixed-support Tulu stage decomposition artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.poc.exp46_tulu_fixed_support_stage_sweep import NON_BASE_STAGES, STAGE_ORDER, STAGES
from src.poc.exp46_tulu_fixed_support_stage_sweep.common import (
    json_rows,
    load_stage,
    rank_from_logits,
    stage_adapter,
    stage_key_for_label,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _seed_offset(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100000


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _read_cell_records(fixed_root: Path) -> tuple[list[dict[str, Any]], torch.Tensor]:
    records: list[dict[str, Any]] = []
    hiddens: list[torch.Tensor] = []
    for host_stage in STAGE_ORDER:
        host_dir = fixed_root / host_stage
        paths = sorted(host_dir.glob("cells_w*.jsonl.gz"))
        if not paths and (host_dir / "cells.jsonl.gz").exists():
            paths = [host_dir / "cells.jsonl.gz"]
        hidden_cache: dict[str, torch.Tensor] = {}
        for path in paths:
            for row in json_rows(path):
                hidden_file = str(row["hidden_file"])
                if hidden_file not in hidden_cache:
                    hidden_cache[hidden_file] = torch.load(host_dir / hidden_file, map_location="cpu")["final_hidden"]
                idx = int(row["hidden_index"])
                hidden = hidden_cache[hidden_file][idx].detach().cpu().to(torch.bfloat16)
                if hidden.ndim != 1 or int(hidden.numel()) != 4096:
                    raise RuntimeError(
                        "Exp46 fixed-factorial final hidden has invalid shape; "
                        f"host={host_stage} file={hidden_file} index={idx} shape={tuple(hidden.shape)}"
                    )
                row = dict(row)
                row["_global_hidden_index"] = len(hiddens)
                hiddens.append(hidden)
                records.append(row)
    if not records:
        raise FileNotFoundError(f"No fixed factorial cell records under {fixed_root}")
    return records, torch.stack(hiddens, dim=0)


@torch.no_grad()
def _score_stage_readout(
    *,
    stage: str,
    hiddens: torch.Tensor,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    stage = stage_key_for_label(stage)
    model, tokenizer = load_stage(stage, device)
    adapter = stage_adapter()
    final_norm = adapter.get_final_norm(model)
    lm_head = adapter.get_lm_head(model)
    real_mask = adapter.real_token_mask(tokenizer, device, model)
    norm_dtype = next(final_norm.parameters()).dtype
    head_dtype = next(lm_head.parameters()).dtype
    out: list[dict[str, Any]] = []
    log.info("[exp46] scoring readout stage=%s n=%d", stage, len(records))
    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        batch = hiddens[start:end].to(device=device, dtype=norm_dtype)
        normed = final_norm(batch.view(end - start, 1, -1)).view(end - start, -1)
        logits = lm_head(normed.to(dtype=head_dtype)).float()
        logits[:, ~real_mask] = float("-inf")
        top1 = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        for offset, rec in enumerate(records[start:end]):
            row_idx = start + offset
            t_base = int(rec["t_base_id"])
            t_rlvr = int(rec["t_rlvr_id"])
            row_logits = logits[offset]
            top = int(top1[offset])
            out.append(
                {
                    "record_index": row_idx,
                    "readout": f"common_{stage.lower()}",
                    "readout_stage": stage,
                    "prompt_id": rec["prompt_id"],
                    "event_kind": rec.get("event_kind", "first_diff"),
                    "cell": rec["cell"],
                    "upstream_stage": rec["upstream_stage"],
                    "late_stage": rec["late_stage"],
                    "margin_rlvr_minus_base": float((row_logits[t_rlvr] - row_logits[t_base]).item()),
                    "rlvr_logit": float(row_logits[t_rlvr].item()),
                    "base_logit": float(row_logits[t_base].item()),
                    "final_top1_id": top,
                    "winner": "rlvr" if top == t_rlvr else ("base" if top == t_base else "other"),
                    "rlvr_rank": rank_from_logits(row_logits, t_rlvr),
                    "base_rank": rank_from_logits(row_logits, t_base),
                }
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, separators=(",", ":")) + "\n")


def _cluster_bootstrap(values: list[tuple[str, float]], n_boot: int, seed: int) -> dict[str, Any]:
    clean = [(str(cluster), float(value)) for cluster, value in values if np.isfinite(value)]
    if not clean:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n_units": 0, "n_prompt_clusters": 0}
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for cluster, value in clean:
        by_cluster[cluster].append(value)
    clusters = np.array(sorted(by_cluster), dtype=object)
    cluster_values = {cluster: np.array(by_cluster[str(cluster)], dtype=float) for cluster in clusters}
    point = float(np.mean([v for _, v in clean]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        vals = np.concatenate([cluster_values[str(cluster)] for cluster in sample])
        boots.append(float(np.mean(vals)))
    lo, hi = np.percentile(np.array(boots, dtype=float), [2.5, 97.5])
    return {
        "estimate": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_units": len(clean),
        "n_prompt_clusters": len(clusters),
    }


def _pivot_scored(scored: list[dict[str, Any]], readout: str) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    piv: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scored:
        if row["readout"] != readout:
            continue
        piv[(row["prompt_id"], row.get("event_kind", "first_diff"))][row["cell"]] = row
    return piv


def _unit_effects(scored: list[dict[str, Any]], readout: str, stage: str) -> list[dict[str, Any]]:
    stage = stage_key_for_label(stage)
    piv = _pivot_scored(scored, readout)
    out = []
    for (pid, event_kind), cells in piv.items():
        needed = [f"U_{stage}__L_{stage}", f"U_{stage}__L_B", f"U_B__L_{stage}", "U_B__L_B"]
        if any(name not in cells for name in needed):
            continue
        y_vv = cells[f"U_{stage}__L_{stage}"]["margin_rlvr_minus_base"]
        y_vb = cells[f"U_{stage}__L_B"]["margin_rlvr_minus_base"]
        y_bv = cells[f"U_B__L_{stage}"]["margin_rlvr_minus_base"]
        y_bb = cells["U_B__L_B"]["margin_rlvr_minus_base"]
        out.append(
            {
                "prompt_id": pid,
                "event_kind": event_kind,
                "stage": stage,
                "interaction": y_vv - y_vb - y_bv + y_bb,
                "late_effect_from_base": y_bv - y_bb,
                "late_effect_from_matched": y_vv - y_vb,
            }
        )
    return out


def _summarize_effects(scored: list[dict[str, Any]], metadata: dict[tuple[str, str], dict[str, Any]], n_boot: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    readouts = sorted({row["readout"] for row in scored} | {"native"})
    common_by_record_cell = {
        (row["record_index"], row["readout"]): row
        for row in scored
        if row["readout"].startswith("common_")
    }
    native_rows = []
    if "native" in readouts:
        for row in scored:
            stage_readout = f"common_{str(row['late_stage']).lower()}"
            if row["readout"] == stage_readout:
                native = dict(row)
                native["readout"] = "native"
                native_rows.append(native)
    scored_all = scored + native_rows
    for readout in sorted({row["readout"] for row in scored_all}):
        summary[readout] = {}
        for stage in NON_BASE_STAGES:
            units = _unit_effects(scored_all, readout, stage)
            for metric in ("interaction", "late_effect_from_base", "late_effect_from_matched"):
                values = [(unit["prompt_id"], unit[metric]) for unit in units]
                est = _cluster_bootstrap(values, n_boot=n_boot, seed=seed + _seed_offset(readout, stage, metric))
                summary[readout].setdefault(stage, {})[metric] = est
                rows.append(
                    {
                        "readout": readout,
                        "stage": stage,
                        "metric": metric,
                        **est,
                    }
                )
            # Required position/category subsets for interaction.
            for subgroup_name, predicate in (
                ("position_ge3", lambda meta: int(meta.get("divergence_step", -1)) >= 3),
                ("all", lambda meta: True),
            ):
                vals = []
                for unit in units:
                    meta = metadata.get((unit["prompt_id"], unit["event_kind"]), {})
                    if predicate(meta):
                        vals.append((unit["prompt_id"], unit["interaction"]))
                summary[readout][stage][f"interaction_{subgroup_name}"] = _cluster_bootstrap(
                    vals, n_boot=n_boot, seed=seed + _seed_offset(readout, stage, subgroup_name)
                )
            for field in ("prompt_category", "t_rlvr_category", "t_base_category"):
                groups = sorted(
                    {
                        str(metadata.get((unit["prompt_id"], unit["event_kind"]), {}).get(field))
                        for unit in units
                        if metadata.get((unit["prompt_id"], unit["event_kind"]), {}).get(field) is not None
                    }
                )
                for group in groups:
                    vals = [
                        (unit["prompt_id"], unit["interaction"])
                        for unit in units
                        if str(metadata.get((unit["prompt_id"], unit["event_kind"]), {}).get(field)) == group
                    ]
                    summary[readout][stage].setdefault("category_breakdowns", {}).setdefault(field, {})[group] = _cluster_bootstrap(
                        vals, n_boot=n_boot, seed=seed + _seed_offset(readout, stage, field, group)
                    )
    return summary, rows


def _paired_ratio_bootstrap(
    numerator_units: list[dict[str, Any]],
    denominator_units: list[dict[str, Any]],
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    denom_by_key = {
        (unit["prompt_id"], unit.get("event_kind", "first_diff")): float(unit[metric])
        for unit in denominator_units
        if np.isfinite(float(unit[metric]))
    }
    pairs = []
    for unit in numerator_units:
        key = (unit["prompt_id"], unit.get("event_kind", "first_diff"))
        if key in denom_by_key and np.isfinite(float(unit[metric])):
            pairs.append((str(unit["prompt_id"]), float(unit[metric]), denom_by_key[key]))
    if not pairs:
        return {
            "ratio_estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "denominator_mean": None,
            "numerator_mean": None,
            "denominator_ci_excludes_zero": False,
            "n_units": 0,
            "n_prompt_clusters": 0,
            "status": "no_overlap",
        }
    by_cluster: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for cluster, numerator, denominator in pairs:
        by_cluster[cluster].append((numerator, denominator))
    clusters = np.array(sorted(by_cluster), dtype=object)
    nums = np.array([num for _, num, _ in pairs], dtype=float)
    dens = np.array([den for _, _, den in pairs], dtype=float)
    denominator_summary = _cluster_bootstrap(
        [(cluster, den) for cluster, _, den in pairs],
        n_boot=n_boot,
        seed=seed + 17,
    )
    denom_mean = float(dens.mean())
    ratio = float(nums.mean() / denom_mean) if abs(denom_mean) > 1e-12 else None
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        vals = [pair for cluster in sample for pair in by_cluster[str(cluster)]]
        num_mean = float(np.mean([num for num, _ in vals]))
        den_mean = float(np.mean([den for _, den in vals]))
        if abs(den_mean) > 1e-12:
            boots.append(num_mean / den_mean)
    if boots:
        lo, hi = np.percentile(np.array(boots, dtype=float), [2.5, 97.5])
    else:
        lo, hi = np.nan, np.nan
    denom_low = denominator_summary.get("ci95_low")
    denom_high = denominator_summary.get("ci95_high")
    denom_excludes_zero = (
        denom_low is not None
        and denom_high is not None
        and ((float(denom_low) > 0 and float(denom_high) > 0) or (float(denom_low) < 0 and float(denom_high) < 0))
    )
    return {
        "ratio_estimate": ratio,
        "ci95_low": float(lo) if np.isfinite(lo) else None,
        "ci95_high": float(hi) if np.isfinite(hi) else None,
        "denominator_mean": denom_mean,
        "numerator_mean": float(nums.mean()),
        "denominator_ci_excludes_zero": bool(denom_excludes_zero),
        "n_units": len(pairs),
        "n_prompt_clusters": len(clusters),
        "status": "ok" if denom_excludes_zero else "denominator_ci_includes_zero_do_not_interpret_as_fraction",
    }


def _summarize_stage_fractions(scored: list[dict[str, Any]], n_boot: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_all = list(scored)
    native_rows = []
    for row in scored:
        if row["readout"] == f"common_{str(row['late_stage']).lower()}":
            payload = dict(row)
            payload["readout"] = "native"
            native_rows.append(payload)
    scored_all.extend(native_rows)
    out: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for readout in sorted({row["readout"] for row in scored_all}):
        denom_units = _unit_effects(scored_all, readout, "R")
        out[readout] = {}
        for stage in NON_BASE_STAGES:
            num_units = _unit_effects(scored_all, readout, stage)
            est = _paired_ratio_bootstrap(
                num_units,
                denom_units,
                metric="interaction",
                n_boot=n_boot,
                seed=seed + _seed_offset("fraction", readout, stage),
            )
            out[readout][stage] = est
            rows.append({"readout": readout, "stage": stage, "metric": "interaction_fraction_of_final", **est})
    return out, rows


def _label_swap_null(scored: list[dict[str, Any]], readout: str, n_perm: int, seed: int) -> dict[str, Any]:
    units = _unit_effects(scored, readout, "R")
    values = np.array([unit["interaction"] for unit in units], dtype=float)
    if values.size == 0:
        return {"readout": readout, "n_units": 0}
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=values.shape[0], replace=True)
        null.append(float(np.mean(values * signs)))
    arr = np.array(null, dtype=float)
    return {
        "readout": readout,
        "observed": observed,
        "n_units": int(values.size),
        "null_mean": float(arr.mean()),
        "null_p999": float(np.percentile(arr, 99.9)),
        "p_upper": float((np.sum(arr >= observed) + 1) / (len(arr) + 1)),
    }


def _metadata_from_cells(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    meta = {}
    for rec in records:
        key = (rec["prompt_id"], rec.get("event_kind", "first_diff"))
        if key not in meta:
            meta[key] = {
                "divergence_step": rec.get("divergence_step"),
                "prompt_category": rec.get("prompt_category"),
                "source": rec.get("source"),
                "t_base_category": rec.get("t_base_category"),
                "t_rlvr_category": rec.get("t_rlvr_category"),
                "t_base_assistant_marker": rec.get("t_base_assistant_marker"),
                "t_rlvr_assistant_marker": rec.get("t_rlvr_assistant_marker"),
            }
    return meta


def _summarize_stage_cache(stage_cache_root: Path) -> dict[str, Any]:
    out = {}
    for stage in STAGE_ORDER:
        values = []
        classes = []
        for path in sorted((stage_cache_root / stage).glob("records_w*.jsonl.gz")):
            for row in json_rows(path):
                if row.get("late_delta_cosine_mean") is not None:
                    values.append(float(row["late_delta_cosine_mean"]))
                if row.get("native_top1_class") is not None:
                    classes.append(str(row["native_top1_class"]))
        counts = {cls: classes.count(cls) for cls in sorted(set(classes))}
        total = sum(counts.values()) or 1
        out[stage] = {
            "n": len(classes),
            "native_top1_rates": {cls: count / total for cls, count in counts.items()},
            "late_delta_cosine_mean": float(np.mean(values)) if values else None,
        }
    return out


def _summarize_identity(identity_root: Path | None) -> dict[str, Any]:
    if identity_root is None or not identity_root.exists():
        return {}
    out = {}
    for stage in NON_BASE_STAGES:
        rows = []
        merged = identity_root / stage / "identity_margin.jsonl.gz"
        paths = [merged] if merged.exists() else sorted((identity_root / stage).glob("identity_margin_w*.jsonl.gz"))
        for path in paths:
            rows.extend(list(json_rows(path)))
        if not rows:
            continue
        stage_payload = {}
        for cond in ("B_mid", "B_late", "B_midlate", "D_mid", "D_late", "D_midlate", "A_base", "C_stage"):
            winners = []
            margins = []
            for row in rows:
                payload = (row.get("conditions") or {}).get(cond) or {}
                if payload.get("winner") is not None:
                    winners.append(str(payload["winner"]))
                if payload.get("margin_rlvr_minus_base") is not None:
                    margins.append(float(payload["margin_rlvr_minus_base"]))
            total = len(winners) or 1
            stage_payload[cond] = {
                "n": len(winners),
                "winner_rates": {cls: winners.count(cls) / total for cls in sorted(set(winners))},
                "margin_mean": float(np.mean(margins)) if margins else None,
            }
        out[stage] = stage_payload
    return out


def _write_effects_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "readout",
        "stage",
        "metric",
        "estimate",
        "ci95_low",
        "ci95_high",
        "n_units",
        "n_prompt_clusters",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_stage_fraction_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "readout",
        "stage",
        "metric",
        "ratio_estimate",
        "ci95_low",
        "ci95_high",
        "numerator_mean",
        "denominator_mean",
        "denominator_ci_excludes_zero",
        "n_units",
        "n_prompt_clusters",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _read_json_if_exists(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _plot(summary: dict[str, Any], out_path: Path) -> None:
    common = summary.get("effects", {}).get("common_r", {})
    stages = list(NON_BASE_STAGES)
    vals = [((common.get(stage) or {}).get("interaction") or {}).get("estimate", np.nan) for stage in stages]
    lows = [((common.get(stage) or {}).get("interaction") or {}).get("ci95_low", np.nan) for stage in stages]
    highs = [((common.get(stage) or {}).get("interaction") or {}).get("ci95_high", np.nan) for stage in stages]
    adoption = summary.get("stage_cache", {})
    rlvr_rates = [(adoption.get(stage, {}).get("native_top1_rates", {}).get("rlvr", 0.0)) for stage in STAGE_ORDER]
    delta = [(adoption.get(stage, {}).get("late_delta_cosine_mean", np.nan)) for stage in STAGE_ORDER]
    identity = summary.get("identity_margin", {})
    mid_rates = [identity.get(stage, {}).get("B_mid", {}).get("winner_rates", {}).get("rlvr", np.nan) for stage in stages]
    late_rates = [identity.get(stage, {}).get("B_late", {}).get("winner_rates", {}).get("rlvr", np.nan) for stage in stages]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(stages))
    axes[0, 0].bar(x, vals, color="#4C78A8")
    yerr = np.vstack([np.array(vals) - np.array(lows), np.array(highs) - np.array(vals)])
    axes[0, 0].errorbar(x, vals, yerr=np.nan_to_num(yerr), fmt="none", color="black", capsize=4)
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set_xticks(x, stages)
    axes[0, 0].set_title("Fixed-support cumulative interaction")
    axes[0, 0].set_ylabel("logit margin interaction")

    x2 = np.arange(len(STAGE_ORDER))
    axes[0, 1].bar(x2, rlvr_rates, color="#54A24B")
    axes[0, 1].set_xticks(x2, STAGE_ORDER)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Stage native top-1 adopts t_R")

    axes[1, 0].plot(x2, delta, marker="o", color="#E45756")
    axes[1, 0].axhline(0, color="black", lw=0.8)
    axes[1, 0].set_xticks(x2, STAGE_ORDER)
    axes[1, 0].set_title("Late MLP delta-cosine")

    axes[1, 1].bar(x - 0.18, mid_rates, width=0.36, label="Base + stage mid", color="#72B7B2")
    axes[1, 1].bar(x + 0.18, late_rates, width=0.36, label="Base + stage late", color="#F58518")
    axes[1, 1].set_xticks(x, stages)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Fixed-support RLVR-token identity transfer")
    axes[1, 1].legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    records, hiddens = _read_cell_records(args.fixed_root)
    metadata = _metadata_from_cells(records)
    all_scored: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        all_scored.extend(
            _score_stage_readout(
                stage=stage,
                hiddens=hiddens,
                records=records,
                device=device,
                batch_size=args.batch_size,
            )
        )
    # Add native readouts by aliasing each cell to its late-stage common readout.
    native_rows = []
    for row in all_scored:
        if row["readout"] == f"common_{str(row['late_stage']).lower()}":
            payload = dict(row)
            payload["readout"] = "native"
            native_rows.append(payload)
    scored_with_native = all_scored + native_rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl_gz(args.out_dir / "scored_cells.jsonl.gz", scored_with_native)
    effects, effect_rows = _summarize_effects(all_scored, metadata, n_boot=args.n_boot, seed=args.seed)
    stage_fractions, stage_fraction_rows = _summarize_stage_fractions(
        all_scored,
        n_boot=args.n_boot,
        seed=args.seed + 461,
    )
    label_swap = {
        readout: _label_swap_null(scored_with_native, readout, n_perm=args.n_permutations, seed=args.seed + idx)
        for idx, readout in enumerate(["common_r", "common_b", "common_s", "common_d", "native"])
    }
    summary = {
        "experiment": "exp46_tulu_fixed_support_stage_sweep",
        "fixed_root": str(args.fixed_root),
        "stage_cache_root": str(args.stage_cache_root),
        "identity_root": str(args.identity_root) if args.identity_root else None,
        "preflight_dir": str(args.preflight_dir) if args.preflight_dir else None,
        "n_cell_records": len(records),
        "n_scored_records": len(scored_with_native),
        "effects": effects,
        "stage_fractions": stage_fractions,
        "label_swap_null": label_swap,
        "stage_cache": _summarize_stage_cache(args.stage_cache_root),
        "identity_margin": _summarize_identity(args.identity_root),
        "preflight": _read_json_if_exists(args.preflight_dir / "summary.json") if args.preflight_dir else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_effects_csv(effect_rows, args.out_dir / "effects.csv")
    _write_stage_fraction_csv(stage_fraction_rows, args.out_dir / "stage_fraction_ratios.csv")
    _plot(summary, args.out_dir / "exp46_stage_decomposition.png")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--stage-cache-root", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, default=None)
    parser.add_argument("--preflight-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=35)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps({"out": summary.get("experiment"), "n_cell_records": summary["n_cell_records"]}, indent=2))


if __name__ == "__main__":
    main()
