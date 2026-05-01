"""Train per-layer BatchTopK crosscoders from Exp28 activation caches."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import (
    BatchTopKCrossCoder,
    CrosscoderConfig,
    fvu,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _stack_cache(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([payload["pt_mlp"], payload["it_mlp"]], dim=1)
    is_val = payload.get("is_val")
    if is_val is None:
        is_val = torch.zeros(x.shape[0], dtype=torch.bool)
        is_val[::10] = True
    return x, is_val.bool()


def _lr_for_dict_size(dict_size: int, base_lr: float | None) -> float:
    if base_lr is not None:
        return float(base_lr)
    return 2e-4 / math.sqrt(float(dict_size) / float(2**14))


def _sample_batch(
    x: torch.Tensor,
    train_idx: torch.Tensor,
    *,
    batch_tokens: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    choices = train_idx[torch.randint(len(train_idx), (batch_tokens,), generator=generator)]
    return x[choices].to(device=device, dtype=torch.float32, non_blocking=True)


def _auxk_loss(
    model: BatchTopKCrossCoder,
    x: torch.Tensor,
    recon: torch.Tensor,
    features: torch.Tensor,
    preacts: torch.Tensor,
    *,
    auxk: int,
) -> torch.Tensor:
    if auxk <= 0:
        return x.new_tensor(0.0)
    acts = torch.relu(preacts)
    inactive = features <= 0
    if not bool(inactive.any().item()):
        return x.new_tensor(0.0)
    aux_acts = acts.masked_fill(~inactive, 0.0)
    budget = min(aux_acts.numel(), aux_acts.shape[0] * auxk)
    if budget <= 0:
        return x.new_tensor(0.0)
    cutoff = torch.topk(aux_acts.flatten(), k=budget, largest=True).values[-1]
    aux_features = aux_acts * (aux_acts >= cutoff).to(aux_acts.dtype)
    aux_recon = model.decode(aux_features) - model.decoder_bias
    residual = (x.float() - recon.float()).detach()
    return F.mse_loss(aux_recon, residual)


@torch.no_grad()
def _eval_model(
    model: BatchTopKCrossCoder,
    x: torch.Tensor,
    val_idx: torch.Tensor,
    *,
    batch_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    if len(val_idx) == 0:
        val_idx = torch.arange(min(len(x), batch_tokens))
    take = val_idx[: min(len(val_idx), max(batch_tokens, 1024))]
    xb = x[take].to(device=device, dtype=torch.float32)
    recon, features = model(xb, output_features=True)
    fv = fvu(recon.detach().cpu(), xb.detach().cpu())
    l0 = (features > 0).float().sum(dim=1).mean()
    alive = (features > 0).any(dim=0).float().mean()
    mse = F.mse_loss(recon.float(), xb.float())
    return {
        "heldout_fvu_pt": float(fv[0].item()),
        "heldout_fvu_it": float(fv[1].item()),
        "heldout_variance_explained_pt": float(1.0 - fv[0].item()),
        "heldout_variance_explained_it": float(1.0 - fv[1].item()),
        "heldout_mse": float(mse.item()),
        "effective_l0": float(l0.item()),
        "alive_fraction": float(alive.item()),
        "dead_latents": int(features.shape[1] - int((features > 0).any(dim=0).sum().item())),
        "inference_threshold": float(model.inference_threshold.item()),
    }


def train_layer(args: argparse.Namespace, layer: int) -> dict[str, Any]:
    cache_path = args.run_root / "cache" / f"layer_{layer}.pt"
    payload = _load_cache(cache_path)
    x, is_val = _stack_cache(payload)
    train_idx = torch.nonzero(~is_val, as_tuple=False).flatten()
    val_idx = torch.nonzero(is_val, as_tuple=False).flatten()
    if len(train_idx) == 0:
        raise RuntimeError(f"No train tokens in {cache_path}")

    activation_dim = int(x.shape[-1])
    input_mean = x[train_idx].float().mean(dim=0)
    config = CrosscoderConfig(
        activation_dim=activation_dim,
        dict_size=args.dict_size,
        k=args.k,
        threshold_beta=args.threshold_beta,
        threshold_start_step=args.threshold_start_step,
    )
    device = torch.device(args.device)
    model = BatchTopKCrossCoder(config, input_mean=input_mean, device=device)
    model.train()
    lr = _lr_for_dict_size(args.dict_size, args.lr)
    out_dir = args.run_root / "dictionaries" / f"layer_{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    generator = torch.Generator().manual_seed(args.seed + layer)
    ckpt_path = out_dir / "checkpoint_latest.pt"
    final_path = out_dir / "crosscoder.pt"
    start_step = 1
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        generator.set_state(ckpt["generator_state"].cpu())
        start_step = int(ckpt["step"]) + 1
        log.info("[exp28-train] layer=%d resuming from step=%d", layer, start_step - 1)
    elif args.resume and final_path.exists():
        payload_final = torch.load(final_path, map_location=device, weights_only=False)
        saved_config = payload_final.get("config", {})
        expected = config.__dict__
        for key in ("activation_dim", "dict_size", "k", "n_branches"):
            if int(saved_config.get(key, -1)) != int(expected[key]):
                raise RuntimeError(
                    f"Cannot resume layer={layer} from {final_path}: "
                    f"saved {key}={saved_config.get(key)} but requested {expected[key]}"
                )
        model.load_state_dict(payload_final["state_dict"])
        previous_steps = int(payload_final.get("extra", {}).get("steps", 0))
        start_step = previous_steps + 1
        log.info(
            "[exp28-train] layer=%d continuing from final weights at step=%d",
            layer,
            previous_steps,
        )
    auxk = int(args.auxk if args.auxk is not None else args.k)

    log_mode = "a" if args.resume and start_step > 1 else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_f:
        for step in range(start_step, args.steps + 1):
            xb = _sample_batch(
                x,
                train_idx,
                batch_tokens=args.batch_tokens,
                generator=generator,
                device=device,
            )
            opt.zero_grad(set_to_none=True)
            recon, features, preacts = model(xb, output_features=True, return_preacts=True)
            main_loss = F.mse_loss(recon.float(), xb.float())
            aux_loss = _auxk_loss(
                model,
                xb,
                recon,
                features,
                preacts,
                auxk=auxk,
            )
            loss = main_loss + float(args.auxk_alpha) * aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at layer={layer} step={step}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            if args.warmup_steps > 0 and step <= args.warmup_steps:
                scale = step / float(args.warmup_steps)
                for group in opt.param_groups:
                    group["lr"] = lr * scale
            else:
                for group in opt.param_groups:
                    group["lr"] = lr
            opt.step()
            model.normalize_decoder_()
            model.update_threshold(preacts, step=step)

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                metrics = _eval_model(
                    model,
                    x,
                    val_idx,
                    batch_tokens=min(args.batch_tokens, 2048),
                    device=device,
                )
                row = {
                    "layer": layer,
                    "step": step,
                    "loss": float(loss.item()),
                    "main_loss": float(main_loss.item()),
                    "aux_loss": float(aux_loss.item()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    **metrics,
                }
                log_f.write(json.dumps(row, separators=(",", ":")) + "\n")
                log_f.flush()
                log.info(
                    "[exp28-train] layer=%d step=%d loss=%.4g ve_pt=%.3f ve_it=%.3f l0=%.1f",
                    layer,
                    step,
                    row["loss"],
                    row["heldout_variance_explained_pt"],
                    row["heldout_variance_explained_it"],
                    row["effective_l0"],
                )
            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0):
                torch.save(
                    {
                        "step": int(step),
                        "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "generator_state": generator.get_state(),
                        "config": config.__dict__,
                    },
                    ckpt_path,
                )

    metrics = _eval_model(
        model,
        x,
        val_idx,
        batch_tokens=min(max(args.batch_tokens, 2048), 8192),
        device=device,
    )
    extra = {
        "model": payload.get("model"),
        "layer": int(layer),
        "train_tokens": int(len(train_idx)),
        "val_tokens": int(len(val_idx)),
        "lr": lr,
        "steps": int(args.steps),
        "batch_tokens": int(args.batch_tokens),
        "metrics": metrics,
    }
    model.save(out_dir / "crosscoder.pt", extra=extra)
    if ckpt_path.exists():
        ckpt_path.unlink()
    (out_dir / "config.json").write_text(
        json.dumps({"crosscoder": config.__dict__, **extra}, indent=2) + "\n",
        encoding="utf-8",
    )
    return extra


def run(args: argparse.Namespace) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for layer in args.layers:
        summaries.append(train_layer(args, int(layer)))
    out = {
        "layers": [int(x) for x in args.layers],
        "dict_size": int(args.dict_size),
        "k": int(args.k),
        "steps": int(args.steps),
        "batch_tokens": int(args.batch_tokens),
        "summaries": summaries,
    }
    summary_path = args.run_root / "dictionaries" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--dict-size", type=int, default=32768)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--threshold-beta", type=float, default=0.999)
    parser.add_argument("--threshold-start-step", type=int, default=1000)
    parser.add_argument("--auxk-alpha", type=float, default=1.0 / 32.0)
    parser.add_argument("--auxk", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
