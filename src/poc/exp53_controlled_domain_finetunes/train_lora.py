"""Train an Exp53 LoRA continuation adapter on packed domain text."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .common import BASE_MODEL_ID, BASE_REVISION, iter_jsonl, paths_for, write_json


def _lazy_imports():
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # type: ignore

    return LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class PackedJsonlDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        *,
        seq_len: int,
        rank: int,
        world_size: int,
        seed_offset: int = 0,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed_offset = seed_offset

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        n_workers = worker.num_workers if worker is not None else 1
        shard_id = self.rank * n_workers + worker_id
        n_shards = self.world_size * n_workers
        eos = self.tokenizer.eos_token_id
        buffer: list[int] = []
        epoch = 0
        while True:
            for i, row in enumerate(iter_jsonl(self.path)):
                if (i + self.seed_offset + epoch) % n_shards != shard_id:
                    continue
                ids = self.tokenizer.encode(str(row["text"]), add_special_tokens=False)
                if eos is not None:
                    ids.append(int(eos))
                buffer.extend(ids)
                while len(buffer) >= self.seq_len:
                    out = torch.tensor(buffer[: self.seq_len], dtype=torch.long)
                    del buffer[: self.seq_len]
                    yield out
            epoch += 1


def _dist_setup() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _is_main(rank: int) -> bool:
    return rank == 0


def train(args: argparse.Namespace) -> dict[str, Any]:
    rank, local_rank, world_size = _dist_setup()
    device = torch.device(f"cuda:{local_rank}")
    paths = paths_for(args.run_root, args.domain)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup = _lazy_imports()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        trust_remote_code=True,
        token=args.hf_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=args.hf_token,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(device)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    dataset = PackedJsonlDataset(
        paths.train_jsonl,
        tokenizer,
        seq_len=args.seq_len,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        num_workers=args.dataloader_workers,
        pin_memory=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    global_tokens_per_step = args.seq_len * args.micro_batch_size * world_size * args.grad_accum
    max_steps = int(math.ceil(args.train_tokens / global_tokens_per_step))
    warmup_steps = max(1, int(args.warmup_ratio * max_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    if _is_main(rank):
        write_json(
            paths.train_manifest,
            {
                "domain": args.domain,
                "base_model": args.base_model,
                "base_revision": args.base_revision,
                "train_tokens_target": args.train_tokens,
                "global_tokens_per_step": global_tokens_per_step,
                "max_optimizer_steps": max_steps,
                "seq_len": args.seq_len,
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "warmup_steps": warmup_steps,
                "world_size": world_size,
            },
        )
        paths.train_log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        paths.train_log_jsonl.write_text("", encoding="utf-8")

    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    micro = 0
    opt_step = 0
    start = time.time()
    data_iter = iter(loader)
    while opt_step < max_steps:
        batch = next(data_iter).to(device, non_blocking=True)
        attention_mask = torch.ones_like(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=batch, attention_mask=attention_mask, labels=batch)
            loss = out.loss / args.grad_accum
        loss.backward()
        running += float(loss.detach().cpu()) * args.grad_accum
        micro += 1
        if micro % args.grad_accum != 0:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1
        if _is_main(rank) and (opt_step == 1 or opt_step % args.log_every == 0 or opt_step == max_steps):
            row = {
                "domain": args.domain,
                "step": opt_step,
                "max_steps": max_steps,
                "loss": running / max(1, args.log_every),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_sec": time.time() - start,
            }
            with paths.train_log_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            print(json.dumps(row), flush=True)
            running = 0.0

    if world_size > 1:
        dist.barrier()
    if _is_main(rank):
        save_model = model.module if isinstance(model, DDP) else model
        save_model.save_pretrained(paths.adapter_dir)
        tokenizer.save_pretrained(paths.adapter_dir)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return {"domain": args.domain, "adapter_dir": str(paths.adapter_dir), "steps": max_steps}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--domain", choices=["code", "biomed"], required=True)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--train-tokens", type=int, default=20_000_000)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    payload = train(parse_args())
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

