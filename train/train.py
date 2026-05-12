"""Training loop — sprint and promotion share this code (spec §4).

AGENT-EDITABLE (Tier 2). The agent may modify this file inside experiment branches
to introduce hyperparameter or optimizer changes.
"""
from __future__ import annotations
import math
import time
from pathlib import Path
import yaml
import torch
from train.arch import AsenaConfig, AsenaModel
from train.data_loader import ParquetTokenStream


def _cosine_lr(step: int, peak: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run_training(
    config_path: Path,
    tokenizer_path: Path,
    train_glob: str,
    checkpoint_out: Path,
    max_steps: int | None = None,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Run a training run defined by config_path; save final checkpoint; return metrics."""
    torch.manual_seed(seed)
    cfg = yaml.safe_load(open(config_path))
    mcfg, tcfg = cfg["model"], cfg["training"]

    from tokenizers import Tokenizer
    vocab_size = Tokenizer.from_file(str(tokenizer_path)).get_vocab_size()
    model_cfg = AsenaConfig(
        vocab_size=vocab_size,
        n_layers=mcfg["n_layers"], n_embd=mcfg["n_embd"],
        n_head=mcfg["n_head"], n_kv_heads=mcfg["n_kv_heads"],
        mlp_ratio=mcfg["mlp_ratio"], rope_theta=float(mcfg["rope_theta"]),
        tie_embeddings=mcfg["tie_embeddings"], init_std=mcfg["init_std"],
        max_seq_len=mcfg["max_seq_len"],
    )
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    dtype = torch.bfloat16 if tcfg["precision"] == "bf16" and dev.type == "cuda" else torch.float32
    model = AsenaModel(model_cfg).to(dev).to(dtype)

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr_peak"], betas=tuple(tcfg["betas"]),
        weight_decay=tcfg["weight_decay"],
    )

    stream = ParquetTokenStream(
        train_glob=train_glob, tokenizer_path=tokenizer_path,
        seq_len=tcfg["seq_len"], batch_size=tcfg["batch_size"],
        mix=cfg["data"]["mix"], seed=seed,
    )

    steps_from_tokens = tcfg["total_tokens"] // (tcfg["seq_len"] * tcfg["batch_size"])
    total_steps = min(max_steps or steps_from_tokens, steps_from_tokens) if max_steps else steps_from_tokens
    warmup = tcfg["warmup_steps"]
    losses: list[float] = []
    t0 = time.time()
    it = iter(stream)
    for step in range(total_steps):
        x, y = next(it)
        x = x.to(dev); y = y.to(dev)
        lr = _cosine_lr(step, tcfg["lr_peak"], warmup, total_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        losses.append(loss.item())

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {**cfg, "vocab_size": vocab_size},
        "step": total_steps,
    }, checkpoint_out)

    return {
        "losses": losses,
        "wall_seconds": time.time() - t0,
        "final_loss": losses[-1] if losses else float("nan"),
        "tokens_seen": total_steps * tcfg["seq_len"] * tcfg["batch_size"],
    }
