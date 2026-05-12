"""Bits-per-byte on the frozen held-out corpus (spec §5.1).

val_bpb = sum(cross_entropy * tokens) / (total_bytes * ln(2))
Vocab-size independent — fair comparisons even if tokenizer changes in Phase 2.

IMMUTABLE (Tier 1). Do not modify without an unfreeze + clear-ledger event.
"""
from __future__ import annotations
import math
import glob as _glob
from pathlib import Path
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train.arch import AsenaConfig, AsenaModel


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[AsenaModel, int]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    model_cfg = AsenaConfig(
        vocab_size=cfg["vocab_size"],
        n_layers=mcfg["n_layers"], n_embd=mcfg["n_embd"],
        n_head=mcfg["n_head"], n_kv_heads=mcfg["n_kv_heads"],
        mlp_ratio=mcfg["mlp_ratio"], rope_theta=float(mcfg["rope_theta"]),
        tie_embeddings=mcfg["tie_embeddings"], init_std=mcfg["init_std"],
        max_seq_len=mcfg["max_seq_len"],
    )
    model = AsenaModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg["vocab_size"]


@torch.inference_mode()
def compute_heldout_bpb(
    checkpoint_path: Path,
    tokenizer_path: Path,
    heldout_glob: str,
    device: str = "cuda",
    max_seq_len: int = 1024,
) -> float:
    """Compute val_bpb across all heldout parquet files."""
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, vocab_size = _load_model(Path(checkpoint_path), dev)
    tok = Tokenizer.from_file(str(tokenizer_path))

    total_nll, total_bytes = 0.0, 0
    for path in sorted(_glob.glob(heldout_glob)):
        table = pq.read_table(path, columns=["text"])
        for text in table.column("text").to_pylist():
            if not text:
                continue
            ids = tok.encode(text).ids
            if len(ids) < 2:
                continue
            for i in range(0, len(ids) - 1, max_seq_len):
                chunk = ids[i:i + max_seq_len + 1]
                if len(chunk) < 2:
                    continue
                x = torch.tensor(chunk[:-1], dtype=torch.long, device=dev).unsqueeze(0)
                y = torch.tensor(chunk[1:], dtype=torch.long, device=dev).unsqueeze(0)
                logits = model(x)
                nll = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1),
                    reduction="sum",
                ).item()
                total_nll += nll
            total_bytes += len(text.encode("utf-8"))

    if total_bytes == 0:
        return float("inf")
    return total_nll / (total_bytes * math.log(2))
