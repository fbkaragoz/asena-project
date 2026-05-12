"""Train the asena BPE tokenizer (spec §3.4)."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import glob as _glob
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers


def _iter_texts(train_glob: str) -> Iterator[str]:
    for path in sorted(_glob.glob(train_glob)):
        table = pq.read_table(path, columns=["text"])
        for t in table.column("text").to_pylist():
            if t:
                yield t


def train_bpe(
    train_glob: str,
    out_path: Path,
    vocab_size: int = 24_000,
    special_tokens: list[str] | None = None,
) -> None:
    """Train BPE on the parquet files matching train_glob; write tokenizer.json to out_path."""
    if special_tokens is None:
        special_tokens = ["<|bos|>", "<|eos|>", "<|pad|>"] + [f"<|reserved_{i}|>" for i in range(8)]

    tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(_iter_texts(train_glob), trainer=trainer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
