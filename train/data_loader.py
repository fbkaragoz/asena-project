"""Streaming data loader: parquet → tokenized batches with era-mix weighting (spec §3.3)."""
from __future__ import annotations
from pathlib import Path
import random
import glob as _glob
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer


class ParquetTokenStream:
    """Iterable yielding (x, y) long-tensor batches of shape (B, T).

    Reads cleaned parquet rows, samples by era weight, tokenizes, and concatenates
    until seq_len*batch_size tokens are available. Loops indefinitely.
    """

    def __init__(
        self,
        train_glob: str,
        tokenizer_path: Path,
        seq_len: int,
        batch_size: int,
        mix: dict[str, float],
        seed: int = 0,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

        rows_by_era: dict[str, list[str]] = {}
        for path in sorted(_glob.glob(train_glob)):
            table = pq.read_table(path, columns=["text", "era"])
            for text, era in zip(table.column("text").to_pylist(),
                                 table.column("era").to_pylist()):
                if era not in mix or mix[era] <= 0:
                    continue
                rows_by_era.setdefault(era, []).append(text)
        self.rows_by_era = rows_by_era
        total = sum(mix.get(e, 0) for e in rows_by_era)
        self.eras = list(rows_by_era.keys())
        self.weights = [mix[e] / total for e in self.eras] if total else []
        if not self.weights:
            raise ValueError("data loader: empty corpus after applying era mix")

    def _sample_text(self) -> str:
        era = self.rng.choices(self.eras, weights=self.weights, k=1)[0]
        return self.rng.choice(self.rows_by_era[era])

    def __iter__(self):
        buf: list[int] = []
        need = self.seq_len * self.batch_size + 1
        while True:
            while len(buf) < need:
                ids = self.tokenizer.encode(self._sample_text()).ids
                buf.append(self.bos_id)
                buf.extend(ids)
                if self.eos_id is not None:
                    buf.append(self.eos_id)
            chunk = buf[:need]
            buf = buf[need - 1:]
            t = torch.tensor(chunk, dtype=torch.long)
            x = t[:-1].view(self.batch_size, self.seq_len)
            y = t[1:].view(self.batch_size, self.seq_len)
            yield x, y
