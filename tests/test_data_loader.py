from pathlib import Path
import torch
from train.data_loader import ParquetTokenStream


def test_stream_yields_batches(tiny_corpus_dir, tmp_path):
    from data.pipeline import run_prepare_data
    from tokenizer.train_bpe import train_bpe

    clean = tmp_path / "clean"
    run_prepare_data(raw_dir=tiny_corpus_dir, out_dir=clean,
                     rules_path=Path("data/cleaning_rules.yaml"), heldout_pct=0)
    tok_path = tmp_path / "tok.json"
    train_bpe(train_glob=str(clean / "train" / "*.parquet"), out_path=tok_path, vocab_size=300)

    stream = ParquetTokenStream(
        train_glob=str(clean / "train" / "*.parquet"),
        tokenizer_path=tok_path,
        seq_len=16, batch_size=2,
        mix={"classical": 1.0, "late_ottoman": 1.0, "tanzimat": 1.0},
        seed=42,
    )
    x, y = next(iter(stream))
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert x.dtype == torch.long


def test_stream_respects_mix_weights(tiny_corpus_dir, tmp_path):
    """Same seed → same stream output."""
    from data.pipeline import run_prepare_data
    from tokenizer.train_bpe import train_bpe
    clean = tmp_path / "clean"
    run_prepare_data(raw_dir=tiny_corpus_dir, out_dir=clean,
                     rules_path=Path("data/cleaning_rules.yaml"), heldout_pct=0)
    tok_path = tmp_path / "tok.json"
    train_bpe(train_glob=str(clean / "train" / "*.parquet"), out_path=tok_path, vocab_size=300)

    s1 = ParquetTokenStream(train_glob=str(clean / "train" / "*.parquet"),
                            tokenizer_path=tok_path, seq_len=16, batch_size=2,
                            mix={"late_ottoman": 1.0}, seed=7)
    s2 = ParquetTokenStream(train_glob=str(clean / "train" / "*.parquet"),
                            tokenizer_path=tok_path, seq_len=16, batch_size=2,
                            mix={"late_ottoman": 1.0}, seed=7)
    x1, _ = next(iter(s1))
    x2, _ = next(iter(s2))
    assert torch.equal(x1, x2)
