import math
from pathlib import Path
import pytest
from eval.heldout_ppl import compute_heldout_bpb


@pytest.mark.slow
def test_compute_bpb_finite_and_nonnegative(tiny_corpus_dir, tmp_path):
    from data.pipeline import run_prepare_data
    from tokenizer.train_bpe import train_bpe
    from train.train import run_training

    clean = tmp_path / "clean"
    # heldout_pct=27: the 3 fixture PDFs hash to buckets 26, 32, 40.
    # pct=27 sends bucket-26 to heldout and keeps the other two in train,
    # giving non-empty sets on both sides.
    run_prepare_data(raw_dir=tiny_corpus_dir, out_dir=clean,
                     rules_path=Path("data/cleaning_rules.yaml"), heldout_pct=27)
    tok = tmp_path / "tok.json"
    train_bpe(train_glob=str(clean / "train" / "*.parquet"), out_path=tok, vocab_size=300)
    out = tmp_path / "ckpt.pt"
    run_training(config_path=Path("train/configs/sprint.yaml"),
                 tokenizer_path=tok,
                 train_glob=str(clean / "train" / "*.parquet"),
                 checkpoint_out=out, max_steps=2, device="cpu")

    bpb = compute_heldout_bpb(
        checkpoint_path=out,
        tokenizer_path=tok,
        heldout_glob=str(clean / "heldout" / "*.parquet"),
        device="cpu", max_seq_len=64,
    )
    assert math.isfinite(bpb)
    assert bpb >= 0
