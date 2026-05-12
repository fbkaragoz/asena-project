from pathlib import Path
import pytest
from train.train import run_training


@pytest.mark.slow
def test_30_second_smoke_training(tiny_corpus_dir, tmp_path):
    """Verify 10 training steps produce decreasing loss + a valid checkpoint."""
    from data.pipeline import run_prepare_data
    from tokenizer.train_bpe import train_bpe

    clean = tmp_path / "clean"
    run_prepare_data(raw_dir=tiny_corpus_dir, out_dir=clean,
                     rules_path=Path("data/cleaning_rules.yaml"), heldout_pct=0)
    tok_path = tmp_path / "tok.json"
    train_bpe(train_glob=str(clean / "train" / "*.parquet"), out_path=tok_path, vocab_size=300)

    out = tmp_path / "checkpoint.pt"
    result = run_training(
        config_path=Path("train/configs/sprint.yaml"),
        tokenizer_path=tok_path,
        train_glob=str(clean / "train" / "*.parquet"),
        checkpoint_out=out,
        max_steps=10,
        seed=42,
        device="cpu",
    )
    assert out.exists()
    losses = result["losses"]
    assert len(losses) == 10
    assert all(l == l for l in losses)            # no NaN
    assert losses[-1] < losses[0]
