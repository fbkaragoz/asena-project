from pathlib import Path
from tokenizer.train_bpe import train_bpe


def test_train_bpe_produces_loadable_tokenizer(tiny_corpus_dir, tmp_path):
    # First, run the data pipeline to produce a train split.
    from data.pipeline import run_prepare_data
    clean = tmp_path / "clean"
    run_prepare_data(raw_dir=tiny_corpus_dir, out_dir=clean,
                     rules_path=Path("data/cleaning_rules.yaml"), heldout_pct=0)

    out_path = tmp_path / "tok.json"
    train_bpe(train_glob=str(clean / "train" / "*.parquet"), out_path=out_path,
              vocab_size=300, special_tokens=["<|bos|>", "<|eos|>", "<|pad|>"])
    assert out_path.exists()

    # Round-trip: load and encode/decode.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(out_path))
    enc = tok.encode("şehrin bedesteninde")
    assert len(enc.ids) > 0
    dec = tok.decode(enc.ids)
    assert "şehrin" in dec or "ş" in dec  # round-trips at least character-level


def test_train_bpe_reserves_special_tokens(tiny_corpus_dir, tmp_path):
    from data.pipeline import run_prepare_data
    clean = tmp_path / "clean"
    run_prepare_data(raw_dir=tiny_corpus_dir, out_dir=clean,
                     rules_path=Path("data/cleaning_rules.yaml"), heldout_pct=0)

    out_path = tmp_path / "tok.json"
    specials = ["<|bos|>", "<|eos|>", "<|pad|>"] + [f"<|reserved_{i}|>" for i in range(8)]
    train_bpe(train_glob=str(clean / "train" / "*.parquet"), out_path=out_path,
              vocab_size=300, special_tokens=specials)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(out_path))
    for s in specials:
        assert tok.token_to_id(s) is not None
