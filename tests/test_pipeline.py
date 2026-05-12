from pathlib import Path
import pyarrow.parquet as pq
from data.pipeline import run_prepare_data


def test_run_prepare_data_produces_train_and_heldout(tiny_corpus_dir, tmp_path):
    out = tmp_path / "clean"
    run_prepare_data(raw_dir=tiny_corpus_dir, out_dir=out,
                     rules_path=Path("data/cleaning_rules.yaml"),
                     heldout_pct=33)  # high pct so tiny corpus splits visibly
    assert (out / "train").exists()
    assert (out / "heldout").exists()
    # At least one of the two has files
    train_files = list((out / "train").glob("*.parquet"))
    heldout_files = list((out / "heldout").glob("*.parquet"))
    assert (train_files or heldout_files)


def test_run_prepare_data_keeps_required_columns(tiny_corpus_dir, tmp_path):
    out = tmp_path / "clean"
    run_prepare_data(raw_dir=tiny_corpus_dir, out_dir=out,
                     rules_path=Path("data/cleaning_rules.yaml"),
                     heldout_pct=0)
    files = list((out / "train").glob("*.parquet"))
    assert files
    table = pq.read_table(files[0])
    for col in ("text", "source", "era", "genre", "source_pdf"):
        assert col in table.column_names
