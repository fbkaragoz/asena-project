import pyarrow.parquet as pq
import pytest
from data.schema import RAW_SCHEMA, validate_raw_parquet, SchemaError


def test_validate_passes_on_good_parquet(tiny_corpus_dir):
    path = tiny_corpus_dir / "tiny.parquet"
    table = pq.read_table(path)
    validate_raw_parquet(table)  # should not raise


def test_validate_fails_on_missing_column(tiny_corpus_dir, tmp_path):
    import pyarrow as pa
    bad = pa.table({"text": ["foo"], "source": ["x"]})  # missing required cols
    with pytest.raises(SchemaError, match="missing required column"):
        validate_raw_parquet(bad)


def test_schema_lists_required_columns():
    required = {"text", "source", "era", "genre", "language_variant",
                "source_pdf", "extraction_method", "length_chars"}
    assert required.issubset(set(RAW_SCHEMA.keys()))


def test_validate_accepts_null_extraction_confidence():
    """Optional column extraction_confidence may be null — validator must not reject it."""
    import pyarrow as pa
    table = pa.table({
        "text": ["مثال"],
        "source": ["doc_null"],
        "era": ["classical"],
        "genre": ["literary"],
        "language_variant": ["ottoman_istanbul"],
        "source_pdf": ["none.pdf"],
        "extraction_method": ["synthetic_test"],
        "extraction_confidence": pa.array([None], type=pa.float64()),
        "length_chars": pa.array([5], type=pa.int64()),
    })
    validate_raw_parquet(table)  # should not raise
