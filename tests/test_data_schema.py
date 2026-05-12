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


def test_validate_accepts_large_string_for_string_column():
    import pyarrow as pa
    rows = {
        "text": pa.array(["x" * 10], type=pa.large_string()),
        "source": pa.array(["s"], type=pa.large_string()),
        "era": pa.array(["late_ottoman"]),
        "genre": pa.array(["literary"]),
        "language_variant": pa.array(["ottoman_istanbul"]),
        "source_pdf": pa.array(["x.pdf"]),
        "extraction_method": pa.array(["m"]),
        "extraction_confidence": pa.array([1.0]),
        "length_chars": pa.array([10], type=pa.int32()),  # int32 accepted as int
    }
    table = pa.table(rows)
    validate_raw_parquet(table)  # must not raise


def test_validate_fails_on_unknown_language_variant(tiny_corpus_dir):
    import pyarrow as pa, pyarrow.parquet as pq
    table = pq.read_table(tiny_corpus_dir / "tiny.parquet")
    # Mutate language_variant column
    new_table = table.set_column(
        table.column_names.index("language_variant"),
        "language_variant",
        pa.array(["wrong_variant"] * len(table)),
    )
    with pytest.raises(SchemaError, match="unknown language_variant"):
        validate_raw_parquet(new_table)


def test_value_check_scans_full_table(tiny_corpus_dir):
    import pyarrow as pa, pyarrow.parquet as pq
    base = pq.read_table(tiny_corpus_dir / "tiny.parquet")
    # Concatenate 1500 valid rows then 1 bad at the end (was hidden under 1000-row sampling).
    good_rows = pa.concat_tables([base] * 500)  # 1500 rows
    bad = base.slice(0, 1).set_column(
        base.column_names.index("era"), "era", pa.array(["WRONG"]),
    )
    poisoned = pa.concat_tables([good_rows, bad])  # 1501 rows
    with pytest.raises(SchemaError, match="unknown era"):
        validate_raw_parquet(poisoned)


def test_validate_fails_on_null_era(tiny_corpus_dir):
    import pyarrow as pa, pyarrow.parquet as pq
    table = pq.read_table(tiny_corpus_dir / "tiny.parquet")
    new_table = table.set_column(
        table.column_names.index("era"),
        "era",
        pa.array([None, "late_ottoman", "tanzimat"]),
    )
    with pytest.raises(SchemaError, match="null values"):
        validate_raw_parquet(new_table)
