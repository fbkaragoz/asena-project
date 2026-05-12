"""Raw corpus parquet schema validator (spec §3.1)."""
from __future__ import annotations
import pyarrow as pa

RAW_SCHEMA: dict[str, pa.DataType] = {
    "text": pa.string(),
    "source": pa.string(),
    "era": pa.string(),
    "genre": pa.string(),
    "language_variant": pa.string(),
    "source_pdf": pa.string(),
    "extraction_method": pa.string(),
    "extraction_confidence": pa.float64(),  # nullable
    "length_chars": pa.int64(),
}

ALLOWED_ERAS = {"classical", "late_ottoman", "tanzimat"}
ALLOWED_GENRES = {"newspaper", "literary", "legal", "religious", "official", "poetry", "other"}
ALLOWED_VARIANTS = {"ottoman_istanbul"}  # v1 only


class SchemaError(ValueError):
    pass


def validate_raw_parquet(table: pa.Table) -> None:
    """Validate that a pyarrow Table conforms to the raw corpus schema.

    Raises SchemaError on the first violation. `extraction_confidence` is the
    only nullable column.
    """
    have = set(table.column_names)
    required = set(RAW_SCHEMA.keys()) - {"extraction_confidence"}
    missing = required - have
    if missing:
        raise SchemaError(f"missing required column(s): {sorted(missing)}")

    for col, expected in RAW_SCHEMA.items():
        if col not in have:
            continue  # optional column absent is fine
        actual = table.schema.field(col).type
        if not actual.equals(expected):
            raise SchemaError(f"column {col!r} has type {actual} (expected {expected})")

    # Lightweight value-domain checks on a sample (first 1000 rows).
    sample = table.slice(0, min(1000, len(table)))
    for era in sample.column("era").to_pylist():
        if era is None or era not in ALLOWED_ERAS:
            raise SchemaError(f"unknown era: {era!r} (allowed: {sorted(ALLOWED_ERAS)})")
    for genre in sample.column("genre").to_pylist():
        if genre is None or genre not in ALLOWED_GENRES:
            raise SchemaError(f"unknown genre: {genre!r} (allowed: {sorted(ALLOWED_GENRES)})")
