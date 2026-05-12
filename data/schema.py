"""Raw corpus parquet schema validator (spec §3.1)."""
from __future__ import annotations
import pyarrow as pa
import pyarrow.compute as pc

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


def _type_compatible(actual: pa.DataType, expected: pa.DataType) -> bool:
    """Check if `actual` type is compatible with `expected` at the family level.

    We accept any string-like for string, any integer for int, any float for float.
    The spec at §3.1 defines semantic types, not specific pyarrow physical types.
    """
    if pa.types.is_string(expected) or pa.types.is_large_string(expected):
        return pa.types.is_string(actual) or pa.types.is_large_string(actual)
    if pa.types.is_integer(expected):
        return pa.types.is_integer(actual)
    if pa.types.is_floating(expected):
        return pa.types.is_floating(actual)
    return actual.equals(expected)


def _check_value_domain(table: pa.Table, col_name: str, allowed: set[str], label: str) -> None:
    col = table.column(col_name)
    allowed_arr = pa.array(sorted(allowed))
    bad_mask = pc.invert(pc.is_in(col, value_set=allowed_arr))
    if pc.any(bad_mask).as_py():
        idx = int(pc.indices_nonzero(bad_mask)[0].as_py())
        bad_val = col[idx].as_py()
        if bad_val is None:
            raise SchemaError(f"{label} column contains null values (required column)")
        raise SchemaError(f"unknown {label} at row {idx}: {bad_val!r} (allowed: {sorted(allowed)})")


class SchemaError(ValueError):
    pass


def validate_raw_parquet(table: pa.Table) -> None:
    """Validate that a pyarrow Table conforms to the raw corpus schema.

    Raises SchemaError on the first violation. `extraction_confidence` is the
    only nullable column.

    Value-domain checks (era, genre, language_variant) scan the full table via pyarrow.compute.
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
        if not _type_compatible(actual, expected):
            raise SchemaError(f"column {col!r} has type {actual} (expected {expected})")

    _check_value_domain(table, "era", ALLOWED_ERAS, "era")
    _check_value_domain(table, "genre", ALLOWED_GENRES, "genre")
    _check_value_domain(table, "language_variant", ALLOWED_VARIANTS, "language_variant")
