"""HuggingFace OCR-corpus → RAW_SCHEMA importer (spec §3.1).

Maps two HF dataset schemas (Anadolu OpenCR export, Evliya Seyahatname) into
data/schema.py's RAW_SCHEMA. Pure column/value mapping plus page-level filtering;
no text cleaning, no dedup, no split. Run `cli.py prepare-data` afterwards for
Stages 1-4.

Era boundaries (3-bucket taxonomy, spec §3.1):
    classical:    year ≤ 1838
    tanzimat:     1839 ≤ year ≤ 1875
    late_ottoman: year ≥ 1876

Script/direction filter drops pages that aren't Latin-LTR; validation filter
keeps only `pass`/`warn` pages (drops `fail`/`empty`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ANADOLU_SOURCE = "anadolu-ocr-corpus"
EVLIYA_SOURCE = "evliya-celebi-seyahatname"

_CLASSICAL_MAX = 1838
_TANZIMAT_END = 1875
_MIN_VALID_YEAR, _MAX_VALID_YEAR = 1400, 1928

_LATIN_SCRIPTS = {"latin", "latin_extended"}
_LTR_DIRECTIONS = {"ltr"}
_KEEP_VALIDATION = {"pass", "warn"}

_YEAR_RE = re.compile(r"(\d{4})")

_POETRY_KEYS = ("divan", "dîvân", "dîvâni", "divanı", "mecmua", "mecmuası", "mecmua-i")
_PERIODICAL_KEYS = ("sebilürreşad", "sebilurresad", "sıratımüstakim", "siratimustakim", "gazete")
_LITERARY_KEYS = ("seyahatname", "seyahatnâme", "tezkire", "tezkere")
_LEGAL_KEYS = ("kanun", "nizamname")
_RELIGIOUS_KEYS = ("risale", "hutbe")
_OFFICIAL_KEYS = ("salname", "yıllık")

_EVLIYA_DEFAULT_YEAR = 1671

# Known publication years for periodicals whose filenames are unreliable
# (e.g. anadolu-ocr-corpus uses cilt_N_1850.pdf for Sebilürreşad volumes even
# though Sebilürreşad ran 1908-1925). Periodical-title match wins over filename
# year extraction. Lowercase, ASCII-fold-aware substrings.
_PERIODICAL_PUB_YEARS = {
    "sebilürreşad": 1908,
    "sebilurresad": 1908,
    "sebilurressad": 1908,
    "sıratımüstakim": 1908,
    "siratimustakim": 1908,
    "sırat-ı müstakim": 1908,
}


def _periodical_year_from_title(title) -> int | None:
    if not title:
        return None
    low = str(title).lower()
    for key, year in _PERIODICAL_PUB_YEARS.items():
        if key in low:
            return year
    return None

_RAW_PA_SCHEMA = pa.schema([
    ("text", pa.string()),
    ("source", pa.string()),
    ("era", pa.string()),
    ("genre", pa.string()),
    ("language_variant", pa.string()),
    ("source_pdf", pa.string()),
    ("extraction_method", pa.string()),
    ("extraction_confidence", pa.float64()),
    ("length_chars", pa.int64()),
])


def date_to_era(value, *, default: str | None = None) -> str | None:
    """Map a date-like value (str or int) to {classical, tanzimat, late_ottoman}.

    Returns `default` if value is None, missing a 4-digit year, or has a year
    outside the plausible Ottoman-Latinized range (1400-1928).
    """
    if value is None:
        return default
    if isinstance(value, int):
        year = value
    else:
        m = _YEAR_RE.search(str(value))
        if not m:
            return default
        year = int(m.group(1))
    if not (_MIN_VALID_YEAR <= year <= _MAX_VALID_YEAR):
        return default
    if year <= _CLASSICAL_MAX:
        return "classical"
    if year <= _TANZIMAT_END:
        return "tanzimat"
    return "late_ottoman"


def extract_year_from_name(name) -> int | None:
    """First 4-digit substring within the plausible Ottoman year range, or None."""
    if name is None:
        return None
    for m in _YEAR_RE.finditer(str(name)):
        y = int(m.group(1))
        if _MIN_VALID_YEAR <= y <= _MAX_VALID_YEAR:
            return y
    return None


def title_to_genre(title, group_path=None) -> str:
    """Heuristic genre inference. Returns one of the seven allowed genres."""
    haystack = " ".join(str(s) for s in (title, group_path) if s).lower()
    if not haystack:
        return "other"
    if any(k in haystack for k in _PERIODICAL_KEYS):
        return "newspaper"
    if any(k in haystack for k in _POETRY_KEYS):
        return "poetry"
    if any(k in haystack for k in _LITERARY_KEYS):
        return "literary"
    if any(k in haystack for k in _LEGAL_KEYS):
        return "legal"
    if any(k in haystack for k in _RELIGIOUS_KEYS):
        return "religious"
    if any(k in haystack for k in _OFFICIAL_KEYS):
        return "official"
    return "other"


def should_drop_page(*, primary_script, script_direction, validation_status) -> bool:
    if primary_script not in _LATIN_SCRIPTS:
        return True
    if script_direction not in _LTR_DIRECTIONS:
        return True
    if validation_status not in _KEEP_VALIDATION:
        return True
    return False


def _to_raw_row(*, text, source, era, genre, source_pdf, extraction_method) -> dict:
    return {
        "text": text,
        "source": source,
        "era": era,
        "genre": genre,
        "language_variant": "ottoman_istanbul",
        "source_pdf": source_pdf,
        "extraction_method": extraction_method,
        "extraction_confidence": None,
        "length_chars": len(text),
    }


def anadolu_row_to_raw(row: dict, *, era_default: str | None = None) -> dict | None:
    """Map an Anadolu-OCR `pages` row → RAW_SCHEMA dict, or None if filtered out."""
    if should_drop_page(
        primary_script=row.get("primary_script"),
        script_direction=row.get("script_direction"),
        validation_status=row.get("validation_status"),
    ):
        return None
    text = row.get("clean_text") or ""
    if not text.strip():
        return None
    era = date_to_era(row.get("document_date_label"))
    if era is None:
        # Known-periodical title wins over filename year (anadolu corpus has
        # misleading print-run years in cilt_N_1850.pdf / cilt_N_1900.pdf).
        py = _periodical_year_from_title(row.get("title"))
        if py is not None:
            era = date_to_era(py)
    if era is None:
        era = date_to_era(extract_year_from_name(row.get("document_name")), default=era_default)
    if era is None:
        return None
    return _to_raw_row(
        text=text,
        source=row.get("document_id") or "unknown",
        era=era,
        genre=title_to_genre(row.get("title"), row.get("group_path")),
        source_pdf=row.get("document_name") or row.get("document_id") or "unknown.pdf",
        extraction_method=row.get("model_used") or "deepseek-ai/DeepSeek-OCR",
    )


def evliya_row_to_raw(row: dict, *, era_default: str | None = None) -> dict | None:
    """Map an Evliya Seyahatname `pages` row → RAW_SCHEMA dict, or None if filtered."""
    if should_drop_page(
        primary_script=row.get("script"),
        script_direction=row.get("direction"),
        validation_status=row.get("ocr_status"),
    ):
        return None
    text = row.get("text") or ""
    if not text.strip():
        return None
    era = date_to_era(row.get("pdf_creation_date"))
    if era is None:
        era = date_to_era(extract_year_from_name(row.get("id")))
    if era is None:
        # Seyahatname's composition era is well-known — fall back to classical.
        era = date_to_era(_EVLIYA_DEFAULT_YEAR, default=era_default)
    return _to_raw_row(
        text=text,
        source=row.get("id") or row.get("source_pdf") or "evliya_unknown",
        era=era,
        genre=title_to_genre(row.get("work") or "Seyahatname"),
        source_pdf=row.get("source_pdf") or "evliya_unknown.pdf",
        extraction_method=row.get("ocr_model") or "deepseek-ai/DeepSeek-OCR",
    )


_MAPPERS = {ANADOLU_SOURCE: anadolu_row_to_raw, EVLIYA_SOURCE: evliya_row_to_raw}


def transform_table(table: pa.Table, *, source: str) -> pa.Table:
    """Transform an HF-source pyarrow Table → RAW_SCHEMA-compliant Table.

    Dropped rows (filter or null era) are silently removed. The returned Table
    is built against `_RAW_PA_SCHEMA` so the validator always accepts it.
    """
    mapper = _MAPPERS.get(source)
    if mapper is None:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(_MAPPERS)}")
    out_rows = []
    for row in table.to_pylist():
        mapped = mapper(row)
        if mapped is not None:
            out_rows.append(mapped)
    return pa.Table.from_pylist(out_rows, schema=_RAW_PA_SCHEMA)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HF OCR-corpus → asena RAW_SCHEMA parquet.")
    ap.add_argument("--source", required=True, choices=sorted(_MAPPERS))
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to local HF parquet file (use `datasets.load_dataset(...).to_parquet(...)` first).")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    in_tbl = pq.read_table(args.input)
    out_tbl = transform_table(in_tbl, source=args.source)
    pq.write_table(out_tbl, args.output)
    print(f"{args.input} → {args.output}: {in_tbl.num_rows} rows in → {out_tbl.num_rows} rows kept")
