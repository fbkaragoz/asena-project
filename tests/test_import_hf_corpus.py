"""Tests for tools/import_hf_corpus.py — dual-schema HF → RAW_SCHEMA importer.

Covers the pure helpers (date_to_era, extract_year_from_name, title_to_genre,
should_drop_page) and the two row-mappers (anadolu_row_to_raw, evliya_row_to_raw),
plus an end-to-end transform_table over synthetic in-memory tables.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from tools.import_hf_corpus import (
    date_to_era,
    extract_year_from_name,
    title_to_genre,
    should_drop_page,
    anadolu_row_to_raw,
    evliya_row_to_raw,
    transform_table,
    ANADOLU_SOURCE,
    EVLIYA_SOURCE,
)
from data.schema import RAW_SCHEMA, validate_raw_parquet


# ---------------------------------------------------------------------------
# date_to_era
# ---------------------------------------------------------------------------

def test_date_to_era_classical_year_only():
    assert date_to_era("1660") == "classical"


def test_date_to_era_classical_iso_date():
    assert date_to_era("1500-01-01") == "classical"


def test_date_to_era_tanzimat_start_inclusive():
    # 1839 is the conventional Tanzimat start (Gülhane edict).
    assert date_to_era("1839-11-03") == "tanzimat"


def test_date_to_era_tanzimat_middle():
    assert date_to_era("1850") == "tanzimat"


def test_date_to_era_late_ottoman_start_inclusive():
    # 1876 is the inclusive lower bound of late_ottoman (Abdülhamid II reign / I. Meşrutiyet).
    assert date_to_era("1876-12-23") == "late_ottoman"


def test_date_to_era_late_ottoman_periodical():
    # Sebilürreşad / Sıratımüstakim (1908+) — must land in late_ottoman.
    assert date_to_era("1908") == "late_ottoman"


def test_date_to_era_just_below_tanzimat():
    assert date_to_era("1838") == "classical"


def test_date_to_era_just_below_late_ottoman():
    assert date_to_era("1875") == "tanzimat"


def test_date_to_era_none_returns_default():
    assert date_to_era(None, default="late_ottoman") == "late_ottoman"


def test_date_to_era_unparseable_returns_default():
    assert date_to_era("not a date", default="classical") == "classical"


def test_date_to_era_accepts_integer():
    assert date_to_era(1700) == "classical"


# ---------------------------------------------------------------------------
# extract_year_from_name
# ---------------------------------------------------------------------------

def test_extract_year_simple_filename():
    assert extract_year_from_name("Necati Bey - Divan (1500)") == 1500


def test_extract_year_within_underscore():
    assert extract_year_from_name("sebilurresad_1908_vol3") == 1908


def test_extract_year_rejects_out_of_range():
    # 1234 is too early for Ottoman Latinized texts; 2024 is post-corpus.
    assert extract_year_from_name("doc_1234_random") is None
    assert extract_year_from_name("file_2024_modern") is None


def test_extract_year_returns_first_valid_match():
    # "1800-1900" appearing in a title — return the first plausible year.
    assert extract_year_from_name("Tezkire 1800-1900 collection") == 1800


def test_extract_year_none_for_no_digits():
    assert extract_year_from_name("undated_manuscript") is None


def test_extract_year_handles_none():
    assert extract_year_from_name(None) is None


# ---------------------------------------------------------------------------
# title_to_genre
# ---------------------------------------------------------------------------

def test_title_to_genre_divan_with_diacritics():
    assert title_to_genre("Necati Bey - Emrî Dîvânı") == "poetry"


def test_title_to_genre_divan_plain():
    assert title_to_genre("Vahyî Divan") == "poetry"


def test_title_to_genre_mecmua():
    assert title_to_genre("Pervâne Bey Mecmuası") == "poetry"


def test_title_to_genre_seyahatname():
    assert title_to_genre("Evliya Çelebi Seyahatnamesi") == "literary"


def test_title_to_genre_periodical_sebilurresad():
    assert title_to_genre("Sebilürreşad cilt 5") == "newspaper"


def test_title_to_genre_periodical_siratimustakim():
    assert title_to_genre("Sıratımüstakim 1908") == "newspaper"


def test_title_to_genre_falls_back_to_group_path():
    # Title is unhelpful; group_path indicates poetry collection.
    assert title_to_genre("untitled", group_path="literary/divan/17c") == "poetry"


def test_title_to_genre_unknown_defaults_to_other():
    assert title_to_genre("xyzzy") == "other"


def test_title_to_genre_handles_none():
    assert title_to_genre(None) == "other"


# ---------------------------------------------------------------------------
# should_drop_page
# ---------------------------------------------------------------------------

def test_should_drop_arabic_script():
    assert should_drop_page(primary_script="arabic", script_direction="rtl",
                            validation_status="pass") is True


def test_should_drop_rtl_direction():
    assert should_drop_page(primary_script="latin_extended", script_direction="rtl",
                            validation_status="pass") is True


def test_should_drop_mixed_script():
    assert should_drop_page(primary_script="mixed", script_direction="mixed",
                            validation_status="pass") is True


def test_should_drop_fail_validation():
    assert should_drop_page(primary_script="latin_extended", script_direction="ltr",
                            validation_status="fail") is True


def test_should_drop_empty_validation():
    assert should_drop_page(primary_script="latin_extended", script_direction="ltr",
                            validation_status="empty") is True


def test_should_keep_clean_latin_pass():
    assert should_drop_page(primary_script="latin_extended", script_direction="ltr",
                            validation_status="pass") is False


def test_should_keep_warn_pages():
    # `warn` is acceptable — mostly hyphenation issues per the dataset card.
    assert should_drop_page(primary_script="latin_extended", script_direction="ltr",
                            validation_status="warn") is False


# ---------------------------------------------------------------------------
# anadolu_row_to_raw
# ---------------------------------------------------------------------------

def _good_anadolu_row(**overrides):
    base = {
        "page_id": "p_abc123",
        "document_id": "doc0001",
        "document_name": "Necati_Bey_Divani_1500",
        "title": "Necati Bey - Emrî Dîvânı",
        "group_path": "literary/divan",
        "document_date_label": "1500-01-01",
        "primary_script": "latin_extended",
        "script_direction": "ltr",
        "validation_status": "pass",
        "clean_text": "Şu beytün manası ki gülşen-i hüsne nazar etmek …",
        "model_used": "deepseek-ai/DeepSeek-OCR-2",
    }
    base.update(overrides)
    return base


def test_anadolu_row_maps_required_columns():
    row = anadolu_row_to_raw(_good_anadolu_row())
    assert row is not None
    # All RAW_SCHEMA required columns present.
    for col in ("text", "source", "era", "genre", "language_variant",
                "source_pdf", "extraction_method", "length_chars"):
        assert col in row, f"missing {col}"
    assert row["text"] == "Şu beytün manası ki gülşen-i hüsne nazar etmek …"
    assert row["era"] == "classical"
    assert row["genre"] == "poetry"
    assert row["language_variant"] == "ottoman_istanbul"
    assert row["length_chars"] == len(row["text"])
    assert "DeepSeek-OCR" in row["extraction_method"]


def test_anadolu_row_dropped_arabic_script_returns_none():
    row = anadolu_row_to_raw(_good_anadolu_row(primary_script="arabic", script_direction="rtl"))
    assert row is None


def test_anadolu_row_dropped_fail_returns_none():
    assert anadolu_row_to_raw(_good_anadolu_row(validation_status="fail")) is None


def test_anadolu_row_missing_date_falls_back_to_name():
    row = anadolu_row_to_raw(_good_anadolu_row(
        document_date_label=None,
        document_name="Sebilurresad_1908_vol3",
    ))
    assert row is not None
    assert row["era"] == "late_ottoman"


def test_anadolu_row_periodical_title_overrides_misleading_filename_year():
    # Real-world case observed in anadolu-ocr-corpus: title is "Sebilürreşad 8.Cilt"
    # but document_name is "cilt_8_1850.pdf" (1850 is the print-run year of an old reprint
    # series, not the publication year). The periodical is 1908+, so its era must be
    # late_ottoman; the filename year must NOT win.
    row = anadolu_row_to_raw(_good_anadolu_row(
        document_date_label=None,
        document_name="cilt_8_1850.pdf",
        title="Sebilürreşad 8.Cilt",
    ))
    assert row is not None
    assert row["era"] == "late_ottoman"
    assert row["genre"] == "newspaper"


def test_anadolu_row_siratimustakim_periodical_title():
    row = anadolu_row_to_raw(_good_anadolu_row(
        document_date_label=None,
        document_name="cilt_3_1900.pdf",
        title="Sıratımüstakim 3.Cilt",
    ))
    assert row is not None
    assert row["era"] == "late_ottoman"
    assert row["genre"] == "newspaper"


def test_anadolu_row_with_empty_text_returns_none():
    # An OCR page with no clean_text is effectively empty.
    row = anadolu_row_to_raw(_good_anadolu_row(clean_text=""))
    assert row is None


def test_anadolu_row_source_pdf_uses_document_name():
    row = anadolu_row_to_raw(_good_anadolu_row(document_name="Necati_Bey_Divani_1500"))
    assert row["source_pdf"] == "Necati_Bey_Divani_1500"


# ---------------------------------------------------------------------------
# evliya_row_to_raw
# ---------------------------------------------------------------------------

def _good_evliya_row(**overrides):
    base = {
        "id": "evliya_book1_p42",
        "author": "Evliya Çelebi",
        "work": "Seyahatname",
        "book": "1",
        "page": 42,
        "language": "ota-latn",
        "direction": "ltr",
        "script": "latin_extended",
        "text": "İstanbul kal'asınun ahvâli ve binâ-yı şehri …",
        "ocr_status": "pass",
        "source_pdf": "evliya_seyahatname_book1.pdf",
        "pdf_creation_date": "1671",
        "ocr_model": "deepseek-ai/DeepSeek-OCR-2",
    }
    base.update(overrides)
    return base


def test_evliya_row_maps_required_columns():
    row = evliya_row_to_raw(_good_evliya_row())
    assert row is not None
    assert row["text"].startswith("İstanbul")
    assert row["era"] == "classical"  # 1671 → classical
    assert row["genre"] == "literary"  # seyahatname → literary
    assert row["language_variant"] == "ottoman_istanbul"
    assert row["source_pdf"] == "evliya_seyahatname_book1.pdf"


def test_evliya_row_dropped_fail_returns_none():
    assert evliya_row_to_raw(_good_evliya_row(ocr_status="fail")) is None


def test_evliya_row_dropped_rtl_returns_none():
    assert evliya_row_to_raw(_good_evliya_row(direction="rtl", script="arabic")) is None


def test_evliya_row_missing_date_uses_known_default():
    # If pdf_creation_date is None and id has no year, Evliya should default
    # to its known composition era (classical, ~1671).
    row = evliya_row_to_raw(_good_evliya_row(pdf_creation_date=None))
    assert row is not None
    assert row["era"] == "classical"


# ---------------------------------------------------------------------------
# transform_table — end-to-end over an in-memory pyarrow Table
# ---------------------------------------------------------------------------

def test_transform_table_anadolu_produces_valid_raw_schema():
    table = pa.table({
        "page_id":           ["p1", "p2", "p3"],
        "document_id":       ["d1", "d1", "d2"],
        "document_name":     ["Vahyi_Divan_1660", "Vahyi_Divan_1660", "Sebilurresad_1908"],
        "title":             ["Vahyî Divan", "Vahyî Divan", "Sebilürreşad cilt 5"],
        "group_path":        ["literary/divan", "literary/divan", "periodical/press"],
        "document_date_label": ["1660-01-01", "1660-01-01", "1908-01-01"],
        "primary_script":    ["latin_extended", "arabic", "latin_extended"],
        "script_direction":  ["ltr", "rtl", "ltr"],
        "validation_status": ["pass", "pass", "warn"],
        "clean_text":        ["güzel beyit bir", "rejected arabic page", "Sebilürreşad makalesi"],
        "model_used":        ["deepseek-ai/DeepSeek-OCR-2"] * 3,
    })
    out = transform_table(table, source=ANADOLU_SOURCE)
    # Row 2 (arabic/rtl) must be dropped.
    assert out.num_rows == 2
    # Validator must accept the output.
    validate_raw_parquet(out)
    eras = out.column("era").to_pylist()
    assert eras == ["classical", "late_ottoman"]
    genres = out.column("genre").to_pylist()
    assert genres == ["poetry", "newspaper"]


def test_transform_table_evliya_produces_valid_raw_schema():
    table = pa.table({
        "id":                ["e1", "e2"],
        "author":            ["Evliya Çelebi", "Evliya Çelebi"],
        "work":              ["Seyahatname", "Seyahatname"],
        "book":              ["1", "1"],
        "page":              [1, 2],
        "language":          ["ota-latn", "ota-latn"],
        "direction":         ["ltr", "ltr"],
        "script":            ["latin_extended", "latin_extended"],
        "text":              ["İstanbul kal'asınun ahvâli", "Bursa ahalisi der ki"],
        "ocr_status":        ["pass", "pass"],
        "source_pdf":        ["evliya_book1.pdf", "evliya_book1.pdf"],
        "pdf_creation_date": ["1671", "1671"],
        "ocr_model":         ["deepseek-ai/DeepSeek-OCR-2", "deepseek-ai/DeepSeek-OCR-2"],
    })
    out = transform_table(table, source=EVLIYA_SOURCE)
    assert out.num_rows == 2
    validate_raw_parquet(out)
    assert out.column("genre").to_pylist() == ["literary", "literary"]
    assert all(e == "classical" for e in out.column("era").to_pylist())


def test_transform_table_unknown_source_raises():
    table = pa.table({"x": [1]})
    with pytest.raises(ValueError, match="unknown source"):
        transform_table(table, source="not_a_real_source")
