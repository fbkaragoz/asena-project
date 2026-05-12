"""Shared pytest fixtures for asena-project tests."""
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def tiny_corpus_dir(tmp_path: Path) -> Path:
    """Return a temp dir containing one tiny raw .parquet matching the §3.1 schema."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        {"text": "Sultan Abdülhamid'in saltanatı sırasında devlet-i aliyye büyük tahavvüllere şahid olmuştur.",
         "source": "doc_001", "era": "late_ottoman", "genre": "official",
         "language_variant": "ottoman_istanbul", "source_pdf": "salname_1882.pdf",
         "extraction_method": "synthetic_test", "extraction_confidence": 1.0, "length_chars": 95},
        {"text": "Tanzimat fermanı ile birlikte memalik-i osmaniyede yeni bir devre başlamıştır.",
         "source": "doc_002", "era": "tanzimat", "genre": "literary",
         "language_variant": "ottoman_istanbul", "source_pdf": "tarih_1850.pdf",
         "extraction_method": "synthetic_test", "extraction_confidence": 1.0, "length_chars": 80},
        {"text": "Şehrin bedesteninde bezzazlar ve sarraflar müşterilere mal arz ederlerdi.",
         "source": "doc_003", "era": "classical", "genre": "literary",
         "language_variant": "ottoman_istanbul", "source_pdf": "evliya_1660.pdf",
         "extraction_method": "synthetic_test", "extraction_confidence": 1.0, "length_chars": 75},
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, raw_dir / "tiny.parquet")
    return raw_dir
