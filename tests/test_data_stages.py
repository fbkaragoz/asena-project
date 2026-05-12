from pathlib import Path
import pytest
from data.stages import normalize, apply_cleaning_rules, load_cleaning_rules


def test_normalize_collapses_whitespace():
    assert normalize("foo   bar\t\tbaz\r\nquux") == "foo bar baz quux"


def test_normalize_strips_control_chars():
    assert normalize("hello\x00world\x07") == "helloworld"


def test_normalize_applies_nfc():
    # "â" can be either NFC (single codepoint) or NFD (a + combining circumflex)
    nfd = "â"  # a + combining circumflex
    nfc = "â"   # â precomposed
    assert normalize(nfd) == nfc


def test_normalize_normalizes_line_endings():
    assert normalize("line1\r\nline2\rline3") == "line1 line2 line3"


def test_normalize_preserves_ottoman_diacritics():
    text = "şehrin bedesteninde âlim sarraflar bulunurdu"
    assert "â" in normalize(text)
    assert "ş" in normalize(text)


def test_substitution_removes_bare_page_numbers(tmp_path):
    rules = load_cleaning_rules(Path("data/cleaning_rules.yaml"))
    assert apply_cleaning_rules("12", rules) is None     # filtered (too short anyway)
    # but inside a longer text, page-number patterns get stripped:
    text = " 42  şehrin bedesteninde âlim ve sarraflar mevcut idi ve müşterilere mal arz ederlerdi"
    out = apply_cleaning_rules(text, rules)
    assert out is not None
    assert "42" not in out.split()[:1]   # leading bare number gone


def test_modern_loanword_filter_rejects_high_ratio():
    rules = load_cleaning_rules(Path("data/cleaning_rules.yaml"))
    # 4 of 8 tokens are modern → 50% > 4% → rejected
    text = "internet bilgisayar metro otobüs şehir mahalle ev sokak"
    assert apply_cleaning_rules(text, rules) is None


def test_modern_loanword_filter_allows_low_ratio():
    rules = load_cleaning_rules(Path("data/cleaning_rules.yaml"))
    # 1 of ~30 tokens modern → < 4% → allowed
    text = ("şehrin bedesteninde âlim ve sarraflar mevcut idi ve müşterilere mal "
            "arz ederlerdi devlet-i aliyye memuru her gün buraya uğrar internet")
    assert apply_cleaning_rules(text, rules) is not None


def test_length_filter_rejects_too_short():
    rules = load_cleaning_rules(Path("data/cleaning_rules.yaml"))
    assert apply_cleaning_rules("kısa metin", rules) is None


from data.stages import dedup_minhash


def test_dedup_removes_near_duplicates():
    texts = [
        "şehrin bedesteninde âlim ve sarraflar mevcut idi ve müşterilere mal arz ederlerdi",
        "şehrin bedesteninde âlim ve sarraflar mevcut idi ve müşterilere mal arz ederler",  # near-dup
        "Sultan Abdülhamid'in saltanatı sırasında devlet-i aliyye büyük tahavvüllere uğradı",
    ]
    kept = dedup_minhash(texts, threshold=0.85)
    # The first two are near-dups; only one should remain.
    assert len(kept) == 2


def test_dedup_keeps_distinct_texts():
    texts = [
        "Sultan Abdülhamid'in saltanatı sırasında devlet-i aliyye büyük tahavvüllere uğradı",
        "Tanzimat fermanı ile birlikte memalik-i osmaniyede yeni bir devre başlamıştır",
    ]
    kept = dedup_minhash(texts, threshold=0.85)
    assert len(kept) == 2
