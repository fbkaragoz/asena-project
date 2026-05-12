from data.stages import normalize


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
