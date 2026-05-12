import pytest
from eval.lexicon_score import lexicon_score_from_text


def test_high_score_for_ottoman_text():
    text = "Sultan Abdülhamid'in saltanatı sırasında devlet-i aliyye divan ve vezir"
    s = lexicon_score_from_text(text, lexicon_path="eval/heldout/ottoman_lexicon.txt")
    assert s > 0


def test_lower_for_modern_text():
    ott_text = "Sultan Abdülhamid'in saltanatı sırasında devlet-i aliyye divan ve vezir"
    mod_text = "internet bilgisayar televizyon araba metro otobüs"
    s_ott = lexicon_score_from_text(ott_text, lexicon_path="eval/heldout/ottoman_lexicon.txt")
    s_mod = lexicon_score_from_text(mod_text, lexicon_path="eval/heldout/ottoman_lexicon.txt")
    assert s_ott < s_mod


def test_empty_text_returns_inf_like():
    s = lexicon_score_from_text("", lexicon_path="eval/heldout/ottoman_lexicon.txt")
    assert s > 10
