import pytest
from pathlib import Path
from eval.smoke import (
    SmokePromptResult, evaluate_smoke_prompts, _check_rules,
)


def test_check_rules_passes_clean_output():
    out = "devlet-i aliyye vezir kadı medrese müderris ulema şeyhülislam fetva " \
          "sadrazam padişah divan-ı hümayun saltanat hilafet rumeli anadolu " \
          "memalik-i osmaniyye hudud hicret kasvet selâmet asayiş emn ü emân"
    result = _check_rules(out, rules={"min_tokens": 10,
                                     "no_modern_loanwords": True,
                                     "no_repetition_5gram": True},
                         blacklist_path="data/modern_loanwords.txt")
    assert result.passed


def test_check_rules_fails_modern_loanword():
    out = "devlet vezir internet bilgisayar metro araba otobüs televizyon kadı medrese"
    result = _check_rules(out, rules={"min_tokens": 5,
                                     "no_modern_loanwords": True,
                                     "no_repetition_5gram": True},
                         blacklist_path="data/modern_loanwords.txt")
    assert not result.passed
    assert "loanword" in result.reason


def test_check_rules_fails_repetition():
    out = "kadı kadı kadı kadı kadı kadı kadı kadı kadı kadı"
    result = _check_rules(out, rules={"min_tokens": 5, "no_repetition_5gram": True,
                                     "no_modern_loanwords": False},
                         blacklist_path="data/modern_loanwords.txt")
    assert not result.passed


def test_check_rules_fails_too_short():
    out = "kısa"
    result = _check_rules(out, rules={"min_tokens": 30, "no_modern_loanwords": False,
                                     "no_repetition_5gram": False},
                         blacklist_path="data/modern_loanwords.txt")
    assert not result.passed
    assert "min_tokens" in result.reason
