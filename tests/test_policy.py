from eval.policy import decide, Accept, Reject, Scores


BASE = Scores(ppl_bpb=4.20, lexicon=1.50, flatness=0.020, smoke=0.10)


def test_accept_when_all_improve():
    new = Scores(ppl_bpb=4.10, lexicon=1.40, flatness=0.010, smoke=0.05)
    d = decide(BASE, new)
    assert isinstance(d, Accept)


def test_reject_when_one_regresses():
    new = Scores(ppl_bpb=4.10, lexicon=1.60, flatness=0.020, smoke=0.10)
    d = decide(BASE, new)
    assert isinstance(d, Reject)
    assert "lexicon" in d.reason


def test_reject_when_smoke_regresses_at_all():
    # smoke has REGRESSION_TOLERANCE=0; any increase rejects
    new = Scores(ppl_bpb=4.10, lexicon=1.40, flatness=0.010, smoke=0.11)
    d = decide(BASE, new)
    assert isinstance(d, Reject)
    assert "smoke" in d.reason


def test_reject_when_all_flat():
    new = Scores(ppl_bpb=4.200, lexicon=1.500, flatness=0.0200, smoke=0.100)
    d = decide(BASE, new)
    assert isinstance(d, Reject)
    assert "no real improvement" in d.reason


def test_accept_when_one_clearly_improves():
    new = Scores(ppl_bpb=4.180, lexicon=1.500, flatness=0.0200, smoke=0.100)
    d = decide(BASE, new)
    assert isinstance(d, Accept)
