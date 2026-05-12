from eval.flatness import compute_flatness


def test_zero_flatness_for_pure_ottoman():
    gens = ["Sultan Abdülhamid devlet-i aliyye divan vezir kadı medrese müderris"]
    s = compute_flatness(gens, blacklist_path="data/modern_loanwords.txt")
    assert s == 0.0


def test_high_flatness_for_modern_text():
    gens = ["internet bilgisayar televizyon araba metro"]
    s = compute_flatness(gens, blacklist_path="data/modern_loanwords.txt")
    assert s == 1.0


def test_partial_flatness():
    gens = ["sultan divan internet bilgisayar"]
    s = compute_flatness(gens, blacklist_path="data/modern_loanwords.txt")
    assert 0.4 <= s <= 0.6
