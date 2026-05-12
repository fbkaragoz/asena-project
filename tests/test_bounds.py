from factory.bounds import (
    estimate_param_count, BoundsViolation, check_sprint_bounds,
    check_promotion_bounds, free_vram_mb,
)
import pytest


def test_estimate_param_count_sprint():
    # depth-6, n_embd=384, n_head=6, mlp_ratio=2.67, vocab=24000
    n = estimate_param_count(n_layers=6, n_embd=384, n_kv_heads=2, n_head=6,
                             mlp_ratio=2.67, vocab_size=24000, tied=True)
    # Order-of-magnitude sanity check: ~19M with GQA kv_heads=2
    assert 10_000_000 < n < 80_000_000


def test_estimate_param_count_promotion():
    n = estimate_param_count(n_layers=18, n_embd=768, n_kv_heads=4, n_head=12,
                             mlp_ratio=2.67, vocab_size=24000, tied=False)
    assert 150_000_000 < n < 280_000_000


def test_check_sprint_bounds_rejects_too_big():
    with pytest.raises(BoundsViolation, match="param count"):
        check_sprint_bounds(params=200_000_000, estimated_seconds=60, estimated_vram_mb=10000)


def test_check_sprint_bounds_rejects_too_slow():
    with pytest.raises(BoundsViolation, match="wall clock"):
        check_sprint_bounds(params=30_000_000, estimated_seconds=500, estimated_vram_mb=10000)


def test_check_sprint_bounds_passes_normal():
    check_sprint_bounds(params=30_000_000, estimated_seconds=290, estimated_vram_mb=15000)


def test_check_promotion_bounds_passes():
    check_promotion_bounds(params=200_000_000, estimated_seconds=86400, estimated_vram_mb=20000)


def test_free_vram_mb_returns_int():
    v = free_vram_mb()
    assert v >= 0
