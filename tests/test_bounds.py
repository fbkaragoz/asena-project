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
    # Post-second-amendment promotion config: n_layers=12, n_embd=640 → ~84M params.
    n = estimate_param_count(n_layers=12, n_embd=640, n_kv_heads=4, n_head=10,
                             mlp_ratio=2.67, vocab_size=24000, tied=False)
    assert 70_000_000 < n < 100_000_000


def test_check_sprint_bounds_rejects_too_big():
    with pytest.raises(BoundsViolation, match="param count"):
        check_sprint_bounds(params=200_000_000, estimated_seconds=60, estimated_vram_mb=10000)


def test_check_sprint_bounds_rejects_too_slow():
    with pytest.raises(BoundsViolation, match="wall clock"):
        check_sprint_bounds(params=30_000_000, estimated_seconds=500, estimated_vram_mb=10000)


def test_check_sprint_bounds_passes_normal():
    check_sprint_bounds(params=30_000_000, estimated_seconds=290, estimated_vram_mb=15000)


def test_check_promotion_bounds_passes():
    # Sized for the actual 7.97M-token budget: ~84M params, ~1-2h wall clock.
    check_promotion_bounds(params=84_000_000, estimated_seconds=7200, estimated_vram_mb=20000)


def test_check_promotion_bounds_rejects_old_200m():
    with pytest.raises(BoundsViolation, match="param count"):
        check_promotion_bounds(params=200_000_000, estimated_seconds=7200, estimated_vram_mb=20000)


def test_check_promotion_bounds_rejects_above_new_ceiling():
    # After the second amendment, promotion is capped at 130M (was 180M).
    with pytest.raises(BoundsViolation, match="param count"):
        check_promotion_bounds(params=150_000_000, estimated_seconds=7200, estimated_vram_mb=20000)


def test_check_promotion_bounds_rejects_too_small():
    with pytest.raises(BoundsViolation, match="param count"):
        check_promotion_bounds(params=50_000_000, estimated_seconds=7200, estimated_vram_mb=20000)


def test_check_promotion_bounds_rejects_too_slow():
    # Promotion wall-clock cap tightened to 4h after the second amendment
    # (60M tokens, ~1-2h expected; >4h is a red flag for misconfiguration).
    with pytest.raises(BoundsViolation, match="wall clock"):
        check_promotion_bounds(params=84_000_000, estimated_seconds=5 * 3600, estimated_vram_mb=20000)


def test_free_vram_mb_returns_int():
    v = free_vram_mb()
    assert v >= 0
