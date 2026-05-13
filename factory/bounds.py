"""Param/time/VRAM bounds for sprint and promotion profiles (spec §4.2, §7.4)."""
from __future__ import annotations
import subprocess


class BoundsViolation(RuntimeError):
    pass


SPRINT_PARAM_MIN, SPRINT_PARAM_MAX = 10_000_000, 80_000_000
SPRINT_SECONDS_MAX = 360
SPRINT_VRAM_MB_MAX = 22_000

PROMOTION_PARAM_MIN, PROMOTION_PARAM_MAX = 60_000_000, 130_000_000
PROMOTION_SECONDS_MAX = 4 * 3600
PROMOTION_VRAM_MB_MAX = 22_000


def estimate_param_count(
    n_layers: int, n_embd: int, n_head: int, n_kv_heads: int,
    mlp_ratio: float, vocab_size: int, tied: bool,
) -> int:
    """Closed-form parameter count for the decoder defined in train/arch.py.

    Assumes RMSNorm (no bias), SwiGLU MLP (3 matrices: gate, up, down).
    """
    head_dim = n_embd // n_head
    kv_dim = n_kv_heads * head_dim
    per_block = (
        n_embd * n_embd
        + n_embd * kv_dim
        + n_embd * kv_dim
        + n_embd * n_embd
        + 3 * n_embd * int(n_embd * mlp_ratio)
        + 2 * n_embd
    )
    embed = vocab_size * n_embd
    head = 0 if tied else vocab_size * n_embd
    final_ln = n_embd
    return embed + n_layers * per_block + final_ln + head


def check_sprint_bounds(params: int, estimated_seconds: float, estimated_vram_mb: int) -> None:
    if not (SPRINT_PARAM_MIN <= params <= SPRINT_PARAM_MAX):
        raise BoundsViolation(f"sprint param count {params} outside [{SPRINT_PARAM_MIN}, {SPRINT_PARAM_MAX}]")
    if estimated_seconds > SPRINT_SECONDS_MAX:
        raise BoundsViolation(f"sprint wall clock {estimated_seconds:.0f}s > {SPRINT_SECONDS_MAX}s")
    if estimated_vram_mb > SPRINT_VRAM_MB_MAX:
        raise BoundsViolation(f"sprint VRAM {estimated_vram_mb}MB > {SPRINT_VRAM_MB_MAX}MB")


def check_promotion_bounds(params: int, estimated_seconds: float, estimated_vram_mb: int) -> None:
    if not (PROMOTION_PARAM_MIN <= params <= PROMOTION_PARAM_MAX):
        raise BoundsViolation(f"promotion param count {params} outside bounds")
    if estimated_seconds > PROMOTION_SECONDS_MAX:
        raise BoundsViolation(f"promotion wall clock {estimated_seconds:.0f}s > {PROMOTION_SECONDS_MAX}s")
    if estimated_vram_mb > PROMOTION_VRAM_MB_MAX:
        raise BoundsViolation(f"promotion VRAM {estimated_vram_mb}MB > {PROMOTION_VRAM_MB_MAX}MB")


def free_vram_mb() -> int:
    """Return free VRAM in MB on cuda:0 via nvidia-smi. Returns 0 if not on a GPU host."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, check=True, timeout=5,
        ).stdout.decode().strip().splitlines()
        return int(out[0])
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, ValueError):
        return 0
